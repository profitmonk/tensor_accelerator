#==============================================================================
# ResNet Convolution Layer Kernel
#
# Implements: Conv2D + BatchNorm + ReLU
#
# Method: im2col transformation + GEMM
# - Convert convolution to matrix multiplication
# - Conv2D: [N,C,H,W] * [K,C,kH,kW] -> [N,K,H',W']
# - As GEMM: [N*H'*W', C*kH*kW] × [C*kH*kW, K] -> [N*H'*W', K]
#
# Example configuration (ResNet-18 layer):
# - Input: 1×64×56×56 (NCHW)
# - Filter: 64×64×3×3
# - Output: 1×64×56×56
# - im2col: [3136, 576] × [576, 64] -> [3136, 64]
#
# For 16×16 systolic array, we tile:
# - M_tile = 16 (output spatial)
# - K_tile = 16 (input channels * kernel)
# - N_tile = 16 (output channels)
#==============================================================================

# Configuration
.equ    BATCH_SIZE,     1
.equ    IN_CHANNELS,    64
.equ    OUT_CHANNELS,   64
.equ    IN_HEIGHT,      56
.equ    IN_WIDTH,       56
.equ    KERNEL_SIZE,    3
.equ    STRIDE,         1
.equ    PADDING,        1

# Derived dimensions
.equ    OUT_HEIGHT,     56          # (56 + 2*1 - 3) / 1 + 1 = 56
.equ    OUT_WIDTH,      56
.equ    GEMM_M,         3136        # OUT_HEIGHT * OUT_WIDTH = 56*56
.equ    GEMM_K,         576         # IN_CHANNELS * KERNEL_SIZE^2 = 64*9
.equ    GEMM_N,         64          # OUT_CHANNELS

# Tile sizes (for 16×16 array)
.equ    TILE_M,         16
.equ    TILE_K,         16
.equ    TILE_N,         16

# Number of tiles
.equ    M_TILES,        196         # 3136 / 16
.equ    K_TILES,        36          # 576 / 16
.equ    N_TILES,        4           # 64 / 16

# SRAM layout
.equ    ACT_BUF_A,      0x0000      # Activation double buffer A
.equ    ACT_BUF_B,      0x0200      # Activation double buffer B
.equ    WT_BUF_A,       0x0400      # Weight double buffer A
.equ    WT_BUF_B,       0x0600      # Weight double buffer B
.equ    OUT_BUF,        0x0800      # Output buffer
.equ    BIAS_BUF,       0x0A00      # Bias buffer
.equ    BN_SCALE,       0x0A80      # BatchNorm scale (gamma)
.equ    BN_BIAS,        0x0B00      # BatchNorm bias (beta)
.equ    BN_MEAN,        0x0B80      # BatchNorm running mean
.equ    BN_VAR,         0x0C00      # BatchNorm running variance

# External memory addresses (HBM)
.equ    HBM_IM2COL,     0x80000000  # im2col transformed input
.equ    HBM_WEIGHTS,    0x80100000  # Filter weights [K, C*kH*kW]
.equ    HBM_BIAS,       0x80200000  # Bias
.equ    HBM_OUTPUT,     0x80300000  # Output
.equ    HBM_BN_PARAMS,  0x80400000  # BatchNorm parameters

#==============================================================================
# Main Kernel
#==============================================================================

main:
    # -------------------------------------------------------------------------
    # Step 1: Load BatchNorm parameters (once per layer)
    # -------------------------------------------------------------------------
    
    DMA.LOAD_1D     BN_SCALE, HBM_BN_PARAMS, 64, 64
    DMA.LOAD_1D     BN_BIAS, 0x80400100, 64, 64
    DMA.LOAD_1D     BN_MEAN, 0x80400200, 64, 64
    DMA.LOAD_1D     BN_VAR, 0x80400300, 64, 64
    SYNC.WAIT_DMA

    # -------------------------------------------------------------------------
    # Step 2: Main GEMM loop over output tiles
    # 
    # For each (m_tile, n_tile):
    #   - Load weight tile (once per n_tile)
    #   - Loop over k_tiles, accumulating partial products
    #   - Apply BatchNorm + ReLU
    #   - Store output tile
    # -------------------------------------------------------------------------

    # Outer loop: N tiles (output channels)
n_loop:
    LOOP            N_TILES

    # Load first weight tile for this N column
    # Weight layout: [K, N] where K = C*kH*kW, N = out_channels
    DMA.LOAD_2D     WT_BUF_A, HBM_WEIGHTS, TILE_K, TILE_N, GEMM_N, TILE_N
    SYNC.WAIT_DMA

    # Middle loop: M tiles (output spatial positions)
m_loop:
    LOOP            M_TILES
    
    # Initialize output accumulator to zero
    VEC.ZERO        v0
    VEC.STORE       v0, OUT_BUF, 256

    # Inner loop: K tiles (input channels * kernel)
k_loop:
    LOOP            K_TILES

    # ---- Double-buffered: Load next tiles while computing ----
    
    # Load activation tile: im2col[m_tile*TILE_M : , k_tile*TILE_K : ]
    # For now, simplified single-buffer load
    DMA.LOAD_2D     ACT_BUF_A, HBM_IM2COL, TILE_M, TILE_K, GEMM_K, TILE_K
    SYNC.WAIT_DMA

    # GEMM: OUT_BUF += ACT_BUF_A × WT_BUF_A
    # Accumulate into output buffer
    TENSOR.GEMM_ACC OUT_BUF, ACT_BUF_A, WT_BUF_A, TILE_M, TILE_N, TILE_K, 1

    SYNC.WAIT_MXU

    ENDLOOP         # k_loop

    # -------------------------------------------------------------------------
    # Step 3: Apply BatchNorm + ReLU to output tile
    #
    # BatchNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
    # Simplified: y = scale * x + bias (precomputed scale/bias)
    # Then: ReLU(y)
    # -------------------------------------------------------------------------

    # Load output tile into vector registers
    VEC.LOAD        v0, OUT_BUF, 256

    # Load precomputed BN scale and bias for this output channel
    VEC.LOAD        v1, BN_SCALE, 16
    VEC.LOAD        v2, BN_BIAS, 16

    # Apply: y = scale * x + bias
    VEC.MUL         v3, v0, v1
    VEC.ADD         v3, v3, v2

    # Apply ReLU
    VEC.RELU        v4, v3

    # Store result
    VEC.STORE       v4, OUT_BUF, 256

    # -------------------------------------------------------------------------
    # Step 4: Store output tile to HBM
    # -------------------------------------------------------------------------

    DMA.STORE_2D    HBM_OUTPUT, OUT_BUF, TILE_M, TILE_N, GEMM_N, TILE_N
    SYNC.WAIT_DMA

    ENDLOOP         # m_loop

    ENDLOOP         # n_loop

    # -------------------------------------------------------------------------
    # Done
    # -------------------------------------------------------------------------

    HALT


#==============================================================================
# Optimized Version with Double Buffering
#==============================================================================

# This would be the optimized version with proper double-buffering:
#
# main_optimized:
#     # Prologue: Load first tiles
#     DMA.LOAD_2D     ACT_BUF_A, HBM_IM2COL, TILE_M, TILE_K, GEMM_K, TILE_K
#     DMA.LOAD_2D     WT_BUF_A, HBM_WEIGHTS, TILE_K, TILE_N, GEMM_N, TILE_N
#     SYNC.WAIT_DMA
#
# main_loop:
#     LOOP            TOTAL_TILES
#
#     # Start loading next tile (to buffer B)
#     DMA.LOAD_2D     ACT_BUF_B, HBM_IM2COL, TILE_M, TILE_K, GEMM_K, TILE_K
#     DMA.LOAD_2D     WT_BUF_B, HBM_WEIGHTS, TILE_K, TILE_N, GEMM_N, TILE_N
#
#     # Compute on current tile (buffer A)
#     TENSOR.GEMM     OUT_BUF, ACT_BUF_A, WT_BUF_A, TILE_M, TILE_N, TILE_K, 1
#
#     SYNC.WAIT_ALL
#
#     # Swap buffer pointers (handled by compiler/address calculation)
#     # ...
#
#     ENDLOOP
#
#     HALT

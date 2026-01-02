# ============================================================
# Generated code for: mlp_medium
# Nodes: 3
# Tensors: 6
# ============================================================

# Memory addresses
.equ WEIGHT_W1_ADDR, 0x00100000
.equ WEIGHT_W1_SIZE, 0x8000
.equ WEIGHT_W2_ADDR, 0x00108000
.equ WEIGHT_W2_SIZE, 0x500

.equ SRAM_ACT_A, 0x0000
.equ SRAM_ACT_B, 0x1000
.equ SRAM_WT_A, 0x2000
.equ SRAM_WT_B, 0x4000
.equ SRAM_OUT, 0x6000

# ============================================================
# Main Program
# ============================================================


# Node: fc1 (tile 0)
# Op: GEMM
# GEMM: 8x256 @ 256x128
TENSOR.GEMM 0x6000, 0x0000, 0x2000, 8, 128, 256
SYNC.WAIT_MXU

# Node: relu
# Op: RELU
# Load H1 from DDR
DMA.LOAD_1D 0x0000, 0x00000000, 512
SYNC.WAIT_DMA
VECTOR.RELU 0x180000, 0x0000, 128
SYNC.WAIT_VPU

# Node: fc2 (tile 0)
# Op: GEMM
# GEMM: 8x128 @ 128x8
TENSOR.GEMM 0x6000, 0x0000, 0x2000, 8, 8, 128
SYNC.WAIT_MXU

# Node: fc2 (tile 1)
# Op: GEMM
# GEMM: 8x128 @ 128x8
TENSOR.GEMM 0x6000, 0x0000, 0x2000, 8, 8, 128
SYNC.WAIT_MXU

# End of program
HALT

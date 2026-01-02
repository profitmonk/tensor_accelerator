# ============================================================
# Generated code for: gemm_16x16
# Nodes: 1
# Tensors: 3
# ============================================================

# Memory addresses
.equ WEIGHT_B_ADDR, 0x00100000
.equ WEIGHT_B_SIZE, 0x100

.equ SRAM_ACT_A, 0x0000
.equ SRAM_ACT_B, 0x1000
.equ SRAM_WT_A, 0x2000
.equ SRAM_WT_B, 0x4000
.equ SRAM_OUT, 0x6000

# ============================================================
# Main Program
# ============================================================


# Node: gemm (tile 0)
# Op: GEMM
# GEMM: 16x16 @ 16x16
TENSOR.GEMM 0x6000, 0x0000, 0x2000, 16, 16, 16
SYNC.WAIT_MXU

# End of program
HALT

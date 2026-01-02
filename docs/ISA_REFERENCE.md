# Tensor Accelerator ISA Reference

## Version 1.0

---

## Table of Contents

1. [Overview](#overview)
2. [Instruction Format](#instruction-format)
3. [Memory Model](#memory-model)
4. [Instruction Classes](#instruction-classes)
   - [TENSOR Instructions](#tensor-instructions)
   - [VECTOR Instructions](#vector-instructions)
   - [DMA Instructions](#dma-instructions)
   - [CONTROL Instructions](#control-instructions)
   - [SYNC Instructions](#sync-instructions)
5. [Complete Instruction Reference](#complete-instruction-reference)
6. [Programming Examples](#programming-examples)
7. [Performance Guidelines](#performance-guidelines)

---

## Overview

The Tensor Accelerator ISA is designed for efficient neural network inference. It provides:

- **Matrix operations** via a 16×16 systolic array (MXU)
- **Vector operations** via a 256-wide vector unit (VPU)  
- **Data movement** via a high-bandwidth DMA engine
- **Control flow** via the Local Command Processor (LCP)

### Architecture Summary

| Component | Specification |
|-----------|---------------|
| Systolic Array | 16×16 INT8 MACs, weight-stationary |
| Vector Unit | 256-wide SIMD, INT8/INT32 |
| SRAM | 2MB total (512KB act, 768KB weight, 768KB scratch) |
| DMA Bandwidth | 256 bits/cycle |
| Clock Target | 500 MHz |

---

## Instruction Format

Instructions are 128 bits (16 bytes) for uniform fetch/decode:

```
┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│ OPCODE │ FLAGS  │  DST   │  SRC1  │  SRC2  │  SRC3  │  IMM1  │  IMM2  │
│ [7:0]  │ [15:8] │ [31:16]│ [47:32]│ [63:48]│ [79:64]│ [95:80]│[127:96]│
│ 8 bits │ 8 bits │ 16 bits│ 16 bits│ 16 bits│ 16 bits│ 16 bits│ 32 bits│
└────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
```

### Opcode Encoding

| Class | Opcode Range | Description |
|-------|--------------|-------------|
| TENSOR | 0x00-0x1F | Matrix/tensor operations |
| VECTOR | 0x20-0x5F | Vector/activation operations |
| DMA | 0x60-0x7F | Data movement |
| CONTROL | 0x80-0x9F | Loops, branches |
| SYNC | 0xA0-0xAF | Synchronization |
| MISC | 0xF0-0xFF | NOP, HALT, debug |

### Flag Bits

| Bit | Name | Description |
|-----|------|-------------|
| 0 | ACC | Accumulate mode (GEMM_ACC) |
| 1 | RELU | Fused ReLU activation |
| 2 | SAT | Saturate to INT8 |
| 3 | ASYNC | Don't block on completion |
| 4-7 | Reserved | Future use |

---

## Memory Model

### Address Spaces

| Address Range | Size | Description |
|---------------|------|-------------|
| 0x0000-0x7FFF | 32KB | Activation Buffer A |
| 0x8000-0xFFFF | 32KB | Activation Buffer B |
| 0x10000-0x1FFFF | 64KB | Weight Buffer A |
| 0x20000-0x2FFFF | 64KB | Weight Buffer B |
| 0x30000-0x3FFFF | 64KB | Output Buffer |
| 0x40000-0x5FFFF | 128KB | Scratch Buffer |

### DDR Memory Map

| Address Range | Description |
|---------------|-------------|
| 0x00100000 | Weights base |
| 0x00200000 | Input activations base |
| 0x00300000 | Output activations base |
| 0x00400000 | Intermediate buffers |

---

## Instruction Classes

### TENSOR Instructions

Matrix and tensor operations executed on the MXU (Matrix Unit).

| Opcode | Mnemonic | Description |
|--------|----------|-------------|
| 0x00 | TENSOR.NOP | No operation |
| 0x01 | TENSOR.GEMM | General matrix multiply C = A @ B |
| 0x02 | TENSOR.GEMM_ACC | GEMM with accumulate C += A @ B |
| 0x03 | TENSOR.IM2COL | Image-to-column transform |
| 0x04 | TENSOR.TRANSPOSE | Matrix transpose |
| 0x05 | TENSOR.MATMUL_TRANS_B | A @ B.T |
| 0x06 | TENSOR.DEPTHWISE_CONV | Depthwise convolution |
| 0x07 | TENSOR.MAXPOOL | Max pooling |
| 0x08 | TENSOR.AVGPOOL | Average pooling |

### VECTOR Instructions

Element-wise and reduction operations on the VPU.

| Opcode | Mnemonic | Description |
|--------|----------|-------------|
| 0x20 | VECTOR.ADD | dst = src1 + src2 |
| 0x21 | VECTOR.SUB | dst = src1 - src2 |
| 0x22 | VECTOR.MUL | dst = src1 * src2 |
| 0x23 | VECTOR.DIV | dst = src1 / src2 |
| 0x24 | VECTOR.SCALE | dst = src1 * imm |
| 0x25 | VECTOR.BIAS_ADD | Broadcast add |
| 0x26 | VECTOR.ADD_BCAST | Add with broadcasting |
| 0x28 | VECTOR.RELU | ReLU: max(0, x) |
| 0x29 | VECTOR.RELU6 | ReLU6: clamp(x, 0, 6) |
| 0x2A | VECTOR.SIGMOID | 1/(1+exp(-x)) |
| 0x2B | VECTOR.TANH | tanh(x) |
| 0x2C | VECTOR.GELU | x * Φ(x) |
| 0x30 | VECTOR.REDUCE_SUM | Sum reduction |
| 0x31 | VECTOR.REDUCE_MAX | Max reduction |
| 0x32 | VECTOR.REDUCE_MIN | Min reduction |
| 0x33 | VECTOR.GLOBAL_AVG | Global average |
| 0x38 | VECTOR.BATCHNORM | scale * x + bias |
| 0x39 | VECTOR.LAYERNORM_* | LayerNorm passes |
| 0x3A | VECTOR.GROUPNORM | Group normalization |
| 0x40 | VECTOR.SOFTMAX_P1 | Softmax pass 1 (max) |
| 0x41 | VECTOR.SOFTMAX_P2 | Softmax pass 2 (exp-sum) |
| 0x42 | VECTOR.SOFTMAX_P3 | Softmax pass 3 (div) |
| 0x48 | VECTOR.SCALE_Q8 | Q8 fixed-point scale |
| 0x49 | VECTOR.MASKED_FILL | Conditional fill |
| 0x4A | VECTOR.SCALE_SHIFT | scale * x + shift |

### DMA Instructions

Data movement between SRAM and DDR.

| Opcode | Mnemonic | Description |
|--------|----------|-------------|
| 0x60 | DMA.LOAD_1D | Linear load from DDR |
| 0x61 | DMA.STORE_1D | Linear store to DDR |
| 0x62 | DMA.LOAD_2D | 2D strided load |
| 0x63 | DMA.STORE_2D | 2D strided store |
| 0x64 | DMA.COPY | SRAM-to-SRAM copy |
| 0x65 | DMA.FILL | Fill with constant |
| 0x66 | DMA.GATHER | Indexed gather |
| 0x67 | DMA.SCATTER | Indexed scatter |

### CONTROL Instructions

Program control flow.

| Opcode | Mnemonic | Description |
|--------|----------|-------------|
| 0x80 | CTRL.LOOP | Start loop (count in imm) |
| 0x81 | CTRL.ENDLOOP | End loop |
| 0x82 | CTRL.BRANCH | Unconditional branch |
| 0x83 | CTRL.BRANCH_EQ | Branch if equal |
| 0x84 | CTRL.BRANCH_NE | Branch if not equal |
| 0x85 | CTRL.CALL | Subroutine call |
| 0x86 | CTRL.RETURN | Return from subroutine |

### SYNC Instructions

Synchronization primitives.

| Opcode | Mnemonic | Description |
|--------|----------|-------------|
| 0xA0 | SYNC.WAIT_MXU | Wait for MXU completion |
| 0xA1 | SYNC.WAIT_VPU | Wait for VPU completion |
| 0xA2 | SYNC.WAIT_DMA | Wait for DMA completion |
| 0xA3 | SYNC.WAIT_ALL | Wait for all units |
| 0xA4 | SYNC.BARRIER | Global TPC barrier |
| 0xA5 | SYNC.FENCE | Memory fence |

### MISC Instructions

| Opcode | Mnemonic | Description |
|--------|----------|-------------|
| 0xF0 | NOP | No operation |
| 0xFE | DEBUG | Debug breakpoint |
| 0xFF | HALT | Stop execution |

---

## Complete Instruction Reference

### TENSOR.GEMM

General Matrix Multiply: C = A × B

```
TENSOR.GEMM dst, src_a, src_b, M, N, K
```

| Field | Bits | Description |
|-------|------|-------------|
| dst | [31:16] | Output SRAM address |
| src_a | [47:32] | Matrix A SRAM address |
| src_b | [63:48] | Matrix B SRAM address |
| M | [79:64] | Rows of A and C |
| N | [95:80] | Columns of B and C |
| K | [111:96] | Columns of A, Rows of B |

**Behavior:**
```
for m in 0..M:
  for n in 0..N:
    acc = 0
    for k in 0..K:
      acc += A[m,k] * B[k,n]
    C[m,n] = saturate_int8(acc >> shift)
```

**Latency:** M + N + K + 2*ARRAY_SIZE cycles (pipelined)

---

### TENSOR.DEPTHWISE_CONV

Depthwise Convolution (one filter per channel)

```
TENSOR.DEPTHWISE_CONV dst, src, weight, C, H, W, kH, kW, sH, sW, pad
```

| Field | Description |
|-------|-------------|
| C | Number of channels |
| H, W | Input height/width |
| kH, kW | Kernel size |
| sH, sW | Stride |
| pad | Padding (symmetric) |

---

### VECTOR.RELU

ReLU Activation

```
VECTOR.RELU dst, src, count
```

**Behavior:**
```
for i in 0..count:
  dst[i] = max(0, src[i])
```

**Throughput:** 256 elements/cycle

---

### VECTOR.SOFTMAX_P1/P2/P3

Three-pass Softmax implementation:

```
# Pass 1: Find row maxima
VECTOR.SOFTMAX_P1 scratch, src, axis_size, num_rows

# Pass 2: Compute exp(x - max) and sum
VECTOR.SOFTMAX_P2 dst, src, scratch, axis_size, num_rows

# Pass 3: Normalize by sum
VECTOR.SOFTMAX_P3 dst, dst, scratch, axis_size, num_rows
```

---

### DMA.LOAD_1D

Linear DMA transfer from DDR to SRAM

```
DMA.LOAD_1D sram_addr, ddr_addr, byte_count
```

| Field | Bits | Description |
|-------|------|-------------|
| sram_addr | [31:16] | Destination in SRAM |
| ddr_addr | [95:64] | Source in DDR (32-bit) |
| byte_count | [47:32] | Transfer size in bytes |

**Throughput:** 256 bits/cycle (32 bytes/cycle)

---

## Programming Examples

### Example 1: Simple GEMM

```asm
# 16x16 GEMM: C = A @ B
# A at SRAM 0x0000, B at 0x2000, C at 0x6000

TENSOR.GEMM 0x6000, 0x0000, 0x2000, 16, 16, 16
SYNC.WAIT_MXU
```

### Example 2: Conv2D via im2col

```asm
# Conv2D: [1,32,56,56] * [64,32,3,3] -> [1,64,56,56]
# Input at 0x0000, Weight at 0x2000, Output at 0x6000

# Step 1: im2col transform
TENSOR.IM2COL 0x7000, 0x0000, 32, 56, 56, 3, 3, 1, 1
SYNC.WAIT_MXU

# Step 2: GEMM
# M = 56*56 = 3136, K = 32*3*3 = 288, N = 64
TENSOR.GEMM 0x6000, 0x7000, 0x2000, 3136, 64, 288
SYNC.WAIT_MXU
```

### Example 3: Multi-Head Attention

```asm
# Q @ K.T -> scores
TENSOR.MATMUL_TRANS_B 0x7000, 0x0000, 0x1000, 16, 16, 64
SYNC.WAIT_MXU

# Scale by 1/sqrt(d_k)
VECTOR.SCALE_Q8 0x7000, 0x7000, 32, 256   # 32 = 256/sqrt(64)
SYNC.WAIT_VPU

# Softmax
VECTOR.SOFTMAX_P1 0x8000, 0x7000, 16, 16
SYNC.WAIT_VPU
VECTOR.SOFTMAX_P2 0x7000, 0x7000, 0x8000, 16, 16
SYNC.WAIT_VPU
VECTOR.SOFTMAX_P3 0x7000, 0x7000, 0x8000, 16, 16
SYNC.WAIT_VPU

# scores @ V -> output
TENSOR.GEMM 0x6000, 0x7000, 0x2000, 16, 64, 16
SYNC.WAIT_MXU
```

### Example 4: MobileNet Block

```asm
# Depthwise conv 3x3
TENSOR.DEPTHWISE_CONV 0x6000, 0x0000, 0x2000, 32, 56, 56, 3, 3, 1, 1, 1
SYNC.WAIT_MXU

# BatchNorm
VECTOR.BATCHNORM 0x6000, 0x6000, 0x3000, 0x3080, 32, 3136
SYNC.WAIT_VPU

# ReLU6
VECTOR.RELU6 0x6000, 0x6000, 100352
SYNC.WAIT_VPU
```

---

## Performance Guidelines

### Maximizing MXU Utilization

1. **Tile to 16×16**: Best efficiency when M, N, K are multiples of 16
2. **Weight-stationary**: Load weights once, stream activations
3. **Double-buffer**: Overlap DMA with compute

### Memory Bandwidth

| Operation | Bytes/Op | Arithmetic Intensity |
|-----------|----------|---------------------|
| GEMM (16×16) | 512 | 16 |
| Conv2D 3×3 | 9C | ~1.8 |
| ReLU | 2 | 0.5 |

### Latency Hiding

```asm
# Double-buffered GEMM
DMA.LOAD_1D 0x0000, 0x100000, 512   # Load tile 0
SYNC.WAIT_DMA

.loop:
  DMA.LOAD_1D 0x1000, 0x100200, 512 # Load tile 1 (async)
  TENSOR.GEMM 0x6000, 0x0000, ...   # Compute tile 0
  SYNC.WAIT_ALL
  
  DMA.LOAD_1D 0x0000, 0x100400, 512 # Load tile 2 (async)
  TENSOR.GEMM 0x6000, 0x1000, ...   # Compute tile 1
  SYNC.WAIT_ALL
```

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-01 | Initial release |

---

*Tensor Accelerator ISA Reference v1.0 - Confidential*

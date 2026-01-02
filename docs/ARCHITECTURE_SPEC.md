# Tensor Accelerator Architecture Specification

## Version 1.0

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Tensor Processing Cluster (TPC)](#tensor-processing-cluster)
4. [Matrix Unit (MXU)](#matrix-unit-mxu)
5. [Vector Processing Unit (VPU)](#vector-processing-unit-vpu)
6. [DMA Engine](#dma-engine)
7. [Memory Subsystem](#memory-subsystem)
8. [Control Architecture](#control-architecture)
9. [Network-on-Chip (NoC)](#network-on-chip)
10. [Performance Analysis](#performance-analysis)
11. [Power Considerations](#power-considerations)
12. [Supported Workloads](#supported-workloads)

---

## Executive Summary

The Tensor Accelerator is a high-performance neural network inference engine designed for edge and datacenter deployment. Key specifications:

| Parameter | Value |
|-----------|-------|
| Peak Performance | 2048 INT8 TOPS (4 TPCs @ 512 TOPS each) |
| On-chip SRAM | 8 MB (2 MB per TPC) |
| DDR Bandwidth | 128 GB/s (HBM2e) |
| Process Node | 7nm / 5nm target |
| Power Envelope | 15-75W (configurable) |

### Key Features

- **Scalable architecture**: 1-16 TPCs per chip
- **Flexible dataflow**: Weight-stationary, output-stationary configurable
- **INT8/INT4 support**: Native quantized inference
- **Transformer optimized**: Attention, LayerNorm, GELU accelerated
- **CNN optimized**: im2col, depthwise conv, pooling

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TENSOR ACCELERATOR TOP                           │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                    │
│  │  TPC 0  │  │  TPC 1  │  │  TPC 2  │  │  TPC 3  │                    │
│  │         │  │         │  │         │  │         │                    │
│  │  ┌───┐  │  │  ┌───┐  │  │  ┌───┐  │  │  ┌───┐  │                    │
│  │  │MXU│  │  │  │MXU│  │  │  │MXU│  │  │  │MXU│  │                    │
│  │  └───┘  │  │  └───┘  │  │  └───┘  │  │  └───┘  │                    │
│  │  ┌───┐  │  │  ┌───┐  │  │  ┌───┐  │  │  ┌───┐  │                    │
│  │  │VPU│  │  │  │VPU│  │  │  │VPU│  │  │  │VPU│  │                    │
│  │  └───┘  │  │  └───┘  │  │  └───┘  │  │  └───┘  │                    │
│  │  ┌───┐  │  │  ┌───┐  │  │  ┌───┐  │  │  ┌───┐  │                    │
│  │  │DMA│  │  │  │DMA│  │  │  │DMA│  │  │  │DMA│  │                    │
│  │  └───┘  │  │  └───┘  │  │  └───┘  │  │  └───┘  │                    │
│  │  SRAM   │  │  SRAM   │  │  SRAM   │  │  SRAM   │                    │
│  │  2 MB   │  │  2 MB   │  │  2 MB   │  │  2 MB   │                    │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘                    │
│       │            │            │            │                          │
│  ┌────┴────────────┴────────────┴────────────┴────┐                    │
│  │              Network-on-Chip (2x2 Mesh)         │                    │
│  └────────────────────────┬───────────────────────┘                    │
│                           │                                             │
│  ┌────────────────────────┴───────────────────────┐                    │
│  │         Global Command Processor (GCP)          │                    │
│  │              + AXI4 Interface                   │                    │
│  └────────────────────────┬───────────────────────┘                    │
│                           │                                             │
└───────────────────────────┼─────────────────────────────────────────────┘
                            │
                   ┌────────┴────────┐
                   │   DDR / HBM2e   │
                   │   256-512 GB    │
                   └─────────────────┘
```

---

## Tensor Processing Cluster

Each TPC is an independent compute unit with:

### TPC Block Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    TENSOR PROCESSING CLUSTER                  │
├──────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Local Command Processor (LCP)               │ │
│  │  ┌──────────┐  ┌───────────┐  ┌───────────┐            │ │
│  │  │ Inst Mem │  │ Decoder   │  │ Sequencer │            │ │
│  │  │  4 KB    │  │           │  │           │            │ │
│  │  └──────────┘  └───────────┘  └───────────┘            │ │
│  └─────────────────────────────────────────────────────────┘ │
│                              │                                │
│           ┌──────────────────┼──────────────────┐            │
│           ▼                  ▼                  ▼            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │  SYSTOLIC   │    │   VECTOR    │    │    DMA      │      │
│  │   ARRAY     │    │    UNIT     │    │   ENGINE    │      │
│  │   (MXU)     │    │   (VPU)     │    │             │      │
│  │             │    │             │    │             │      │
│  │ 16×16 PEs   │    │ 256-wide    │    │ 2D strided  │      │
│  │ INT8 MAC    │    │ SIMD        │    │ AXI master  │      │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘      │
│         │                  │                  │              │
│  ┌──────┴──────────────────┴──────────────────┴──────┐      │
│  │               SRAM SUBSYSTEM (2 MB)                │      │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐           │      │
│  │  │ Act Buf │  │ Wgt Buf │  │ Scratch │           │      │
│  │  │ 512 KB  │  │ 768 KB  │  │ 768 KB  │           │      │
│  │  │ 2 banks │  │ 4 banks │  │ 4 banks │           │      │
│  │  └─────────┘  └─────────┘  └─────────┘           │      │
│  └──────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────┘
```

### TPC Specifications

| Parameter | Value |
|-----------|-------|
| Peak TOPS | 512 INT8 |
| MXU Size | 16×16 |
| VPU Width | 256 elements |
| SRAM Total | 2 MB |
| Instruction Memory | 4 KB (256 instructions) |
| Clock Frequency | 1 GHz target |

---

## Matrix Unit (MXU)

The MXU implements a weight-stationary systolic array for matrix multiplication.

### Systolic Array Architecture

```
                    Weight Loading (vertical)
                           │
              ┌────────────┼────────────┐
              │            ▼            │
              │   ┌───┐ ┌───┐ ┌───┐    │
    Act ────► │   │PE │→│PE │→│PE │→ ··· → Results
   Input      │   └───┘ └───┘ └───┘    │
              │     │     │     │      │
              │   ┌───┐ ┌───┐ ┌───┐    │
              │   │PE │→│PE │→│PE │→   │
              │   └───┘ └───┘ └───┘    │
              │     │     │     │      │
              │    ···   ···   ···     │
              │                        │
              └────────────────────────┘
                    16 × 16 PEs
```

### Processing Element (PE)

```
           ┌─────────────────────────────────┐
           │         MAC Processing Element   │
           │                                  │
           │   weight_in ──┐                  │
           │               │                  │
           │   act_in ───┬─┴─┐  ┌────────┐   │
           │             │ × │──│   +    │   │  acc_out
           │             └───┘  │ (acc)  │───┼───────►
           │                    └────────┘   │
           │   act_out ◄───── act_in         │
           │                                  │
           └─────────────────────────────────┘
```

### PE Specifications

| Parameter | Value |
|-----------|-------|
| Data Type | INT8 × INT8 → INT32 |
| Accumulator | 32-bit |
| Latency | 1 cycle (pipelined) |
| Weight Register | 8-bit, loadable |

### Dataflow

**Weight-Stationary Mode:**
1. Weights loaded column-by-column (K cycles)
2. Activations streamed row-by-row (M cycles)
3. Results accumulate in place
4. Output shifted out (N cycles)

**Total GEMM Latency:** K + M + N + 2×ARRAY_SIZE cycles

### Supported Operations

| Operation | Description | Mapping |
|-----------|-------------|---------|
| GEMM | C = A × B | Direct |
| Conv2D | Convolution | im2col + GEMM |
| DepthwiseConv | Channel-wise conv | Sequential per-channel |
| Attention | Q×K.T, Scores×V | Two GEMMs |

---

## Vector Processing Unit (VPU)

The VPU handles element-wise operations, activations, and reductions.

### VPU Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    VECTOR PROCESSING UNIT                       │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input A (256 × INT8)        Input B (256 × INT8)              │
│       │                           │                             │
│       ▼                           ▼                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              VECTOR ALU (256-wide SIMD)                  │   │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐        ┌─────┐        │   │
│  │  │ ADD │ │ MUL │ │ MAX │ │ CMP │  ...   │ LUT │        │   │
│  │  └─────┘ └─────┘ └─────┘ └─────┘        └─────┘        │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              REDUCTION TREE (log2 stages)                │   │
│  │                                                          │   │
│  │   256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1             │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                           │                             │
│       ▼                           ▼                             │
│  Vector Output (256)         Scalar Output (1)                 │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### VPU Specifications

| Parameter | Value |
|-----------|-------|
| SIMD Width | 256 elements |
| Data Types | INT8, INT16, INT32 |
| Throughput | 256 ops/cycle |
| Reduction Latency | 8 cycles |

### Activation Functions

| Function | Implementation | Throughput |
|----------|----------------|------------|
| ReLU | MAX(0, x) | 256/cycle |
| ReLU6 | CLAMP(x, 0, 6) | 256/cycle |
| Sigmoid | 256-entry LUT | 256/cycle |
| Tanh | 256-entry LUT | 256/cycle |
| GELU | x × sigmoid(1.702x) | 85/cycle |
| Swish | x × sigmoid(x) | 128/cycle |
| Softmax | 3-pass algorithm | Variable |

### Normalization

| Operation | Passes | Complexity |
|-----------|--------|------------|
| BatchNorm | 1 | O(N) |
| LayerNorm | 4 | O(N) per instance |
| GroupNorm | 4 | O(N) per group |

---

## DMA Engine

The DMA engine provides high-bandwidth data movement.

### DMA Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      DMA ENGINE                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐    ┌──────────────────┐              │
│  │ Command Queue    │    │ Descriptor Pool  │              │
│  │ (32 entries)     │    │ (64 descriptors) │              │
│  └────────┬─────────┘    └────────┬─────────┘              │
│           │                       │                         │
│           ▼                       ▼                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Address Generator                       │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│  │  │ 1D Mode  │  │ 2D Mode  │  │ Gather/  │          │   │
│  │  │ Linear   │  │ Strided  │  │ Scatter  │          │   │
│  │  └──────────┘  └──────────┘  └──────────┘          │   │
│  └─────────────────────────────────────────────────────┘   │
│                         │                                   │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              AXI4 Master Interface                   │   │
│  │  - 256-bit data width                               │   │
│  │  - Outstanding transactions: 16                      │   │
│  │  - Burst length: up to 256 beats                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### DMA Specifications

| Parameter | Value |
|-----------|-------|
| Data Width | 256 bits |
| Peak Bandwidth | 32 GB/s per TPC |
| Max Outstanding | 16 transactions |
| Max Burst | 256 beats (8 KB) |
| 2D Stride | 24-bit |

### Transfer Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| 1D Linear | Contiguous transfer | Weight loading |
| 2D Strided | Row-major with stride | Feature maps |
| Gather | Indexed access | Embedding lookup |
| Scatter | Indexed write | Sparse update |

---

## Memory Subsystem

### SRAM Organization

```
┌─────────────────────────────────────────────────────────────┐
│                    SRAM SUBSYSTEM (2 MB)                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ ACTIVATION BUFFER (512 KB)                           │    │
│  │  ┌─────────────┐  ┌─────────────┐                   │    │
│  │  │  Bank 0     │  │  Bank 1     │                   │    │
│  │  │  256 KB     │  │  256 KB     │                   │    │
│  │  │  (Ping)     │  │  (Pong)     │                   │    │
│  │  └─────────────┘  └─────────────┘                   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ WEIGHT BUFFER (768 KB)                               │    │
│  │  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐        │    │
│  │  │Bank 0 │  │Bank 1 │  │Bank 2 │  │Bank 3 │        │    │
│  │  │192 KB │  │192 KB │  │192 KB │  │192 KB │        │    │
│  │  └───────┘  └───────┘  └───────┘  └───────┘        │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ SCRATCH BUFFER (768 KB)                              │    │
│  │  - Output accumulation                               │    │
│  │  - Intermediate results                              │    │
│  │  - im2col workspace                                  │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Memory Hierarchy

| Level | Size | Bandwidth | Latency |
|-------|------|-----------|---------|
| Register File | 16 KB | 1024 GB/s | 0 cycles |
| SRAM | 2 MB/TPC | 512 GB/s | 1-2 cycles |
| NoC | - | 256 GB/s | 4-8 cycles |
| DDR/HBM | 256-512 GB | 128 GB/s | 50-200 cycles |

---

## Control Architecture

### Command Processor Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│              GLOBAL COMMAND PROCESSOR (GCP)                  │
│                                                              │
│  - Receives commands from host via AXI4-Lite                │
│  - Dispatches work to TPCs                                   │
│  - Manages global synchronization                            │
│  - Handles interrupts and status                             │
│                                                              │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│    LCP 0      │   │    LCP 1      │   │    LCP 2,3..  │
│               │   │               │   │               │
│ - Instruction │   │ - Instruction │   │ - Instruction │
│   fetch/decode│   │   fetch/decode│   │   fetch/decode│
│ - Loop control│   │ - Loop control│   │ - Loop control│
│ - Unit control│   │ - Unit control│   │ - Unit control│
└───────────────┘   └───────────────┘   └───────────────┘
```

### Instruction Pipeline

```
┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐
│ Fetch │→│Decode │→│ Issue │→│Execute│→│Commit │
└───────┘  └───────┘  └───────┘  └───────┘  └───────┘
   1 cyc     1 cyc     1 cyc    Variable    1 cyc
```

---

## Network-on-Chip

### NoC Topology

```
2×2 Mesh (4 TPCs):

    ┌─────┐    ┌─────┐
    │TPC 0│────│TPC 1│
    └──┬──┘    └──┬──┘
       │          │
    ┌──┴──┐    ┌──┴──┐
    │TPC 2│────│TPC 3│
    └─────┘    └─────┘

4×4 Mesh (16 TPCs):

    ┌─────┬─────┬─────┬─────┐
    │ 0   │ 1   │ 2   │ 3   │
    ├─────┼─────┼─────┼─────┤
    │ 4   │ 5   │ 6   │ 7   │
    ├─────┼─────┼─────┼─────┤
    │ 8   │ 9   │ 10  │ 11  │
    ├─────┼─────┼─────┼─────┤
    │ 12  │ 13  │ 14  │ 15  │
    └─────┴─────┴─────┴─────┘
```

### NoC Specifications

| Parameter | Value |
|-----------|-------|
| Topology | 2D Mesh |
| Link Width | 256 bits |
| Routing | XY Dimension-Order |
| Virtual Channels | 4 |
| Buffer Depth | 4 flits per VC |

---

## Performance Analysis

### Peak Performance

| Configuration | INT8 TOPS | FP16 TFLOPS |
|---------------|-----------|-------------|
| 1 TPC | 512 | 64 |
| 4 TPC | 2048 | 256 |
| 16 TPC | 8192 | 1024 |

### Roofline Model

```
Performance (TOPS)
    │
2048│                           ┌─────────────
    │                          /
    │                         /
    │                        /
1024│                       /
    │                      /
    │                     /
 512│                    /
    │                   /
    │                  /
    │_________________/________________________
                     16              128
              Arithmetic Intensity (Ops/Byte)
```

### Model Performance Estimates

| Model | Params | MACs | Latency (4 TPC) |
|-------|--------|------|-----------------|
| ResNet-50 | 25M | 4.1G | 1.2 ms |
| MobileNetV2 | 3.4M | 0.3G | 0.3 ms |
| BERT-Base | 110M | 22G | 5.5 ms |
| GPT-2 Small | 124M | 25G | 6.2 ms |

---

## Power Considerations

### Power Breakdown (4 TPC @ 1 GHz)

| Component | Power | Percentage |
|-----------|-------|------------|
| MXU (4×) | 15W | 42% |
| VPU (4×) | 6W | 17% |
| SRAM (8 MB) | 4W | 11% |
| DMA (4×) | 3W | 8% |
| NoC | 2W | 6% |
| Control | 2W | 6% |
| I/O + PLL | 3W | 8% |
| **Total** | **35W** | 100% |

### Power Modes

| Mode | Performance | Power |
|------|-------------|-------|
| Full | 100% | 35W |
| Eco | 50% | 18W |
| Idle | 0% | 2W |
| Sleep | 0% | 0.5W |

---

## Supported Workloads

### CNN Architectures

| Architecture | Status | Notes |
|--------------|--------|-------|
| ResNet-18/34/50 | ✅ Full | Standard inference |
| MobileNetV1/V2/V3 | ✅ Full | Depthwise conv |
| EfficientNet | ✅ Full | Swish, SE blocks |
| VGG-16/19 | ✅ Full | Large weight buffers |
| YOLO v3/v4/v5 | ✅ Full | Detection heads |

### Transformer Architectures

| Architecture | Status | Notes |
|--------------|--------|-------|
| BERT-Base | ✅ Full | Multi-head attention |
| GPT-2 | ✅ Full | Causal attention |
| ViT | ✅ Full | Patch embedding |
| CLIP | ⚠️ Partial | Image encoder only |

### Supported Operations

| Category | Operations |
|----------|------------|
| Linear | GEMM, Conv2D, DepthwiseConv, Conv1D |
| Activation | ReLU, ReLU6, GELU, Swish, Sigmoid, Tanh |
| Normalization | BatchNorm, LayerNorm, GroupNorm, InstanceNorm |
| Pooling | MaxPool, AvgPool, GlobalAvgPool, AdaptivePool |
| Attention | MultiHeadAttention, ScaledDotProduct |
| Shape | Reshape, Transpose, Concat, Split, Squeeze |

---

## Appendix: Register Map

### TPC Control Registers (per TPC)

| Offset | Name | Description |
|--------|------|-------------|
| 0x000 | CTRL | Control register |
| 0x004 | STATUS | Status register |
| 0x008 | PC | Program counter |
| 0x00C | INSTR_ADDR | Instruction base address |
| 0x010 | MXU_CTRL | MXU control |
| 0x014 | VPU_CTRL | VPU control |
| 0x018 | DMA_CTRL | DMA control |
| 0x01C | INT_STATUS | Interrupt status |
| 0x020 | INT_ENABLE | Interrupt enable |

### Global Registers

| Offset | Name | Description |
|--------|------|-------------|
| 0x1000 | GCP_CTRL | Global control |
| 0x1004 | GCP_STATUS | Global status |
| 0x1008 | TPC_ENABLE | TPC enable bitmap |
| 0x100C | SYNC_CTRL | Synchronization control |
| 0x1010 | PERF_CNT0 | Performance counter 0 |
| 0x1014 | PERF_CNT1 | Performance counter 1 |

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-01 | Initial specification |

---

*Tensor Accelerator Architecture Specification v1.0 - Confidential*

# Building an AI Accelerator from Scratch
## Part 1: Architecture Deep Dive — Design Decisions and Microarchitecture

*A comprehensive exploration of the architectural choices behind a production-quality tensor processing unit*

---

## Introduction

This article series documents the complete journey of building an AI accelerator from scratch—not using off-the-shelf IP cores, but designing every RTL module and compiler pass by hand. This first part focuses on the architecture: the key questions we asked, the decisions we made, and why.

**What this series covers:**
- **Part 1** (this article): Architecture deep dive — GCP, TPC, MXU, VPU microarchitecture
- **Part 2**: Python functional models and RTL implementation
- **Part 3**: Verification plan, test coverage, and open-source EDA tooling
- **Part 4**: Complete software flow from ONNX to binary

---

## The First Question: What Are We Actually Building?

Before writing a single line of RTL, we needed to nail down the fundamental constraints:

### Key Questions Asked

**Q1: Inference only, or training support?**
*Decision*: Inference only for v1. Training requires backward pass computation, gradient accumulation, and 2× the memory bandwidth. For an FPGA prototype, inference is the right scope.

**Q2: What model types?**
*Decision*: Transformers/LLMs as the primary target, with CNN support. This means optimizing for GEMM (the backbone of both architectures) plus attention-specific operations like softmax.

**Q3: What precision?**
*Decision*: INT8 primary with INT32 accumulation. This gives 95%+ of FP32 accuracy for inference while being 4× more efficient in compute and memory bandwidth than FP32.

**Q4: Batch size assumptions?**
*Decision*: Batch 1-8, latency-optimized. Edge and interactive use cases don't batch heavily—they need low latency per request.

### Design Goals Summary

| Parameter | Decision | Rationale |
|-----------|----------|-----------|
| Workload | Inference (LLM/Transformer focus) | Simpler than training, latency-sensitive |
| Batch size | 1-8 (latency-optimized) | Edge/interactive use case |
| Precision | INT8 primary, INT32 accumulation | Good accuracy/efficiency tradeoff |
| Target platform | Xilinx FPGA (prototyping) | Validate before ASIC investment |
| Power envelope | FPGA-constrained (~10-25W) | Prototype, not production ASIC |

---

## The Fundamental Architecture Question: Systolic Array vs. Sea of MACs

This was *the* defining architectural decision. Let me walk through the discussion exactly as it happened.

### The Question

> "systolic array or sea of MACs?"

### The Tradeoffs We Analyzed

**Systolic Array (TPU-style)**
```
Data flows through a 2D grid of MACs
Weight-stationary or output-stationary dataflow
Good for dense matrix multiplies
Simpler control, but data movement can be tricky
```

**Sea of MACs / Spatial Array (Cerebras-style)**
```
More flexible interconnect between PEs
Each PE has local SRAM and can communicate with neighbors
Handles irregular shapes and sparsity better
But: routing is a nightmare, higher energy per MAC
```

### Quantitative Analysis

| Metric | Systolic (256×256) | Sea of MACs (64K PEs) |
|--------|-------------------|----------------------|
| Peak TOPS (INT8) | 131 TOPS | 131 TOPS |
| Utilization @ 256×256 GEMM | ~95% | ~80-90% (routing overhead) |
| Utilization @ 64×64 GEMM | ~6% | ~70-80% |
| Control overhead | ~1% of area | ~10-20% of area |
| Energy per MAC | ~0.1 pJ (register) | ~0.5-2 pJ (router + SRAM) |

### The Decision

**We chose systolic array** for these reasons:

1. **FPGA-friendly**: Simpler control fits better in FPGA fabric
2. **Tractable complexity**: Debuggable without drowning in NoC design
3. **Proven architecture**: TPU, Groq, and most production accelerators use systolic
4. **Dense workloads**: Transformers and CNNs are mostly dense GEMMs
5. **Energy efficiency**: Neighbor-to-neighbor data movement is optimal

The insight from industry: Modern chips use **hierarchical hybrids**—systolic arrays *inside* compute clusters, connected by a mesh NoC *between* clusters. This is exactly what we implemented.

---

## Top-Level Architecture: The Tensor Processing Cluster (TPC)

### The Multi-TPC Question

> "What about scaling to hundreds of PetaOPS?"

A single 16×16 array can't achieve datacenter-scale throughput. The architecture must be **scalable from the start**.

### The Decision: Replicated TPCs with NoC

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        TENSOR ACCELERATOR TOP                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐ │
│  │                     Global Command Processor (GCP)                        │ │
│  │                  Distributes work across TPCs, handles sync              │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                         │
│  ┌───────────────────────────────────┼───────────────────────────────────┐    │
│  │                         Network-on-Chip (NoC)                          │    │
│  │                    2×2 Mesh with XY Routing                            │    │
│  │                                                                         │    │
│  │    ┌──────────────┐         ┌──────────────┐                           │    │
│  │    │    TPC 0     │◄═══════►│    TPC 1     │                           │    │
│  │    │  ┌────────┐  │         │  ┌────────┐  │                           │    │
│  │    │  │ 16×16  │  │         │  │ 16×16  │  │                           │    │
│  │    │  │Systolic│  │         │  │Systolic│  │                           │    │
│  │    │  │ Array  │  │         │  │ Array  │  │                           │    │
│  │    │  └────────┘  │         │  └────────┘  │                           │    │
│  │    │  + VPU, DMA  │         │  + VPU, DMA  │                           │    │
│  │    │  + 2MB SRAM  │         │  + 2MB SRAM  │                           │    │
│  │    └──────┬───────┘         └──────┬───────┘                           │    │
│  │           ║                        ║                                   │    │
│  │    ┌──────────────┐         ┌──────────────┐                           │    │
│  │    │    TPC 2     │◄═══════►│    TPC 3     │                           │    │
│  │    │  (same)      │         │  (same)      │                           │    │
│  │    └──────────────┘         └──────────────┘                           │    │
│  └───────────────────────────────────────────────────────────────────────┘    │
│                                                                                │
│  ┌──────────────────────────────────────────────────────────────────────────┐ │
│  │                    AXI4 Memory Interface (DDR4/HBM)                       │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Why 4 TPCs in a 2×2 Mesh?

| Reason | Explanation |
|--------|-------------|
| **Scalability** | Architecture naturally extends to 4×4, 8×8, or larger grids |
| **Parallelism** | Four TPCs can process different layers or batch elements simultaneously |
| **Fault isolation** | A hung TPC doesn't bring down the system |
| **FPGA fit** | 4 TPCs with 16×16 arrays fits in VCU118 |

---

## Global Command Processor (GCP) Microarchitecture

### The Control Hierarchy Question

> "How do we coordinate multiple TPCs without making control a bottleneck?"

### The Decision: Two-Level Control

**Global Command Processor (GCP)**: Distributes work, handles global sync
**Local Command Processor (LCP)**: Per-TPC instruction execution

This separation means:
- GCP sets up the high-level schedule once
- Each LCP runs independently, fetching from its own instruction memory
- TPCs only synchronize at explicit BARRIER instructions

### GCP Responsibilities

| Function | Implementation |
|----------|----------------|
| Work distribution | Writes program counters for each TPC |
| Global barriers | Monitors per-TPC done signals |
| Host interface | AXI-Lite slave for configuration |
| Interrupt generation | Signals completion to host |

### GCP Register Map

| Offset | Name | R/W | Description |
|--------|------|-----|-------------|
| 0x000 | CTRL | R/W | `[0]` start, `[15:8]` tpc_enable mask |
| 0x004 | STATUS | R | `[3:0]` busy, `[11:8]` done, `[19:16]` error |
| 0x008 | IRQ_EN | R/W | `[0]` completion IRQ enable |
| 0x00C | IRQ_STATUS | R/W1C | `[0]` completion IRQ (write 1 to clear) |
| 0x100 | TPC0_PC | R/W | TPC0 start program counter |
| 0x110 | TPC1_PC | R/W | TPC1 start program counter |
| 0x120 | TPC2_PC | R/W | TPC2 start program counter |
| 0x130 | TPC3_PC | R/W | TPC3 start program counter |

---

## Tensor Processing Cluster (TPC) Microarchitecture

Each TPC is a self-contained compute engine with these components:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Tensor Processing Cluster                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Local Command Processor (LCP)                 │  │
│  │  • Instruction fetch from local IMEM                      │  │
│  │  • Decode 128-bit instructions                            │  │
│  │  • Hardware loop stack (4 levels)                         │  │
│  │  • Scoreboard for dependency tracking                     │  │
│  │  • Dispatch to MXU/VPU/DMA                               │  │
│  └────────────────────────┬──────────────────────────────────┘  │
│                           │                                      │
│     ┌─────────────────────┼─────────────────────┐               │
│     │                     │                     │               │
│     ▼                     ▼                     ▼               │
│  ┌─────────┐       ┌─────────────┐       ┌─────────┐           │
│  │   MXU   │       │    VPU      │       │   DMA   │           │
│  │ 16×16   │       │  64-lane    │       │  2D     │           │
│  │Systolic │       │   SIMD      │       │Strided  │           │
│  │ Array   │       │             │       │         │           │
│  └────┬────┘       └──────┬──────┘       └────┬────┘           │
│       │                   │                   │                 │
│       └───────────────────┴───────────────────┘                 │
│                           │                                      │
│                           ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              SRAM Subsystem (16 Banks × 2MB)               │  │
│  │  • Multi-port arbitration (MXU > VPU > DMA priority)      │  │
│  │  • XOR-based bank mapping (reduce stride conflicts)       │  │
│  │  • 256-bit wide access (32 bytes per read)               │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Key TPC Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Systolic array | 16×16 | 256 MACs/cycle, fits FPGA |
| VPU lanes | 64 | Matches common tensor dimensions |
| SRAM | 2 MB (16 banks × 128KB) | Fits several tiles for double-buffering |
| Instruction width | 128 bits | Rich encoding, no variable-length complexity |
| Loop stack depth | 4 | Sufficient for tiled nested loops |

---

## Matrix eXecution Unit (MXU) — The Systolic Array

### The Dataflow Question

> "Weight-stationary, output-stationary, or row-stationary?"

### Analysis of Options

| Dataflow | Stationary | Streams | Best For |
|----------|------------|---------|----------|
| Weight-stationary | Weights | Activations, Psums | Inference (weights reused across batches) |
| Output-stationary | Partial sums | Weights, Activations | Training (accumulate gradients) |
| Row-stationary | Varies | Varies | Flexible, complex control |

### The Decision: Weight-Stationary

**Rationale:**
1. **Inference focus**: Weights are constant during inference—load once per layer
2. **Simpler control**: No complex psum accumulation across tiles
3. **Better for small batches**: Edge inference often has batch=1
4. **Proven**: Google's TPUv1-v3 all use weight-stationary

### MXU Microarchitecture

```
        ─── Activations flow horizontally ───►
       
    │   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
    │   │ PE  │──►│ PE  │──►│ PE  │──►│ PE  │──► partial sums out
    │   │ w00 │   │ w01 │   │ w02 │   │ w03 │
  W │   └──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘
  e │      │         │         │         │
  i │      ▼         ▼         ▼         ▼
  g │   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
  h │   │ PE  │──►│ PE  │──►│ PE  │──►│ PE  │──►
  t │   │ w10 │   │ w11 │   │ w12 │   │ w13 │
  s │   └──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘
    │      │         │         │         │
  f │      ▼         ▼         ▼         ▼
  l │   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
  o │   │ PE  │──►│ PE  │──►│ PE  │──►│ PE  │──►
  w │   │ w20 │   │ w21 │   │ w22 │   │ w23 │
    │   └─────┘   └─────┘   └─────┘   └─────┘
    ▼
```

### Key MXU Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Array size | 16×16 | Fits FPGA BRAM, good utilization for common layers |
| Input precision | INT8 | Industry standard for inference |
| Accumulator width | INT32 | Prevents overflow for K up to 65K |
| Weight loading | Column broadcast | Load one column per cycle (16 cycles total) |
| Activation skewing | Input-side registers | Proper diagonal wavefront alignment |
| Output draining | Row-wise with de-skewing | N cycles to drain, realigned at output |

### The Skewing Question

> "How do we handle the timing alignment in a systolic array?"

**Input Skewing**: Activations must enter with staggered timing:
```
Cycle 0: Row 0 gets A[0,0]
Cycle 1: Row 0 gets A[0,1], Row 1 gets A[1,0]
Cycle 2: Row 0 gets A[0,2], Row 1 gets A[1,1], Row 2 gets A[2,0]
...
```

**Output De-skewing**: Results emerge staggered, must be realigned:
```
Column 0: 6 delay cycles (2*(N-1-0) for N=4)
Column 1: 4 delay cycles
Column 2: 2 delay cycles
Column 3: 0 delay cycles
```

This was one of the trickiest parts to get right—off-by-one errors in skewing cause silent data corruption.

---

## Vector Processing Unit (VPU) Microarchitecture

### The VPU Question

> "What operations can't the systolic array handle?"

**Answer**: Everything that's not a dense matrix multiply:
- Activation functions (ReLU, GELU, Sigmoid, Softmax)
- Normalization (LayerNorm, BatchNorm)
- Elementwise operations (Add, Multiply for residuals)
- Reductions (sum, max for softmax)

### VPU Design Decisions

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| SIMD lanes | 64 | Matches common tensor dimensions, power of 2 |
| Lane width | 16-bit (BF16) | Full precision for non-GEMM ops |
| Vector registers | 32 × 64-wide | 4KB register file, enough for complex ops |
| Pipeline depth | 3-4 stages | Balanced for FPGA timing |

### Supported Operations

| Category | Operations | Notes |
|----------|------------|-------|
| Arithmetic | ADD, SUB, MUL, MADD (fused) | Basic vector math |
| Activation | RELU, GELU (approx), SILU, Sigmoid | Via LUT for transcendentals |
| Reduction | SUM, MAX, MIN (horizontal) | For softmax, pooling |
| Movement | LOAD, STORE, BROADCAST, PERMUTE | Data manipulation |
| Conversion | INT8↔BF16, quantize/dequantize | Precision conversion |

### The GELU/Softmax Question

> "How do we compute transcendental functions in hardware?"

**Decision**: 256-entry lookup tables with linear interpolation

For GELU: `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`
- This is too complex for direct hardware implementation
- We use a 256-entry LUT indexed by the INT8 input value
- Error is <1% which is acceptable for inference

For Softmax (3-pass implementation):
```
Pass 1: Find max (reduction across vector)
Pass 2: Compute exp(x - max) via LUT
Pass 3: Normalize (divide by sum)
```

---

## DMA Engine Microarchitecture

### The DMA Question

> "How do we hide memory latency?"

**Answer**: 2D strided transfers with double-buffering support.

### DMA Design Decisions

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Addressing | 2D strided | Essential for tiled tensor access |
| Max burst | 256 bytes | AXI4 compatible |
| Channels | 2 | Concurrent load/store for double-buffering |
| Features | Zero-padding, transpose-on-fly | Handle edge tiles and layout conversion |

### 2D Transfer Format

```
DMA Command:
- src_addr: Starting address in DDR/HBM
- dst_addr: Starting address in SRAM
- rows, cols: 2D shape
- src_stride, dst_stride: Bytes between rows
```

This enables efficient:
- Column extraction from row-major matrices
- Channel-first to channel-last conversion
- Padding insertion/removal for edge tiles

### The Double-Buffering Pattern

```
Time →
┌─────────┬─────────┬─────────┬─────────┬─────────┐
│ Load A0 │ Comp A0 │ Comp A1 │ Comp A2 │ Comp A3 │  Compute
│         │ Load A1 │ Load A2 │ Load A3 │ Load A4 │  DMA
└─────────┴─────────┴─────────┴─────────┴─────────┘
          │←─ Overlap: DMA hidden by compute ─→│
```

---

## SRAM Subsystem Microarchitecture

### The Banking Question

> "How do we achieve high bandwidth without conflicts?"

### SRAM Design Decisions

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Total capacity | 2 MB per TPC | Fits several tiles for double-buffering |
| Bank count | 16 | Parallel access for MXU rows |
| Bank width | 256 bits | 32 bytes per access |
| Ports per bank | 2 (dual-port) | Concurrent read/write |
| Bank mapping | XOR-based | Reduce stride conflicts |

### The XOR-Based Bank Mapping

**Problem**: Simple modulo mapping causes all elements of a matrix column to map to the same bank—serializing column access.

**Solution**: XOR upper and lower address bits:
```verilog
bank = addr[3:0] ^ addr[15:12];
```

This distributes strided accesses across banks. Same technique used in GPU shared memory.

### SRAM Address Space Layout

```
0x00000 - 0x3FFFF: Buffer 0 Activations (256 KB)
0x40000 - 0x7FFFF: Buffer 0 Weights (256 KB)
0x80000 - 0xBFFFF: Buffer 1 Activations (256 KB)
0xC0000 - 0xFFFFF: Buffer 1 Weights (256 KB)
0x100000 - 0x17FFFF: Output (512 KB)
0x180000 - 0x1FFFFF: Scratch (512 KB)
```

### Arbitration Priority

When multiple units request the same bank:
1. **MXU** (highest): Compute should never stall
2. **VPU**: Activations are on critical path
3. **DMA** (lowest): Can absorb latency

---

## Instruction Set Architecture (ISA)

### The ISA Question

> "Why not use RISC-V with custom extensions?"

### Analysis

| Aspect | Custom ISA | RISC-V + Extensions |
|--------|-----------|---------------------|
| Instruction width | 128-bit (rich encoding) | 32-bit base (cramped) |
| Decode complexity | Single fixed format | Multiple formats, extensions |
| Hardware loops | Native support | Requires software emulation |
| Coarse-grained ops | Natural fit | Awkward mapping |
| Toolchain | Custom (more work) | Leverage existing (less work) |

### The Decision: Custom 128-bit ISA

**Rationale**: For a dedicated accelerator where every transistor counts, custom ISA wins. For a more general system with CPU integration needs, RISC-V would be attractive.

### Instruction Format

```
Bit Position:  127    120 119    112 111      96 95       80 79       64
               ┌────────┬────────┬───────────┬───────────┬───────────┐
               │ Opcode │ Subop  │    Dst    │   Src0    │   Src1    │
               │ (8b)   │ (8b)   │   (16b)   │   (16b)   │   (16b)   │
               └────────┴────────┴───────────┴───────────┴───────────┘

Bit Position:  63       48 47       32 31       16 15        0
               ┌───────────┬───────────┬───────────┬───────────┐
               │   Dim_M   │   Dim_N   │   Dim_K   │   Flags   │
               │   (16b)   │   (16b)   │   (16b)   │   (16b)   │
               └───────────┴───────────┴───────────┴───────────┘
```

### Instruction Classes

| Opcode | Class | Key Operations |
|--------|-------|----------------|
| 0x01 | TENSOR | GEMM, GEMM_ACC |
| 0x02 | VECTOR | ADD, MUL, RELU, GELU, SOFTMAX |
| 0x03 | DMA | LOAD_2D, STORE_2D |
| 0x04 | SYNC | WAIT_MXU, WAIT_VPU, WAIT_DMA |
| 0x05 | LOOP | Hardware loop start |
| 0x06 | ENDLOOP | Hardware loop end |
| 0x07 | BARRIER | Multi-TPC synchronization |
| 0xFF | HALT | Stop execution |

### Hardware Loops

**Why hardware loops?**
Amortizing instruction fetch is critical. A tiled GEMM might execute 4096 inner loop iterations. With software loops, that's 4096 instruction fetches. With hardware loops, it's ~10 instructions with the loop count in a register.

```
LOOP count           # Push to loop stack
    # ... loop body ...
ENDLOOP              # Decrement counter, branch if not zero
```

Maximum nesting: 4 levels (sufficient for M/N/K tiling)

---

## Summary: The Architecture Decisions

| Component | Decision | Key Rationale |
|-----------|----------|---------------|
| **Compute style** | Systolic array | Simple control, FPGA-friendly, proven |
| **Dataflow** | Weight-stationary | Inference-optimized, weights load once |
| **Array size** | 16×16 | Fits FPGA, good utilization |
| **Precision** | INT8 / INT32 acc | Industry standard, prevents overflow |
| **TPC count** | 4 (2×2 mesh) | Scalable, fits target FPGA |
| **VPU lanes** | 64 | Matches tensor dimensions |
| **SRAM** | 2MB, 16 banks | Fits tiles, XOR mapping |
| **ISA** | Custom 128-bit | Rich encoding, hardware loops |
| **Control** | GCP + LCP | Hierarchy scales to many TPCs |

---

## What's Next

In **Part 2**, we'll dive into the implementation: cycle-accurate Python models that validated the algorithms, and the RTL that implements them. We'll see exactly how the systolic array timing works, how we debugged the skewing logic, and how the Python model caught bugs before we ever ran a Verilog simulation.

---

*All code is available on GitHub. Questions or feedback? Let's connect!*

**Tags**: #AI #HardwareDesign #FPGA #ASIC #ComputerArchitecture #Semiconductor #SystemDesign #TPU

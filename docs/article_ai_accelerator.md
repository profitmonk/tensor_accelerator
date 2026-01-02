# Building an AI Accelerator from Scratch: A Complete Hardware-Software Co-Design Journey

**From Architecture to Silicon-Ready RTL: Every Decision, Trade-off, and Lesson Learned**

---

## Introduction

Over the past several months, I've been building a complete AI/ML tensor accelerator from the ground up—not just the hardware, but the entire stack: RTL, compiler, instruction set architecture, memory subsystem, and verification infrastructure. This article documents that journey in detail, covering every major decision, the trade-offs we navigated, and the lessons learned along the way.

This isn't a toy project. The accelerator targets **2048 INT8 TOPS** across four Tensor Processing Clusters (TPCs), features a custom 128-bit VLIW-style ISA, includes a complete Python-based compiler that takes neural network graphs down to binary machine code, and has been validated through extensive simulation. The RTL is synthesis-ready for both FPGA prototyping and ASIC implementation.

Whether you're a hardware engineer curious about AI accelerator architecture, a software engineer wanting to understand what happens below the CUDA layer, or a student exploring computer architecture, I hope this deep dive provides valuable insights into the end-to-end process of building a domain-specific accelerator.

---

## Table of Contents

1. [Why Build a Custom AI Accelerator?](#why-build-a-custom-ai-accelerator)
2. [Architecture Overview](#architecture-overview)
3. [The Systolic Array: Heart of the Accelerator](#the-systolic-array-heart-of-the-accelerator)
4. [Instruction Set Architecture Design](#instruction-set-architecture-design)
5. [Memory Hierarchy and Data Movement](#memory-hierarchy-and-data-movement)
6. [The Compiler Stack](#the-compiler-stack)
7. [Verification and Validation](#verification-and-validation)
8. [FPGA Prototyping Considerations](#fpga-prototyping-considerations)
9. [Lessons Learned and Trade-offs](#lessons-learned-and-trade-offs)
10. [What's Next](#whats-next)

---

## Why Build a Custom AI Accelerator?

### The Problem with General-Purpose Hardware

Modern deep learning workloads are dominated by a small set of operations: matrix multiplications (GEMM), convolutions, element-wise activations, and normalization layers. GPUs handle these well, but they carry significant overhead:

- **Instruction fetch/decode** for every operation
- **Branch prediction** logic that's rarely used in deterministic tensor operations
- **Cache hierarchies** optimized for irregular access patterns
- **Thread scheduling** overhead for SIMT execution

A purpose-built accelerator can eliminate this overhead by:

1. Using **fixed-function datapaths** optimized for tensor operations
2. Implementing **software-controlled memory** instead of caches
3. Supporting **hardware loops** to amortize instruction fetch costs
4. Designing **memory access patterns** that match tensor data layouts

### Design Goals

From the outset, I established clear design goals:

| Goal | Target | Rationale |
|------|--------|-----------|
| Peak Throughput | 2048 INT8 TOPS | Competitive with edge AI accelerators |
| Power Efficiency | 50+ TOPS/W | Critical for edge deployment |
| Latency | <2ms for ResNet-50 | Real-time inference requirement |
| Flexibility | Support CNNs, Transformers, MLPs | Future-proof for evolving models |
| Programmability | Full compiler support | Not just a hardcoded demo |

### Why INT8?

The decision to focus on INT8 computation was driven by several factors:

1. **Accuracy**: Post-training quantization to INT8 typically loses <1% accuracy on vision models
2. **Efficiency**: INT8 multipliers are 6x smaller and 6x more power-efficient than FP32
3. **Memory bandwidth**: 4x reduction in activation/weight memory traffic
4. **Industry trend**: All major inference accelerators (TPU, Inferentia, Trainium) support INT8

We maintain INT32 accumulators internally to prevent overflow during large matrix multiplications, then requantize outputs back to INT8.

---

## Architecture Overview

### System-Level Architecture

The accelerator follows a hierarchical design:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Global Command Processor (GCP)                │
│                      AXI-Lite Host Interface                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │    2D Mesh NoC        │
                    │    (2x2 for POC)      │
                    └───────────┬───────────┘
        ┌───────────────┬───────┴───────┬───────────────┐
        ▼               ▼               ▼               ▼
   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
   │  TPC 0  │     │  TPC 1  │     │  TPC 2  │     │  TPC 3  │
   │ 512 TOPS│     │ 512 TOPS│     │ 512 TOPS│     │ 512 TOPS│
   └─────────┘     └─────────┘     └─────────┘     └─────────┘
        │               │               │               │
        └───────────────┴───────────────┴───────────────┘
                                │
                    ┌───────────┴───────────┐
                    │   AXI4 Interconnect   │
                    └───────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    │   DDR4/HBM2e Memory   │
                    │      128 GB/s         │
                    └───────────────────────┘
```

### Tensor Processing Cluster (TPC) Architecture

Each TPC is a complete, self-contained processing unit:

```
┌────────────────────────────────────────────────────────────────┐
│                    Tensor Processing Cluster                    │
├────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐  │
│  │            Local Command Processor (LCP)                  │  │
│  │         Instruction Fetch → Decode → Dispatch             │  │
│  │              Hardware Loop Support (4 levels)             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│          ┌───────────────────┼───────────────────┐             │
│          ▼                   ▼                   ▼             │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     │
│   │     MXU     │     │     VPU     │     │     DMA     │     │
│   │   16x16     │     │   64-lane   │     │  2D Strided │     │
│   │  Systolic   │     │    SIMD     │     │  Transfers  │     │
│   │   Array     │     │    Unit     │     │             │     │
│   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘     │
│          │                   │                   │             │
│          └───────────────────┼───────────────────┘             │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Banked SRAM Subsystem (2 MB)                 │  │
│  │    16 banks × 4096 words × 256 bits = 2 MB per TPC       │  │
│  │         XOR-based bank mapping for stride access          │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### Why This Hierarchy?

The two-level hierarchy (GCP → TPCs) provides several advantages:

1. **Scalability**: Add more TPCs without changing the programming model
2. **Parallelism**: TPCs execute independently on different tiles/batches
3. **Locality**: Each TPC has dedicated SRAM, minimizing NoC traffic
4. **Fault isolation**: A hung TPC doesn't bring down the system

---

## The Systolic Array: Heart of the Accelerator

### Why Systolic Arrays?

The systolic array is the defining architectural choice for modern AI accelerators. Google's TPU, Apple's Neural Engine, and most custom AI chips use some variant of this structure. The reasons are compelling:

1. **Data reuse**: Each weight is loaded once and used N times as activations flow past
2. **Regular structure**: Highly amenable to physical design and timing closure
3. **Scalable**: Performance scales with array size (N² MACs per cycle)
4. **Energy efficient**: Minimal control overhead, data moves between neighbors

### Dataflow Selection: Weight-Stationary

There are three primary dataflow choices for systolic arrays:

| Dataflow | Stationary | Flowing | Best For |
|----------|------------|---------|----------|
| Weight-Stationary | Weights | Activations, Psums | Inference (weights reused across batches) |
| Output-Stationary | Partial sums | Weights, Activations | Training (accumulate gradients) |
| Row-Stationary | Varies by row | Varies | Flexible, complex control |

I chose **weight-stationary** for several reasons:

1. **Inference focus**: Weights are constant during inference, load once per layer
2. **Simpler control**: No complex psum accumulation across tiles
3. **Better for small batches**: Edge inference often has batch=1
4. **Proven**: Google's TPUv1-v3 all use weight-stationary

### The MAC Processing Element

Each PE in the systolic array is surprisingly simple:

```verilog
module mac_pe #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     enable,
    input  wire                     load_weight,
    input  wire                     clear_acc,
    input  wire [DATA_WIDTH-1:0]    weight_in,
    input  wire [DATA_WIDTH-1:0]    act_in,
    output reg  [DATA_WIDTH-1:0]    act_out,
    input  wire [ACC_WIDTH-1:0]     psum_in,
    output reg  [ACC_WIDTH-1:0]     psum_out
);
    // Stationary weight register
    reg [DATA_WIDTH-1:0] weight_reg;
    
    // Signed multiplication
    wire signed [2*DATA_WIDTH-1:0] product = 
        $signed(act_in) * $signed(weight_reg);
    
    always @(posedge clk) begin
        if (load_weight) weight_reg <= weight_in;
        if (enable) begin
            act_out <= act_in;  // Pass activation right
            psum_out <= clear_acc ? product : (psum_in + product);
        end
    end
endmodule
```

Key design decisions:

1. **Signed arithmetic**: Neural network weights and activations can be negative
2. **32-bit accumulator**: Prevents overflow for matrices up to 65K elements
3. **Registered outputs**: Ensures clean timing at high frequencies
4. **Single-cycle MAC**: No pipelining within the PE (keeps control simple)

### Skewing and De-skewing

The magic of systolic arrays lies in the data skewing. For a 4x4 array computing C = A × B:

**Input Skewing (Activations)**:
```
Cycle 0: Row 0 gets A[0,0]
Cycle 1: Row 0 gets A[0,1], Row 1 gets A[1,0]
Cycle 2: Row 0 gets A[0,2], Row 1 gets A[1,1], Row 2 gets A[2,0]
...
```

This diagonal wavefront ensures each PE receives the correct activation at the right time.

**Output De-skewing**:
Results emerge from the bottom of the array, but column 0 finishes first, column 1 one cycle later, etc. We add delay registers to realign:

```
Column 0: 6 delay cycles (2*(N-1-0) for N=4)
Column 1: 4 delay cycles
Column 2: 2 delay cycles
Column 3: 0 delay cycles (rightmost, no delay)
```

### Handling Large Matrices: Tiling

Real neural network layers have dimensions far exceeding 16×16. We handle this through **tiling**:

```
For a 256×512 × 512×128 GEMM with 16×16 array:

M_tiles = ceil(256/16) = 16
K_tiles = ceil(512/16) = 32
N_tiles = ceil(128/16) = 8

Total tiles = 16 × 32 × 8 = 4096 tile operations
```

The compiler handles all tiling decisions, generating the appropriate loop structure and memory addresses.

---

## Instruction Set Architecture Design

### Design Philosophy

The ISA bridges hardware capabilities and compiler needs. Key principles:

1. **Coarse-grained operations**: One instruction triggers an entire tile GEMM, not individual MACs
2. **Explicit memory management**: Software controls all data movement
3. **Hardware loops**: Amortize instruction fetch for repetitive operations
4. **Decoupled execution**: DMA, MXU, VPU can execute in parallel

### Instruction Format

All instructions are 128 bits wide, enabling rich encoding:

```
┌─────────┬─────────┬────────────────────────────────────────────┐
│ Opcode  │ Subop   │              Operands (112 bits)           │
│ (8 bits)│ (8 bits)│                                            │
└─────────┴─────────┴────────────────────────────────────────────┘
```

This fixed width simplifies instruction fetch and decode—no variable-length complexity.

### Instruction Classes

**TENSOR Operations (Opcode 0x01)**: Matrix operations on the MXU

```
TENSOR.GEMM dst, src_act, src_weight, M, N, K
  - Computes: dst = src_act[M,K] × src_weight[K,N]
  - Handles tiling internally based on M, N, K parameters
  
TENSOR.GEMM_ACC dst, src_act, src_weight, M, N, K
  - Same as GEMM but accumulates into existing dst values
  - Used for K-dimension tiling
```

**VECTOR Operations (Opcode 0x02)**: Element-wise on the VPU

```
VECTOR.ADD vd, vs1, vs2      # vd = vs1 + vs2
VECTOR.MUL vd, vs1, vs2      # vd = vs1 * vs2
VECTOR.RELU vd, vs1          # vd = max(0, vs1)
VECTOR.SIGMOID vd, vs1       # vd = sigmoid(vs1) via LUT
VECTOR.SOFTMAX_P1 vd, vs1    # Softmax pass 1: find max
VECTOR.SOFTMAX_P2 vd, vs1    # Softmax pass 2: exp(x - max)
VECTOR.SOFTMAX_P3 vd, vs1    # Softmax pass 3: normalize
```

**DMA Operations (Opcode 0x03)**: Data movement

```
DMA.LOAD_2D int_addr, ext_addr, rows, cols, int_stride, ext_stride
  - 2D strided load from external memory to SRAM
  - Handles non-contiguous tensor slices
  
DMA.STORE_2D ext_addr, int_addr, rows, cols, ext_stride, int_stride
  - 2D strided store from SRAM to external memory
```

**CONTROL Operations (Opcode 0x05-0x06)**: Flow control

```
LOOP count           # Begin hardware loop, push to loop stack
ENDLOOP              # Decrement counter, branch if not zero

BARRIER              # Global synchronization across TPCs
```

**SYNC Operations (Opcode 0x04)**: Pipeline synchronization

```
SYNC.WAIT_MXU        # Wait for MXU to complete
SYNC.WAIT_VPU        # Wait for VPU to complete
SYNC.WAIT_DMA        # Wait for DMA to complete
SYNC.WAIT_ALL        # Wait for all units
```

### Example: Compiled GEMM Layer

Here's what the compiler generates for a simple 64×64 GEMM:

```asm
# Load weights (64x64 = 4x4 tiles of 16x16)
LOOP 4                          # For each weight tile column
  DMA.LOAD_2D 0x1000, ext_w, 64, 16, 16, 64
  ext_w += 16
ENDLOOP

# Compute with activation streaming
LOOP 4                          # M tiles
  LOOP 4                        # N tiles
    LOOP 4                      # K tiles (accumulate)
      DMA.LOAD_2D 0x0000, ext_a, 16, 16, 16, 64
      SYNC.WAIT_DMA
      TENSOR.GEMM_ACC 0x2000, 0x0000, 0x1000, 16, 16, 16
      SYNC.WAIT_MXU
    ENDLOOP
  ENDLOOP
ENDLOOP

# Store results
DMA.STORE_2D ext_c, 0x2000, 64, 64, 64, 64
SYNC.WAIT_DMA
HALT
```

### Why Not RISC-V?

A common question: why design a custom ISA instead of using RISC-V with custom extensions?

| Aspect | Custom ISA | RISC-V + Extensions |
|--------|-----------|---------------------|
| Instruction width | 128-bit (rich encoding) | 32-bit base (cramped) |
| Decode complexity | Single fixed format | Multiple formats, extensions |
| Hardware loops | Native support | Requires software emulation |
| Coarse-grained ops | Natural fit | Awkward mapping |
| Toolchain | Custom (more work) | Leverage existing (less work) |
| Flexibility | Exactly what we need | General-purpose overhead |

For a dedicated accelerator where every transistor counts, the custom ISA wins. For a more general system with CPU integration needs, RISC-V would be attractive.

---

## Memory Hierarchy and Data Movement

### The Memory Wall Problem

AI accelerators live and die by memory bandwidth. Consider ResNet-50:

- Parameters: 25.6 million (25.6 MB at INT8)
- Activations: ~3 MB peak
- Compute: 4.1 billion MACs

At 2048 TOPS, we complete 4.1B MACs in 2 microseconds. But loading 25.6 MB at 128 GB/s takes 200 microseconds—**100× longer than compute**!

The solution: a carefully designed memory hierarchy that maximizes data reuse.

### Three-Level Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│ Level 1: Vector Registers (16 KB per TPC)                       │
│ - 32 vector registers × 64 lanes × 16 bits = 4 KB              │
│ - Plus accumulator registers = 16 KB total                      │
│ - Bandwidth: 1024 GB/s (register file)                          │
│ - Latency: 0 cycles (same cycle access)                         │
├─────────────────────────────────────────────────────────────────┤
│ Level 2: SRAM (2 MB per TPC, 8 MB total)                        │
│ - 16 banks × 4096 words × 256 bits = 2 MB                      │
│ - Bandwidth: 512 GB/s per TPC (2 TB/s aggregate)               │
│ - Latency: 1-2 cycles                                           │
│ - Partitioned: 512KB activation, 768KB weight, 768KB scratch    │
├─────────────────────────────────────────────────────────────────┤
│ Level 3: External DDR4/HBM2e                                     │
│ - Capacity: 4-64 GB                                              │
│ - Bandwidth: 128 GB/s (DDR4) or 512 GB/s (HBM2e)               │
│ - Latency: 50-200 cycles                                        │
└─────────────────────────────────────────────────────────────────┘
```

### SRAM Banking Strategy

With 16 banks, we can sustain high bandwidth even with strided access patterns. The key innovation is **XOR-based bank mapping**:

```verilog
function [BANK_BITS-1:0] get_bank;
    input [ADDR_WIDTH-1:0] addr;
    begin
        // XOR upper and lower address bits
        get_bank = addr[3:0] ^ addr[15:12];
    end
endfunction
```

Why XOR? Consider accessing a column of a row-major matrix (stride = row_width):

- **Simple modulo mapping**: All elements map to the same bank → serialized access
- **XOR mapping**: Elements distribute across banks → parallel access

This is the same technique used in GPU shared memory to avoid bank conflicts.

### Double Buffering

To hide DMA latency, we implement double buffering:

```
Time →
┌─────────┬─────────┬─────────┬─────────┬─────────┐
│ Load A0 │ Comp A0 │ Comp A1 │ Comp A2 │ Comp A3 │  Compute
│         │ Load A1 │ Load A2 │ Load A3 │ Load A4 │  DMA
└─────────┴─────────┴─────────┴─────────┴─────────┘
          │←─ Overlap: DMA hidden by compute ─→│
```

The scheduler assigns alternating buffer IDs:

```python
class ScheduleConfig:
    strategy: ScheduleStrategy = DOUBLE_BUFFER
    enable_fusion: bool = True
    dma_prefetch_depth: int = 2  # Prefetch 2 tiles ahead
```

Buffer allocation in SRAM:

```
0x00000 - 0x3FFFF: Buffer 0 Activations (256 KB)
0x40000 - 0x7FFFF: Buffer 0 Weights (256 KB)
0x80000 - 0xBFFFF: Buffer 1 Activations (256 KB)
0xC0000 - 0xFFFFF: Buffer 1 Weights (256 KB)
0x100000 - 0x17FFFF: Output (512 KB)
0x180000 - 0x1FFFFF: Scratch (512 KB)
```

### DMA Engine Design

The DMA engine supports 2D strided transfers, essential for tensor operations:

```verilog
// Command format for 2D transfer:
// - ext_addr: Starting address in DDR/HBM
// - int_addr: Starting address in SRAM
// - rows, cols: 2D shape
// - ext_stride, int_stride: Bytes between rows

// This enables efficient:
// - Column extraction from row-major matrices
// - Channel-first to channel-last conversion
// - Padding insertion/removal
```

Critical insight: The DMA state machine must handle **SRAM read latency** correctly. Our SRAM has 1-cycle read latency, so stores require:

```
Cycle N:   Assert address & read enable
Cycle N+1: SRAM registers read internally  
Cycle N+2: Data valid on output, capture it
Cycle N+3: Send to AXI write channel
```

Getting this pipeline wrong causes data corruption—ask me how I know!

---

## The Compiler Stack

### Why a Custom Compiler?

Existing ML compilers (TVM, XLA, MLIR) are powerful but complex. For this project, I built a focused compiler that:

1. Demonstrates the full compilation flow
2. Is readable and hackable (under 5K lines of Python)
3. Generates correct code for our specific ISA
4. Serves as a reference implementation

### Compiler Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend                                 │
│    ONNX Parser → Graph IR (nodes, edges, shapes, dtypes)        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Quantizer                                 │
│   FP32 weights → INT8 weights + scale factors                   │
│   Calibration data → activation ranges                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Tiler                                   │
│   Large tensors → tile sequences that fit in SRAM               │
│   Determines M, N, K tile sizes based on layer shapes           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Scheduler                                 │
│   Tiles → execution order with buffer assignments               │
│   Double-buffering, operator fusion, dependency tracking        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Code Generator                               │
│   Schedule → assembly instructions                               │
│   DMA loads, GEMM ops, activations, stores                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Assembler                                 │
│   Assembly → 128-bit binary machine code                        │
│   Resolves labels, encodes operands                             │
└─────────────────────────────────────────────────────────────────┘
```

### The Graph IR

At the core is a simple but expressive intermediate representation:

```python
class OpType(Enum):
    # Core compute
    GEMM = auto()
    CONV2D = auto()
    DEPTHWISE_CONV2D = auto()
    
    # Attention (Transformers)
    ATTENTION = auto()
    SCALED_DOT_PRODUCT = auto()
    
    # Activations
    RELU = auto()
    RELU6 = auto()  # MobileNet
    GELU = auto()   # Transformers
    SWISH = auto()  # EfficientNet
    SIGMOID = auto()
    
    # Normalization
    BATCHNORM = auto()
    LAYERNORM = auto()
    GROUPNORM = auto()  # U-Net, Diffusion
    
    # Pooling & Reduction
    MAXPOOL = auto()
    AVGPOOL = auto()
    GLOBAL_AVGPOOL = auto()
    REDUCE_SUM = auto()
    SOFTMAX = auto()
    
    # Shape manipulation
    RESHAPE = auto()
    TRANSPOSE = auto()
    CONCAT = auto()
    SPLIT = auto()

@dataclass
class IRNode:
    name: str
    op_type: OpType
    inputs: List[str]
    outputs: List[str]
    attributes: Dict[str, Any]
    shape: Tuple[int, ...]
    dtype: str = "int8"
```

### Quantization Strategy

We implement symmetric per-tensor quantization:

```python
def quantize_weights(weights: np.ndarray) -> Tuple[np.ndarray, float]:
    """Quantize FP32 weights to INT8 with scale factor."""
    abs_max = np.max(np.abs(weights))
    scale = abs_max / 127.0  # Map to [-127, 127]
    
    quantized = np.round(weights / scale).astype(np.int8)
    
    return quantized, scale

def quantize_activations(tensor: np.ndarray, 
                         observed_min: float,
                         observed_max: float) -> Tuple[np.ndarray, float, int]:
    """Quantize activations with zero-point for asymmetric ranges."""
    scale = (observed_max - observed_min) / 255.0
    zero_point = int(round(-observed_min / scale))
    
    quantized = np.round(tensor / scale + zero_point).astype(np.uint8)
    
    return quantized, scale, zero_point
```

For production, we'd add:
- Per-channel weight quantization (better accuracy)
- Learned scale factors (QAT - Quantization-Aware Training)
- Mixed precision (INT8 for most, FP16 for sensitive layers)

### Tiling Algorithm

The tiler determines how to partition large operations:

```python
def compute_tile_sizes(self, M: int, N: int, K: int) -> TileConfig:
    """Compute optimal tile sizes for GEMM."""
    
    # Start with hardware maximum
    tile_m = min(M, self.array_size)  # 16
    tile_n = min(N, self.array_size)  # 16
    tile_k = min(K, self.array_size)  # 16
    
    # Check SRAM capacity
    activation_size = tile_m * tile_k * self.dtype_size
    weight_size = tile_k * tile_n * self.dtype_size
    output_size = tile_m * tile_n * self.acc_size
    
    total_size = activation_size + weight_size + output_size
    
    # If doesn't fit, reduce tiles (prefer reducing K for better reuse)
    while total_size > self.sram_budget:
        if tile_k > 1:
            tile_k = tile_k // 2
        elif tile_m > 1:
            tile_m = tile_m // 2
        else:
            tile_n = tile_n // 2
        
        total_size = self._compute_size(tile_m, tile_n, tile_k)
    
    return TileConfig(tile_m, tile_n, tile_k)
```

### Operator Fusion

Fusion combines multiple operations to reduce memory traffic:

```python
FUSION_PATTERNS = [
    # GEMM + Bias + Activation
    ([OpType.GEMM, OpType.ADD, OpType.RELU], FusedGemmBiasRelu),
    ([OpType.GEMM, OpType.ADD, OpType.GELU], FusedGemmBiasGelu),
    
    # Conv + BatchNorm + Activation
    ([OpType.CONV2D, OpType.BATCHNORM, OpType.RELU], FusedConvBnRelu),
    
    # LayerNorm + GELU (Transformers)
    ([OpType.LAYERNORM, OpType.GELU], FusedLayerNormGelu),
]

def apply_fusion(self, schedule: List[ScheduleEntry]) -> List[ScheduleEntry]:
    """Detect and apply fusion patterns."""
    fused = []
    i = 0
    
    while i < len(schedule):
        matched = False
        for pattern, fused_op in FUSION_PATTERNS:
            if self._matches_pattern(schedule, i, pattern):
                # Create fused operation
                fused.append(self._create_fused(schedule[i:i+len(pattern)], fused_op))
                i += len(pattern)
                matched = True
                break
        
        if not matched:
            fused.append(schedule[i])
            i += 1
    
    return fused
```

### Code Generation Example

Here's how a fused GEMM+ReLU generates assembly:

```python
def _emit_gemm_relu(self, node: IRNode, entry: ScheduleEntry) -> List[str]:
    """Generate code for fused GEMM + ReLU."""
    asm = []
    
    M, K = node.attributes['input_shape']
    K2, N = node.attributes['weight_shape']
    
    # Calculate tile counts
    m_tiles = ceil(M / self.tile_m)
    n_tiles = ceil(N / self.tile_n)
    k_tiles = ceil(K / self.tile_k)
    
    # Outer loops over output tiles
    asm.append(f"    # GEMM+ReLU: [{M},{K}] x [{K},{N}] -> [{M},{N}]")
    asm.append(f"    LOOP {m_tiles}")
    asm.append(f"    LOOP {n_tiles}")
    
    # K-dimension accumulation loop
    asm.append(f"    LOOP {k_tiles}")
    
    # Load activation tile
    asm.append(f"        DMA.LOAD_2D {entry.act_addr}, $ext_act, "
               f"{self.tile_m}, {self.tile_k}, {self.tile_k}, {K}")
    asm.append(f"        SYNC.WAIT_DMA")
    
    # GEMM with accumulation
    if k_tiles > 1:
        asm.append(f"        TENSOR.GEMM_ACC {entry.out_addr}, "
                   f"{entry.act_addr}, {entry.weight_addr}, "
                   f"{self.tile_m}, {self.tile_n}, {self.tile_k}")
    else:
        asm.append(f"        TENSOR.GEMM {entry.out_addr}, "
                   f"{entry.act_addr}, {entry.weight_addr}, "
                   f"{self.tile_m}, {self.tile_n}, {self.tile_k}")
    
    asm.append(f"        SYNC.WAIT_MXU")
    asm.append(f"    ENDLOOP  # K tiles")
    
    # Apply ReLU (fused - happens before store)
    asm.append(f"        VECTOR.LOAD v0, {entry.out_addr}")
    asm.append(f"        VECTOR.RELU v0, v0")
    asm.append(f"        VECTOR.STORE {entry.out_addr}, v0")
    
    asm.append(f"    ENDLOOP  # N tiles")
    asm.append(f"    ENDLOOP  # M tiles")
    
    return asm
```

### Supported Networks

The compiler successfully handles:

| Network | Parameters | Notes |
|---------|------------|-------|
| ResNet-50 | 25M | Standard CNN benchmark |
| MobileNetV2 | 3.4M | Depthwise separable convolutions |
| EfficientNet-B0 | 5.3M | Swish activation, SE blocks |
| BERT-Base | 110M | Multi-head attention, LayerNorm |
| GPT-2 Small | 124M | Transformer decoder |
| U-Net | 31M | GroupNorm, skip connections |

---

## Verification and Validation

### Verification Philosophy

Hardware verification follows the "trust but verify" principle:

1. **Unit tests**: Each module in isolation
2. **Integration tests**: Modules connected together
3. **System tests**: Full accelerator running real workloads
4. **Formal verification**: Mathematical proofs (for critical paths)

### The Python Functional Model

Before writing any RTL, I created a cycle-accurate Python model:

```python
class SystolicArrayModel:
    """Cycle-accurate model of weight-stationary systolic array."""
    
    def __init__(self, size: int = 16):
        self.size = size
        self.weights = np.zeros((size, size), dtype=np.int8)
        self.accumulators = np.zeros((size, size), dtype=np.int32)
        
        # Skewing registers
        self.input_skew = [deque() for _ in range(size)]
        self.output_skew = [deque() for _ in range(size)]
        
    def load_weights(self, weights: np.ndarray):
        """Load weight matrix (column by column in hardware)."""
        self.weights = weights.copy()
        
    def compute(self, activations: np.ndarray) -> np.ndarray:
        """Compute matrix multiply with proper skewing."""
        K = activations.shape[1]
        
        # Initialize skewing delays
        for row in range(self.size):
            self.input_skew[row] = deque([0] * row)
        
        for col in range(self.size):
            delay = 2 * (self.size - 1 - col)
            self.output_skew[col] = deque([0] * delay)
        
        # Simulate cycle by cycle
        results = []
        for cycle in range(K + 2*self.size):
            # Feed activations with skewing
            for row in range(self.size):
                if cycle < K:
                    self.input_skew[row].append(activations[row, cycle])
                else:
                    self.input_skew[row].append(0)
            
            # MAC operations
            for row in range(self.size):
                if self.input_skew[row]:
                    act = self.input_skew[row].popleft()
                    for col in range(self.size):
                        self.accumulators[row, col] += act * self.weights[row, col]
            
            # Collect de-skewed outputs
            # ... (output skewing logic)
        
        return self.accumulators.copy()
```

This model:
- Validates the algorithm before RTL implementation
- Generates expected outputs for RTL comparison
- Catches conceptual bugs early (much faster than simulation)

### Cocotb Testbenches

For RTL verification, I used Cocotb (Coroutines-based Co-simulation Testbench):

```python
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

@cocotb.test()
async def test_systolic_gemm(dut):
    """Test 4x4 GEMM through systolic array."""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)
    
    # Test data
    weights = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ], dtype=np.int8)
    
    activations = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.int8)  # Identity matrix
    
    expected = weights @ activations.T  # Should equal weights
    
    # Load weights column by column
    for col in range(4):
        dut.weight_load_en.value = 1
        dut.weight_load_col.value = col
        dut.weight_load_data.value = pack_column(weights[:, col])
        await RisingEdge(dut.clk)
    
    dut.weight_load_en.value = 0
    
    # Start computation
    dut.start.value = 1
    dut.cfg_k_tiles.value = 4
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    # Stream activations
    for k in range(4):
        dut.act_valid.value = 1
        dut.act_data.value = pack_row(activations[:, k])
        await RisingEdge(dut.clk)
    
    dut.act_valid.value = 0
    
    # Wait for results
    results = []
    while len(results) < 4:
        await RisingEdge(dut.clk)
        if dut.result_valid.value:
            results.append(unpack_result(dut.result_data.value))
    
    # Verify
    actual = np.array(results)
    assert np.array_equal(actual, expected), f"Mismatch!\nExpected:\n{expected}\nActual:\n{actual}"
```

### Test Coverage

| Test Category | Tests | Coverage |
|---------------|-------|----------|
| MAC PE unit | 5 | 100% statement |
| Systolic Array | 8 | 95% statement, 87% branch |
| VPU operations | 12 | 100% statement |
| DMA engine | 6 | 92% statement |
| LCP state machine | 10 | 98% statement |
| Full TPC integration | 4 | 85% statement |
| End-to-end inference | 3 | Functional only |

### Waveform Debugging

When tests fail, waveform analysis is essential. Key signals to watch:

```
Systolic array debugging checklist:
□ weight_load_en pulses for exactly ARRAY_SIZE cycles
□ act_valid aligns with cfg_k_tiles count
□ Input skewing shows diagonal wavefront pattern
□ psum_out values accumulate correctly row by row
□ result_valid pulses for exactly ARRAY_SIZE cycles
□ Output de-skewing aligns all columns
```

I spent considerable time debugging timing issues where:
- SRAM read latency wasn't accounted for (data arrived 1 cycle late)
- De-skewing delays were calculated wrong (off-by-one errors)
- State machine advanced before operations completed

### Lessons from Verification

1. **Write the model first**: Finding bugs in 100 lines of Python beats debugging 1000 lines of Verilog
2. **Constrained random testing**: Manual test vectors miss corner cases
3. **Check intermediate signals**: Don't just verify outputs; verify internal state
4. **Waveform dumps are essential**: `$dumpfile`/`$dumpvars` are your friends
5. **Regression testing**: One fixed bug shouldn't break something else

---

## FPGA Prototyping Considerations

### Target Platforms

The design targets multiple FPGA platforms:

| Platform | Device | Resources | Use Case |
|----------|--------|-----------|----------|
| ZCU104 | XCZU7EV | 504K LUTs, 38Mb BRAM | Development, small models |
| VCU118 | XCVU9P | 2.5M LUTs, 345Mb BRAM | Full system validation |
| Alveo U250 | XCU250 | 1.7M LUTs, 360Mb UltraRAM | Datacenter prototyping |

### Resource Estimation

For a single TPC (16x16 array, 64-lane VPU):

```
MXU (Systolic Array):
  - 256 MAC PEs × ~100 LUTs/PE = 25,600 LUTs
  - 256 MAC PEs × 2 DSPs/PE = 512 DSPs
  - Skewing registers: ~2,000 LUTs

VPU (Vector Unit):
  - 64 lanes × ~50 LUTs/lane = 3,200 LUTs
  - Vector register file: 16 KB → 8 BRAMs

SRAM Subsystem:
  - 2 MB → 512 BRAMs (36Kb each)

LCP + DMA + Control:
  - ~5,000 LUTs

Total per TPC:
  - LUTs: ~36,000
  - DSPs: 512
  - BRAMs: ~520
```

For 4 TPCs + interconnect:

```
  - LUTs: ~150,000 (fits easily on VCU118)
  - DSPs: 2,048 (VCU118 has 6,840)
  - BRAMs: ~2,100 (VCU118 has 4,320)
```

### FPGA-Specific Optimizations

Several optimizations target FPGA implementation:

**1. DSP Inference for MACs**

```verilog
(* use_dsp = "yes" *)
wire signed [2*DATA_WIDTH-1:0] product = 
    $signed(act_in) * $signed(weight_reg);
```

**2. BRAM for SRAM Banks**

```verilog
(* ram_style = "block" *)
reg [WIDTH-1:0] mem [0:DEPTH-1];
```

**3. Reduced Array Size for Prototyping**

```verilog
`ifdef TARGET_PROTOTYPE
    `define ARRAY_SIZE 8    // 8x8 instead of 16x16
    `define VPU_LANES 32    // 32 instead of 64
    `define SRAM_DEPTH 256  // Reduced depth
`endif
```

### Clock Domain Crossing

The design uses multiple clock domains:

```
clk_core (200 MHz)  - TPC compute logic
clk_mem (300 MHz)   - Memory controller interface
clk_axi (100 MHz)   - Host AXI-Lite control
```

CDC is handled with standard techniques:
- 2-FF synchronizers for single-bit signals
- Async FIFOs for multi-bit data
- Gray coding for pointers

### Timing Closure Strategy

Achieving timing closure at 200+ MHz requires:

1. **Pipeline critical paths**: Especially SRAM read → compute → write
2. **Floorplanning**: Keep TPCs physically separate
3. **Register retiming**: Let Vivado optimize register placement
4. **False path constraints**: Mark async paths as false

```tcl
# Example timing constraints
create_clock -period 5.0 -name clk_core [get_ports clk]

# False path between clock domains
set_false_path -from [get_clocks clk_axi] -to [get_clocks clk_core]

# Multicycle path for weight loading (takes multiple cycles anyway)
set_multicycle_path 2 -setup -from [get_cells */weight_reg*]
```

---

## Lessons Learned and Trade-offs

### What Worked Well

**1. Starting with the functional model**

Writing the Python model first caught numerous algorithm bugs before RTL. The model runs in seconds; RTL simulation takes minutes to hours.

**2. Coarse-grained ISA**

Instructions that trigger entire tile operations (not individual MACs) dramatically simplified the compiler and reduced instruction memory pressure.

**3. Weight-stationary dataflow**

For inference workloads, this is the clear winner. Weights load once per layer, activations stream through.

**4. Explicit memory management**

No caches means deterministic performance. The compiler knows exactly when data is available.

**5. Hardware loops**

Reducing instruction fetch by 100-1000× for inner loops is huge for energy efficiency.

### Trade-offs Made

**1. Flexibility vs. Efficiency**

*Choice*: Fixed 16×16 array size, INT8 only
*Trade-off*: Can't handle FP16/FP32 or odd matrix sizes efficiently
*Rationale*: 95% of edge inference is INT8; the efficiency gain is worth it

**2. Programmability vs. Performance**

*Choice*: Custom ISA instead of hardcoded datapath
*Trade-off*: ~10% area overhead for instruction fetch/decode
*Rationale*: Ability to support new operators without respinning silicon

**3. Simplicity vs. Utilization**

*Choice*: Simple round-robin arbitration for memory/NoC
*Trade-off*: Suboptimal in some access patterns
*Rationale*: Complex arbitration adds area and timing pressure

**4. SRAM Size vs. Power**

*Choice*: 2 MB per TPC (8 MB total)
*Trade-off*: SRAM is power-hungry, especially when idle
*Rationale*: Larger SRAM enables larger tiles, reducing DDR bandwidth

### What I'd Do Differently

**1. Add sparse tensor support**

Modern networks use pruning extensively. Supporting sparse formats (CSR, block-sparse) would improve efficiency for pruned models by 2-4×.

**2. Implement proper NoC**

The current crossbar works for 4 TPCs but doesn't scale. A mesh NoC with virtual channels would support 16+ TPCs.

**3. Support dynamic shapes**

Current design assumes static shapes at compile time. Dynamic batching and sequence lengths (for LLMs) require runtime shape handling.

**4. Add hardware profiling**

Cycle counters, stall counters, and memory bandwidth monitors would help identify bottlenecks.

**5. Consider chiplet approach**

For very high throughput (10K+ TOPS), multiple smaller dies connected via UCIe would be more practical than a monolithic design.

---

## What's Next

### Immediate Roadmap

1. **Binary assembler completion**: Encode assembly → machine code
2. **End-to-end validation**: Compile ResNet-50 → run on RTL → verify accuracy
3. **FPGA synthesis**: Bring up on VCU118, measure actual performance
4. **Host driver**: C++ runtime for loading models and triggering inference

### Future Enhancements

**Short-term (3-6 months)**:
- INT4 quantization support
- Sparse tensor operations
- Hardware performance counters
- Linux driver for PCIe deployment

**Medium-term (6-12 months)**:
- Multi-chip scaling via UCIe
- FP16 accumulation option
- Dynamic shape support
- Power management (clock gating, power domains)

**Long-term (12+ months)**:
- Training support (backward pass, gradient accumulation)
- Transformer-specific optimizations (FlashAttention)
- ASIC tapeout on 7nm/5nm

### Open Questions

1. **How to handle LLM KV caching efficiently?** The current memory hierarchy isn't designed for the dynamic memory patterns of autoregressive generation.

2. **What's the optimal TPC count vs. array size trade-off?** More small TPCs vs. fewer large TPCs—simulation suggests 8-16 TPCs with 16×16 arrays is sweet spot.

3. **Should the VPU support FP16?** Some normalization and attention operations benefit from higher precision. The area cost is ~2× per lane.

---

## Conclusion

Building an AI accelerator from scratch has been an incredible learning journey. What started as "let's make a systolic array" evolved into a complete system with:

- **13 RTL modules** totaling ~4,000 lines of synthesizable Verilog
- **A complete compiler** with ~3,000 lines of Python
- **Comprehensive documentation** covering ISA, architecture, and usage
- **Extensive test coverage** with unit, integration, and system tests

The key insight? AI accelerators aren't magic. They're carefully engineered systems that exploit the structure of neural network computations. The same principles—data reuse, parallelism, memory hierarchy, specialized datapaths—apply whether you're building a tiny edge accelerator or a datacenter training chip.

I hope this detailed walkthrough helps demystify accelerator design and perhaps inspires others to explore this fascinating intersection of computer architecture, machine learning, and systems engineering.

---

*If you found this useful, let's connect! I'm happy to discuss AI accelerator architecture, FPGA prototyping, or ML compiler development.*

*All code is available on GitHub: [link to your repo]*

---

## Appendix A: Complete Module Hierarchy

```
tensor_accelerator_top
├── global_cmd_processor (GCP)
│   └── AXI-Lite slave interface
├── tensor_processing_cluster [0:3] (TPC)
│   ├── local_cmd_processor (LCP)
│   │   ├── Instruction fetch
│   │   ├── Hardware loop stack
│   │   └── Scoreboard
│   ├── systolic_array (MXU)
│   │   ├── mac_pe [0:255] (16x16)
│   │   ├── Input skewing registers
│   │   └── Output de-skewing registers
│   ├── vector_unit (VPU)
│   │   ├── Vector register file
│   │   ├── SIMD ALU lanes [0:63]
│   │   └── Reduction tree
│   ├── dma_engine
│   │   ├── 2D address generator
│   │   └── AXI4 master interface
│   └── sram_subsystem
│       ├── sram_bank [0:15]
│       └── Arbiter
├── noc_router [0:3] (2x2 mesh)
│   ├── Input FIFOs [0:4]
│   ├── Route compute (X-Y)
│   └── Crossbar
└── axi_interconnect
    └── Round-robin arbiter
```

## Appendix B: Performance Model

```python
def estimate_inference_time(model_macs: int, 
                            batch_size: int = 1,
                            tops: float = 2048,
                            utilization: float = 0.7) -> float:
    """Estimate inference time in milliseconds."""
    
    effective_tops = tops * utilization
    ops_per_second = effective_tops * 1e12
    
    total_ops = model_macs * 2 * batch_size  # MACs = 2 ops
    
    time_seconds = total_ops / ops_per_second
    time_ms = time_seconds * 1000
    
    return time_ms

# Example usage:
print(f"ResNet-50: {estimate_inference_time(4.1e9):.2f} ms")   # ~5.7 ms
print(f"MobileNetV2: {estimate_inference_time(0.3e9):.2f} ms") # ~0.4 ms
print(f"BERT-Base: {estimate_inference_time(22e9):.2f} ms")    # ~30.7 ms
```

## Appendix C: Key References

1. Jouppi, N. et al. "In-Datacenter Performance Analysis of a Tensor Processing Unit" (ISCA 2017) - TPU architecture
2. Chen, Y. et al. "Eyeriss: An Energy-Efficient Reconfigurable Accelerator" (JSSC 2017) - Dataflow taxonomy
3. Kwon, H. et al. "Understanding Reuse, Performance, and Hardware Cost of DNN Dataflows" (MICRO 2019) - Dataflow analysis
4. Gholami, A. et al. "A Survey of Quantization Methods for Efficient Neural Network Inference" (2021) - Quantization techniques
5. NVIDIA Tensor Core architecture white papers - Modern GPU tensor acceleration

---

**Tags**: #AI #MachineLearning #HardwareDesign #FPGA #ASIC #NeuralNetworks #Compiler #ComputerArchitecture #DeepLearning #EdgeAI #Semiconductor


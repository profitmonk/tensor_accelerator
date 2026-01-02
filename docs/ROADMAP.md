# Tensor Accelerator Development Roadmap

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Date | December 31, 2025 |
| Status | Active |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State](#2-current-state)
3. [Software Stack Plan](#3-software-stack-plan)
4. [Verification Plan](#4-verification-plan)
5. [Integration Plan](#5-integration-plan)
6. [Implementation Phases](#6-implementation-phases)
7. [Risk Assessment](#7-risk-assessment)

---

## 1. Executive Summary

### Project Goal
Build a complete, production-ready INT8 tensor accelerator with:
- Verified RTL (FPGA/ASIC ready)
- Complete software toolchain (ONNX → binary)
- AXI4 interface for system integration
- Comprehensive verification with coverage metrics

### Current Status
- ✅ RTL complete (2×2 TPC grid, 8×8 systolic arrays)
- ✅ 53 functional tests passing
- ✅ Basic assembler working
- ⚠️ AXI protocol not verified
- ❌ No compiler (hand-written assembly only)
- ❌ No coverage metrics

### Target State
```
PyTorch/ONNX Model → Compiler → Binary → AXI Interface → Accelerator → Output
        │                                      │
        └──────── Verified & Measured ─────────┘
```

---

## 2. Current State

### 2.1 RTL Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Tensor Accelerator Top                          │
├─────────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────┐     │
│  │                Global Command Processor                     │     │
│  └────────────────────────────────────────────────────────────┘     │
│                              │                                       │
│  ┌───────────────────────────┴───────────────────────────┐          │
│  │                   NoC Mesh (2×2)                       │          │
│  └───────────────────────────────────────────────────────┘          │
│                              │                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐│
│  │    TPC 0     │  │    TPC 1     │  │    TPC 2     │  │  TPC 3   ││
│  │  8×8 Array   │  │  8×8 Array   │  │  8×8 Array   │  │ 8×8 Array││
│  │  2MB SRAM    │  │  2MB SRAM    │  │  2MB SRAM    │  │ 2MB SRAM ││
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────┘│
│                              │                                       │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │                    AXI4 Interface                           │     │
│  │              (NOT YET VERIFIED)                             │     │
│  └────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Software Toolchain (Current)

```
Hand-Written Assembly (.asm)
         │
         ▼
┌─────────────────┐
│  assembler.py   │  ✅ Working
│                 │
│  - Parse ASM    │
│  - Encode 128b  │
│  - Output .hex  │
└────────┬────────┘
         │
         ▼
    .hex / .bin / .coe
         │
         ▼
    Verilog $readmemh
```

### 2.3 Verification Status

| Category | Tests | Coverage (Est.) |
|----------|:-----:|:---------------:|
| Unit | 9 | ~95% |
| Integration | 2 | ~85% |
| End-to-End | 5 | ~80% |
| Model | 10 | ~75% |
| Realistic (Phase A-F) | 26 | ~70% |
| Python Golden | 1 | N/A |
| **Total** | **53** | **~70%** |

### 2.4 Key Gaps

| Gap | Impact | Priority |
|-----|--------|:--------:|
| AXI Protocol Verification | Cannot integrate with SoC | **Critical** |
| Coverage Metrics | Unknown verification quality | **High** |
| ONNX Compiler | Cannot run real models | **High** |
| Runtime/Driver | Cannot deploy to hardware | Medium |
| Double-buffering Optimization | Performance | Low |

---

## 3. Software Stack Plan

### 3.1 Target Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     COMPLETE SOFTWARE STACK                          │
└─────────────────────────────────────────────────────────────────────┘

     ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
     │   PyTorch    │    │    ONNX      │    │ TensorFlow   │
     │    Model     │    │    Model     │    │    Model     │
     └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
            │                   │                   │
            └───────────────────┼───────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Frontend Parser     │
                    │                       │
                    │ • ONNX graph import   │
                    │ • Shape inference     │
                    │ • Type checking       │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Quantization        │
                    │                       │
                    │ • FP32 → INT8         │
                    │ • Scale computation   │
                    │ • Calibration         │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Graph Optimizer     │
                    │                       │
                    │ • Operator fusion     │
                    │ • Constant folding    │
                    │ • Layout optimization │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Scheduler           │
                    │                       │
                    │ • Layer ordering      │
                    │ • TPC assignment      │
                    │ • Memory planning     │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Tiling Engine       │
                    │                       │
                    │ • GEMM tiling         │
                    │ • Loop generation     │
                    │ • DMA scheduling      │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Code Generator      │
                    │                       │
                    │ • Emit instructions   │
                    │ • Register alloc      │
                    │ • Sync insertion      │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Assembler           │  ✅ EXISTS
                    │   (assembler.py)      │
                    └───────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
            ┌───────────────┐       ┌───────────────┐
            │ Program Binary│       │ Weight Binary │
            │    (.bin)     │       │   (.weights)  │
            └───────┬───────┘       └───────┬───────┘
                    │                       │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Runtime / Driver    │
                    │                       │
                    │ • Memory allocation   │
                    │ • DMA setup           │
                    │ • Command submission  │
                    │ • Interrupt handling  │
                    └───────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
            ┌───────────────┐       ┌───────────────┐
            │   Simulator   │       │  Hardware     │
            │   (cocotb)    │       │  (FPGA/ASIC)  │
            └───────────────┘       └───────────────┘
```

### 3.2 Compiler Components

#### 3.2.1 Frontend Parser

**Input**: ONNX model file
**Output**: Internal graph representation (IR)

```python
# Target API
from tensor_compiler import frontend

graph = frontend.load_onnx("model.onnx")
graph = frontend.load_pytorch(torch_model, sample_input)

# IR representation
class IRGraph:
    nodes: List[IRNode]      # Operations
    edges: List[IREdge]      # Data dependencies
    inputs: List[IRTensor]   # Model inputs
    outputs: List[IRTensor]  # Model outputs

class IRNode:
    op_type: str             # "Conv", "MatMul", "Relu", etc.
    inputs: List[str]        # Input tensor names
    outputs: List[str]       # Output tensor names
    attributes: Dict         # Kernel size, strides, etc.
```

**Supported ONNX Operators (Initial)**:
| Operator | Priority | Notes |
|----------|:--------:|-------|
| Conv | High | im2col + GEMM |
| MatMul | High | Direct GEMM |
| Gemm | High | Direct GEMM |
| Relu | High | VPU or fused |
| MaxPool | High | VPU |
| AveragePool | High | VPU |
| Add | High | Residual connections |
| Flatten | High | Reshape only |
| Softmax | Medium | Multi-pass VPU |
| LayerNorm | Medium | VPU |
| GELU | Medium | LUT-based |
| Transpose | Medium | DMA or in-place |
| Reshape | Low | Metadata only |

#### 3.2.2 Quantization Module

**Input**: FP32 graph + calibration data
**Output**: INT8 graph + scale factors

```python
# Target API
from tensor_compiler import quantize

# Post-training quantization
quant_graph = quantize.ptq(
    graph,
    calibration_data=dataloader,
    method="minmax"  # or "percentile", "entropy"
)

# Access scales
for node in quant_graph.nodes:
    print(f"{node.name}: scale={node.output_scale}, zp={node.zero_point}")
```

**Quantization Scheme**:
```
INT8 Symmetric: q = round(x / scale)
                x ≈ q * scale

Scale Computation:
  scale = max(|x|) / 127

GEMM Output:
  Y_int32 = X_int8 @ W_int8
  Y_fp32 = Y_int32 * (scale_x * scale_w)
  Y_int8 = round(Y_fp32 / scale_y)

Requantization (fused):
  Y_int8 = (Y_int32 * M) >> shift
  where M and shift are precomputed
```

#### 3.2.3 Graph Optimizer

**Optimizations**:

1. **Operator Fusion**
   ```
   Conv → BatchNorm → ReLU  →  Conv_BN_ReLU (single kernel)
   MatMul → Add → ReLU      →  GEMM_BIAS_RELU
   ```

2. **Constant Folding**
   ```
   BatchNorm with fixed weights → Precompute scale/bias
   Reshape of constant         → Apply at compile time
   ```

3. **Layout Optimization**
   ```
   NCHW → NHWC if beneficial for im2col
   Weight transposition for efficient loading
   ```

#### 3.2.4 Scheduler

**Input**: Optimized graph
**Output**: Execution schedule

```python
class Schedule:
    layers: List[LayerSchedule]
    
class LayerSchedule:
    layer_id: int
    tpc_assignment: List[int]    # Which TPCs run this
    dependencies: List[int]      # Wait for these layers
    memory_allocation: Dict      # SRAM addresses
```

**Scheduling Strategy**:
```
1. Topological sort of graph
2. For each layer:
   a. Estimate compute cycles
   b. Estimate memory requirements
   c. Assign to TPC(s)
   d. Insert sync barriers
3. Optimize for:
   - Minimize memory transfers
   - Maximize TPC utilization
   - Enable double-buffering
```

#### 3.2.5 Tiling Engine

**Input**: Layer schedule
**Output**: Tiled loops with DMA

```python
# Example: GEMM tiling for M=256, N=128, K=512

def tile_gemm(M, N, K, tile_m=16, tile_n=16, tile_k=16):
    """Generate tiled GEMM schedule"""
    
    schedule = []
    
    for m in range(0, M, tile_m):
        for n in range(0, N, tile_n):
            # Initialize accumulator
            schedule.append(('ZERO_ACC', m, n))
            
            for k in range(0, K, tile_k):
                # Load tiles
                schedule.append(('DMA_LOAD_A', m, k, tile_m, tile_k))
                schedule.append(('DMA_LOAD_B', k, n, tile_k, tile_n))
                schedule.append(('SYNC_DMA',))
                
                # Compute
                schedule.append(('GEMM_ACC', tile_m, tile_n, tile_k))
                schedule.append(('SYNC_MXU',))
            
            # Store result
            schedule.append(('DMA_STORE_C', m, n, tile_m, tile_n))
    
    return schedule
```

#### 3.2.6 Code Generator

**Input**: Tiled schedule
**Output**: Assembly or direct binary

```python
class CodeGenerator:
    def __init__(self):
        self.instructions = []
        self.sram_allocator = SRAMAllocator()
    
    def emit_gemm(self, dst, src_a, src_b, M, N, K, accumulate=False):
        subop = 'GEMM_ACC' if accumulate else 'GEMM'
        self.instructions.append(
            f"TENSOR.{subop} {dst}, {src_a}, {src_b}, {M}, {N}, {K}, 1"
        )
    
    def emit_dma_load(self, dst, src, rows, cols, stride):
        self.instructions.append(
            f"DMA.LOAD_2D {dst}, {src}, {rows}, {cols}, {stride}, {cols}"
        )
    
    def generate(self) -> str:
        return '\n'.join(self.instructions)
```

### 3.3 Runtime / Driver

```python
# Target API for simulation
class Simulator:
    def __init__(self, rtl_path: str):
        self.runner = cocotb.runner(rtl_path)
    
    def load_program(self, program: bytes):
        """Load program binary via AXI"""
        self.axi_write(PROG_BASE, program)
    
    def load_weights(self, weights: bytes):
        """Load weights via AXI"""
        self.axi_write(WEIGHT_BASE, weights)
    
    def run(self, input_data: np.ndarray) -> np.ndarray:
        """Execute inference"""
        self.axi_write(INPUT_BASE, input_data.tobytes())
        self.start_execution()
        self.wait_completion()
        return self.axi_read(OUTPUT_BASE)

# Target API for hardware
class Device:
    def __init__(self, device_path: str = "/dev/tensor0"):
        self.fd = os.open(device_path, os.O_RDWR)
        self.mmap = mmap.mmap(...)
    
    # Same interface as Simulator
```

### 3.4 Implementation Priority

| Component | Priority | Effort | Dependencies |
|-----------|:--------:|:------:|--------------|
| ONNX Parser | P0 | 2 days | None |
| Basic Quantizer | P0 | 1 day | Parser |
| Simple Scheduler | P0 | 2 days | Parser |
| Tiling Engine | P0 | 2 days | Scheduler |
| Code Generator | P0 | 2 days | Tiling |
| **MVP Compiler** | **P0** | **~9 days** | All above |
| Graph Optimizer | P1 | 3 days | MVP |
| Advanced Quantizer | P1 | 2 days | MVP |
| Runtime (Sim) | P1 | 2 days | AXI verified |
| Runtime (HW) | P2 | 3 days | FPGA working |

---

## 4. Verification Plan

### 4.1 AXI Protocol Verification

#### 4.1.1 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    cocotb AXI Verification                          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Test Cases    │     │   AXI Master    │     │   Protocol      │
│                 │────▶│   BFM           │────▶│   Checker       │
│ • Single R/W    │     │                 │     │                 │
│ • Burst R/W     │     │ • Read channel  │     │ • Handshake     │
│ • Outstanding   │     │ • Write channel │     │ • Ordering      │
│ • Error inject  │     │ • Response      │     │ • Timing        │
└─────────────────┘     └────────┬────────┘     └────────┬────────┘
                                 │                       │
                                 ▼                       ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │   DUT           │     │   Coverage      │
                        │   (Accelerator) │     │   Collector     │
                        │                 │     │                 │
                        │ • AXI Slave     │     │ • Functional    │
                        │ • Memory        │     │ • Protocol      │
                        │ • Compute       │     │ • Cross         │
                        └─────────────────┘     └─────────────────┘
```

#### 4.1.2 Test Categories

**Basic Transactions**:
| Test | Description | Priority |
|------|-------------|:--------:|
| single_read | Single beat read | P0 |
| single_write | Single beat write | P0 |
| burst_read_incr | INCR burst read | P0 |
| burst_write_incr | INCR burst write | P0 |
| burst_read_wrap | WRAP burst read | P1 |
| burst_write_wrap | WRAP burst write | P1 |

**Protocol Compliance**:
| Test | Description | Priority |
|------|-------------|:--------:|
| handshake_valid_first | VALID asserted before READY | P0 |
| handshake_ready_first | READY asserted before VALID | P0 |
| response_okay | OKAY response handling | P0 |
| response_slverr | SLVERR response handling | P1 |
| response_decerr | DECERR response handling | P2 |

**Advanced Features**:
| Test | Description | Priority |
|------|-------------|:--------:|
| outstanding_reads | Multiple pending reads | P1 |
| outstanding_writes | Multiple pending writes | P1 |
| interleaved | Interleaved R/W | P1 |
| narrow_transfer | Sub-width transfers | P2 |
| unaligned | Unaligned addresses | P2 |

**Stress Tests**:
| Test | Description | Priority |
|------|-------------|:--------:|
| back_to_back | No gaps between transactions | P1 |
| max_outstanding | Fill transaction buffers | P1 |
| random_delays | Random READY timing | P1 |
| mixed_traffic | Random R/W mix | P1 |

#### 4.1.3 Protocol Checker Rules

```python
# AXI4 Protocol Rules to Verify

class AXI4ProtocolChecker:
    """
    Checks AXI4 protocol compliance per ARM IHI 0022E
    """
    
    # Handshake rules
    RULE_VALID_STABLE = "VALID must remain asserted until READY"
    RULE_DATA_STABLE = "Data must remain stable while VALID high"
    RULE_RESP_ORDER = "Responses must be in order (per ID)"
    
    # Write rules  
    RULE_WLAST = "WLAST must be asserted on final beat"
    RULE_WSTRB = "WSTRB must be valid for write data"
    RULE_BRESP_AFTER_WLAST = "BRESP only after WLAST"
    
    # Read rules
    RULE_RLAST = "RLAST must be asserted on final beat"
    RULE_RRESP_PER_BEAT = "RRESP valid for each beat"
    
    # Ordering rules
    RULE_WRITE_ORDER = "Writes to same ID in order"
    RULE_READ_ORDER = "Reads to same ID in order"
```

### 4.2 Coverage Plan

#### 4.2.1 Coverage Categories

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Coverage Hierarchy                              │
└─────────────────────────────────────────────────────────────────────┘

Coverage
├── Functional Coverage
│   ├── Operation Coverage
│   │   ├── GEMM dimensions (M, N, K ranges)
│   │   ├── Activation functions (ReLU, GELU, etc.)
│   │   ├── Pooling types (Max, Avg)
│   │   └── DMA patterns (1D, 2D, strides)
│   │
│   ├── Data Coverage
│   │   ├── INT8 value ranges (boundaries, zero, negative)
│   │   ├── Accumulator overflow cases
│   │   └── Requantization shifts
│   │
│   └── Control Coverage
│       ├── Multi-TPC configurations
│       ├── Sync barrier patterns
│       └── Loop nesting depths
│
├── Protocol Coverage (AXI)
│   ├── Transaction types (read, write, burst)
│   ├── Burst lengths (1, 4, 8, 16, 256)
│   ├── Response types (OKAY, SLVERR)
│   └── Outstanding transactions (1, 2, 4, max)
│
├── Cross Coverage
│   ├── Operation × Data size
│   ├── TPC count × Operation type
│   └── Burst length × Outstanding count
│
└── Code Coverage (from simulator)
    ├── Line coverage
    ├── Branch coverage
    ├── Toggle coverage
    └── FSM coverage
```

#### 4.2.2 Coverage Goals

| Category | Current | Target | Gap |
|----------|:-------:|:------:|:---:|
| Functional - Operations | ~80% | 95% | 15% |
| Functional - Data | ~70% | 90% | 20% |
| Functional - Control | ~60% | 85% | 25% |
| Protocol - AXI | 0% | 90% | 90% |
| Cross Coverage | ~40% | 80% | 40% |
| Line Coverage | ~70% | 90% | 20% |
| Branch Coverage | ~60% | 85% | 25% |
| FSM Coverage | ~80% | 95% | 15% |

#### 4.2.3 Coverage Implementation

```python
# cocotb-coverage implementation

from cocotb_coverage.coverage import CoverPoint, CoverCross, coverage_db

# Functional coverage
@CoverPoint(
    "gemm.dimensions.M",
    bins=[1, 8, 16, 32, 64, 128, 256],
    rel=lambda x, bins: x in bins
)
def sample_gemm_m(m_dim):
    pass

@CoverPoint(
    "axi.burst.length",
    bins=[1, 2, 4, 8, 16, 32, 64, 128, 256],
)
def sample_burst_length(length):
    pass

# Cross coverage
@CoverCross(
    "gemm_x_tpc",
    items=["gemm.dimensions.M", "tpc.count"]
)
def sample_gemm_tpc_cross():
    pass

# Report generation
def generate_coverage_report():
    coverage_db.report_coverage(
        output_file="coverage_report.html",
        format="html"
    )
```

### 4.3 Verification Metrics

#### 4.3.1 Exit Criteria

| Metric | Threshold | Current |
|--------|:---------:|:-------:|
| All tests passing | 100% | ✅ 100% |
| Functional coverage | >90% | ⚠️ ~70% |
| Protocol coverage | >90% | ❌ 0% |
| Line coverage | >85% | ⚠️ ~70% |
| Branch coverage | >80% | ⚠️ ~60% |
| No critical bugs | 0 | ✅ 0 |
| No major bugs | 0 | ✅ 0 |

#### 4.3.2 Bug Classification

| Severity | Definition | Action |
|----------|------------|--------|
| Critical | Data corruption, hang, crash | Block tape-out |
| Major | Incorrect results, performance | Must fix |
| Minor | Edge cases, cosmetic | Should fix |
| Enhancement | Optimization opportunity | Nice to have |

---

## 5. Integration Plan

### 5.1 System Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SoC Integration                              │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│     CPU      │     │    Tensor    │     │     DDR     │
│   (ARM/x86)  │     │  Accelerator │     │   Memory    │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       │     AXI4           │      AXI4          │
       ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      AXI Interconnect                                │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Memory Map     │
                    │                 │
                    │ 0x0000: Config  │
                    │ 0x1000: Program │
                    │ 0x2000: Weights │
                    │ 0x4000: Input   │
                    │ 0x6000: Output  │
                    │ 0x8000: Status  │
                    └─────────────────┘
```

### 5.2 Memory Map

| Address Range | Size | Description |
|---------------|------|-------------|
| 0x0000_0000 - 0x0000_0FFF | 4 KB | Control registers |
| 0x0000_1000 - 0x0000_FFFF | 60 KB | Program memory |
| 0x0001_0000 - 0x000F_FFFF | 960 KB | Weight memory |
| 0x0010_0000 - 0x001F_FFFF | 1 MB | Input buffer |
| 0x0020_0000 - 0x002F_FFFF | 1 MB | Output buffer |
| 0x0030_0000 - 0x003F_FFFF | 1 MB | Scratch memory |

### 5.3 Control Registers

| Offset | Name | Access | Description |
|--------|------|:------:|-------------|
| 0x000 | CTRL | RW | Control register |
| 0x004 | STATUS | RO | Status register |
| 0x008 | IRQ_EN | RW | Interrupt enable |
| 0x00C | IRQ_STATUS | RO | Interrupt status |
| 0x010 | PROG_ADDR | RW | Program start address |
| 0x014 | PROG_LEN | RW | Program length |
| 0x018 | INPUT_ADDR | RW | Input buffer address |
| 0x01C | OUTPUT_ADDR | RW | Output buffer address |

---

## 6. Implementation Phases

### Phase 1: AXI + Coverage (Current)
**Duration**: 3-4 hours
**Deliverables**:
- cocotb test infrastructure
- AXI4 BFM (Bus Functional Model)
- Protocol checker
- Coverage collector
- Basic AXI tests (20+)

### Phase 2: ONNX Compiler MVP
**Duration**: 1-2 weeks
**Deliverables**:
- ONNX parser
- Basic quantizer
- Simple scheduler
- Tiling engine
- Code generator
- LeNet-5 and ResNet-18 compilation

### Phase 3: Integration Testing
**Duration**: 1 week
**Deliverables**:
- Compile real models
- Run through cocotb
- Measure coverage
- Fix bugs

### Phase 4: FPGA Prototype
**Duration**: 2-3 weeks
**Deliverables**:
- Synthesis scripts
- Timing closure
- Board bring-up
- Driver development

### Phase 5: Performance Optimization
**Duration**: 2-4 weeks
**Deliverables**:
- Double-buffering
- Kernel fusion
- Memory optimization
- Benchmark results

---

## 7. Risk Assessment

### 7.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|:-----------:|:------:|------------|
| AXI timing issues | Medium | High | Early verification |
| Compiler bugs | High | Medium | Extensive testing |
| FPGA resource overflow | Low | High | Early synthesis |
| Performance shortfall | Medium | Medium | Profiling tools |

### 7.2 Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|:-----------:|:------:|------------|
| Verification delays | Medium | Medium | Parallel work |
| Tool issues | Low | High | Docker environment |
| Scope creep | Medium | Medium | Clear MVP definition |

### 7.3 Contingency Plans

1. **If AXI verification fails**: Fall back to simplified interface
2. **If compiler is delayed**: Continue with hand-written assembly
3. **If FPGA resources tight**: Reduce TPC count to 2×1

---

## Appendix A: File Structure

```
tensor_accelerator/
├── rtl/                      # RTL source
│   ├── core/                # Compute units
│   ├── memory/              # Memory subsystem
│   ├── noc/                 # Network on chip
│   ├── control/             # Command processors
│   └── top/                 # Top-level modules
├── tb/                       # Verilog testbenches
├── tests/                    # Test organization
│   └── realistic/           # Realistic tests
├── cocotb/                   # cocotb tests (NEW)
│   ├── axi/                 # AXI verification
│   ├── coverage/            # Coverage collection
│   └── bfm/                 # Bus functional models
├── sw/                       # Software
│   ├── assembler/           # Assembler
│   ├── compiler/            # Compiler (NEW)
│   └── runtime/             # Runtime (NEW)
├── model/                    # Python models
├── docs/                     # Documentation
└── scripts/                  # Build scripts
```

---

## Appendix B: Tool Versions

| Tool | Version | Purpose |
|------|---------|---------|
| Icarus Verilog | 12.0+ | RTL simulation |
| cocotb | 1.8+ | Python testbenches |
| cocotb-bus | 0.2+ | AXI BFM |
| cocotb-coverage | 1.1+ | Coverage |
| Python | 3.10+ | Compiler, models |
| ONNX | 1.14+ | Model format |
| NumPy | 1.24+ | Numerical |

---

## Appendix C: References

1. ARM AMBA AXI Protocol Specification (IHI 0022E)
2. ONNX Operator Specifications
3. INT8 Quantization Best Practices
4. cocotb Documentation

---

*End of Roadmap Document*

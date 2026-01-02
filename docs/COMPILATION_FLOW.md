# Tensor Accelerator: Compilation Flow

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Date | January 1, 2026 |
| Status | Phase 2 Complete |

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Component Details](#3-component-details)
4. [File Formats](#4-file-formats)
5. [Usage Guide](#5-usage-guide)
6. [Testing Strategy](#6-testing-strategy)
7. [Known Limitations](#7-known-limitations)
8. [Phase 3 Integration](#8-phase-3-integration)

---

## 1. Overview

### 1.1 Purpose

This document describes the complete flow from neural network models to executable code for the tensor accelerator. The flow enables:

- Compiling ONNX models to accelerator assembly
- Generating test vectors with golden references
- Preparing files for RTL simulation

### 1.2 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT SOURCES                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ONNX Model          PyTorch Model         Hand-built IR Graph    │
│   (.onnx)             (via export)          (Python API)           │
│       │                    │                      │                 │
│       └────────────────────┼──────────────────────┘                 │
│                            │                                        │
│                            ▼                                        │
│                  ┌─────────────────┐                               │
│                  │   IR Graph      │                               │
│                  │   (in memory)   │                               │
│                  └────────┬────────┘                               │
│                           │                                        │
└───────────────────────────┼─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      COMPILER PIPELINE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐              │
│   │  Quantizer  │──▶│   Tiler     │──▶│  Scheduler  │              │
│   │ FP32→INT8   │   │ Break into  │   │ Order ops,  │              │
│   │             │   │ HW tiles    │   │ assign TPCs │              │
│   └─────────────┘   └─────────────┘   └──────┬──────┘              │
│                                              │                      │
│                                              ▼                      │
│                                       ┌─────────────┐              │
│                                       │  CodeGen    │              │
│                                       │ Emit ASM    │              │
│                                       └──────┬──────┘              │
│                                              │                      │
└──────────────────────────────────────────────┼──────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         ASSEMBLER                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   program.asm                                                       │
│   (human-readable)                                                  │
│        │                                                            │
│        ▼                                                            │
│   ┌─────────────┐                                                  │
│   │  Assembler  │  Parse instructions, encode to 128-bit binary    │
│   │             │                                                  │
│   └──────┬──────┘                                                  │
│          │                                                          │
│          ▼                                                          │
│   program.hex                                                       │
│   (128-bit hex, $readmemh format)                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      RTL SIMULATION                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Test Package:                                                     │
│   ├── program.hex   ──▶  Instruction Memory ($readmemh)            │
│   ├── weights.memh  ──▶  Weight SRAM ($readmemh)                   │
│   ├── input.memh    ──▶  Input SRAM ($readmemh)                    │
│   └── golden.memh   ──▶  Expected Output (for comparison)          │
│                                                                     │
│   Verilog Testbench / cocotb:                                      │
│   1. Initialize memories from .memh files                          │
│   2. Start accelerator execution                                   │
│   3. Wait for completion (HALT instruction)                        │
│   4. Read output from SRAM                                         │
│   5. Compare against golden.memh                                   │
│   6. Report PASS/FAIL                                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Architecture

### 2.1 Hardware Target

The compiler targets the tensor accelerator with these specifications:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Systolic Array | 8×8 | 64 MACs per cycle |
| Data Type | INT8 | Weights and activations |
| Accumulator | INT32 | Prevents overflow |
| SRAM per TPC | 2 MB | Local scratchpad |
| Number of TPCs | 4 | 2×2 grid |
| Total Compute | 256 MACs/cycle | 4 TPCs × 64 MACs |

### 2.2 Memory Map

```
SRAM Address Space (per TPC):
┌────────────────────────────────────────┐
│ 0x000000 - 0x03FFFF: Activation A      │  256 KB
├────────────────────────────────────────┤
│ 0x040000 - 0x07FFFF: Activation B      │  256 KB
├────────────────────────────────────────┤
│ 0x080000 - 0x17FFFF: Weights           │  1 MB
├────────────────────────────────────────┤
│ 0x180000 - 0x1BFFFF: Output            │  256 KB
├────────────────────────────────────────┤
│ 0x1C0000 - 0x1FFFFF: Scratch           │  256 KB
└────────────────────────────────────────┘

DDR Address Space:
┌────────────────────────────────────────┐
│ 0x00100000: Weight Storage             │
├────────────────────────────────────────┤
│ 0x00200000: Input Storage              │
├────────────────────────────────────────┤
│ 0x00300000: Output Storage             │
└────────────────────────────────────────┘
```

### 2.3 Instruction Set Summary

| Category | Instructions | Description |
|----------|--------------|-------------|
| TENSOR | GEMM, GEMM_ACC, GEMM_RELU | Matrix operations on systolic array |
| VECTOR | ADD, MUL, RELU, SOFTMAX, etc. | Elementwise operations |
| DMA | LOAD_1D, LOAD_2D, STORE_1D, STORE_2D | Memory transfers |
| SYNC | WAIT_MXU, WAIT_VPU, WAIT_DMA | Synchronization |
| CONTROL | NOP, HALT, LOOP, ENDLOOP | Program control |

---

## 3. Component Details

### 3.1 Frontend (ONNX Parser)

**Location:** `sw/compiler/frontend/onnx_parser.py`

**Purpose:** Load ONNX models and convert to internal IR

**Supported Operations:**

| ONNX Op | IR OpType | Status |
|---------|-----------|--------|
| Gemm | GEMM | ✅ Full |
| MatMul | MATMUL | ✅ Full |
| Conv | CONV2D | ✅ Via im2col |
| Relu | RELU | ✅ Full |
| Add/Sub/Mul/Div | ADD/SUB/MUL/DIV | ✅ Full |
| Softmax | SOFTMAX | ✅ 3-pass |
| MaxPool | MAXPOOL | ⚠️ Partial |
| BatchNormalization | BATCHNORM | 🔜 Planned |

**Usage:**
```python
from frontend.onnx_parser import load_onnx

graph = load_onnx("model.onnx", verbose=True)
```

### 3.2 Quantizer

**Location:** `sw/compiler/quantizer/quantizer.py`

**Purpose:** Convert FP32 weights/activations to INT8

**Features:**
- Symmetric quantization (zero_point = 0)
- Asymmetric quantization (optimized range)
- Per-tensor quantization
- Per-channel quantization (for weights)

**Usage:**
```python
from quantizer.quantizer import Quantizer

quantizer = Quantizer(method='symmetric')
q_graph = quantizer.quantize_weights_only(graph)
# Or with calibration:
q_graph = quantizer.quantize(graph, calibration_data)
```

### 3.3 Tiler

**Location:** `sw/compiler/tiler/tiler.py`

**Purpose:** Break large operations into hardware-sized tiles

**Strategy for GEMM (C = A × B):**
1. Tile K dimension first (minimize accumulation passes)
2. Tile M dimension (fits in activation buffer)
3. Tile N dimension (fits in output buffer)
4. Align all tiles to 8×8 systolic array

**Example:**
```
GEMM(256×256×256):
  Tile size: 256×256×256 (fits in SRAM)
  Tiles: 1×1×1 = 1 total

GEMM(1024×1024×1024):
  Tile size: 512×512×512
  Tiles: 2×2×2 = 8 total
```

### 3.4 Scheduler

**Location:** `sw/compiler/scheduler/scheduler.py`

**Purpose:** Determine execution order and resource allocation

**Responsibilities:**
- Topological sort of operations
- TPC assignment (round-robin)
- SRAM address allocation
- DMA transfer scheduling
- Dependency tracking for tiles

### 3.5 Code Generator

**Location:** `sw/compiler/codegen/codegen.py`

**Purpose:** Emit assembly instructions

**Output Format:**
```asm
# Comment
.equ SYMBOL, VALUE

TENSOR.GEMM dst, src_a, src_b, M, N, K
SYNC.WAIT_MXU
VECTOR.RELU dst, src, count
SYNC.WAIT_VPU
DMA.LOAD_1D sram_addr, ddr_addr, size
SYNC.WAIT_DMA
HALT
```

### 3.6 Assembler

**Location:** `sw/assembler/assembler.py`

**Purpose:** Convert assembly to binary

**Instruction Encoding (128 bits):**
```
[127:120] opcode     (8 bits)
[119:112] subop      (8 bits)
[111:96]  dst        (16 bits)
[95:80]   src0       (16 bits)
[79:64]   src1       (16 bits)
[63:48]   dim_m      (16 bits)
[47:32]   dim_n      (16 bits)
[31:16]   dim_k      (16 bits)
[15:0]    flags      (16 bits)
```

---

## 4. File Formats

### 4.1 Assembly (.asm)

Human-readable source code:

```asm
# Matrix multiplication test
.equ WEIGHT_ADDR, 0x00100000
.equ SRAM_ACT_A, 0x0000
.equ SRAM_WT_A, 0x2000
.equ SRAM_OUT, 0x6000

# Load weights from DDR
DMA.LOAD_1D SRAM_WT_A, WEIGHT_ADDR, 256
SYNC.WAIT_DMA

# Execute GEMM
TENSOR.GEMM SRAM_OUT, SRAM_ACT_A, SRAM_WT_A, 16, 16, 16
SYNC.WAIT_MXU

HALT
```

### 4.2 Hex Instructions (.hex)

Binary format for `$readmemh`:

```
01016000000020000010001000100000
04010000000000000000000000000000
ff000000000000000000000000000000
```

Each line is 32 hex characters = 128 bits = one instruction.

### 4.3 Memory Data (.memh)

Data format for `$readmemh` with configurable width:

```
// 256-bit wide (32 bytes per line)
0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20
2122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f40
```

### 4.4 Test Configuration (.json)

Metadata for test execution:

```json
{
  "name": "gemm_16x16",
  "type": "gemm",
  "dimensions": {"M": 16, "N": 16, "K": 16},
  "seed": 42,
  "input_range": {"min": -64, "max": 63},
  "output_range": {"min": -20543, "max": 16616},
  "instructions": 3,
  "files": {
    "hex": "program.hex",
    "weights_memh": "weights.memh",
    "input_memh": "input.memh",
    "golden_memh": "golden.memh"
  }
}
```

---

## 5. Usage Guide

### 5.1 Command Line

**Compile ONNX model:**
```bash
cd sw/compiler
python compile.py model.onnx -o output.asm -v
```

**Run test compilation:**
```bash
python compile.py --test
```

**Generate E2E test packages:**
```bash
python generate_e2e_tests.py --all -o ../../tests/e2e
```

### 5.2 Python API

**Basic compilation:**
```python
from compile import Compiler

compiler = Compiler(verbose=True)
asm_code, weight_data, graph = compiler.compile("model.onnx")

# Write outputs
with open("program.asm", "w") as f:
    f.write(asm_code)
```

**Custom graph:**
```python
from ir.graph import Graph, Node, Tensor, OpType, DataType

# Create graph
g = Graph(name="my_model")
g.add_tensor(Tensor("input", (1, 64), DataType.INT8))
g.add_tensor(Tensor("weight", (64, 32), DataType.INT8, data=weights))
g.add_tensor(Tensor("output", (1, 32), DataType.INT8))
g.add_node(Node("fc", OpType.GEMM, ["input", "weight"], ["output"]))
g.inputs.append("input")
g.outputs.append("output")

# Compile
asm, weights = compiler.compile_graph(g)
```

**Generate test package:**
```python
from generate_e2e_tests import E2ETestGenerator

gen = E2ETestGenerator("./tests")
config = gen.generate_gemm_test(M=32, N=32, K=32, name="my_gemm")
```

---

## 6. Testing Strategy

### 6.1 Current Test Coverage

| Test Suite | Location | Tests | Status |
|------------|----------|:-----:|--------|
| Compiler Unit | `test_compiler.py` | 7 | ✅ All passing |
| Integration | `test_integration.py` | 3 | ✅ All passing |
| E2E Generation | `generate_e2e_tests.py` | 6 packages | ✅ Generated |

### 6.2 Test Categories

**Unit Tests (test_compiler.py):**
1. IR creation and validation
2. Quantization correctness
3. Tiling computations
4. Scheduler ordering
5. Code generation syntax
6. Full pipeline execution
7. Custom graph (LeNet-like)

**Integration Tests (test_integration.py):**
1. Compiler → Assembler syntax compatibility
2. MLP end-to-end compilation
3. Golden reference generation

### 6.3 What Phase 2 Tests Verify

| Aspect | Tested | Method |
|--------|:------:|--------|
| IR graph construction | ✅ | Unit tests |
| Topological sort | ✅ | Unit tests |
| Quantization math | ✅ | NumPy comparison |
| Tile size computation | ✅ | Memory bound checks |
| Assembly syntax | ✅ | Assembler parsing |
| Hex encoding | ✅ | Length validation |
| Golden reference | ✅ | NumPy matmul |

### 6.4 What Phase 2 Does NOT Test

| Aspect | Status | Resolution |
|--------|--------|------------|
| RTL execution | ❌ | Phase 3 |
| Verilog $readmemh loading | ❌ | Phase 3 |
| Hardware correctness | ❌ | Phase 3 |
| Real ONNX models | ⚠️ | Needs ONNX install |
| Large model tiling | ⚠️ | Manual verification |
| DMA correctness | ❌ | Phase 3 |

### 6.5 Recommended Additional Phase 2 Tests

Before Phase 3, consider testing:

1. **Verilog Format Validation**
   - Load .hex file in Icarus Verilog
   - Verify $readmemh succeeds

2. **Assembly Round-Trip**
   - Assemble → Disassemble → Compare

3. **Large Matrix Tiling**
   - Test 1024×1024 GEMM tiling
   - Verify tile dependencies

4. **ONNX Model Import** (requires `pip install onnx`)
   - Export PyTorch model to ONNX
   - Import and compile

---

## 7. Known Limitations

### 7.1 Compiler Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| No operator fusion | Suboptimal code | Manual fusion in IR |
| Simple scheduler | May not minimize DMA | Future optimization |
| No double-buffering | Compute/DMA not overlapped | Phase 5 |
| Limited ONNX ops | Some models won't compile | Add op support |

### 7.2 Unsupported Operations

| Operation | Reason | Plan |
|-----------|--------|------|
| Transpose | Not in ISA | Add VECTOR.TRANSPOSE |
| Concat | Multi-output | Future |
| Split | Multi-output | Future |
| Dynamic shapes | Static compiler | Not planned |

### 7.3 Known Issues

1. **DMA addresses not fully connected** - Codegen emits addresses but DMA loads may need adjustment for actual memory layout

2. **Tile dependencies simplified** - Only K-dimension dependencies tracked; M/N parallelism not exploited

3. **No error recovery** - Assembler failures stop immediately

---

## 8. Phase 3 Integration

### 8.1 What Phase 3 Will Add

```
Phase 2 Output                    Phase 3 Addition
─────────────────                 ────────────────
program.hex      ─────────────▶   cocotb test that loads
weights.memh     ─────────────▶   into RTL and executes
input.memh       ─────────────▶   
golden.memh      ─────────────▶   comparison logic
```

### 8.2 cocotb Test Structure

```python
@cocotb.test()
async def test_gemm_16x16(dut):
    # 1. Load test package
    test_dir = "tests/e2e/gemm_16x16"
    
    # 2. Initialize memories
    load_memh(dut.instr_mem, f"{test_dir}/program.hex")
    load_memh(dut.weight_sram, f"{test_dir}/weights.memh")
    load_memh(dut.input_sram, f"{test_dir}/input.memh")
    
    # 3. Start execution
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    # 4. Wait for HALT
    while dut.running.value:
        await RisingEdge(dut.clk)
    
    # 5. Read and compare output
    output = read_sram(dut.output_sram)
    golden = load_memh(f"{test_dir}/golden.memh")
    
    assert np.array_equal(output, golden), "Output mismatch!"
```

### 8.3 Success Criteria for Phase 3

| Test | Requirement |
|------|-------------|
| gemm_8x8 | Output matches golden |
| gemm_16x16 | Output matches golden |
| gemm_32x32 | Output matches golden |
| gemm_64x64 | Output matches golden |
| mlp_small | Output matches golden |
| mlp_medium | Output matches golden |

---

## Appendix A: File Tree

```
tensor_accelerator/
├── sw/
│   ├── assembler/
│   │   └── assembler.py          # ASM → HEX
│   └── compiler/
│       ├── compile.py            # Main entry point
│       ├── ir/
│       │   └── graph.py          # IR definitions
│       ├── frontend/
│       │   └── onnx_parser.py    # ONNX import
│       ├── quantizer/
│       │   └── quantizer.py      # FP32 → INT8
│       ├── tiler/
│       │   └── tiler.py          # Operation tiling
│       ├── scheduler/
│       │   └── scheduler.py      # Execution ordering
│       ├── codegen/
│       │   └── codegen.py        # ASM generation
│       ├── test_compiler.py      # Unit tests
│       ├── test_integration.py   # Integration tests
│       └── generate_e2e_tests.py # E2E test generator
└── tests/
    └── e2e/
        ├── gemm_8x8/
        ├── gemm_16x16/
        ├── gemm_32x32/
        ├── gemm_64x64/
        ├── mlp_small/
        └── mlp_medium/
```

---

## Appendix B: Quick Reference

**Compile a model:**
```bash
python sw/compiler/compile.py model.onnx -o program.asm
```

**Run compiler tests:**
```bash
python sw/compiler/test_compiler.py
python sw/compiler/test_integration.py
```

**Generate test packages:**
```bash
python sw/compiler/generate_e2e_tests.py --all -o tests/e2e
```

**Assemble to hex:**
```bash
python sw/assembler/assembler.py program.asm -o program.hex
```

---

*End of Document*

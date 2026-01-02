# Building an AI Accelerator from Scratch
## Part 3: Verification — Testing Strategy, Coverage, and Open-Source EDA Tooling

*How we achieved 100% test pass rate across 53 tests without any commercial verification tools*

---

## Introduction

This is Part 3 of our series on building an AI accelerator from scratch. In the previous articles, we covered the architecture decisions (Part 1) and the Python models plus RTL implementation (Part 2). Now we tackle something equally important: **how do you know it actually works?**

Verification is where most hardware projects fail. The RTL might simulate correctly for toy examples but break on real workloads. We took a systematic approach: bottom-up verification starting from individual PEs, building up to full model inference, using entirely open-source tools.

**Final Result**: 53 tests, 100% pass rate, covering everything from single MAC operations to complete LeNet-5 inference.

---

## The Verification Philosophy

### Why Verification is Hard for Accelerators

Traditional CPU verification focuses on instruction coverage—every opcode, every corner case. Accelerators are different:

1. **Data-dependent behavior**: A GEMM with all-zero inputs behaves differently than one with saturating values
2. **Timing complexity**: Systolic arrays have intricate data skewing requirements—off-by-one errors cause silent corruption
3. **Scale**: Testing a 16×16 systolic array exhaustively would take longer than the age of the universe
4. **Emergent bugs**: Individual modules can pass unit tests but fail when composed

### Our Approach: Bottom-Up with Python Golden Models

```
Level 5: Full Model E2E      ─────────────────────────────────────
         (LeNet-5, ResNet)              Phase F
                                           │
Level 4: Stress Testing      ─────────────────────────────────────
         (Multi-TPC, Pipeline)          Phase E
                                           │
Level 3: Transformer Ops     ─────────────────────────────────────
         (LayerNorm, Attention)         Phase D
                                           │
Level 2: Layer Operations    ─────────────────────────────────────
         (Requant, Chaining)            Phase C
                                           │
Level 1: Realistic Workloads ─────────────────────────────────────
         (LeNet layers, ResNet block)   Phase A/B
                                           │
Level 0: Unit/Integration    ─────────────────────────────────────
         (PE, Array, TPC, NoC)          Baseline
```

The key insight: **Python models generate the expected outputs.** The RTL testbenches compare against these golden references. If they diverge, we debug the RTL. If the Python model has a bug, we fix it first—Python is much easier to debug than Verilog.

---

## Test Categories and Coverage

### The Complete Test Suite: 53 Tests

| Category | Purpose | Count |
|----------|---------|-------|
| Unit Tests | Individual module verification | 9 |
| Integration Tests | Multi-module interaction | 2 |
| End-to-End Tests | Full datapath verification | 5 |
| Model Tests | Algorithm-level verification | 10 |
| Realistic Tests | Real workload patterns | 21 |
| Python Model | Cycle-accurate reference | 6 |
| **Total** | | **53** |

### Unit Tests: The Foundation

Every major module has a standalone testbench:

| Test | Module | What It Verifies |
|------|--------|------------------|
| `tb_mac_pe` | MAC PE | Single multiply-accumulate with weight loading |
| `tb_systolic_array` | Systolic Array | 8×8 array operation, skewing, result collection |
| `tb_vector_unit` | VPU | Element-wise ops (ADD, MUL, RELU, GELU) |
| `tb_dma_engine` | DMA | 2D strided transfers, burst handling |
| `tb_sram_subsystem` | SRAM | Bank arbitration, read latency, XOR addressing |
| `tb_noc_router` | NoC Router | Packet routing, flow control |

#### The MAC PE Test: Starting Simple

The MAC PE is the atom of the accelerator—get this wrong and nothing else matters.

```verilog
// tb_mac_pe.v - Key test cases
initial begin
    // Test 1: Basic MAC operation
    // weight=5, activation=3, psum_in=0 -> expect 15
    
    // Test 2: Accumulation
    // weight=5, activation=3, psum_in=10 -> expect 25
    
    // Test 3: Negative values
    // weight=-5, activation=3, psum_in=0 -> expect -15
    
    // Test 4: Saturation (INT8 range)
    // weight=127, activation=127 -> expect 16129 (fits in INT32)
end
```

### Integration Tests: Modules Working Together

Once individual modules pass, we test interactions:

**TPC Integration (`tb_tpc`)**: Tests the complete Tensor Processing Cluster—LCP issuing commands to MXU, VPU, and DMA, with SRAM arbitration.

**NoC Mesh (`tb_noc_mesh_2x2`)**: Tests routing between 4 TPCs. Key scenarios:
- Single packet routing (all 12 paths in 2×2 mesh)
- Multi-packet concurrent routing
- Backpressure handling when destination is busy

### End-to-End Tests: Full Datapath

These are the "money tests"—if these pass, the core functionality works.

| Test | Description | Cycles | MACs |
|------|-------------|--------|------|
| `tb_e2e_gemm` | Basic 8×8 GEMM | 64 | 512 |
| `tb_tiled_gemm` | 16×16 tiled GEMM | 512 | 4,096 |
| `tb_multi_tpc_gemm` | 4 TPCs in parallel | 8,192 | 1.08M |

---

## The Phased Verification Plan

### Phase Baseline: Core Functionality (26 tests)

Before any "realistic" tests, we verified fundamentals:

- MAC PE accumulation across INT8 range
- Systolic array data flow (skewing is correct)
- Weight stationary operation
- TPC integration (LCP→MXU→SRAM)
- NoC routing (all paths)
- SRAM read/write timing
- DMA burst transfers

**Key Bug Found**: Early versions had an off-by-one error in output de-skewing. The Python model caught this:

```python
# WRONG: de-skew delay = N - 1 - col
# CORRECT: de-skew delay = 2 * (N - 1 - col)

# For 8×8 array:
# Column 0: delay 14 cycles
# Column 7: delay 0 cycles
```

This bug would have caused silent data corruption—results would look plausible but be wrong by one cycle.

### Phase A/B: Realistic Workloads (4 tests)

After baseline passed, we tested with real network layer dimensions:

| Test | Network | Layer | Dimensions |
|------|---------|-------|------------|
| LeNet Conv1 | LeNet-5 | First convolution | 28×28×1 → 24×24×6 |
| LeNet Pool3 | LeNet-5 | Max pooling | 12×12×6 → 6×6×6 |
| LeNet FC7 | LeNet-5 | Fully connected | 256 → 120 |
| ResNet Block | ResNet-18 | Basic block | 56×56×16 → 56×56×16 |

**Why These Dimensions Matter**: These aren't nice power-of-2 sizes. LeNet Conv1 uses 5×5 kernels (not 3×3). ResNet basic block has skip connections. Real networks stress the hardware in ways toy tests don't.

### Phase C: Requantization and Layer Chaining (4 tests)

Neural network layers don't operate in isolation—the output of one becomes the input of another. This requires requantization:

```
INT32 Accumulator (from GEMM)
       │
       ▼
┌─────────────────┐
│  >> shift       │  (arithmetic right shift, configurable)
│  (7-10 bits)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Saturate      │  (clip to [-128, 127])
│   to INT8       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Optional      │  (max(x, 0))
│   ReLU          │
└────────┬────────┘
         │
         ▼
    INT8 Output → Input to next layer
```

**Tests in This Phase**:
- `tb_requant`: Shift 7-10 bits, verify saturation
- `tb_bias_fusion`: Add bias + ReLU in single pass
- `tb_layer_chain`: Conv → ReLU → Pool (3-layer chain)
- `tb_lenet_chain`: LeNet-style 2-layer chain

### Phase D: Transformer Operations (4 tests)

Modern AI is all about transformers. We verified the attention-specific operations:

**LayerNorm**: Normalize activations across feature dimension
```
Input (INT8) → Dequantize → Compute mean/var → Normalize → Scale → Requantize → Output (INT8)
```

**Softmax**: Row-wise probability distribution
```
Input (INT8) → Dequantize → Find max → exp(x-max) via LUT → Normalize → Scale to 0-127 → Output (INT8)
```

**GELU**: The activation function of choice for transformers
- Implemented as 256-entry lookup table
- Max error vs. float reference: 0.8% (acceptable for inference)

**Attention**: The complete QK^T → softmax → AV pipeline
```
Q, K, V (INT8)
     │
     ▼
┌─────────────┐
│  QK^T       │  (GEMM on systolic array)
│  INT32 acc  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Scale by   │  (1/√d_k)
│  1/√d_k     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Softmax    │  (row-wise, INT8 output)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Attn × V   │  (GEMM on systolic array)
└──────┬──────┘
       │
       ▼
   Requantize → INT8 Output
```

### Phase E: Stress Testing (3 tests)

Now we tried to break it:

**Back-to-Back GEMM (Pipeline Test)**:
```
Stage 1: A(32,24) × B(24,16) → C     192 cycles
              │
              ▼ requantize
Stage 2: C(32,16) × D(16,12) → E     128 cycles
              │
              ▼ requantize
Stage 3: E(32,12) × F(12,8)  → G      48 cycles
                                    ─────────
                              Total: 368 cycles
                              MACs:  21,504
                              Throughput: 58.4 MACs/cycle
```

**Multi-TPC Parallel**: All 4 TPCs running different operations simultaneously
- TPC0: Query × Key^T attention
- TPC1: Attention × Value
- TPC2: Large FC layer
- TPC3: Another large FC layer

**Result**: 2.06× speedup vs. sequential, 132 MACs/cycle effective throughput

**Boundary Conditions**: Specifically testing edge cases
| Subtest | Input | Result |
|---------|-------|--------|
| Non-aligned | 7×13×5 (weird dimensions) | ✅ Pass |
| Single element | 100×50×1 | ✅ Pass |
| Max INT8 | 127×127×8 | 129,032 ✅ |
| Min INT8 | -128×-128×8 | 131,072 ✅ |
| Mixed extremes | 127×-128×8 | -130,048 ✅ |

### Phase F: Full Model E2E (3 tests)

The ultimate test: can we run complete inference?

#### LeNet-5 Full Inference

```
Input: (1, 1, 28, 28) INT8 ─── "MNIST-like image"
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: Conv1  (1,6,5,5) + ReLU                 ✅ 0 errors   │
│          Output: (1, 6, 24, 24)                                │
├─────────────────────────────────────────────────────────────────┤
│ Layer 2: Pool1  MaxPool 2×2                      ✅ 0 errors   │
│          Output: (1, 6, 12, 12)                                │
├─────────────────────────────────────────────────────────────────┤
│ Layer 3: Conv2  (6,16,5,5) + ReLU                ✅ 0 errors   │
│          Output: (1, 16, 8, 8)                                 │
├─────────────────────────────────────────────────────────────────┤
│ Layer 4: Pool2  MaxPool 2×2                      ✅ 0 errors   │
│          Output: (1, 16, 4, 4) = 256                           │
├─────────────────────────────────────────────────────────────────┤
│ Layer 5: FC1  256→120 + ReLU                     ✅ 0 errors   │
├─────────────────────────────────────────────────────────────────┤
│ Layer 6: FC2  120→84 + ReLU                      ✅ 0 errors   │
├─────────────────────────────────────────────────────────────────┤
│ Layer 7: FC3  84→10 (logits)                     ✅ 0 errors   │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
Output: Logits → Predicted Class: 3 ✅

Total Cycles: 286,120
```

#### ResNet Basic Block

Complete residual block with skip connection:
```
Input: (1, 16, 14, 14) INT8
              │
              ├──────────────────────────────┐
              │                              │
              ▼                              │
┌─────────────────────────┐                  │
│ Conv1: 3×3, pad=1       │                  │
│ + ReLU                  │                  │
│ Output: (1,16,14,14)    │                  │
│ ✅ 0 errors             │                  │
└───────────┬─────────────┘                  │
            │                                │
            ▼                                │
┌─────────────────────────┐                  │
│ Conv2: 3×3, pad=1       │                  │
│ (no ReLU)               │                  │
└───────────┬─────────────┘                  │
            │                                │
            ▼                                │
┌─────────────────────────┐                  │
│ Residual Add            │◄─────────────────┘
│ Conv2 + Input           │    (skip connection)
│ ✅ 0 errors             │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Final ReLU              │
│ ✅ 0 errors             │
└─────────────────────────┘

Total Cycles: 903,168
```

---

## The Open-Source EDA Toolchain

### No Commercial Tools Required

We achieved 100% test coverage using only open-source tools:

| Tool | Purpose | Why We Chose It |
|------|---------|-----------------|
| **Icarus Verilog** | RTL simulation | Fast, reliable, 100% Verilog-2005 support |
| **Verilator** | Fast simulation | 10-100× faster for long tests, lint checking |
| **cocotb** | Python testbenches | Write verification in Python, not SystemVerilog |
| **Yosys** | Synthesis | Check synthesizability, estimate gate counts |
| **GTKWave** | Waveform viewing | Debug timing issues |

### Why cocotb Changed Everything

Traditional Verilog testbenches are painful:
- Verbose syntax for simple operations
- No native data structures (good luck with test vectors)
- String handling is a disaster
- Can't easily compare against Python reference

cocotb lets us write testbenches in Python:

```python
import cocotb
from cocotb.triggers import RisingEdge, Timer
import numpy as np

@cocotb.test()
async def test_gemm(dut):
    """Test 8x8 GEMM operation"""
    # Generate random inputs
    A = np.random.randint(-128, 128, (8, 8), dtype=np.int8)
    B = np.random.randint(-128, 128, (8, 8), dtype=np.int8)
    
    # Python golden model
    expected = A.astype(np.int32) @ B.astype(np.int32)
    
    # Load inputs to DUT
    await load_weights(dut, B)
    await load_activations(dut, A)
    
    # Start computation
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    # Wait for completion
    while not dut.done.value:
        await RisingEdge(dut.clk)
    
    # Read and compare output
    result = await read_output(dut)
    np.testing.assert_array_equal(result, expected)
```

Benefits:
- NumPy for generating test vectors and computing golden outputs
- Real assertions with meaningful error messages
- Easy to parameterize tests (dimensions, ranges, etc.)
- Can reuse Python model code directly

### Test Execution Infrastructure

We built a unified test runner:

```bash
# Run all tests
./run_tests.sh --all

# Run specific phase
./run_tests.sh --phase baseline
./run_tests.sh --phase f

# Run with coverage
./run_tests.sh --coverage

# Run with waveform dump
./run_tests.sh --waves tb_systolic_array
```

Output:
```
============================================================
Tensor Accelerator Test Suite
============================================================
Running 53 tests...

[BASELINE] tb_mac_pe                    PASS (0.3s)
[BASELINE] tb_systolic_array            PASS (1.2s)
[BASELINE] tb_vector_unit               PASS (0.8s)
...
[PHASE F]  tb_lenet5_full               PASS (45.2s)
[PHASE F]  tb_resnet_block_full         PASS (52.1s)
[PHASE F]  tb_batch_inference           PASS (8.3s)

============================================================
Results: 53/53 tests passed (100%)
Total time: 4m 23s
============================================================
```

---

## Key Bugs Found During Verification

### Bug #1: Systolic Array De-Skewing (Found in Phase Baseline)

**Symptom**: Results were shifted by one element

**Root Cause**: De-skewing delay formula was wrong:
```python
# Bug: delay = N - 1 - col
# Fix: delay = 2 * (N - 1 - col)
```

**How Python Model Found It**: The Python model had the same bug initially. When we drew the timing diagram by hand, we realized both were wrong. Fixed Python first, then RTL.

### Bug #2: SRAM Read Latency (Found in Phase A)

**Symptom**: First result element was wrong, rest were correct

**Root Cause**: SRAM has 1-cycle read latency, but DMA wasn't accounting for it. The first read returned stale data.

**Fix**: Added capture state in DMA FSM:
```verilog
S_READ: begin
    sram_re <= 1;
    state <= S_CAPTURE;  // NEW: wait for data
end
S_CAPTURE: begin
    data_buf <= sram_rdata;  // Now data is valid
    state <= S_PROCESS;
end
```

### Bug #3: LCP Halt Timing (Found in Phase E)

**Symptom**: Tests hung indefinitely

**Root Cause**: HALT instruction asserted `done` in same cycle as transitioning to HALTED state. The testbench checked `done` before it was stable.

**Fix**: Assert `done` one cycle after entering HALTED state.

### Bug #4: Multi-TPC SRAM Contention (Found in Phase E)

**Symptom**: Incorrect results when all 4 TPCs ran simultaneously

**Root Cause**: AXI arbiter had a priority inversion bug—lower-priority TPC could starve higher-priority ones under specific timing conditions.

**Fix**: Implemented proper round-robin arbitration with fairness guarantees.

---

## Coverage Analysis

### Operation Coverage Matrix

| Operation | Unit | Integration | E2E | Model | Realistic |
|-----------|:----:|:-----------:|:---:|:-----:|:---------:|
| INT8×INT8→INT32 MAC | ✅ | ✅ | ✅ | ✅ | ✅ |
| Systolic Array GEMM | ✅ | ✅ | ✅ | ✅ | ✅ |
| Tiled GEMM | - | ✅ | ✅ | ✅ | ✅ |
| Conv2D (im2col) | - | - | - | ✅ | ✅ |
| MaxPool 2×2 | - | - | - | ✅ | ✅ |
| ReLU | - | - | - | ✅ | ✅ |
| GELU (LUT) | - | - | - | - | ✅ |
| LayerNorm | - | - | - | - | ✅ |
| Softmax | - | - | - | - | ✅ |
| Attention | - | - | - | ✅ | ✅ |
| Requantization | - | - | - | - | ✅ |
| Multi-TPC Parallel | - | - | ✅ | - | ✅ |

### Quantization Coverage

| Aspect | Coverage | Status |
|--------|----------|--------|
| INT8 weights | Full range [-128, 127] | ✅ |
| INT8 activations | Full range [-128, 127] | ✅ |
| INT32 accumulators | Overflow tested | ✅ |
| Requantization shifts | 7-10 bits | ✅ |
| Saturation | Both directions | ✅ |

### What's NOT Covered (Known Limitations)

- **Random delay injection**: We test fixed timing, not random backpressure
- **Clock domain crossings**: Single clock domain for now
- **Power-on reset sequence**: Assumed clean reset
- **ECC/fault injection**: No memory error testing

---

## Lessons Learned

### 1. Python Models First, Always

Writing the Python model before RTL:
- Forces you to understand the algorithm deeply
- Catches design bugs before they become RTL bugs
- Provides golden reference for all tests

### 2. Test Realistic Dimensions Early

Toy 4×4 tests find basic bugs. But real networks have:
- Non-power-of-2 dimensions
- Odd tiling boundaries
- Skip connections and residuals

Test with LeNet/ResNet dimensions from Day 1.

### 3. Open-Source Tools Are Production-Ready

We achieved professional-grade verification without spending $100K+ on commercial tools:
- Icarus Verilog: Reliable, well-documented
- cocotb: Game-changer for verification productivity
- Verilator: Fast enough for long regression tests

### 4. Invest in Test Infrastructure

Building `run_tests.sh` and proper CI/CD early paid dividends:
- Easy to run regressions
- Consistent output format
- Automatic coverage tracking

---

## Summary

Verification is not optional—it's the difference between a working accelerator and an expensive paperweight. Our approach:

1. **Bottom-up testing**: Unit → Integration → E2E → Model → Full inference
2. **Python golden models**: Every RTL test has a Python reference
3. **Phased verification**: Baseline first, then realistic workloads, then stress
4. **Open-source tooling**: cocotb + Icarus Verilog + Verilator

**Final Score**: 53 tests, 100% pass rate, including complete LeNet-5 and ResNet block inference.

In Part 4, we'll cover the complete software flow: how we compile ONNX models to assembly, generate test vectors, and prepare everything for RTL simulation.

---

*Questions or comments? I'd love to discuss verification strategies for accelerators. The biggest gap between academic papers and working silicon is often the verification—happy to share more details on any aspect of our approach.*

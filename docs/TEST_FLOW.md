# Tensor Accelerator Test Flow

## Overview

This document describes the step-by-step verification flow for the tensor accelerator, covering both ResNet-style CNN layers and LLM attention layers.

---

## Directory Structure

```
tensor_accelerator/
├── rtl/                    # Verilog RTL source files
│   ├── core/               # MAC PE, systolic array, VPU, DMA
│   ├── memory/             # SRAM subsystem
│   ├── control/            # LCP, GCP
│   ├── noc/                # Network-on-chip router
│   └── top/                # TPC and top-level integration
├── tb/                     # Testbenches
├── sim/                    # Simulation outputs
├── sw/                     # Software tools
│   ├── assembler/          # LCP instruction assembler
│   ├── examples/           # Example kernels
│   └── test_generator.py   # Test vector generator
├── constraints/            # FPGA constraints
└── docs/                   # Documentation
```

---

## Step-by-Step Test Flow

### Phase 1: Unit-Level Testing

#### 1.1 Test the MAC Processing Element

```bash
cd tensor_accelerator/sim

# Compile MAC PE
iverilog -o mac_pe_tb ../rtl/core/mac_pe.v ../tb/tb_mac_pe.v

# Run simulation
vvp mac_pe_tb

# View waveforms (optional)
gtkwave mac_pe_tb.vcd
```

**What to verify:**
- Weight loading works correctly
- Activation passes through with 1-cycle delay
- Multiply-accumulate produces correct results
- Sign extension is handled properly for INT8

#### 1.2 Test the Systolic Array

```bash
# Compile systolic array
iverilog -o systolic_tb \
    ../rtl/core/mac_pe.v \
    ../rtl/core/systolic_array.v \
    ../tb/tb_systolic_array.v

# Run simulation
vvp systolic_tb
```

**What to verify:**
- 16×16 GEMM produces correct results
- Input skewing works (activations are properly staggered)
- Output draining works (results emerge in correct order)
- Accumulation across K dimension works

#### 1.3 Test the Vector Unit

```bash
iverilog -o vpu_tb \
    ../rtl/core/vector_unit.v \
    ../tb/tb_vector_unit.v

vvp vpu_tb
```

**What to verify:**
- SIMD operations (ADD, MUL, RELU) work across all lanes
- Reduction operations (SUM, MAX) produce correct scalar
- Load/Store to SRAM works
- Register file read/write works

---

### Phase 2: Block-Level Testing

#### 2.1 Test Single TPC

```bash
# Compile full TPC
iverilog -o tpc_tb \
    ../rtl/core/mac_pe.v \
    ../rtl/core/systolic_array.v \
    ../rtl/core/vector_unit.v \
    ../rtl/core/dma_engine.v \
    ../rtl/memory/sram_subsystem.v \
    ../rtl/control/local_cmd_processor.v \
    ../rtl/top/tensor_processing_cluster.v \
    ../tb/tb_single_tpc.v

vvp tpc_tb
```

**What to verify:**
- LCP fetches and decodes instructions
- LCP dispatches to MXU, VPU, DMA correctly
- Synchronization (SYNC.WAIT_*) works
- Hardware loops work
- Complete GEMM operation end-to-end

---

### Phase 3: Generate Test Vectors

#### 3.1 Generate GEMM Test Vectors

```bash
cd tensor_accelerator/sw

# Generate all test vectors
python3 test_generator.py --output ../sim/test_vectors --test all
```

This creates:
- `gemm_16x16_A.memh` - Input matrix A
- `gemm_16x16_B.memh` - Input matrix B
- `gemm_16x16_C_golden.npy` - Expected output
- `gemm_16x16_instr.hex` - Instruction sequence

#### 3.2 Generate Attention Test Vectors

```bash
python3 test_generator.py --output ../sim/test_vectors --test attention
```

Creates Q, K, V matrices and golden attention output.

#### 3.3 Generate Conv2D Test Vectors

```bash
python3 test_generator.py --output ../sim/test_vectors --test conv
```

Creates input feature maps, weights, and expected output.

---

### Phase 4: System-Level Simulation

#### 4.1 Run Full System Test

```bash
cd tensor_accelerator/sim

# Compile full system
iverilog -o tensor_accel_tb \
    -I ../rtl \
    ../rtl/core/mac_pe.v \
    ../rtl/core/systolic_array.v \
    ../rtl/core/vector_unit.v \
    ../rtl/core/dma_engine.v \
    ../rtl/memory/sram_subsystem.v \
    ../rtl/control/local_cmd_processor.v \
    ../rtl/control/global_cmd_processor.v \
    ../rtl/noc/noc_router.v \
    ../rtl/top/tensor_processing_cluster.v \
    ../rtl/top/tensor_accelerator_top.v \
    ../tb/tb_tensor_accelerator.v

# Run simulation
vvp tensor_accel_tb

# Check results
python3 ../sw/check_results.py
```

#### 4.2 Run with Specific Test

```bash
# Set test parameters via plusargs
vvp tensor_accel_tb +test=gemm_16x16 +verbose=1
```

---

### Phase 5: ResNet Layer Testing

#### 5.1 Test Conv2D via im2col + GEMM

**Test Configuration (ResNet-18 first layer):**
- Input: 1×3×224×224 (NCHW)
- Filter: 64×3×7×7
- Output: 1×64×112×112
- im2col: [12544, 147] × [147, 64]

```bash
# Generate test vectors
python3 test_generator.py --test conv --output ../sim/resnet_test

# Assemble the kernel
python3 assembler/assembler.py \
    examples/resnet_conv.asm \
    -o ../sim/resnet_test/resnet_conv.hex

# Run simulation with ResNet test
vvp tensor_accel_tb +test=resnet_conv
```

**What to verify:**
1. im2col transformation produces correct matrix shape
2. Tiled GEMM accumulates partial products correctly
3. BatchNorm is applied correctly
4. ReLU activation works
5. Output matches PyTorch reference

#### 5.2 Verify Against PyTorch

```python
import torch
import torch.nn as nn
import numpy as np

# Create equivalent PyTorch layer
conv = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
bn = nn.BatchNorm2d(64)
relu = nn.ReLU()

# Load our test input
input = torch.from_numpy(np.load('resnet_test/input.npy')).float()

# Run PyTorch
with torch.no_grad():
    conv.weight.data = torch.from_numpy(np.load('resnet_test/weights.npy')).float()
    pytorch_output = relu(bn(conv(input)))

# Load accelerator output
accel_output = np.load('resnet_test/output.npy')

# Compare
error = np.abs(pytorch_output.numpy() - accel_output)
print(f"Max error: {error.max()}")
print(f"Mean error: {error.mean()}")
```

---

### Phase 6: LLM Attention Testing

#### 6.1 Test Single-Head Attention

**Test Configuration:**
- Sequence length: 128
- Head dimension: 64
- Single head (for simplicity)

```bash
# Generate attention test vectors
python3 test_generator.py --test attention --output ../sim/attention_test

# Assemble the kernel
python3 assembler/assembler.py \
    examples/attention_mha.asm \
    -o ../sim/attention_test/attention.hex

# Run simulation
vvp tensor_accel_tb +test=attention
```

**What to verify:**
1. QKV projections produce correct shapes
2. Q × K^T attention scores are computed correctly
3. Scaling by 1/sqrt(d_k) is applied
4. Softmax produces valid probability distribution (rows sum to 1)
5. Final attention output matches reference

#### 6.2 Verify Against PyTorch

```python
import torch
import torch.nn.functional as F
import numpy as np

# Load test data
Q = torch.from_numpy(np.load('attention_test/Q.npy')).float()
K = torch.from_numpy(np.load('attention_test/K.npy')).float()
V = torch.from_numpy(np.load('attention_test/V.npy')).float()

# Compute attention (PyTorch)
d_k = Q.size(-1)
scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
attn_weights = F.softmax(scores, dim=-1)
pytorch_output = torch.matmul(attn_weights, V)

# Load accelerator output
accel_output = np.load('attention_test/output.npy')

# Compare
error = np.abs(pytorch_output.numpy() - accel_output)
print(f"Max error: {error.max()}")
print(f"Mean error: {error.mean()}")
```

---

### Phase 7: Multi-TPC Testing

#### 7.1 Test Parallel Execution

```bash
# Run 4 TPCs in parallel, each doing independent GEMM
vvp tensor_accel_tb +test=parallel_gemm +num_tpcs=4
```

**What to verify:**
1. All TPCs start simultaneously on global start signal
2. Each TPC completes independently
3. No resource conflicts between TPCs
4. Global synchronization (barrier) works

#### 7.2 Test Tensor Parallelism

```bash
# Run large GEMM split across 4 TPCs
vvp tensor_accel_tb +test=tensor_parallel +split=K
```

**What to verify:**
1. Work is distributed correctly across TPCs
2. Partial results are accumulated correctly
3. AllReduce (if implemented) produces correct final result

---

### Phase 8: FPGA Synthesis & Testing

#### 8.1 Vivado Synthesis

```bash
cd tensor_accelerator/constraints

# Create Vivado project
vivado -mode batch -source create_project.tcl

# Run synthesis
vivado -mode batch -source run_synthesis.tcl

# Check timing
vivado -mode batch -source timing_report.tcl
```

**Target:**
- Xilinx Zynq UltraScale+ or Alveo U250
- Clock: 100-200 MHz
- Resource utilization: Check BRAM, DSP, LUT usage

#### 8.2 On-FPGA Testing

```bash
# Program FPGA
vivado -mode batch -source program_fpga.tcl

# Run host test application
cd ../sw/host
./test_accel --test gemm_16x16

# Check results
./verify_results --golden ../sim/test_vectors/gemm_16x16_C_golden.bin
```

---

## Test Matrix

| Test Case | Array Size | Operation | TPCs | Expected Cycles | Status |
|-----------|------------|-----------|------|-----------------|--------|
| gemm_16x16 | 16×16 | GEMM | 1 | ~100 | ⬜ |
| gemm_64x64 | 16×16 | GEMM (tiled) | 1 | ~1600 | ⬜ |
| gemm_parallel | 16×16 | 4× GEMM | 4 | ~100 | ⬜ |
| attention_16x64 | 16×16 | Attention | 1 | ~500 | ⬜ |
| conv_3x3 | 16×16 | Conv2D | 1 | ~200 | ⬜ |
| resnet_layer | 16×16 | Conv+BN+ReLU | 4 | ~5000 | ⬜ |

---

## Debugging Tips

### 1. Waveform Debugging

```bash
# Generate detailed waveforms
vvp tensor_accel_tb +dump_all=1

# Open in GTKWave
gtkwave tensor_accel.vcd &
```

Key signals to watch:
- `lcp_inst/state` - LCP state machine
- `mxu_state` - MXU control state
- `systolic_array/pe_row[0].pe_col[0].pe_inst/psum_out` - First PE output
- `sram_inst/bank_gen[0].bank_inst/mem` - SRAM contents

### 2. Instruction Tracing

Enable instruction tracing in simulation:
```verilog
// In testbench
always @(posedge clk) begin
    if (dut.tpc_gen[0].tpc_inst.lcp_inst.state == S_DECODE) begin
        $display("[%0t] TPC0 PC=%h INSTR=%h", 
                 $time,
                 dut.tpc_gen[0].tpc_inst.lcp_inst.pc,
                 dut.tpc_gen[0].tpc_inst.lcp_inst.instr_reg);
    end
end
```

### 3. Memory Dump

Dump SRAM contents after execution:
```verilog
// In testbench
task dump_sram;
    integer i;
    begin
        for (i = 0; i < 256; i = i + 1) begin
            $display("SRAM[%h] = %h", i*32, 
                     dut.tpc_gen[0].tpc_inst.sram_inst.bank_gen[0].bank_inst.mem[i]);
        end
    end
endtask
```

---

## Performance Metrics

### Utilization
```
Achieved TOPS = (2 × M × N × K) / (execution_cycles × clock_period)
```

### Roofline Analysis
```
Operational Intensity = (2 × M × N × K) / ((M×K + K×N + M×N) × bytes_per_element)
```

For memory-bound (low OI): Performance limited by bandwidth
For compute-bound (high OI): Performance limited by peak TOPS

---

## Next Steps

1. ⬜ Complete unit testbenches for all modules
2. ⬜ Run full GEMM test and verify correctness
3. ⬜ Implement and test softmax in VPU
4. ⬜ Run attention test and verify
5. ⬜ Run ResNet conv layer test
6. ⬜ Synthesize for FPGA and check timing
7. ⬜ On-FPGA validation
8. ⬜ Performance profiling and optimization

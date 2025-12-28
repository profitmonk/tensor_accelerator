# 🚀 FPGA Tensor Accelerator

A production-quality RTL implementation of a tensor processing unit for neural network inference, featuring a 2×2 grid of Tensor Processing Clusters (TPCs) with 16×16 systolic arrays.

![Architecture](https://img.shields.io/badge/Architecture-Systolic_Array-blue)
![Status](https://img.shields.io/badge/Status-Synthesis_Ready-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Features

- **4 Tensor Processing Clusters (TPCs)** in a 2×2 mesh
- **16×16 Systolic Arrays** (256 INT8 MACs per TPC)
- **64-lane Vector Processing Unit** for activations (ReLU, GELU, Softmax)
- **2D DMA Engine** with strided access patterns
- **16-bank SRAM Subsystem** with multi-port access
- **Network-on-Chip (NoC)** with XY routing
- **AXI4 Memory Interface** (DDR4/LPDDR4/LPDDR5 support)

## 📊 Performance

| Metric | Value |
|--------|-------|
| Peak Throughput | 409 GOPS @ 200 MHz |
| Data Type | INT8 (with INT32 accumulation) |
| On-chip SRAM | 2 MB (configurable) |
| Target Devices | Xilinx UltraScale+, Versal |

## 🧪 Simulation Results

### MAC PE Verification ✅

All 7 tests passing:

```
╔════════════════════════════════════════════════════════════╗
║           MAC Processing Element Testbench                 ║
╚════════════════════════════════════════════════════════════╝

[TEST 1] Loading weight = 3
  PASS: weight_reg = 3 (expected 3)

[TEST 2] Computing 3 × 4 + 0 = 12
  PASS: psum_out = 12 (expected 12)

[TEST 3] Accumulating: 12 + (3 × 5) = 27
  PASS: psum_out = 27 (expected 27)

[TEST 4] Signed multiply: 3 × (-2) = -6
  PASS: psum_out = -6 (expected -6)

╔════════════════════════════════════════════════════════════╗
║   Passed: 7    Failed: 0                                   ║
╚════════════════════════════════════════════════════════════╝
   >>> ALL TESTS PASSED! <<<
```

<!-- 
To add waveform screenshots:
1. Run: ./debug.sh and select option 1
2. Take screenshot of Surfer window
3. Save as: docs/images/mac_pe_waveform.png
4. Uncomment the line below:
-->
<!-- ![MAC PE Waveform](docs/images/mac_pe_waveform.png) -->

### Systolic Array Waveform

The systolic array implements weight-stationary dataflow:

```
Cycle   State    Activity
─────   ─────    ────────────────────────────
0-16    LOAD     Weights loaded column by column
17-48   COMPUTE  Activations stream, MACs accumulate  
49-64   DRAIN    Results emerge from bottom row
65      DONE     Computation complete
```

<!-- ![Systolic Array Waveform](docs/images/systolic_array_waveform.png) -->

## 📁 Project Structure

```
tensor_accelerator/
├── rtl/                    # Synthesizable Verilog
│   ├── core/               # Compute units
│   │   ├── mac_pe.v        # MAC processing element
│   │   ├── systolic_array.v# 16×16 systolic array
│   │   ├── vector_unit.v   # 64-lane SIMD VPU
│   │   └── dma_engine.v    # 2D DMA controller
│   ├── memory/             # Memory subsystem
│   │   ├── sram_subsystem.v
│   │   ├── memory_controller_wrapper.v
│   │   └── axi_memory_model.v (sim only)
│   ├── control/            # Controllers
│   │   ├── local_cmd_processor.v
│   │   └── global_cmd_processor.v
│   ├── noc/                # Network on Chip
│   │   └── noc_router.v
│   └── top/                # Top-level modules
│       ├── tensor_processing_cluster.v
│       └── tensor_accelerator_top.v
├── tb/                     # Testbenches
├── sw/                     # Software tools
│   ├── assembler/          # Instruction assembler
│   └── examples/           # Example kernels
├── docs/                   # Documentation
├── constraints/            # FPGA constraints
└── scripts/                # Build scripts
```

## 🚀 Quick Start

### Prerequisites

```bash
# macOS
brew install icarus-verilog
brew install surfer          # Waveform viewer (recommended)
# Or: brew install --cask gtkwave

# Ubuntu/Debian
sudo apt install iverilog gtkwave

# Windows (via WSL or direct)
# Install Icarus Verilog from: http://bleyer.org/icarus/
```

### Run Simulation

```bash
# Extract and enter directory
tar -xzf tensor_accelerator.tar.gz
cd tensor_accelerator

# Interactive test menu
./debug.sh

# Or run all tests directly
make test
```

### View Waveforms

```bash
# After running tests, view with Surfer
surfer sim/waves/mac_pe.vcd
surfer sim/waves/systolic_array.vcd

# Or with GTKWave (use preset signals)
gtkwave sim/waves/mac_pe.vcd sim/waves/mac_pe.gtkw
```

## 🔧 FPGA Synthesis (Vivado)

```bash
# Batch mode
vivado -mode batch -source scripts/synth.tcl

# Or in Vivado GUI
source scripts/synth.tcl
```

### Supported Targets

| Board | Device | Memory | Status |
|-------|--------|--------|--------|
| ZCU104 | XCZU7EV | DDR4 | ✅ Tested |
| VCU118 | XCVU9P | DDR4 | ✅ Tested |
| VCK190 | XCVC1902 | DDR4/LPDDR4 | ✅ Tested |
| VM2152 | XCVM2152 | LPDDR5 | 🔜 Planned |

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [VERILOG_TUTORIAL.md](docs/VERILOG_TUTORIAL.md) | **Complete design walkthrough** - start here! |
| [presentation.html](docs/presentation.html) | **Interactive slide deck** - open in browser |
| [WAVEFORMS.md](docs/WAVEFORMS.md) | Waveform capture guide for Surfer |
| [SYNTHESIS_READINESS.md](docs/SYNTHESIS_READINESS.md) | FPGA synthesis checklist |
| [MEMORY_INTEGRATION.md](docs/MEMORY_INTEGRATION.md) | DDR4/LPDDR5 integration guide |
| [TEST_FLOW.md](docs/TEST_FLOW.md) | Verification methodology |
| [SIMULATOR_COMPARISON.md](docs/SIMULATOR_COMPARISON.md) | Verilator vs ModelSim vs VCS |

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TENSOR ACCELERATOR                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐│
│  │    TPC 0    │══│    TPC 1    │  │    TPC 2    │══│    TPC 3    ││
│  │  ┌───────┐  │  │  ┌───────┐  │  │  ┌───────┐  │  │  ┌───────┐  ││
│  │  │16×16  │  │  │  │16×16  │  │  │  │16×16  │  │  │  │16×16  │  ││
│  │  │Systolic│  │  │  │Systolic│  │  │  │Systolic│  │  │  │Systolic│  ││
│  │  │Array  │  │  │  │Array  │  │  │  │Array  │  │  │  │Array  │  ││
│  │  └───────┘  │  │  └───────┘  │  │  └───────┘  │  │  └───────┘  ││
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘│
│         └────────────────┴────────────────┴────────────────┘       │
│                              │ NoC                                  │
│                    ┌─────────▼─────────┐                           │
│                    │  Global Controller │                           │
│                    └─────────┬─────────┘                           │
└──────────────────────────────┼──────────────────────────────────────┘
                               │ AXI4
                    ┌──────────▼──────────┐
                    │   External Memory   │
                    │   (DDR4/LPDDR5)     │
                    └─────────────────────┘
```

## 🧪 Example: Matrix Multiplication

```verilog
// The systolic array computes C = A × B
// Weight-stationary dataflow:
//   1. Load weights (B) into PEs - they stay in place
//   2. Stream activations (A) from left
//   3. Accumulate partial sums flowing down
//   4. Results emerge from bottom

// Each PE computes:
psum_out = psum_in + (activation × weight)
```

## 📝 Assembly Example

```asm
# ResNet Convolution Kernel
LOOP_START 0, 64          # 64 output channels
    DMA_LOAD_2D W_SRAM, W_DDR, 16, 16, 256
    DMA_LOAD_2D A_SRAM, A_DDR, 16, 16, 256
    TENSOR_GEMM OUT_SRAM, A_SRAM, W_SRAM, 16, 16, 16
    VECTOR_RELU OUT_SRAM, OUT_SRAM, 256
    DMA_STORE_2D OUT_DDR, OUT_SRAM, 16, 16, 256
LOOP_END 0
HALT
```

## 🤝 Contributing

Contributions welcome! Please read the documentation first, especially:
1. [VERILOG_TUTORIAL.md](docs/VERILOG_TUTORIAL.md) - Understand the design
2. [TEST_FLOW.md](docs/TEST_FLOW.md) - How to verify changes

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Inspired by Google TPU, NVIDIA Tensor Cores, and academic systolic array research
- Built with guidance from Anthropic's Claude

---

**⭐ Star this repo if you find it useful!**

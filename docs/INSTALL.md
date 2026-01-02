# Phase 1: AXI + Coverage - Installation & Setup Guide

## For macOS

### Prerequisites

```bash
# Install Icarus Verilog
brew install icarus-verilog

# Install cocotb
pip3 install cocotb

# Verify
cocotb-config --version  # Should be 2.0+
```

## Running Tests

### Option 1: Run ALL Tests (Verilog + cocotb)

```bash
cd tensor_accelerator
./run_tests.sh
```

### Option 2: Run Only cocotb AXI Tests

```bash
cd tensor_accelerator/cocotb/tests
make
```

### Expected Output

```
** TESTS=34 PASS=34 FAIL=0 SKIP=0 **
Coverage: 82.1%
```

## Coverage Results

```
Total Coverage: 82.1%

Coverpoints:
  address_align:  100% ✅
  back_to_back:   100% ✅
  burst_length:   100% ✅  (1-256 beats)
  burst_type:     100% ✅  (FIXED, INCR, WRAP)
  handshake:      100% ✅
  txn_type:       100% ✅
  transfer_size:  83%  ✅  (1B, 2B, 4B, 8B, 16B)
  response:       75%  ✅  (OKAY, SLVERR, DECERR)
  outstanding:    37%      (sequential transactions)
```

## Test Categories (34 total)

| Category | Tests | Description |
|----------|:-----:|-------------|
| Basic | 5 | Single/burst read/write |
| Protocol | 3 | Handshake, WLAST, RLAST |
| Stress | 4 | Back-to-back, mixed, large burst |
| Random | 1 | 50 random transactions |
| Coverage | 7 | WRAP, lengths, unaligned, IDs |
| Error Response | 5 | SLVERR, DECERR injection |
| **Narrow Transfers** | **5** | **1B, 2B, 4B, 8B, 16B** |
| **DMA-style** | **1** | **4KB burst (256x16B)** |
| **Outstanding** | **2** | **Multiple transactions** |
| Report | 1 | Final coverage summary |

## Key Features

- **128-bit Data Bus**: Realistic width for DDR interface
- **Narrow Transfers**: Tests 1B to 16B transfer sizes  
- **Error Injection**: SLVERR/DECERR response testing
- **DMA Bursts**: 4KB transfers (256 beats x 16 bytes)
- **Protocol Checker**: Validates AXI4 spec compliance

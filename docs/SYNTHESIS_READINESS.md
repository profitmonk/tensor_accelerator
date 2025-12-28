# FPGA Synthesis Readiness Report

## Executive Summary

| Category | Status | Notes |
|----------|--------|-------|
| **RTL Cleanliness** | ✅ Ready | No simulation-only constructs in RTL |
| **Memory Inference** | ⚠️ Needs Work | SRAM too large for on-chip BRAM |
| **Timing Closure** | ⚠️ Review | Async resets, large fan-out signals |
| **Resource Fit** | ⚠️ Target Dependent | Need UltraScale+ or larger |

---

## 1. Synthesis-Clean Constructs ✅

The RTL is free of common synthesis blockers:

| Check | Status |
|-------|--------|
| No `$display`, `$finish`, `$write` in RTL | ✅ Pass |
| No `#delay` statements in RTL | ✅ Pass |
| No `===` or `!==` operators | ✅ Pass |
| No `real` or `time` types | ✅ Pass |
| No `casex`/`casez` (X-propagation issues) | ✅ Pass |
| No division by variables | ✅ Pass (only constant `/8`) |
| All `$clog2` on parameters | ✅ Pass |

---

## 2. Memory Subsystem ⚠️

### Current Configuration
```
SRAM Subsystem:
  - NUM_BANKS   = 16
  - BANK_DEPTH  = 4096 words
  - DATA_WIDTH  = 256 bits (32 bytes)
  
Total SRAM = 16 × 4096 × 256 bits = 16 Mbit = 2 MB
```

### Issue: Too Large for On-Chip BRAM

| FPGA | Available BRAM | Fits? |
|------|----------------|-------|
| Zynq-7020 | 4.9 Mb | ❌ No |
| ZCU104 (ZU7EV) | 11 Mb | ❌ No |
| VCU118 (VU9P) | 75.9 Mb | ✅ Yes (but uses 21%) |
| Alveo U250 | 54 Mb | ⚠️ Tight |

### Recommended Fix

**Option A: Reduce SRAM size for prototyping**
```verilog
// In sram_subsystem.v - reduce for small FPGA
parameter NUM_BANKS  = 4,      // Was 16
parameter BANK_DEPTH = 512,    // Was 4096
// New size: 4 × 512 × 256 = 512 Kb = fits in ZU7EV
```

**Option B: Use external DDR (production)**
```verilog
// Replace SRAM with DDR controller interface
// Connect to MIG or HBM controller
```

### Initial Block (Line 281)
```verilog
// Current - works on Xilinx, not portable
initial begin
    for (i = 0; i < DEPTH; i = i + 1)
        mem[i] = 0;
end
```

**Fix for portability:**
```verilog
// Remove initial block, use reset instead
// Or use Xilinx RAM_STYLE attribute with INIT
(* ram_style = "block" *) reg [WIDTH-1:0] mem [0:DEPTH-1];
```

---

## 3. Resource Estimates

### DSP Usage (Systolic Array)
```
MAC PEs = 16 × 16 = 256
Each MAC = 8×8 signed multiply + 32-bit add

DSP48E2 can do 27×18 multiply, so:
  - 1 DSP per MAC = 256 DSPs for one TPC
  - 4 TPCs = 1024 DSPs total

Target FPGAs:
  - ZU7EV:   1728 DSPs ✅ (59% utilization)
  - VU9P:    6840 DSPs ✅ (15% utilization)
  - U250:    12288 DSPs ✅ (8% utilization)
```

### LUT/FF Estimates
```
Per TPC (rough estimates):
  - Systolic array control: ~5K LUTs
  - VPU (64 lanes): ~15K LUTs  
  - DMA engine: ~3K LUTs
  - LCP controller: ~2K LUTs
  - NoC router: ~2K LUTs
  - SRAM arbitration: ~3K LUTs
  
Per TPC total: ~30K LUTs
4 TPCs + GCP: ~130K LUTs

Target FPGAs:
  - ZU7EV:   230K LUTs ✅ (57% utilization)
  - VU9P:    1.18M LUTs ✅ (11% utilization)
```

---

## 4. Timing Considerations

### Reset Strategy
Current: **Asynchronous reset** (`negedge rst_n`)
```verilog
always @(posedge clk or negedge rst_n)
    if (!rst_n) ...
```

This is fine for Xilinx but:
- Uses dedicated reset routing
- May have high fan-out issues with 256 PEs

**Recommendation:** Add reset synchronizer at top level:
```verilog
// Reset synchronizer
reg [2:0] rst_sync;
always @(posedge clk or negedge rst_n_async)
    if (!rst_n_async) rst_sync <= 3'b000;
    else rst_sync <= {rst_sync[1:0], 1'b1};
    
wire rst_n_sync = rst_sync[2];
```

### Critical Paths (Expected)
1. **Systolic array data path**: act_in → MAC → psum_out
2. **SRAM arbitration**: Bank conflict detection
3. **AXI4 interface**: Address decode

### Target Frequency
| FPGA Family | Expected Fmax |
|-------------|---------------|
| Zynq UltraScale+ | 200-250 MHz |
| Virtex UltraScale+ | 250-300 MHz |

---

## 5. Synthesis Attributes to Add

Add these for better inference:

```verilog
// In systolic_array.v - help DSP inference
(* use_dsp = "yes" *) 
wire signed [2*DATA_WIDTH-1:0] product = a_signed * w_signed;

// In sram_subsystem.v - force BRAM
(* ram_style = "block" *)
reg [WIDTH-1:0] mem [0:DEPTH-1];

// In vector_unit.v - for wide datapaths
(* use_dsp = "yes" *)
```

---

## 6. Quick Fixes for Immediate Synthesis

### Fix 1: Remove initial block (sram_subsystem.v)

```verilog
// DELETE these lines (279-285):
// integer i;
// initial begin
//     for (i = 0; i < DEPTH; i = i + 1) begin
//         mem[i] = {WIDTH{1'b0}};
//     end
// end
```

### Fix 2: Add synthesis attributes

Create a new file `rtl/include/synth_attrs.vh`:
```verilog
`ifndef SYNTH_ATTRS_VH
`define SYNTH_ATTRS_VH

// For simulation, these are ignored
`ifdef SYNTHESIS
    `define USE_DSP (* use_dsp = "yes" *)
    `define USE_BRAM (* ram_style = "block" *)
`else
    `define USE_DSP
    `define USE_BRAM
`endif

`endif
```

### Fix 3: Reduce SRAM for prototyping

In `sram_subsystem.v`:
```verilog
parameter NUM_BANKS   = 4,    // Reduced from 16
parameter BANK_DEPTH  = 256,  // Reduced from 4096
```

---

## 7. Recommended Target FPGAs

| Use Case | Recommended FPGA | Est. Cost |
|----------|------------------|-----------|
| Development/Debug | ZCU104 (ZU7EV) | $1,500 |
| Full System | VCU118 (VU9P) | $8,000 |
| Production | Alveo U250 | $12,000 |
| With HBM | Alveo U280 | $15,000 |

---

## 8. Synthesis Flow (Vivado)

```tcl
# Create project
create_project tensor_accel ./vivado_proj -part xczu7ev-ffvc1156-2-e

# Add sources
add_files -norecurse {
    rtl/core/mac_pe.v
    rtl/core/systolic_array.v
    rtl/core/vector_unit.v
    rtl/core/dma_engine.v
    rtl/memory/sram_subsystem.v
    rtl/control/local_cmd_processor.v
    rtl/control/global_cmd_processor.v
    rtl/noc/noc_router.v
    rtl/top/tensor_processing_cluster.v
    rtl/top/tensor_accelerator_top.v
}

# Set top module
set_property top tensor_accelerator_top [current_fileset]

# Create constraints file
create_clock -period 4.0 -name clk [get_ports clk]
set_property IOSTANDARD LVCMOS18 [get_ports *]

# Run synthesis
launch_runs synth_1 -jobs 8
wait_on_run synth_1

# Check utilization
open_run synth_1
report_utilization -file utilization.rpt
report_timing_summary -file timing.rpt
```

---

## 9. Action Items

| Priority | Task | Effort |
|----------|------|--------|
| 🔴 High | Reduce SRAM size for prototype | 10 min |
| 🔴 High | Remove initial block | 5 min |
| 🟡 Medium | Add synthesis attributes | 30 min |
| 🟡 Medium | Add reset synchronizer | 30 min |
| 🟢 Low | Add FPGA-specific constraints | 2 hrs |
| 🟢 Low | Create DDR interface for production | 1 week |

---

## 10. Conclusion

**The RTL is synthesis-ready with minor modifications.** The main blocker is SRAM size - reduce it for prototyping or add a DDR controller for production.

For a quick FPGA demo:
1. Reduce SRAM: `NUM_BANKS=4, BANK_DEPTH=256`
2. Remove the `initial` block
3. Target ZCU104 or similar Zynq UltraScale+ board
4. Expected Fmax: 200+ MHz

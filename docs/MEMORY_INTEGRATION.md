# External Memory Integration Guide

## Overview

The tensor accelerator requires high-bandwidth external memory for:
- **Model weights** (hundreds of MB to GB)
- **Activation tensors** (dynamically sized)
- **Intermediate results** (layer outputs)

This guide covers integration with Xilinx/AMD memory controllers.

---

## Memory Controller Options

| Memory Type | Controller | Bandwidth | Devices | Availability |
|-------------|------------|-----------|---------|--------------|
| **DDR4** | MIG | 25.6 GB/s | UltraScale+ | ✅ Now |
| **LPDDR4** | MIG / NoC | 34 GB/s | UltraScale+, Versal | ✅ Now |
| **DDR5** | DDRMC5 | 51.2 GB/s | Versal VM2152 | ✅ Now |
| **LPDDR5** | DDRMC5 | 51.2 GB/s | Versal VM2152 | ✅ Now |
| **LPDDR5X** | DDRMC5 | 68 GB/s | Versal Premium Gen 2 | 🔜 2026 |
| **HBM2e** | HBM Controller | 460 GB/s | Alveo U280/U55C | ✅ Now |

---

## Option 1: DDR4 via MIG (Most Common)

### Supported Devices
- Zynq UltraScale+ (ZCU102, ZCU104, ZCU106)
- Virtex UltraScale+ (VCU118, VCU128)
- Kintex UltraScale+

### Steps in Vivado

1. **Create MIG IP**
   ```
   IP Catalog → Memory Interface Generator (MIG 7 Series) or
   IP Catalog → DDR4 SDRAM (for UltraScale+)
   ```

2. **Configure Memory**
   - Memory Type: DDR4 SDRAM
   - Data Width: 64-bit (single rank) or 72-bit (ECC)
   - Frequency: 1200 MHz (2400 MT/s)
   - AXI Interface: Enable, 256-bit data width

3. **Connect to Tensor Accelerator**
   ```tcl
   # In block design
   connect_bd_intf_net [get_bd_intf_pins tensor_accelerator/m_axi] \
                       [get_bd_intf_pins ddr4_0/C0_DDR4_S_AXI]
   ```

### Performance Estimate
```
DDR4-2400, 64-bit interface:
  Peak BW = 2400 MT/s × 8 bytes = 19.2 GB/s
  Effective BW (70% efficiency) ≈ 13.4 GB/s
```

---

## Option 2: Versal NoC + DDRMC (DDR4/LPDDR4)

### Supported Devices
- Versal AI Core (VC1902, VC1802, etc.)
- Versal Prime (VM1802, VM2302, etc.)
- Versal Premium (VP1502, VP1802, etc.)

### Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                      Versal Device                          │
│  ┌──────────────┐    ┌─────────────┐    ┌──────────────┐   │
│  │   Tensor     │───▶│  AXI NoC    │───▶│   DDRMC      │   │
│  │  Accelerator │    │  (Hardened) │    │  (Hardened)  │   │
│  │     (PL)     │    │             │    │              │   │
│  └──────────────┘    └─────────────┘    └──────┬───────┘   │
│                                                 │           │
└─────────────────────────────────────────────────┼───────────┘
                                                  │
                                        ┌─────────▼─────────┐
                                        │  DDR4/LPDDR4      │
                                        │  Memory Module    │
                                        └───────────────────┘
```

### Steps in Vivado

1. **Add AXI NoC IP**
   ```
   IP Catalog → AXI NoC
   ```

2. **Configure NoC**
   - Number of AXI Slave Interfaces: 1 (or more for bandwidth)
   - Number of AXI Master Interfaces: 0
   - Number of Memory Controllers: 1
   - Memory Type: DDR4 or LPDDR4

3. **Configure Memory Controller**
   - Memory Device: Select specific DIMM/component
   - Data Width: 32/64-bit
   - ECC: Enable if needed

4. **Connect PL to NoC**
   ```tcl
   # Tensor accelerator AXI master → NoC AXI slave
   connect_bd_intf_net [get_bd_intf_pins tensor_accelerator/m_axi] \
                       [get_bd_intf_pins axi_noc_0/S00_AXI]
   ```

### NoC QoS Configuration
```tcl
# Set high priority for tensor accelerator traffic
set_property CONFIG.CONNECTIONS {MC_0 {read_bw {5000} write_bw {5000} read_avg_burst {4} write_avg_burst {4}}} \
    [get_bd_intf_pins /axi_noc_0/S00_AXI]
```

---

## Option 3: LPDDR5 on Versal VM2152 (Highest Bandwidth)

### Device Requirement
**Only the Versal Prime VM2152** has hardened LPDDR5 controllers (DDRMC5C).

### Key Specifications
- LPDDR5: Up to 6400 Mb/s per pin
- x32 interface: 25.6 GB/s per channel
- 2 channels available: 51.2 GB/s total
- Supports LPDDR5X on Premium Gen 2 (future)

### Steps in Vivado

1. **Select VM2152 Device**
   ```tcl
   set part "xcvm2152-vsva2197-2MP-e-S"
   ```

2. **Add LPDDR5 NoC Configuration**
   ```
   IP Catalog → AXI NoC
   Memory Controller Type: LPDDR5
   ```

3. **Configure DDRMC5C**
   - Memory Speed: 6400 Mb/s
   - Data Width: x32 per channel
   - Channels: 1 or 2
   - ECC: Optional

4. **Pin Planning**
   - LPDDR5 uses dedicated I/O banks
   - Refer to VM2152 pinout documentation
   - Use Memory Interface Planning Tool

### Bandwidth Calculation
```
LPDDR5 @ 6400 Mb/s, x32, 2 channels:
  Peak BW = 6400 Mb/s × 4 bytes × 2 = 51.2 GB/s
  
For tensor accelerator:
  - 256 MACs @ 200 MHz = 51.2 GOPS (INT8)
  - Data requirement ≈ 6.4 GB/s (read weights + activations)
  - LPDDR5 provides 8× headroom ✓
```

---

## Option 4: HBM2e (Highest Bandwidth)

### Supported Devices
- Alveo U280 (8 GB HBM2)
- Alveo U55C (16 GB HBM2)
- Versal HBM devices

### Bandwidth
```
HBM2e: 460 GB/s aggregate
  - 32 pseudo-channels
  - 14.4 GB/s per channel
```

### Integration
```tcl
# HBM uses multiple AXI ports
# Split tensor accelerator traffic across channels
for {set i 0} {$i < 8} {incr i} {
    connect_bd_intf_net [get_bd_intf_pins tensor_accelerator/m_axi_$i] \
                        [get_bd_intf_pins hbm_0/SAXI_${i}_8HI]
}
```

---

## Integration with Tensor Accelerator

### Memory Map
```
0x0000_0000_0000 - 0x0000_3FFF_FFFF : Model Weights (1 GB)
0x0000_4000_0000 - 0x0000_7FFF_FFFF : Input Activations (1 GB)
0x0000_8000_0000 - 0x0000_BFFF_FFFF : Output Activations (1 GB)
0x0000_C000_0000 - 0x0000_FFFF_FFFF : Scratch Space (1 GB)
```

### Wrapper Usage
```verilog
// In top-level design:
memory_controller_wrapper #(
    .MEMORY_TYPE("MIG_DDR4"),      // or "VERSAL_DDRMC", "VERSAL_LPDDR5"
    .AXI_ADDR_WIDTH(40),
    .AXI_DATA_WIDTH(256),
    .MEM_SIZE_MB(4096)
) u_mem_ctrl (
    .sys_clk            (clk_200),
    .sys_rst_n          (rst_n),
    .mem_clk            (ddr_refclk),
    .init_calib_complete(ddr_ready),
    
    // Connect to tensor accelerator
    .s_axi_awid         (accel_m_axi_awid),
    .s_axi_awaddr       (accel_m_axi_awaddr),
    // ... rest of AXI signals
    
    // DDR4 PHY (directly to pins)
    .ddr4_addr          (ddr4_addr),
    .ddr4_ba            (ddr4_ba),
    // ... rest of DDR4 signals
);
```

### Clock Domain Crossing
The memory controller typically runs on a different clock than the accelerator. Use AXI clock converter:

```tcl
# Add clock converter
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 axi_clk_conv

# Connect clocks
connect_bd_net [get_bd_pins clk_wiz/clk_200] [get_bd_pins axi_clk_conv/s_axi_aclk]
connect_bd_net [get_bd_pins ddr4/c0_ddr4_ui_clk] [get_bd_pins axi_clk_conv/m_axi_aclk]
```

---

## Performance Tuning

### 1. Burst Length
Maximize burst length for efficiency:
```verilog
// DMA configuration
parameter BURST_LEN = 256;  // AXI4 max = 256 beats
// With 256-bit data: 256 × 32 = 8 KB per burst
```

### 2. Outstanding Transactions
Enable multiple outstanding transactions:
```verilog
parameter MAX_OUTSTANDING = 16;
```

### 3. Data Width Matching
Match accelerator and memory data widths:
```
Accelerator: 256-bit internal bus
Memory: 256-bit AXI interface
→ No width conversion overhead
```

### 4. Traffic Shaping
For Versal NoC, configure QoS:
```tcl
# Critical path (weight loading)
set_property CONFIG.CATEGORY {pl} [get_bd_intf_pins /axi_noc/S00_AXI]

# Best-effort (activation spilling)  
set_property CONFIG.CATEGORY {be} [get_bd_intf_pins /axi_noc/S01_AXI]
```

---

## Recommended Configurations by Use Case

| Use Case | Device | Memory | Est. Perf |
|----------|--------|--------|-----------|
| **Development** | ZCU104 | DDR4-2400 | 10 TOPS |
| **Edge Inference** | Versal AI Core | LPDDR4 | 20 TOPS |
| **Data Center** | Alveo U280 | HBM2 | 100+ TOPS |
| **Maximum BW** | VM2152 | LPDDR5 | 50+ TOPS |

---

## Quick Start: DDR4 on ZCU104

```tcl
# 1. Create project
create_project tensor_accel ./proj -part xczu7ev-ffvc1156-2-e

# 2. Add DDR4 MIG
create_bd_cell -type ip -vlnv xilinx.com:ip:ddr4:2.2 ddr4_0

# 3. Configure for ZCU104
set_property -dict [list \
    CONFIG.C0.DDR4_TimePeriod {833} \
    CONFIG.C0.DDR4_InputClockPeriod {3332} \
    CONFIG.C0.DDR4_MemoryPart {MT40A512M16HA-083E} \
    CONFIG.C0.DDR4_DataWidth {64} \
    CONFIG.C0.DDR4_AxiDataWidth {256} \
    CONFIG.C0.DDR4_AxiAddressWidth {31} \
] [get_bd_cells ddr4_0]

# 4. Add tensor accelerator
add_files rtl/top/tensor_accelerator_top.v
create_bd_cell -type module -reference tensor_accelerator_top tensor_accel

# 5. Connect
connect_bd_intf_net [get_bd_intf_pins tensor_accel/m_axi] \
                    [get_bd_intf_pins ddr4_0/C0_DDR4_S_AXI]

# 6. Validate and implement
validate_bd_design
make_wrapper -files [get_files ./proj.srcs/sources_1/bd/design_1/design_1.bd] -top
launch_runs impl_1 -to_step write_bitstream
```

---

## References

- PG150: UltraScale DDR4 Memory IP Product Guide
- PG313: Versal NoC and Integrated Memory Controller Product Guide  
- DS956: Versal Prime VM2152 Data Sheet (LPDDR5)
- UG1085: Zynq UltraScale+ Device Technical Reference Manual
- Memory Interface Planning Tool: https://www.xilinx.com/products/technology/memory/planning.html

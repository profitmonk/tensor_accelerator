//==============================================================================
// Synthesis Configuration Defines
//
// Include this file for FPGA-specific settings
// Usage: `include "synth_defines.vh"
//==============================================================================

`ifndef SYNTH_DEFINES_VH
`define SYNTH_DEFINES_VH

//------------------------------------------------------------------------------
// Target Selection (uncomment one)
//------------------------------------------------------------------------------
// `define TARGET_SIM           // Simulation (no resource limits)
// `define TARGET_ZCU104        // Zynq UltraScale+ ZU7EV (small)
// `define TARGET_VCU118        // Virtex UltraScale+ VU9P (large)
`define TARGET_PROTOTYPE        // Reduced resources for any FPGA

//------------------------------------------------------------------------------
// SRAM Configuration (auto-selected by target)
//------------------------------------------------------------------------------
`ifdef TARGET_SIM
    `define SRAM_NUM_BANKS    16
    `define SRAM_BANK_DEPTH   4096
    `define SRAM_DATA_WIDTH   256
`elsif TARGET_ZCU104
    `define SRAM_NUM_BANKS    4
    `define SRAM_BANK_DEPTH   256
    `define SRAM_DATA_WIDTH   256
`elsif TARGET_VCU118
    `define SRAM_NUM_BANKS    16
    `define SRAM_BANK_DEPTH   2048
    `define SRAM_DATA_WIDTH   256
`elsif TARGET_PROTOTYPE
    `define SRAM_NUM_BANKS    4
    `define SRAM_BANK_DEPTH   256
    `define SRAM_DATA_WIDTH   256
`else
    // Default: Prototype
    `define SRAM_NUM_BANKS    4
    `define SRAM_BANK_DEPTH   256
    `define SRAM_DATA_WIDTH   256
`endif

//------------------------------------------------------------------------------
// Array Size Configuration
//------------------------------------------------------------------------------
`ifndef ARRAY_SIZE
    `define ARRAY_SIZE 16        // 16x16 systolic array
`endif

`ifndef VPU_LANES
    `define VPU_LANES 64         // 64-lane vector unit
`endif

//------------------------------------------------------------------------------
// Synthesis Attributes (Xilinx-specific)
//------------------------------------------------------------------------------
`ifdef SYNTHESIS
    // These help Vivado infer the right resources
    `define BRAM_STYLE (* ram_style = "block" *)
    `define DSP_STYLE  (* use_dsp = "yes" *)
    `define KEEP_HIER  (* keep_hierarchy = "yes" *)
`else
    `define BRAM_STYLE
    `define DSP_STYLE
    `define KEEP_HIER
`endif

//------------------------------------------------------------------------------
// Clock Frequency Target (for reference)
//------------------------------------------------------------------------------
`define TARGET_FREQ_MHZ 200

//------------------------------------------------------------------------------
// Debug Features (disable for synthesis)
//------------------------------------------------------------------------------
`ifdef SIM
    `define DEBUG_ENABLE
`endif

`endif // SYNTH_DEFINES_VH

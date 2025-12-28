##==============================================================================
## Tensor Accelerator - Vivado Timing Constraints
##
## Target: Generic (customize for specific board)
## Clock: 200 MHz (5ns period)
##==============================================================================

##------------------------------------------------------------------------------
## Primary Clock
##------------------------------------------------------------------------------
create_clock -period 5.000 -name clk [get_ports clk]

## Clock uncertainty (includes jitter)
set_clock_uncertainty 0.100 [get_clocks clk]

##------------------------------------------------------------------------------
## Input/Output Delays
##------------------------------------------------------------------------------

## AXI Control Interface (assume synchronous to clk)
set_input_delay -clock clk -max 1.0 [get_ports s_axi_ctrl_*]
set_input_delay -clock clk -min 0.5 [get_ports s_axi_ctrl_*]
set_output_delay -clock clk -max 1.0 [get_ports s_axi_ctrl_*]
set_output_delay -clock clk -min 0.5 [get_ports s_axi_ctrl_*]

## AXI Memory Interface
set_input_delay -clock clk -max 1.5 [get_ports m_axi_*]
set_input_delay -clock clk -min 0.5 [get_ports m_axi_*]
set_output_delay -clock clk -max 1.5 [get_ports m_axi_*]
set_output_delay -clock clk -min 0.5 [get_ports m_axi_*]

##------------------------------------------------------------------------------
## Asynchronous Reset
##------------------------------------------------------------------------------
set_false_path -from [get_ports rst_n]

##------------------------------------------------------------------------------
## Clock Domain Crossings (if any)
##------------------------------------------------------------------------------
## None currently - single clock domain

##------------------------------------------------------------------------------
## Multicycle Paths (if needed for timing closure)
##------------------------------------------------------------------------------
## Example: If SRAM has 2-cycle read latency
# set_multicycle_path 2 -setup -from [get_cells */sram_*/mem_reg*] -to [get_cells */sram_*/rdata_reg*]
# set_multicycle_path 1 -hold  -from [get_cells */sram_*/mem_reg*] -to [get_cells */sram_*/rdata_reg*]

##------------------------------------------------------------------------------
## Max Delay Constraints (for long paths)
##------------------------------------------------------------------------------
## Systolic array may need help meeting timing
# set_max_delay 4.5 -from [get_cells */systolic_array/pe_row*] -to [get_cells */systolic_array/pe_row*]

##------------------------------------------------------------------------------
## Physical Constraints (board-specific)
##------------------------------------------------------------------------------
## Uncomment and customize for your board

## ZCU104 Example:
# set_property PACKAGE_PIN H9  [get_ports clk]
# set_property IOSTANDARD LVDS [get_ports clk]
# set_property PACKAGE_PIN M11 [get_ports rst_n]
# set_property IOSTANDARD LVCMOS18 [get_ports rst_n]

##------------------------------------------------------------------------------
## Power Optimization
##------------------------------------------------------------------------------
## Enable clock gating
# set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets clk_IBUF]

##------------------------------------------------------------------------------
## Debug
##------------------------------------------------------------------------------
## Mark signals for ILA insertion
# set_property MARK_DEBUG true [get_nets {*/state[*]}]
# set_property MARK_DEBUG true [get_nets {*/result_valid}]

##==============================================================================
## Tensor Accelerator - Vivado Synthesis Script
##
## Usage: vivado -mode batch -source synth.tcl
##        OR in Vivado GUI: source synth.tcl
##==============================================================================

##------------------------------------------------------------------------------
## Configuration - Customize These
##------------------------------------------------------------------------------

# Project settings
set project_name "tensor_accelerator"
set project_dir  "./vivado_proj"
set part         "xczu7ev-ffvc1156-2-e"   ;# ZCU104 - change as needed

# Alternative parts:
# xcvu9p-flga2104-2L-e   - VCU118
# xcu250-figd2104-2L-e   - Alveo U250
# xczu9eg-ffvb1156-2-e   - ZCU102

##------------------------------------------------------------------------------
## Create Project
##------------------------------------------------------------------------------

puts "Creating project: $project_name"
create_project $project_name $project_dir -part $part -force

##------------------------------------------------------------------------------
## Add RTL Sources
##------------------------------------------------------------------------------

puts "Adding RTL sources..."

# Core modules
add_files -norecurse {
    rtl/core/mac_pe.v
    rtl/core/systolic_array.v
    rtl/core/vector_unit.v
    rtl/core/dma_engine.v
}

# Memory
add_files -norecurse {
    rtl/memory/sram_subsystem.v
    rtl/memory/memory_controller_wrapper.v
}

# Note: axi_memory_model.v is simulation-only, don't add for synthesis

# Control
add_files -norecurse {
    rtl/control/local_cmd_processor.v
    rtl/control/global_cmd_processor.v
}

# NoC
add_files -norecurse {
    rtl/noc/noc_router.v
}

# Top level
add_files -norecurse {
    rtl/top/tensor_processing_cluster.v
    rtl/top/tensor_accelerator_top.v
}

# Include directories
set_property include_dirs {rtl/include} [current_fileset]

##------------------------------------------------------------------------------
## Add Constraints
##------------------------------------------------------------------------------

puts "Adding constraints..."
add_files -fileset constrs_1 -norecurse {
    constraints/timing.xdc
}

##------------------------------------------------------------------------------
## Set Top Module and Properties
##------------------------------------------------------------------------------

set_property top tensor_accelerator_top [current_fileset]

# Enable SYNTHESIS define
set_property verilog_define {SYNTHESIS=1} [current_fileset]

# For prototype (reduced SRAM)
set_property verilog_define {SYNTHESIS=1 TARGET_PROTOTYPE=1} [current_fileset]

##------------------------------------------------------------------------------
## Synthesis Settings
##------------------------------------------------------------------------------

puts "Configuring synthesis settings..."

# Use OOC for better timing
set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY rebuilt [get_runs synth_1]

# Aggressive optimization
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING on [get_runs synth_1]

# FSM encoding
set_property STEPS.SYNTH_DESIGN.ARGS.FSM_EXTRACTION one_hot [get_runs synth_1]

##------------------------------------------------------------------------------
## Run Synthesis
##------------------------------------------------------------------------------

puts "Launching synthesis..."
launch_runs synth_1 -jobs 8
wait_on_run synth_1

# Check for errors
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: Synthesis failed!"
    exit 1
}

##------------------------------------------------------------------------------
## Open Results and Generate Reports
##------------------------------------------------------------------------------

puts "Generating reports..."
open_run synth_1

# Utilization report
report_utilization -file ${project_dir}/utilization_synth.rpt
report_utilization -hierarchical -file ${project_dir}/utilization_hier.rpt

# Timing report
report_timing_summary -file ${project_dir}/timing_synth.rpt

# Clock report
report_clocks -file ${project_dir}/clocks.rpt

# CDC report
report_cdc -file ${project_dir}/cdc.rpt

##------------------------------------------------------------------------------
## Print Summary
##------------------------------------------------------------------------------

puts ""
puts "=============================================="
puts "  Synthesis Complete!"
puts "=============================================="
puts ""
puts "Reports generated in: ${project_dir}/"
puts "  - utilization_synth.rpt"
puts "  - utilization_hier.rpt"
puts "  - timing_synth.rpt"
puts "  - clocks.rpt"
puts "  - cdc.rpt"
puts ""

# Print quick utilization summary
puts "Quick Utilization Summary:"
puts [report_utilization -return_string]

puts ""
puts "Next steps:"
puts "  1. Review timing report for any violations"
puts "  2. Run implementation: launch_runs impl_1"
puts "  3. Generate bitstream: launch_runs impl_1 -to_step write_bitstream"
puts ""

##------------------------------------------------------------------------------
## Optional: Run Implementation
##------------------------------------------------------------------------------

# Uncomment to automatically run implementation
# puts "Launching implementation..."
# launch_runs impl_1 -jobs 8
# wait_on_run impl_1
# 
# open_run impl_1
# report_utilization -file ${project_dir}/utilization_impl.rpt
# report_timing_summary -file ${project_dir}/timing_impl.rpt
# report_power -file ${project_dir}/power.rpt

##------------------------------------------------------------------------------
## Optional: Generate Bitstream
##------------------------------------------------------------------------------

# Uncomment to automatically generate bitstream
# puts "Generating bitstream..."
# launch_runs impl_1 -to_step write_bitstream -jobs 8
# wait_on_run impl_1
# puts "Bitstream generated: ${project_dir}/${project_name}.runs/impl_1/tensor_accelerator_top.bit"

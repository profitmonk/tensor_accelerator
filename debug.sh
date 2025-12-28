#!/bin/bash
#==============================================================================
# Interactive Test Review Script
#
# This script runs tests and helps you analyze results step by step
#==============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

#==============================================================================
# Menu
#==============================================================================

show_menu() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║          Tensor Accelerator - Test & Debug Menu            ║${NC}"
    echo -e "${BLUE}╠════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${BLUE}║${NC}  1) Run MAC PE test + view waveform                        ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC}  2) Run Systolic Array test + view waveform                ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC}  3) Run all tests                                          ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC}  4) View existing waveform (MAC PE)                        ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC}  5) View existing waveform (Systolic Array)                ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC}  6) Show signal guide                                      ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC}  7) Generate test vectors                                  ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC}  q) Quit                                                   ${BLUE}║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -n "Select option: "
}

#==============================================================================
# Test Functions
#==============================================================================

run_mac_test() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  Running MAC Processing Element Test${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    mkdir -p sim/waves
    
    # Compile
    echo -e "${YELLOW}[1/3] Compiling...${NC}"
    iverilog -o sim/mac_pe_tb rtl/core/mac_pe.v tb/tb_mac_pe.v
    echo -e "${GREEN}      Compiled: sim/mac_pe_tb${NC}"
    
    # Run
    echo ""
    echo -e "${YELLOW}[2/3] Running simulation...${NC}"
    echo -e "${CYAN}─────────────────────────────────────────────────────────────────${NC}"
    (cd sim && vvp mac_pe_tb)
    echo -e "${CYAN}─────────────────────────────────────────────────────────────────${NC}"
    
    # Move VCD
    mv -f sim/mac_pe.vcd sim/waves/ 2>/dev/null || true
    
    echo ""
    echo -e "${YELLOW}[3/3] Results:${NC}"
    if [ -f sim/waves/mac_pe.vcd ]; then
        VCD_SIZE=$(ls -lh sim/waves/mac_pe.vcd | awk '{print $5}')
        echo -e "${GREEN}      ✓ Waveform saved: sim/waves/mac_pe.vcd ($VCD_SIZE)${NC}"
    else
        echo -e "${RED}      ✗ No waveform generated${NC}"
    fi
    
    echo ""
    echo -e "${YELLOW}Open waveform in GTKWave? (y/n)${NC} "
    read -n 1 answer
    echo ""
    if [ "$answer" = "y" ]; then
        open_mac_waveform
    fi
}

run_systolic_test() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  Running Systolic Array Test (16x16)${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    mkdir -p sim/waves
    
    # Compile
    echo -e "${YELLOW}[1/3] Compiling...${NC}"
    iverilog -o sim/systolic_tb \
        rtl/core/mac_pe.v \
        rtl/core/systolic_array.v \
        tb/tb_systolic_array.v
    echo -e "${GREEN}      Compiled: sim/systolic_tb${NC}"
    
    # Run
    echo ""
    echo -e "${YELLOW}[2/3] Running simulation...${NC}"
    echo -e "${CYAN}─────────────────────────────────────────────────────────────────${NC}"
    (cd sim && vvp systolic_tb)
    echo -e "${CYAN}─────────────────────────────────────────────────────────────────${NC}"
    
    # Move VCD
    mv -f sim/systolic_array.vcd sim/waves/ 2>/dev/null || true
    
    echo ""
    echo -e "${YELLOW}[3/3] Results:${NC}"
    if [ -f sim/waves/systolic_array.vcd ]; then
        VCD_SIZE=$(ls -lh sim/waves/systolic_array.vcd | awk '{print $5}')
        echo -e "${GREEN}      ✓ Waveform saved: sim/waves/systolic_array.vcd ($VCD_SIZE)${NC}"
    else
        echo -e "${RED}      ✗ No waveform generated${NC}"
    fi
    
    echo ""
    echo -e "${YELLOW}Open waveform in GTKWave? (y/n)${NC} "
    read -n 1 answer
    echo ""
    if [ "$answer" = "y" ]; then
        open_systolic_waveform
    fi
}

#==============================================================================
# Waveform Viewing
#==============================================================================

open_mac_waveform() {
    if [ ! -f sim/waves/mac_pe.vcd ]; then
        echo -e "${RED}No waveform file found. Run the test first.${NC}"
        return
    fi
    
    echo ""
    echo -e "${CYAN}Opening waveform viewer...${NC}"
    echo ""
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}  MAC PE Waveform Viewing Guide${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "  Key signals to add:"
    echo ""
    echo "  ${GREEN}Clock & Reset:${NC}"
    echo "    • clk              - System clock"
    echo "    • rst_n            - Active-low reset"
    echo ""
    echo "  ${GREEN}Control Signals:${NC}"
    echo "    • enable           - Pipeline enable"
    echo "    • load_weight      - Weight loading strobe"
    echo "    • clear_acc        - Clear accumulator"
    echo ""
    echo "  ${GREEN}Data Path:${NC}"
    echo "    • weight_in[7:0]   - Weight input"
    echo "    • weight_reg[7:0]  - Stored weight (internal)"
    echo "    • act_in[7:0]      - Activation input"
    echo "    • act_out[7:0]     - Activation output (to right neighbor)"
    echo "    • psum_in[31:0]    - Partial sum input (from top)"
    echo "    • psum_out[31:0]   - Partial sum output (to bottom)"
    echo ""
    echo "  ${GREEN}What to look for:${NC}"
    echo "    1. Weight loads when load_weight=1"
    echo "    2. act_out follows act_in by 1 cycle"
    echo "    3. psum_out = psum_in + (act_in × weight_reg)"
    echo ""
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # Try to open with Surfer first, then GTKWave
    if command -v surfer &> /dev/null; then
        echo -e "${GREEN}Opening with Surfer...${NC}"
        surfer sim/waves/mac_pe.vcd &
    elif command -v gtkwave &> /dev/null; then
        echo -e "${GREEN}Opening with GTKWave...${NC}"
        gtkwave sim/waves/mac_pe.vcd sim/waves/mac_pe.gtkw &
    elif [ -d "/Applications/gtkwave.app" ]; then
        open -a gtkwave sim/waves/mac_pe.vcd
    else
        echo -e "${RED}No waveform viewer found.${NC}"
        echo "Install Surfer:  brew install surfer"
        echo "Or GTKWave:      brew install --cask gtkwave"
        echo ""
        echo "VCD file is at: sim/waves/mac_pe.vcd"
    fi
}

open_systolic_waveform() {
    if [ ! -f sim/waves/systolic_array.vcd ]; then
        echo -e "${RED}No waveform file found. Run the test first.${NC}"
        return
    fi
    
    echo ""
    echo -e "${CYAN}Opening waveform viewer...${NC}"
    echo ""
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}  Systolic Array Waveform Viewing Guide${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "  Key signals to add:"
    echo ""
    echo "  ${GREEN}Top Level:${NC}"
    echo "    • clk, rst_n        - Clock and reset"
    echo "    • start             - Start computation"
    echo "    • busy              - Array is processing"
    echo "    • done              - Computation complete"
    echo "    • state[2:0]        - State machine (IDLE=0, LOAD=1, COMPUTE=2, DRAIN=3)"
    echo ""
    echo "  ${GREEN}Weight Loading:${NC}"
    echo "    • weight_load_en    - Weight load enable"
    echo "    • weight_load_col   - Which column being loaded"
    echo "    • weight_load_data  - Weight data (all rows for one column)"
    echo ""
    echo "  ${GREEN}Activation Flow:${NC}"
    echo "    • act_valid         - Activation input valid"
    echo "    • act_data          - Activation input (one element per row)"
    echo ""
    echo "  ${GREEN}Results:${NC}"
    echo "    • result_valid      - Result output valid"
    echo "    • result_data       - Result output (one element per column)"
    echo ""
    echo "  ${GREEN}Individual PEs (drill down):${NC}"
    echo "    • pe_row[0].pe_col[0].pe_inst.weight_reg  - First PE weight"
    echo "    • pe_row[0].pe_col[0].pe_inst.psum_out    - First PE output"
    echo ""
    echo "  ${GREEN}What to look for:${NC}"
    echo "    1. State transitions: IDLE → LOAD → COMPUTE → DRAIN → IDLE"
    echo "    2. Weights load column by column (16 cycles)"
    echo "    3. Activations stream in during COMPUTE"
    echo "    4. Results emerge during DRAIN"
    echo ""
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # Try to open with Surfer first, then GTKWave
    if command -v surfer &> /dev/null; then
        echo -e "${GREEN}Opening with Surfer...${NC}"
        surfer sim/waves/systolic_array.vcd &
    elif command -v gtkwave &> /dev/null; then
        echo -e "${GREEN}Opening with GTKWave...${NC}"
        gtkwave sim/waves/systolic_array.vcd sim/waves/systolic_array.gtkw &
    elif [ -d "/Applications/gtkwave.app" ]; then
        open -a gtkwave sim/waves/systolic_array.vcd
    else
        echo -e "${RED}No waveform viewer found.${NC}"
        echo "Install Surfer:  brew install surfer"
        echo "Or GTKWave:      brew install --cask gtkwave"
        echo ""
        echo "VCD file is at: sim/waves/systolic_array.vcd"
    fi
}

#==============================================================================
# Signal Guide
#==============================================================================

show_signal_guide() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  GTKWave Quick Reference${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${YELLOW}Navigation:${NC}"
    echo "  • Mouse wheel     - Zoom in/out"
    echo "  • Click + drag    - Pan"
    echo "  • Shift + click   - Zoom to selection"
    echo "  • Home            - Go to start"
    echo "  • End             - Go to end"
    echo ""
    echo -e "${YELLOW}Adding Signals:${NC}"
    echo "  1. Left panel (SST) shows module hierarchy"
    echo "  2. Click module to see its signals"
    echo "  3. Double-click signal to add to waveform"
    echo "  4. Or: Select signals → Append (Shift+Ctrl+A)"
    echo ""
    echo -e "${YELLOW}Display Options:${NC}"
    echo "  • Right-click signal → Data Format → Decimal/Hex/Binary"
    echo "  • Right-click signal → Color → Change trace color"
    echo "  • Edit → Expand/Collapse (for buses)"
    echo ""
    echo -e "${YELLOW}Markers & Cursors:${NC}"
    echo "  • Left-click       - Place primary cursor"
    echo "  • Middle-click     - Place secondary cursor"
    echo "  • Status bar shows delta time between cursors"
    echo ""
    echo -e "${YELLOW}Saving Session:${NC}"
    echo "  • File → Write Save File (*.gtkw)"
    echo "  • Saves signal selection and zoom level"
    echo "  • Reload with: gtkwave file.vcd file.gtkw"
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
}

#==============================================================================
# Test Vectors
#==============================================================================

generate_test_vectors() {
    echo ""
    echo -e "${CYAN}Generating test vectors...${NC}"
    echo ""
    
    mkdir -p sim/test_vectors
    
    cd sw
    python3 test_generator.py --output ../sim/test_vectors --test all
    cd ..
    
    echo ""
    echo -e "${GREEN}Test vectors generated:${NC}"
    ls -la sim/test_vectors/
}

#==============================================================================
# Main Loop
#==============================================================================

while true; do
    show_menu
    read -n 1 choice
    echo ""
    
    case $choice in
        1) run_mac_test ;;
        2) run_systolic_test ;;
        3) run_mac_test; run_systolic_test ;;
        4) open_mac_waveform ;;
        5) open_systolic_waveform ;;
        6) show_signal_guide ;;
        7) generate_test_vectors ;;
        q|Q) echo "Goodbye!"; exit 0 ;;
        *) echo -e "${RED}Invalid option${NC}" ;;
    esac
    
    echo ""
    echo -e "${YELLOW}Press any key to continue...${NC}"
    read -n 1
done

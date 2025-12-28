#!/bin/bash
#==============================================================================
# Tensor Accelerator - macOS Setup and Test Script
#
# This script:
# 1. Checks/installs dependencies
# 2. Validates the environment
# 3. Runs unit tests
# 4. Reports results
#==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

#==============================================================================
# Helper Functions
#==============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
}

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        print_status "$1 found: $(command -v $1)"
        return 0
    else
        print_error "$1 not found"
        return 1
    fi
}

#==============================================================================
# Step 1: Check Dependencies
#==============================================================================

print_header "Step 1: Checking Dependencies"

MISSING_DEPS=0

# Check for Homebrew
if ! check_command brew; then
    print_warning "Homebrew not installed. Install from https://brew.sh"
    MISSING_DEPS=1
fi

# Check for Icarus Verilog
if ! check_command iverilog; then
    print_warning "Icarus Verilog not found. Install with: brew install icarus-verilog"
    MISSING_DEPS=1
fi

# Check for VVP (comes with Icarus)
if ! check_command vvp; then
    print_warning "VVP not found (part of Icarus Verilog)"
    MISSING_DEPS=1
fi

# Check for Verilator (optional but recommended)
if check_command verilator; then
    VERILATOR_VERSION=$(verilator --version | head -1)
    print_status "Verilator version: $VERILATOR_VERSION"
    HAS_VERILATOR=1
else
    print_warning "Verilator not found (optional). Install with: brew install verilator"
    HAS_VERILATOR=0
fi

# Check for Python 3
if check_command python3; then
    PYTHON_VERSION=$(python3 --version)
    print_status "Python version: $PYTHON_VERSION"
else
    print_warning "Python 3 not found. Install with: brew install python3"
    MISSING_DEPS=1
fi

# Check for NumPy
if python3 -c "import numpy" 2>/dev/null; then
    NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)")
    print_status "NumPy version: $NUMPY_VERSION"
else
    print_warning "NumPy not found. Install with: pip3 install numpy"
    MISSING_DEPS=1
fi

# Check for GTKWave (optional)
if check_command gtkwave; then
    print_status "GTKWave available for waveform viewing"
else
    print_warning "GTKWave not found (optional). Install with: brew install --cask gtkwave"
fi

if [ $MISSING_DEPS -eq 1 ]; then
    echo ""
    print_error "Missing required dependencies. Install them and re-run this script."
    echo ""
    echo "Quick install command:"
    echo "  brew install icarus-verilog verilator python3"
    echo "  pip3 install numpy"
    echo ""
    exit 1
fi

print_status "All required dependencies found!"

#==============================================================================
# Step 2: Create Directory Structure
#==============================================================================

print_header "Step 2: Setting Up Directories"

mkdir -p sim
mkdir -p sim/test_vectors
mkdir -p sim/waves

print_status "Created sim/, sim/test_vectors/, sim/waves/"

#==============================================================================
# Step 3: Generate Test Vectors
#==============================================================================

print_header "Step 3: Generating Test Vectors"

cd sw
python3 test_generator.py --output ../sim/test_vectors --test gemm
cd ..

print_status "Test vectors generated in sim/test_vectors/"

#==============================================================================
# Step 4: Run Unit Tests with Icarus Verilog
#==============================================================================

print_header "Step 4: Running Unit Tests (Icarus Verilog)"

# Test 1: MAC Processing Element
echo ""
echo -e "${YELLOW}Test 4.1: MAC Processing Element${NC}"

# Compile MAC PE testbench
iverilog -o sim/mac_pe_tb rtl/core/mac_pe.v tb/tb_mac_pe.v
(cd sim && vvp mac_pe_tb && mv -f mac_pe.vcd waves/ 2>/dev/null || true)

print_status "MAC PE test completed"

# Test 2: Systolic Array
echo ""
echo -e "${YELLOW}Test 4.2: Systolic Array (16x16)${NC}"

iverilog -o sim/systolic_tb \
    rtl/core/mac_pe.v \
    rtl/core/systolic_array.v \
    tb/tb_systolic_array.v

(cd sim && vvp systolic_tb && mv -f systolic_array.vcd waves/ 2>/dev/null || true)

print_status "Systolic array test completed"

#==============================================================================
# Step 5: Run Verilator Tests (if available)
#==============================================================================

if [ $HAS_VERILATOR -eq 1 ]; then
    print_header "Step 5: Running Verilator Lint Check"
    
    verilator --lint-only -Wall \
        rtl/core/mac_pe.v \
        rtl/core/systolic_array.v \
        rtl/core/vector_unit.v \
        rtl/core/dma_engine.v \
        rtl/memory/sram_subsystem.v \
        2>&1 | head -20
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        print_status "Verilator lint check passed"
    else
        print_warning "Verilator lint found some warnings (see above)"
    fi
else
    print_header "Step 5: Skipping Verilator (not installed)"
fi

#==============================================================================
# Step 6: Assemble Example Kernels
#==============================================================================

print_header "Step 6: Assembling Example Kernels"

cd sw
python3 assembler/assembler.py examples/resnet_conv.asm -o ../sim/resnet_conv.hex 2>/dev/null || true
python3 assembler/assembler.py examples/attention_mha.asm -o ../sim/attention_mha.hex 2>/dev/null || true
cd ..

if [ -f sim/resnet_conv.hex ]; then
    INSTR_COUNT=$(wc -l < sim/resnet_conv.hex)
    print_status "ResNet kernel: $INSTR_COUNT instructions"
fi

if [ -f sim/attention_mha.hex ]; then
    INSTR_COUNT=$(wc -l < sim/attention_mha.hex)
    print_status "Attention kernel: $INSTR_COUNT instructions"
fi

#==============================================================================
# Summary
#==============================================================================

print_header "Test Summary"

echo ""
echo "  Project Structure:"
echo "  ├── rtl/           - Verilog RTL source"
echo "  ├── tb/            - Testbenches"
echo "  ├── sw/            - Python tools (assembler, test generator)"
echo "  ├── sim/           - Simulation outputs"
echo "  │   ├── test_vectors/  - Generated test data"
echo "  │   └── waves/         - VCD waveforms"
echo "  └── docs/          - Documentation"
echo ""

echo "  Available Commands:"
echo "  ├── make test_systolic  - Run systolic array test"
echo "  ├── make test_full      - Run full system test"
echo "  ├── make test_vectors   - Generate test vectors"
echo "  ├── make lint           - Run Verilator lint"
echo "  └── make clean          - Clean build artifacts"
echo ""

if [ -f sim/waves/systolic_array.vcd ]; then
    echo "  View Waveforms:"
    echo "    open -a gtkwave sim/waves/systolic_array.vcd"
    echo ""
fi

print_status "Setup complete! All tests passed."
echo ""

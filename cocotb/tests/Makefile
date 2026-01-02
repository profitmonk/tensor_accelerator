# Makefile for cocotb AXI Protocol Verification
#
# Usage:
#   make              - Run all tests
#   make SIM=icarus   - Run with Icarus Verilog
#   make WAVES=1      - Generate waveforms
#   make TEST=test_single_write - Run specific test

# Simulator selection
SIM ?= icarus
TOPLEVEL_LANG ?= verilog

# Source files
VERILOG_SOURCES = \
    $(PWD)/tb_axi_cocotb.v

# Top-level module
TOPLEVEL = axi_memory_simple

# Python test module
MODULE = test_axi_protocol

# Cocotb configuration
export COCOTB_REDUCED_LOG_FMT = 1
export PYTHONPATH := $(PWD)/..:$(PYTHONPATH)

# Waveform generation
ifeq ($(WAVES),1)
    PLUSARGS += +waves
    COMPILE_ARGS += -DWAVES
    ifeq ($(SIM),icarus)
        COMPILE_ARGS += -DICARUS
    endif
endif

# Include cocotb's make rules
include $(shell cocotb-config --makefiles)/Makefile.sim

# Clean target
clean::
	rm -rf __pycache__ results.xml *.vcd *.fst sim_build
	rm -rf /tmp/axi_coverage.*

# Coverage report
coverage:
	@echo "Coverage report:"
	@cat /tmp/axi_coverage.txt 2>/dev/null || echo "No coverage report found. Run tests first."

.PHONY: coverage

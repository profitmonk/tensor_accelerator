#==============================================================================
# Tensor Accelerator Makefile
#==============================================================================

# Directories
RTL_DIR      = rtl
TB_DIR       = tb
SIM_DIR      = sim
SW_DIR       = sw
TEST_VEC_DIR = $(SIM_DIR)/test_vectors

# Simulator (iverilog or verilator)
SIM          = iverilog
VVP          = vvp
WAVE_VIEWER  = gtkwave

# Verilator options (for faster simulation)
VERILATOR    = verilator
VERILATOR_FLAGS = --cc --exe --build -Wall

# RTL Source Files
RTL_CORE = \
    $(RTL_DIR)/core/mac_pe.v \
    $(RTL_DIR)/core/systolic_array.v \
    $(RTL_DIR)/core/vector_unit.v \
    $(RTL_DIR)/core/dma_engine.v

RTL_MEMORY = \
    $(RTL_DIR)/memory/sram_subsystem.v

RTL_CONTROL = \
    $(RTL_DIR)/control/local_cmd_processor.v \
    $(RTL_DIR)/control/global_cmd_processor.v

RTL_NOC = \
    $(RTL_DIR)/noc/noc_router.v

RTL_TOP = \
    $(RTL_DIR)/top/tensor_processing_cluster.v \
    $(RTL_DIR)/top/tensor_accelerator_top.v

RTL_ALL = $(RTL_CORE) $(RTL_MEMORY) $(RTL_CONTROL) $(RTL_NOC) $(RTL_TOP)

# Testbenches
TB_MAC     = $(TB_DIR)/tb_mac_pe.v
TB_SYSTOLIC = $(TB_DIR)/tb_systolic_array.v
TB_VPU     = $(TB_DIR)/tb_vector_unit.v
TB_TPC     = $(TB_DIR)/tb_single_tpc.v
TB_TOP     = $(TB_DIR)/tb_tensor_accelerator.v

#==============================================================================
# Targets
#==============================================================================

.PHONY: all clean test test_mac test_systolic test_vpu test_tpc test_full \
        test_vectors waves help lint

all: test

help:
	@echo "Tensor Accelerator Build System"
	@echo "================================"
	@echo ""
	@echo "Targets:"
	@echo "  make test           - Run all tests"
	@echo "  make test_mac       - Test MAC processing element"
	@echo "  make test_systolic  - Test systolic array"
	@echo "  make test_vpu       - Test vector processing unit"
	@echo "  make test_tpc       - Test single TPC"
	@echo "  make test_full      - Test full accelerator"
	@echo "  make test_vectors   - Generate test vectors"
	@echo "  make lint           - Run Verilator lint check"
	@echo "  make waves          - Open waveform viewer"
	@echo "  make clean          - Clean build artifacts"

#==============================================================================
# Create directories
#==============================================================================

$(SIM_DIR):
	mkdir -p $(SIM_DIR)

$(TEST_VEC_DIR):
	mkdir -p $(TEST_VEC_DIR)

#==============================================================================
# Unit Tests
#==============================================================================

# MAC PE Test
test_mac: $(SIM_DIR)/mac_pe_tb | $(SIM_DIR)
	cd $(SIM_DIR) && $(VVP) mac_pe_tb
	@echo "MAC PE test completed"

$(SIM_DIR)/mac_pe_tb: $(RTL_DIR)/core/mac_pe.v $(TB_MAC) | $(SIM_DIR)
	$(SIM) -o $@ $^

# Systolic Array Test
test_systolic: $(SIM_DIR)/systolic_tb | $(SIM_DIR)
	cd $(SIM_DIR) && $(VVP) systolic_tb
	@echo "Systolic array test completed"

$(SIM_DIR)/systolic_tb: $(RTL_DIR)/core/mac_pe.v $(RTL_DIR)/core/systolic_array.v $(TB_SYSTOLIC) | $(SIM_DIR)
	$(SIM) -o $@ $^

# Vector Unit Test
test_vpu: $(SIM_DIR)/vpu_tb | $(SIM_DIR)
	cd $(SIM_DIR) && $(VVP) vpu_tb
	@echo "VPU test completed"

$(SIM_DIR)/vpu_tb: $(RTL_DIR)/core/vector_unit.v $(TB_VPU) | $(SIM_DIR)
	$(SIM) -o $@ $^

# Single TPC Test
test_tpc: $(SIM_DIR)/tpc_tb | $(SIM_DIR)
	cd $(SIM_DIR) && $(VVP) tpc_tb
	@echo "TPC test completed"

$(SIM_DIR)/tpc_tb: $(RTL_CORE) $(RTL_MEMORY) $(RTL_CONTROL) $(RTL_DIR)/top/tensor_processing_cluster.v $(TB_TPC) | $(SIM_DIR)
	$(SIM) -o $@ -I $(RTL_DIR) $^

#==============================================================================
# Full System Test
#==============================================================================

test_full: $(SIM_DIR)/tensor_accel_tb test_vectors | $(SIM_DIR)
	cd $(SIM_DIR) && $(VVP) tensor_accel_tb
	@echo "Full system test completed"

$(SIM_DIR)/tensor_accel_tb: $(RTL_ALL) $(TB_TOP) | $(SIM_DIR)
	$(SIM) -o $@ -I $(RTL_DIR) $^

#==============================================================================
# Test Vector Generation
#==============================================================================

test_vectors: | $(TEST_VEC_DIR)
	cd $(SW_DIR) && python3 test_generator.py --output ../$(TEST_VEC_DIR) --test all
	@echo "Test vectors generated in $(TEST_VEC_DIR)"

#==============================================================================
# Assemble Example Kernels
#==============================================================================

assemble_examples:
	cd $(SW_DIR) && python3 assembler/assembler.py examples/resnet_conv.asm -o ../$(SIM_DIR)/resnet_conv.hex
	cd $(SW_DIR) && python3 assembler/assembler.py examples/attention_mha.asm -o ../$(SIM_DIR)/attention_mha.hex
	@echo "Kernels assembled"

#==============================================================================
# Lint Check
#==============================================================================

lint:
	$(VERILATOR) --lint-only -Wall -I $(RTL_DIR) $(RTL_ALL)
	@echo "Lint check passed"

#==============================================================================
# Waveforms
#==============================================================================

waves:
	$(WAVE_VIEWER) $(SIM_DIR)/*.vcd &

#==============================================================================
# Run all tests
#==============================================================================

test: test_mac test_systolic test_vpu test_tpc
	@echo ""
	@echo "=========================================="
	@echo "All unit tests completed"
	@echo "=========================================="

#==============================================================================
# FPGA Synthesis (Vivado)
#==============================================================================

VIVADO = vivado
VIVADO_FLAGS = -mode batch

synth:
	cd constraints && $(VIVADO) $(VIVADO_FLAGS) -source create_project.tcl
	cd constraints && $(VIVADO) $(VIVADO_FLAGS) -source run_synthesis.tcl

impl:
	cd constraints && $(VIVADO) $(VIVADO_FLAGS) -source run_implementation.tcl

bitstream:
	cd constraints && $(VIVADO) $(VIVADO_FLAGS) -source generate_bitstream.tcl

fpga: synth impl bitstream

#==============================================================================
# Clean
#==============================================================================

clean:
	rm -rf $(SIM_DIR)/*.vcd
	rm -rf $(SIM_DIR)/*_tb
	rm -rf $(SIM_DIR)/test_vectors
	rm -rf *.log
	rm -rf obj_dir
	@echo "Cleaned build artifacts"

clean_all: clean
	rm -rf $(SIM_DIR)
	@echo "Cleaned all simulation files"

#==============================================================================
# Verilator Build (for faster simulation)
#==============================================================================

verilator_build:
	$(VERILATOR) $(VERILATOR_FLAGS) \
		-I $(RTL_DIR) \
		--top-module tensor_accelerator_top \
		$(RTL_ALL) \
		$(TB_DIR)/tb_verilator.cpp

verilator_run: verilator_build
	./obj_dir/Vtensor_accelerator_top

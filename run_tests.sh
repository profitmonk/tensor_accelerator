#!/bin/bash
cd "$(dirname "$0")"
PASS=0
FAIL=0

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║        Tensor Accelerator - Full Test Suite                ║"
echo "╚════════════════════════════════════════════════════════════╝"

mkdir -p sim

run_test() {
    name="$1"; compile_cmd="$2"; sim_exe="$3"; pass_pattern="${4:-ALL TESTS PASSED}"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "TEST: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if eval "$compile_cmd" 2>/dev/null; then
        output=$(cd sim && vvp "$sim_exe" 2>&1)
        echo "$output" | tail -20
        if echo "$output" | grep -q "$pass_pattern"; then
            echo "✓ PASSED"; PASS=$((PASS + 1))
        else
            echo "✗ FAILED"; FAIL=$((FAIL + 1))
        fi
    else
        echo "✗ COMPILE ERROR"; FAIL=$((FAIL + 1))
    fi
}

echo ""
echo "═══════════════════ UNIT TESTS ═══════════════════"

run_test "MAC PE" \
    "iverilog -o sim/tb_mac rtl/core/mac_pe.v tb/tb_mac_pe.v" "tb_mac"

run_test "Systolic Array" \
    "iverilog -o sim/tb_sys rtl/core/mac_pe.v rtl/core/systolic_array.v tb/tb_systolic_array.v" "tb_sys"

run_test "Vector Unit" \
    "iverilog -g2012 -o sim/tb_vpu rtl/core/vector_unit.v tb/tb_vector_unit.v" "tb_vpu"

run_test "DMA Engine" \
    "iverilog -g2012 -o sim/tb_dma rtl/core/dma_engine.v tb/tb_dma_engine.v" "tb_dma"

run_test "Local Command Processor" \
    "iverilog -g2012 -o sim/tb_lcp rtl/control/local_cmd_processor.v tb/tb_local_cmd_processor.v" "tb_lcp"

run_test "Global Command Processor" \
    "iverilog -g2012 -o sim/tb_gcp rtl/control/global_cmd_processor.v tb/tb_global_cmd_processor.v" "tb_gcp"

run_test "SRAM Subsystem" \
    "iverilog -g2012 -DSIM -o sim/tb_sram rtl/memory/sram_subsystem.v tb/tb_sram_subsystem.v" "tb_sram"

run_test "NoC Router" \
    "iverilog -g2012 -o sim/tb_noc rtl/noc/noc_router.v tb/tb_noc_router.v" "tb_noc"

run_test "NoC 2x2 Mesh" \
    "iverilog -g2012 -o sim/tb_noc_mesh rtl/noc/noc_router.v tb/tb_noc_mesh_2x2.v" "tb_noc_mesh" "NOC MESH INTEGRATION TESTS PASSED"

echo ""
echo "═══════════════ INTEGRATION TESTS ═══════════════"

run_test "TPC Integration" \
    "iverilog -g2012 -DSIM -o sim/tb_tpc rtl/control/local_cmd_processor.v rtl/core/systolic_array.v rtl/core/mac_pe.v rtl/core/vector_unit.v rtl/core/dma_engine.v rtl/memory/sram_subsystem.v rtl/top/tensor_processing_cluster.v tb/tb_tpc.v" "tb_tpc"

run_test "Full Chip (Top Level)" \
    "iverilog -g2012 -DSIM -o sim/tb_top rtl/control/local_cmd_processor.v rtl/control/global_cmd_processor.v rtl/core/systolic_array.v rtl/core/mac_pe.v rtl/core/vector_unit.v rtl/core/dma_engine.v rtl/memory/sram_subsystem.v rtl/top/tensor_processing_cluster.v rtl/top/tensor_accelerator_top.v tb/tb_top.v" "tb_top"

echo ""
echo "═══════════════ END-TO-END TESTS ═══════════════"

TPC_FILES="rtl/control/local_cmd_processor.v rtl/core/systolic_array.v rtl/core/mac_pe.v rtl/core/vector_unit.v rtl/core/dma_engine.v rtl/memory/sram_subsystem.v rtl/top/tensor_processing_cluster.v"

run_test "E2E: Identity GEMM" \
    "iverilog -g2012 -DSIM -o sim/tb_e2e_simple $TPC_FILES tb/tb_e2e_simple.v" \
    "tb_e2e_simple" "E2E GEMM TEST PASSED"

run_test "E2E: Random 4×4 GEMM" \
    "iverilog -g2012 -DSIM -o sim/tb_e2e_gemm_random $TPC_FILES tb/tb_e2e_gemm_random.v" \
    "tb_e2e_gemm_random" "RANDOM GEMM TEST PASSED"

run_test "E2E: Multi-GEMM Sequential" \
    "iverilog -g2012 -DSIM -o sim/tb_e2e_multi_gemm $TPC_FILES tb/tb_e2e_multi_gemm.v" \
    "tb_e2e_multi_gemm" "MULTI GEMM TEST PASSED"

run_test "E2E: Conv2D (im2col)" \
    "iverilog -g2012 -DSIM -o sim/tb_e2e_conv2d $TPC_FILES tb/tb_e2e_conv2d.v" \
    "tb_e2e_conv2d" "CONV2D TEST PASSED"

echo ""
echo "═══════════════ STRESS TESTS ═══════════════"

run_test "Stress: Signed/Overflow/Sparse/FixedPt" \
    "iverilog -g2012 -DSIM -o sim/tb_stress_test $TPC_FILES tb/tb_stress_test.v" \
    "tb_stress_test" "ALL STRESS TESTS PASSED"

run_test "Stress: Random Matrix (numpy-verified)" \
    "iverilog -g2012 -DSIM -o sim/tb_random_simple $TPC_FILES tb/tb_random_simple.v" \
    "tb_random_simple" "RANDOM MATRIX TEST PASSED"

echo ""
echo "═══════════════ DMA TESTS ═══════════════"

run_test "DMA: LOAD/STORE with AXI Memory" \
    "iverilog -g2012 -DSIM -o sim/tb_dma_axi rtl/core/dma_engine.v rtl/memory/axi_memory_model.v tb/tb_dma_axi.v" \
    "tb_dma_axi" "ALL DMA TESTS PASSED"

echo ""
echo "═══════════════ MULTI-TPC TESTS ═══════════════"

TOP_FILES="rtl/control/local_cmd_processor.v rtl/control/global_cmd_processor.v rtl/core/systolic_array.v rtl/core/mac_pe.v rtl/core/vector_unit.v rtl/core/dma_engine.v rtl/memory/sram_subsystem.v rtl/memory/axi_memory_model.v rtl/top/tensor_processing_cluster.v rtl/top/tensor_accelerator_top.v"

run_test "Multi-TPC: 4-way Parallel Tiled GEMM" \
    "iverilog -g2012 -DSIM -o sim/tb_multi_tpc_gemm $TOP_FILES tb/tb_multi_tpc_gemm.v" \
    "tb_multi_tpc_gemm" "MULTI-TPC GEMM TEST PASSED"

run_test "K-Accumulation: Tiled GEMM with VPU ADD" \
    "iverilog -g2012 -DSIM -o sim/tb_k_accumulation $TPC_FILES tb/tb_k_accumulation.v" \
    "tb_k_accumulation" "K-ACCUMULATION TEST PASSED"

run_test "E2E Inference: ReLU(X×W + bias)" \
    "iverilog -g2012 -DSIM -o sim/tb_e2e_inference $TPC_FILES tb/tb_e2e_inference.v" \
    "tb_e2e_inference" "E2E INFERENCE TEST PASSED"

run_test "Residual Block: Y = ReLU(X×W+b) + X" \
    "iverilog -g2012 -DSIM -o sim/tb_residual_block $TPC_FILES tb/tb_residual_block.v" \
    "tb_residual_block" "RESIDUAL BLOCK TEST PASSED"

run_test "Batch Processing: N samples, shared weights" \
    "iverilog -g2012 -DSIM -o sim/tb_batch_inference $TPC_FILES tb/tb_batch_inference.v" \
    "tb_batch_inference" "BATCH PROCESSING TEST PASSED"

run_test "2-Layer MLP: Layer chaining" \
    "iverilog -g2012 -DSIM -o sim/tb_mlp_2layer $TPC_FILES tb/tb_mlp_2layer.v" \
    "tb_mlp_2layer" "2-LAYER MLP TEST PASSED"

echo ""
echo "═══════════════ PYTHON MODEL TESTS ═══════════════"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST: Systolic Array Python Model"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
output=$(python3 model/systolic_array_model.py 2>&1)
echo "$output" | grep -E "(TEST|PASS|FAIL|>>>)"
if echo "$output" | grep -q "ALL TESTS PASSED"; then
    echo "✓ PASSED"; PASS=$((PASS + 1))
else
    echo "✗ FAILED"; FAIL=$((FAIL + 1))
fi

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    TEST SUMMARY                            ║"
echo "╠════════════════════════════════════════════════════════════╣"
printf "║   Passed: %-3d                                             ║\n" $PASS
printf "║   Failed: %-3d                                             ║\n" $FAIL
echo "╚════════════════════════════════════════════════════════════╝"

if [ $FAIL -eq 0 ]; then echo ">>> ALL TESTS PASSED! <<<"; exit 0
else echo ">>> SOME TESTS FAILED <<<"; exit 1; fi

echo ""
echo "═══════════════ PYTHON MODEL TESTS (EXTENDED) ═══════════════"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST: DMA Python Model"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd model && python3 dma_model.py
if [ $? -eq 0 ]; then echo "✓ PASSED"; ((PASSED++)); else echo "✗ FAILED"; ((FAILED++)); fi
cd ..

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST: VPU Python Model"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd model && python3 vpu_model.py
if [ $? -eq 0 ]; then echo "✓ PASSED"; ((PASSED++)); else echo "✗ FAILED"; ((FAILED++)); fi
cd ..

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST: LCP Python Model"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd model && python3 lcp_model.py
if [ $? -eq 0 ]; then echo "✓ PASSED"; ((PASSED++)); else echo "✗ FAILED"; ((FAILED++)); fi
cd ..

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST: TPC Integrated Python Model"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd model && python3 tpc_model.py
if [ $? -eq 0 ]; then echo "✓ PASSED"; ((PASSED++)); else echo "✗ FAILED"; ((FAILED++)); fi
cd ..

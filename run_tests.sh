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
        output=$(vvp "sim/$sim_exe" 2>&1)
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

# NEW: Phase 3 comprehensive systolic array tests
run_test "Systolic Array Comprehensive (5 tests)" \
    "iverilog -g2012 -I rtl/include -o sim/tb_sys_comp rtl/core/mac_pe.v rtl/core/systolic_array.v tb/tb_systolic_comprehensive.v" \
    "tb_sys_comp" "5/5 tests passed"

run_test "Systolic Array Simple GEMM" \
    "iverilog -g2012 -I rtl/include -o sim/tb_simple_gemm rtl/core/mac_pe.v rtl/core/systolic_array.v tb/tb_simple_gemm.v" \
    "tb_simple_gemm" "TEST PASSED"

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

run_test "Tiled GEMM: K-accumulation" \
    "iverilog -g2012 -DSIM -o sim/tb_tiled_gemm $TPC_FILES tb/tb_tiled_gemm.v" \
    "tb_tiled_gemm" "TILED GEMM TEST PASSED"

run_test "Conv2D Multi-Channel: im2col + GEMM" \
    "iverilog -g2012 -DSIM -o sim/tb_conv2d_multichannel $TPC_FILES tb/tb_conv2d_multichannel.v" \
    "tb_conv2d_multichannel" "CONV2D MULTICHANNEL TEST PASSED"

run_test "Attention Score: Two-GEMM + VPU ReLU" \
    "iverilog -g2012 -DSIM -o sim/tb_attention $TPC_FILES tb/tb_attention.v" \
    "tb_attention" "ATTENTION TEST PASSED"

# VPU Completeness Tests (Option A)
run_test "VPU MUL: Element-wise multiply" \
    "iverilog -g2012 -DSIM -o sim/tb_vpu_mul rtl/core/vector_unit.v tb/tb_vpu_mul.v" \
    "tb_vpu_mul" "VPU MUL TEST PASSED"

run_test "VPU Reduce: SUM, MAX, MIN" \
    "iverilog -g2012 -DSIM -o sim/tb_vpu_reduce rtl/core/vector_unit.v tb/tb_vpu_reduce.v" \
    "tb_vpu_reduce" "VPU REDUCE TEST PASSED"

run_test "MaxPool 2×2: CNN pooling layer" \
    "iverilog -g2012 -DSIM -o sim/tb_maxpool_2x2 $TPC_FILES tb/tb_maxpool_2x2.v" \
    "tb_maxpool_2x2" "MAXPOOL TEST PASSED"

run_test "BatchNorm: Fused scale*x + bias" \
    "iverilog -g2012 -DSIM -o sim/tb_batchnorm rtl/core/vector_unit.v tb/tb_batchnorm.v" \
    "tb_batchnorm" "BATCHNORM TEST PASSED"

run_test "AvgPool: SUM + divide" \
    "iverilog -g2012 -DSIM -o sim/tb_avgpool rtl/core/vector_unit.v tb/tb_avgpool.v" \
    "tb_avgpool" "AVGPOOL TEST PASSED"

run_test "LeNet-5: Full CNN model" \
    "iverilog -g2012 -DSIM -o sim/tb_lenet5 $TPC_FILES tb/tb_lenet5.v" \
    "tb_lenet5" "LENET-5 TEST PASSED"

run_test "ResNet-18 Block: BN→ReLU→BN→(+skip)→ReLU" \
    "iverilog -g2012 -DSIM -o sim/tb_resnet18_block rtl/core/vector_unit.v tb/tb_resnet18_block.v" \
    "tb_resnet18_block" "RESNET-18 BLOCK TEST PASSED"

echo ""
echo "═══════════════ REALISTIC MODEL TESTS ═══════════════"
echo "(28×28 LeNet, 56×56 ResNet - using real RTL)"

# Generate test vectors if they don't exist
if [ ! -f "tests/realistic/lenet/test_vectors/layer1_im2col_int8.hex" ]; then
    echo ""
    echo "Generating LeNet test vectors..."
    python3 tests/realistic/lenet/golden.py > /dev/null 2>&1
fi

run_test "LeNet Layer1 Conv (28×28 realistic)" \
    "iverilog -g2012 -o sim/tb_lenet_layer1 tests/realistic/lenet/tb_lenet_layer1_conv.v" \
    "tb_lenet_layer1" "LENET LAYER1 CONV TEST PASSED"

run_test "LeNet Layer3 Pool (24×24 realistic)" \
    "iverilog -g2012 -o sim/tb_lenet_layer3 tests/realistic/lenet/tb_lenet_layer3_pool.v" \
    "tb_lenet_layer3" "LENET LAYER3 POOL TEST PASSED"

# Generate ResNet test vectors if they don't exist
if [ ! -f "tests/realistic/resnet_block/test_vectors/input_activations_int8.hex" ]; then
    echo ""
    echo "Generating ResNet test vectors..."
    python3 tests/realistic/resnet_block/golden.py > /dev/null 2>&1
fi

run_test "ResNet Block (56×56 realistic)" \
    "iverilog -g2012 -o sim/tb_resnet_block tests/realistic/resnet_block/tb_resnet_block.v" \
    "tb_resnet_block" "RESNET BLOCK TEST PASSED"

echo ""
echo "═══════════════ PHASE C: FUSED OPERATIONS ═══════════════"
echo "(GEMM+bias, layer chains, LeNet integration)"

# Generate Phase C test vectors if needed
if [ ! -f "tests/realistic/phase_c/test_vectors/test1_A_int8.hex" ]; then
    echo ""
    echo "Generating Phase C test vectors..."
    python3 tests/realistic/phase_c/golden.py > /dev/null 2>&1
fi

run_test "Bias Fusion (GEMM + VPU ADD)" \
    "iverilog -g2012 -o sim/tb_bias_fusion tests/realistic/phase_c/tb_bias_fusion.v" \
    "tb_bias_fusion" "PHASE_C_BIAS_FUSION TEST PASSED"

run_test "Layer Chain (Conv → ReLU → Pool)" \
    "iverilog -g2012 -o sim/tb_layer_chain tests/realistic/phase_c/tb_layer_chain.v" \
    "tb_layer_chain" "PHASE_C_LAYER_CHAIN TEST PASSED"

run_test "LeNet Chain (Conv1 → ReLU → Pool1)" \
    "iverilog -g2012 -o sim/tb_lenet_chain tests/realistic/phase_c/tb_lenet_chain.v" \
    "tb_lenet_chain" "PHASE_C_LENET_CHAIN TEST PASSED"

echo ""
echo "═══════════════ PHASE D: TRANSFORMER OPS ═══════════════"
echo "(LayerNorm, Softmax, GELU, Attention)"

# Generate Phase D test vectors if needed
if [ ! -f "tests/realistic/phase_d/test_vectors/test1_x_int8.hex" ]; then
    echo ""
    echo "Generating Phase D test vectors..."
    python3 tests/realistic/phase_d/golden.py > /dev/null 2>&1
fi

run_test "LayerNorm (hidden=64)" \
    "iverilog -g2012 -o sim/tb_layernorm tests/realistic/phase_d/tb_layernorm.v" \
    "tb_layernorm" "PHASE_D_LAYERNORM TEST PASSED"

run_test "Softmax (16×16 attention scores)" \
    "iverilog -g2012 -o sim/tb_softmax tests/realistic/phase_d/tb_softmax.v" \
    "tb_softmax" "PHASE_D_SOFTMAX TEST PASSED"

run_test "GELU (256-entry LUT)" \
    "iverilog -g2012 -o sim/tb_gelu tests/realistic/phase_d/tb_gelu.v" \
    "tb_gelu" "PHASE_D_GELU TEST PASSED"

run_test "Attention (Q,K,V 8×16)" \
    "iverilog -g2012 -o sim/tb_attention_phd tests/realistic/phase_d/tb_attention.v" \
    "tb_attention_phd" "PHASE_D_ATTENTION TEST PASSED"

echo ""
echo "═══════════════ PHASE E: STRESS TESTING ═══════════════"
echo "(Back-to-back ops, multi-TPC, boundary conditions)"

# Generate Phase E test vectors if needed
if [ ! -f "tests/realistic/phase_e/test_vectors/test1_A_int8.hex" ]; then
    echo ""
    echo "Generating Phase E test vectors..."
    python3 tests/realistic/phase_e/golden.py > /dev/null 2>&1
fi

run_test "Back-to-Back GEMM (3 stages)" \
    "iverilog -g2012 -o sim/tb_back_to_back tests/realistic/phase_e/tb_back_to_back.v" \
    "tb_back_to_back" "PHASE_E_BACK_TO_BACK TEST PASSED"

run_test "Multi-TPC Parallel (4 TPCs)" \
    "iverilog -g2012 -o sim/tb_multi_tpc_phe tests/realistic/phase_e/tb_multi_tpc.v" \
    "tb_multi_tpc_phe" "PHASE_E_MULTI_TPC TEST PASSED"

run_test "Boundary Conditions (5 subtests)" \
    "iverilog -g2012 -o sim/tb_boundary tests/realistic/phase_e/tb_boundary.v" \
    "tb_boundary" "PHASE_E_BOUNDARY TEST PASSED"

echo ""
echo "═══════════════ PHASE F: FULL MODEL E2E ═══════════════"
echo "(LeNet-5, ResNet Block, Batch Inference)"

# Generate Phase F test vectors if needed
if [ ! -f "tests/realistic/phase_f/test_vectors/test1_input.hex" ]; then
    echo ""
    echo "Generating Phase F test vectors..."
    python3 tests/realistic/phase_f/golden.py > /dev/null 2>&1
fi

run_test "LeNet-5 Full (8 layers)" \
    "iverilog -g2012 -o sim/tb_lenet5_full tests/realistic/phase_f/tb_lenet5_full.v" \
    "tb_lenet5_full" "PHASE_F_LENET5 TEST PASSED"

run_test "ResNet Block Full (residual)" \
    "iverilog -g2012 -o sim/tb_resnet_block_full tests/realistic/phase_f/tb_resnet_block_full.v" \
    "tb_resnet_block_full" "PHASE_F_RESNET_BLOCK TEST PASSED"

run_test "Batch Inference (batch=4)" \
    "iverilog -g2012 -o sim/tb_batch_inf_phf tests/realistic/phase_f/tb_batch_inference.v" \
    "tb_batch_inf_phf" "PHASE_F_BATCH TEST PASSED"

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

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST: DMA Python Model"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
output=$(python3 model/dma_model.py 2>&1)
echo "$output" | grep -E "(TEST|PASS|FAIL|>>>)" | tail -5
if echo "$output" | grep -q "ALL TESTS PASSED"; then
    echo "✓ PASSED"; PASS=$((PASS + 1))
else
    echo "✗ FAILED"; FAIL=$((FAIL + 1))
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST: VPU Python Model"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
output=$(python3 model/vpu_model.py 2>&1)
echo "$output" | grep -E "(TEST|PASS|FAIL|>>>)" | tail -5
if echo "$output" | grep -q "ALL TESTS PASSED"; then
    echo "✓ PASSED"; PASS=$((PASS + 1))
else
    echo "✗ FAILED"; FAIL=$((FAIL + 1))
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST: LCP Python Model"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
output=$(python3 model/lcp_model.py 2>&1)
echo "$output" | grep -E "(TEST|PASS|FAIL|>>>)" | tail -5
if echo "$output" | grep -q "ALL TESTS PASSED"; then
    echo "✓ PASSED"; PASS=$((PASS + 1))
else
    echo "✗ FAILED"; FAIL=$((FAIL + 1))
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST: TPC Integrated Python Model"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
output=$(python3 model/tpc_model.py 2>&1)
echo "$output" | grep -E "(TEST|PASS|FAIL|>>>)" | tail -5
if echo "$output" | grep -q "ALL TESTS PASSED"; then
    echo "✓ PASSED"; PASS=$((PASS + 1))
else
    echo "✗ FAILED"; FAIL=$((FAIL + 1))
fi

echo ""
echo "═══════════════ COMPILER TESTS ═══════════════"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST: Compiler Unit Tests (7 tests)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
output=$(python3 sw/compiler/test_compiler.py 2>&1)
echo "$output" | grep -E "(Test|passed|failed|✓|✗)" | tail -10
if echo "$output" | grep -q "0 failed"; then
    echo "✓ PASSED"; PASS=$((PASS + 1))
else
    echo "✗ FAILED"; FAIL=$((FAIL + 1))
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST: Compiler Integration Tests (3 tests)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
output=$(python3 sw/compiler/test_integration.py 2>&1)
echo "$output" | grep -E "(Test|PASS|FAIL|passed|failed)" | tail -5
if echo "$output" | grep -q "0 failed"; then
    echo "✓ PASSED"; PASS=$((PASS + 1))
else
    echo "✗ FAILED"; FAIL=$((FAIL + 1))
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST: Compiler Extended Ops (LeNet/ResNet/Transformer)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
output=$(python3 sw/compiler/test_extended_ops.py 2>&1)
echo "$output" | grep -E "(Test|PASS|FAIL|passed|failed)" | tail -5
if echo "$output" | grep -q "0 failed"; then
    echo "✓ PASSED"; PASS=$((PASS + 1))
else
    echo "✗ FAILED"; FAIL=$((FAIL + 1))
fi

echo ""
echo "═══════════════ COCOTB TESTS ═══════════════"
echo "(AXI4 protocol verification with cocotb)"

# Check if cocotb is available
if command -v cocotb-config &> /dev/null; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "TEST: cocotb AXI Protocol Verification"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Run cocotb tests
    pushd cocotb/tests > /dev/null 2>&1
    
    # Clean previous runs
    rm -rf sim_build __pycache__ results.xml 2>/dev/null
    
    # Run the tests
    output=$(make SIM=icarus 2>&1)
    
    # Extract results
    echo "$output" | grep -E "(PASS|FAIL|test_axi|Coverage)" | tail -30
    
    # Check for success
    if echo "$output" | grep -q "TESTS=.*PASS=.*FAIL=0"; then
        cocotb_pass=$(echo "$output" | grep -oP 'PASS=\K[0-9]+' | tail -1)
        echo ""
        echo "✓ PASSED ($cocotb_pass cocotb tests)"
        PASS=$((PASS + 1))
    else
        cocotb_fail=$(echo "$output" | grep -oP 'FAIL=\K[0-9]+' | tail -1)
        echo ""
        echo "✗ FAILED ($cocotb_fail cocotb tests failed)"
        FAIL=$((FAIL + 1))
    fi
    
    popd > /dev/null 2>&1
else
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "SKIPPED: cocotb not installed"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    TEST SUMMARY                            ║"
echo "╠════════════════════════════════════════════════════════════╣"
printf "║   Total: %-3d passed, %-3d failed                           ║\n" $PASS $FAIL
echo "╚════════════════════════════════════════════════════════════╝"

##########
iverilog -g2012 -I rtl/include -o sim/tb_sys_comp rtl/core/mac_pe.v rtl/core/systolic_array.v tb/tb_systolic_comprehensive.v && vvp sim/tb_sys_comp
python3 sw/compiler/test_extended_ops.py
python3 sw/compiler/test_compiler.py
python3 sw/compiler/test_new_ops.py
##########

if [ $FAIL -eq 0 ]; then 
    echo ">>> ALL TESTS PASSED! <<<"
    exit 0
else 
    echo ">>> SOME TESTS FAILED <<<"
    exit 1
fi

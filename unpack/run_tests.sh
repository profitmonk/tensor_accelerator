#!/bin/bash
# Tensor Accelerator - Complete Test Suite
# Compatible with both Linux and macOS

cd "$(dirname "$0")"
PASS=0
FAIL=0

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║        Tensor Accelerator - Test Suite                     ║"
echo "╚════════════════════════════════════════════════════════════╝"

mkdir -p sim

run_test() {
    name="$1"
    compile_cmd="$2"
    sim_exe="$3"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "TEST: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Compile
    if eval "$compile_cmd" 2>/dev/null; then
        # Run simulation
        output=$(cd sim && vvp "$sim_exe" 2>&1)
        echo "$output" | tail -15
        
        # Check for pass/fail
        if echo "$output" | grep -q "ALL TESTS PASSED"; then
            echo "✓ PASSED"
            PASS=$((PASS + 1))
        else
            echo "✗ FAILED"
            FAIL=$((FAIL + 1))
        fi
    else
        echo "✗ COMPILE ERROR"
        FAIL=$((FAIL + 1))
    fi
}

# Core Unit Tests
run_test "MAC PE" \
    "iverilog -o sim/tb_mac rtl/core/mac_pe.v tb/tb_mac_pe.v" \
    "tb_mac"

run_test "Systolic Array" \
    "iverilog -o sim/tb_sys rtl/core/mac_pe.v rtl/core/systolic_array.v tb/tb_systolic_array.v" \
    "tb_sys"

run_test "Vector Unit" \
    "iverilog -g2012 -o sim/tb_vpu rtl/core/vector_unit.v tb/tb_vector_unit.v" \
    "tb_vpu"

# Control Unit Tests
run_test "Local Command Processor" \
    "iverilog -g2012 -o sim/tb_lcp rtl/control/local_cmd_processor.v tb/tb_local_cmd_processor.v" \
    "tb_lcp"

# Memory Unit Tests
run_test "SRAM Subsystem" \
    "iverilog -g2012 -DSIM -o sim/tb_sram rtl/memory/sram_subsystem.v tb/tb_sram_subsystem.v" \
    "tb_sram"

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    TEST SUMMARY                            ║"
echo "╠════════════════════════════════════════════════════════════╣"
printf "║   Passed: %-3d                                             ║\n" $PASS
printf "║   Failed: %-3d                                             ║\n" $FAIL
echo "╚════════════════════════════════════════════════════════════╝"

if [ $FAIL -eq 0 ]; then
    echo ">>> ALL TESTS PASSED! <<<"
    exit 0
else
    echo ">>> SOME TESTS FAILED <<<"
    exit 1
fi

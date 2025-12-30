#!/usr/bin/env python3
"""
VPU Integration Tests - Python Model

Tests the complete GEMM → VPU → SRAM datapath.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from tpc_model import (TPCModel, make_mxu_instr, make_vpu_instr, 
                       make_dma_instr, make_halt)
from vpu_model import VPUOp
from dma_model import DMAOp


def test_gemm_relu_basic():
    """Test 1: Basic GEMM → ReLU flow with known values"""
    print("\n" + "="*70)
    print("TEST 1: GEMM → ReLU (Basic 2x2)")
    print("="*70)
    
    tpc = TPCModel(verbose=False)
    
    # Design matrices to produce both positive and negative results
    A = np.array([[1, 2], [3, 4]], dtype=np.int8)
    B = np.array([[1, -1], [-1, 1]], dtype=np.int8)
    
    expected_gemm = np.dot(A.astype(np.int32), B.astype(np.int32)).astype(np.int8)
    expected_relu = np.maximum(expected_gemm, 0).astype(np.int8)
    
    print(f"A:\n{A}")
    print(f"B:\n{B}")
    print(f"Expected GEMM (C = A @ B):\n{expected_gemm}")
    print(f"Expected ReLU(C):\n{expected_relu}")
    
    tpc.load_sram(0x000, B)
    tpc.load_sram(0x100, A)
    
    program = [
        make_mxu_instr(weight_addr=0x000, act_addr=0x100, out_addr=0x200, M=2, K=2, N=2),
        make_vpu_instr(VPUOp.RELU, src_a=0x200, src_b=0, dst=0x300, length=1),
        make_halt()
    ]
    tpc.load_program(program)
    
    assert tpc.run(max_cycles=200), "Timeout!"
    
    gemm_result = tpc.read_sram(0x200, 4).reshape(2, 2)
    relu_result = tpc.read_sram(0x300, 4).reshape(2, 2)
    
    print(f"\nActual GEMM result:\n{gemm_result}")
    print(f"Actual ReLU result:\n{relu_result}")
    
    assert np.array_equal(gemm_result, expected_gemm), f"GEMM mismatch!"
    assert np.array_equal(relu_result, expected_relu), f"ReLU mismatch!"
    
    print("\n>>> TEST 1 PASSED <<<")
    return True


def test_gemm_relu_with_negatives():
    """Test 2: GEMM → ReLU ensuring negative handling"""
    print("\n" + "="*70)
    print("TEST 2: GEMM → ReLU (Negative Value Handling)")
    print("="*70)
    
    tpc = TPCModel(verbose=False)
    
    # Designed to produce mix of positives and negatives
    A = np.array([[5, -3], [-2, 4]], dtype=np.int8)
    B = np.array([[1, 2], [3, 4]], dtype=np.int8)
    
    expected_gemm = np.dot(A.astype(np.int32), B.astype(np.int32))
    expected_gemm = np.clip(expected_gemm, -128, 127).astype(np.int8)
    expected_relu = np.maximum(expected_gemm, 0).astype(np.int8)
    
    print(f"A:\n{A}")
    print(f"B:\n{B}")
    print(f"Expected GEMM:\n{expected_gemm}")
    print(f"Expected ReLU:\n{expected_relu}")
    
    num_negatives = np.sum(expected_gemm < 0)
    print(f"Number of negative GEMM outputs: {num_negatives}")
    
    tpc.load_sram(0x000, B)
    tpc.load_sram(0x100, A)
    
    program = [
        make_mxu_instr(0x000, 0x100, 0x200, M=2, K=2, N=2),
        make_vpu_instr(VPUOp.RELU, src_a=0x200, src_b=0, dst=0x300, length=1),
        make_halt()
    ]
    tpc.load_program(program)
    
    assert tpc.run(max_cycles=200), "Timeout!"
    
    gemm_result = tpc.read_sram(0x200, 4).reshape(2, 2)
    relu_result = tpc.read_sram(0x300, 4).reshape(2, 2)
    
    print(f"\nActual GEMM:\n{gemm_result}")
    print(f"Actual ReLU:\n{relu_result}")
    
    assert np.array_equal(gemm_result, expected_gemm), "GEMM mismatch!"
    assert np.array_equal(relu_result, expected_relu), "ReLU mismatch!"
    assert np.all(relu_result >= 0), "ReLU has negatives!"
    
    print("\n>>> TEST 2 PASSED <<<")
    return True


def test_gemm_add_bias():
    """Test 3: GEMM → Add bias vector"""
    print("\n" + "="*70)
    print("TEST 3: GEMM → Add Bias")
    print("="*70)
    
    tpc = TPCModel(verbose=False)
    
    A = np.array([[1, 0], [0, 1]], dtype=np.int8)
    B = np.array([[5, 6], [7, 8]], dtype=np.int8)
    bias = np.array([[10, 10], [10, 10]], dtype=np.int8)
    
    expected_gemm = np.dot(A.astype(np.int32), B.astype(np.int32)).astype(np.int8)
    expected_add = np.clip(expected_gemm.astype(np.int16) + bias.astype(np.int16), 
                          -128, 127).astype(np.int8)
    
    print(f"A:\n{A}")
    print(f"B:\n{B}")
    print(f"Bias:\n{bias}")
    print(f"Expected GEMM + Bias:\n{expected_add}")
    
    tpc.load_sram(0x000, B)
    tpc.load_sram(0x100, A)
    tpc.load_sram(0x400, bias)
    
    program = [
        make_mxu_instr(0x000, 0x100, 0x200, M=2, K=2, N=2),
        make_vpu_instr(VPUOp.ADD, src_a=0x200, src_b=0x400, dst=0x300, length=1),
        make_halt()
    ]
    tpc.load_program(program)
    
    assert tpc.run(max_cycles=200), "Timeout!"
    
    add_result = tpc.read_sram(0x300, 4).reshape(2, 2)
    
    print(f"\nActual GEMM + Bias:\n{add_result}")
    
    assert np.array_equal(add_result, expected_add), "Add mismatch!"
    
    print("\n>>> TEST 3 PASSED <<<")
    return True


def test_gemm_relu_chain():
    """Test 4: Chained GEMM → ReLU → ReLU"""
    print("\n" + "="*70)
    print("TEST 4: GEMM → ReLU → ReLU (chain)")
    print("="*70)
    
    tpc = TPCModel(verbose=False)
    
    A = np.array([[2, -1], [-1, 2]], dtype=np.int8)
    B = np.array([[1, 1], [1, 1]], dtype=np.int8)
    
    expected_gemm = np.dot(A.astype(np.int32), B.astype(np.int32)).astype(np.int8)
    expected_relu = np.maximum(expected_gemm, 0).astype(np.int8)
    
    print(f"A:\n{A}")
    print(f"B:\n{B}")
    print(f"GEMM:\n{expected_gemm}")
    print(f"ReLU:\n{expected_relu}")
    
    tpc.load_sram(0x000, B)
    tpc.load_sram(0x100, A)
    
    program = [
        make_mxu_instr(0x000, 0x100, 0x200, M=2, K=2, N=2),
        make_vpu_instr(VPUOp.RELU, src_a=0x200, src_b=0, dst=0x300, length=1),
        make_vpu_instr(VPUOp.RELU, src_a=0x300, src_b=0, dst=0x400, length=1),
        make_halt()
    ]
    tpc.load_program(program)
    
    assert tpc.run(max_cycles=300), "Timeout!"
    
    relu1 = tpc.read_sram(0x300, 4).reshape(2, 2)
    relu2 = tpc.read_sram(0x400, 4).reshape(2, 2)
    
    print(f"\nActual ReLU1:\n{relu1}")
    print(f"Actual ReLU2:\n{relu2}")
    
    assert np.array_equal(relu1, expected_relu), "ReLU1 mismatch!"
    assert np.array_equal(relu2, expected_relu), "ReLU2 mismatch!"
    
    print("\n>>> TEST 4 PASSED <<<")
    return True


def test_full_pipeline_with_dma():
    """Test 5: Full pipeline - DMA LOAD → GEMM → ReLU → DMA STORE"""
    print("\n" + "="*70)
    print("TEST 5: DMA LOAD → GEMM → ReLU → DMA STORE (Full Pipeline)")
    print("="*70)
    
    tpc = TPCModel(verbose=False)
    
    A = np.array([[3, -2], [-1, 4]], dtype=np.int8)
    B = np.array([[1, 2], [3, 4]], dtype=np.int8)
    
    expected_gemm = np.dot(A.astype(np.int32), B.astype(np.int32)).astype(np.int8)
    expected_relu = np.maximum(expected_gemm, 0).astype(np.int8)
    
    print(f"A:\n{A}")
    print(f"B:\n{B}")
    print(f"Expected GEMM:\n{expected_gemm}")
    print(f"Expected ReLU:\n{expected_relu}")
    
    # Load into external memory
    tpc.axi_mem.mem[0] = tpc._pack(B.flatten())
    tpc.axi_mem.mem[1] = tpc._pack(A.flatten())
    
    program = [
        make_dma_instr(DMAOp.LOAD, ext_addr=0x000, int_addr=0x000, rows=1, cols=1),
        make_dma_instr(DMAOp.LOAD, ext_addr=0x020, int_addr=0x100, rows=1, cols=1),
        make_mxu_instr(0x000, 0x100, 0x200, M=2, K=2, N=2),
        make_vpu_instr(VPUOp.RELU, src_a=0x200, src_b=0, dst=0x300, length=1),
        make_dma_instr(DMAOp.STORE, ext_addr=0x100, int_addr=0x300, rows=1, cols=1),
        make_halt()
    ]
    tpc.load_program(program)
    
    assert tpc.run(max_cycles=500), "Timeout!"
    
    result_packed = int(tpc.axi_mem.mem[0x100 >> 5])
    result = tpc._unpack(result_packed)[:4].reshape(2, 2)
    
    print(f"\nResult from external memory:\n{result}")
    print(f"Expected:\n{expected_relu}")
    
    assert np.array_equal(result, expected_relu), "Full pipeline mismatch!"
    assert np.all(result >= 0), "Output has negative values!"
    
    print("\n>>> TEST 5 PASSED <<<")
    return True


def main():
    print("="*70)
    print("VPU INTEGRATION TESTS - PYTHON MODEL")
    print("="*70)
    
    tests = [
        ("GEMM → ReLU (Basic)", test_gemm_relu_basic),
        ("GEMM → ReLU (Negative Handling)", test_gemm_relu_with_negatives),
        ("GEMM → Add Bias", test_gemm_add_bias),
        ("GEMM → ReLU Chain", test_gemm_relu_chain),
        ("Full Pipeline (DMA → GEMM → ReLU → DMA)", test_full_pipeline_with_dma),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"\n!!! TEST FAILED: {name}")
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print(f"VPU INTEGRATION SUMMARY: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed == 0:
        print(">>> ALL VPU INTEGRATION TESTS PASSED! <<<")
        return 0
    else:
        print(">>> SOME TESTS FAILED <<<")
        return 1


if __name__ == "__main__":
    sys.exit(main())

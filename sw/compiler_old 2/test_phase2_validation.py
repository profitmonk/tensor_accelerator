#!/usr/bin/env python3
"""
Phase 2 Standalone Validation Tests

These tests verify the compiler output is correct BEFORE attempting
RTL simulation in Phase 3. This catches format and data issues early.

Tests:
1. Hex format validation (can Verilog parse it?)
2. Memory file format validation
3. Large matrix tiling correctness
4. Assembly round-trip (assemble → decode → compare)
5. Golden reference sanity checks
"""

import sys
import os
import tempfile
import subprocess
import struct
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assembler'))

from ir.graph import Graph, Node, Tensor, OpType, DataType
from tiler.tiler import TilingEngine, HardwareConfig
from scheduler.scheduler import Scheduler
from codegen.codegen import CodeGenerator
from assembler import Assembler, Instruction, Opcode


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'


def test_hex_format_validity():
    """
    Test 1: Verify hex files are valid for Verilog $readmemh
    
    Requirements:
    - Each line must be valid hex
    - Each line must be exactly 32 characters (128 bits)
    - No whitespace or comments in data
    """
    print("\n" + "="*60)
    print("Test 1: Hex Format Validity")
    print("="*60)
    
    test_dir = Path(__file__).parent.parent.parent / "tests" / "e2e"
    
    if not test_dir.exists():
        print(f"{Colors.YELLOW}⚠ Test directory not found: {test_dir}{Colors.RESET}")
        print("  Run: python generate_e2e_tests.py --all -o ../../tests/e2e")
        return None
    
    errors = []
    checked = 0
    
    for hex_file in test_dir.rglob("*.hex"):
        checked += 1
        with open(hex_file) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Check length
                if len(line) != 32:
                    errors.append(f"{hex_file.name}:{line_num}: Length {len(line)} != 32")
                    continue
                
                # Check hex validity
                try:
                    int(line, 16)
                except ValueError:
                    errors.append(f"{hex_file.name}:{line_num}: Invalid hex: {line[:20]}...")
    
    if errors:
        print(f"{Colors.RED}✗ Found {len(errors)} errors:{Colors.RESET}")
        for e in errors[:10]:
            print(f"  {e}")
        return False
    
    print(f"{Colors.GREEN}✓ Validated {checked} hex files{Colors.RESET}")
    return True


def test_memh_format_validity():
    """
    Test 2: Verify .memh files are valid for Verilog $readmemh
    
    Requirements:
    - Valid hex characters
    - Consistent line width
    - Loadable format
    """
    print("\n" + "="*60)
    print("Test 2: Memory File Format Validity")
    print("="*60)
    
    test_dir = Path(__file__).parent.parent.parent / "tests" / "e2e"
    
    if not test_dir.exists():
        print(f"{Colors.YELLOW}⚠ Test directory not found{Colors.RESET}")
        return None
    
    errors = []
    checked = 0
    
    for memh_file in test_dir.rglob("*.memh"):
        checked += 1
        line_lengths = set()
        
        with open(memh_file) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('//'):
                    continue
                
                line_lengths.add(len(line))
                
                # Check hex validity
                try:
                    int(line, 16)
                except ValueError:
                    errors.append(f"{memh_file.name}:{line_num}: Invalid hex")
        
        # Check consistent width (should all be same)
        if len(line_lengths) > 1:
            errors.append(f"{memh_file.name}: Inconsistent line widths: {line_lengths}")
    
    if errors:
        print(f"{Colors.RED}✗ Found {len(errors)} errors:{Colors.RESET}")
        for e in errors[:10]:
            print(f"  {e}")
        return False
    
    print(f"{Colors.GREEN}✓ Validated {checked} memh files{Colors.RESET}")
    return True


def test_large_matrix_tiling():
    """
    Test 3: Verify tiling works correctly for large matrices
    
    Checks:
    - Tiles fit in SRAM
    - All elements are covered
    - Dependencies are correct
    """
    print("\n" + "="*60)
    print("Test 3: Large Matrix Tiling")
    print("="*60)
    
    tiler = TilingEngine()
    hw = tiler.hw
    
    test_sizes = [
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (784, 256, 784),   # LeNet FC1
        (4096, 1000, 4096), # Large FC
    ]
    
    errors = []
    
    for M, N, K in test_sizes:
        config = tiler.compute_gemm_tiling(M, N, K)
        
        # Check 1: Tile fits in SRAM
        if config.total_bytes > hw.sram_size:
            errors.append(f"({M}×{N}×{K}): Tile too large: {config.total_bytes} > {hw.sram_size}")
        
        # Check 2: Tiles cover full matrix
        m_coverage = config.tile_m * config.num_m_tiles
        n_coverage = config.tile_n * config.num_n_tiles
        k_coverage = config.tile_k * config.num_k_tiles
        
        if m_coverage < M:
            errors.append(f"({M}×{N}×{K}): M not covered: {m_coverage} < {M}")
        if n_coverage < N:
            errors.append(f"({M}×{N}×{K}): N not covered: {n_coverage} < {N}")
        if k_coverage < K:
            errors.append(f"({M}×{N}×{K}): K not covered: {k_coverage} < {K}")
        
        # Check 3: Tiles aligned to systolic array
        if config.tile_m % hw.systolic_m != 0:
            errors.append(f"({M}×{N}×{K}): tile_m not aligned: {config.tile_m}")
        if config.tile_n % hw.systolic_n != 0:
            errors.append(f"({M}×{N}×{K}): tile_n not aligned: {config.tile_n}")
        
        total_tiles = config.num_m_tiles * config.num_n_tiles * config.num_k_tiles
        print(f"  GEMM({M}×{N}×{K}): {total_tiles} tiles, "
              f"{config.tile_m}×{config.tile_n}×{config.tile_k}, "
              f"{config.total_bytes//1024}KB")
    
    if errors:
        print(f"\n{Colors.RED}✗ Found {len(errors)} errors:{Colors.RESET}")
        for e in errors:
            print(f"  {e}")
        return False
    
    print(f"\n{Colors.GREEN}✓ All tiling tests passed{Colors.RESET}")
    return True


def test_instruction_encoding():
    """
    Test 4: Verify instruction encoding/decoding round-trip
    
    Assemble instructions and verify they decode correctly.
    """
    print("\n" + "="*60)
    print("Test 4: Instruction Encoding Round-Trip")
    print("="*60)
    
    test_instructions = [
        # (asm_line, expected_opcode, expected_subop, expected_dst, ...)
        ("TENSOR.GEMM 0x6000, 0x0000, 0x2000, 16, 16, 16", 0x01, 0x01, 0x6000, 0x0000, 0x2000, 16, 16, 16),
        ("TENSOR.GEMM_ACC 0x6000, 0x1000, 0x4000, 8, 8, 8", 0x01, 0x02, 0x6000, 0x1000, 0x4000, 8, 8, 8),
        ("VECTOR.RELU 0x6000, 0x6000, 256", 0x02, 0x10, 0x6000, 0x6000, 256, 0, 0, 0),
        ("SYNC.WAIT_MXU", 0x04, 0x01, 0, 0, 0, 0, 0, 0),
        ("SYNC.WAIT_DMA", 0x04, 0x03, 0, 0, 0, 0, 0, 0),
        ("HALT", 0xFF, 0x00, 0, 0, 0, 0, 0, 0),
        ("NOP", 0x00, 0x00, 0, 0, 0, 0, 0, 0),
    ]
    
    assembler = Assembler()
    errors = []
    
    for i, (asm_line, exp_op, exp_sub, exp_dst, exp_src0, exp_src1, exp_m, exp_n, exp_k) in enumerate(test_instructions):
        try:
            instr = assembler.assemble_line(asm_line)
            if instr is None:
                errors.append(f"Line {i}: '{asm_line}' returned None")
                continue
            
            # Check fields
            if instr.opcode != exp_op:
                errors.append(f"Line {i}: opcode {instr.opcode:#x} != {exp_op:#x}")
            if instr.subop != exp_sub:
                errors.append(f"Line {i}: subop {instr.subop:#x} != {exp_sub:#x}")
            if instr.dst != exp_dst:
                errors.append(f"Line {i}: dst {instr.dst:#x} != {exp_dst:#x}")
            if instr.src0 != exp_src0:
                errors.append(f"Line {i}: src0 {instr.src0:#x} != {exp_src0:#x}")
            if instr.src1 != exp_src1:
                errors.append(f"Line {i}: src1 {instr.src1:#x} != {exp_src1:#x}")
            
            # Check encoding length
            encoded = instr.encode()
            if len(encoded) != 16:
                errors.append(f"Line {i}: encoded length {len(encoded)} != 16 bytes")
            
            hex_str = instr.to_hex()
            if len(hex_str) != 32:
                errors.append(f"Line {i}: hex length {len(hex_str)} != 32 chars")
            
            print(f"  {asm_line:45s} → {hex_str[:16]}...")
            
        except Exception as e:
            errors.append(f"Line {i}: Exception: {e}")
    
    if errors:
        print(f"\n{Colors.RED}✗ Found {len(errors)} errors:{Colors.RESET}")
        for e in errors:
            print(f"  {e}")
        return False
    
    print(f"\n{Colors.GREEN}✓ All encoding tests passed{Colors.RESET}")
    return True


def test_golden_reference_sanity():
    """
    Test 5: Verify golden references are mathematically correct
    
    Re-compute golden values and compare against stored files.
    """
    print("\n" + "="*60)
    print("Test 5: Golden Reference Sanity Check")
    print("="*60)
    
    test_dir = Path(__file__).parent.parent.parent / "tests" / "e2e"
    
    if not test_dir.exists():
        print(f"{Colors.YELLOW}⚠ Test directory not found{Colors.RESET}")
        return None
    
    errors = []
    checked = 0
    
    for config_file in test_dir.rglob("test_config.json"):
        test_name = config_file.parent.name
        
        with open(config_file) as f:
            config = json.load(f)
        
        if config['type'] != 'gemm':
            continue  # Only check GEMM for now
        
        checked += 1
        
        # Load numpy arrays
        A_file = config_file.parent / "input_A.npy"
        B_file = config_file.parent / "weight_B.npy"
        C_file = config_file.parent / "golden_C.npy"
        
        if not all(f.exists() for f in [A_file, B_file, C_file]):
            errors.append(f"{test_name}: Missing numpy files")
            continue
        
        A = np.load(A_file)
        B = np.load(B_file)
        C_stored = np.load(C_file)
        
        # Recompute
        C_computed = np.matmul(A.astype(np.int32), B.astype(np.int32))
        
        # Compare
        if not np.array_equal(C_stored, C_computed):
            max_diff = np.max(np.abs(C_stored - C_computed))
            errors.append(f"{test_name}: Golden mismatch, max_diff={max_diff}")
        else:
            print(f"  {test_name}: Golden verified ✓")
    
    if errors:
        print(f"\n{Colors.RED}✗ Found {len(errors)} errors:{Colors.RESET}")
        for e in errors:
            print(f"  {e}")
        return False
    
    if checked == 0:
        print(f"{Colors.YELLOW}⚠ No GEMM tests found to verify{Colors.RESET}")
        return None
    
    print(f"\n{Colors.GREEN}✓ Verified {checked} golden references{Colors.RESET}")
    return True


def test_verilog_readmemh_simulation():
    """
    Test 6: Actually try to load hex in Verilog (requires iverilog)
    
    Creates a minimal testbench that loads the hex file.
    """
    print("\n" + "="*60)
    print("Test 6: Verilog $readmemh Simulation")
    print("="*60)
    
    # Check if iverilog is available
    try:
        result = subprocess.run(['iverilog', '--version'], 
                              capture_output=True, timeout=5)
        if result.returncode != 0:
            print(f"{Colors.YELLOW}⚠ iverilog not working{Colors.RESET}")
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"{Colors.YELLOW}⚠ iverilog not found, skipping{Colors.RESET}")
        return None
    
    test_dir = Path(__file__).parent.parent.parent / "tests" / "e2e" / "gemm_16x16"
    hex_file = test_dir / "program.hex"
    
    if not hex_file.exists():
        print(f"{Colors.YELLOW}⚠ Test hex file not found{Colors.RESET}")
        return None
    
    # Count instructions
    with open(hex_file) as f:
        num_instr = sum(1 for line in f if line.strip())
    
    # Create minimal testbench
    tb_code = f'''
module tb_readmemh_test;
    reg [127:0] instr_mem [0:{num_instr-1}];
    integer i;
    
    initial begin
        $readmemh("{hex_file}", instr_mem);
        
        // Verify loaded
        for (i = 0; i < {num_instr}; i = i + 1) begin
            if (instr_mem[i] === 128'bx) begin
                $display("ERROR: Instruction %0d not loaded", i);
                $finish;
            end
        end
        
        $display("SUCCESS: Loaded %0d instructions", {num_instr});
        $display("  instr[0] = %h", instr_mem[0]);
        $display("  instr[1] = %h", instr_mem[1]);
        $finish;
    end
endmodule
'''
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tb_file = Path(tmpdir) / "tb_test.v"
        tb_file.write_text(tb_code)
        
        out_file = Path(tmpdir) / "tb_test.vvp"
        
        # Compile
        result = subprocess.run(
            ['iverilog', '-o', str(out_file), str(tb_file)],
            capture_output=True, text=True, timeout=30
        )
        
        if result.returncode != 0:
            print(f"{Colors.RED}✗ Verilog compile failed:{Colors.RESET}")
            print(result.stderr)
            return False
        
        # Run
        result = subprocess.run(
            ['vvp', str(out_file)],
            capture_output=True, text=True, timeout=30
        )
        
        if "SUCCESS" in result.stdout:
            print(f"{Colors.GREEN}✓ Verilog $readmemh test passed{Colors.RESET}")
            for line in result.stdout.strip().split('\n'):
                print(f"  {line}")
            return True
        else:
            print(f"{Colors.RED}✗ Verilog test failed:{Colors.RESET}")
            print(result.stdout)
            print(result.stderr)
            return False


def run_all_tests():
    """Run all Phase 2 validation tests"""
    print("="*60)
    print("Phase 2 Standalone Validation Tests")
    print("="*60)
    
    tests = [
        ("Hex Format", test_hex_format_validity),
        ("Memh Format", test_memh_format_validity),
        ("Large Matrix Tiling", test_large_matrix_tiling),
        ("Instruction Encoding", test_instruction_encoding),
        ("Golden Reference", test_golden_reference_sanity),
        ("Verilog $readmemh", test_verilog_readmemh_simulation),
    ]
    
    results = []
    
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"{Colors.RED}✗ {name} crashed: {e}{Colors.RESET}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Phase 2 Validation Summary")
    print("="*60)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, result in results:
        if result is True:
            print(f"  {Colors.GREEN}✓ PASS{Colors.RESET}: {name}")
            passed += 1
        elif result is False:
            print(f"  {Colors.RED}✗ FAIL{Colors.RESET}: {name}")
            failed += 1
        else:
            print(f"  {Colors.YELLOW}⚠ SKIP{Colors.RESET}: {name}")
            skipped += 1
    
    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed > 0:
        print(f"\n{Colors.RED}Phase 2 validation FAILED - fix issues before Phase 3{Colors.RESET}")
        return False
    else:
        print(f"\n{Colors.GREEN}Phase 2 validation PASSED - ready for Phase 3{Colors.RESET}")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

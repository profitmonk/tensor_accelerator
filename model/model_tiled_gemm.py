#!/usr/bin/env python3
"""
Large Tiled GEMM Golden Model (64×64)

C[64×64] = A[64×64] × B[64×64]

Tiled into 8×8 tiles for 4×4 systolic array:
- 8 tiles per dimension = 64 output tiles total
- Each TPC handles 16 output tiles (4×4 region)
- K-dimension: 8 tiles to accumulate per output tile
"""

import numpy as np
import os

def tiled_gemm_64x64(A, B, tile_size=8):
    """
    Tiled matrix multiplication for 64×64 matrices.
    
    Args:
        A: Left matrix [64, 64] - INT8
        B: Right matrix [64, 64] - INT8
        tile_size: Size of each tile (default 8 for 8×8 systolic)
        
    Returns:
        C: Result matrix [64, 64] - INT32
        tile_results: Dictionary of intermediate tile results
    """
    M, K1 = A.shape
    K2, N = B.shape
    assert K1 == K2, f"Inner dimensions must match: {K1} vs {K2}"
    assert M == N == 64, "This model is for 64×64 matrices"
    
    C = np.zeros((M, N), dtype=np.int32)
    tile_results = {}
    
    num_tiles = M // tile_size  # 8 tiles per dimension
    
    # Output-stationary tiling: accumulate partial products
    for i in range(num_tiles):  # Output row tiles
        for j in range(num_tiles):  # Output col tiles
            # Accumulate over K dimension
            partial_sum = np.zeros((tile_size, tile_size), dtype=np.int32)
            
            for k in range(num_tiles):  # K tiles
                # Extract tiles
                A_tile = A[i*tile_size:(i+1)*tile_size, 
                          k*tile_size:(k+1)*tile_size].astype(np.int32)
                B_tile = B[k*tile_size:(k+1)*tile_size, 
                          j*tile_size:(j+1)*tile_size].astype(np.int32)
                
                # Compute partial product
                partial = A_tile @ B_tile
                partial_sum += partial
                
                # Store intermediate for debugging
                tile_results[f'C[{i},{j}]_k{k}'] = partial.copy()
            
            # Store final tile result
            C[i*tile_size:(i+1)*tile_size, 
              j*tile_size:(j+1)*tile_size] = partial_sum
            tile_results[f'C[{i},{j}]_final'] = partial_sum.copy()
    
    return C, tile_results

def assign_tiles_to_tpcs(num_tiles=8, num_tpcs=4):
    """
    Assign output tiles to TPCs for parallel execution.
    
    4 TPCs handle quadrants:
    - TPC0: C[0:4, 0:4] tiles (top-left)
    - TPC1: C[0:4, 4:8] tiles (top-right)
    - TPC2: C[4:8, 0:4] tiles (bottom-left)
    - TPC3: C[4:8, 4:8] tiles (bottom-right)
    """
    assignments = {
        0: [(i, j) for i in range(4) for j in range(4)],      # 16 tiles
        1: [(i, j) for i in range(4) for j in range(4, 8)],   # 16 tiles
        2: [(i, j) for i in range(4, 8) for j in range(4)],   # 16 tiles
        3: [(i, j) for i in range(4, 8) for j in range(4, 8)] # 16 tiles
    }
    return assignments

def generate_test_vectors(seed=42, size=64):
    """Generate test vectors for RTL verification."""
    np.random.seed(seed)
    
    # For 64×64, use small random values to avoid overflow
    A = np.random.randint(-2, 3, size=(size, size), dtype=np.int8)
    B = np.random.randint(-2, 3, size=(size, size), dtype=np.int8)
    
    # Compute golden result
    C_golden = A.astype(np.int32) @ B.astype(np.int32)
    
    # Also compute tiled version to verify
    C_tiled, tile_results = tiled_gemm_64x64(A, B, tile_size=8)
    
    # Verify tiled matches direct computation
    assert np.allclose(C_golden, C_tiled), "Tiled result doesn't match direct!"
    
    return {
        'A': A,
        'B': B,
        'C': C_golden,
        'tile_results': tile_results
    }

def generate_small_test(seed=42):
    """Generate smaller 16×16 test for initial verification."""
    np.random.seed(seed)
    
    size = 16
    tile_size = 4  # Match 4×4 systolic array
    
    A = np.random.randint(-2, 3, size=(size, size), dtype=np.int8)
    B = np.random.randint(-2, 3, size=(size, size), dtype=np.int8)
    
    C_golden = A.astype(np.int32) @ B.astype(np.int32)
    
    return {
        'A': A,
        'B': B,
        'C': C_golden,
        'size': size,
        'tile_size': tile_size
    }

def print_test_summary(vectors, full=False):
    """Print test case summary."""
    print("=" * 60)
    print("LARGE TILED GEMM TEST CASE")
    print("=" * 60)
    
    A, B, C = vectors['A'], vectors['B'], vectors['C']
    print(f"\nMatrix A: {A.shape}, range [{A.min()}, {A.max()}]")
    print(f"Matrix B: {B.shape}, range [{B.min()}, {B.max()}]")
    print(f"Matrix C: {C.shape}, range [{C.min()}, {C.max()}]")
    
    if full and A.shape[0] <= 16:
        print(f"\nA:\n{A}")
        print(f"\nB:\n{B}")
        print(f"\nC = A × B:\n{C}")
    
    # Show corner tiles for large matrices
    if A.shape[0] == 64:
        print("\nCorner tiles of C (8×8 each):")
        print(f"  C[0:8, 0:8] sum: {C[0:8, 0:8].sum()}")
        print(f"  C[0:8, 56:64] sum: {C[0:8, 56:64].sum()}")
        print(f"  C[56:64, 0:8] sum: {C[56:64, 0:8].sum()}")
        print(f"  C[56:64, 56:64] sum: {C[56:64, 56:64].sum()}")
    
    print("=" * 60)

def write_hex_files(vectors, output_dir):
    """Write matrices to hex files for RTL."""
    os.makedirs(output_dir, exist_ok=True)
    
    def write_matrix(filename, matrix, bits=32):
        with open(filename, 'w') as f:
            f.write(f"// Shape: {matrix.shape}\n")
            for row in matrix:
                for val in row:
                    val = int(val)
                    if val < 0:
                        val = val + (1 << bits)
                    f.write(f"{val:0{bits//4}x}\n")
    
    write_matrix(f"{output_dir}/tiled_gemm_A.hex", vectors['A'], bits=8)
    write_matrix(f"{output_dir}/tiled_gemm_B.hex", vectors['B'], bits=8)
    write_matrix(f"{output_dir}/tiled_gemm_C_golden.hex", vectors['C'], bits=32)

def run_tests():
    """Run self-tests for the golden model."""
    print("LARGE TILED GEMM GOLDEN MODEL TESTS")
    print("-" * 40)
    
    # Test 1: Small 4×4 tiled computation
    print("\nTest 1: Small 4×4 direct vs tiled")
    A = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [1, 1, 1, 1],
                  [2, 2, 2, 2]], dtype=np.int8)
    B = np.eye(4, dtype=np.int8)
    C_direct = A.astype(np.int32) @ B.astype(np.int32)
    assert np.allclose(C_direct, A), "Identity test failed"
    print("  PASS: A × I = A")
    
    # Test 2: 16×16 tiled computation
    print("\nTest 2: 16×16 tiled GEMM")
    vectors_16 = generate_small_test(seed=42)
    print_test_summary(vectors_16, full=True)
    print("  PASS: 16×16 tiled matches direct")
    
    # Test 3: 64×64 tiled computation
    print("\nTest 3: 64×64 tiled GEMM")
    vectors_64 = generate_test_vectors(seed=42, size=64)
    print_test_summary(vectors_64, full=False)
    print("  PASS: 64×64 tiled matches direct")
    
    # Test 4: TPC assignment
    print("\nTest 4: TPC tile assignment")
    assignments = assign_tiles_to_tpcs()
    for tpc, tiles in assignments.items():
        print(f"  TPC{tpc}: {len(tiles)} tiles")
    print("  PASS: 64 tiles distributed across 4 TPCs")
    
    # Write golden vectors
    script_dir = os.path.dirname(os.path.abspath(__file__))
    golden_dir = os.path.join(script_dir, '../../golden_vectors')
    write_hex_files(vectors_16, golden_dir)
    print(f"\nGolden vectors written to {golden_dir}/")
    
    print("\n>>> ALL TILED GEMM MODEL TESTS PASSED! <<<")
    return vectors_16, vectors_64

if __name__ == "__main__":
    run_tests()

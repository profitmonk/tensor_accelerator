#!/usr/bin/env python3
"""
Residual Block Golden Model

Computes: Y = F(X) + X
Where:    F(X) = ReLU(X @ W + b)

This implements a skip/residual connection commonly used in ResNets.
"""

import numpy as np
import os

def relu(x):
    """ReLU activation: max(0, x)"""
    return np.maximum(0, x)

def residual_block(X, W, b):
    """
    Residual block with identity skip connection.
    
    Args:
        X: Input matrix [N, features] - INT8 or INT32
        W: Weight matrix [features, features] - INT8
        b: Bias vector [features] - INT32
        
    Returns:
        Y: Output = ReLU(X @ W + b) + X
        F_X: Main path output (for debugging)
    """
    # Main path: GEMM -> Add bias -> ReLU
    Z = X.astype(np.int32) @ W.astype(np.int32)  # GEMM
    Z_bias = Z + b.astype(np.int32)               # Add bias
    F_X = relu(Z_bias)                            # ReLU
    
    # Skip connection: Add input
    Y = F_X + X.astype(np.int32)
    
    return Y, F_X

def generate_test_vectors(seed=42):
    """Generate test vectors for RTL verification."""
    np.random.seed(seed)
    
    # Parameters (must match RTL test)
    N = 4        # Batch size (rows)
    features = 4 # Feature dimension
    
    # Generate random inputs
    X = np.random.randint(-4, 5, size=(N, features), dtype=np.int8)
    W = np.random.randint(-2, 3, size=(features, features), dtype=np.int8)
    b = np.random.randint(-8, 9, size=(features,), dtype=np.int32)
    
    # Compute golden output
    Y, F_X = residual_block(X, W, b)
    
    # Also compute intermediate values for debugging
    Z = X.astype(np.int32) @ W.astype(np.int32)
    Z_bias = Z + b
    
    return {
        'X': X,
        'W': W,
        'b': b,
        'Z': Z,           # GEMM output
        'Z_bias': Z_bias, # After bias add
        'F_X': F_X,       # After ReLU
        'Y': Y            # Final output (F_X + X)
    }

def write_golden_hex(vectors, output_dir):
    """Write golden vectors to hex files for RTL."""
    os.makedirs(output_dir, exist_ok=True)
    
    def write_matrix_hex(filename, matrix, bits=32):
        """Write matrix to hex file, row-major order."""
        with open(filename, 'w') as f:
            f.write(f"// Shape: {matrix.shape}\n")
            flat = matrix.flatten()
            for val in flat:
                # Convert to Python int to avoid numpy overflow
                val = int(val)
                # Handle signed values
                if val < 0:
                    val = val + (1 << bits)
                f.write(f"{val:0{bits//4}x}\n")
    
    def write_vector_hex(filename, vec, bits=32):
        """Write vector to hex file."""
        with open(filename, 'w') as f:
            f.write(f"// Length: {len(vec)}\n")
            for val in vec:
                val = int(val)
                if val < 0:
                    val = val + (1 << bits)
                f.write(f"{val:0{bits//4}x}\n")
    
    # Write all vectors
    write_matrix_hex(f"{output_dir}/residual_X.hex", vectors['X'], bits=8)
    write_matrix_hex(f"{output_dir}/residual_W.hex", vectors['W'], bits=8)
    write_vector_hex(f"{output_dir}/residual_b.hex", vectors['b'], bits=32)
    write_matrix_hex(f"{output_dir}/residual_Y_golden.hex", vectors['Y'], bits=32)
    
    # Also write intermediate values for debugging
    write_matrix_hex(f"{output_dir}/residual_Z_golden.hex", vectors['Z'], bits=32)
    write_matrix_hex(f"{output_dir}/residual_FX_golden.hex", vectors['F_X'], bits=32)

def print_test_case(vectors):
    """Print test case for manual verification."""
    print("=" * 60)
    print("RESIDUAL BLOCK TEST CASE")
    print("=" * 60)
    print(f"\nInput X ({vectors['X'].shape}):")
    print(vectors['X'])
    print(f"\nWeights W ({vectors['W'].shape}):")
    print(vectors['W'])
    print(f"\nBias b ({vectors['b'].shape}):")
    print(vectors['b'])
    print(f"\nGEMM output Z = X @ W ({vectors['Z'].shape}):")
    print(vectors['Z'])
    print(f"\nAfter bias: Z + b ({vectors['Z_bias'].shape}):")
    print(vectors['Z_bias'])
    print(f"\nAfter ReLU: F(X) ({vectors['F_X'].shape}):")
    print(vectors['F_X'])
    print(f"\nFinal output Y = F(X) + X ({vectors['Y'].shape}):")
    print(vectors['Y'])
    print("=" * 60)

def run_tests():
    """Run self-tests for the golden model."""
    print("RESIDUAL BLOCK GOLDEN MODEL TESTS")
    print("-" * 40)
    
    # Test 1: Identity weights (W=I, b=0)
    print("\nTest 1: Identity weights")
    X = np.array([[1, 2], [3, 4]], dtype=np.int8)
    W = np.eye(2, dtype=np.int8)
    b = np.zeros(2, dtype=np.int32)
    Y, F_X = residual_block(X, W, b)
    # F(X) = ReLU(X @ I + 0) = ReLU(X) = X (since X > 0)
    # Y = F(X) + X = X + X = 2*X
    expected = 2 * X
    assert np.allclose(Y, expected), f"Expected {expected}, got {Y}"
    print("  PASS: Y = 2X when W=I, b=0, X>0")
    
    # Test 2: ReLU zeroing negatives
    print("\nTest 2: ReLU with negative values")
    X = np.array([[1, 1], [1, 1]], dtype=np.int8)
    W = np.array([[1, 0], [0, 1]], dtype=np.int8)
    b = np.array([-10, 0], dtype=np.int32)  # First output will be negative
    Y, F_X = residual_block(X, W, b)
    # Z = [[1,1], [1,1]]
    # Z + b = [[-9,1], [-9,1]]
    # F(X) = ReLU = [[0,1], [0,1]]
    # Y = F(X) + X = [[0,1], [0,1]] + [[1,1], [1,1]] = [[1,2], [1,2]]
    expected = np.array([[1, 2], [1, 2]], dtype=np.int32)
    assert np.allclose(Y, expected), f"Expected {expected}, got {Y}"
    print("  PASS: ReLU correctly zeroes negative values")
    
    # Test 3: Generate and save test vectors
    print("\nTest 3: Generate random test vectors")
    vectors = generate_test_vectors(seed=42)
    print_test_case(vectors)
    
    # Write golden vectors
    script_dir = os.path.dirname(os.path.abspath(__file__))
    golden_dir = os.path.join(script_dir, '../../golden_vectors')
    write_golden_hex(vectors, golden_dir)
    print(f"\nGolden vectors written to {golden_dir}/")
    
    print("\n>>> ALL RESIDUAL MODEL TESTS PASSED! <<<")
    return vectors

if __name__ == "__main__":
    run_tests()

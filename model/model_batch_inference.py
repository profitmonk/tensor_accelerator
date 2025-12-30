#!/usr/bin/env python3
"""
Batch Processing Golden Model

Processes N input samples through same weights:
Y[n] = ReLU(X[n] × W + b)  for n = 0..N-1

Weights W and bias b are reused across all batches.
"""

import numpy as np
import os

def relu(x):
    """ReLU activation: max(0, x)"""
    return np.maximum(0, x)

def batch_inference(X_batch, W, b):
    """
    Batch inference with shared weights.
    
    Args:
        X_batch: Input batch [N, in_features] - INT8
        W: Weight matrix [in_features, out_features] - INT8
        b: Bias vector [out_features] - INT32
        
    Returns:
        Y: Output batch [N, out_features]
    """
    # Matrix multiply: [N, in] @ [in, out] = [N, out]
    Z = X_batch.astype(np.int32) @ W.astype(np.int32)
    
    # Add bias (broadcast across batch)
    Z_bias = Z + b.astype(np.int32)
    
    # ReLU activation
    Y = relu(Z_bias)
    
    return Y, Z, Z_bias

def generate_test_vectors(seed=42):
    """Generate test vectors for RTL verification."""
    np.random.seed(seed)
    
    # Parameters (must match RTL test)
    N = 4              # Batch size (4 samples)
    in_features = 4    # Input features
    out_features = 4   # Output features
    
    # Generate random inputs
    X_batch = np.random.randint(-3, 4, size=(N, in_features), dtype=np.int8)
    W = np.random.randint(-2, 3, size=(in_features, out_features), dtype=np.int8)
    b = np.random.randint(-5, 6, size=(out_features,), dtype=np.int32)
    
    # Compute golden output
    Y, Z, Z_bias = batch_inference(X_batch, W, b)
    
    return {
        'X_batch': X_batch,
        'W': W,
        'b': b,
        'Z': Z,           # GEMM output [N, out]
        'Z_bias': Z_bias, # After bias add
        'Y': Y            # After ReLU
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
                val = int(val)
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
    write_matrix_hex(f"{output_dir}/batch_X.hex", vectors['X_batch'], bits=8)
    write_matrix_hex(f"{output_dir}/batch_W.hex", vectors['W'], bits=8)
    write_vector_hex(f"{output_dir}/batch_b.hex", vectors['b'], bits=32)
    write_matrix_hex(f"{output_dir}/batch_Y_golden.hex", vectors['Y'], bits=32)

def print_test_case(vectors):
    """Print test case for manual verification."""
    print("=" * 60)
    print("BATCH PROCESSING TEST CASE")
    print("=" * 60)
    print(f"\nInput Batch X ({vectors['X_batch'].shape}):")
    print(vectors['X_batch'])
    print(f"\nWeights W ({vectors['W'].shape}) - shared across batch:")
    print(vectors['W'])
    print(f"\nBias b ({vectors['b'].shape}) - shared across batch:")
    print(vectors['b'])
    print(f"\nGEMM output Z = X @ W ({vectors['Z'].shape}):")
    print(vectors['Z'])
    print(f"\nAfter bias: Z + b ({vectors['Z_bias'].shape}):")
    print(vectors['Z_bias'])
    print(f"\nOutput Y = ReLU(Z + b) ({vectors['Y'].shape}):")
    print(vectors['Y'])
    print("=" * 60)

def run_tests():
    """Run self-tests for the golden model."""
    print("BATCH PROCESSING GOLDEN MODEL TESTS")
    print("-" * 40)
    
    # Test 1: Simple batch with identity weights
    print("\nTest 1: Batch with identity weights")
    X_batch = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int8)
    W = np.eye(2, dtype=np.int8)
    b = np.zeros(2, dtype=np.int32)
    Y, Z, Z_bias = batch_inference(X_batch, W, b)
    # Y should equal X since W=I, b=0, all values > 0
    assert np.allclose(Y, X_batch), f"Expected {X_batch}, got {Y}"
    print("  PASS: Y = X when W=I, b=0, X>0")
    
    # Test 2: Bias broadcast across batch
    print("\nTest 2: Bias broadcast")
    X_batch = np.array([[1, 1], [2, 2]], dtype=np.int8)
    W = np.eye(2, dtype=np.int8)
    b = np.array([10, 20], dtype=np.int32)
    Y, Z, Z_bias = batch_inference(X_batch, W, b)
    expected = np.array([[11, 21], [12, 22]], dtype=np.int32)
    assert np.allclose(Y, expected), f"Expected {expected}, got {Y}"
    print("  PASS: Bias correctly broadcast to all samples")
    
    # Test 3: ReLU activation
    print("\nTest 3: ReLU zeroing negatives in batch")
    X_batch = np.array([[1, 1], [1, 1]], dtype=np.int8)
    W = np.eye(2, dtype=np.int8)
    b = np.array([-5, 0], dtype=np.int32)
    Y, Z, Z_bias = batch_inference(X_batch, W, b)
    # Z = [[1,1], [1,1]]
    # Z + b = [[-4,1], [-4,1]]
    # ReLU = [[0,1], [0,1]]
    expected = np.array([[0, 1], [0, 1]], dtype=np.int32)
    assert np.allclose(Y, expected), f"Expected {expected}, got {Y}"
    print("  PASS: ReLU zeros negatives across batch")
    
    # Test 4: Generate and save test vectors
    print("\nTest 4: Generate random test vectors")
    vectors = generate_test_vectors(seed=123)
    print_test_case(vectors)
    
    # Write golden vectors
    script_dir = os.path.dirname(os.path.abspath(__file__))
    golden_dir = os.path.join(script_dir, '../../golden_vectors')
    write_golden_hex(vectors, golden_dir)
    print(f"\nGolden vectors written to {golden_dir}/")
    
    print("\n>>> ALL BATCH PROCESSING MODEL TESTS PASSED! <<<")
    return vectors

if __name__ == "__main__":
    run_tests()

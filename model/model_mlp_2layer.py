#!/usr/bin/env python3
"""
2-Layer MLP Golden Model

Layer 1: H = ReLU(X × W1 + b1)   [Input → Hidden]
Layer 2: Y = ReLU(H × W2 + b2)   [Hidden → Output]

This tests layer chaining and intermediate storage.
"""

import numpy as np
import os

def relu(x):
    """ReLU activation: max(0, x)"""
    return np.maximum(0, x)

def mlp_2layer(X, W1, b1, W2, b2):
    """
    2-layer MLP with ReLU activations.
    
    Args:
        X: Input [batch, in_features] - INT8
        W1: Layer 1 weights [in_features, hidden] - INT8
        b1: Layer 1 bias [hidden] - INT32
        W2: Layer 2 weights [hidden, out_features] - INT8
        b2: Layer 2 bias [out_features] - INT32
        
    Returns:
        Y: Output [batch, out_features]
        H: Hidden activations [batch, hidden]
    """
    # Layer 1: H = ReLU(X @ W1 + b1)
    Z1 = X.astype(np.int32) @ W1.astype(np.int32)
    Z1_bias = Z1 + b1.astype(np.int32)
    H = relu(Z1_bias)
    
    # Layer 2: Y = ReLU(H @ W2 + b2)
    Z2 = H @ W2.astype(np.int32)
    Z2_bias = Z2 + b2.astype(np.int32)
    Y = relu(Z2_bias)
    
    return Y, H, {'Z1': Z1, 'Z1_bias': Z1_bias, 'Z2': Z2, 'Z2_bias': Z2_bias}

def generate_test_vectors(seed=42):
    """Generate test vectors for RTL verification."""
    np.random.seed(seed)
    
    # Parameters - keep small for 4×4 systolic array
    batch = 4
    in_features = 4
    hidden = 4
    out_features = 4
    
    # Generate inputs that will produce positive hidden activations
    X = np.array([[1, 1, 1, 1],
                  [2, 2, 2, 2],
                  [1, 2, 1, 2],
                  [2, 1, 2, 1]], dtype=np.int8)
    
    # W1: identity matrix (H = X after ReLU since X > 0)
    W1 = np.eye(4, dtype=np.int8)
    b1 = np.zeros(4, dtype=np.int32)
    
    # W2: simple weights
    W2 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [1, 1, 0, 0],
                   [0, 0, 1, 1]], dtype=np.int8)
    b2 = np.zeros(4, dtype=np.int32)
    
    # Compute golden output
    Y, H, intermediates = mlp_2layer(X, W1, b1, W2, b2)
    
    return {
        'X': X,
        'W1': W1, 'b1': b1,
        'W2': W2, 'b2': b2,
        'H': H,           # Hidden layer output
        'Y': Y,           # Final output
        **intermediates
    }

def print_test_case(vectors):
    """Print test case for manual verification."""
    print("=" * 60)
    print("2-LAYER MLP TEST CASE")
    print("=" * 60)
    print(f"\nInput X ({vectors['X'].shape}):")
    print(vectors['X'])
    print(f"\nLayer 1 Weights W1 ({vectors['W1'].shape}):")
    print(vectors['W1'])
    print(f"\nLayer 1 Bias b1: {vectors['b1']}")
    print(f"\nZ1 = X @ W1:")
    print(vectors['Z1'])
    print(f"\nH = ReLU(Z1 + b1):")
    print(vectors['H'])
    print(f"\nLayer 2 Weights W2 ({vectors['W2'].shape}):")
    print(vectors['W2'])
    print(f"\nLayer 2 Bias b2: {vectors['b2']}")
    print(f"\nZ2 = H @ W2:")
    print(vectors['Z2'])
    print(f"\nY = ReLU(Z2 + b2) - Final Output:")
    print(vectors['Y'])
    print("=" * 60)

def run_tests():
    """Run self-tests for the golden model."""
    print("2-LAYER MLP GOLDEN MODEL TESTS")
    print("-" * 40)
    
    # Test 1: Identity network (both layers identity)
    print("\nTest 1: Identity network")
    X = np.array([[1, 2], [3, 4]], dtype=np.int8)
    W1 = np.eye(2, dtype=np.int8)
    b1 = np.zeros(2, dtype=np.int32)
    W2 = np.eye(2, dtype=np.int8)
    b2 = np.zeros(2, dtype=np.int32)
    Y, H, _ = mlp_2layer(X, W1, b1, W2, b2)
    assert np.allclose(Y, X), f"Expected {X}, got {Y}"
    print("  PASS: Y = X for identity network")
    
    # Test 2: ReLU activation
    print("\nTest 2: ReLU zeroing negatives")
    X = np.array([[1, 1]], dtype=np.int8)
    W1 = np.array([[1, 0], [0, 1]], dtype=np.int8)
    b1 = np.array([-5, 0], dtype=np.int32)  # First hidden will be negative
    W2 = np.eye(2, dtype=np.int8)
    b2 = np.zeros(2, dtype=np.int32)
    Y, H, _ = mlp_2layer(X, W1, b1, W2, b2)
    assert H[0, 0] == 0, f"Expected H[0,0]=0, got {H[0,0]}"
    assert H[0, 1] == 1, f"Expected H[0,1]=1, got {H[0,1]}"
    print("  PASS: ReLU zeros negative hidden activations")
    
    # Test 3: Generate full test vectors
    print("\nTest 3: Generate test vectors")
    vectors = generate_test_vectors(seed=42)
    print_test_case(vectors)
    
    print("\n>>> ALL 2-LAYER MLP MODEL TESTS PASSED! <<<")
    return vectors

if __name__ == "__main__":
    run_tests()

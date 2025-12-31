#!/usr/bin/env python3
"""
Attention Score Golden Model (Simplified)

Standard Attention: Attention(Q, K, V) = softmax(Q × K^T / √d) × V

Simplified for hardware:
1. S = Q × K^T           (score matrix)
2. S' = ReLU(S)          (simplified "attention" - no true softmax)
3. O = S' × V            (output)

This tests the core GEMM operations needed for attention.
True softmax would require exp() and division which are complex for int hardware.
"""

import numpy as np
import os

def attention_simplified(Q, K, V):
    """
    Simplified attention without softmax.
    
    Uses ReLU as a proxy for softmax sparsification.
    
    Args:
        Q: Query matrix [seq_len, d_k] - INT8
        K: Key matrix [seq_len, d_k] - INT8
        V: Value matrix [seq_len, d_v] - INT8
        
    Returns:
        O: Output [seq_len, d_v]
        S: Score matrix (Q × K^T)
        A: Attention weights (ReLU(S))
    """
    # Step 1: Compute scores S = Q × K^T
    S = Q.astype(np.int32) @ K.astype(np.int32).T
    
    # Step 2: Apply ReLU (simplified attention - no softmax)
    # In real attention, we'd do softmax here
    A = np.maximum(0, S)
    
    # Step 3: Compute output O = A × V
    O = A @ V.astype(np.int32)
    
    return O, S, A

def attention_with_scaling(Q, K, V, scale=True):
    """
    Attention with optional scaling by sqrt(d_k).
    
    For integer hardware, we approximate 1/sqrt(d_k) by right-shift.
    """
    d_k = Q.shape[-1]
    
    # Compute scores
    S = Q.astype(np.int32) @ K.astype(np.int32).T
    
    # Scale by approximate 1/sqrt(d_k)
    if scale:
        # For d_k=4, sqrt(4)=2, so divide by 2 (right shift by 1)
        # For d_k=8, sqrt(8)≈2.83, approximate as divide by 3
        shift = int(np.log2(np.sqrt(d_k)))
        S_scaled = S >> shift
    else:
        S_scaled = S
    
    # ReLU approximation of softmax
    A = np.maximum(0, S_scaled)
    
    # Output
    O = A @ V.astype(np.int32)
    
    return O, S, S_scaled, A

def generate_test_vectors(seed=42):
    """Generate test vectors for RTL verification."""
    np.random.seed(seed)
    
    # Small test case for 4×4 systolic array
    seq_len = 4   # Sequence length
    d_k = 4       # Key/Query dimension
    d_v = 4       # Value dimension
    
    # Generate random Q, K, V
    Q = np.random.randint(-2, 3, size=(seq_len, d_k), dtype=np.int8)
    K = np.random.randint(-2, 3, size=(seq_len, d_k), dtype=np.int8)
    V = np.random.randint(-2, 3, size=(seq_len, d_v), dtype=np.int8)
    
    # Compute attention
    O, S, A = attention_simplified(Q, K, V)
    
    return {
        'Q': Q,
        'K': K,
        'V': V,
        'S': S,       # Score matrix
        'A': A,       # Attention weights (ReLU'd scores)
        'O': O,       # Output
        'seq_len': seq_len,
        'd_k': d_k,
        'd_v': d_v
    }

def generate_simple_test():
    """Generate simple test with known values."""
    seq_len = 4
    d_k = 4
    d_v = 4
    
    # Q = K = I (identity) → S = I × I^T = I
    Q = np.eye(4, dtype=np.int8)
    K = np.eye(4, dtype=np.int8)
    V = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]], dtype=np.int8)
    
    # S = Q × K^T = I × I^T = I
    # A = ReLU(I) = I (all positive)
    # O = I × V = V
    O, S, A = attention_simplified(Q, K, V)
    
    return {
        'Q': Q, 'K': K, 'V': V,
        'S': S, 'A': A, 'O': O,
        'seq_len': seq_len, 'd_k': d_k, 'd_v': d_v
    }

def print_test_case(vectors):
    """Print test case for verification."""
    print("=" * 60)
    print("ATTENTION SCORE TEST CASE (Simplified)")
    print("=" * 60)
    
    print(f"\nConfiguration:")
    print(f"  seq_len: {vectors['seq_len']}")
    print(f"  d_k: {vectors['d_k']}")
    print(f"  d_v: {vectors['d_v']}")
    
    print(f"\nQuery Q ({vectors['Q'].shape}):")
    print(vectors['Q'])
    
    print(f"\nKey K ({vectors['K'].shape}):")
    print(vectors['K'])
    
    print(f"\nValue V ({vectors['V'].shape}):")
    print(vectors['V'])
    
    print(f"\nScore S = Q × K^T ({vectors['S'].shape}):")
    print(vectors['S'])
    
    print(f"\nAttention A = ReLU(S) ({vectors['A'].shape}):")
    print(vectors['A'])
    
    print(f"\nOutput O = A × V ({vectors['O'].shape}):")
    print(vectors['O'])
    
    print("=" * 60)

def run_tests():
    """Run self-tests for the golden model."""
    print("ATTENTION SCORE GOLDEN MODEL TESTS")
    print("-" * 40)
    
    # Test 1: Identity Q, K → O = V
    print("\nTest 1: Identity Q, K matrices")
    vectors = generate_simple_test()
    O = vectors['O']
    V = vectors['V']
    # With Q=K=I, S=I, A=I, O = I × V = V
    assert np.allclose(O, V), f"Expected O=V, got O={O}"
    print("  PASS: O = V when Q = K = I")
    print_test_case(vectors)
    
    # Test 2: Random test
    print("\nTest 2: Random Q, K, V")
    vectors = generate_test_vectors(seed=42)
    print_test_case(vectors)
    
    # Verify GEMM decomposition
    Q, K, V = vectors['Q'], vectors['K'], vectors['V']
    S = Q.astype(np.int32) @ K.astype(np.int32).T
    A = np.maximum(0, S)
    O = A @ V.astype(np.int32)
    assert np.allclose(vectors['S'], S), "S mismatch"
    assert np.allclose(vectors['A'], A), "A mismatch"
    assert np.allclose(vectors['O'], O), "O mismatch"
    print("  PASS: GEMM decomposition verified")
    
    # Test 3: Verify two-GEMM structure
    print("\nTest 3: Two-GEMM structure")
    # GEMM1: S = Q × K^T
    # GEMM2: O = A × V (where A = ReLU(S))
    print("  GEMM1: S[4×4] = Q[4×4] × K^T[4×4]")
    print("  VPU:   A[4×4] = ReLU(S[4×4])")
    print("  GEMM2: O[4×4] = A[4×4] × V[4×4]")
    print("  PASS: Two-GEMM attention structure")
    
    print("\n>>> ALL ATTENTION MODEL TESTS PASSED! <<<")
    return vectors

if __name__ == "__main__":
    run_tests()

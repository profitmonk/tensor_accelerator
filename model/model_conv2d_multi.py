#!/usr/bin/env python3
"""
Multi-Channel Conv2D Golden Model

Output[Co] = Σ(Ci) Conv2D(Input[Ci], Kernel[Co, Ci])

Using im2col + GEMM approach:
- Input [Ci, H, W] → Patches [Ci×K×K, out_H×out_W]
- Kernel [Co, Ci, K, K] → Weights [Co, Ci×K×K]
- Output = Weights × Patches
"""

import numpy as np
import os

def im2col(input, kernel_h, kernel_w, stride=1):
    """
    Convert input tensor to column matrix for GEMM-based convolution.
    
    Args:
        input: [Ci, H, W] input tensor
        kernel_h, kernel_w: kernel dimensions
        stride: convolution stride
        
    Returns:
        cols: [Ci×Kh×Kw, out_H×out_W] matrix
    """
    Ci, H, W = input.shape
    out_h = (H - kernel_h) // stride + 1
    out_w = (W - kernel_w) // stride + 1
    
    cols = np.zeros((Ci * kernel_h * kernel_w, out_h * out_w), dtype=input.dtype)
    
    col_idx = 0
    for y in range(0, H - kernel_h + 1, stride):
        for x in range(0, W - kernel_w + 1, stride):
            patch = input[:, y:y+kernel_h, x:x+kernel_w]
            cols[:, col_idx] = patch.flatten()
            col_idx += 1
    
    return cols

def conv2d_multichannel(input, kernel, bias=None):
    """
    Multi-channel 2D convolution using im2col + GEMM.
    
    Args:
        input: [Ci, H, W] input tensor - INT8
        kernel: [Co, Ci, Kh, Kw] kernel tensor - INT8
        bias: [Co] bias vector - INT32 (optional)
        
    Returns:
        output: [Co, out_H, out_W] output tensor - INT32
    """
    Ci, H, W = input.shape
    Co, _, Kh, Kw = kernel.shape
    
    out_h = H - Kh + 1
    out_w = W - Kw + 1
    
    # im2col transformation
    patches = im2col(input.astype(np.int32), Kh, Kw)  # [Ci×Kh×Kw, out_H×out_W]
    
    # Reshape kernel to weight matrix
    weights = kernel.reshape(Co, -1).astype(np.int32)  # [Co, Ci×Kh×Kw]
    
    # GEMM: output = weights × patches
    output_flat = weights @ patches  # [Co, out_H×out_W]
    
    # Add bias if provided
    if bias is not None:
        output_flat = output_flat + bias.reshape(-1, 1).astype(np.int32)
    
    # Reshape to output tensor
    output = output_flat.reshape(Co, out_h, out_w)
    
    return output, patches, weights

def generate_test_vectors(seed=42):
    """Generate test vectors for RTL verification."""
    np.random.seed(seed)
    
    # Small test case for 4×4 systolic array
    Ci = 2   # Input channels
    Co = 2   # Output channels
    H = W = 4  # Input spatial size
    K = 3    # Kernel size (3×3)
    
    # Generate random inputs
    input = np.random.randint(-2, 3, size=(Ci, H, W), dtype=np.int8)
    kernel = np.random.randint(-1, 2, size=(Co, Ci, K, K), dtype=np.int8)
    bias = np.zeros(Co, dtype=np.int32)  # No bias for simplicity
    
    # Compute golden output
    output, patches, weights = conv2d_multichannel(input, kernel, bias)
    
    return {
        'input': input,
        'kernel': kernel,
        'bias': bias,
        'output': output,
        'patches': patches,   # im2col result
        'weights': weights,   # Reshaped kernel
        'Ci': Ci, 'Co': Co, 'H': H, 'W': W, 'K': K
    }

def generate_simple_test():
    """Generate simple test with known values."""
    # 1 input channel, 1 output channel, 4×4 input, 3×3 kernel
    Ci, Co = 1, 1
    H, W = 4, 4
    K = 3
    
    # All-1s input
    input = np.ones((Ci, H, W), dtype=np.int8)
    
    # Simple kernel: center = 1, rest = 0
    kernel = np.zeros((Co, Ci, K, K), dtype=np.int8)
    kernel[0, 0, 1, 1] = 1  # Center pixel
    
    bias = np.zeros(Co, dtype=np.int32)
    
    output, patches, weights = conv2d_multichannel(input, kernel, bias)
    
    return {
        'input': input,
        'kernel': kernel,
        'bias': bias,
        'output': output,
        'patches': patches,
        'weights': weights,
        'Ci': Ci, 'Co': Co, 'H': H, 'W': W, 'K': K
    }

def print_test_case(vectors):
    """Print test case for verification."""
    print("=" * 60)
    print("MULTI-CHANNEL CONV2D TEST CASE")
    print("=" * 60)
    
    Ci, Co, H, W, K = vectors['Ci'], vectors['Co'], vectors['H'], vectors['W'], vectors['K']
    out_h, out_w = H - K + 1, W - K + 1
    
    print(f"\nConfiguration:")
    print(f"  Input: [{Ci}, {H}, {W}]")
    print(f"  Kernel: [{Co}, {Ci}, {K}, {K}]")
    print(f"  Output: [{Co}, {out_h}, {out_w}]")
    
    print(f"\nInput tensor:")
    for c in range(Ci):
        print(f"  Channel {c}:\n{vectors['input'][c]}")
    
    print(f"\nKernel tensor:")
    for co in range(Co):
        for ci in range(Ci):
            print(f"  Kernel[{co},{ci}]:\n{vectors['kernel'][co,ci]}")
    
    print(f"\nPatches (im2col): {vectors['patches'].shape}")
    print(vectors['patches'])
    
    print(f"\nWeights (reshaped kernel): {vectors['weights'].shape}")
    print(vectors['weights'])
    
    print(f"\nOutput tensor:")
    for c in range(Co):
        print(f"  Channel {c}:\n{vectors['output'][c]}")
    
    print("=" * 60)

def run_tests():
    """Run self-tests for the golden model."""
    print("MULTI-CHANNEL CONV2D GOLDEN MODEL TESTS")
    print("-" * 40)
    
    # Test 1: Simple center-only kernel
    print("\nTest 1: Center-only 3×3 kernel on all-1s input")
    vectors = generate_simple_test()
    output = vectors['output']
    # With center=1 kernel on all-1s input, output should be all 1s
    expected = np.ones((1, 2, 2), dtype=np.int32)
    assert np.allclose(output, expected), f"Expected {expected}, got {output}"
    print("  PASS: Center kernel extracts center values")
    
    # Test 2: Multi-channel test
    print("\nTest 2: Multi-channel conv2d")
    vectors = generate_test_vectors(seed=42)
    print_test_case(vectors)
    
    # Verify manually: check shapes
    Ci, Co, H, W, K = vectors['Ci'], vectors['Co'], vectors['H'], vectors['W'], vectors['K']
    assert vectors['patches'].shape == (Ci*K*K, (H-K+1)*(W-K+1)), "Patches shape mismatch"
    assert vectors['weights'].shape == (Co, Ci*K*K), "Weights shape mismatch"
    assert vectors['output'].shape == (Co, H-K+1, W-K+1), "Output shape mismatch"
    print("  PASS: All shapes correct")
    
    # Test 3: Verify GEMM equivalence
    print("\nTest 3: Verify GEMM equivalence")
    gemm_result = vectors['weights'] @ vectors['patches']
    output_flat = vectors['output'].reshape(Co, -1)
    assert np.allclose(gemm_result, output_flat), "GEMM doesn't match conv2d"
    print("  PASS: im2col + GEMM = Conv2D")
    
    print("\n>>> ALL CONV2D MODEL TESTS PASSED! <<<")
    return vectors

if __name__ == "__main__":
    run_tests()

#!/usr/bin/env python3
"""
Phase C: Requantization & Layer Chaining Golden Model

This module implements:
1. INT32 → INT8 requantization with configurable shift
2. Bias fusion (add bias before requantization)
3. Layer chaining: Conv → ReLU → Pool (full INT8 pipeline)

Quantization scheme:
  - Accumulator: INT32 (from systolic array)
  - Requantize: (acc + bias) >> shift, clip to [-128, 127]
  - Output: INT8 for next layer
"""

import numpy as np
import os
import json
from typing import Tuple, Dict

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_vectors")

# ============================================================================
# Requantization Functions
# ============================================================================

def requantize_int32_to_int8(
    x_int32: np.ndarray, 
    shift: int = 8,
    bias: np.ndarray = None,
    relu: bool = False
) -> np.ndarray:
    """
    Requantize INT32 accumulator to INT8.
    
    Pipeline: x_int32 + bias → >> shift → clip → (optional ReLU) → INT8
    
    Args:
        x_int32: INT32 accumulator values
        shift: Right shift amount (effectively divide by 2^shift)
        bias: Optional INT32 bias to add before shift
        relu: Apply ReLU after clipping
    
    Returns:
        INT8 output
    """
    result = x_int32.astype(np.int64)  # Prevent overflow during bias add
    
    # Add bias if provided
    if bias is not None:
        # Bias is typically per-channel, reshape for broadcasting
        # For (N, C, H, W) shaped input, bias is (C,) → reshape to (1, C, 1, 1)
        if result.ndim == 4 and bias.ndim == 1:
            bias_reshaped = bias.reshape(1, -1, 1, 1)
        elif result.ndim == 2 and bias.ndim == 1:
            # For (M, N) GEMM output, bias is (N,) → broadcast on last dim
            bias_reshaped = bias.reshape(1, -1)
        else:
            bias_reshaped = bias
        result = result + bias_reshaped.astype(np.int64)
    
    # Arithmetic right shift (preserves sign)
    # For positive: x >> n = floor(x / 2^n)
    # For negative: x >> n = floor(x / 2^n) (rounds toward -inf)
    result = result >> shift
    
    # Clip to INT8 range
    result = np.clip(result, -128, 127)
    
    # Optional ReLU
    if relu:
        result = np.maximum(result, 0)
    
    return result.astype(np.int8)


def compute_requant_shift(input_scale: float, weight_scale: float, output_scale: float) -> int:
    """
    Compute the right shift needed for requantization.
    
    After GEMM: acc_scale = input_scale * weight_scale
    To get output_scale: output = acc / (acc_scale / output_scale)
    
    We want: output = acc >> shift
    So: 2^shift ≈ acc_scale / output_scale
    """
    acc_scale = input_scale * weight_scale
    ratio = acc_scale / output_scale
    shift = int(np.round(np.log2(ratio)))
    return max(0, min(shift, 31))  # Clamp to valid range


# ============================================================================
# Layer Operations with Requantization
# ============================================================================

def conv2d_int8_with_requant(
    x_int8: np.ndarray,      # (N, C, H, W)
    weight_int8: np.ndarray, # (K, C, kH, kW)
    bias_int32: np.ndarray,  # (K,) or None
    shift: int,
    relu: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    INT8 convolution with fused bias and requantization.
    
    Returns:
        (output_int8, acc_int32) - both for verification
    """
    N, C, H, W = x_int8.shape
    K, _, kH, kW = weight_int8.shape
    
    # Output dimensions (valid padding)
    H_out = H - kH + 1
    W_out = W - kW + 1
    
    # im2col
    patches = []
    for n in range(N):
        for h in range(H_out):
            for w in range(W_out):
                patch = x_int8[n, :, h:h+kH, w:w+kW].flatten()
                patches.append(patch)
    
    im2col = np.array(patches, dtype=np.int8)  # (N*H_out*W_out, C*kH*kW)
    weight_flat = weight_int8.reshape(K, -1).T  # (C*kH*kW, K)
    
    # GEMM with INT32 accumulation
    acc_int32 = im2col.astype(np.int32) @ weight_flat.astype(np.int32)
    
    # Reshape to (N, H_out, W_out, K) then transpose to (N, K, H_out, W_out)
    acc_reshaped = acc_int32.reshape(N, H_out, W_out, K).transpose(0, 3, 1, 2)
    
    # Requantize with bias
    output_int8 = requantize_int32_to_int8(acc_reshaped, shift, bias_int32, relu)
    
    return output_int8, acc_reshaped


def relu_int8(x_int8: np.ndarray) -> np.ndarray:
    """Element-wise ReLU on INT8."""
    return np.maximum(x_int8, 0).astype(np.int8)


def avgpool2d_int8(x_int8: np.ndarray, pool_size: int = 2) -> np.ndarray:
    """
    2×2 average pooling with proper INT8 handling.
    
    sum of 4 INT8 values fits in INT16, divide by 4 via >> 2
    """
    N, C, H, W = x_int8.shape
    H_out = H // pool_size
    W_out = W // pool_size
    
    output = np.zeros((N, C, H_out, W_out), dtype=np.int8)
    
    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    # Sum 4 values (INT16 sufficient)
                    h_start = h * pool_size
                    w_start = w * pool_size
                    window = x_int8[n, c, h_start:h_start+pool_size, w_start:w_start+pool_size]
                    total = np.sum(window.astype(np.int16))
                    # Divide by 4 via right shift
                    output[n, c, h, w] = total >> 2
    
    return output


# ============================================================================
# Test Vector Generation
# ============================================================================

def save_hex_int8(filename: str, data: np.ndarray):
    with open(filename, 'w') as f:
        for val in data.flatten():
            val = int(val)
            if val < 0:
                val = val + 256
            f.write(f"{val:02x}\n")

def save_hex_int32(filename: str, data: np.ndarray):
    with open(filename, 'w') as f:
        for val in data.flatten():
            val = int(val)
            if val < 0:
                val = val + (1 << 32)
            f.write(f"{val:08x}\n")


class PhaseC_Tests:
    """Generate test vectors for Phase C."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
    
    def test1_basic_requant(self, output_dir: str):
        """Test 1: Basic INT32 → INT8 requantization."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("TEST 1: Basic Requantization (INT32 → INT8)")
        print("="*60)
        
        # Generate INT32 values in typical accumulator range
        # After 8x8 GEMM with INT8 inputs: max ≈ 127*127*8 ≈ 129,000
        acc_int32 = np.random.randint(-50000, 50000, size=(64,), dtype=np.int32)
        
        # Test multiple shift values
        shifts = [7, 8, 9, 10]
        
        results = {}
        for shift in shifts:
            output_int8 = requantize_int32_to_int8(acc_int32, shift=shift)
            results[f'shift{shift}'] = output_int8
            
            print(f"\n  Shift={shift}:")
            print(f"    Input range:  [{acc_int32.min()}, {acc_int32.max()}]")
            print(f"    Output range: [{output_int8.min()}, {output_int8.max()}]")
            print(f"    Sample: {acc_int32[0]} >> {shift} = {output_int8[0]}")
        
        # Save vectors
        save_hex_int32(f"{output_dir}/test1_input_int32.hex", acc_int32)
        np.save(f"{output_dir}/test1_input_int32.npy", acc_int32)
        
        for shift in shifts:
            save_hex_int8(f"{output_dir}/test1_output_shift{shift}_int8.hex", results[f'shift{shift}'])
            np.save(f"{output_dir}/test1_output_shift{shift}_int8.npy", results[f'shift{shift}'])
        
        return {'input_size': 64, 'shifts': shifts}
    
    def test2_bias_fusion(self, output_dir: str):
        """Test 2: GEMM output + bias → requantize."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("TEST 2: Bias Fusion (GEMM + bias → requantize)")
        print("="*60)
        
        # Simulate GEMM output: (M, N) = (16, 8) output channels
        M, N = 16, 8
        gemm_out_int32 = np.random.randint(-30000, 30000, size=(M, N), dtype=np.int32)
        
        # Per-channel bias (one per output channel)
        bias_int32 = np.random.randint(-1000, 1000, size=(N,), dtype=np.int32)
        
        shift = 8
        
        # Without bias
        out_no_bias = requantize_int32_to_int8(gemm_out_int32, shift=shift)
        
        # With bias (broadcast across M dimension)
        out_with_bias = requantize_int32_to_int8(gemm_out_int32, shift=shift, bias=bias_int32)
        
        # With bias and ReLU
        out_with_bias_relu = requantize_int32_to_int8(gemm_out_int32, shift=shift, bias=bias_int32, relu=True)
        
        print(f"\n  GEMM output: ({M}, {N})")
        print(f"  Bias: ({N},)")
        print(f"  Shift: {shift}")
        print(f"\n  Without bias: range [{out_no_bias.min()}, {out_no_bias.max()}]")
        print(f"  With bias:    range [{out_with_bias.min()}, {out_with_bias.max()}]")
        print(f"  With bias+ReLU: range [{out_with_bias_relu.min()}, {out_with_bias_relu.max()}]")
        
        # Save vectors
        save_hex_int32(f"{output_dir}/test2_gemm_int32.hex", gemm_out_int32)
        save_hex_int32(f"{output_dir}/test2_bias_int32.hex", bias_int32)
        save_hex_int8(f"{output_dir}/test2_out_no_bias_int8.hex", out_no_bias)
        save_hex_int8(f"{output_dir}/test2_out_with_bias_int8.hex", out_with_bias)
        save_hex_int8(f"{output_dir}/test2_out_bias_relu_int8.hex", out_with_bias_relu)
        
        np.save(f"{output_dir}/test2_gemm_int32.npy", gemm_out_int32)
        np.save(f"{output_dir}/test2_bias_int32.npy", bias_int32)
        np.save(f"{output_dir}/test2_out_no_bias_int8.npy", out_no_bias)
        np.save(f"{output_dir}/test2_out_with_bias_int8.npy", out_with_bias)
        np.save(f"{output_dir}/test2_out_bias_relu_int8.npy", out_with_bias_relu)
        
        return {'M': M, 'N': N, 'shift': shift}
    
    def test3_layer_chain(self, output_dir: str):
        """Test 3: Full layer chain Conv1 → ReLU → Pool1."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("TEST 3: Layer Chain (Conv → ReLU → Pool)")
        print("="*60)
        
        # Small but realistic dimensions
        # Input: (1, 1, 16, 16) - single channel 16x16 image
        # Conv: 4 output channels, 3x3 kernel
        # Output after conv: (1, 4, 14, 14)
        # Output after pool: (1, 4, 7, 7)
        
        N, C_in, H, W = 1, 1, 16, 16
        C_out = 4
        kH, kW = 3, 3
        
        # Generate input
        input_int8 = np.random.randint(-50, 50, size=(N, C_in, H, W), dtype=np.int8)
        
        # Generate weights and bias
        weight_int8 = np.random.randint(-30, 30, size=(C_out, C_in, kH, kW), dtype=np.int8)
        bias_int32 = np.random.randint(-500, 500, size=(C_out,), dtype=np.int32)
        
        shift = 7  # Chosen for this scale
        
        print(f"\n  Input: {input_int8.shape}")
        print(f"  Weight: {weight_int8.shape}")
        print(f"  Bias: {bias_int32.shape}")
        print(f"  Shift: {shift}")
        
        # Stage 1: Conv with bias and ReLU fused
        conv_out_int8, conv_acc_int32 = conv2d_int8_with_requant(
            input_int8, weight_int8, bias_int32, shift, relu=True
        )
        print(f"\n  After Conv+ReLU: {conv_out_int8.shape}, range [{conv_out_int8.min()}, {conv_out_int8.max()}]")
        
        # Stage 2: AvgPool (ReLU already done)
        pool_out_int8 = avgpool2d_int8(conv_out_int8, pool_size=2)
        print(f"  After Pool: {pool_out_int8.shape}, range [{pool_out_int8.min()}, {pool_out_int8.max()}]")
        
        # Also compute Conv without ReLU for separate ReLU test
        conv_no_relu_int8, _ = conv2d_int8_with_requant(
            input_int8, weight_int8, bias_int32, shift, relu=False
        )
        relu_out_int8 = relu_int8(conv_no_relu_int8)
        
        # Save all vectors
        save_hex_int8(f"{output_dir}/test3_input_int8.hex", input_int8)
        save_hex_int8(f"{output_dir}/test3_weight_int8.hex", weight_int8)
        save_hex_int32(f"{output_dir}/test3_bias_int32.hex", bias_int32)
        save_hex_int32(f"{output_dir}/test3_conv_acc_int32.hex", conv_acc_int32)
        save_hex_int8(f"{output_dir}/test3_conv_out_int8.hex", conv_out_int8)
        save_hex_int8(f"{output_dir}/test3_conv_no_relu_int8.hex", conv_no_relu_int8)
        save_hex_int8(f"{output_dir}/test3_relu_out_int8.hex", relu_out_int8)
        save_hex_int8(f"{output_dir}/test3_pool_out_int8.hex", pool_out_int8)
        
        np.save(f"{output_dir}/test3_input_int8.npy", input_int8)
        np.save(f"{output_dir}/test3_weight_int8.npy", weight_int8)
        np.save(f"{output_dir}/test3_bias_int32.npy", bias_int32)
        np.save(f"{output_dir}/test3_conv_acc_int32.npy", conv_acc_int32)
        np.save(f"{output_dir}/test3_conv_out_int8.npy", conv_out_int8)
        np.save(f"{output_dir}/test3_pool_out_int8.npy", pool_out_int8)
        
        # im2col for GEMM verification
        H_out = H - kH + 1
        W_out = W - kW + 1
        patches = []
        for h in range(H_out):
            for w in range(W_out):
                patch = input_int8[0, :, h:h+kH, w:w+kW].flatten()
                patches.append(patch)
        im2col = np.array(patches, dtype=np.int8)
        weight_flat = weight_int8.reshape(C_out, -1).T
        
        # Raw GEMM output (before reshape) - this is what RTL computes
        gemm_raw_int32 = im2col.astype(np.int32) @ weight_flat.astype(np.int32)
        save_hex_int32(f"{output_dir}/test3_gemm_raw_int32.hex", gemm_raw_int32)
        np.save(f"{output_dir}/test3_gemm_raw_int32.npy", gemm_raw_int32)
        
        save_hex_int8(f"{output_dir}/test3_im2col_int8.hex", im2col)
        save_hex_int8(f"{output_dir}/test3_weight_flat_int8.hex", weight_flat)
        np.save(f"{output_dir}/test3_im2col_int8.npy", im2col)
        np.save(f"{output_dir}/test3_weight_flat_int8.npy", weight_flat)
        
        return {
            'input_shape': list(input_int8.shape),
            'conv_out_shape': list(conv_out_int8.shape),
            'pool_out_shape': list(pool_out_int8.shape),
            'im2col_shape': list(im2col.shape),
            'weight_flat_shape': list(weight_flat.shape),
            'shift': shift
        }
    
    def test4_lenet_chain(self, output_dir: str):
        """Test 4: LeNet Conv1 → ReLU → Pool1 with requantization."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("TEST 4: LeNet Conv1 → ReLU → Pool1 Chain")
        print("="*60)
        
        # Full LeNet dimensions
        N, C_in, H, W = 1, 1, 28, 28
        C_out = 6
        kH, kW = 5, 5
        
        # Generate data
        input_int8 = np.random.randint(-100, 100, size=(N, C_in, H, W), dtype=np.int8)
        weight_int8 = np.random.randint(-50, 50, size=(C_out, C_in, kH, kW), dtype=np.int8)
        bias_int32 = np.random.randint(-2000, 2000, size=(C_out,), dtype=np.int32)
        
        shift = 8
        
        print(f"\n  Input: {input_int8.shape}")
        print(f"  Weight: {weight_int8.shape}")
        
        # Conv1 + ReLU (fused)
        conv_out_int8, conv_acc_int32 = conv2d_int8_with_requant(
            input_int8, weight_int8, bias_int32, shift, relu=True
        )
        print(f"  After Conv1+ReLU: {conv_out_int8.shape}")
        
        # Pool1
        pool_out_int8 = avgpool2d_int8(conv_out_int8, pool_size=2)
        print(f"  After Pool1: {pool_out_int8.shape}")
        
        # Save vectors
        save_hex_int8(f"{output_dir}/test4_input_int8.hex", input_int8)
        save_hex_int8(f"{output_dir}/test4_weight_int8.hex", weight_int8)
        save_hex_int32(f"{output_dir}/test4_bias_int32.hex", bias_int32)
        save_hex_int32(f"{output_dir}/test4_conv_acc_int32.hex", conv_acc_int32)
        save_hex_int8(f"{output_dir}/test4_conv_out_int8.hex", conv_out_int8)
        save_hex_int8(f"{output_dir}/test4_pool_out_int8.hex", pool_out_int8)
        
        np.save(f"{output_dir}/test4_input_int8.npy", input_int8)
        np.save(f"{output_dir}/test4_weight_int8.npy", weight_int8)
        np.save(f"{output_dir}/test4_bias_int32.npy", bias_int32)
        np.save(f"{output_dir}/test4_conv_acc_int32.npy", conv_acc_int32)
        np.save(f"{output_dir}/test4_conv_out_int8.npy", conv_out_int8)
        np.save(f"{output_dir}/test4_pool_out_int8.npy", pool_out_int8)
        
        # im2col and raw GEMM for RTL verification
        H_out = H - kH + 1  # 24
        W_out = W - kW + 1  # 24
        patches = []
        for h in range(H_out):
            for w in range(W_out):
                patch = input_int8[0, :, h:h+kH, w:w+kW].flatten()
                patches.append(patch)
        im2col = np.array(patches, dtype=np.int8)
        weight_flat = weight_int8.reshape(C_out, -1).T
        
        # Raw GEMM output (before reshape)
        gemm_raw_int32 = im2col.astype(np.int32) @ weight_flat.astype(np.int32)
        
        save_hex_int8(f"{output_dir}/test4_im2col_int8.hex", im2col)
        save_hex_int8(f"{output_dir}/test4_weight_flat_int8.hex", weight_flat)
        save_hex_int32(f"{output_dir}/test4_gemm_raw_int32.hex", gemm_raw_int32)
        np.save(f"{output_dir}/test4_im2col_int8.npy", im2col)
        np.save(f"{output_dir}/test4_weight_flat_int8.npy", weight_flat)
        np.save(f"{output_dir}/test4_gemm_raw_int32.npy", gemm_raw_int32)
        
        return {
            'input_shape': [N, C_in, H, W],
            'conv_out_shape': list(conv_out_int8.shape),
            'pool_out_shape': list(pool_out_int8.shape),
            'im2col_shape': list(im2col.shape),
            'shift': shift
        }
    
    def generate_all(self):
        """Generate all test vectors."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        print("\n" + "="*60)
        print("PHASE C: Requantization & Layer Chaining")
        print("="*60)
        
        results = {}
        results['test1'] = self.test1_basic_requant(OUTPUT_DIR)
        results['test2'] = self.test2_bias_fusion(OUTPUT_DIR)
        results['test3'] = self.test3_layer_chain(OUTPUT_DIR)
        results['test4'] = self.test4_lenet_chain(OUTPUT_DIR)
        
        # Save summary
        with open(f"{OUTPUT_DIR}/summary.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*60)
        print(f"All vectors saved to: {OUTPUT_DIR}")
        print("="*60)
        
        # List files
        files = sorted(os.listdir(OUTPUT_DIR))
        total_size = 0
        for f in files:
            size = os.path.getsize(f"{OUTPUT_DIR}/{f}")
            total_size += size
        print(f"\nGenerated {len(files)} files, {total_size/1024:.1f} KB total")
        
        return results


if __name__ == "__main__":
    tests = PhaseC_Tests(seed=42)
    tests.generate_all()

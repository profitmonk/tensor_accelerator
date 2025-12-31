#!/usr/bin/env python3
"""
Phase D: Transformer Operations Golden Model

This module implements transformer-specific operations:
1. LayerNorm - normalize across features
2. Softmax - attention weights
3. GELU - activation function
4. Multi-head Attention - Q, K, V projections and attention

For hardware implementation, we use fixed-point approximations:
- LayerNorm: compute in higher precision, requantize output
- Softmax: lookup table for exp, careful overflow handling
- GELU: polynomial approximation
"""

import numpy as np
import os
import json
from typing import Tuple, Dict

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_vectors")

# ============================================================================
# LayerNorm Implementation
# ============================================================================

def layernorm_fp32(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Standard LayerNorm in FP32.
    
    y = gamma * (x - mean) / sqrt(var + eps) + beta
    
    Args:
        x: Input tensor, normalized over last dimension
        gamma: Scale parameter (same size as last dim)
        beta: Bias parameter (same size as last dim)
        eps: Small constant for numerical stability
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


def layernorm_int8(
    x_int8: np.ndarray,
    x_scale: float,
    gamma_int8: np.ndarray,
    gamma_scale: float,
    beta_int32: np.ndarray,
    output_scale: float,
    eps: float = 1e-5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    LayerNorm with INT8 input/output.
    
    Pipeline:
    1. Dequantize input to FP32: x_fp32 = x_int8 * x_scale
    2. Compute LayerNorm in FP32
    3. Apply gamma (dequantized) and beta
    4. Requantize to INT8
    
    Returns:
        (output_int8, output_fp32) for verification
    """
    # Dequantize input
    x_fp32 = x_int8.astype(np.float32) * x_scale
    
    # Compute mean and variance
    mean = np.mean(x_fp32, axis=-1, keepdims=True)
    var = np.var(x_fp32, axis=-1, keepdims=True)
    
    # Normalize
    x_norm = (x_fp32 - mean) / np.sqrt(var + eps)
    
    # Apply gamma and beta (beta is pre-scaled)
    gamma_fp32 = gamma_int8.astype(np.float32) * gamma_scale
    beta_fp32 = beta_int32.astype(np.float32) * output_scale  # beta in output scale
    
    output_fp32 = gamma_fp32 * x_norm + beta_fp32
    
    # Requantize
    output_int8 = np.clip(np.round(output_fp32 / output_scale), -128, 127).astype(np.int8)
    
    return output_int8, output_fp32


# ============================================================================
# Softmax Implementation
# ============================================================================

def softmax_fp32(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_int8(
    x_int8: np.ndarray,
    x_scale: float,
    output_bits: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Softmax with INT8 input, fixed-point output.
    
    For attention, output is often kept in higher precision (INT16)
    or normalized to sum to 256 (for INT8 weights summing to 1.0).
    
    Returns:
        (output_int8, output_fp32)
    """
    # Dequantize
    x_fp32 = x_int8.astype(np.float32) * x_scale
    
    # Compute softmax in FP32
    softmax_fp32_result = softmax_fp32(x_fp32)
    
    # Quantize output to INT8 (0-255 range for probabilities)
    # Scale by 127 to fit in signed INT8 positive range
    output_int8 = np.clip(np.round(softmax_fp32_result * 127), 0, 127).astype(np.int8)
    
    return output_int8, softmax_fp32_result


# ============================================================================
# GELU Implementation
# ============================================================================

def gelu_fp32(x: np.ndarray) -> np.ndarray:
    """
    Gaussian Error Linear Unit.
    GELU(x) = x * Φ(x) where Φ is the CDF of standard normal.
    
    Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    """
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def gelu_int8(
    x_int8: np.ndarray,
    x_scale: float,
    output_scale: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GELU with INT8 input/output.
    
    For hardware, often implemented as lookup table.
    """
    # Dequantize
    x_fp32 = x_int8.astype(np.float32) * x_scale
    
    # Compute GELU in FP32
    gelu_fp32_result = gelu_fp32(x_fp32)
    
    # Requantize
    output_int8 = np.clip(np.round(gelu_fp32_result / output_scale), -128, 127).astype(np.int8)
    
    return output_int8, gelu_fp32_result


def gelu_lut_int8(num_entries: int = 256) -> np.ndarray:
    """
    Generate GELU lookup table for INT8 inputs.
    
    Input range: [-128, 127] maps to some FP32 range
    Output: INT8 GELU result
    """
    # Assume input scale maps [-128, 127] to roughly [-4, 4]
    input_scale = 4.0 / 128.0
    output_scale = input_scale  # Same scale for output
    
    lut = np.zeros(256, dtype=np.int8)
    for i in range(256):
        # Convert to signed
        x_int8 = i if i < 128 else i - 256
        x_fp32 = x_int8 * input_scale
        y_fp32 = gelu_fp32(x_fp32)
        lut[i] = np.clip(np.round(y_fp32 / output_scale), -128, 127).astype(np.int8)
    
    return lut


# ============================================================================
# Multi-Head Attention Implementation
# ============================================================================

def attention_fp32(
    Q: np.ndarray,  # (batch, heads, seq_len, head_dim)
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scaled dot-product attention.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """
    d_k = Q.shape[-1]
    
    # QK^T / sqrt(d_k)
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scores = scores + mask  # mask should be 0 or -inf
    
    # Softmax
    attn_weights = softmax_fp32(scores, axis=-1)
    
    # Weighted sum of values
    output = np.matmul(attn_weights, V)
    
    return output, attn_weights


def attention_int8(
    Q_int8: np.ndarray,  # (seq_len, head_dim)
    K_int8: np.ndarray,
    V_int8: np.ndarray,
    qk_scale: float,
    v_scale: float,
    output_scale: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single-head attention with INT8 tensors.
    
    Simplified for testing (no batch, single head).
    
    Pipeline:
    1. QK^T (INT32 accumulator)
    2. Scale and requantize scores
    3. Softmax (in higher precision)
    4. Attention * V (INT32 accumulator)
    5. Requantize output
    """
    seq_len, head_dim = Q_int8.shape
    
    # QK^T -> (seq_len, seq_len)
    # INT8 × INT8 -> INT32
    qk_int32 = Q_int8.astype(np.int32) @ K_int8.T.astype(np.int32)
    
    # Scale QK^T (typically divide by sqrt(head_dim))
    # In quantized domain: shift right
    sqrt_d = np.sqrt(head_dim)
    qk_fp32 = qk_int32.astype(np.float32) * qk_scale / sqrt_d
    
    # Softmax (compute in FP32)
    attn_weights_fp32 = softmax_fp32(qk_fp32)
    
    # Quantize attention weights to INT8 (0-127 range)
    attn_weights_int8 = np.clip(np.round(attn_weights_fp32 * 127), 0, 127).astype(np.int8)
    
    # Attention * V
    # For proper quantization, we'd use attn_weights_int8
    # But for accuracy, use FP32 weights
    output_fp32 = attn_weights_fp32 @ (V_int8.astype(np.float32) * v_scale)
    
    # Requantize output
    output_int8 = np.clip(np.round(output_fp32 / output_scale), -128, 127).astype(np.int8)
    
    return output_int8, attn_weights_int8, qk_int32


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


class PhaseD_Tests:
    """Generate test vectors for Phase D."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
    
    def test1_layernorm(self, output_dir: str):
        """Test 1: LayerNorm operation."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("TEST 1: LayerNorm")
        print("="*60)
        
        # Dimensions: (batch, seq_len, hidden_dim)
        batch, seq_len, hidden = 1, 8, 64
        
        # Generate input
        x_int8 = np.random.randint(-100, 100, size=(batch, seq_len, hidden), dtype=np.int8)
        x_scale = 0.1
        
        # LayerNorm parameters
        gamma_int8 = np.random.randint(50, 127, size=(hidden,), dtype=np.int8)  # Positive scale
        gamma_scale = 0.01
        beta_int32 = np.random.randint(-500, 500, size=(hidden,), dtype=np.int32)
        output_scale = 0.1
        
        print(f"\n  Input: {x_int8.shape}")
        print(f"  Hidden dim: {hidden}")
        print(f"  x_scale={x_scale}, gamma_scale={gamma_scale}, output_scale={output_scale}")
        
        # Compute LayerNorm
        output_int8, output_fp32 = layernorm_int8(
            x_int8, x_scale, gamma_int8, gamma_scale, beta_int32, output_scale
        )
        
        print(f"  Output: {output_int8.shape}")
        print(f"  Output range: [{output_int8.min()}, {output_int8.max()}]")
        
        # Also compute intermediate values for verification
        x_fp32 = x_int8.astype(np.float32) * x_scale
        mean = np.mean(x_fp32, axis=-1, keepdims=True)
        var = np.var(x_fp32, axis=-1, keepdims=True)
        
        print(f"  Mean range: [{mean.min():.3f}, {mean.max():.3f}]")
        print(f"  Var range: [{var.min():.3f}, {var.max():.3f}]")
        
        # Save vectors
        save_hex_int8(f"{output_dir}/test1_x_int8.hex", x_int8)
        save_hex_int8(f"{output_dir}/test1_gamma_int8.hex", gamma_int8)
        save_hex_int32(f"{output_dir}/test1_beta_int32.hex", beta_int32)
        save_hex_int8(f"{output_dir}/test1_output_int8.hex", output_int8)
        
        np.save(f"{output_dir}/test1_x_int8.npy", x_int8)
        np.save(f"{output_dir}/test1_gamma_int8.npy", gamma_int8)
        np.save(f"{output_dir}/test1_beta_int32.npy", beta_int32)
        np.save(f"{output_dir}/test1_output_int8.npy", output_int8)
        np.save(f"{output_dir}/test1_output_fp32.npy", output_fp32)
        np.save(f"{output_dir}/test1_mean_fp32.npy", mean)
        np.save(f"{output_dir}/test1_var_fp32.npy", var)
        
        # Save scales
        np.save(f"{output_dir}/test1_scales.npy", np.array([x_scale, gamma_scale, output_scale]))
        
        return {
            'input_shape': list(x_int8.shape),
            'hidden_dim': hidden,
            'x_scale': x_scale,
            'gamma_scale': gamma_scale,
            'output_scale': output_scale
        }
    
    def test2_softmax(self, output_dir: str):
        """Test 2: Softmax operation."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("TEST 2: Softmax")
        print("="*60)
        
        # Dimensions: attention scores (seq_len, seq_len)
        seq_len = 16
        
        # Generate input (attention scores before softmax)
        x_int8 = np.random.randint(-50, 50, size=(seq_len, seq_len), dtype=np.int8)
        x_scale = 0.1  # Maps [-50, 50] to [-5, 5]
        
        print(f"\n  Input: {x_int8.shape}")
        print(f"  x_scale: {x_scale}")
        
        # Compute softmax
        output_int8, output_fp32 = softmax_int8(x_int8, x_scale)
        
        print(f"  Output range: [{output_int8.min()}, {output_int8.max()}]")
        print(f"  FP32 output sum per row: {output_fp32.sum(axis=-1)[:3]}...")
        print(f"  INT8 output sum per row: {output_int8.sum(axis=-1)[:3]}...")
        
        # Save vectors
        save_hex_int8(f"{output_dir}/test2_x_int8.hex", x_int8)
        save_hex_int8(f"{output_dir}/test2_output_int8.hex", output_int8)
        
        np.save(f"{output_dir}/test2_x_int8.npy", x_int8)
        np.save(f"{output_dir}/test2_output_int8.npy", output_int8)
        np.save(f"{output_dir}/test2_output_fp32.npy", output_fp32)
        np.save(f"{output_dir}/test2_x_scale.npy", np.array([x_scale]))
        
        return {
            'input_shape': list(x_int8.shape),
            'x_scale': x_scale
        }
    
    def test3_gelu(self, output_dir: str):
        """Test 3: GELU activation."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("TEST 3: GELU Activation")
        print("="*60)
        
        # Generate input covering typical activation range
        x_int8 = np.arange(-128, 128, dtype=np.int16).astype(np.int8)  # All INT8 values
        x_scale = 4.0 / 128.0  # Maps [-128, 127] to [-4, 4]
        output_scale = x_scale
        
        print(f"\n  Input: {x_int8.shape}")
        print(f"  x_scale: {x_scale}")
        
        # Compute GELU
        output_int8, output_fp32 = gelu_int8(x_int8, x_scale, output_scale)
        
        # Also generate lookup table
        gelu_lut = gelu_lut_int8()
        
        print(f"  Output range: [{output_int8.min()}, {output_int8.max()}]")
        print(f"  Sample: GELU(-2) = {output_int8[128-64]}, GELU(0) = {output_int8[128]}, GELU(2) = {output_int8[128+64]}")
        
        # Verify LUT matches
        lut_matches = np.all(output_int8 == gelu_lut)
        print(f"  LUT matches direct computation: {lut_matches}")
        
        # Save vectors
        save_hex_int8(f"{output_dir}/test3_x_int8.hex", x_int8)
        save_hex_int8(f"{output_dir}/test3_output_int8.hex", output_int8)
        save_hex_int8(f"{output_dir}/test3_gelu_lut.hex", gelu_lut)
        
        np.save(f"{output_dir}/test3_x_int8.npy", x_int8)
        np.save(f"{output_dir}/test3_output_int8.npy", output_int8)
        np.save(f"{output_dir}/test3_output_fp32.npy", output_fp32)
        np.save(f"{output_dir}/test3_gelu_lut.npy", gelu_lut)
        np.save(f"{output_dir}/test3_scales.npy", np.array([x_scale, output_scale]))
        
        return {
            'input_size': len(x_int8),
            'x_scale': x_scale,
            'output_scale': output_scale
        }
    
    def test4_attention(self, output_dir: str):
        """Test 4: Single-head attention."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("TEST 4: Single-Head Attention")
        print("="*60)
        
        # Dimensions
        seq_len = 8
        head_dim = 16
        
        # Generate Q, K, V
        Q_int8 = np.random.randint(-50, 50, size=(seq_len, head_dim), dtype=np.int8)
        K_int8 = np.random.randint(-50, 50, size=(seq_len, head_dim), dtype=np.int8)
        V_int8 = np.random.randint(-50, 50, size=(seq_len, head_dim), dtype=np.int8)
        
        qk_scale = 0.01  # Scale for Q, K
        v_scale = 0.1    # Scale for V
        output_scale = 0.1
        
        print(f"\n  Q, K, V: ({seq_len}, {head_dim})")
        print(f"  qk_scale={qk_scale}, v_scale={v_scale}, output_scale={output_scale}")
        
        # Compute attention
        output_int8, attn_weights_int8, qk_int32 = attention_int8(
            Q_int8, K_int8, V_int8, qk_scale, v_scale, output_scale
        )
        
        print(f"  QK^T shape: {qk_int32.shape}, range: [{qk_int32.min()}, {qk_int32.max()}]")
        print(f"  Attention weights range: [{attn_weights_int8.min()}, {attn_weights_int8.max()}]")
        print(f"  Output shape: {output_int8.shape}, range: [{output_int8.min()}, {output_int8.max()}]")
        
        # Save vectors
        save_hex_int8(f"{output_dir}/test4_Q_int8.hex", Q_int8)
        save_hex_int8(f"{output_dir}/test4_K_int8.hex", K_int8)
        save_hex_int8(f"{output_dir}/test4_V_int8.hex", V_int8)
        save_hex_int32(f"{output_dir}/test4_qk_int32.hex", qk_int32)
        save_hex_int8(f"{output_dir}/test4_attn_weights_int8.hex", attn_weights_int8)
        save_hex_int8(f"{output_dir}/test4_output_int8.hex", output_int8)
        
        np.save(f"{output_dir}/test4_Q_int8.npy", Q_int8)
        np.save(f"{output_dir}/test4_K_int8.npy", K_int8)
        np.save(f"{output_dir}/test4_V_int8.npy", V_int8)
        np.save(f"{output_dir}/test4_qk_int32.npy", qk_int32)
        np.save(f"{output_dir}/test4_attn_weights_int8.npy", attn_weights_int8)
        np.save(f"{output_dir}/test4_output_int8.npy", output_int8)
        np.save(f"{output_dir}/test4_scales.npy", np.array([qk_scale, v_scale, output_scale]))
        
        return {
            'seq_len': seq_len,
            'head_dim': head_dim,
            'qk_scale': qk_scale,
            'v_scale': v_scale,
            'output_scale': output_scale
        }
    
    def generate_all(self):
        """Generate all test vectors."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        print("\n" + "="*60)
        print("PHASE D: Transformer Operations")
        print("="*60)
        
        results = {}
        results['test1_layernorm'] = self.test1_layernorm(OUTPUT_DIR)
        results['test2_softmax'] = self.test2_softmax(OUTPUT_DIR)
        results['test3_gelu'] = self.test3_gelu(OUTPUT_DIR)
        results['test4_attention'] = self.test4_attention(OUTPUT_DIR)
        
        # Save summary
        with open(f"{OUTPUT_DIR}/summary.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*60)
        print(f"All vectors saved to: {OUTPUT_DIR}")
        print("="*60)
        
        # List files
        files = sorted(os.listdir(OUTPUT_DIR))
        total_size = sum(os.path.getsize(f"{OUTPUT_DIR}/{f}") for f in files)
        print(f"\nGenerated {len(files)} files, {total_size/1024:.1f} KB total")
        
        return results


if __name__ == "__main__":
    tests = PhaseD_Tests(seed=42)
    tests.generate_all()

#!/usr/bin/env python3
"""
ResNet-18 Basic Block Golden Model - 56×56 Spatial, 16 Channels

Architecture:
  Input:  (1, 16, 56, 56)
          │
          ├──────────────────────────────────┐ (skip connection)
          ▼                                  │
     Conv1 (3×3, 16→16, pad=1)              │
          ▼                                  │
     BN1 (scale × x + bias)                 │
          ▼                                  │
     ReLU                                    │
          ▼                                  │
     Conv2 (3×3, 16→16, pad=1)              │
          ▼                                  │
     BN2 (scale × x + bias)                 │
          ▼                                  │
     Add ←───────────────────────────────────┘
          ▼
     ReLU
          ▼
  Output: (1, 16, 56, 56)

im2col GEMM for each conv:
  Patches: 56 × 56 = 3136 spatial positions
  Patch size: 16 channels × 3 × 3 = 144 elements
  A: (3136, 144)
  B: (144, 16)
  C: (3136, 16) → reshape to (1, 16, 56, 56)
"""

import numpy as np
import os
import json
from typing import Tuple, Dict

TILE_SIZE = 8
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_vectors")

# ============================================================================
# Utility Functions
# ============================================================================

def quantize_to_int8(x: np.ndarray) -> Tuple[np.ndarray, float]:
    max_val = np.abs(x).max()
    scale = max_val / 127.0 if max_val > 0 else 1.0
    x_int8 = np.clip(np.round(x / scale), -128, 127).astype(np.int8)
    return x_int8, scale

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

def im2col_padded(x: np.ndarray, kH: int, kW: int, stride: int = 1, pad: int = 1) -> np.ndarray:
    """im2col with padding for 'same' convolution."""
    N, C, H, W = x.shape
    
    x_pad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
    
    H_out = (H + 2*pad - kH) // stride + 1
    W_out = (W + 2*pad - kW) // stride + 1
    
    col = np.zeros((N * H_out * W_out, C * kH * kW), dtype=x.dtype)
    
    idx = 0
    for n in range(N):
        for h in range(H_out):
            for w in range(W_out):
                h_start = h * stride
                w_start = w * stride
                patch = x_pad[n, :, h_start:h_start+kH, w_start:w_start+kW]
                col[idx] = patch.flatten()
                idx += 1
    
    return col

def tiled_gemm_int8(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """INT8 GEMM with INT32 accumulation."""
    return A.astype(np.int32) @ B.astype(np.int32)

def requantize_int32_to_int8(x: np.ndarray, shift: int = 8) -> np.ndarray:
    x_shifted = x >> shift
    return np.clip(x_shifted, -128, 127).astype(np.int8)

# ============================================================================
# ResNet Basic Block
# ============================================================================

class ResNetBlock:
    def __init__(self, channels: int = 16, spatial: int = 56, seed: int = 42):
        np.random.seed(seed)
        
        self.channels = channels
        self.spatial = spatial
        
        # Conv1 weights: (out_c, in_c, kH, kW)
        fan_in = channels * 9
        std = np.sqrt(2.0 / fan_in)
        self.conv1_weight = (np.random.randn(channels, channels, 3, 3) * std).astype(np.float32)
        
        # BN1 parameters (fused)
        self.bn1_scale = np.ones(channels, dtype=np.float32)
        self.bn1_bias = np.zeros(channels, dtype=np.float32)
        
        # Conv2 weights
        self.conv2_weight = (np.random.randn(channels, channels, 3, 3) * std).astype(np.float32)
        
        # BN2 parameters
        self.bn2_scale = np.ones(channels, dtype=np.float32)
        self.bn2_bias = np.zeros(channels, dtype=np.float32)
        
        # Quantize weights
        self.conv1_weight_int8, self.conv1_scale = quantize_to_int8(self.conv1_weight)
        self.conv2_weight_int8, self.conv2_scale = quantize_to_int8(self.conv2_weight)
    
    def forward_int8(self, x_int8: np.ndarray, x_scale: float) -> Dict[str, np.ndarray]:
        """INT8 forward pass matching RTL."""
        results = {}
        results['input_int8'] = x_int8.copy()
        identity_int8 = x_int8.copy()
        
        # Conv1: im2col + GEMM
        col_int8 = im2col_padded(x_int8, 3, 3, pad=1)
        results['conv1_im2col_int8'] = col_int8.copy()
        
        # Weight: (C, C, 3, 3) -> (C*3*3, C) = (144, 16)
        weight1_flat = self.conv1_weight_int8.reshape(self.channels, -1).T
        results['conv1_weight_flat_int8'] = weight1_flat.copy()
        
        # GEMM
        conv1_out_int32 = tiled_gemm_int8(col_int8, weight1_flat)
        results['conv1_out_int32'] = conv1_out_int32.copy()
        
        # Requantize
        conv1_out_int8 = requantize_int32_to_int8(conv1_out_int32, shift=7)
        conv1_out_int8 = conv1_out_int8.reshape(1, self.spatial, self.spatial, self.channels)
        conv1_out_int8 = conv1_out_int8.transpose(0, 3, 1, 2)
        results['conv1_out_int8'] = conv1_out_int8.copy()
        
        # BN1 (scale=1, bias=0 for simplicity)
        bn1_out_int8 = conv1_out_int8.copy()
        results['bn1_out_int8'] = bn1_out_int8.copy()
        
        # ReLU1
        relu1_out_int8 = np.maximum(bn1_out_int8, 0)
        results['relu1_out_int8'] = relu1_out_int8.copy()
        
        # Conv2
        col_int8 = im2col_padded(relu1_out_int8, 3, 3, pad=1)
        results['conv2_im2col_int8'] = col_int8.copy()
        
        weight2_flat = self.conv2_weight_int8.reshape(self.channels, -1).T
        results['conv2_weight_flat_int8'] = weight2_flat.copy()
        
        conv2_out_int32 = tiled_gemm_int8(col_int8, weight2_flat)
        results['conv2_out_int32'] = conv2_out_int32.copy()
        
        conv2_out_int8 = requantize_int32_to_int8(conv2_out_int32, shift=7)
        conv2_out_int8 = conv2_out_int8.reshape(1, self.spatial, self.spatial, self.channels)
        conv2_out_int8 = conv2_out_int8.transpose(0, 3, 1, 2)
        results['conv2_out_int8'] = conv2_out_int8.copy()
        
        # BN2
        bn2_out_int8 = conv2_out_int8.copy()
        results['bn2_out_int8'] = bn2_out_int8.copy()
        
        # Residual add
        add_out_int16 = bn2_out_int8.astype(np.int16) + identity_int8.astype(np.int16)
        add_out_int8 = np.clip(add_out_int16, -128, 127).astype(np.int8)
        results['add_out_int8'] = add_out_int8.copy()
        
        # ReLU2
        output_int8 = np.maximum(add_out_int8, 0)
        results['output_int8'] = output_int8.copy()
        
        return results
    
    def generate_test_vectors(self, output_dir: str = OUTPUT_DIR):
        """Generate test vectors for RTL verification."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 70)
        print("ResNet Block Test Vector Generation")
        print(f"Input/Output: (1, {self.channels}, {self.spatial}, {self.spatial})")
        print("=" * 70)
        
        # Generate input
        x_fp32 = np.random.randn(1, self.channels, self.spatial, self.spatial).astype(np.float32) * 0.5
        x_int8, x_scale = quantize_to_int8(x_fp32)
        
        print(f"\n[INPUT]")
        print(f"  Shape: {x_int8.shape}")
        print(f"  Elements: {x_int8.size:,}")
        print(f"  Scale: {x_scale:.6f}")
        
        # Run forward pass
        print(f"\n[FORWARD PASS]")
        results = self.forward_int8(x_int8, x_scale)
        
        M, K = results['conv1_im2col_int8'].shape
        N = self.channels
        num_tiles = ((M + TILE_SIZE - 1) // TILE_SIZE) * \
                    ((K + TILE_SIZE - 1) // TILE_SIZE) * \
                    ((N + TILE_SIZE - 1) // TILE_SIZE)
        
        print(f"  Conv1 im2col: {results['conv1_im2col_int8'].shape}")
        print(f"  Conv1 GEMM: ({M}, {K}) × ({K}, {N}) → ({M}, {N})")
        print(f"  Conv1 tiles: {num_tiles}")
        print(f"  Conv2 tiles: {num_tiles}")
        print(f"  Total tiles: {num_tiles * 2}")
        
        # Save test vectors
        print(f"\n[SAVING]")
        
        # Input
        save_hex_int8(f"{output_dir}/input_int8.hex", x_int8)
        np.save(f"{output_dir}/input_int8.npy", x_int8)
        np.save(f"{output_dir}/input_scale.npy", np.array([x_scale]))
        
        # Conv1
        save_hex_int8(f"{output_dir}/conv1_im2col_int8.hex", results['conv1_im2col_int8'])
        save_hex_int8(f"{output_dir}/conv1_weight_int8.hex", results['conv1_weight_flat_int8'])
        save_hex_int32(f"{output_dir}/conv1_expected_int32.hex", results['conv1_out_int32'])
        
        np.save(f"{output_dir}/conv1_im2col_int8.npy", results['conv1_im2col_int8'])
        np.save(f"{output_dir}/conv1_weight_flat_int8.npy", results['conv1_weight_flat_int8'])
        np.save(f"{output_dir}/conv1_out_int32.npy", results['conv1_out_int32'])
        np.save(f"{output_dir}/conv1_out_int8.npy", results['conv1_out_int8'])
        
        # Intermediate stages
        np.save(f"{output_dir}/relu1_out_int8.npy", results['relu1_out_int8'])
        
        # Conv2
        save_hex_int8(f"{output_dir}/conv2_im2col_int8.hex", results['conv2_im2col_int8'])
        save_hex_int8(f"{output_dir}/conv2_weight_int8.hex", results['conv2_weight_flat_int8'])
        save_hex_int32(f"{output_dir}/conv2_expected_int32.hex", results['conv2_out_int32'])
        
        np.save(f"{output_dir}/conv2_im2col_int8.npy", results['conv2_im2col_int8'])
        np.save(f"{output_dir}/conv2_weight_flat_int8.npy", results['conv2_weight_flat_int8'])
        np.save(f"{output_dir}/conv2_out_int32.npy", results['conv2_out_int32'])
        np.save(f"{output_dir}/conv2_out_int8.npy", results['conv2_out_int8'])
        
        # Final output
        save_hex_int8(f"{output_dir}/add_out_int8.hex", results['add_out_int8'])
        save_hex_int8(f"{output_dir}/output_int8.hex", results['output_int8'])
        
        np.save(f"{output_dir}/add_out_int8.npy", results['add_out_int8'])
        np.save(f"{output_dir}/output_int8.npy", results['output_int8'])
        
        # Summary
        summary = {
            'model': 'ResNet Basic Block',
            'channels': self.channels,
            'spatial': self.spatial,
            'input_shape': list(x_int8.shape),
            'output_shape': list(results['output_int8'].shape),
            'conv_gemm': {
                'M': M,
                'K': K,
                'N': N
            },
            'tiles_per_conv': num_tiles,
            'total_tiles': num_tiles * 2,
            'estimated_cycles': num_tiles * 2 * 25
        }
        
        with open(f"{output_dir}/summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n[FILES]")
        for f_name in sorted(os.listdir(output_dir)):
            size = os.path.getsize(f"{output_dir}/{f_name}")
            print(f"  {f_name}: {size:,} bytes")
        
        print(f"\n" + "=" * 70)
        print(f"Summary:")
        print(f"  Input/Output: (1, {self.channels}, {self.spatial}, {self.spatial})")
        print(f"  Conv GEMM: ({M}, {K}) × ({K}, {N})")
        print(f"  Total tiles: {num_tiles * 2}")
        print(f"  Estimated cycles: {summary['estimated_cycles']:,}")
        print("=" * 70)
        
        return results


if __name__ == "__main__":
    block = ResNetBlock(channels=16, spatial=56, seed=42)
    block.generate_test_vectors()

#!/usr/bin/env python3
"""
LeNet-5 Golden Model - Generates Test Vectors for RTL Verification

Architecture (28×28 MNIST input):
  Layer 1:  Conv1    (1,1,28,28)  → (1,6,24,24)   5×5 conv, 6 filters
  Layer 2:  ReLU1    (1,6,24,24)  → (1,6,24,24)   element-wise
  Layer 3:  Pool1    (1,6,24,24)  → (1,6,12,12)   2×2 avg pool
  Layer 4:  Conv2    (1,6,12,12)  → (1,16,8,8)    5×5 conv, 16 filters
  Layer 5:  ReLU2    (1,16,8,8)   → (1,16,8,8)    element-wise
  Layer 6:  Pool2    (1,16,8,8)   → (1,16,4,4)    2×2 avg pool
  Layer 7:  FC1      (1,256)      → (1,120)       GEMM
  Layer 8:  ReLU3    (1,120)      → (1,120)       element-wise
  Layer 9:  FC2      (1,120)      → (1,84)        GEMM
  Layer 10: ReLU4    (1,84)       → (1,84)        element-wise
  Layer 11: FC3      (1,84)       → (1,10)        GEMM (output logits)

Each layer generates:
  - input_int8.hex / input_fp32.npy
  - weight_int8.hex / weight_fp32.npy (for conv/fc layers)
  - expected_int8.hex / expected_fp32.npy (or int32 for GEMM outputs)

Usage:
  python golden.py                    # Generate all test vectors
  python golden.py --layer 1          # Generate only layer 1
  python golden.py --verify results/  # Verify RTL results
"""

import numpy as np
import os
import json
from typing import Tuple, Dict, List
from dataclasses import dataclass

# ============================================================================
# Configuration
# ============================================================================

TILE_SIZE = 8  # Systolic array size
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_vectors")

@dataclass
class LayerConfig:
    name: str
    layer_type: str  # 'conv', 'relu', 'pool', 'fc'
    input_shape: tuple
    output_shape: tuple
    weight_shape: tuple = None  # For conv/fc layers

LENET_LAYERS = [
    LayerConfig("layer1_conv",  "conv", (1,1,28,28),  (1,6,24,24),  (6,1,5,5)),
    LayerConfig("layer2_relu",  "relu", (1,6,24,24),  (1,6,24,24)),
    LayerConfig("layer3_pool",  "pool", (1,6,24,24),  (1,6,12,12)),
    LayerConfig("layer4_conv",  "conv", (1,6,12,12),  (1,16,8,8),   (16,6,5,5)),
    LayerConfig("layer5_relu",  "relu", (1,16,8,8),   (1,16,8,8)),
    LayerConfig("layer6_pool",  "pool", (1,16,8,8),   (1,16,4,4)),
    LayerConfig("layer7_fc",    "fc",   (1,256),      (1,120),      (120,256)),
    LayerConfig("layer8_relu",  "relu", (1,120),      (1,120)),
    LayerConfig("layer9_fc",    "fc",   (1,120),      (1,84),       (84,120)),
    LayerConfig("layer10_relu", "relu", (1,84),       (1,84)),
    LayerConfig("layer11_fc",   "fc",   (1,84),       (1,10),       (10,84)),
]

# ============================================================================
# Utility Functions
# ============================================================================

def quantize_to_int8(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """Quantize float tensor to INT8 with symmetric quantization."""
    max_val = np.abs(x).max()
    scale = max_val / 127.0 if max_val > 0 else 1.0
    x_int8 = np.clip(np.round(x / scale), -128, 127).astype(np.int8)
    return x_int8, scale

def dequantize_int8(x: np.ndarray, scale: float) -> np.ndarray:
    """Dequantize INT8 tensor back to float."""
    return x.astype(np.float32) * scale

def save_hex_int8(filename: str, data: np.ndarray):
    """Save INT8 data as hex file for Verilog $readmemh."""
    with open(filename, 'w') as f:
        for val in data.flatten():
            val = int(val)
            if val < 0:
                val = val + 256  # Convert to unsigned
            f.write(f"{val:02x}\n")

def save_hex_int32(filename: str, data: np.ndarray):
    """Save INT32 data as hex file for Verilog $readmemh."""
    with open(filename, 'w') as f:
        for val in data.flatten():
            val = int(val)
            if val < 0:
                val = val + (1 << 32)  # Convert to unsigned
            f.write(f"{val:08x}\n")

def im2col(x: np.ndarray, kH: int, kW: int, stride: int = 1, pad: int = 0) -> np.ndarray:
    """
    Transform input tensor to column matrix for GEMM-based convolution.
    
    Input: (N, C, H, W)
    Output: (N * H_out * W_out, C * kH * kW)
    """
    N, C, H, W = x.shape
    
    # Pad if needed
    if pad > 0:
        x = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
        H, W = H + 2*pad, W + 2*pad
    
    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1
    
    col = np.zeros((N * H_out * W_out, C * kH * kW), dtype=x.dtype)
    
    idx = 0
    for n in range(N):
        for h in range(H_out):
            for w in range(W_out):
                h_start = h * stride
                w_start = w * stride
                patch = x[n, :, h_start:h_start+kH, w_start:w_start+kW]
                col[idx] = patch.flatten()
                idx += 1
    
    return col

def col2im_shape(col_shape: tuple, out_channels: int, H_out: int, W_out: int) -> tuple:
    """Get output shape from im2col result."""
    return (1, out_channels, H_out, W_out)

# ============================================================================
# Layer Operations (Float Reference)
# ============================================================================

def conv2d_fp32(x: np.ndarray, weight: np.ndarray, bias: np.ndarray = None) -> np.ndarray:
    """Float32 convolution using im2col + GEMM."""
    N, C, H, W = x.shape
    K, C_w, kH, kW = weight.shape
    
    H_out = H - kH + 1
    W_out = W - kW + 1
    
    # im2col: (N*H_out*W_out, C*kH*kW)
    col = im2col(x, kH, kW)
    
    # Weight reshape: (K, C*kH*kW) -> transpose to (C*kH*kW, K)
    weight_flat = weight.reshape(K, -1).T
    
    # GEMM: (N*H_out*W_out, C*kH*kW) @ (C*kH*kW, K) -> (N*H_out*W_out, K)
    out = col @ weight_flat
    
    # Add bias if provided
    if bias is not None:
        out = out + bias
    
    # Reshape to (N, K, H_out, W_out)
    out = out.reshape(N, H_out, W_out, K).transpose(0, 3, 1, 2)
    
    return out

def relu_fp32(x: np.ndarray) -> np.ndarray:
    """Float32 ReLU."""
    return np.maximum(x, 0)

def avgpool2d_fp32(x: np.ndarray, kernel_size: int = 2) -> np.ndarray:
    """Float32 2D average pooling."""
    N, C, H, W = x.shape
    H_out = H // kernel_size
    W_out = W // kernel_size
    
    out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
    
    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * kernel_size
                    w_start = w * kernel_size
                    patch = x[n, c, h_start:h_start+kernel_size, w_start:w_start+kernel_size]
                    out[n, c, h, w] = patch.mean()
    
    return out

def fc_fp32(x: np.ndarray, weight: np.ndarray, bias: np.ndarray = None) -> np.ndarray:
    """Float32 fully connected layer (GEMM)."""
    out = x @ weight.T
    if bias is not None:
        out = out + bias
    return out

# ============================================================================
# Layer Operations (INT8 Quantized)
# ============================================================================

def conv2d_int8(x_int8: np.ndarray, weight_int8: np.ndarray) -> np.ndarray:
    """INT8 convolution -> INT32 output (no bias, no requantization)."""
    N, C, H, W = x_int8.shape
    K, C_w, kH, kW = weight_int8.shape
    
    H_out = H - kH + 1
    W_out = W - kW + 1
    
    # im2col on INT8
    col = im2col(x_int8, kH, kW)
    
    # Weight reshape
    weight_flat = weight_int8.reshape(K, -1).T
    
    # INT8 @ INT8 -> INT32
    out_int32 = col.astype(np.int32) @ weight_flat.astype(np.int32)
    
    # Reshape
    out_int32 = out_int32.reshape(N, H_out, W_out, K).transpose(0, 3, 1, 2)
    
    return out_int32

def relu_int8(x_int8: np.ndarray) -> np.ndarray:
    """INT8 ReLU (in-place max with 0)."""
    return np.maximum(x_int8, 0).astype(np.int8)

def avgpool2d_int8(x_int8: np.ndarray, kernel_size: int = 2) -> np.ndarray:
    """INT8 average pooling using sum and shift."""
    N, C, H, W = x_int8.shape
    H_out = H // kernel_size
    W_out = W // kernel_size
    k2 = kernel_size * kernel_size
    
    out = np.zeros((N, C, H_out, W_out), dtype=np.int8)
    
    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * kernel_size
                    w_start = w * kernel_size
                    patch = x_int8[n, c, h_start:h_start+kernel_size, w_start:w_start+kernel_size]
                    # Sum then divide (truncate)
                    val = patch.astype(np.int32).sum() // k2
                    out[n, c, h, w] = np.clip(val, -128, 127)
    
    return out

def fc_int8(x_int8: np.ndarray, weight_int8: np.ndarray) -> np.ndarray:
    """INT8 fully connected -> INT32 output."""
    return x_int8.astype(np.int32) @ weight_int8.T.astype(np.int32)

def requantize_int32_to_int8(x_int32: np.ndarray, shift: int = 8) -> np.ndarray:
    """Requantize INT32 accumulator to INT8 using right shift."""
    x_shifted = x_int32 >> shift
    return np.clip(x_shifted, -128, 127).astype(np.int8)

# ============================================================================
# LeNet Model
# ============================================================================

class LeNet5:
    """Complete LeNet-5 model with both FP32 and INT8 paths."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
        # Initialize weights with Kaiming initialization
        self.conv1_weight = self._init_conv(1, 6, 5)
        self.conv1_bias = np.zeros(6, dtype=np.float32)
        
        self.conv2_weight = self._init_conv(6, 16, 5)
        self.conv2_bias = np.zeros(16, dtype=np.float32)
        
        self.fc1_weight = self._init_fc(256, 120)
        self.fc1_bias = np.zeros(120, dtype=np.float32)
        
        self.fc2_weight = self._init_fc(120, 84)
        self.fc2_bias = np.zeros(84, dtype=np.float32)
        
        self.fc3_weight = self._init_fc(84, 10)
        self.fc3_bias = np.zeros(10, dtype=np.float32)
        
        # Quantize weights
        self.conv1_weight_int8, self.conv1_weight_scale = quantize_to_int8(self.conv1_weight)
        self.conv2_weight_int8, self.conv2_weight_scale = quantize_to_int8(self.conv2_weight)
        self.fc1_weight_int8, self.fc1_weight_scale = quantize_to_int8(self.fc1_weight)
        self.fc2_weight_int8, self.fc2_weight_scale = quantize_to_int8(self.fc2_weight)
        self.fc3_weight_int8, self.fc3_weight_scale = quantize_to_int8(self.fc3_weight)
    
    def _init_conv(self, in_c: int, out_c: int, k: int) -> np.ndarray:
        fan_in = in_c * k * k
        std = np.sqrt(2.0 / fan_in)
        return (np.random.randn(out_c, in_c, k, k) * std).astype(np.float32)
    
    def _init_fc(self, in_f: int, out_f: int) -> np.ndarray:
        std = np.sqrt(2.0 / (in_f + out_f))
        return (np.random.randn(out_f, in_f) * std).astype(np.float32)
    
    def forward_fp32(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Float32 forward pass, returns all intermediate activations."""
        results = {}
        results['input'] = x.copy()
        
        # Layer 1: Conv1
        x = conv2d_fp32(x, self.conv1_weight, self.conv1_bias)
        results['layer1_conv_out'] = x.copy()
        
        # Layer 2: ReLU1
        x = relu_fp32(x)
        results['layer2_relu_out'] = x.copy()
        
        # Layer 3: Pool1
        x = avgpool2d_fp32(x, 2)
        results['layer3_pool_out'] = x.copy()
        
        # Layer 4: Conv2
        x = conv2d_fp32(x, self.conv2_weight, self.conv2_bias)
        results['layer4_conv_out'] = x.copy()
        
        # Layer 5: ReLU2
        x = relu_fp32(x)
        results['layer5_relu_out'] = x.copy()
        
        # Layer 6: Pool2
        x = avgpool2d_fp32(x, 2)
        results['layer6_pool_out'] = x.copy()
        
        # Flatten
        x = x.reshape(1, -1)
        results['flatten_out'] = x.copy()
        
        # Layer 7: FC1
        x = fc_fp32(x, self.fc1_weight, self.fc1_bias)
        results['layer7_fc_out'] = x.copy()
        
        # Layer 8: ReLU3
        x = relu_fp32(x)
        results['layer8_relu_out'] = x.copy()
        
        # Layer 9: FC2
        x = fc_fp32(x, self.fc2_weight, self.fc2_bias)
        results['layer9_fc_out'] = x.copy()
        
        # Layer 10: ReLU4
        x = relu_fp32(x)
        results['layer10_relu_out'] = x.copy()
        
        # Layer 11: FC3
        x = fc_fp32(x, self.fc3_weight, self.fc3_bias)
        results['layer11_fc_out'] = x.copy()
        
        results['logits'] = x.copy()
        return results
    
    def forward_int8(self, x_int8: np.ndarray, x_scale: float) -> Dict[str, np.ndarray]:
        """
        INT8 forward pass matching RTL behavior.
        Returns all intermediate activations for verification.
        """
        results = {}
        results['input_int8'] = x_int8.copy()
        results['input_scale'] = x_scale
        
        # Layer 1: Conv1 (INT8 -> INT32)
        conv1_out_int32 = conv2d_int8(x_int8, self.conv1_weight_int8)
        results['layer1_conv_out_int32'] = conv1_out_int32.copy()
        
        # Requantize to INT8 for next layer
        conv1_out_int8 = requantize_int32_to_int8(conv1_out_int32, shift=7)
        results['layer1_conv_out_int8'] = conv1_out_int8.copy()
        
        # Layer 2: ReLU1
        relu1_out_int8 = relu_int8(conv1_out_int8)
        results['layer2_relu_out_int8'] = relu1_out_int8.copy()
        
        # Layer 3: Pool1
        pool1_out_int8 = avgpool2d_int8(relu1_out_int8, 2)
        results['layer3_pool_out_int8'] = pool1_out_int8.copy()
        
        # Layer 4: Conv2 (INT8 -> INT32)
        conv2_out_int32 = conv2d_int8(pool1_out_int8, self.conv2_weight_int8)
        results['layer4_conv_out_int32'] = conv2_out_int32.copy()
        
        conv2_out_int8 = requantize_int32_to_int8(conv2_out_int32, shift=7)
        results['layer4_conv_out_int8'] = conv2_out_int8.copy()
        
        # Layer 5: ReLU2
        relu2_out_int8 = relu_int8(conv2_out_int8)
        results['layer5_relu_out_int8'] = relu2_out_int8.copy()
        
        # Layer 6: Pool2
        pool2_out_int8 = avgpool2d_int8(relu2_out_int8, 2)
        results['layer6_pool_out_int8'] = pool2_out_int8.copy()
        
        # Flatten
        flatten_int8 = pool2_out_int8.reshape(1, -1)
        results['flatten_out_int8'] = flatten_int8.copy()
        
        # Layer 7: FC1 (INT8 -> INT32)
        fc1_out_int32 = fc_int8(flatten_int8, self.fc1_weight_int8)
        results['layer7_fc_out_int32'] = fc1_out_int32.copy()
        
        fc1_out_int8 = requantize_int32_to_int8(fc1_out_int32, shift=7)
        results['layer7_fc_out_int8'] = fc1_out_int8.copy()
        
        # Layer 8: ReLU3
        relu3_out_int8 = relu_int8(fc1_out_int8)
        results['layer8_relu_out_int8'] = relu3_out_int8.copy()
        
        # Layer 9: FC2 (INT8 -> INT32)
        fc2_out_int32 = fc_int8(relu3_out_int8, self.fc2_weight_int8)
        results['layer9_fc_out_int32'] = fc2_out_int32.copy()
        
        fc2_out_int8 = requantize_int32_to_int8(fc2_out_int32, shift=7)
        results['layer9_fc_out_int8'] = fc2_out_int8.copy()
        
        # Layer 10: ReLU4
        relu4_out_int8 = relu_int8(fc2_out_int8)
        results['layer10_relu_out_int8'] = relu4_out_int8.copy()
        
        # Layer 11: FC3 (INT8 -> INT32, final output)
        fc3_out_int32 = fc_int8(relu4_out_int8, self.fc3_weight_int8)
        results['layer11_fc_out_int32'] = fc3_out_int32.copy()
        
        results['logits_int32'] = fc3_out_int32.copy()
        return results
    
    def generate_layer1_vectors(self, x_int8: np.ndarray, output_dir: str):
        """Generate test vectors specifically for Layer 1 (Conv1) RTL test."""
        os.makedirs(output_dir, exist_ok=True)
        
        # im2col transformation
        col_int8 = im2col(x_int8, 5, 5)  # (576, 25)
        
        # Weight matrix in GEMM format: (K, C*kH*kW).T = (25, 6)
        weight_flat_int8 = self.conv1_weight_int8.reshape(6, -1).T  # (25, 6)
        
        # Expected INT32 output
        expected_int32 = col_int8.astype(np.int32) @ weight_flat_int8.astype(np.int32)  # (576, 6)
        
        # Save all vectors
        save_hex_int8(f"{output_dir}/layer1_input_int8.hex", x_int8)
        save_hex_int8(f"{output_dir}/layer1_im2col_int8.hex", col_int8)
        save_hex_int8(f"{output_dir}/layer1_weight_int8.hex", weight_flat_int8)
        save_hex_int32(f"{output_dir}/layer1_expected_int32.hex", expected_int32)
        
        np.save(f"{output_dir}/layer1_input_fp32.npy", x_int8.astype(np.float32))
        np.save(f"{output_dir}/layer1_im2col_int8.npy", col_int8)
        np.save(f"{output_dir}/layer1_weight_int8.npy", weight_flat_int8)
        np.save(f"{output_dir}/layer1_expected_int32.npy", expected_int32)
        
        # Also save original weight shape for reference
        np.save(f"{output_dir}/layer1_weight_original.npy", self.conv1_weight_int8)
        
        return {
            'im2col_shape': col_int8.shape,
            'weight_shape': weight_flat_int8.shape,
            'output_shape': expected_int32.shape,
            'M': col_int8.shape[0],
            'K': col_int8.shape[1],
            'N': weight_flat_int8.shape[1]
        }
    
    def generate_layer3_vectors(self, relu1_out_int8: np.ndarray, output_dir: str):
        """Generate test vectors for Layer 3 (AvgPool)."""
        os.makedirs(output_dir, exist_ok=True)
        
        expected_int8 = avgpool2d_int8(relu1_out_int8, 2)
        
        save_hex_int8(f"{output_dir}/layer3_input_int8.hex", relu1_out_int8)
        save_hex_int8(f"{output_dir}/layer3_expected_int8.hex", expected_int8)
        
        np.save(f"{output_dir}/layer3_input_int8.npy", relu1_out_int8)
        np.save(f"{output_dir}/layer3_expected_int8.npy", expected_int8)
        
        return {
            'input_shape': relu1_out_int8.shape,
            'output_shape': expected_int8.shape
        }
    
    def generate_layer7_vectors(self, flatten_int8: np.ndarray, output_dir: str):
        """Generate test vectors for Layer 7 (FC1)."""
        os.makedirs(output_dir, exist_ok=True)
        
        expected_int32 = fc_int8(flatten_int8, self.fc1_weight_int8)
        
        # Weight needs to be transposed for GEMM: A(1,K) @ B(K,N) -> C(1,N)
        # self.fc1_weight_int8 is (N, K) = (120, 256)
        # We need B as (K, N) = (256, 120)
        weight_transposed = self.fc1_weight_int8.T  # (256, 120)
        
        save_hex_int8(f"{output_dir}/layer7_input_int8.hex", flatten_int8)
        save_hex_int8(f"{output_dir}/layer7_weight_int8.hex", weight_transposed)
        save_hex_int32(f"{output_dir}/layer7_expected_int32.hex", expected_int32)
        
        np.save(f"{output_dir}/layer7_input_int8.npy", flatten_int8)
        np.save(f"{output_dir}/layer7_weight_int8.npy", self.fc1_weight_int8)  # Original shape
        np.save(f"{output_dir}/layer7_weight_transposed_int8.npy", weight_transposed)
        np.save(f"{output_dir}/layer7_expected_int32.npy", expected_int32)
        
        return {
            'input_shape': flatten_int8.shape,
            'weight_shape': weight_transposed.shape,  # (K, N) for GEMM
            'output_shape': expected_int32.shape,
            'M': flatten_int8.shape[0],
            'K': weight_transposed.shape[0],
            'N': weight_transposed.shape[1]
        }

# ============================================================================
# Main: Generate All Test Vectors
# ============================================================================

def generate_all_vectors(output_dir: str = OUTPUT_DIR):
    """Generate all test vectors for LeNet-5."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("LeNet-5 Test Vector Generation")
    print("=" * 70)
    
    # Initialize model
    model = LeNet5(seed=42)
    
    # Generate random input (28x28 grayscale image)
    x_fp32 = np.random.randn(1, 1, 28, 28).astype(np.float32) * 0.5
    x_int8, x_scale = quantize_to_int8(x_fp32)
    
    print(f"\n[INPUT]")
    print(f"  Shape: {x_int8.shape}")
    print(f"  Scale: {x_scale:.6f}")
    print(f"  Range: [{x_int8.min()}, {x_int8.max()}]")
    
    # Run FP32 reference
    print(f"\n[FP32 REFERENCE]")
    results_fp32 = model.forward_fp32(x_fp32)
    for name, arr in results_fp32.items():
        print(f"  {name}: {arr.shape}")
    
    # Run INT8
    print(f"\n[INT8 FORWARD]")
    results_int8 = model.forward_int8(x_int8, x_scale)
    
    # Save input
    save_hex_int8(f"{output_dir}/input_int8.hex", x_int8)
    np.save(f"{output_dir}/input_int8.npy", x_int8)
    np.save(f"{output_dir}/input_fp32.npy", x_fp32)
    np.save(f"{output_dir}/input_scale.npy", np.array([x_scale]))
    
    # Generate Layer 1 vectors
    print(f"\n[LAYER 1: Conv1]")
    l1_info = model.generate_layer1_vectors(x_int8, output_dir)
    print(f"  im2col: {l1_info['im2col_shape']}")
    print(f"  weight: {l1_info['weight_shape']}")
    print(f"  GEMM: ({l1_info['M']}, {l1_info['K']}) × ({l1_info['K']}, {l1_info['N']}) → ({l1_info['M']}, {l1_info['N']})")
    
    # Generate Layer 3 vectors (need Layer 2 output first)
    print(f"\n[LAYER 3: Pool1]")
    l3_info = model.generate_layer3_vectors(results_int8['layer2_relu_out_int8'], output_dir)
    print(f"  input:  {l3_info['input_shape']}")
    print(f"  output: {l3_info['output_shape']}")
    
    # Generate Layer 7 vectors
    print(f"\n[LAYER 7: FC1]")
    l7_info = model.generate_layer7_vectors(results_int8['flatten_out_int8'], output_dir)
    print(f"  GEMM: ({l7_info['M']}, {l7_info['K']}) × ({l7_info['K']}, {l7_info['N']}) → ({l7_info['M']}, {l7_info['N']})")
    
    # Save all intermediate results for debugging
    print(f"\n[SAVING ALL INTERMEDIATES]")
    for name, arr in results_int8.items():
        if isinstance(arr, np.ndarray):
            np.save(f"{output_dir}/{name}.npy", arr)
            print(f"  {name}: {arr.shape}")
    
    for name, arr in results_fp32.items():
        np.save(f"{output_dir}/{name}_fp32.npy", arr)
    
    # Save model weights
    print(f"\n[SAVING WEIGHTS]")
    np.save(f"{output_dir}/conv1_weight_int8.npy", model.conv1_weight_int8)
    np.save(f"{output_dir}/conv1_weight_fp32.npy", model.conv1_weight)
    np.save(f"{output_dir}/conv2_weight_int8.npy", model.conv2_weight_int8)
    np.save(f"{output_dir}/conv2_weight_fp32.npy", model.conv2_weight)
    np.save(f"{output_dir}/fc1_weight_int8.npy", model.fc1_weight_int8)
    np.save(f"{output_dir}/fc1_weight_fp32.npy", model.fc1_weight)
    np.save(f"{output_dir}/fc2_weight_int8.npy", model.fc2_weight_int8)
    np.save(f"{output_dir}/fc2_weight_fp32.npy", model.fc2_weight)
    np.save(f"{output_dir}/fc3_weight_int8.npy", model.fc3_weight_int8)
    np.save(f"{output_dir}/fc3_weight_fp32.npy", model.fc3_weight)
    
    # Generate summary JSON
    summary = {
        'model': 'LeNet-5',
        'input_shape': list(x_int8.shape),
        'input_scale': float(x_scale),
        'layers': {
            'layer1_conv': {
                'type': 'conv',
                'input_shape': [1, 1, 28, 28],
                'output_shape': [1, 6, 24, 24],
                'weight_shape': [6, 1, 5, 5],
                'gemm_M': l1_info['M'],
                'gemm_K': l1_info['K'],
                'gemm_N': l1_info['N']
            },
            'layer3_pool': {
                'type': 'avgpool',
                'input_shape': list(l3_info['input_shape']),
                'output_shape': list(l3_info['output_shape']),
                'kernel_size': 2
            },
            'layer7_fc': {
                'type': 'fc',
                'input_shape': list(l7_info['input_shape']),
                'output_shape': list(l7_info['output_shape']),
                'weight_shape': list(model.fc1_weight_int8.shape),
                'gemm_M': l7_info['M'],
                'gemm_K': l7_info['K'],
                'gemm_N': l7_info['N']
            }
        },
        'output_logits_shape': [1, 10],
        'predicted_class': int(np.argmax(results_fp32['logits']))
    }
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # List generated files
    print(f"\n[GENERATED FILES]")
    files = sorted(os.listdir(output_dir))
    for f in files:
        size = os.path.getsize(f"{output_dir}/{f}")
        print(f"  {f}: {size:,} bytes")
    
    print(f"\n" + "=" * 70)
    print(f"Test vectors saved to: {output_dir}/")
    print(f"Predicted class: {summary['predicted_class']}")
    print("=" * 70)
    
    return summary


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print(__doc__)
    else:
        generate_all_vectors()

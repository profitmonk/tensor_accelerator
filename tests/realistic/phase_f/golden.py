#!/usr/bin/env python3
"""
Phase F: Full Model End-to-End Golden Model

This module implements complete model inference:
1. LeNet-5 full inference (all 11 layers)
2. ResNet-18 basic block with full quantization pipeline
3. Multi-batch inference

All computations use INT8 quantization with proper scaling.
"""

import numpy as np
import os
import json
from typing import Tuple, List, Dict

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_vectors")

# ============================================================================
# Quantization Helpers
# ============================================================================

def quantize_int8(x: np.ndarray, scale: float) -> np.ndarray:
    """Quantize FP32 to INT8."""
    return np.clip(np.round(x / scale), -128, 127).astype(np.int8)

def dequantize(x: np.ndarray, scale: float) -> np.ndarray:
    """Dequantize INT8 to FP32."""
    return x.astype(np.float32) * scale

def requantize_int32_to_int8(x: np.ndarray, shift: int, apply_relu: bool = False) -> np.ndarray:
    """Requantize INT32 accumulator to INT8."""
    # Arithmetic right shift
    result = x >> shift
    # Saturate
    result = np.clip(result, -128, 127)
    # Optional ReLU
    if apply_relu:
        result = np.maximum(result, 0)
    return result.astype(np.int8)

# ============================================================================
# Layer Operations
# ============================================================================

def conv2d_int8(
    x: np.ndarray,      # (N, C_in, H, W) INT8
    weight: np.ndarray, # (C_out, C_in, kH, kW) INT8
    bias: np.ndarray,   # (C_out,) INT32
    stride: int = 1,
    padding: int = 0,
    shift: int = 8,
    relu: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    INT8 Conv2D with bias and requantization.
    Returns (output_int8, output_int32_raw)
    """
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape
    
    # Output dimensions
    H_out = (H + 2*padding - kH) // stride + 1
    W_out = (W + 2*padding - kW) // stride + 1
    
    # Pad input
    if padding > 0:
        x_pad = np.pad(x, ((0,0), (0,0), (padding,padding), (padding,padding)), mode='constant')
    else:
        x_pad = x
    
    # Compute convolution
    output_int32 = np.zeros((N, C_out, H_out, W_out), dtype=np.int32)
    
    for n in range(N):
        for c_out in range(C_out):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride
                    w_start = w * stride
                    
                    # Extract patch
                    patch = x_pad[n, :, h_start:h_start+kH, w_start:w_start+kW]
                    
                    # Compute dot product
                    acc = np.sum(patch.astype(np.int32) * weight[c_out].astype(np.int32))
                    
                    # Add bias
                    acc += bias[c_out]
                    
                    output_int32[n, c_out, h, w] = acc
    
    # Requantize
    output_int8 = requantize_int32_to_int8(output_int32, shift, relu)
    
    return output_int8, output_int32


def maxpool2d(x: np.ndarray, kernel: int = 2, stride: int = 2) -> np.ndarray:
    """Max pooling (works with any dtype)."""
    N, C, H, W = x.shape
    H_out = (H - kernel) // stride + 1
    W_out = (W - kernel) // stride + 1
    
    output = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
    
    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride
                    w_start = w * stride
                    output[n, c, h, w] = np.max(x[n, c, h_start:h_start+kernel, w_start:w_start+kernel])
    
    return output


def avgpool2d_int8(x: np.ndarray, kernel: int = 2, stride: int = 2) -> np.ndarray:
    """Average pooling for INT8 (uses right shift for divide)."""
    N, C, H, W = x.shape
    H_out = (H - kernel) // stride + 1
    W_out = (W - kernel) // stride + 1
    
    output = np.zeros((N, C, H_out, W_out), dtype=np.int8)
    
    # Shift for averaging (kernel^2)
    shift = int(np.log2(kernel * kernel))
    
    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride
                    w_start = w * stride
                    patch = x[n, c, h_start:h_start+kernel, w_start:w_start+kernel]
                    # Sum and divide
                    acc = np.sum(patch.astype(np.int16))
                    output[n, c, h, w] = np.clip(acc >> shift, -128, 127)
    
    return output


def fc_int8(
    x: np.ndarray,      # (N, in_features) INT8
    weight: np.ndarray, # (out_features, in_features) INT8
    bias: np.ndarray,   # (out_features,) INT32
    shift: int = 8,
    relu: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fully connected layer with INT8 input/output.
    Returns (output_int8, output_int32_raw)
    """
    # Matrix multiply
    output_int32 = x.astype(np.int32) @ weight.T.astype(np.int32)
    
    # Add bias
    output_int32 = output_int32 + bias
    
    # Requantize
    output_int8 = requantize_int32_to_int8(output_int32, shift, relu)
    
    return output_int8, output_int32


# ============================================================================
# LeNet-5 Full Model
# ============================================================================

class LeNet5_INT8:
    """
    LeNet-5 architecture:
    Input: (1, 1, 28, 28) - MNIST
    
    Layer 1: Conv2D(1, 6, 5×5) + ReLU     → (1, 6, 24, 24)
    Layer 2: MaxPool(2×2)                  → (1, 6, 12, 12)
    Layer 3: Conv2D(6, 16, 5×5) + ReLU    → (1, 16, 8, 8)
    Layer 4: MaxPool(2×2)                  → (1, 16, 4, 4)
    Layer 5: Flatten                       → (1, 256)
    Layer 6: FC(256, 120) + ReLU          → (1, 120)
    Layer 7: FC(120, 84) + ReLU           → (1, 84)
    Layer 8: FC(84, 10)                   → (1, 10) - logits
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
        # Initialize weights (random for testing)
        self.conv1_weight = np.random.randint(-30, 30, size=(6, 1, 5, 5), dtype=np.int8)
        self.conv1_bias = np.random.randint(-500, 500, size=(6,), dtype=np.int32)
        
        self.conv2_weight = np.random.randint(-30, 30, size=(16, 6, 5, 5), dtype=np.int8)
        self.conv2_bias = np.random.randint(-500, 500, size=(16,), dtype=np.int32)
        
        self.fc1_weight = np.random.randint(-30, 30, size=(120, 256), dtype=np.int8)
        self.fc1_bias = np.random.randint(-1000, 1000, size=(120,), dtype=np.int32)
        
        self.fc2_weight = np.random.randint(-30, 30, size=(84, 120), dtype=np.int8)
        self.fc2_bias = np.random.randint(-1000, 1000, size=(84,), dtype=np.int32)
        
        self.fc3_weight = np.random.randint(-30, 30, size=(10, 84), dtype=np.int8)
        self.fc3_bias = np.random.randint(-1000, 1000, size=(10,), dtype=np.int32)
        
        # Requantization shifts
        self.shifts = {
            'conv1': 8, 'conv2': 8,
            'fc1': 8, 'fc2': 8, 'fc3': 8
        }
    
    def forward(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Full forward pass with all intermediate outputs.
        x: (N, 1, 28, 28) INT8 input
        """
        outputs = {'input': x}
        
        # Conv1 + ReLU
        conv1_out, conv1_acc = conv2d_int8(
            x, self.conv1_weight, self.conv1_bias,
            shift=self.shifts['conv1'], relu=True
        )
        outputs['conv1_acc'] = conv1_acc
        outputs['conv1'] = conv1_out
        
        # Pool1 (MaxPool)
        pool1_out = maxpool2d(conv1_out)
        outputs['pool1'] = pool1_out
        
        # Conv2 + ReLU
        conv2_out, conv2_acc = conv2d_int8(
            pool1_out, self.conv2_weight, self.conv2_bias,
            shift=self.shifts['conv2'], relu=True
        )
        outputs['conv2_acc'] = conv2_acc
        outputs['conv2'] = conv2_out
        
        # Pool2 (MaxPool)
        pool2_out = maxpool2d(conv2_out)
        outputs['pool2'] = pool2_out
        
        # Flatten
        N = x.shape[0]
        flatten_out = pool2_out.reshape(N, -1)
        outputs['flatten'] = flatten_out
        
        # FC1 + ReLU
        fc1_out, fc1_acc = fc_int8(
            flatten_out, self.fc1_weight, self.fc1_bias,
            shift=self.shifts['fc1'], relu=True
        )
        outputs['fc1_acc'] = fc1_acc
        outputs['fc1'] = fc1_out
        
        # FC2 + ReLU
        fc2_out, fc2_acc = fc_int8(
            fc1_out, self.fc2_weight, self.fc2_bias,
            shift=self.shifts['fc2'], relu=True
        )
        outputs['fc2_acc'] = fc2_acc
        outputs['fc2'] = fc2_out
        
        # FC3 (no ReLU - logits)
        fc3_out, fc3_acc = fc_int8(
            fc2_out, self.fc3_weight, self.fc3_bias,
            shift=self.shifts['fc3'], relu=False
        )
        outputs['fc3_acc'] = fc3_acc
        outputs['logits'] = fc3_out
        
        return outputs


# ============================================================================
# ResNet Basic Block
# ============================================================================

class ResNetBlock_INT8:
    """
    ResNet-18 basic block:
    
    Input: (N, C, H, W)
    
    Path A (main):
        Conv1: 3×3, same channels, stride=1, pad=1 + ReLU
        Conv2: 3×3, same channels, stride=1, pad=1 (no ReLU)
    
    Path B (residual):
        Identity (same as input)
    
    Output = ReLU(Path_A + Path_B)
    """
    
    def __init__(self, channels: int = 16, seed: int = 42):
        np.random.seed(seed)
        
        self.channels = channels
        
        # Conv1: (C, C, 3, 3)
        self.conv1_weight = np.random.randint(-20, 20, size=(channels, channels, 3, 3), dtype=np.int8)
        self.conv1_bias = np.random.randint(-500, 500, size=(channels,), dtype=np.int32)
        
        # Conv2: (C, C, 3, 3)
        self.conv2_weight = np.random.randint(-20, 20, size=(channels, channels, 3, 3), dtype=np.int8)
        self.conv2_bias = np.random.randint(-500, 500, size=(channels,), dtype=np.int32)
        
        self.shifts = {'conv1': 8, 'conv2': 8}
    
    def forward(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Forward pass.
        x: (N, C, H, W) INT8 input
        """
        outputs = {'input': x}
        
        # Conv1 + ReLU
        conv1_out, conv1_acc = conv2d_int8(
            x, self.conv1_weight, self.conv1_bias,
            padding=1, shift=self.shifts['conv1'], relu=True
        )
        outputs['conv1_acc'] = conv1_acc
        outputs['conv1'] = conv1_out
        
        # Conv2 (no ReLU before residual add)
        conv2_out, conv2_acc = conv2d_int8(
            conv1_out, self.conv2_weight, self.conv2_bias,
            padding=1, shift=self.shifts['conv2'], relu=False
        )
        outputs['conv2_acc'] = conv2_acc
        outputs['conv2'] = conv2_out
        
        # Residual add (in INT16 to avoid overflow, then clip)
        residual_add = conv2_out.astype(np.int16) + x.astype(np.int16)
        residual_add = np.clip(residual_add, -128, 127).astype(np.int8)
        outputs['residual_add'] = residual_add
        
        # Final ReLU
        output = np.maximum(residual_add, 0).astype(np.int8)
        outputs['output'] = output
        
        return outputs


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


class PhaseF_Tests:
    """Generate test vectors for Phase F."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
    
    def test1_lenet5_full(self, output_dir: str):
        """Test 1: Complete LeNet-5 inference."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("TEST 1: LeNet-5 Full Inference")
        print("="*60)
        
        np.random.seed(self.seed)
        
        # Create model
        model = LeNet5_INT8(seed=self.seed + 100)
        
        # Generate input (MNIST-like)
        x = np.random.randint(-50, 50, size=(1, 1, 28, 28), dtype=np.int8)
        
        print(f"\n  Input: {x.shape}")
        
        # Forward pass
        outputs = model.forward(x)
        
        # Print layer shapes
        print(f"  Conv1: {outputs['conv1'].shape}, range [{outputs['conv1'].min()}, {outputs['conv1'].max()}]")
        print(f"  Pool1: {outputs['pool1'].shape}")
        print(f"  Conv2: {outputs['conv2'].shape}, range [{outputs['conv2'].min()}, {outputs['conv2'].max()}]")
        print(f"  Pool2: {outputs['pool2'].shape}")
        print(f"  Flatten: {outputs['flatten'].shape}")
        print(f"  FC1: {outputs['fc1'].shape}, range [{outputs['fc1'].min()}, {outputs['fc1'].max()}]")
        print(f"  FC2: {outputs['fc2'].shape}, range [{outputs['fc2'].min()}, {outputs['fc2'].max()}]")
        print(f"  Logits: {outputs['logits'].shape}, range [{outputs['logits'].min()}, {outputs['logits'].max()}]")
        
        # Predicted class
        pred = np.argmax(outputs['logits'][0])
        print(f"\n  Predicted class: {pred}")
        
        # Save weights
        save_hex_int8(f"{output_dir}/test1_conv1_weight.hex", model.conv1_weight)
        save_hex_int32(f"{output_dir}/test1_conv1_bias.hex", model.conv1_bias)
        save_hex_int8(f"{output_dir}/test1_conv2_weight.hex", model.conv2_weight)
        save_hex_int32(f"{output_dir}/test1_conv2_bias.hex", model.conv2_bias)
        save_hex_int8(f"{output_dir}/test1_fc1_weight.hex", model.fc1_weight)
        save_hex_int32(f"{output_dir}/test1_fc1_bias.hex", model.fc1_bias)
        save_hex_int8(f"{output_dir}/test1_fc2_weight.hex", model.fc2_weight)
        save_hex_int32(f"{output_dir}/test1_fc2_bias.hex", model.fc2_bias)
        save_hex_int8(f"{output_dir}/test1_fc3_weight.hex", model.fc3_weight)
        save_hex_int32(f"{output_dir}/test1_fc3_bias.hex", model.fc3_bias)
        
        # Save input and outputs
        save_hex_int8(f"{output_dir}/test1_input.hex", x)
        save_hex_int8(f"{output_dir}/test1_conv1_out.hex", outputs['conv1'])
        save_hex_int8(f"{output_dir}/test1_pool1_out.hex", outputs['pool1'])
        save_hex_int8(f"{output_dir}/test1_conv2_out.hex", outputs['conv2'])
        save_hex_int8(f"{output_dir}/test1_pool2_out.hex", outputs['pool2'])
        save_hex_int8(f"{output_dir}/test1_fc1_out.hex", outputs['fc1'])
        save_hex_int8(f"{output_dir}/test1_fc2_out.hex", outputs['fc2'])
        save_hex_int8(f"{output_dir}/test1_logits.hex", outputs['logits'])
        
        # Save numpy arrays
        np.save(f"{output_dir}/test1_input.npy", x)
        np.save(f"{output_dir}/test1_conv1_weight.npy", model.conv1_weight)
        np.save(f"{output_dir}/test1_conv2_weight.npy", model.conv2_weight)
        np.save(f"{output_dir}/test1_fc1_weight.npy", model.fc1_weight)
        np.save(f"{output_dir}/test1_fc2_weight.npy", model.fc2_weight)
        np.save(f"{output_dir}/test1_fc3_weight.npy", model.fc3_weight)
        np.save(f"{output_dir}/test1_logits.npy", outputs['logits'])
        
        # Save intermediate outputs for verification
        for key, val in outputs.items():
            np.save(f"{output_dir}/test1_{key}.npy", val)
        
        return {
            'input_shape': list(x.shape),
            'output_shape': list(outputs['logits'].shape),
            'predicted_class': int(pred)
        }
    
    def test2_resnet_block(self, output_dir: str):
        """Test 2: ResNet basic block."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("TEST 2: ResNet Basic Block")
        print("="*60)
        
        np.random.seed(self.seed + 200)
        
        # Create block
        channels = 16
        block = ResNetBlock_INT8(channels=channels, seed=self.seed + 200)
        
        # Input: (1, 16, 14, 14) - typical ResNet dimensions
        H, W = 14, 14
        x = np.random.randint(-50, 50, size=(1, channels, H, W), dtype=np.int8)
        
        print(f"\n  Input: {x.shape}")
        
        # Forward pass
        outputs = block.forward(x)
        
        print(f"  Conv1: {outputs['conv1'].shape}, range [{outputs['conv1'].min()}, {outputs['conv1'].max()}]")
        print(f"  Conv2: {outputs['conv2'].shape}, range [{outputs['conv2'].min()}, {outputs['conv2'].max()}]")
        print(f"  Residual add: range [{outputs['residual_add'].min()}, {outputs['residual_add'].max()}]")
        print(f"  Output: {outputs['output'].shape}, range [{outputs['output'].min()}, {outputs['output'].max()}]")
        
        # Verify residual connection
        # output should have similar statistics to input where conv2 is ~0
        
        # Save weights
        save_hex_int8(f"{output_dir}/test2_conv1_weight.hex", block.conv1_weight)
        save_hex_int32(f"{output_dir}/test2_conv1_bias.hex", block.conv1_bias)
        save_hex_int8(f"{output_dir}/test2_conv2_weight.hex", block.conv2_weight)
        save_hex_int32(f"{output_dir}/test2_conv2_bias.hex", block.conv2_bias)
        
        # Save input and outputs
        save_hex_int8(f"{output_dir}/test2_input.hex", x)
        save_hex_int8(f"{output_dir}/test2_conv1_out.hex", outputs['conv1'])
        save_hex_int8(f"{output_dir}/test2_conv2_out.hex", outputs['conv2'])
        save_hex_int8(f"{output_dir}/test2_residual_add.hex", outputs['residual_add'])
        save_hex_int8(f"{output_dir}/test2_output.hex", outputs['output'])
        
        # Save numpy arrays
        np.save(f"{output_dir}/test2_input.npy", x)
        np.save(f"{output_dir}/test2_conv1_weight.npy", block.conv1_weight)
        np.save(f"{output_dir}/test2_conv2_weight.npy", block.conv2_weight)
        for key, val in outputs.items():
            np.save(f"{output_dir}/test2_{key}.npy", val)
        
        return {
            'channels': channels,
            'spatial': [H, W],
            'input_shape': list(x.shape),
            'output_shape': list(outputs['output'].shape)
        }
    
    def test3_batch_inference(self, output_dir: str):
        """Test 3: Multi-batch inference."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("TEST 3: Multi-Batch Inference (batch=4)")
        print("="*60)
        
        np.random.seed(self.seed + 300)
        
        # Create model (simple 2-layer MLP for batch testing)
        in_features = 64
        hidden = 32
        out_features = 10
        batch_size = 4
        
        # Weights
        W1 = np.random.randint(-30, 30, size=(hidden, in_features), dtype=np.int8)
        b1 = np.random.randint(-500, 500, size=(hidden,), dtype=np.int32)
        W2 = np.random.randint(-30, 30, size=(out_features, hidden), dtype=np.int8)
        b2 = np.random.randint(-500, 500, size=(out_features,), dtype=np.int32)
        
        # Input batch
        x = np.random.randint(-50, 50, size=(batch_size, in_features), dtype=np.int8)
        
        print(f"\n  Input: {x.shape}")
        print(f"  W1: {W1.shape}, W2: {W2.shape}")
        
        # Forward pass
        h, h_acc = fc_int8(x, W1, b1, shift=8, relu=True)
        y, y_acc = fc_int8(h, W2, b2, shift=8, relu=False)
        
        print(f"  Hidden: {h.shape}, range [{h.min()}, {h.max()}]")
        print(f"  Output: {y.shape}, range [{y.min()}, {y.max()}]")
        
        # Predictions for each batch element
        preds = np.argmax(y, axis=1)
        print(f"  Predictions: {preds}")
        
        # Save vectors
        save_hex_int8(f"{output_dir}/test3_input.hex", x)
        save_hex_int8(f"{output_dir}/test3_W1.hex", W1)
        save_hex_int32(f"{output_dir}/test3_b1.hex", b1)
        save_hex_int8(f"{output_dir}/test3_W2.hex", W2)
        save_hex_int32(f"{output_dir}/test3_b2.hex", b2)
        save_hex_int8(f"{output_dir}/test3_hidden.hex", h)
        save_hex_int8(f"{output_dir}/test3_output.hex", y)
        
        np.save(f"{output_dir}/test3_input.npy", x)
        np.save(f"{output_dir}/test3_W1.npy", W1)
        np.save(f"{output_dir}/test3_W2.npy", W2)
        np.save(f"{output_dir}/test3_hidden.npy", h)
        np.save(f"{output_dir}/test3_output.npy", y)
        
        return {
            'batch_size': batch_size,
            'in_features': in_features,
            'hidden': hidden,
            'out_features': out_features,
            'predictions': preds.tolist()
        }
    
    def generate_all(self):
        """Generate all test vectors."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        print("\n" + "="*60)
        print("PHASE F: Full Model E2E")
        print("="*60)
        
        results = {}
        results['test1_lenet5'] = self.test1_lenet5_full(OUTPUT_DIR)
        results['test2_resnet_block'] = self.test2_resnet_block(OUTPUT_DIR)
        results['test3_batch'] = self.test3_batch_inference(OUTPUT_DIR)
        
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
    tests = PhaseF_Tests(seed=42)
    tests.generate_all()

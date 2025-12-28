#!/usr/bin/env python3
"""
Test Vector Generator for Tensor Accelerator

Generates:
1. Input matrices (activations, weights)
2. Golden reference outputs (using NumPy)
3. Memory initialization files
4. Instruction sequences

Supports:
- GEMM (matrix multiplication)
- Conv2D (via im2col + GEMM)
- Attention (Q, K, V projections + attention computation)
"""

import numpy as np
import struct
import os
from typing import Tuple, List, Optional
from dataclasses import dataclass
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assembler.assembler import Assembler, Instruction, Opcode, TensorSubop, VectorSubop, DMASubop, SyncSubop

#==============================================================================
# Configuration
#==============================================================================

@dataclass
class TestConfig:
    """Test configuration parameters"""
    name: str
    array_size: int = 16
    data_width: int = 8
    acc_width: int = 32
    sram_base: int = 0x0000
    hbm_base: int = 0x80000000

#==============================================================================
# Data Generation
#==============================================================================

def generate_random_matrix(rows: int, cols: int, dtype: str = 'int8', 
                          scale: float = 1.0, seed: Optional[int] = None) -> np.ndarray:
    """Generate random matrix with specified shape and dtype"""
    if seed is not None:
        np.random.seed(seed)
    
    if dtype == 'int8':
        # INT8: -128 to 127, but we use smaller range for easier testing
        return (np.random.randn(rows, cols) * scale * 10).astype(np.int8)
    elif dtype == 'int16':
        return (np.random.randn(rows, cols) * scale * 100).astype(np.int16)
    elif dtype == 'float16':
        return (np.random.randn(rows, cols) * scale).astype(np.float16)
    elif dtype == 'float32':
        return (np.random.randn(rows, cols) * scale).astype(np.float32)
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

def generate_identity_matrix(size: int, dtype: str = 'int8') -> np.ndarray:
    """Generate identity matrix (for simple testing)"""
    return np.eye(size).astype(dtype)

def generate_sequential_matrix(rows: int, cols: int, dtype: str = 'int8') -> np.ndarray:
    """Generate matrix with sequential values (for debugging)"""
    return np.arange(rows * cols).reshape(rows, cols).astype(dtype)

#==============================================================================
# Golden Reference Computation
#==============================================================================

def golden_gemm(A: np.ndarray, B: np.ndarray, 
                accumulate: bool = False, 
                C: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute golden GEMM reference: C = A × B (+ C if accumulate)"""
    result = np.matmul(A.astype(np.int32), B.astype(np.int32))
    if accumulate and C is not None:
        result += C.astype(np.int32)
    return result.astype(np.int32)

def golden_conv2d_im2col(input: np.ndarray, weights: np.ndarray,
                         stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    Compute Conv2D using im2col transformation
    
    Args:
        input: [N, C, H, W] input tensor
        weights: [K, C, kH, kW] weight tensor
        stride: Convolution stride
        padding: Zero padding
    
    Returns:
        output: [N, K, H', W'] output tensor
    """
    N, C, H, W = input.shape
    K, _, kH, kW = weights.shape
    
    # Output dimensions
    H_out = (H + 2*padding - kH) // stride + 1
    W_out = (W + 2*padding - kW) // stride + 1
    
    # Pad input
    if padding > 0:
        input = np.pad(input, ((0,0), (0,0), (padding, padding), (padding, padding)))
    
    # im2col transformation
    # Output shape: [N * H_out * W_out, C * kH * kW]
    col = np.zeros((N, H_out * W_out, C * kH * kW), dtype=input.dtype)
    
    for n in range(N):
        idx = 0
        for h in range(H_out):
            for w in range(W_out):
                # Extract patch
                h_start = h * stride
                w_start = w * stride
                patch = input[n, :, h_start:h_start+kH, w_start:w_start+kW]
                col[n, idx, :] = patch.flatten()
                idx += 1
    
    col = col.reshape(N * H_out * W_out, C * kH * kW)
    
    # Reshape weights: [K, C*kH*kW]
    weights_col = weights.reshape(K, -1).T  # [C*kH*kW, K]
    
    # GEMM: [N*H'*W', C*kH*kW] × [C*kH*kW, K] -> [N*H'*W', K]
    output = golden_gemm(col.astype(np.int8), weights_col.astype(np.int8))
    
    # Reshape to [N, K, H', W']
    output = output.reshape(N, H_out, W_out, K).transpose(0, 3, 1, 2)
    
    return output

def golden_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                     scale: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute golden attention reference
    
    Attention(Q, K, V) = softmax(Q × K^T / sqrt(d_k)) × V
    
    Args:
        Q: [seq, d_k] Query matrix
        K: [seq, d_k] Key matrix  
        V: [seq, d_v] Value matrix
        scale: Scaling factor (default: 1/sqrt(d_k))
    
    Returns:
        output: [seq, d_v] Attention output
        attention_weights: [seq, seq] Attention weights
    """
    seq, d_k = Q.shape
    
    if scale is None:
        scale = 1.0 / np.sqrt(d_k)
    
    # Compute attention scores: Q × K^T
    scores = np.matmul(Q.astype(np.float32), K.T.astype(np.float32))
    
    # Scale
    scores = scores * scale
    
    # Softmax
    scores_exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
    
    # Output: attention_weights × V
    output = np.matmul(attention_weights, V.astype(np.float32))
    
    return output.astype(np.float32), attention_weights.astype(np.float32)

def golden_layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                     eps: float = 1e-5) -> np.ndarray:
    """Compute LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta"""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta

def golden_relu(x: np.ndarray) -> np.ndarray:
    """Compute ReLU activation"""
    return np.maximum(x, 0)

def golden_gelu(x: np.ndarray) -> np.ndarray:
    """Compute GELU activation (approximation)"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

#==============================================================================
# Memory File Generation
#==============================================================================

def matrix_to_bytes(matrix: np.ndarray, dtype: str = 'int8') -> bytes:
    """Convert matrix to bytes for memory initialization"""
    flat = matrix.flatten()
    
    if dtype == 'int8':
        return flat.astype(np.int8).tobytes()
    elif dtype == 'int16':
        return flat.astype(np.int16).tobytes()
    elif dtype == 'int32':
        return flat.astype(np.int32).tobytes()
    elif dtype == 'float16':
        return flat.astype(np.float16).tobytes()
    elif dtype == 'float32':
        return flat.astype(np.float32).tobytes()
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

def write_memh(filename: str, data: bytes, width: int = 256):
    """
    Write memory initialization file in Verilog $readmemh format
    
    Args:
        filename: Output file path
        data: Raw bytes to write
        width: Memory width in bits (256 = 32 bytes per line)
    """
    bytes_per_line = width // 8
    
    with open(filename, 'w') as f:
        for i in range(0, len(data), bytes_per_line):
            chunk = data[i:i+bytes_per_line]
            # Pad if necessary
            if len(chunk) < bytes_per_line:
                chunk = chunk + bytes(bytes_per_line - len(chunk))
            # Write as hex (big-endian)
            hex_str = chunk[::-1].hex()  # Reverse for Verilog convention
            f.write(f"{hex_str}\n")

def write_binary(filename: str, data: bytes):
    """Write raw binary file"""
    with open(filename, 'wb') as f:
        f.write(data)

#==============================================================================
# Test Case Generators
#==============================================================================

class TestGenerator:
    """Generate complete test cases with inputs, golden outputs, and instructions"""
    
    def __init__(self, output_dir: str, config: TestConfig):
        self.output_dir = output_dir
        self.config = config
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_gemm_test(self, M: int, N: int, K: int, 
                           name: str = "gemm", seed: int = 42) -> dict:
        """
        Generate GEMM test case
        
        Args:
            M: Rows of A and C
            N: Columns of B and C
            K: Columns of A, rows of B
            name: Test name
            seed: Random seed
        
        Returns:
            dict with paths to generated files
        """
        print(f"Generating GEMM test: {name} ({M}×{K} × {K}×{N})")
        
        # Generate inputs
        A = generate_random_matrix(M, K, 'int8', seed=seed)
        B = generate_random_matrix(K, N, 'int8', seed=seed+1)
        
        # Compute golden output
        C = golden_gemm(A, B)
        
        # Save to files
        files = {}
        
        # Input matrices
        files['A'] = os.path.join(self.output_dir, f"{name}_A.bin")
        files['B'] = os.path.join(self.output_dir, f"{name}_B.bin")
        files['C_golden'] = os.path.join(self.output_dir, f"{name}_C_golden.bin")
        
        write_binary(files['A'], matrix_to_bytes(A, 'int8'))
        write_binary(files['B'], matrix_to_bytes(B, 'int8'))
        write_binary(files['C_golden'], matrix_to_bytes(C, 'int32'))
        
        # Memory initialization files
        files['A_memh'] = os.path.join(self.output_dir, f"{name}_A.memh")
        files['B_memh'] = os.path.join(self.output_dir, f"{name}_B.memh")
        
        write_memh(files['A_memh'], matrix_to_bytes(A, 'int8'))
        write_memh(files['B_memh'], matrix_to_bytes(B, 'int8'))
        
        # Generate instruction sequence
        files['instr'] = os.path.join(self.output_dir, f"{name}_instr.hex")
        self._generate_gemm_instructions(files['instr'], M, N, K)
        
        # Save numpy arrays for debugging
        files['A_npy'] = os.path.join(self.output_dir, f"{name}_A.npy")
        files['B_npy'] = os.path.join(self.output_dir, f"{name}_B.npy")
        files['C_npy'] = os.path.join(self.output_dir, f"{name}_C_golden.npy")
        
        np.save(files['A_npy'], A)
        np.save(files['B_npy'], B)
        np.save(files['C_npy'], C)
        
        # Metadata
        files['config'] = {
            'name': name,
            'op': 'GEMM',
            'M': M, 'N': N, 'K': K,
            'seed': seed,
            'A_shape': A.shape,
            'B_shape': B.shape,
            'C_shape': C.shape,
        }
        
        return files
    
    def _generate_gemm_instructions(self, filename: str, M: int, N: int, K: int):
        """Generate instruction sequence for tiled GEMM"""
        tile_m = min(M, self.config.array_size)
        tile_n = min(N, self.config.array_size)
        tile_k = min(K, self.config.array_size)
        
        asm = Assembler()
        instrs = []
        
        # Simple single-tile GEMM for POC
        # For larger matrices, would need loops
        
        # NOP (pipeline warmup)
        instrs.append(Instruction(opcode=Opcode.NOP))
        
        # TENSOR.GEMM
        instr = Instruction(
            opcode=Opcode.TENSOR,
            subop=TensorSubop.GEMM,
            dst=0x6000,      # Output buffer
            src0=0x0000,     # Activation buffer
            src1=0x2000,     # Weight buffer
            dim_m=tile_m,
            dim_n=tile_n,
            dim_k=tile_k,
            flags=0
        )
        instrs.append(instr)
        
        # SYNC.WAIT_MXU
        instrs.append(Instruction(opcode=Opcode.SYNC, subop=SyncSubop.WAIT_MXU))
        
        # HALT
        instrs.append(Instruction(opcode=Opcode.HALT))
        
        # Write to file
        with open(filename, 'w') as f:
            for i, instr in enumerate(instrs):
                f.write(f"{instr.to_hex()}  // {i:04d}\n")
    
    def generate_attention_test(self, seq_len: int, d_model: int,
                                name: str = "attention", seed: int = 42) -> dict:
        """
        Generate attention test case
        
        Args:
            seq_len: Sequence length
            d_model: Model dimension (head dimension for single-head)
            name: Test name
            seed: Random seed
        """
        print(f"Generating attention test: {name} (seq={seq_len}, d={d_model})")
        
        # Generate Q, K, V
        Q = generate_random_matrix(seq_len, d_model, 'int8', seed=seed)
        K = generate_random_matrix(seq_len, d_model, 'int8', seed=seed+1)
        V = generate_random_matrix(seq_len, d_model, 'int8', seed=seed+2)
        
        # Compute golden output
        output, attn_weights = golden_attention(
            Q.astype(np.float32), 
            K.astype(np.float32), 
            V.astype(np.float32)
        )
        
        # Save files
        files = {}
        
        files['Q'] = os.path.join(self.output_dir, f"{name}_Q.bin")
        files['K'] = os.path.join(self.output_dir, f"{name}_K.bin")
        files['V'] = os.path.join(self.output_dir, f"{name}_V.bin")
        files['output_golden'] = os.path.join(self.output_dir, f"{name}_output_golden.bin")
        files['attn_weights'] = os.path.join(self.output_dir, f"{name}_attn_weights.bin")
        
        write_binary(files['Q'], matrix_to_bytes(Q, 'int8'))
        write_binary(files['K'], matrix_to_bytes(K, 'int8'))
        write_binary(files['V'], matrix_to_bytes(V, 'int8'))
        write_binary(files['output_golden'], matrix_to_bytes(output, 'float32'))
        write_binary(files['attn_weights'], matrix_to_bytes(attn_weights, 'float32'))
        
        # Memory init files
        write_memh(os.path.join(self.output_dir, f"{name}_Q.memh"), 
                   matrix_to_bytes(Q, 'int8'))
        write_memh(os.path.join(self.output_dir, f"{name}_K.memh"), 
                   matrix_to_bytes(K, 'int8'))
        write_memh(os.path.join(self.output_dir, f"{name}_V.memh"), 
                   matrix_to_bytes(V, 'int8'))
        
        # Save numpy arrays
        np.save(os.path.join(self.output_dir, f"{name}_Q.npy"), Q)
        np.save(os.path.join(self.output_dir, f"{name}_K.npy"), K)
        np.save(os.path.join(self.output_dir, f"{name}_V.npy"), V)
        np.save(os.path.join(self.output_dir, f"{name}_output_golden.npy"), output)
        np.save(os.path.join(self.output_dir, f"{name}_attn_weights.npy"), attn_weights)
        
        files['config'] = {
            'name': name,
            'op': 'ATTENTION',
            'seq_len': seq_len,
            'd_model': d_model,
            'seed': seed,
        }
        
        return files
    
    def generate_conv_test(self, N: int, C: int, H: int, W: int,
                           K: int, kH: int, kW: int,
                           stride: int = 1, padding: int = 0,
                           name: str = "conv", seed: int = 42) -> dict:
        """
        Generate Conv2D test case
        
        Args:
            N: Batch size
            C: Input channels
            H, W: Input height/width
            K: Output channels
            kH, kW: Kernel height/width
            stride: Convolution stride
            padding: Zero padding
        """
        print(f"Generating Conv2D test: {name} ({N},{C},{H},{W}) * ({K},{C},{kH},{kW})")
        
        # Generate input and weights
        input = generate_random_matrix(N * C * H * W, 1, 'int8', seed=seed).reshape(N, C, H, W)
        weights = generate_random_matrix(K * C * kH * kW, 1, 'int8', seed=seed+1).reshape(K, C, kH, kW)
        
        # Compute golden output
        output = golden_conv2d_im2col(input, weights, stride, padding)
        
        # Save files
        files = {}
        
        files['input'] = os.path.join(self.output_dir, f"{name}_input.bin")
        files['weights'] = os.path.join(self.output_dir, f"{name}_weights.bin")
        files['output_golden'] = os.path.join(self.output_dir, f"{name}_output_golden.bin")
        
        write_binary(files['input'], matrix_to_bytes(input, 'int8'))
        write_binary(files['weights'], matrix_to_bytes(weights, 'int8'))
        write_binary(files['output_golden'], matrix_to_bytes(output, 'int32'))
        
        # Save numpy arrays
        np.save(os.path.join(self.output_dir, f"{name}_input.npy"), input)
        np.save(os.path.join(self.output_dir, f"{name}_weights.npy"), weights)
        np.save(os.path.join(self.output_dir, f"{name}_output_golden.npy"), output)
        
        files['config'] = {
            'name': name,
            'op': 'CONV2D',
            'N': N, 'C': C, 'H': H, 'W': W,
            'K': K, 'kH': kH, 'kW': kW,
            'stride': stride, 'padding': padding,
            'seed': seed,
        }
        
        return files

#==============================================================================
# Main
#==============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate test vectors for tensor accelerator')
    parser.add_argument('--output', '-o', default='test_vectors', help='Output directory')
    parser.add_argument('--test', '-t', choices=['gemm', 'attention', 'conv', 'all'], 
                       default='all', help='Test type to generate')
    args = parser.parse_args()
    
    config = TestConfig(name="test")
    generator = TestGenerator(args.output, config)
    
    if args.test in ['gemm', 'all']:
        # Simple 16x16 GEMM
        generator.generate_gemm_test(16, 16, 16, name="gemm_16x16")
        
        # Larger GEMM requiring tiling
        generator.generate_gemm_test(64, 64, 64, name="gemm_64x64")
        
        # Non-square GEMM
        generator.generate_gemm_test(32, 64, 48, name="gemm_32x64x48")
    
    if args.test in ['attention', 'all']:
        # Simple attention (single head)
        generator.generate_attention_test(16, 64, name="attention_16x64")
        
        # Larger attention
        generator.generate_attention_test(64, 64, name="attention_64x64")
    
    if args.test in ['conv', 'all']:
        # Simple 3x3 convolution
        generator.generate_conv_test(
            N=1, C=3, H=8, W=8,
            K=16, kH=3, kW=3,
            stride=1, padding=1,
            name="conv_3x3"
        )
        
        # ResNet-style convolution
        generator.generate_conv_test(
            N=1, C=64, H=16, W=16,
            K=64, kH=3, kW=3,
            stride=1, padding=1,
            name="conv_resnet"
        )
    
    print(f"\nTest vectors generated in: {args.output}/")

if __name__ == '__main__':
    main()

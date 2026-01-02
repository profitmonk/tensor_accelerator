#!/usr/bin/env python3
"""
End-to-End Test Generator

Generates complete test packages ready for RTL simulation:
1. Compiler: IR → Assembly
2. Assembler: Assembly → Hex instructions
3. Golden reference: NumPy computation
4. Memory files: .memh format for Verilog $readmemh

Output structure:
    test_name/
    ├── program.asm      # Human-readable assembly
    ├── program.hex      # 128-bit hex instructions for $readmemh
    ├── weights.bin      # Raw weight data
    ├── weights.memh     # Weight data for $readmemh
    ├── input.bin        # Test input data
    ├── input.memh       # Input data for $readmemh
    ├── golden.bin       # Expected output
    ├── golden.memh      # Golden output for comparison
    └── test_config.json # Test metadata
"""

import sys
import os
import json
import struct
import numpy as np
from typing import Dict, Any, Tuple
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assembler'))

from ir.graph import Graph, Node, Tensor, OpType, DataType
from tiler.tiler import TilingEngine
from scheduler.scheduler import Scheduler
from codegen.codegen import CodeGenerator

from assembler import Assembler


class E2ETestGenerator:
    """Generate complete end-to-end test packages"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Compiler components
        self.tiler = TilingEngine()
        self.scheduler = Scheduler()
        self.codegen = CodeGenerator()
        self.assembler = Assembler()
        
    def generate_gemm_test(self, M: int, N: int, K: int, 
                           name: str = "gemm", seed: int = 42) -> Dict[str, Any]:
        """
        Generate a complete GEMM test
        
        Returns dict with file paths and test metadata
        """
        test_dir = self.output_dir / name
        test_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Generating GEMM test: {name}")
        print(f"  Dimensions: ({M}×{K}) × ({K}×{N}) → ({M}×{N})")
        print(f"{'='*60}")
        
        # Generate random data
        np.random.seed(seed)
        A = np.random.randint(-64, 64, (M, K), dtype=np.int8)  # Smaller range to avoid overflow
        B = np.random.randint(-64, 64, (K, N), dtype=np.int8)
        
        # Compute golden reference
        C_golden = np.matmul(A.astype(np.int32), B.astype(np.int32))
        
        print(f"  Input A: [{A.min()}, {A.max()}]")
        print(f"  Input B: [{B.min()}, {B.max()}]")
        print(f"  Golden C: [{C_golden.min()}, {C_golden.max()}]")
        
        # Create graph
        graph = Graph(name=name)
        graph.add_tensor(Tensor("A", (M, K), DataType.INT8))
        graph.add_tensor(Tensor("B", (K, N), DataType.INT8, data=B))
        graph.add_tensor(Tensor("C", (M, N), DataType.INT32))
        graph.add_node(Node("gemm", OpType.GEMM, ["A", "B"], ["C"]))
        graph.inputs.append("A")
        graph.outputs.append("C")
        
        # Compile
        print("\n  Compiling...")
        self.tiler.tile_graph(graph)
        schedule = self.scheduler.schedule(graph)
        asm_code = self.codegen.generate(graph, schedule)
        
        # Assemble
        print("  Assembling...")
        instructions = self._assemble(asm_code)
        
        # Generate output files
        files = {}
        
        # Assembly source
        files['asm'] = test_dir / 'program.asm'
        files['asm'].write_text(asm_code)
        print(f"  Written: {files['asm']}")
        
        # Hex instructions
        files['hex'] = test_dir / 'program.hex'
        self._write_hex(files['hex'], instructions)
        print(f"  Written: {files['hex']} ({len(instructions)} instructions)")
        
        # Weight data (B matrix)
        files['weights_bin'] = test_dir / 'weights.bin'
        files['weights_bin'].write_bytes(B.tobytes())
        
        files['weights_memh'] = test_dir / 'weights.memh'
        self._write_memh(files['weights_memh'], B.tobytes(), width=256)
        print(f"  Written: {files['weights_memh']}")
        
        # Input data (A matrix)
        files['input_bin'] = test_dir / 'input.bin'
        files['input_bin'].write_bytes(A.tobytes())
        
        files['input_memh'] = test_dir / 'input.memh'
        self._write_memh(files['input_memh'], A.tobytes(), width=256)
        print(f"  Written: {files['input_memh']}")
        
        # Golden output
        files['golden_bin'] = test_dir / 'golden.bin'
        files['golden_bin'].write_bytes(C_golden.tobytes())
        
        files['golden_memh'] = test_dir / 'golden.memh'
        self._write_memh(files['golden_memh'], C_golden.tobytes(), width=256)
        print(f"  Written: {files['golden_memh']}")
        
        # NumPy arrays for debugging
        np.save(test_dir / 'input_A.npy', A)
        np.save(test_dir / 'weight_B.npy', B)
        np.save(test_dir / 'golden_C.npy', C_golden)
        
        # Test metadata
        config = {
            'name': name,
            'type': 'gemm',
            'dimensions': {'M': M, 'N': N, 'K': K},
            'seed': seed,
            'input_range': {'min': int(A.min()), 'max': int(A.max())},
            'weight_range': {'min': int(B.min()), 'max': int(B.max())},
            'output_range': {'min': int(C_golden.min()), 'max': int(C_golden.max())},
            'instructions': len(instructions),
            'files': {k: str(v) for k, v in files.items()}
        }
        
        files['config'] = test_dir / 'test_config.json'
        files['config'].write_text(json.dumps(config, indent=2))
        print(f"  Written: {files['config']}")
        
        print(f"\n  ✓ Test package complete: {test_dir}")
        
        return config
    
    def generate_mlp_test(self, input_size: int, hidden_size: int, 
                          output_size: int, name: str = "mlp", 
                          seed: int = 42) -> Dict[str, Any]:
        """Generate a complete MLP test"""
        test_dir = self.output_dir / name
        test_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Generating MLP test: {name}")
        print(f"  Architecture: {input_size} → {hidden_size} → {output_size}")
        print(f"{'='*60}")
        
        np.random.seed(seed)
        
        # Generate weights (smaller range to avoid overflow)
        W1 = np.random.randint(-32, 32, (input_size, hidden_size), dtype=np.int8)
        W2 = np.random.randint(-32, 32, (hidden_size, output_size), dtype=np.int8)
        
        # Test input
        X = np.random.randint(-64, 64, (1, input_size), dtype=np.int8)
        
        # Golden forward pass
        H = np.matmul(X.astype(np.int32), W1.astype(np.int32))  # FC1
        H = np.maximum(H, 0)  # ReLU (simplified for INT32)
        Y_golden = np.matmul(H, W2.astype(np.int32))  # FC2
        
        print(f"  Input X: [{X.min()}, {X.max()}]")
        print(f"  Hidden H: [{H.min()}, {H.max()}]")
        print(f"  Output Y: [{Y_golden.min()}, {Y_golden.max()}]")
        
        # Create graph
        graph = Graph(name=name)
        graph.add_tensor(Tensor("input", (1, input_size), DataType.INT8))
        graph.add_tensor(Tensor("W1", (input_size, hidden_size), DataType.INT8, data=W1))
        graph.add_tensor(Tensor("H1", (1, hidden_size), DataType.INT32))
        graph.add_tensor(Tensor("H1_relu", (1, hidden_size), DataType.INT32))
        graph.add_tensor(Tensor("W2", (hidden_size, output_size), DataType.INT8, data=W2))
        graph.add_tensor(Tensor("output", (1, output_size), DataType.INT32))
        
        graph.add_node(Node("fc1", OpType.GEMM, ["input", "W1"], ["H1"]))
        graph.add_node(Node("relu", OpType.RELU, ["H1"], ["H1_relu"]))
        graph.add_node(Node("fc2", OpType.GEMM, ["H1_relu", "W2"], ["output"]))
        
        graph.inputs.append("input")
        graph.outputs.append("output")
        
        # Compile
        print("\n  Compiling...")
        self.tiler.tile_graph(graph)
        schedule = self.scheduler.schedule(graph)
        asm_code = self.codegen.generate(graph, schedule)
        
        # Assemble
        print("  Assembling...")
        instructions = self._assemble(asm_code)
        
        # Generate files
        files = {}
        
        files['asm'] = test_dir / 'program.asm'
        files['asm'].write_text(asm_code)
        
        files['hex'] = test_dir / 'program.hex'
        self._write_hex(files['hex'], instructions)
        
        # Weights
        all_weights = np.concatenate([W1.flatten(), W2.flatten()])
        files['weights_memh'] = test_dir / 'weights.memh'
        self._write_memh(files['weights_memh'], all_weights.tobytes(), width=256)
        
        # Input
        files['input_memh'] = test_dir / 'input.memh'
        self._write_memh(files['input_memh'], X.tobytes(), width=256)
        
        # Golden
        files['golden_memh'] = test_dir / 'golden.memh'
        self._write_memh(files['golden_memh'], Y_golden.tobytes(), width=256)
        
        # Save numpy
        np.save(test_dir / 'input.npy', X)
        np.save(test_dir / 'W1.npy', W1)
        np.save(test_dir / 'W2.npy', W2)
        np.save(test_dir / 'golden.npy', Y_golden)
        
        # Config
        config = {
            'name': name,
            'type': 'mlp',
            'architecture': [input_size, hidden_size, output_size],
            'seed': seed,
            'instructions': len(instructions),
        }
        
        files['config'] = test_dir / 'test_config.json'
        files['config'].write_text(json.dumps(config, indent=2))
        
        print(f"\n  ✓ Test package complete: {test_dir}")
        print(f"    Instructions: {len(instructions)}")
        
        return config
    
    def _assemble(self, asm_code: str) -> list:
        """Assemble code and return instruction list"""
        self.assembler = Assembler()  # Fresh assembler
        instructions = []
        
        for i, line in enumerate(asm_code.split('\n'), 1):
            self.assembler.line_num = i
            instr = self.assembler.assemble_line(line)
            if instr:
                instructions.append(instr)
        
        return instructions
    
    def _write_hex(self, path: Path, instructions: list):
        """Write instructions in $readmemh format"""
        with open(path, 'w') as f:
            for instr in instructions:
                f.write(f"{instr.to_hex()}\n")
    
    def _write_memh(self, path: Path, data: bytes, width: int = 256):
        """Write data in $readmemh format"""
        bytes_per_line = width // 8
        
        with open(path, 'w') as f:
            for i in range(0, len(data), bytes_per_line):
                chunk = data[i:i+bytes_per_line]
                if len(chunk) < bytes_per_line:
                    chunk = chunk + bytes(bytes_per_line - len(chunk))
                # Write as hex (big-endian for Verilog)
                hex_str = chunk[::-1].hex()
                f.write(f"{hex_str}\n")


def main():
    """Generate test suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate E2E test packages')
    parser.add_argument('-o', '--output', default='./e2e_tests', help='Output directory')
    parser.add_argument('--gemm', action='store_true', help='Generate GEMM tests')
    parser.add_argument('--mlp', action='store_true', help='Generate MLP tests')
    parser.add_argument('--all', action='store_true', help='Generate all tests')
    
    args = parser.parse_args()
    
    gen = E2ETestGenerator(args.output)
    
    if args.all or args.gemm:
        # Generate various GEMM tests
        gen.generate_gemm_test(8, 8, 8, name="gemm_8x8")
        gen.generate_gemm_test(16, 16, 16, name="gemm_16x16")
        gen.generate_gemm_test(32, 32, 32, name="gemm_32x32")
        gen.generate_gemm_test(64, 64, 64, name="gemm_64x64")
    
    if args.all or args.mlp:
        # Generate MLP tests
        gen.generate_mlp_test(64, 32, 10, name="mlp_small")
        gen.generate_mlp_test(256, 128, 10, name="mlp_medium")
    
    if not (args.all or args.gemm or args.mlp):
        # Default: generate one of each
        gen.generate_gemm_test(16, 16, 16, name="gemm_16x16")
        gen.generate_mlp_test(64, 32, 10, name="mlp_small")
    
    print("\n" + "="*60)
    print("Test generation complete!")
    print(f"Output directory: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()

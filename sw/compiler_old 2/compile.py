#!/usr/bin/env python3
"""
Tensor Accelerator Compiler

Main entry point for compiling ONNX/PyTorch models to accelerator code.

Usage:
    python compile.py model.onnx -o output.asm
    python compile.py model.onnx --quantize -o output.asm
"""

import argparse
import sys
import os
from typing import Optional, Tuple, Dict
from pathlib import Path

# Add compiler package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ir.graph import Graph, create_simple_test_graph
from frontend.onnx_parser import ONNXParser, load_onnx, ONNX_AVAILABLE
from quantizer.quantizer import Quantizer, quantize_graph
from tiler.tiler import TilingEngine, HardwareConfig, tile_graph
from scheduler.scheduler import Scheduler, schedule_graph
from codegen.codegen import CodeGenerator, CodeGenConfig, generate_code


class Compiler:
    """
    Main compiler class
    
    Pipeline:
    1. Frontend: Load ONNX model
    2. Quantize: FP32 -> INT8
    3. Optimize: Graph optimizations (future)
    4. Tile: Break large ops into hardware tiles
    5. Schedule: Determine execution order
    6. Codegen: Emit assembly
    """
    
    def __init__(self, 
                 hw_config: Optional[HardwareConfig] = None,
                 codegen_config: Optional[CodeGenConfig] = None,
                 verbose: bool = False):
        self.hw_config = hw_config or HardwareConfig()
        self.codegen_config = codegen_config or CodeGenConfig()
        self.verbose = verbose
        
        # Components
        self.parser = ONNXParser(verbose=verbose)
        self.quantizer = Quantizer(method='symmetric')
        self.tiler = TilingEngine(hw_config)
        self.scheduler = Scheduler(hw_config)
        self.codegen = CodeGenerator(codegen_config, hw_config)
        
        # Stats
        self.stats = {}
    
    def compile(self, 
                input_path: str,
                quantize: bool = True,
                calibration_data: Optional[Dict] = None) -> Tuple[str, bytes, Graph]:
        """
        Compile a model to accelerator code
        
        Args:
            input_path: Path to ONNX model
            quantize: Whether to quantize to INT8
            calibration_data: Optional calibration data for quantization
        
        Returns:
            Tuple of (assembly_code, weight_binary, final_graph)
        """
        if self.verbose:
            print(f"Compiling: {input_path}")
            print("=" * 60)
        
        # 1. Parse ONNX
        if self.verbose:
            print("\n[1/5] Parsing ONNX model...")
        graph = self.parser.load(input_path)
        self.stats['original_nodes'] = len(graph.nodes)
        self.stats['original_tensors'] = len(graph.tensors)
        
        # 2. Quantize
        if quantize:
            if self.verbose:
                print("\n[2/5] Quantizing to INT8...")
            if calibration_data:
                graph = self.quantizer.quantize(graph, calibration_data)
            else:
                graph = self.quantizer.quantize_weights_only(graph)
            self.stats['quantized'] = True
        else:
            self.stats['quantized'] = False
        
        # 3. Tile
        if self.verbose:
            print("\n[3/5] Computing tiling...")
        graph = self.tiler.tile_graph(graph)
        total_tiles = sum(len(n.tiles) for n in graph.nodes)
        self.stats['total_tiles'] = total_tiles
        
        # 4. Schedule
        if self.verbose:
            print("\n[4/5] Scheduling operations...")
        schedule = self.scheduler.schedule(graph)
        self.stats['schedule_length'] = len(schedule)
        
        # 5. Generate code
        if self.verbose:
            print("\n[5/5] Generating assembly...")
        asm_code = self.codegen.generate(graph, schedule)
        weight_data = self.codegen.generate_weights(graph)
        
        self.stats['asm_lines'] = asm_code.count('\n')
        self.stats['weight_bytes'] = len(weight_data)
        
        # Estimates
        mem_usage = self.scheduler.estimate_memory_usage(graph)
        timing = self.scheduler.estimate_execution_time(schedule, graph)
        self.stats['memory'] = mem_usage
        self.stats['timing'] = timing
        
        if self.verbose:
            self._print_summary()
        
        return asm_code, weight_data, graph
    
    def compile_graph(self, graph: Graph) -> Tuple[str, bytes]:
        """Compile an already-parsed graph"""
        graph = self.tiler.tile_graph(graph)
        schedule = self.scheduler.schedule(graph)
        asm_code = self.codegen.generate(graph, schedule)
        weight_data = self.codegen.generate_weights(graph)
        return asm_code, weight_data
    
    def _print_summary(self):
        """Print compilation summary"""
        print("\n" + "=" * 60)
        print("Compilation Summary")
        print("=" * 60)
        print(f"  Nodes: {self.stats.get('original_nodes', 0)}")
        print(f"  Tensors: {self.stats.get('original_tensors', 0)}")
        print(f"  Quantized: {self.stats.get('quantized', False)}")
        print(f"  Total tiles: {self.stats.get('total_tiles', 0)}")
        print(f"  Schedule entries: {self.stats.get('schedule_length', 0)}")
        print(f"  Assembly lines: {self.stats.get('asm_lines', 0)}")
        print(f"  Weight data: {self.stats.get('weight_bytes', 0):,} bytes")
        
        if 'memory' in self.stats:
            mem = self.stats['memory']
            print(f"\nMemory Usage:")
            print(f"  Weights: {mem.get('weights', 0):,} bytes")
            print(f"  Activations: {mem.get('activations', 0):,} bytes")
            print(f"  Total: {mem.get('total', 0):,} bytes")
        
        if 'timing' in self.stats:
            timing = self.stats['timing']
            print(f"\nPerformance Estimate:")
            print(f"  MACs: {timing.get('total_macs', 0):,.0f}")
            print(f"  DMA bytes: {timing.get('total_dma_bytes', 0):,.0f}")
            print(f"  Estimated time: {timing.get('time_us', 0):,.2f} µs")
            if timing.get('ops_per_sec', 0) > 0:
                gops = timing['ops_per_sec'] / 1e9
                print(f"  Throughput: {gops:,.2f} GOPS")


def compile_model(input_path: str,
                  output_path: Optional[str] = None,
                  quantize: bool = True,
                  verbose: bool = False) -> Tuple[str, bytes]:
    """
    Convenience function to compile a model
    
    Args:
        input_path: Path to ONNX model
        output_path: Optional output path for assembly
        quantize: Whether to quantize
        verbose: Print progress
    
    Returns:
        Tuple of (assembly_code, weight_binary)
    """
    compiler = Compiler(verbose=verbose)
    asm_code, weight_data, graph = compiler.compile(input_path, quantize=quantize)
    
    if output_path:
        # Write assembly
        with open(output_path, 'w') as f:
            f.write(asm_code)
        
        # Write weights
        weight_path = output_path.replace('.asm', '.weights')
        with open(weight_path, 'wb') as f:
            f.write(weight_data)
        
        if verbose:
            print(f"\nOutput written to:")
            print(f"  Assembly: {output_path}")
            print(f"  Weights: {weight_path}")
    
    return asm_code, weight_data


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Tensor Accelerator Compiler',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python compile.py model.onnx -o program.asm
  python compile.py model.onnx --no-quantize -o program.asm
  python compile.py --test  # Run test compilation
'''
    )
    
    parser.add_argument('input', nargs='?', help='Input ONNX model file')
    parser.add_argument('-o', '--output', help='Output assembly file')
    parser.add_argument('--no-quantize', action='store_true', help='Skip INT8 quantization')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--test', action='store_true', help='Run test compilation')
    
    args = parser.parse_args()
    
    if args.test:
        # Run test with built-in graph
        print("Running test compilation...")
        print("=" * 60)
        
        graph = create_simple_test_graph()
        print(f"Test graph: {graph.name}")
        print(graph.summary())
        
        compiler = Compiler(verbose=True)
        asm_code, weight_data = compiler.compile_graph(graph)
        
        print("\nGenerated Assembly:")
        print("-" * 40)
        print(asm_code)
        
        print(f"\nWeight data: {len(weight_data)} bytes")
        return
    
    if not args.input:
        parser.print_help()
        print("\nError: Input file required (or use --test)")
        sys.exit(1)
    
    if not ONNX_AVAILABLE:
        print("Error: ONNX not installed. Run: pip install onnx")
        sys.exit(1)
    
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    
    output_path = args.output or args.input.replace('.onnx', '.asm')
    
    try:
        compile_model(
            args.input,
            output_path,
            quantize=not args.no_quantize,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

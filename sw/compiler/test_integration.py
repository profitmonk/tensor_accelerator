#!/usr/bin/env python3
"""
Compiler Integration Test

Tests the complete flow:
  Compiler (IR → ASM) → Assembler (ASM → HEX) → [Ready for RTL simulation]

This verifies that:
1. Compiler generates valid assembly syntax
2. Assembly can be parsed by the assembler
3. Generated hex can be loaded by Verilog testbench
"""

import sys
import os
import tempfile
import numpy as np

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assembler'))

from ir.graph import Graph, Node, Tensor, OpType, DataType
from tiler.tiler import TilingEngine
from scheduler.scheduler import Scheduler
from codegen.codegen import CodeGenerator

# Import assembler
from assembler import Assembler


def create_mlp_graph(input_size=784, hidden_size=256, output_size=10):
    """Create a simple MLP graph"""
    g = Graph(name="test_mlp")
    
    # Input
    g.add_tensor(Tensor("input", (1, input_size), DataType.INT8))
    g.inputs.append("input")
    
    # FC1
    fc1_w = np.random.randint(-128, 127, (input_size, hidden_size), dtype=np.int8)
    fc1_b = np.zeros(hidden_size, dtype=np.int32)
    
    g.add_tensor(Tensor("fc1_weight", (input_size, hidden_size), DataType.INT8, data=fc1_w))
    g.add_tensor(Tensor("fc1_bias", (hidden_size,), DataType.INT32, data=fc1_b))
    g.add_tensor(Tensor("fc1_out", (1, hidden_size), DataType.INT8))
    
    g.add_node(Node("fc1", OpType.GEMM, 
                    ["input", "fc1_weight", "fc1_bias"], ["fc1_out"],
                    attrs={'transB': True}))
    
    # ReLU
    g.add_tensor(Tensor("relu_out", (1, hidden_size), DataType.INT8))
    g.add_node(Node("relu1", OpType.RELU, ["fc1_out"], ["relu_out"]))
    
    # FC2
    fc2_w = np.random.randint(-128, 127, (hidden_size, output_size), dtype=np.int8)
    fc2_b = np.zeros(output_size, dtype=np.int32)
    
    g.add_tensor(Tensor("fc2_weight", (hidden_size, output_size), DataType.INT8, data=fc2_w))
    g.add_tensor(Tensor("fc2_bias", (output_size,), DataType.INT32, data=fc2_b))
    g.add_tensor(Tensor("output", (1, output_size), DataType.INT8))
    
    g.add_node(Node("fc2", OpType.GEMM,
                    ["relu_out", "fc2_weight", "fc2_bias"], ["output"],
                    attrs={'transB': True}))
    
    g.outputs.append("output")
    return g


def create_simple_gemm_graph(M=8, N=8, K=8):
    """Create minimal GEMM graph for testing"""
    g = Graph(name="simple_gemm")
    
    # Input
    g.add_tensor(Tensor("A", (M, K), DataType.INT8))
    g.inputs.append("A")
    
    # Weight
    B = np.random.randint(-128, 127, (K, N), dtype=np.int8)
    g.add_tensor(Tensor("B", (K, N), DataType.INT8, data=B))
    
    # Output
    g.add_tensor(Tensor("C", (M, N), DataType.INT8))
    g.outputs.append("C")
    
    # GEMM node
    g.add_node(Node("gemm", OpType.GEMM, ["A", "B"], ["C"]))
    
    return g


def test_compiler_to_assembler():
    """Test that compiler output can be assembled"""
    print("=" * 60)
    print("Test: Compiler → Assembler Integration")
    print("=" * 60)
    
    # Create test graph
    graph = create_simple_gemm_graph(M=16, N=16, K=16)
    print(f"\nGraph: {graph.name}")
    print(f"  Nodes: {len(graph.nodes)}")
    
    # Run compiler pipeline
    tiler = TilingEngine()
    tiler.tile_graph(graph)
    
    scheduler = Scheduler()
    schedule = scheduler.schedule(graph)
    
    codegen = CodeGenerator()
    asm_code = codegen.generate(graph, schedule)
    
    print(f"\nGenerated assembly ({asm_code.count(chr(10))} lines)")
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.asm', delete=False) as f:
        f.write(asm_code)
        asm_file = f.name
    
    print(f"  Written to: {asm_file}")
    
    # Try to assemble it
    print("\nRunning assembler...")
    assembler = Assembler()
    
    try:
        # Read and assemble each line
        with open(asm_file, 'r') as f:
            lines = f.readlines()
        
        errors = []
        instructions = []
        
        for i, line in enumerate(lines, 1):
            try:
                assembler.line_num = i
                instr = assembler.assemble_line(line)
                if instr:
                    instructions.append(instr)
            except Exception as e:
                errors.append(f"Line {i}: {e}")
                errors.append(f"  Content: {line.strip()}")
        
        if errors:
            print(f"\n❌ Assembly errors ({len(errors)}):")
            for e in errors[:10]:  # Show first 10
                print(f"  {e}")
            return False
        
        print(f"✓ Assembled {len(instructions)} instructions successfully")
        
        # Generate hex output
        hex_lines = [instr.to_hex() for instr in instructions]
        print(f"✓ Generated {len(hex_lines)} hex lines")
        
        # Verify hex format (should be 32 hex chars = 128 bits)
        for i, h in enumerate(hex_lines):
            if len(h) != 32:
                print(f"❌ Hex line {i} has wrong length: {len(h)} (expected 32)")
                return False
        
        print("✓ All hex lines are correct length (128 bits)")
        
        # Show sample
        print(f"\nSample instructions:")
        for i, (line, hex_val) in enumerate(zip(lines, hex_lines)):
            if i < 5:
                clean_line = line.strip()
                if clean_line and not clean_line.startswith(';'):
                    print(f"  {clean_line[:50]:50s} → {hex_val[:16]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Assembler failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.unlink(asm_file)


def test_mlp_compilation():
    """Test MLP compilation end-to-end"""
    print("\n" + "=" * 60)
    print("Test: MLP Compilation")
    print("=" * 60)
    
    # Create MLP graph
    graph = create_mlp_graph(input_size=256, hidden_size=128, output_size=10)
    print(f"\nGraph: {graph.name}")
    print(f"  Layers: FC(256→128) → ReLU → FC(128→10)")
    
    # Compile
    tiler = TilingEngine()
    tiler.tile_graph(graph)
    
    scheduler = Scheduler()
    schedule = scheduler.schedule(graph)
    
    codegen = CodeGenerator()
    asm_code = codegen.generate(graph, schedule)
    weight_data = codegen.generate_weights(graph)
    
    print(f"\nCompilation results:")
    print(f"  Assembly lines: {asm_code.count(chr(10))}")
    print(f"  Weight bytes: {len(weight_data):,}")
    print(f"  Schedule entries: {len(schedule)}")
    
    # Count instruction types
    tensor_gemm = asm_code.count("TENSOR.GEMM")
    vector_relu = asm_code.count("VECTOR.RELU")
    sync_wait = asm_code.count("SYNC.WAIT")
    dma_load = asm_code.count("DMA.LOAD")
    
    print(f"\nInstruction breakdown:")
    print(f"  TENSOR.GEMM: {tensor_gemm}")
    print(f"  VECTOR.RELU: {vector_relu}")
    print(f"  SYNC.WAIT: {sync_wait}")
    print(f"  DMA.LOAD: {dma_load}")
    
    # Try to assemble
    assembler = Assembler()
    lines = asm_code.split('\n')
    instructions = []
    errors = []
    
    for i, line in enumerate(lines, 1):
        try:
            assembler.line_num = i
            instr = assembler.assemble_line(line)
            if instr:
                instructions.append(instr)
        except Exception as e:
            errors.append(f"Line {i}: {e} - '{line.strip()}'")
    
    if errors:
        print(f"\n❌ Assembly errors: {len(errors)}")
        for e in errors[:5]:
            print(f"  {e}")
        return False
    
    print(f"\n✓ Successfully assembled {len(instructions)} instructions")
    return True


def test_golden_reference():
    """Test that we can generate golden reference for verification"""
    print("\n" + "=" * 60)
    print("Test: Golden Reference Generation")
    print("=" * 60)
    
    # Create simple GEMM
    M, N, K = 16, 16, 16
    
    # Random inputs
    np.random.seed(42)
    A = np.random.randint(-128, 127, (M, K), dtype=np.int8)
    B = np.random.randint(-128, 127, (K, N), dtype=np.int8)
    
    # Golden output (INT32 accumulator)
    C_golden = np.matmul(A.astype(np.int32), B.astype(np.int32))
    
    print(f"\nGEMM: ({M}×{K}) × ({K}×{N}) → ({M}×{N})")
    print(f"  Input A range: [{A.min()}, {A.max()}]")
    print(f"  Input B range: [{B.min()}, {B.max()}]")
    print(f"  Output C range: [{C_golden.min()}, {C_golden.max()}]")
    
    # Create graph with these values
    g = Graph(name="golden_gemm")
    g.add_tensor(Tensor("A", (M, K), DataType.INT8, data=A))
    g.add_tensor(Tensor("B", (K, N), DataType.INT8, data=B))
    g.add_tensor(Tensor("C", (M, N), DataType.INT32))
    g.add_node(Node("gemm", OpType.GEMM, ["A", "B"], ["C"]))
    g.inputs.append("A")
    g.outputs.append("C")
    
    # Compile
    tiler = TilingEngine()
    tiler.tile_graph(g)
    
    scheduler = Scheduler()
    schedule = scheduler.schedule(g)
    
    codegen = CodeGenerator()
    asm_code = codegen.generate(g, schedule)
    
    print(f"\n✓ Generated assembly for golden test")
    print(f"  Instructions: {asm_code.count('TENSOR.GEMM')} GEMM ops")
    
    # This golden reference would be used by cocotb to verify RTL output
    print(f"\nGolden reference ready for RTL verification:")
    print(f"  C[0,0] = {C_golden[0,0]}")
    print(f"  C[7,7] = {C_golden[7,7]}")
    print(f"  C[15,15] = {C_golden[15,15]}")
    
    return True


def main():
    """Run all integration tests"""
    print("Compiler Integration Tests")
    print("=" * 60)
    
    results = []
    
    # Test 1: Basic compiler → assembler
    results.append(("Compiler→Assembler", test_compiler_to_assembler()))
    
    # Test 2: MLP compilation
    results.append(("MLP Compilation", test_mlp_compilation()))
    
    # Test 3: Golden reference
    results.append(("Golden Reference", test_golden_reference()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Integration Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    failed = len(results) - passed
    
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

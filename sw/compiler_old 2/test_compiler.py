#!/usr/bin/env python3
"""
Compiler Test Suite

Tests the compiler with various model configurations.
"""

import sys
import os
import numpy as np

# Add compiler to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ir.graph import Graph, Node, Tensor, OpType, DataType, create_simple_test_graph
from quantizer.quantizer import Quantizer
from tiler.tiler import TilingEngine, HardwareConfig
from scheduler.scheduler import Scheduler
from codegen.codegen import CodeGenerator
from compile import Compiler


def test_ir():
    """Test IR creation and validation"""
    print("=" * 60)
    print("Test 1: IR Creation and Validation")
    print("=" * 60)
    
    graph = create_simple_test_graph()
    
    # Validate
    errors = graph.validate()
    assert len(errors) == 0, f"Validation errors: {errors}"
    
    # Check structure
    assert len(graph.nodes) == 3  # fc1, relu1, fc2
    assert len(graph.inputs) == 1
    assert len(graph.outputs) == 1
    
    # Test topological sort
    sorted_nodes = graph.topological_sort()
    assert sorted_nodes[0].name == "fc1"
    assert sorted_nodes[1].name == "relu1"
    assert sorted_nodes[2].name == "fc2"
    
    print("✓ IR creation passed")
    print("✓ Graph validation passed")
    print("✓ Topological sort passed")
    return True


def test_quantizer():
    """Test quantization"""
    print("\n" + "=" * 60)
    print("Test 2: Quantization")
    print("=" * 60)
    
    graph = create_simple_test_graph()
    
    quantizer = Quantizer(method='symmetric')
    q_graph = quantizer.quantize_weights_only(graph)
    
    # Check weights are quantized
    fc1_weight = q_graph.get_tensor("fc1_weight")
    assert fc1_weight is not None
    assert fc1_weight.dtype == DataType.INT8
    assert fc1_weight.data.dtype == np.int8
    assert fc1_weight.quant is not None
    
    print("✓ Weight quantization passed")
    print(f"  fc1_weight scale: {fc1_weight.quant.scale:.6f}")
    
    return True


def test_tiler():
    """Test tiling engine"""
    print("\n" + "=" * 60)
    print("Test 3: Tiling Engine")
    print("=" * 60)
    
    engine = TilingEngine()
    
    # Test various GEMM sizes
    test_cases = [
        (8, 8, 8),        # Fits in one systolic pass
        (16, 16, 16),     # Small
        (256, 256, 256),  # Medium
        (784, 256, 784),  # LeNet FC1
    ]
    
    for M, N, K in test_cases:
        config = engine.compute_gemm_tiling(M, N, K)
        
        # Verify tile fits in memory
        assert config.total_bytes < engine.hw.sram_size
        
        # Verify tiles cover full matrix
        assert config.tile_m * config.num_m_tiles >= M
        assert config.tile_n * config.num_n_tiles >= N
        assert config.tile_k * config.num_k_tiles >= K
        
        total_tiles = config.num_m_tiles * config.num_n_tiles * config.num_k_tiles
        print(f"  GEMM({M}x{N}x{K}): {total_tiles} tiles, "
              f"tile size {config.tile_m}x{config.tile_n}x{config.tile_k}")
    
    print("✓ Tiling passed")
    return True


def test_scheduler():
    """Test scheduler"""
    print("\n" + "=" * 60)
    print("Test 4: Scheduler")
    print("=" * 60)
    
    graph = create_simple_test_graph()
    
    tiler = TilingEngine()
    tiler.tile_graph(graph)
    
    scheduler = Scheduler()
    schedule = scheduler.schedule(graph)
    
    # Check schedule is non-empty
    assert len(schedule) > 0
    
    # Check order is monotonic
    orders = [e.order for e in schedule]
    assert orders == sorted(orders)
    
    print(f"✓ Scheduler produced {len(schedule)} entries")
    
    # Check estimates
    mem = scheduler.estimate_memory_usage(graph)
    print(f"  Memory estimate: {mem['total']:,} bytes")
    
    timing = scheduler.estimate_execution_time(schedule, graph)
    print(f"  Time estimate: {timing['time_us']:.2f} µs")
    
    return True


def test_codegen():
    """Test code generation"""
    print("\n" + "=" * 60)
    print("Test 5: Code Generation")
    print("=" * 60)
    
    graph = create_simple_test_graph()
    
    tiler = TilingEngine()
    tiler.tile_graph(graph)
    
    scheduler = Scheduler()
    schedule = scheduler.schedule(graph)
    
    codegen = CodeGenerator()
    asm_code = codegen.generate(graph, schedule)
    weight_data = codegen.generate_weights(graph)
    
    # Check assembly is non-empty
    assert len(asm_code) > 0
    
    # Check it has key instructions
    assert "TENSOR.GEMM" in asm_code
    assert "VECTOR.RELU" in asm_code
    assert "HALT" in asm_code
    
    # Check weight data is generated
    assert len(weight_data) > 0
    
    print(f"✓ Generated {asm_code.count(chr(10))} lines of assembly")
    print(f"✓ Generated {len(weight_data):,} bytes of weights")
    
    return True


def test_full_pipeline():
    """Test full compilation pipeline"""
    print("\n" + "=" * 60)
    print("Test 6: Full Compilation Pipeline")
    print("=" * 60)
    
    compiler = Compiler(verbose=False)
    graph = create_simple_test_graph()
    
    asm_code, weight_data = compiler.compile_graph(graph)
    
    assert len(asm_code) > 0
    assert len(weight_data) > 0
    
    print("✓ Full pipeline passed")
    print(f"  Assembly: {asm_code.count(chr(10))} lines")
    print(f"  Weights: {len(weight_data):,} bytes")
    
    return True


def test_custom_graph():
    """Test with a custom graph (LeNet-like)"""
    print("\n" + "=" * 60)
    print("Test 7: Custom Graph (LeNet-5 FC Layers)")
    print("=" * 60)
    
    # Create LeNet-5 fully connected layers
    # After conv/pool: 256 features
    # FC1: 256 -> 120
    # FC2: 120 -> 84
    # FC3: 84 -> 10
    
    g = Graph(name="lenet_fc")
    
    # Input
    g.add_tensor(Tensor("input", (1, 256), DataType.INT8))
    g.inputs.append("input")
    
    # FC1
    g.add_tensor(Tensor("fc1_w", (256, 120), DataType.INT8,
                        data=np.random.randint(-128, 127, (256, 120), dtype=np.int8)))
    g.add_tensor(Tensor("fc1_b", (120,), DataType.INT32,
                        data=np.zeros(120, dtype=np.int32)))
    g.add_tensor(Tensor("fc1_out", (1, 120), DataType.INT8))
    g.add_node(Node("fc1", OpType.GEMM, ["input", "fc1_w", "fc1_b"], ["fc1_out"]))
    
    g.add_tensor(Tensor("relu1_out", (1, 120), DataType.INT8))
    g.add_node(Node("relu1", OpType.RELU, ["fc1_out"], ["relu1_out"]))
    
    # FC2
    g.add_tensor(Tensor("fc2_w", (120, 84), DataType.INT8,
                        data=np.random.randint(-128, 127, (120, 84), dtype=np.int8)))
    g.add_tensor(Tensor("fc2_b", (84,), DataType.INT32,
                        data=np.zeros(84, dtype=np.int32)))
    g.add_tensor(Tensor("fc2_out", (1, 84), DataType.INT8))
    g.add_node(Node("fc2", OpType.GEMM, ["relu1_out", "fc2_w", "fc2_b"], ["fc2_out"]))
    
    g.add_tensor(Tensor("relu2_out", (1, 84), DataType.INT8))
    g.add_node(Node("relu2", OpType.RELU, ["fc2_out"], ["relu2_out"]))
    
    # FC3 (output)
    g.add_tensor(Tensor("fc3_w", (84, 10), DataType.INT8,
                        data=np.random.randint(-128, 127, (84, 10), dtype=np.int8)))
    g.add_tensor(Tensor("fc3_b", (10,), DataType.INT32,
                        data=np.zeros(10, dtype=np.int32)))
    g.add_tensor(Tensor("output", (1, 10), DataType.INT8))
    g.add_node(Node("fc3", OpType.GEMM, ["relu2_out", "fc3_w", "fc3_b"], ["output"]))
    
    g.outputs.append("output")
    
    # Compile
    compiler = Compiler(verbose=False)
    asm_code, weight_data = compiler.compile_graph(g)
    
    print(f"✓ LeNet FC layers compiled")
    print(f"  Nodes: {len(g.nodes)}")
    print(f"  Assembly: {asm_code.count(chr(10))} lines")
    print(f"  Weights: {len(weight_data):,} bytes")
    
    # Count GEMM instructions
    gemm_count = asm_code.count("TENSOR.GEMM")
    relu_count = asm_code.count("VECTOR.RELU")
    print(f"  GEMM instructions: {gemm_count}")
    print(f"  RELU instructions: {relu_count}")
    
    return True


def run_all_tests():
    """Run all compiler tests"""
    print("Tensor Accelerator Compiler Tests")
    print("=" * 60)
    
    tests = [
        test_ir,
        test_quantizer,
        test_tiler,
        test_scheduler,
        test_codegen,
        test_full_pipeline,
        test_custom_graph,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

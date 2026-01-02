#!/usr/bin/env python3
"""
Test Extended Operations

Validates compiler support for:
- LeNet: Conv2D, MaxPool, ReLU, GEMM
- ResNet: Conv2D, BatchNorm, ReLU, Add, GlobalAvgPool, GEMM
- LLMs: MatMul, LayerNorm, GELU, Softmax, Add
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assembler'))

from ir.graph import Graph, Node, Tensor, OpType, DataType
from tiler.tiler import TilingEngine
from scheduler.scheduler import Scheduler
from codegen.codegen import CodeGenerator
from assembler import Assembler


def create_lenet_graph():
    """Create LeNet-5 style graph"""
    g = Graph(name="lenet")
    
    # Input: 1x1x28x28
    g.add_tensor(Tensor("input", (1, 1, 28, 28), DataType.INT8))
    g.inputs.append("input")
    
    # Conv1: 1->6 channels, 5x5 kernel
    conv1_w = np.random.randint(-64, 64, (6, 1, 5, 5), dtype=np.int8)
    conv1_b = np.zeros(6, dtype=np.int32)
    g.add_tensor(Tensor("conv1_w", (6, 1, 5, 5), DataType.INT8, data=conv1_w))
    g.add_tensor(Tensor("conv1_b", (6,), DataType.INT32, data=conv1_b))
    g.add_tensor(Tensor("conv1_out", (1, 6, 24, 24), DataType.INT8))
    g.add_node(Node("conv1", OpType.CONV2D, ["input", "conv1_w", "conv1_b"], ["conv1_out"],
                    attrs={'kernel_shape': [5, 5], 'strides': [1, 1], 'pads': [0, 0, 0, 0]}))
    
    # ReLU1
    g.add_tensor(Tensor("relu1_out", (1, 6, 24, 24), DataType.INT8))
    g.add_node(Node("relu1", OpType.RELU, ["conv1_out"], ["relu1_out"]))
    
    # MaxPool1: 2x2
    g.add_tensor(Tensor("pool1_out", (1, 6, 12, 12), DataType.INT8))
    g.add_node(Node("pool1", OpType.MAXPOOL, ["relu1_out"], ["pool1_out"],
                    attrs={'kernel_shape': [2, 2], 'strides': [2, 2]}))
    
    # Conv2: 6->16 channels, 5x5 kernel
    conv2_w = np.random.randint(-64, 64, (16, 6, 5, 5), dtype=np.int8)
    conv2_b = np.zeros(16, dtype=np.int32)
    g.add_tensor(Tensor("conv2_w", (16, 6, 5, 5), DataType.INT8, data=conv2_w))
    g.add_tensor(Tensor("conv2_b", (16,), DataType.INT32, data=conv2_b))
    g.add_tensor(Tensor("conv2_out", (1, 16, 8, 8), DataType.INT8))
    g.add_node(Node("conv2", OpType.CONV2D, ["pool1_out", "conv2_w", "conv2_b"], ["conv2_out"],
                    attrs={'kernel_shape': [5, 5], 'strides': [1, 1], 'pads': [0, 0, 0, 0]}))
    
    # ReLU2
    g.add_tensor(Tensor("relu2_out", (1, 16, 8, 8), DataType.INT8))
    g.add_node(Node("relu2", OpType.RELU, ["conv2_out"], ["relu2_out"]))
    
    # MaxPool2: 2x2
    g.add_tensor(Tensor("pool2_out", (1, 16, 4, 4), DataType.INT8))
    g.add_node(Node("pool2", OpType.MAXPOOL, ["relu2_out"], ["pool2_out"],
                    attrs={'kernel_shape': [2, 2], 'strides': [2, 2]}))
    
    # Flatten (implicit) -> FC1: 256->120
    fc1_w = np.random.randint(-64, 64, (256, 120), dtype=np.int8)
    fc1_b = np.zeros(120, dtype=np.int32)
    g.add_tensor(Tensor("fc1_w", (256, 120), DataType.INT8, data=fc1_w))
    g.add_tensor(Tensor("fc1_b", (120,), DataType.INT32, data=fc1_b))
    g.add_tensor(Tensor("fc1_out", (1, 120), DataType.INT8))
    g.add_node(Node("fc1", OpType.GEMM, ["pool2_out", "fc1_w", "fc1_b"], ["fc1_out"]))
    
    # ReLU3
    g.add_tensor(Tensor("relu3_out", (1, 120), DataType.INT8))
    g.add_node(Node("relu3", OpType.RELU, ["fc1_out"], ["relu3_out"]))
    
    # FC2: 120->84
    fc2_w = np.random.randint(-64, 64, (120, 84), dtype=np.int8)
    fc2_b = np.zeros(84, dtype=np.int32)
    g.add_tensor(Tensor("fc2_w", (120, 84), DataType.INT8, data=fc2_w))
    g.add_tensor(Tensor("fc2_b", (84,), DataType.INT32, data=fc2_b))
    g.add_tensor(Tensor("fc2_out", (1, 84), DataType.INT8))
    g.add_node(Node("fc2", OpType.GEMM, ["relu3_out", "fc2_w", "fc2_b"], ["fc2_out"]))
    
    # ReLU4
    g.add_tensor(Tensor("relu4_out", (1, 84), DataType.INT8))
    g.add_node(Node("relu4", OpType.RELU, ["fc2_out"], ["relu4_out"]))
    
    # FC3: 84->10
    fc3_w = np.random.randint(-64, 64, (84, 10), dtype=np.int8)
    fc3_b = np.zeros(10, dtype=np.int32)
    g.add_tensor(Tensor("fc3_w", (84, 10), DataType.INT8, data=fc3_w))
    g.add_tensor(Tensor("fc3_b", (10,), DataType.INT32, data=fc3_b))
    g.add_tensor(Tensor("output", (1, 10), DataType.INT8))
    g.add_node(Node("fc3", OpType.GEMM, ["relu4_out", "fc3_w", "fc3_b"], ["output"]))
    
    g.outputs.append("output")
    return g


def create_resnet_block():
    """Create ResNet-style residual block"""
    g = Graph(name="resnet_block")
    
    # Input: 1x64x56x56
    g.add_tensor(Tensor("input", (1, 64, 56, 56), DataType.INT8))
    g.inputs.append("input")
    
    # Conv1: 64->64, 3x3
    conv1_w = np.random.randint(-32, 32, (64, 64, 3, 3), dtype=np.int8)
    g.add_tensor(Tensor("conv1_w", (64, 64, 3, 3), DataType.INT8, data=conv1_w))
    g.add_tensor(Tensor("conv1_out", (1, 64, 56, 56), DataType.INT8))
    g.add_node(Node("conv1", OpType.CONV2D, ["input", "conv1_w"], ["conv1_out"],
                    attrs={'kernel_shape': [3, 3], 'strides': [1, 1], 'pads': [1, 1, 1, 1]}))
    
    # BatchNorm1
    bn1_scale = np.ones(64, dtype=np.float32)
    bn1_bias = np.zeros(64, dtype=np.float32)
    g.add_tensor(Tensor("bn1_scale", (64,), DataType.FLOAT32, data=bn1_scale))
    g.add_tensor(Tensor("bn1_bias", (64,), DataType.FLOAT32, data=bn1_bias))
    g.add_tensor(Tensor("bn1_out", (1, 64, 56, 56), DataType.INT8))
    g.add_node(Node("bn1", OpType.BATCHNORM, ["conv1_out", "bn1_scale", "bn1_bias"], ["bn1_out"]))
    
    # ReLU1
    g.add_tensor(Tensor("relu1_out", (1, 64, 56, 56), DataType.INT8))
    g.add_node(Node("relu1", OpType.RELU, ["bn1_out"], ["relu1_out"]))
    
    # Conv2: 64->64, 3x3
    conv2_w = np.random.randint(-32, 32, (64, 64, 3, 3), dtype=np.int8)
    g.add_tensor(Tensor("conv2_w", (64, 64, 3, 3), DataType.INT8, data=conv2_w))
    g.add_tensor(Tensor("conv2_out", (1, 64, 56, 56), DataType.INT8))
    g.add_node(Node("conv2", OpType.CONV2D, ["relu1_out", "conv2_w"], ["conv2_out"],
                    attrs={'kernel_shape': [3, 3], 'strides': [1, 1], 'pads': [1, 1, 1, 1]}))
    
    # BatchNorm2
    bn2_scale = np.ones(64, dtype=np.float32)
    bn2_bias = np.zeros(64, dtype=np.float32)
    g.add_tensor(Tensor("bn2_scale", (64,), DataType.FLOAT32, data=bn2_scale))
    g.add_tensor(Tensor("bn2_bias", (64,), DataType.FLOAT32, data=bn2_bias))
    g.add_tensor(Tensor("bn2_out", (1, 64, 56, 56), DataType.INT8))
    g.add_node(Node("bn2", OpType.BATCHNORM, ["conv2_out", "bn2_scale", "bn2_bias"], ["bn2_out"]))
    
    # Residual Add
    g.add_tensor(Tensor("add_out", (1, 64, 56, 56), DataType.INT8))
    g.add_node(Node("residual_add", OpType.ADD, ["bn2_out", "input"], ["add_out"]))
    
    # ReLU2
    g.add_tensor(Tensor("output", (1, 64, 56, 56), DataType.INT8))
    g.add_node(Node("relu2", OpType.RELU, ["add_out"], ["output"]))
    
    g.outputs.append("output")
    return g


def create_transformer_block():
    """Create Transformer-style attention block"""
    g = Graph(name="transformer_block")
    
    # Input: batch=1, seq_len=64, hidden=256 (scaled down for SRAM fit)
    seq_len = 64
    hidden = 256
    num_heads = 4
    head_dim = hidden // num_heads
    
    g.add_tensor(Tensor("input", (1, seq_len, hidden), DataType.INT8))
    g.inputs.append("input")
    
    # LayerNorm1
    ln1_scale = np.ones(hidden, dtype=np.float32)
    ln1_bias = np.zeros(hidden, dtype=np.float32)
    g.add_tensor(Tensor("ln1_scale", (hidden,), DataType.FLOAT32, data=ln1_scale))
    g.add_tensor(Tensor("ln1_bias", (hidden,), DataType.FLOAT32, data=ln1_bias))
    g.add_tensor(Tensor("ln1_out", (1, seq_len, hidden), DataType.INT8))
    g.add_node(Node("ln1", OpType.LAYERNORM, ["input", "ln1_scale", "ln1_bias"], ["ln1_out"],
                    attrs={'normalized_shape': [hidden]}))
    
    # Q, K, V projections (simplified - combined into one GEMM each)
    qkv_w = np.random.randint(-32, 32, (hidden, hidden), dtype=np.int8)
    g.add_tensor(Tensor("q_w", (hidden, hidden), DataType.INT8, data=qkv_w.copy()))
    g.add_tensor(Tensor("k_w", (hidden, hidden), DataType.INT8, data=qkv_w.copy()))
    g.add_tensor(Tensor("v_w", (hidden, hidden), DataType.INT8, data=qkv_w.copy()))
    
    g.add_tensor(Tensor("q", (1, seq_len, hidden), DataType.INT8))
    g.add_tensor(Tensor("k", (1, seq_len, hidden), DataType.INT8))
    g.add_tensor(Tensor("v", (1, seq_len, hidden), DataType.INT8))
    
    g.add_node(Node("q_proj", OpType.MATMUL, ["ln1_out", "q_w"], ["q"]))
    g.add_node(Node("k_proj", OpType.MATMUL, ["ln1_out", "k_w"], ["k"]))
    g.add_node(Node("v_proj", OpType.MATMUL, ["ln1_out", "v_w"], ["v"]))
    
    # Attention scores: Q @ K^T / sqrt(d)
    g.add_tensor(Tensor("attn_scores", (1, seq_len, seq_len), DataType.INT8))
    g.add_node(Node("attn_matmul", OpType.MATMUL, ["q", "k"], ["attn_scores"]))
    
    # Softmax
    g.add_tensor(Tensor("attn_probs", (1, seq_len, seq_len), DataType.INT8))
    g.add_node(Node("softmax", OpType.SOFTMAX, ["attn_scores"], ["attn_probs"],
                    attrs={'axis': -1}))
    
    # Attention output: attn @ V
    g.add_tensor(Tensor("attn_out", (1, seq_len, hidden), DataType.INT8))
    g.add_node(Node("attn_output", OpType.MATMUL, ["attn_probs", "v"], ["attn_out"]))
    
    # Output projection
    out_w = np.random.randint(-32, 32, (hidden, hidden), dtype=np.int8)
    g.add_tensor(Tensor("out_w", (hidden, hidden), DataType.INT8, data=out_w))
    g.add_tensor(Tensor("proj_out", (1, seq_len, hidden), DataType.INT8))
    g.add_node(Node("out_proj", OpType.MATMUL, ["attn_out", "out_w"], ["proj_out"]))
    
    # Residual add
    g.add_tensor(Tensor("residual1", (1, seq_len, hidden), DataType.INT8))
    g.add_node(Node("residual1", OpType.ADD, ["proj_out", "input"], ["residual1"]))
    
    # LayerNorm2
    ln2_scale = np.ones(hidden, dtype=np.float32)
    ln2_bias = np.zeros(hidden, dtype=np.float32)
    g.add_tensor(Tensor("ln2_scale", (hidden,), DataType.FLOAT32, data=ln2_scale))
    g.add_tensor(Tensor("ln2_bias", (hidden,), DataType.FLOAT32, data=ln2_bias))
    g.add_tensor(Tensor("ln2_out", (1, seq_len, hidden), DataType.INT8))
    g.add_node(Node("ln2", OpType.LAYERNORM, ["residual1", "ln2_scale", "ln2_bias"], ["ln2_out"],
                    attrs={'normalized_shape': [hidden]}))
    
    # FFN: hidden -> 4*hidden -> hidden with GELU
    ffn_w1 = np.random.randint(-32, 32, (hidden, hidden * 4), dtype=np.int8)
    ffn_w2 = np.random.randint(-32, 32, (hidden * 4, hidden), dtype=np.int8)
    g.add_tensor(Tensor("ffn_w1", (hidden, hidden * 4), DataType.INT8, data=ffn_w1))
    g.add_tensor(Tensor("ffn_w2", (hidden * 4, hidden), DataType.INT8, data=ffn_w2))
    
    g.add_tensor(Tensor("ffn_hidden", (1, seq_len, hidden * 4), DataType.INT8))
    g.add_node(Node("ffn1", OpType.MATMUL, ["ln2_out", "ffn_w1"], ["ffn_hidden"]))
    
    # GELU activation
    g.add_tensor(Tensor("ffn_gelu", (1, seq_len, hidden * 4), DataType.INT8))
    g.add_node(Node("gelu", OpType.GELU, ["ffn_hidden"], ["ffn_gelu"]))
    
    g.add_tensor(Tensor("ffn_out", (1, seq_len, hidden), DataType.INT8))
    g.add_node(Node("ffn2", OpType.MATMUL, ["ffn_gelu", "ffn_w2"], ["ffn_out"]))
    
    # Final residual
    g.add_tensor(Tensor("output", (1, seq_len, hidden), DataType.INT8))
    g.add_node(Node("residual2", OpType.ADD, ["ffn_out", "residual1"], ["output"]))
    
    g.outputs.append("output")
    return g


def compile_and_assemble(graph):
    """Compile graph and verify assembly"""
    tiler = TilingEngine()
    tiler.tile_graph(graph)
    
    scheduler = Scheduler()
    schedule = scheduler.schedule(graph)
    
    codegen = CodeGenerator()
    asm_code = codegen.generate(graph, schedule)
    
    # Try to assemble
    assembler = Assembler()
    instructions = []
    errors = []
    
    for i, line in enumerate(asm_code.split('\n'), 1):
        try:
            assembler.line_num = i
            instr = assembler.assemble_line(line)
            if instr:
                instructions.append(instr)
        except Exception as e:
            errors.append(f"Line {i}: {e}")
    
    return asm_code, instructions, errors


def test_lenet():
    """Test LeNet compilation"""
    print("\n" + "=" * 60)
    print("Test: LeNet-5")
    print("=" * 60)
    
    graph = create_lenet_graph()
    print(f"Graph: {graph.name}")
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Tensors: {len(graph.tensors)}")
    
    asm_code, instructions, errors = compile_and_assemble(graph)
    
    if errors:
        print(f"❌ Assembly errors: {len(errors)}")
        for e in errors[:5]:
            print(f"  {e}")
        return False
    
    # Count operations
    conv_count = asm_code.count("TENSOR.IM2COL") + asm_code.count("Conv2D")
    pool_count = asm_code.count("TENSOR.MAXPOOL")
    relu_count = asm_code.count("VECTOR.RELU")
    gemm_count = asm_code.count("TENSOR.GEMM")
    
    print(f"\nGenerated code:")
    print(f"  Lines: {asm_code.count(chr(10))}")
    print(f"  Instructions: {len(instructions)}")
    print(f"  Conv ops: {conv_count}")
    print(f"  Pool ops: {pool_count}")
    print(f"  ReLU ops: {relu_count}")
    print(f"  GEMM ops: {gemm_count}")
    
    print(f"\n✓ LeNet compilation successful")
    return True


def test_resnet_block():
    """Test ResNet block compilation"""
    print("\n" + "=" * 60)
    print("Test: ResNet Block")
    print("=" * 60)
    
    graph = create_resnet_block()
    print(f"Graph: {graph.name}")
    print(f"  Nodes: {len(graph.nodes)}")
    
    asm_code, instructions, errors = compile_and_assemble(graph)
    
    if errors:
        print(f"❌ Assembly errors: {len(errors)}")
        for e in errors[:5]:
            print(f"  {e}")
        return False
    
    # Count operations
    conv_count = asm_code.count("Conv2D")
    bn_count = asm_code.count("VECTOR.BATCHNORM")
    add_count = asm_code.count("VECTOR.ADD")
    relu_count = asm_code.count("VECTOR.RELU")
    
    print(f"\nGenerated code:")
    print(f"  Lines: {asm_code.count(chr(10))}")
    print(f"  Instructions: {len(instructions)}")
    print(f"  Conv ops: {conv_count}")
    print(f"  BatchNorm ops: {bn_count}")
    print(f"  Add ops: {add_count}")
    print(f"  ReLU ops: {relu_count}")
    
    print(f"\n✓ ResNet block compilation successful")
    return True


def test_transformer():
    """Test Transformer block compilation"""
    print("\n" + "=" * 60)
    print("Test: Transformer Block")
    print("=" * 60)
    
    graph = create_transformer_block()
    print(f"Graph: {graph.name}")
    print(f"  Nodes: {len(graph.nodes)}")
    
    asm_code, instructions, errors = compile_and_assemble(graph)
    
    if errors:
        print(f"❌ Assembly errors: {len(errors)}")
        for e in errors[:5]:
            print(f"  {e}")
        return False
    
    # Count operations  
    matmul_count = asm_code.count("TENSOR.GEMM")
    ln_count = asm_code.count("LAYERNORM")
    gelu_count = asm_code.count("GELU")
    softmax_count = asm_code.count("SOFTMAX")
    add_count = asm_code.count("VECTOR.ADD")
    
    print(f"\nGenerated code:")
    print(f"  Lines: {asm_code.count(chr(10))}")
    print(f"  Instructions: {len(instructions)}")
    print(f"  MatMul/GEMM ops: {matmul_count}")
    print(f"  LayerNorm ops: {ln_count}")
    print(f"  GELU ops: {gelu_count}")
    print(f"  Softmax ops: {softmax_count}")
    print(f"  Add ops: {add_count}")
    
    print(f"\n✓ Transformer block compilation successful")
    return True


def main():
    """Run all architecture tests"""
    print("=" * 60)
    print("Extended Operation Tests")
    print("=" * 60)
    
    results = []
    
    results.append(("LeNet-5", test_lenet()))
    results.append(("ResNet Block", test_resnet_block()))
    results.append(("Transformer Block", test_transformer()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
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

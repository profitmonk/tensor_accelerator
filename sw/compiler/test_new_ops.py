#!/usr/bin/env python3
"""
Test Suite for Extended Operations

Tests new operations added for:
- MobileNet: DepthwiseConv2D, ReLU6
- EfficientNet: Swish (SiLU), DepthwiseConv2D
- Transformer: Multi-Head Attention, GroupNorm
- General: Concat, Split, Squeeze, Unsqueeze
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from codegen.codegen import CodeGenerator, CodeGenConfig
from ir.graph import Graph, Node, Tensor, OpType, DataType
from scheduler.scheduler import ScheduleEntry, Scheduler
from tiler.tiler import TilingEngine, HardwareConfig


def test_mobilenet_block():
    """Test MobileNet-style depthwise separable convolution block
    
    Architecture:
    - DepthwiseConv 3x3
    - BatchNorm
    - ReLU6
    - PointwiseConv 1x1 (regular conv)
    - BatchNorm
    - ReLU6
    """
    print("\n" + "="*60)
    print("TEST: MobileNet Depthwise Separable Block")
    print("="*60)
    
    g = Graph(name="mobilenet_block")
    
    # Input: [1, 32, 56, 56]
    g.add_tensor(Tensor("input", (1, 32, 56, 56), DataType.INT8))
    g.inputs.append("input")
    
    # Depthwise conv weights: [32, 1, 3, 3]
    dw_weight = np.random.randint(-128, 127, (32, 1, 3, 3), dtype=np.int8)
    g.add_tensor(Tensor("dw_weight", (32, 1, 3, 3), DataType.INT8, data=dw_weight))
    g.add_tensor(Tensor("dw_bias", (32,), DataType.INT32, 
                        data=np.zeros(32, dtype=np.int32)))
    g.add_tensor(Tensor("dw_out", (1, 32, 56, 56), DataType.INT8))
    
    # Depthwise conv
    g.add_node(Node(
        name="dw_conv",
        op_type=OpType.DEPTHWISE_CONV2D,
        inputs=["input", "dw_weight", "dw_bias"],
        outputs=["dw_out"],
        attrs={"kernel_shape": [3, 3], "strides": [1, 1], "pads": [1, 1, 1, 1]}
    ))
    
    # BatchNorm
    g.add_tensor(Tensor("bn1_scale", (32,), DataType.INT8,
                        data=np.ones(32, dtype=np.int8) * 64))  # ~1.0 in Q7
    g.add_tensor(Tensor("bn1_bias", (32,), DataType.INT32,
                        data=np.zeros(32, dtype=np.int32)))
    g.add_tensor(Tensor("bn1_out", (1, 32, 56, 56), DataType.INT8))
    
    g.add_node(Node(
        name="bn1",
        op_type=OpType.BATCHNORM,
        inputs=["dw_out", "bn1_scale", "bn1_bias"],
        outputs=["bn1_out"]
    ))
    
    # ReLU6
    g.add_tensor(Tensor("relu6_out", (1, 32, 56, 56), DataType.INT8))
    g.add_node(Node(
        name="relu6",
        op_type=OpType.RELU6,
        inputs=["bn1_out"],
        outputs=["relu6_out"]
    ))
    
    # Pointwise conv 1x1: expand 32 -> 64
    pw_weight = np.random.randint(-128, 127, (64, 32, 1, 1), dtype=np.int8)
    g.add_tensor(Tensor("pw_weight", (64, 32, 1, 1), DataType.INT8, data=pw_weight))
    g.add_tensor(Tensor("output", (1, 64, 56, 56), DataType.INT8))
    
    g.add_node(Node(
        name="pw_conv",
        op_type=OpType.CONV2D,
        inputs=["relu6_out", "pw_weight"],
        outputs=["output"],
        attrs={"kernel_shape": [1, 1], "strides": [1, 1], "pads": [0, 0, 0, 0]}
    ))
    
    g.outputs.append("output")
    
    # Validate
    errors = g.validate()
    assert not errors, f"Graph validation failed: {errors}"
    
    # Schedule
    scheduler = Scheduler()
    schedule = scheduler.schedule(g)
    
    # Generate code
    codegen = CodeGenerator()
    asm_code = codegen.generate(g, schedule)
    
    print(f"Graph: {g.name}")
    print(f"  Nodes: {len(g.nodes)}")
    print(f"  Schedule entries: {len(schedule)}")
    print(f"  Generated {len(asm_code.splitlines())} lines of assembly")
    
    # Verify expected ops
    assert "DEPTHWISE_CONV" in asm_code, "Missing DEPTHWISE_CONV"
    assert "RELU6" in asm_code, "Missing RELU6"
    assert "BATCHNORM" in asm_code, "Missing BATCHNORM"
    
    print("✓ MobileNet block test passed")
    return True


def test_transformer_attention():
    """Test Transformer-style multi-head attention
    
    Architecture:
    - Q, K, V projections (linear)
    - Multi-Head Attention
    - Output projection
    """
    print("\n" + "="*60)
    print("TEST: Transformer Multi-Head Attention")
    print("="*60)
    
    # Config
    batch = 1
    seq_len = 16
    hidden = 64
    num_heads = 4
    head_dim = hidden // num_heads
    
    g = Graph(name="transformer_attention")
    
    # Input: [batch, seq_len, hidden]
    g.add_tensor(Tensor("input", (batch, seq_len, hidden), DataType.INT8))
    g.inputs.append("input")
    
    # Q, K, V projection weights
    for name in ["q", "k", "v"]:
        weight = np.random.randint(-128, 127, (hidden, hidden), dtype=np.int8)
        g.add_tensor(Tensor(f"{name}_weight", (hidden, hidden), DataType.INT8, data=weight))
        g.add_tensor(Tensor(f"{name}_proj", (batch, seq_len, hidden), DataType.INT8))
        
        g.add_node(Node(
            name=f"{name}_linear",
            op_type=OpType.GEMM,
            inputs=["input" if name == "q" else f"input", f"{name}_weight"],
            outputs=[f"{name}_proj"],
            attrs={"transB": True}
        ))
    
    # Multi-Head Attention
    g.add_tensor(Tensor("attn_out", (batch, seq_len, hidden), DataType.INT8))
    g.add_node(Node(
        name="attention",
        op_type=OpType.ATTENTION,
        inputs=["q_proj", "k_proj", "v_proj"],
        outputs=["attn_out"],
        attrs={"num_heads": num_heads, "head_dim": head_dim}
    ))
    
    # Output projection
    out_weight = np.random.randint(-128, 127, (hidden, hidden), dtype=np.int8)
    g.add_tensor(Tensor("out_weight", (hidden, hidden), DataType.INT8, data=out_weight))
    g.add_tensor(Tensor("output", (batch, seq_len, hidden), DataType.INT8))
    
    g.add_node(Node(
        name="out_linear",
        op_type=OpType.GEMM,
        inputs=["attn_out", "out_weight"],
        outputs=["output"],
        attrs={"transB": True}
    ))
    
    g.outputs.append("output")
    
    # Validate
    errors = g.validate()
    assert not errors, f"Graph validation failed: {errors}"
    
    # Schedule
    scheduler = Scheduler()
    schedule = scheduler.schedule(g)
    
    # Generate code
    codegen = CodeGenerator()
    asm_code = codegen.generate(g, schedule)
    
    print(f"Graph: {g.name}")
    print(f"  Batch: {batch}, SeqLen: {seq_len}, Hidden: {hidden}")
    print(f"  Num heads: {num_heads}, Head dim: {head_dim}")
    print(f"  Nodes: {len(g.nodes)}")
    print(f"  Generated {len(asm_code.splitlines())} lines of assembly")
    
    # Verify attention components
    assert "Multi-Head Attention" in asm_code, "Missing attention comment"
    assert "SOFTMAX" in asm_code, "Missing softmax in attention"
    
    print("✓ Transformer attention test passed")
    return True


def test_efficientnet_block():
    """Test EfficientNet-style MBConv block with Swish activation
    
    Architecture:
    - Expansion 1x1 conv
    - Swish activation
    - Depthwise 3x3 conv
    - Swish activation
    - Squeeze-and-Excitation (skip for now)
    - Projection 1x1 conv
    - Residual add
    """
    print("\n" + "="*60)
    print("TEST: EfficientNet MBConv Block (Swish)")
    print("="*60)
    
    g = Graph(name="efficientnet_mbconv")
    
    # Input: [1, 32, 28, 28]
    g.add_tensor(Tensor("input", (1, 32, 28, 28), DataType.INT8))
    g.inputs.append("input")
    
    # Expansion 1x1: 32 -> 128 (4x expansion)
    exp_weight = np.random.randint(-128, 127, (128, 32, 1, 1), dtype=np.int8)
    g.add_tensor(Tensor("expand_weight", (128, 32, 1, 1), DataType.INT8, data=exp_weight))
    g.add_tensor(Tensor("expand_out", (1, 128, 28, 28), DataType.INT8))
    
    g.add_node(Node(
        name="expand_conv",
        op_type=OpType.CONV2D,
        inputs=["input", "expand_weight"],
        outputs=["expand_out"],
        attrs={"kernel_shape": [1, 1], "strides": [1, 1], "pads": [0, 0, 0, 0]}
    ))
    
    # Swish after expansion
    g.add_tensor(Tensor("swish1_out", (1, 128, 28, 28), DataType.INT8))
    g.add_node(Node(
        name="swish1",
        op_type=OpType.SWISH,
        inputs=["expand_out"],
        outputs=["swish1_out"]
    ))
    
    # Depthwise 3x3
    dw_weight = np.random.randint(-128, 127, (128, 1, 3, 3), dtype=np.int8)
    g.add_tensor(Tensor("dw_weight", (128, 1, 3, 3), DataType.INT8, data=dw_weight))
    g.add_tensor(Tensor("dw_out", (1, 128, 28, 28), DataType.INT8))
    
    g.add_node(Node(
        name="dw_conv",
        op_type=OpType.DEPTHWISE_CONV2D,
        inputs=["swish1_out", "dw_weight"],
        outputs=["dw_out"],
        attrs={"kernel_shape": [3, 3], "strides": [1, 1], "pads": [1, 1, 1, 1]}
    ))
    
    # Swish after depthwise
    g.add_tensor(Tensor("swish2_out", (1, 128, 28, 28), DataType.INT8))
    g.add_node(Node(
        name="swish2",
        op_type=OpType.SWISH,
        inputs=["dw_out"],
        outputs=["swish2_out"]
    ))
    
    # Projection 1x1: 128 -> 32
    proj_weight = np.random.randint(-128, 127, (32, 128, 1, 1), dtype=np.int8)
    g.add_tensor(Tensor("proj_weight", (32, 128, 1, 1), DataType.INT8, data=proj_weight))
    g.add_tensor(Tensor("proj_out", (1, 32, 28, 28), DataType.INT8))
    
    g.add_node(Node(
        name="proj_conv",
        op_type=OpType.CONV2D,
        inputs=["swish2_out", "proj_weight"],
        outputs=["proj_out"],
        attrs={"kernel_shape": [1, 1], "strides": [1, 1], "pads": [0, 0, 0, 0]}
    ))
    
    # Residual add
    g.add_tensor(Tensor("output", (1, 32, 28, 28), DataType.INT8))
    g.add_node(Node(
        name="residual",
        op_type=OpType.ADD,
        inputs=["input", "proj_out"],
        outputs=["output"]
    ))
    
    g.outputs.append("output")
    
    # Validate and generate
    errors = g.validate()
    assert not errors, f"Graph validation failed: {errors}"
    
    scheduler = Scheduler()
    schedule = scheduler.schedule(g)
    
    codegen = CodeGenerator()
    asm_code = codegen.generate(g, schedule)
    
    print(f"Graph: {g.name}")
    print(f"  Nodes: {len(g.nodes)}")
    print(f"  Schedule entries: {len(schedule)}")
    print(f"  Generated {len(asm_code.splitlines())} lines of assembly")
    
    # Verify
    assert "Swish" in asm_code, "Missing Swish"
    assert "SIGMOID" in asm_code, "Swish should use sigmoid internally"
    assert "DEPTHWISE_CONV" in asm_code, "Missing depthwise conv"
    
    print("✓ EfficientNet MBConv test passed")
    return True


def test_groupnorm_unet():
    """Test GroupNorm (commonly used in U-Net, diffusion models)"""
    print("\n" + "="*60)
    print("TEST: GroupNorm (U-Net style)")
    print("="*60)
    
    g = Graph(name="groupnorm_test")
    
    # Input: [1, 256, 32, 32] - typical U-Net feature map
    g.add_tensor(Tensor("input", (1, 256, 32, 32), DataType.INT8))
    g.inputs.append("input")
    
    # GroupNorm with 32 groups (8 channels per group)
    g.add_tensor(Tensor("gn_scale", (256,), DataType.INT8,
                        data=np.ones(256, dtype=np.int8) * 64))
    g.add_tensor(Tensor("gn_bias", (256,), DataType.INT32,
                        data=np.zeros(256, dtype=np.int32)))
    g.add_tensor(Tensor("output", (1, 256, 32, 32), DataType.INT8))
    
    g.add_node(Node(
        name="groupnorm",
        op_type=OpType.GROUPNORM,
        inputs=["input", "gn_scale", "gn_bias"],
        outputs=["output"],
        attrs={"num_groups": 32}
    ))
    
    g.outputs.append("output")
    
    # Validate and generate
    errors = g.validate()
    assert not errors, f"Graph validation failed: {errors}"
    
    scheduler = Scheduler()
    schedule = scheduler.schedule(g)
    
    codegen = CodeGenerator()
    asm_code = codegen.generate(g, schedule)
    
    print(f"Graph: {g.name}")
    print(f"  Input: [1, 256, 32, 32]")
    print(f"  Groups: 32 (8 channels/group)")
    print(f"  Generated {len(asm_code.splitlines())} lines of assembly")
    
    assert "GroupNorm" in asm_code, "Missing GroupNorm comment"
    assert "GROUPNORM" in asm_code, "Missing GROUPNORM instruction"
    
    print("✓ GroupNorm test passed")
    return True


def test_concat_split():
    """Test Concat and Split operations"""
    print("\n" + "="*60)
    print("TEST: Concat and Split Operations")
    print("="*60)
    
    g = Graph(name="concat_split_test")
    
    # Two inputs to concat
    g.add_tensor(Tensor("a", (1, 64, 16, 16), DataType.INT8))
    g.add_tensor(Tensor("b", (1, 64, 16, 16), DataType.INT8))
    g.inputs.extend(["a", "b"])
    
    # Concat along channel axis
    g.add_tensor(Tensor("concat_out", (1, 128, 16, 16), DataType.INT8))
    g.add_node(Node(
        name="concat",
        op_type=OpType.CONCAT,
        inputs=["a", "b"],
        outputs=["concat_out"],
        attrs={"axis": 1}
    ))
    
    # Process concatenated tensor
    weight = np.random.randint(-128, 127, (128, 128, 1, 1), dtype=np.int8)
    g.add_tensor(Tensor("conv_weight", (128, 128, 1, 1), DataType.INT8, data=weight))
    g.add_tensor(Tensor("conv_out", (1, 128, 16, 16), DataType.INT8))
    
    g.add_node(Node(
        name="conv",
        op_type=OpType.CONV2D,
        inputs=["concat_out", "conv_weight"],
        outputs=["conv_out"],
        attrs={"kernel_shape": [1, 1]}
    ))
    
    # Split back
    g.add_tensor(Tensor("out_a", (1, 64, 16, 16), DataType.INT8))
    g.add_tensor(Tensor("out_b", (1, 64, 16, 16), DataType.INT8))
    
    g.add_node(Node(
        name="split",
        op_type=OpType.SPLIT,
        inputs=["conv_out"],
        outputs=["out_a", "out_b"],
        attrs={"axis": 1, "split": [64, 64]}
    ))
    
    g.outputs.extend(["out_a", "out_b"])
    
    # Validate and generate
    errors = g.validate()
    assert not errors, f"Graph validation failed: {errors}"
    
    scheduler = Scheduler()
    schedule = scheduler.schedule(g)
    
    codegen = CodeGenerator()
    asm_code = codegen.generate(g, schedule)
    
    print(f"Graph: {g.name}")
    print(f"  Nodes: {len(g.nodes)}")
    print(f"  Generated {len(asm_code.splitlines())} lines of assembly")
    
    assert "Concat" in asm_code, "Missing Concat"
    assert "Split" in asm_code, "Missing Split"
    
    print("✓ Concat/Split test passed")
    return True


def run_all_tests():
    """Run all new operation tests"""
    print("\n" + "="*60)
    print("EXTENDED OPERATIONS TEST SUITE")
    print("="*60)
    
    tests = [
        ("MobileNet Block", test_mobilenet_block),
        ("Transformer Attention", test_transformer_attention),
        ("EfficientNet MBConv", test_efficientnet_block),
        ("GroupNorm U-Net", test_groupnorm_unet),
        ("Concat/Split", test_concat_split),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
                print(f"✗ {name} failed")
        except Exception as e:
            failed += 1
            print(f"✗ {name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

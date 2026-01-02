"""
Code Generator

Translates scheduled IR into assembly instructions for the tensor accelerator.
Generates code compatible with the LCP assembler (assembler.py).

Supported architectures:
- LeNet: Conv2D, MaxPool, ReLU, GEMM
- ResNet: Conv2D, BatchNorm, ReLU, Add, GlobalAvgPool, GEMM
- LLMs/Transformers: MatMul, LayerNorm, GELU, Softmax, Add
"""

import sys
import os
from typing import Dict, List, Optional, Tuple, TextIO
from dataclasses import dataclass
from io import StringIO
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir.graph import Graph, Node, Tensor, OpType, DataType
from scheduler.scheduler import ScheduleEntry, Scheduler
from tiler.tiler import TilingEngine, HardwareConfig, GEMMTileConfig


@dataclass
class CodeGenConfig:
    """Code generation configuration"""
    # Memory map
    ddr_weights_base: int = 0x00100000
    ddr_input_base: int = 0x00200000
    ddr_output_base: int = 0x00300000
    
    # SRAM addresses (per TPC)
    sram_act_a: int = 0x0000
    sram_act_b: int = 0x1000
    sram_weight_a: int = 0x2000
    sram_weight_b: int = 0x4000
    sram_output: int = 0x6000
    sram_scratch: int = 0x7000
    sram_scratch2: int = 0x8000
    
    emit_comments: bool = True
    use_double_buffering: bool = False


class CodeGenerator:
    """
    Generates assembly code for the tensor accelerator
    
    Supports:
    - Compute: GEMM, MatMul, Conv2D
    - Activations: ReLU, GELU, Sigmoid, Tanh, Softmax
    - Normalization: BatchNorm, LayerNorm
    - Pooling: MaxPool, AvgPool, GlobalAvgPool
    - Elementwise: Add, Sub, Mul, Div
    - Shape: Reshape, Flatten, Transpose
    """
    
    def __init__(self, config: Optional[CodeGenConfig] = None,
                 hw_config: Optional[HardwareConfig] = None):
        self.config = config or CodeGenConfig()
        self.hw = hw_config or HardwareConfig()
        self.output = StringIO()
        self.indent_level = 0
        self.label_counter = 0
        self.weights: List[Tuple[str, int, bytes]] = []
        
    def generate(self, graph: Graph, schedule: List[ScheduleEntry]) -> str:
        """Generate assembly code for the scheduled graph"""
        self.output = StringIO()
        self.weights = []
        
        self._emit_header(graph)
        self._emit_constants(graph)
        self._emit_newline()
        
        self._emit_comment("=" * 60)
        self._emit_comment("Main Program")
        self._emit_comment("=" * 60)
        
        for entry in schedule:
            node = graph.get_node(entry.node_name)
            if node:
                self._emit_node(graph, node, entry)
        
        self._emit_newline()
        self._emit_comment("End of program")
        self._emit("HALT")
        
        return self.output.getvalue()
    
    def generate_weights(self, graph: Graph) -> bytes:
        """Generate binary weight data"""
        weight_data = bytearray()
        
        for name, tensor in graph.tensors.items():
            if tensor.data is not None:
                data_bytes = tensor.data.tobytes()
                weight_data.extend(data_bytes)
                padding = (16 - (len(data_bytes) % 16)) % 16
                weight_data.extend(bytes(padding))
        
        return bytes(weight_data)
    
    # =========================================================================
    # Assembly emission helpers
    # =========================================================================
    
    def _emit(self, line: str):
        self.output.write(line + "\n")
    
    def _emit_comment(self, text: str):
        if self.config.emit_comments:
            self._emit(f"# {text}")
    
    def _emit_newline(self):
        self.output.write("\n")
    
    def _emit_label(self, label: str):
        self._emit(f"{label}:")
    
    def _new_label(self, prefix: str = "L") -> str:
        label = f"{prefix}{self.label_counter}"
        self.label_counter += 1
        return label
    
    def _emit_header(self, graph: Graph):
        self._emit_comment("=" * 60)
        self._emit_comment(f"Generated code for: {graph.name}")
        self._emit_comment(f"Nodes: {len(graph.nodes)}")
        self._emit_comment(f"Tensors: {len(graph.tensors)}")
        self._emit_comment("=" * 60)
        self._emit_newline()
    
    def _emit_constants(self, graph: Graph):
        self._emit_comment("Memory addresses")
        
        offset = 0
        for name, tensor in graph.tensors.items():
            if tensor.data is not None:
                self._emit(f".equ WEIGHT_{name.upper()}_ADDR, 0x{self.config.ddr_weights_base + offset:08X}")
                self._emit(f".equ WEIGHT_{name.upper()}_SIZE, 0x{tensor.size_bytes:X}")
                offset += tensor.size_bytes
                offset = (offset + 15) & ~15
        
        self._emit_newline()
        self._emit(f".equ SRAM_ACT_A, 0x{self.config.sram_act_a:04X}")
        self._emit(f".equ SRAM_ACT_B, 0x{self.config.sram_act_b:04X}")
        self._emit(f".equ SRAM_WT_A, 0x{self.config.sram_weight_a:04X}")
        self._emit(f".equ SRAM_WT_B, 0x{self.config.sram_weight_b:04X}")
        self._emit(f".equ SRAM_OUT, 0x{self.config.sram_output:04X}")
        self._emit(f".equ SRAM_SCRATCH, 0x{self.config.sram_scratch:04X}")
    
    # =========================================================================
    # Node emission dispatcher
    # =========================================================================
    
    def _emit_node(self, graph: Graph, node: Node, entry: ScheduleEntry):
        self._emit_newline()
        tile_str = f" (tile {entry.tile_id})" if entry.tile_id is not None else ""
        self._emit_comment(f"Node: {node.name}{tile_str}")
        self._emit_comment(f"Op: {node.op_type.name}")
        
        for tensor_name, ddr_addr, sram_addr, size in entry.dma_loads:
            self._emit_dma_load(tensor_name, ddr_addr, sram_addr, size)
        
        # Dispatch table
        handlers = {
            OpType.GEMM: self._emit_gemm,
            OpType.MATMUL: self._emit_matmul,
            OpType.CONV2D: self._emit_conv2d,
            OpType.RELU: self._emit_relu,
            OpType.GELU: self._emit_gelu,
            OpType.SIGMOID: self._emit_sigmoid,
            OpType.TANH: self._emit_tanh,
            OpType.SOFTMAX: self._emit_softmax,
            OpType.BATCHNORM: self._emit_batchnorm,
            OpType.LAYERNORM: self._emit_layernorm,
            OpType.MAXPOOL: self._emit_maxpool,
            OpType.AVGPOOL: self._emit_avgpool,
            OpType.GLOBALAVGPOOL: self._emit_globalavgpool,
            OpType.ADD: self._emit_add,
            OpType.SUB: self._emit_sub,
            OpType.MUL: self._emit_mul,
            OpType.DIV: self._emit_div,
            OpType.RESHAPE: self._emit_reshape,
            OpType.FLATTEN: self._emit_flatten,
            OpType.TRANSPOSE: self._emit_transpose,
        }
        
        handler = handlers.get(node.op_type)
        if handler:
            handler(graph, node, entry)
        else:
            self._emit_comment(f"WARNING: {node.op_type.name} not implemented")
            self._emit("NOP")
        
        for tensor_name, ddr_addr, sram_addr, size in entry.dma_stores:
            self._emit_dma_store(tensor_name, ddr_addr, sram_addr, size)
    
    def _emit_dma_load(self, name: str, ddr_addr: int, sram_addr: int, size: int):
        self._emit_comment(f"Load {name} from DDR")
        self._emit(f"DMA.LOAD_1D 0x{sram_addr:04X}, 0x{ddr_addr:08X}, {size}")
        self._emit("SYNC.WAIT_DMA")
    
    def _emit_dma_store(self, name: str, ddr_addr: int, sram_addr: int, size: int):
        self._emit_comment(f"Store {name} to DDR")
        self._emit(f"DMA.STORE_1D 0x{ddr_addr:08X}, 0x{sram_addr:04X}, {size}")
        self._emit("SYNC.WAIT_DMA")
    
    # =========================================================================
    # Compute operations
    # =========================================================================
    
    def _emit_gemm(self, graph: Graph, node: Node, entry: ScheduleEntry):
        """GEMM: C = A @ B + bias"""
        input_name = node.inputs[0]
        weight_name = node.inputs[1]
        
        input_tensor = graph.get_tensor(input_name)
        weight_tensor = graph.get_tensor(weight_name)
        
        if not input_tensor or not weight_tensor:
            self._emit_comment("ERROR: Missing tensors for GEMM")
            return
        
        tile_config = node.attrs.get('tile_config')
        
        if tile_config:
            M, N, K = tile_config.tile_m, tile_config.tile_n, tile_config.tile_k
        else:
            M = input_tensor.shape[-2] if len(input_tensor.shape) >= 2 else 1
            K = input_tensor.shape[-1]
            transB = node.get_attr('transB', 0)
            N = weight_tensor.shape[0] if transB else weight_tensor.shape[-1]
        
        input_addr = entry.input_addrs.get(input_name, self.config.sram_act_a)
        weight_addr = entry.input_addrs.get(weight_name, self.config.sram_weight_a)
        output_addr = entry.output_addr or self.config.sram_output
        
        is_accumulate = entry.tile_id is not None and entry.tile_id > 0 and entry.depends_on
        
        if is_accumulate:
            self._emit_comment(f"GEMM accumulate: {M}x{K} @ {K}x{N}")
            self._emit(f"TENSOR.GEMM_ACC 0x{output_addr:04X}, 0x{input_addr:04X}, 0x{weight_addr:04X}, {M}, {N}, {K}")
        else:
            self._emit_comment(f"GEMM: {M}x{K} @ {K}x{N}")
            self._emit(f"TENSOR.GEMM 0x{output_addr:04X}, 0x{input_addr:04X}, 0x{weight_addr:04X}, {M}, {N}, {K}")
        
        self._emit("SYNC.WAIT_MXU")
        
        if len(node.inputs) > 2:
            bias_name = node.inputs[2]
            bias_addr = entry.input_addrs.get(bias_name, 0)
            if bias_addr:
                self._emit_comment("Add bias")
                self._emit(f"VECTOR.ADD 0x{output_addr:04X}, 0x{output_addr:04X}, 0x{bias_addr:04X}, {M * N}")
                self._emit("SYNC.WAIT_VPU")
    
    def _emit_matmul(self, graph: Graph, node: Node, entry: ScheduleEntry):
        """MatMul (same as GEMM without bias)"""
        self._emit_gemm(graph, node, entry)
    
    def _emit_conv2d(self, graph: Graph, node: Node, entry: ScheduleEntry):
        """Conv2D via im2col + GEMM"""
        input_name = node.inputs[0]
        weight_name = node.inputs[1]
        
        input_tensor = graph.get_tensor(input_name)
        weight_tensor = graph.get_tensor(weight_name)
        
        if not input_tensor or not weight_tensor:
            self._emit_comment("ERROR: Missing tensors for Conv2D")
            return
        
        kernel_shape = node.get_attr('kernel_shape', [3, 3])
        strides = node.get_attr('strides', [1, 1])
        pads = node.get_attr('pads', [0, 0, 0, 0])
        
        if len(input_tensor.shape) == 4:
            N, C, H, W = input_tensor.shape
        else:
            C, H, W = input_tensor.shape
            N = 1
        
        K_out = weight_tensor.shape[0]
        kH, kW = kernel_shape
        
        H_out = (H + pads[0] + pads[2] - kH) // strides[0] + 1
        W_out = (W + pads[1] + pads[3] - kW) // strides[1] + 1
        
        M = N * H_out * W_out
        K_gemm = C * kH * kW
        N_gemm = K_out
        
        self._emit_comment(f"Conv2D via im2col + GEMM")
        self._emit_comment(f"  Input: [{N},{C},{H},{W}], Kernel: {kH}x{kW}")
        self._emit_comment(f"  Output: [{N},{K_out},{H_out},{W_out}]")
        self._emit_comment(f"  GEMM: {M}x{K_gemm} @ {K_gemm}x{N_gemm}")
        
        input_addr = entry.input_addrs.get(input_name, self.config.sram_act_a)
        weight_addr = entry.input_addrs.get(weight_name, self.config.sram_weight_a)
        output_addr = entry.output_addr or self.config.sram_output
        scratch_addr = self.config.sram_scratch
        
        # im2col + GEMM
        self._emit(f"TENSOR.IM2COL 0x{scratch_addr:04X}, 0x{input_addr:04X}, {C}, {H}, {W}, {kH}, {kW}, {strides[0]}, {pads[0]}")
        self._emit("SYNC.WAIT_MXU")
        self._emit(f"TENSOR.GEMM 0x{output_addr:04X}, 0x{scratch_addr:04X}, 0x{weight_addr:04X}, {M}, {N_gemm}, {K_gemm}")
        self._emit("SYNC.WAIT_MXU")
        
        if len(node.inputs) > 2:
            bias_name = node.inputs[2]
            bias_addr = entry.input_addrs.get(bias_name, 0)
            if bias_addr:
                self._emit_comment("Add bias (broadcast)")
                self._emit(f"VECTOR.BIAS_ADD 0x{output_addr:04X}, 0x{output_addr:04X}, 0x{bias_addr:04X}, {M}, {N_gemm}")
                self._emit("SYNC.WAIT_VPU")
    
    # =========================================================================
    # Activation functions
    # =========================================================================
    
    def _emit_relu(self, graph: Graph, node: Node, entry: ScheduleEntry):
        """ReLU: y = max(0, x)"""
        input_name = node.inputs[0]
        input_tensor = graph.get_tensor(input_name)
        
        if input_tensor:
            num_elements = input_tensor.numel
            input_addr = entry.input_addrs.get(input_name, self.config.sram_act_a)
            output_addr = entry.output_addr or input_addr
            
            self._emit(f"VECTOR.RELU 0x{output_addr:04X}, 0x{input_addr:04X}, {num_elements}")
            self._emit("SYNC.WAIT_VPU")
    
    def _emit_gelu(self, graph: Graph, node: Node, entry: ScheduleEntry):
        """GELU approximation: x * sigmoid(1.702 * x)"""
        input_name = node.inputs[0]
        input_tensor = graph.get_tensor(input_name)
        
        if input_tensor:
            num_elements = input_tensor.numel
            input_addr = entry.input_addrs.get(input_name, self.config.sram_act_a)
            output_addr = entry.output_addr or input_addr
            scratch_addr = self.config.sram_scratch
            
            self._emit_comment("GELU: x * sigmoid(1.702 * x)")
            self._emit(f"VECTOR.SCALE 0x{scratch_addr:04X}, 0x{input_addr:04X}, 1702, {num_elements}")
            self._emit("SYNC.WAIT_VPU")
            self._emit(f"VECTOR.SIGMOID 0x{scratch_addr:04X}, 0x{scratch_addr:04X}, {num_elements}")
            self._emit("SYNC.WAIT_VPU")
            self._emit(f"VECTOR.MUL 0x{output_addr:04X}, 0x{input_addr:04X}, 0x{scratch_addr:04X}, {num_elements}")
            self._emit("SYNC.WAIT_VPU")
    
    def _emit_sigmoid(self, graph: Graph, node: Node, entry: ScheduleEntry):
        """Sigmoid: y = 1 / (1 + exp(-x))"""
        input_name = node.inputs[0]
        input_tensor = graph.get_tensor(input_name)
        
        if input_tensor:
            num_elements = input_tensor.numel
            input_addr = entry.input_addrs.get(input_name, self.config.sram_act_a)
            output_addr = entry.output_addr or input_addr
            
            self._emit(f"VECTOR.SIGMOID 0x{output_addr:04X}, 0x{input_addr:04X}, {num_elements}")
            self._emit("SYNC.WAIT_VPU")
    
    def _emit_tanh(self, graph: Graph, node: Node, entry: ScheduleEntry):
        """Tanh"""
        input_name = node.inputs[0]
        input_tensor = graph.get_tensor(input_name)
        
        if input_tensor:
            num_elements = input_tensor.numel
            input_addr = entry.input_addrs.get(input_name, self.config.sram_act_a)
            output_addr = entry.output_addr or input_addr
            
            self._emit(f"VECTOR.TANH 0x{output_addr:04X}, 0x{input_addr:04X}, {num_elements}")
            self._emit("SYNC.WAIT_VPU")
    
    def _emit_softmax(self, graph: Graph, node: Node, entry: ScheduleEntry):
        """Softmax: 3-pass algorithm"""
        input_name = node.inputs[0]
        input_tensor = graph.get_tensor(input_name)
        
        if input_tensor:
            axis = node.get_attr('axis', -1)
            if axis < 0:
                axis = len(input_tensor.shape) + axis
            
            num_elements = input_tensor.numel
            axis_size = input_tensor.shape[axis] if axis < len(input_tensor.shape) else num_elements
            num_vectors = num_elements // axis_size
            
            input_addr = entry.input_addrs.get(input_name, self.config.sram_act_a)
            output_addr = entry.output_addr or input_addr
            scratch_addr = self.config.sram_scratch
            
            self._emit_comment(f"Softmax: axis={axis}, size={axis_size}")
            self._emit(f"VECTOR.SOFTMAX_P1 0x{scratch_addr:04X}, 0x{input_addr:04X}, {axis_size}, {num_vectors}")
            self._emit("SYNC.WAIT_VPU")
            self._emit(f"VECTOR.SOFTMAX_P2 0x{output_addr:04X}, 0x{input_addr:04X}, 0x{scratch_addr:04X}, {axis_size}, {num_vectors}")
            self._emit("SYNC.WAIT_VPU")
            self._emit(f"VECTOR.SOFTMAX_P3 0x{output_addr:04X}, 0x{output_addr:04X}, 0x{scratch_addr:04X}, {axis_size}, {num_vectors}")
            self._emit("SYNC.WAIT_VPU")
    
    # =========================================================================
    # Normalization operations
    # =========================================================================
    
    def _emit_batchnorm(self, graph: Graph, node: Node, entry: ScheduleEntry):
        """BatchNorm (inference): y = x * scale + bias (precomputed)"""
        input_name = node.inputs[0]
        input_tensor = graph.get_tensor(input_name)
        
        if not input_tensor:
            self._emit_comment("ERROR: Missing input for BatchNorm")
            return
        
        if len(node.inputs) < 3:
            self._emit_comment("ERROR: BatchNorm missing scale/bias")
            return
        
        scale_name = node.inputs[1]
        bias_name = node.inputs[2]
        
        if len(input_tensor.shape) == 4:
            N, C, H, W = input_tensor.shape
            spatial_size = H * W
        elif len(input_tensor.shape) == 2:
            N, C = input_tensor.shape
            spatial_size = 1
        else:
            C = input_tensor.shape[-1]
            spatial_size = input_tensor.numel // C
        
        input_addr = entry.input_addrs.get(input_name, self.config.sram_act_a)
        scale_addr = entry.input_addrs.get(scale_name, self.config.sram_weight_a)
        bias_addr = entry.input_addrs.get(bias_name, self.config.sram_weight_a + C * 4)
        output_addr = entry.output_addr or input_addr
        
        self._emit_comment(f"BatchNorm: {C} channels, spatial={spatial_size}")
        self._emit(f"VECTOR.BATCHNORM 0x{output_addr:04X}, 0x{input_addr:04X}, 0x{scale_addr:04X}, 0x{bias_addr:04X}, {C}, {spatial_size}")
        self._emit("SYNC.WAIT_VPU")
    
    def _emit_layernorm(self, graph: Graph, node: Node, entry: ScheduleEntry):
        """LayerNorm (compute mean/var at runtime)"""
        input_name = node.inputs[0]
        input_tensor = graph.get_tensor(input_name)
        
        if not input_tensor:
            self._emit_comment("ERROR: Missing input for LayerNorm")
            return
        
        normalized_shape = node.get_attr('normalized_shape', [input_tensor.shape[-1]])
        normalized_size = 1
        for s in normalized_shape:
            normalized_size *= s
        
        num_instances = input_tensor.numel // normalized_size
        
        input_addr = entry.input_addrs.get(input_name, self.config.sram_act_a)
        output_addr = entry.output_addr or input_addr
        scratch_addr = self.config.sram_scratch
        
        has_affine = len(node.inputs) >= 3
        if has_affine:
            scale_name = node.inputs[1]
            bias_name = node.inputs[2]
            scale_addr = entry.input_addrs.get(scale_name, self.config.sram_weight_a)
            bias_addr = entry.input_addrs.get(bias_name, self.config.sram_weight_a + normalized_size * 4)
        
        self._emit_comment(f"LayerNorm: size={normalized_size}, instances={num_instances}")
        
        # 4-pass: mean, var, normalize, scale+shift
        self._emit(f"VECTOR.LAYERNORM_MEAN 0x{scratch_addr:04X}, 0x{input_addr:04X}, {normalized_size}, {num_instances}")
        self._emit("SYNC.WAIT_VPU")
        self._emit(f"VECTOR.LAYERNORM_VAR 0x{scratch_addr + 0x1000:04X}, 0x{input_addr:04X}, 0x{scratch_addr:04X}, {normalized_size}, {num_instances}")
        self._emit("SYNC.WAIT_VPU")
        self._emit(f"VECTOR.LAYERNORM_NORM 0x{output_addr:04X}, 0x{input_addr:04X}, 0x{scratch_addr:04X}, 0x{scratch_addr + 0x1000:04X}, {normalized_size}, {num_instances}")
        self._emit("SYNC.WAIT_VPU")
        
        if has_affine:
            self._emit(f"VECTOR.SCALE_SHIFT 0x{output_addr:04X}, 0x{output_addr:04X}, 0x{scale_addr:04X}, 0x{bias_addr:04X}, {normalized_size}, {num_instances}")
            self._emit("SYNC.WAIT_VPU")
    
    # =========================================================================
    # Pooling operations
    # =========================================================================
    
    def _emit_maxpool(self, graph: Graph, node: Node, entry: ScheduleEntry):
        """MaxPool2D"""
        input_name = node.inputs[0]
        input_tensor = graph.get_tensor(input_name)
        
        if not input_tensor:
            self._emit_comment("ERROR: Missing input for MaxPool")
            return
        
        kernel_shape = node.get_attr('kernel_shape', [2, 2])
        strides = node.get_attr('strides', kernel_shape)
        pads = node.get_attr('pads', [0, 0, 0, 0])
        
        if len(input_tensor.shape) == 4:
            N, C, H, W = input_tensor.shape
        else:
            C, H, W = input_tensor.shape
            N = 1
        
        kH, kW = kernel_shape
        sH, sW = strides
        
        H_out = (H + pads[0] + pads[2] - kH) // sH + 1
        W_out = (W + pads[1] + pads[3] - kW) // sW + 1
        
        input_addr = entry.input_addrs.get(input_name, self.config.sram_act_a)
        output_addr = entry.output_addr or self.config.sram_output
        
        self._emit_comment(f"MaxPool: {kH}x{kW}, stride={sH}x{sW}")
        self._emit_comment(f"  [{N},{C},{H},{W}] -> [{N},{C},{H_out},{W_out}]")
        self._emit(f"TENSOR.MAXPOOL 0x{output_addr:04X}, 0x{input_addr:04X}, {C}, {H}, {W}, {kH}, {kW}, {sH}, {sW}")
        self._emit("SYNC.WAIT_MXU")
    
    def _emit_avgpool(self, graph: Graph, node: Node, entry: ScheduleEntry):
        """AvgPool2D"""
        input_name = node.inputs[0]
        input_tensor = graph.get_tensor(input_name)
        
        if not input_tensor:
            self._emit_comment("ERROR: Missing input for AvgPool")
            return
        
        kernel_shape = node.get_attr('kernel_shape', [2, 2])
        strides = node.get_attr('strides', kernel_shape)
        pads = node.get_attr('pads', [0, 0, 0, 0])
        
        if len(input_tensor.shape) == 4:
            N, C, H, W = input_tensor.shape
        else:
            C, H, W = input_tensor.shape
            N = 1
        
        kH, kW = kernel_shape
        sH, sW = strides
        
        H_out = (H + pads[0] + pads[2] - kH) // sH + 1
        W_out = (W + pads[1] + pads[3] - kW) // sW + 1
        
        input_addr = entry.input_addrs.get(input_name, self.config.sram_act_a)
        output_addr = entry.output_addr or self.config.sram_output
        
        self._emit_comment(f"AvgPool: {kH}x{kW}, stride={sH}x{sW}")
        self._emit_comment(f"  [{N},{C},{H},{W}] -> [{N},{C},{H_out},{W_out}]")
        self._emit(f"TENSOR.AVGPOOL 0x{output_addr:04X}, 0x{input_addr:04X}, {C}, {H}, {W}, {kH}, {kW}, {sH}, {sW}")
        self._emit("SYNC.WAIT_MXU")
    
    def _emit_globalavgpool(self, graph: Graph, node: Node, entry: ScheduleEntry):
        """GlobalAvgPool: [N,C,H,W] -> [N,C,1,1]"""
        input_name = node.inputs[0]
        input_tensor = graph.get_tensor(input_name)
        
        if not input_tensor:
            self._emit_comment("ERROR: Missing input for GlobalAvgPool")
            return
        
        if len(input_tensor.shape) == 4:
            N, C, H, W = input_tensor.shape
        else:
            C, H, W = input_tensor.shape
            N = 1
        
        spatial_size = H * W
        
        input_addr = entry.input_addrs.get(input_name, self.config.sram_act_a)
        output_addr = entry.output_addr or self.config.sram_output
        
        self._emit_comment(f"GlobalAvgPool: [{N},{C},{H},{W}] -> [{N},{C},1,1]")
        self._emit(f"VECTOR.GLOBAL_AVG 0x{output_addr:04X}, 0x{input_addr:04X}, {C}, {spatial_size}")
        self._emit("SYNC.WAIT_VPU")
    
    # =========================================================================
    # Elementwise operations
    # =========================================================================
    
    def _emit_add(self, graph: Graph, node: Node, entry: ScheduleEntry):
        """Element-wise add with broadcasting support"""
        input_a = node.inputs[0]
        input_b = node.inputs[1]
        
        tensor_a = graph.get_tensor(input_a)
        tensor_b = graph.get_tensor(input_b)
        
        if tensor_a:
            num_a = tensor_a.numel
            num_b = tensor_b.numel if tensor_b else num_a
            
            addr_a = entry.input_addrs.get(input_a, self.config.sram_act_a)
            addr_b = entry.input_addrs.get(input_b, self.config.sram_act_b)
            output_addr = entry.output_addr or addr_a
            
            if num_a == num_b:
                self._emit(f"VECTOR.ADD 0x{output_addr:04X}, 0x{addr_a:04X}, 0x{addr_b:04X}, {num_a}")
            else:
                self._emit_comment(f"Add with broadcast: {num_a} + {num_b}")
                self._emit(f"VECTOR.ADD_BCAST 0x{output_addr:04X}, 0x{addr_a:04X}, 0x{addr_b:04X}, {num_a}, {num_b}")
            self._emit("SYNC.WAIT_VPU")
    
    def _emit_sub(self, graph: Graph, node: Node, entry: ScheduleEntry):
        """Element-wise subtract"""
        input_a, input_b = node.inputs[0], node.inputs[1]
        tensor_a = graph.get_tensor(input_a)
        
        if tensor_a:
            num_elements = tensor_a.numel
            addr_a = entry.input_addrs.get(input_a, self.config.sram_act_a)
            addr_b = entry.input_addrs.get(input_b, self.config.sram_act_b)
            output_addr = entry.output_addr or addr_a
            
            self._emit(f"VECTOR.SUB 0x{output_addr:04X}, 0x{addr_a:04X}, 0x{addr_b:04X}, {num_elements}")
            self._emit("SYNC.WAIT_VPU")
    
    def _emit_mul(self, graph: Graph, node: Node, entry: ScheduleEntry):
        """Element-wise multiply"""
        input_a, input_b = node.inputs[0], node.inputs[1]
        tensor_a = graph.get_tensor(input_a)
        
        if tensor_a:
            num_elements = tensor_a.numel
            addr_a = entry.input_addrs.get(input_a, self.config.sram_act_a)
            addr_b = entry.input_addrs.get(input_b, self.config.sram_act_b)
            output_addr = entry.output_addr or addr_a
            
            self._emit(f"VECTOR.MUL 0x{output_addr:04X}, 0x{addr_a:04X}, 0x{addr_b:04X}, {num_elements}")
            self._emit("SYNC.WAIT_VPU")
    
    def _emit_div(self, graph: Graph, node: Node, entry: ScheduleEntry):
        """Element-wise divide"""
        input_a, input_b = node.inputs[0], node.inputs[1]
        tensor_a = graph.get_tensor(input_a)
        
        if tensor_a:
            num_elements = tensor_a.numel
            addr_a = entry.input_addrs.get(input_a, self.config.sram_act_a)
            addr_b = entry.input_addrs.get(input_b, self.config.sram_act_b)
            output_addr = entry.output_addr or addr_a
            
            self._emit(f"VECTOR.DIV 0x{output_addr:04X}, 0x{addr_a:04X}, 0x{addr_b:04X}, {num_elements}")
            self._emit("SYNC.WAIT_VPU")
    
    # =========================================================================
    # Shape operations
    # =========================================================================
    
    def _emit_reshape(self, graph: Graph, node: Node, entry: ScheduleEntry):
        """Reshape (no-op at runtime)"""
        input_tensor = graph.get_tensor(node.inputs[0])
        output_tensor = graph.get_tensor(node.outputs[0])
        
        old_shape = input_tensor.shape if input_tensor else "?"
        new_shape = output_tensor.shape if output_tensor else "?"
        
        self._emit_comment(f"Reshape: {old_shape} -> {new_shape} (no-op)")
        self._emit("NOP")
    
    def _emit_flatten(self, graph: Graph, node: Node, entry: ScheduleEntry):
        """Flatten (no-op at runtime)"""
        axis = node.get_attr('axis', 1)
        self._emit_comment(f"Flatten (axis={axis}) (no-op)")
        self._emit("NOP")
    
    def _emit_transpose(self, graph: Graph, node: Node, entry: ScheduleEntry):
        """Transpose (requires data movement)"""
        input_name = node.inputs[0]
        input_tensor = graph.get_tensor(input_name)
        
        if not input_tensor:
            self._emit_comment("ERROR: Missing input for Transpose")
            return
        
        perm = node.get_attr('perm', list(range(len(input_tensor.shape) - 1, -1, -1)))
        
        input_addr = entry.input_addrs.get(input_name, self.config.sram_act_a)
        output_addr = entry.output_addr or self.config.sram_output
        
        if len(input_tensor.shape) == 2 and perm == [1, 0]:
            M, N = input_tensor.shape
            self._emit_comment(f"Transpose 2D: [{M},{N}] -> [{N},{M}]")
            self._emit(f"TENSOR.TRANSPOSE 0x{output_addr:04X}, 0x{input_addr:04X}, {M}, {N}")
        else:
            shape_str = ",".join(str(s) for s in input_tensor.shape)
            perm_str = ",".join(str(p) for p in perm)
            self._emit_comment(f"Transpose: perm={perm}")
            self._emit(f"TENSOR.TRANSPOSE_ND 0x{output_addr:04X}, 0x{input_addr:04X}, {shape_str}, {perm_str}")
        self._emit("SYNC.WAIT_MXU")


def generate_code(graph: Graph, schedule: List[ScheduleEntry],
                  config: Optional[CodeGenConfig] = None) -> Tuple[str, bytes]:
    """Convenience function to generate code"""
    codegen = CodeGenerator(config)
    asm_code = codegen.generate(graph, schedule)
    weight_data = codegen.generate_weights(graph)
    return asm_code, weight_data


__all__ = ['CodeGenerator', 'CodeGenConfig', 'generate_code']

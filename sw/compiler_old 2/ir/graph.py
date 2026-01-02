"""
Intermediate Representation (IR) for Tensor Accelerator Compiler

Defines the graph representation used throughout the compilation pipeline.
Supports ONNX-style operations with extensions for quantization and tiling.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set
from enum import Enum, auto
import numpy as np


class DataType(Enum):
    """Supported data types"""
    FLOAT32 = auto()
    FLOAT16 = auto()
    INT32 = auto()
    INT8 = auto()
    UINT8 = auto()
    
    @property
    def bits(self) -> int:
        return {
            DataType.FLOAT32: 32,
            DataType.FLOAT16: 16,
            DataType.INT32: 32,
            DataType.INT8: 8,
            DataType.UINT8: 8,
        }[self]
    
    @property
    def bytes(self) -> int:
        return self.bits // 8


class OpType(Enum):
    """Supported operation types"""
    # Compute ops
    CONV2D = auto()
    MATMUL = auto()
    GEMM = auto()
    
    # Elementwise ops
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    
    # Activation ops
    RELU = auto()
    GELU = auto()
    SIGMOID = auto()
    TANH = auto()
    SOFTMAX = auto()
    
    # Normalization
    BATCHNORM = auto()
    LAYERNORM = auto()
    
    # Pooling
    MAXPOOL = auto()
    AVGPOOL = auto()
    GLOBALAVGPOOL = auto()
    
    # Shape ops
    RESHAPE = auto()
    TRANSPOSE = auto()
    FLATTEN = auto()
    CONCAT = auto()
    SPLIT = auto()
    
    # Quantization ops
    QUANTIZE = auto()
    DEQUANTIZE = auto()
    REQUANTIZE = auto()
    
    # Memory ops
    CONSTANT = auto()
    INPUT = auto()
    OUTPUT = auto()


@dataclass
class QuantInfo:
    """Quantization parameters for a tensor"""
    scale: float = 1.0
    zero_point: int = 0
    dtype: DataType = DataType.INT8
    
    # Per-channel quantization
    per_channel: bool = False
    channel_axis: int = 0
    scales: Optional[np.ndarray] = None
    zero_points: Optional[np.ndarray] = None
    
    def quantize(self, data: np.ndarray) -> np.ndarray:
        """Quantize float data to int8"""
        if self.per_channel and self.scales is not None:
            # Expand scales for broadcasting
            shape = [1] * data.ndim
            shape[self.channel_axis] = -1
            scales = self.scales.reshape(shape)
            zps = self.zero_points.reshape(shape) if self.zero_points is not None else 0
            return np.clip(np.round(data / scales) + zps, -128, 127).astype(np.int8)
        else:
            return np.clip(np.round(data / self.scale) + self.zero_point, -128, 127).astype(np.int8)
    
    def dequantize(self, data: np.ndarray) -> np.ndarray:
        """Dequantize int8 data to float"""
        if self.per_channel and self.scales is not None:
            shape = [1] * data.ndim
            shape[self.channel_axis] = -1
            scales = self.scales.reshape(shape)
            zps = self.zero_points.reshape(shape) if self.zero_points is not None else 0
            return (data.astype(np.float32) - zps) * scales
        else:
            return (data.astype(np.float32) - self.zero_point) * self.scale


@dataclass
class Tensor:
    """Represents a tensor (input, output, weight, or intermediate)"""
    name: str
    shape: Tuple[int, ...]
    dtype: DataType = DataType.FLOAT32
    
    # Data (for constants/weights)
    data: Optional[np.ndarray] = None
    
    # Quantization info
    quant: Optional[QuantInfo] = None
    
    # Memory allocation (filled by scheduler)
    sram_addr: Optional[int] = None
    ddr_addr: Optional[int] = None
    
    @property
    def size_bytes(self) -> int:
        """Total size in bytes"""
        return int(np.prod(self.shape)) * self.dtype.bytes
    
    @property
    def numel(self) -> int:
        """Number of elements"""
        return int(np.prod(self.shape))
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return isinstance(other, Tensor) and self.name == other.name


@dataclass
class Node:
    """Represents an operation in the graph"""
    name: str
    op_type: OpType
    
    # Inputs and outputs (tensor names)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    
    # Operation-specific attributes
    attrs: Dict[str, Any] = field(default_factory=dict)
    
    # Scheduling info (filled by scheduler)
    tpc_id: Optional[int] = None
    schedule_order: Optional[int] = None
    
    # Tiling info (filled by tiler)
    tiles: List['TileInfo'] = field(default_factory=list)
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return isinstance(other, Node) and self.name == other.name
    
    def get_attr(self, key: str, default=None):
        """Get attribute with default"""
        return self.attrs.get(key, default)


@dataclass
class TileInfo:
    """Tiling information for a single tile of an operation"""
    tile_id: int
    
    # Input tile coordinates [start, end) for each dimension
    input_ranges: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    
    # Output tile coordinates
    output_range: List[Tuple[int, int]] = field(default_factory=list)
    
    # Memory addresses for this tile
    input_addrs: Dict[str, int] = field(default_factory=dict)
    output_addr: int = 0
    
    # Dependencies (other tile IDs that must complete first)
    depends_on: List[int] = field(default_factory=list)


@dataclass
class Graph:
    """
    Complete computation graph
    
    The graph maintains:
    - nodes: Operations to execute
    - tensors: All tensors (inputs, outputs, weights, intermediates)
    - Topological ordering for execution
    """
    name: str = "model"
    
    # Graph structure
    nodes: List[Node] = field(default_factory=list)
    tensors: Dict[str, Tensor] = field(default_factory=dict)
    
    # Graph inputs/outputs
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    
    # Metadata
    opset_version: int = 13
    
    def add_tensor(self, tensor: Tensor) -> Tensor:
        """Add a tensor to the graph"""
        self.tensors[tensor.name] = tensor
        return tensor
    
    def add_node(self, node: Node) -> Node:
        """Add a node to the graph"""
        self.nodes.append(node)
        return node
    
    def get_tensor(self, name: str) -> Optional[Tensor]:
        """Get tensor by name"""
        return self.tensors.get(name)
    
    def get_node(self, name: str) -> Optional[Node]:
        """Get node by name"""
        for node in self.nodes:
            if node.name == name:
                return node
        return None
    
    def get_node_by_output(self, tensor_name: str) -> Optional[Node]:
        """Get the node that produces a given tensor"""
        for node in self.nodes:
            if tensor_name in node.outputs:
                return node
        return None
    
    def get_consumers(self, tensor_name: str) -> List[Node]:
        """Get all nodes that consume a given tensor"""
        return [n for n in self.nodes if tensor_name in n.inputs]
    
    def topological_sort(self) -> List[Node]:
        """Return nodes in topological order"""
        # Build adjacency list
        in_degree = {n.name: 0 for n in self.nodes}
        deps = {n.name: [] for n in self.nodes}
        
        for node in self.nodes:
            for inp in node.inputs:
                producer = self.get_node_by_output(inp)
                if producer:
                    deps[node.name].append(producer.name)
                    in_degree[node.name] += 1
        
        # Kahn's algorithm
        queue = [n for n in self.nodes if in_degree[n.name] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for other in self.nodes:
                if node.name in deps[other.name]:
                    in_degree[other.name] -= 1
                    if in_degree[other.name] == 0:
                        queue.append(other)
        
        if len(result) != len(self.nodes):
            raise ValueError("Graph has cycles!")
        
        return result
    
    def validate(self) -> List[str]:
        """Validate graph consistency, return list of errors"""
        errors = []
        
        # Check all node inputs exist
        for node in self.nodes:
            for inp in node.inputs:
                if inp not in self.tensors:
                    errors.append(f"Node '{node.name}' references unknown input '{inp}'")
        
        # Check all node outputs are registered
        for node in self.nodes:
            for out in node.outputs:
                if out not in self.tensors:
                    errors.append(f"Node '{node.name}' output '{out}' not in tensors")
        
        # Check graph inputs exist
        for inp in self.inputs:
            if inp not in self.tensors:
                errors.append(f"Graph input '{inp}' not in tensors")
        
        # Check graph outputs exist
        for out in self.outputs:
            if out not in self.tensors:
                errors.append(f"Graph output '{out}' not in tensors")
        
        # Try topological sort (checks for cycles)
        try:
            self.topological_sort()
        except ValueError as e:
            errors.append(str(e))
        
        return errors
    
    def summary(self) -> str:
        """Return human-readable summary"""
        lines = [
            f"Graph: {self.name}",
            f"  Inputs: {self.inputs}",
            f"  Outputs: {self.outputs}",
            f"  Nodes: {len(self.nodes)}",
            f"  Tensors: {len(self.tensors)}",
            "",
            "Operations:",
        ]
        
        for node in self.topological_sort():
            inp_shapes = [str(self.tensors[i].shape) if i in self.tensors else "?" for i in node.inputs]
            out_shapes = [str(self.tensors[o].shape) if o in self.tensors else "?" for o in node.outputs]
            lines.append(f"  {node.name}: {node.op_type.name}")
            lines.append(f"    inputs: {node.inputs} {inp_shapes}")
            lines.append(f"    outputs: {node.outputs} {out_shapes}")
        
        return "\n".join(lines)
    
    def count_ops(self) -> Dict[OpType, int]:
        """Count operations by type"""
        counts = {}
        for node in self.nodes:
            counts[node.op_type] = counts.get(node.op_type, 0) + 1
        return counts
    
    def total_weight_bytes(self) -> int:
        """Total weight memory required"""
        total = 0
        for tensor in self.tensors.values():
            if tensor.data is not None:
                total += tensor.size_bytes
        return total
    
    def peak_activation_bytes(self) -> int:
        """Estimate peak activation memory (rough)"""
        # Simple: sum of all non-weight tensor sizes
        total = 0
        for tensor in self.tensors.values():
            if tensor.data is None:  # Not a weight
                total += tensor.size_bytes
        return total


def create_simple_test_graph() -> Graph:
    """Create a simple test graph for validation"""
    g = Graph(name="test_mlp")
    
    # Input
    g.add_tensor(Tensor("input", (1, 784), DataType.INT8))
    g.inputs.append("input")
    
    # FC1 weights and bias
    g.add_tensor(Tensor("fc1_weight", (784, 256), DataType.INT8, 
                        data=np.random.randint(-128, 127, (784, 256), dtype=np.int8)))
    g.add_tensor(Tensor("fc1_bias", (256,), DataType.INT32,
                        data=np.random.randint(-1000, 1000, (256,), dtype=np.int32)))
    
    # FC1 output
    g.add_tensor(Tensor("fc1_out", (1, 256), DataType.INT8))
    
    # FC1 node
    g.add_node(Node(
        name="fc1",
        op_type=OpType.GEMM,
        inputs=["input", "fc1_weight", "fc1_bias"],
        outputs=["fc1_out"],
        attrs={"transB": True}
    ))
    
    # ReLU
    g.add_tensor(Tensor("relu1_out", (1, 256), DataType.INT8))
    g.add_node(Node(
        name="relu1",
        op_type=OpType.RELU,
        inputs=["fc1_out"],
        outputs=["relu1_out"]
    ))
    
    # FC2
    g.add_tensor(Tensor("fc2_weight", (256, 10), DataType.INT8,
                        data=np.random.randint(-128, 127, (256, 10), dtype=np.int8)))
    g.add_tensor(Tensor("fc2_bias", (10,), DataType.INT32,
                        data=np.random.randint(-1000, 1000, (10,), dtype=np.int32)))
    g.add_tensor(Tensor("output", (1, 10), DataType.INT8))
    
    g.add_node(Node(
        name="fc2",
        op_type=OpType.GEMM,
        inputs=["relu1_out", "fc2_weight", "fc2_bias"],
        outputs=["output"],
        attrs={"transB": True}
    ))
    
    g.outputs.append("output")
    
    return g


if __name__ == "__main__":
    # Test the IR
    g = create_simple_test_graph()
    print(g.summary())
    print()
    errors = g.validate()
    if errors:
        print("Validation errors:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("Graph is valid!")
    print()
    print(f"Op counts: {g.count_ops()}")
    print(f"Weight bytes: {g.total_weight_bytes():,}")
    print(f"Activation bytes: {g.peak_activation_bytes():,}")

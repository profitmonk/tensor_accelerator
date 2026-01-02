"""
ONNX Frontend Parser

Loads ONNX models and converts them to our IR format.
Supports common CNN and MLP operations.
"""

import sys
import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir.graph import Graph, Node, Tensor, OpType, DataType, QuantInfo

# Try to import onnx, provide helpful error if not available
try:
    import onnx
    from onnx import numpy_helper, TensorProto
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


# Map ONNX op types to our IR op types
ONNX_OP_MAP = {
    # Compute
    'Conv': OpType.CONV2D,
    'MatMul': OpType.MATMUL,
    'Gemm': OpType.GEMM,
    
    # Elementwise
    'Add': OpType.ADD,
    'Sub': OpType.SUB,
    'Mul': OpType.MUL,
    'Div': OpType.DIV,
    
    # Activations
    'Relu': OpType.RELU,
    'Gelu': OpType.GELU,
    'Sigmoid': OpType.SIGMOID,
    'Tanh': OpType.TANH,
    'Softmax': OpType.SOFTMAX,
    
    # Normalization
    'BatchNormalization': OpType.BATCHNORM,
    'LayerNormalization': OpType.LAYERNORM,
    
    # Pooling
    'MaxPool': OpType.MAXPOOL,
    'AveragePool': OpType.AVGPOOL,
    'GlobalAveragePool': OpType.GLOBALAVGPOOL,
    
    # Shape
    'Reshape': OpType.RESHAPE,
    'Transpose': OpType.TRANSPOSE,
    'Flatten': OpType.FLATTEN,
    'Concat': OpType.CONCAT,
    'Split': OpType.SPLIT,
    
    # Quantization
    'QuantizeLinear': OpType.QUANTIZE,
    'DequantizeLinear': OpType.DEQUANTIZE,
}


# Map ONNX data types to our types
ONNX_DTYPE_MAP = {
    TensorProto.FLOAT: DataType.FLOAT32,
    TensorProto.FLOAT16: DataType.FLOAT16,
    TensorProto.INT32: DataType.INT32,
    TensorProto.INT8: DataType.INT8,
    TensorProto.UINT8: DataType.UINT8,
} if ONNX_AVAILABLE else {}


class ONNXParser:
    """
    Parse ONNX models into our IR
    
    Example:
        parser = ONNXParser()
        graph = parser.load("model.onnx")
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.graph: Optional[Graph] = None
        self.initializers: Dict[str, np.ndarray] = {}
        self.value_info: Dict[str, Tuple[DataType, Tuple[int, ...]]] = {}
        
    def load(self, path: str) -> Graph:
        """Load ONNX model from file"""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not installed. Run: pip install onnx")
        
        if self.verbose:
            print(f"Loading ONNX model: {path}")
        
        model = onnx.load(path)
        return self.parse_model(model)
    
    def load_from_bytes(self, data: bytes) -> Graph:
        """Load ONNX model from bytes"""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not installed. Run: pip install onnx")
        
        model = onnx.load_model_from_string(data)
        return self.parse_model(model)
    
    def parse_model(self, model: 'onnx.ModelProto') -> Graph:
        """Parse ONNX model proto"""
        # Get model metadata
        self.graph = Graph(
            name=model.graph.name or "model",
            opset_version=model.opset_import[0].version if model.opset_import else 13
        )
        
        if self.verbose:
            print(f"  Model name: {self.graph.name}")
            print(f"  Opset version: {self.graph.opset_version}")
        
        # Parse initializers (weights)
        self._parse_initializers(model.graph)
        
        # Parse value info (tensor shapes)
        self._parse_value_info(model.graph)
        
        # Parse inputs
        self._parse_inputs(model.graph)
        
        # Parse outputs
        self._parse_outputs(model.graph)
        
        # Parse nodes
        self._parse_nodes(model.graph)
        
        # Validate
        errors = self.graph.validate()
        if errors:
            print("Warning: Graph validation errors:")
            for e in errors:
                print(f"  - {e}")
        
        return self.graph
    
    def _parse_initializers(self, graph: 'onnx.GraphProto'):
        """Parse model weights/constants"""
        for init in graph.initializer:
            name = init.name
            data = numpy_helper.to_array(init)
            self.initializers[name] = data
            
            # Create tensor
            dtype = self._convert_dtype(init.data_type)
            tensor = Tensor(
                name=name,
                shape=tuple(data.shape),
                dtype=dtype,
                data=data
            )
            self.graph.add_tensor(tensor)
            
            if self.verbose:
                print(f"  Initializer: {name} {data.shape} {dtype.name}")
    
    def _parse_value_info(self, graph: 'onnx.GraphProto'):
        """Parse tensor shape information"""
        for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
            name = vi.name
            if name in self.initializers:
                continue  # Already handled
                
            shape = self._get_shape(vi.type.tensor_type)
            dtype = self._convert_dtype(vi.type.tensor_type.elem_type)
            self.value_info[name] = (dtype, shape)
    
    def _parse_inputs(self, graph: 'onnx.GraphProto'):
        """Parse model inputs"""
        for inp in graph.input:
            name = inp.name
            if name in self.initializers:
                continue  # This is a weight, not an input
            
            shape = self._get_shape(inp.type.tensor_type)
            dtype = self._convert_dtype(inp.type.tensor_type.elem_type)
            
            tensor = Tensor(name=name, shape=shape, dtype=dtype)
            self.graph.add_tensor(tensor)
            self.graph.inputs.append(name)
            
            if self.verbose:
                print(f"  Input: {name} {shape} {dtype.name}")
    
    def _parse_outputs(self, graph: 'onnx.GraphProto'):
        """Parse model outputs"""
        for out in graph.output:
            name = out.name
            shape = self._get_shape(out.type.tensor_type)
            dtype = self._convert_dtype(out.type.tensor_type.elem_type)
            
            # Output tensor may already exist from a node
            if name not in self.graph.tensors:
                tensor = Tensor(name=name, shape=shape, dtype=dtype)
                self.graph.add_tensor(tensor)
            
            self.graph.outputs.append(name)
            
            if self.verbose:
                print(f"  Output: {name} {shape} {dtype.name}")
    
    def _parse_nodes(self, graph: 'onnx.GraphProto'):
        """Parse operation nodes"""
        for node in graph.node:
            ir_node = self._convert_node(node)
            if ir_node:
                self.graph.add_node(ir_node)
                
                # Create output tensors if they don't exist
                for out_name in ir_node.outputs:
                    if out_name not in self.graph.tensors:
                        # Infer shape from value_info or leave unknown
                        if out_name in self.value_info:
                            dtype, shape = self.value_info[out_name]
                        else:
                            dtype = DataType.FLOAT32
                            shape = self._infer_output_shape(ir_node)
                        
                        tensor = Tensor(name=out_name, shape=shape, dtype=dtype)
                        self.graph.add_tensor(tensor)
    
    def _convert_node(self, node: 'onnx.NodeProto') -> Optional[Node]:
        """Convert ONNX node to IR node"""
        op_type = ONNX_OP_MAP.get(node.op_type)
        
        if op_type is None:
            if self.verbose:
                print(f"  Warning: Unsupported op type '{node.op_type}', skipping")
            return None
        
        # Parse attributes
        attrs = {}
        for attr in node.attribute:
            attrs[attr.name] = self._parse_attribute(attr)
        
        ir_node = Node(
            name=node.name or f"{node.op_type}_{id(node)}",
            op_type=op_type,
            inputs=list(node.input),
            outputs=list(node.output),
            attrs=attrs
        )
        
        if self.verbose:
            print(f"  Node: {ir_node.name} ({node.op_type}) -> {op_type.name}")
        
        return ir_node
    
    def _parse_attribute(self, attr: 'onnx.AttributeProto') -> Any:
        """Parse ONNX attribute to Python value"""
        if attr.type == onnx.AttributeProto.FLOAT:
            return attr.f
        elif attr.type == onnx.AttributeProto.INT:
            return attr.i
        elif attr.type == onnx.AttributeProto.STRING:
            return attr.s.decode('utf-8')
        elif attr.type == onnx.AttributeProto.FLOATS:
            return list(attr.floats)
        elif attr.type == onnx.AttributeProto.INTS:
            return list(attr.ints)
        elif attr.type == onnx.AttributeProto.TENSOR:
            return numpy_helper.to_array(attr.t)
        else:
            return None
    
    def _convert_dtype(self, onnx_type: int) -> DataType:
        """Convert ONNX data type to IR data type"""
        return ONNX_DTYPE_MAP.get(onnx_type, DataType.FLOAT32)
    
    def _get_shape(self, tensor_type: 'onnx.TypeProto.Tensor') -> Tuple[int, ...]:
        """Extract shape from ONNX tensor type"""
        shape = []
        for dim in tensor_type.shape.dim:
            if dim.dim_param:
                # Dynamic dimension - use -1 or a default
                shape.append(-1)
            else:
                shape.append(dim.dim_value)
        return tuple(shape)
    
    def _infer_output_shape(self, node: Node) -> Tuple[int, ...]:
        """Infer output shape from input shapes (simple cases)"""
        if not node.inputs:
            return (1,)
        
        # Get first input shape
        first_input = node.inputs[0]
        if first_input in self.graph.tensors:
            input_shape = self.graph.tensors[first_input].shape
        else:
            return (1,)
        
        # Simple shape inference for common ops
        if node.op_type in [OpType.RELU, OpType.GELU, OpType.SIGMOID, OpType.TANH]:
            return input_shape
        
        if node.op_type == OpType.GEMM:
            # GEMM: (M, K) x (K, N) -> (M, N)
            if len(node.inputs) >= 2 and node.inputs[1] in self.graph.tensors:
                weight_shape = self.graph.tensors[node.inputs[1]].shape
                transB = node.get_attr('transB', 0)
                if transB:
                    N = weight_shape[0]
                else:
                    N = weight_shape[1]
                return (input_shape[0], N)
        
        return input_shape


def load_onnx(path: str, verbose: bool = False) -> Graph:
    """Convenience function to load ONNX model"""
    parser = ONNXParser(verbose=verbose)
    return parser.load(path)


# Export from module
__all__ = ['ONNXParser', 'load_onnx', 'ONNX_AVAILABLE']


if __name__ == "__main__":
    # Test with a simple model if available
    import sys
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
        try:
            graph = load_onnx(path, verbose=True)
            print()
            print(graph.summary())
        except Exception as e:
            print(f"Error loading {path}: {e}")
    else:
        print("ONNX Parser")
        print(f"  ONNX available: {ONNX_AVAILABLE}")
        if not ONNX_AVAILABLE:
            print("  Install with: pip install onnx")
        print()
        print("Usage: python onnx_parser.py model.onnx")

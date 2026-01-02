"""
Quantizer Module

Converts floating-point models to INT8 for efficient hardware execution.
Supports both static (calibration-based) and dynamic quantization.
"""

import sys
import os
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir.graph import Graph, Node, Tensor, OpType, DataType, QuantInfo


class Quantizer:
    """
    Quantizes a floating-point graph to INT8
    
    Supports:
    - Per-tensor and per-channel quantization
    - Symmetric and asymmetric quantization
    - Static (with calibration) and weight-only quantization
    
    Example:
        quantizer = Quantizer(method='symmetric')
        quantized_graph = quantizer.quantize(fp32_graph, calibration_data)
    """
    
    def __init__(self, 
                 method: str = 'symmetric',
                 per_channel_weights: bool = True,
                 bits: int = 8):
        """
        Args:
            method: 'symmetric' or 'asymmetric'
            per_channel_weights: Use per-channel for weights
            bits: Quantization bits (8 for INT8)
        """
        self.method = method
        self.per_channel_weights = per_channel_weights
        self.bits = bits
        
        # Quantization range
        if method == 'symmetric':
            self.qmin = -(1 << (bits - 1))
            self.qmax = (1 << (bits - 1)) - 1
        else:
            self.qmin = 0
            self.qmax = (1 << bits) - 1
        
        # Collected statistics for calibration
        self.activation_stats: Dict[str, Dict] = {}
        
    def quantize(self, graph: Graph, 
                 calibration_data: Optional[Dict[str, np.ndarray]] = None) -> Graph:
        """
        Quantize the entire graph
        
        Args:
            graph: Input FP32 graph
            calibration_data: Dict of tensor_name -> representative data samples
                             for activation quantization
        
        Returns:
            Quantized INT8 graph
        """
        # Create copy of graph
        q_graph = Graph(
            name=graph.name + "_int8",
            opset_version=graph.opset_version
        )
        q_graph.inputs = graph.inputs.copy()
        q_graph.outputs = graph.outputs.copy()
        
        # First pass: quantize weights
        for name, tensor in graph.tensors.items():
            q_tensor = self._quantize_tensor(tensor, is_weight=(tensor.data is not None))
            q_graph.add_tensor(q_tensor)
        
        # Second pass: determine activation scales if calibration data provided
        if calibration_data:
            self._calibrate(graph, calibration_data)
            
            # Apply calibrated scales
            for name, stats in self.activation_stats.items():
                if name in q_graph.tensors:
                    tensor = q_graph.tensors[name]
                    tensor.quant = self._compute_quant_params(
                        stats['min'], stats['max'], is_weight=False
                    )
                    tensor.dtype = DataType.INT8
        
        # Copy nodes (they operate on quantized tensors now)
        for node in graph.nodes:
            q_node = Node(
                name=node.name,
                op_type=node.op_type,
                inputs=node.inputs.copy(),
                outputs=node.outputs.copy(),
                attrs=node.attrs.copy()
            )
            
            # Add requantization info for accumulator outputs
            if node.op_type in [OpType.GEMM, OpType.CONV2D, OpType.MATMUL]:
                # GEMM produces INT32 accumulator, needs requant to INT8
                q_node.attrs['output_quant'] = self._get_output_quant(
                    graph, node, q_graph
                )
            
            q_graph.add_node(q_node)
        
        return q_graph
    
    def quantize_weights_only(self, graph: Graph) -> Graph:
        """
        Quantize only weights, leave activations in FP32
        (Simpler, no calibration needed)
        """
        q_graph = Graph(
            name=graph.name + "_w8a32",
            opset_version=graph.opset_version
        )
        q_graph.inputs = graph.inputs.copy()
        q_graph.outputs = graph.outputs.copy()
        
        # Quantize weights, keep activations as-is
        for name, tensor in graph.tensors.items():
            if tensor.data is not None:
                # This is a weight - quantize it
                q_tensor = self._quantize_tensor(tensor, is_weight=True)
            else:
                # This is an activation - copy as-is
                q_tensor = Tensor(
                    name=tensor.name,
                    shape=tensor.shape,
                    dtype=tensor.dtype,
                    data=tensor.data,
                    quant=tensor.quant
                )
            q_graph.add_tensor(q_tensor)
        
        # Copy nodes
        for node in graph.nodes:
            q_node = Node(
                name=node.name,
                op_type=node.op_type,
                inputs=node.inputs.copy(),
                outputs=node.outputs.copy(),
                attrs=node.attrs.copy()
            )
            q_graph.add_node(q_node)
        
        return q_graph
    
    def _quantize_tensor(self, tensor: Tensor, is_weight: bool) -> Tensor:
        """Quantize a single tensor"""
        if tensor.data is None:
            # No data - just create tensor with quant placeholder
            return Tensor(
                name=tensor.name,
                shape=tensor.shape,
                dtype=DataType.INT8,
                quant=QuantInfo(scale=1.0, zero_point=0, dtype=DataType.INT8)
            )
        
        # Get data
        data = tensor.data.astype(np.float32)
        
        # Compute quantization params
        if is_weight and self.per_channel_weights and len(data.shape) >= 2:
            # Per-channel for weights (usually channel is axis 0)
            quant = self._compute_per_channel_quant(data, axis=0)
        else:
            # Per-tensor
            quant = self._compute_quant_params(data.min(), data.max(), is_weight)
        
        # Quantize the data
        q_data = quant.quantize(data)
        
        return Tensor(
            name=tensor.name,
            shape=tensor.shape,
            dtype=DataType.INT8,
            data=q_data,
            quant=quant
        )
    
    def _compute_quant_params(self, min_val: float, max_val: float, 
                               is_weight: bool) -> QuantInfo:
        """Compute scale and zero-point for a tensor"""
        if self.method == 'symmetric':
            # Symmetric: zero_point = 0
            abs_max = max(abs(min_val), abs(max_val))
            if abs_max == 0:
                abs_max = 1.0
            scale = abs_max / self.qmax
            return QuantInfo(scale=scale, zero_point=0, dtype=DataType.INT8)
        else:
            # Asymmetric: optimize range usage
            if max_val == min_val:
                scale = 1.0
                zero_point = 0
            else:
                scale = (max_val - min_val) / (self.qmax - self.qmin)
                zero_point = int(round(self.qmin - min_val / scale))
                zero_point = max(self.qmin, min(self.qmax, zero_point))
            return QuantInfo(scale=scale, zero_point=zero_point, dtype=DataType.INT8)
    
    def _compute_per_channel_quant(self, data: np.ndarray, axis: int) -> QuantInfo:
        """Compute per-channel quantization parameters"""
        # Move channel axis to front
        data = np.moveaxis(data, axis, 0)
        num_channels = data.shape[0]
        
        scales = np.zeros(num_channels, dtype=np.float32)
        zero_points = np.zeros(num_channels, dtype=np.int32)
        
        for c in range(num_channels):
            channel_data = data[c].flatten()
            if self.method == 'symmetric':
                abs_max = max(abs(channel_data.min()), abs(channel_data.max()))
                if abs_max == 0:
                    abs_max = 1.0
                scales[c] = abs_max / self.qmax
                zero_points[c] = 0
            else:
                min_val, max_val = channel_data.min(), channel_data.max()
                if max_val == min_val:
                    scales[c] = 1.0
                    zero_points[c] = 0
                else:
                    scales[c] = (max_val - min_val) / (self.qmax - self.qmin)
                    zero_points[c] = int(round(self.qmin - min_val / scales[c]))
        
        return QuantInfo(
            scale=float(scales.mean()),  # Store average for compatibility
            zero_point=int(zero_points.mean()),
            dtype=DataType.INT8,
            per_channel=True,
            channel_axis=axis,
            scales=scales,
            zero_points=zero_points
        )
    
    def _calibrate(self, graph: Graph, calibration_data: Dict[str, np.ndarray]):
        """Run calibration to collect activation statistics"""
        # Initialize stats collection
        for tensor_name in graph.tensors:
            if graph.tensors[tensor_name].data is None:  # Activations only
                self.activation_stats[tensor_name] = {
                    'min': float('inf'),
                    'max': float('-inf'),
                    'count': 0
                }
        
        # Run inference to collect stats
        # Note: This is a simplified version - real calibration would run
        # the actual model on representative data
        
        # For now, just use provided data directly
        for tensor_name, data in calibration_data.items():
            if tensor_name in self.activation_stats:
                self.activation_stats[tensor_name]['min'] = min(
                    self.activation_stats[tensor_name]['min'],
                    float(data.min())
                )
                self.activation_stats[tensor_name]['max'] = max(
                    self.activation_stats[tensor_name]['max'],
                    float(data.max())
                )
                self.activation_stats[tensor_name]['count'] += 1
        
        # For uncalibrated tensors, use conservative defaults
        for tensor_name, stats in self.activation_stats.items():
            if stats['count'] == 0:
                # Assume typical activation range for ReLU outputs
                stats['min'] = 0.0
                stats['max'] = 6.0  # Common for ReLU6
    
    def _get_output_quant(self, graph: Graph, node: Node, 
                          q_graph: Graph) -> QuantInfo:
        """
        Compute output quantization for a node
        
        For GEMM: output_scale = input_scale * weight_scale
        """
        if node.op_type == OpType.GEMM:
            # Get input scales
            input_name = node.inputs[0]
            weight_name = node.inputs[1]
            
            input_scale = 1.0
            weight_scale = 1.0
            
            if input_name in q_graph.tensors and q_graph.tensors[input_name].quant:
                input_scale = q_graph.tensors[input_name].quant.scale
            if weight_name in q_graph.tensors and q_graph.tensors[weight_name].quant:
                weight_scale = q_graph.tensors[weight_name].quant.scale
            
            # Output scale = input_scale * weight_scale
            output_scale = input_scale * weight_scale
            
            # For output after requant, we need a scale that brings it to ~[-128, 127]
            # This depends on expected output range
            if node.outputs[0] in self.activation_stats:
                stats = self.activation_stats[node.outputs[0]]
                output_quant = self._compute_quant_params(
                    stats['min'], stats['max'], is_weight=False
                )
            else:
                output_quant = QuantInfo(scale=output_scale, zero_point=0)
            
            output_quant.dtype = DataType.INT8
            return output_quant
        
        return QuantInfo(scale=1.0, zero_point=0, dtype=DataType.INT8)


def quantize_graph(graph: Graph, 
                   calibration_data: Optional[Dict[str, np.ndarray]] = None,
                   method: str = 'symmetric') -> Graph:
    """Convenience function to quantize a graph"""
    quantizer = Quantizer(method=method)
    if calibration_data:
        return quantizer.quantize(graph, calibration_data)
    else:
        return quantizer.quantize_weights_only(graph)


# Export from module
__all__ = ['Quantizer', 'quantize_graph']


if __name__ == "__main__":
    # Test quantization
    from ir.graph import create_simple_test_graph
    
    print("Testing Quantizer")
    print("=" * 60)
    
    # Create test graph
    graph = create_simple_test_graph()
    print("\nOriginal graph:")
    print(graph.summary())
    
    # Quantize weights only
    quantizer = Quantizer(method='symmetric')
    q_graph = quantizer.quantize_weights_only(graph)
    
    print("\n\nQuantized graph (weights only):")
    print(q_graph.summary())
    
    # Check a weight
    fc1_weight = q_graph.get_tensor("fc1_weight")
    if fc1_weight:
        print(f"\nfc1_weight:")
        print(f"  dtype: {fc1_weight.dtype}")
        print(f"  shape: {fc1_weight.shape}")
        print(f"  scale: {fc1_weight.quant.scale if fc1_weight.quant else 'N/A'}")
        print(f"  data range: [{fc1_weight.data.min()}, {fc1_weight.data.max()}]")

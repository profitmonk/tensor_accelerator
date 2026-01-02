"""
Tiling Engine

Breaks large tensor operations into tiles that fit in the accelerator's SRAM.
This is critical for efficient hardware utilization.

Hardware constraints:
- Systolic array: 8x8 (can process 8x8 matrix multiply per cycle)
- SRAM per TPC: 2MB (shared for activations, weights, outputs)
- Data width: INT8 (1 byte per element)
"""

import sys
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir.graph import Graph, Node, Tensor, OpType, DataType, TileInfo


@dataclass
class HardwareConfig:
    """Hardware configuration for tiling decisions"""
    # Systolic array dimensions
    systolic_m: int = 8
    systolic_n: int = 8
    systolic_k: int = 8
    
    # SRAM sizes per TPC (bytes)
    sram_size: int = 2 * 1024 * 1024  # 2MB
    
    # Partitioning: how much for each purpose
    # Increased for larger models like ResNet
    sram_act_a: int = 512 * 1024   # Input A buffer (512KB)
    sram_act_b: int = 256 * 1024   # Input B buffer (256KB)
    sram_weight: int = 768 * 1024  # Weight buffer (768KB)
    sram_output: int = 256 * 1024  # Output buffer (256KB)
    sram_scratch: int = 256 * 1024  # Scratch space (256KB)
    
    # Number of TPCs
    num_tpcs: int = 4
    
    # DMA constraints
    max_dma_transfer: int = 4096  # Max bytes per DMA transfer
    
    def max_tile_elements(self, buffer_name: str) -> int:
        """Max elements that fit in a buffer (assuming INT8)"""
        sizes = {
            'act_a': self.sram_act_a,
            'act_b': self.sram_act_b,
            'weight': self.sram_weight,
            'output': self.sram_output,
            'scratch': self.sram_scratch,
        }
        return sizes.get(buffer_name, self.sram_size)


@dataclass
class GEMMTileConfig:
    """Configuration for a GEMM tile"""
    # Tile dimensions
    tile_m: int  # Output rows per tile
    tile_n: int  # Output cols per tile
    tile_k: int  # Reduction dimension per tile
    
    # Number of tiles
    num_m_tiles: int
    num_n_tiles: int
    num_k_tiles: int
    
    # Memory footprint
    input_a_bytes: int
    input_b_bytes: int
    output_bytes: int
    total_bytes: int
    
    def __str__(self):
        return (f"GEMMTile(M={self.tile_m}, N={self.tile_n}, K={self.tile_k}, "
                f"tiles={self.num_m_tiles}x{self.num_n_tiles}x{self.num_k_tiles}, "
                f"mem={self.total_bytes//1024}KB)")


class TilingEngine:
    """
    Tiles operations to fit hardware constraints
    
    Key responsibilities:
    1. Determine optimal tile sizes for each operation
    2. Generate tile schedule (order of execution)
    3. Compute memory addresses for each tile
    """
    
    def __init__(self, hw_config: Optional[HardwareConfig] = None):
        self.hw = hw_config or HardwareConfig()
        
    def tile_graph(self, graph: Graph) -> Graph:
        """
        Tile all operations in the graph
        
        Returns: Graph with TileInfo filled in for each node
        """
        for node in graph.nodes:
            if node.op_type == OpType.GEMM:
                self._tile_gemm(graph, node)
            elif node.op_type == OpType.CONV2D:
                self._tile_conv2d(graph, node)
            elif node.op_type == OpType.MATMUL:
                self._tile_matmul(graph, node)
            # Elementwise ops typically don't need tiling
            # (they process one element at a time)
        
        return graph
    
    def compute_gemm_tiling(self, M: int, N: int, K: int) -> GEMMTileConfig:
        """
        Compute optimal tiling for GEMM: C[M,N] = A[M,K] @ B[K,N]
        
        Strategy:
        1. Tile K to fit weights in SRAM
        2. Tile M and N to fit activations and outputs
        3. Align tiles to systolic array size (8x8)
        4. Verify total fits in SRAM
        
        Memory layout:
        - A tile: M × K bytes (INT8)
        - B tile: K × N bytes (INT8)
        - C tile: M × N × 4 bytes (INT32 accumulator)
        """
        # Start with systolic array aligned sizes
        tile_m = self.hw.systolic_m
        tile_n = self.hw.systolic_n
        tile_k = self.hw.systolic_k
        
        def compute_total_bytes(m, n, k):
            """Compute total SRAM needed for a tile"""
            a_bytes = m * k           # INT8
            b_bytes = k * n           # INT8
            c_bytes = m * n * 4       # INT32 accumulator
            return a_bytes + b_bytes + c_bytes
        
        # Maximum total bytes we can use
        max_total = self.hw.sram_size // 2  # Leave headroom for double-buffering
        
        # Grow tiles while they fit in memory
        # Priority: maximize K first (minimize accumulation passes)
        
        # Try to grow K
        while tile_k * 2 <= K:
            new_k = tile_k * 2
            if compute_total_bytes(tile_m, tile_n, new_k) <= max_total:
                tile_k = new_k
            else:
                break
        
        # Then try to grow M
        while tile_m * 2 <= M:
            new_m = tile_m * 2
            if compute_total_bytes(new_m, tile_n, tile_k) <= max_total:
                tile_m = new_m
            else:
                break
        
        # Then try to grow N
        while tile_n * 2 <= N:
            new_n = tile_n * 2
            if compute_total_bytes(tile_m, new_n, tile_k) <= max_total:
                tile_n = new_n
            else:
                break
        
        # Ensure we cover the full dimensions (for small matrices)
        tile_m = min(tile_m, M)
        tile_n = min(tile_n, N)
        tile_k = min(tile_k, K)
        
        # Align to systolic array (but don't exceed original dimension)
        if tile_m > self.hw.systolic_m:
            tile_m = (tile_m // self.hw.systolic_m) * self.hw.systolic_m
        if tile_n > self.hw.systolic_n:
            tile_n = (tile_n // self.hw.systolic_n) * self.hw.systolic_n
        
        # Final sanity check - shrink if still too large
        while compute_total_bytes(tile_m, tile_n, tile_k) > max_total:
            # Shrink largest dimension first
            if tile_k > self.hw.systolic_k and tile_k >= tile_m and tile_k >= tile_n:
                tile_k = tile_k // 2
            elif tile_n > self.hw.systolic_n and tile_n >= tile_m:
                tile_n = tile_n // 2
            elif tile_m > self.hw.systolic_m:
                tile_m = tile_m // 2
            else:
                break  # Can't shrink further
        
        # Compute number of tiles
        num_m_tiles = math.ceil(M / tile_m)
        num_n_tiles = math.ceil(N / tile_n)
        num_k_tiles = math.ceil(K / tile_k)
        
        # Memory footprint
        input_a_bytes = tile_m * tile_k
        input_b_bytes = tile_k * tile_n
        output_bytes = tile_m * tile_n * 4  # INT32 accumulator
        
        return GEMMTileConfig(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            num_m_tiles=num_m_tiles,
            num_n_tiles=num_n_tiles,
            num_k_tiles=num_k_tiles,
            input_a_bytes=input_a_bytes,
            input_b_bytes=input_b_bytes,
            output_bytes=output_bytes,
            total_bytes=input_a_bytes + input_b_bytes + output_bytes
        )
    
    def _tile_gemm(self, graph: Graph, node: Node):
        """Generate tiles for GEMM operation"""
        # Get dimensions
        # GEMM: Y = alpha * A @ B + beta * C
        # A: [M, K], B: [K, N] or [N, K] if transB
        
        input_a_name = node.inputs[0]
        input_b_name = node.inputs[1]
        
        a_tensor = graph.get_tensor(input_a_name)
        b_tensor = graph.get_tensor(input_b_name)
        
        if not a_tensor or not b_tensor:
            return
        
        # Handle transpose
        transB = node.get_attr('transB', 0)
        
        M = a_tensor.shape[-2] if len(a_tensor.shape) >= 2 else a_tensor.shape[0]
        K = a_tensor.shape[-1]
        
        if transB:
            N = b_tensor.shape[0]
        else:
            N = b_tensor.shape[-1]
        
        # Compute tiling
        tile_config = self.compute_gemm_tiling(M, N, K)
        
        # Store tile config in node attributes
        node.attrs['tile_config'] = tile_config
        
        # Generate individual tiles
        tile_id = 0
        node.tiles = []
        
        for m_idx in range(tile_config.num_m_tiles):
            for n_idx in range(tile_config.num_n_tiles):
                for k_idx in range(tile_config.num_k_tiles):
                    # Compute ranges
                    m_start = m_idx * tile_config.tile_m
                    m_end = min((m_idx + 1) * tile_config.tile_m, M)
                    
                    n_start = n_idx * tile_config.tile_n
                    n_end = min((n_idx + 1) * tile_config.tile_n, N)
                    
                    k_start = k_idx * tile_config.tile_k
                    k_end = min((k_idx + 1) * tile_config.tile_k, K)
                    
                    tile = TileInfo(
                        tile_id=tile_id,
                        input_ranges={
                            input_a_name: [(m_start, m_end), (k_start, k_end)],
                            input_b_name: [(k_start, k_end), (n_start, n_end)] if not transB 
                                         else [(n_start, n_end), (k_start, k_end)],
                        },
                        output_range=[(m_start, m_end), (n_start, n_end)],
                    )
                    
                    # Dependencies: K accumulation requires previous K tile
                    if k_idx > 0:
                        prev_tile_id = tile_id - 1
                        tile.depends_on.append(prev_tile_id)
                    
                    node.tiles.append(tile)
                    tile_id += 1
    
    def _tile_conv2d(self, graph: Graph, node: Node):
        """Generate tiles for Conv2D operation (via im2col + GEMM)"""
        # Conv2D: Y[N,Co,Ho,Wo] = Conv(X[N,Ci,Hi,Wi], W[Co,Ci,Kh,Kw])
        # We convert to GEMM:
        #   im2col(X) -> [N*Ho*Wo, Ci*Kh*Kw]
        #   W reshaped -> [Ci*Kh*Kw, Co]
        #   Y reshaped -> [N*Ho*Wo, Co]
        
        input_name = node.inputs[0]
        weight_name = node.inputs[1]
        
        x_tensor = graph.get_tensor(input_name)
        w_tensor = graph.get_tensor(weight_name)
        
        if not x_tensor or not w_tensor:
            return
        
        # Get conv parameters
        kernel_shape = node.get_attr('kernel_shape', [3, 3])
        strides = node.get_attr('strides', [1, 1])
        pads = node.get_attr('pads', [0, 0, 0, 0])
        
        # Input shape: [N, Ci, Hi, Wi]
        N = x_tensor.shape[0]
        Ci = x_tensor.shape[1]
        Hi = x_tensor.shape[2]
        Wi = x_tensor.shape[3]
        
        # Weight shape: [Co, Ci, Kh, Kw]
        Co = w_tensor.shape[0]
        Kh = kernel_shape[0]
        Kw = kernel_shape[1]
        
        # Output spatial dims
        Ho = (Hi + pads[0] + pads[2] - Kh) // strides[0] + 1
        Wo = (Wi + pads[1] + pads[3] - Kw) // strides[1] + 1
        
        # GEMM dimensions
        M = N * Ho * Wo
        K = Ci * Kh * Kw
        gemm_N = Co
        
        # Use GEMM tiling
        tile_config = self.compute_gemm_tiling(M, gemm_N, K)
        
        node.attrs['tile_config'] = tile_config
        node.attrs['im2col'] = {
            'input_shape': (N, Ci, Hi, Wi),
            'kernel_shape': (Kh, Kw),
            'strides': strides,
            'pads': pads,
            'output_shape': (N, Co, Ho, Wo),
            'gemm_shape': (M, gemm_N, K),
        }
    
    def _tile_matmul(self, graph: Graph, node: Node):
        """Generate tiles for MatMul operation"""
        # Same as GEMM but without bias
        self._tile_gemm(graph, node)


def tile_graph(graph: Graph, hw_config: Optional[HardwareConfig] = None) -> Graph:
    """Convenience function to tile a graph"""
    engine = TilingEngine(hw_config)
    return engine.tile_graph(graph)


# Export from module
__all__ = ['TilingEngine', 'HardwareConfig', 'GEMMTileConfig', 'tile_graph']


if __name__ == "__main__":
    # Test tiling
    print("Testing Tiling Engine")
    print("=" * 60)
    
    engine = TilingEngine()
    
    # Test GEMM tiling for various sizes
    test_cases = [
        (16, 16, 16),      # Tiny: fits in one tile
        (64, 64, 64),      # Small
        (256, 256, 256),   # Medium
        (784, 256, 784),   # LeNet FC1-like
        (1024, 1024, 1024),  # Large
        (4096, 4096, 4096),  # Very large
    ]
    
    for M, N, K in test_cases:
        config = engine.compute_gemm_tiling(M, N, K)
        total_tiles = config.num_m_tiles * config.num_n_tiles * config.num_k_tiles
        print(f"\nGEMM({M}x{K} @ {K}x{N} -> {M}x{N}):")
        print(f"  Tile size: {config.tile_m}x{config.tile_n}x{config.tile_k}")
        print(f"  Num tiles: {config.num_m_tiles}x{config.num_n_tiles}x{config.num_k_tiles} = {total_tiles}")
        print(f"  Memory per tile: {config.total_bytes//1024}KB")
        print(f"  Fits in SRAM: {config.total_bytes < engine.hw.sram_size}")

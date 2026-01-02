"""
Scheduler Module

Determines execution order and assigns operations to TPCs (Tensor Processing Clusters).
Handles memory allocation and generates a schedule for the code generator.

Optimizations:
- Double-buffering for overlapping DMA and compute
- Operator fusion for reduced memory traffic
- Multi-TPC parallelism
"""

import sys
import os
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import heapq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir.graph import Graph, Node, Tensor, OpType, TileInfo
from tiler.tiler import HardwareConfig


class ScheduleStrategy(Enum):
    """Scheduling strategies"""
    SEQUENTIAL = auto()      # Simple sequential execution
    DOUBLE_BUFFER = auto()   # Overlap DMA and compute
    PIPELINED = auto()       # Full pipeline with multiple stages
    PARALLEL = auto()        # Multi-TPC parallel execution


@dataclass
class ScheduleConfig:
    """Configuration for the scheduler"""
    strategy: ScheduleStrategy = ScheduleStrategy.DOUBLE_BUFFER
    enable_fusion: bool = True           # Fuse consecutive ops
    enable_recompute: bool = False       # Trade compute for memory
    max_tiles_per_batch: int = 4         # Tiles to prefetch
    dma_prefetch_depth: int = 2          # DMA operations to queue ahead


@dataclass
class MemoryBlock:
    """Represents an allocated memory block"""
    name: str
    address: int
    size: int
    tensor_name: str
    buffer_id: int = 0  # For double-buffering: 0 or 1
    
    @property
    def end_address(self) -> int:
        return self.address + self.size


@dataclass
class ScheduleEntry:
    """Single entry in the execution schedule"""
    order: int              # Execution order
    node_name: str          # Node being executed
    tile_id: Optional[int]  # Tile index (for tiled ops)
    tpc_id: int             # Which TPC executes this
    
    # DMA operations needed
    dma_loads: List[Tuple[str, int, int, int]] = field(default_factory=list)  # (tensor, ddr_addr, sram_addr, size)
    dma_stores: List[Tuple[str, int, int, int]] = field(default_factory=list)
    
    # Memory addresses for this operation
    input_addrs: Dict[str, int] = field(default_factory=dict)
    output_addr: int = 0
    
    # Dependencies (schedule orders that must complete first)
    depends_on: List[int] = field(default_factory=list)
    
    # Double-buffering info
    buffer_id: int = 0               # Which buffer set (0 or 1)
    async_dma: bool = False          # DMA runs async with previous compute
    wait_for_dma: bool = True        # Wait for DMA before compute
    
    # Fusion info
    fused_ops: List[str] = field(default_factory=list)  # Additional fused operations


class MemoryAllocator:
    """
    Memory allocator with double-buffering support
    
    Memory layout per TPC (2MB total):
    Buffer 0 (Ping):
      0x00000 - 0x3FFFF: Activation A buffer (256KB)
      0x40000 - 0x7FFFF: Weight buffer (256KB)
    Buffer 1 (Pong):
      0x80000 - 0xBFFFF: Activation B buffer (256KB)
      0xC0000 - 0xFFFFF: Weight buffer B (256KB)
    Shared:
      0x100000 - 0x17FFFF: Output buffer (512KB)
      0x180000 - 0x1FFFFF: Scratch (512KB)
    """
    
    def __init__(self, hw_config: HardwareConfig, double_buffer: bool = True):
        self.hw = hw_config
        self.double_buffer = double_buffer
        self.current_buffer = 0
        self.reset()
        
    def reset(self):
        """Reset allocator state"""
        if self.double_buffer:
            # Double-buffer layout
            self.regions = {
                'act_0': {'base': 0x00000, 'size': self.hw.sram_act_a // 2, 'offset': 0},
                'act_1': {'base': 0x80000, 'size': self.hw.sram_act_b, 'offset': 0},
                'weight_0': {'base': 0x40000, 'size': self.hw.sram_weight // 2, 'offset': 0},
                'weight_1': {'base': 0xC0000, 'size': self.hw.sram_weight // 2, 'offset': 0},
                'output': {'base': 0x100000, 'size': self.hw.sram_output, 'offset': 0},
                'scratch': {'base': 0x180000, 'size': self.hw.sram_scratch, 'offset': 0},
            }
        else:
            # Single-buffer layout (original)
            self.regions = {
                'act_a': {'base': 0x00000, 'size': self.hw.sram_act_a, 'offset': 0},
                'act_b': {'base': 0x80000, 'size': self.hw.sram_act_b, 'offset': 0},
                'weight': {'base': 0xC0000, 'size': self.hw.sram_weight, 'offset': 0},
                'output': {'base': 0x180000, 'size': self.hw.sram_output, 'offset': 0},
                'scratch': {'base': 0x1C0000, 'size': self.hw.sram_scratch, 'offset': 0},
            }
        self.allocations: Dict[str, MemoryBlock] = {}
    
    def swap_buffers(self):
        """Swap active buffer for double-buffering"""
        self.current_buffer = 1 - self.current_buffer
        # Reset offset for new buffer
        if self.double_buffer:
            self.regions[f'act_{self.current_buffer}']['offset'] = 0
            self.regions[f'weight_{self.current_buffer}']['offset'] = 0
    
    def allocate(self, name: str, size: int, region: str = 'scratch') -> int:
        """Allocate memory in specified region"""
        # Map to current buffer if double-buffering
        if self.double_buffer and region in ['act', 'weight']:
            region = f'{region}_{self.current_buffer}'
        elif region == 'act_a':
            region = f'act_{self.current_buffer}' if self.double_buffer else 'act_a'
        
        if region not in self.regions:
            raise ValueError(f"Unknown region: {region}")
        
        reg = self.regions[region]
        if reg['offset'] + size > reg['size']:
            raise MemoryError(f"Out of {region} memory: need {size}, have {reg['size'] - reg['offset']}")
        
        addr = reg['base'] + reg['offset']
        reg['offset'] += size
        
        # Align to 16 bytes
        reg['offset'] = (reg['offset'] + 15) & ~15
        
        self.allocations[name] = MemoryBlock(name, addr, size, name, self.current_buffer)
        return addr
    
    def free(self, name: str):
        """Free allocation (simple: just remove tracking)"""
        if name in self.allocations:
            del self.allocations[name]
    
    def reset_region(self, region: str):
        """Reset a region's allocator"""
        if region in self.regions:
            self.regions[region]['offset'] = 0
    
    def get_address(self, name: str) -> Optional[int]:
        """Get address of allocated block"""
        if name in self.allocations:
            return self.allocations[name].address
        return None


class Scheduler:
    """
    Schedules graph execution across TPCs
    
    Responsibilities:
    1. Topological ordering of operations
    2. TPC assignment (parallel execution where possible)
    3. Memory allocation for each operation
    4. DMA schedule generation
    5. Double-buffering for DMA/compute overlap
    6. Operator fusion
    """
    
    def __init__(self, hw_config: Optional[HardwareConfig] = None,
                 schedule_config: Optional[ScheduleConfig] = None):
        self.hw = hw_config or HardwareConfig()
        self.config = schedule_config or ScheduleConfig()
        self.double_buffer = self.config.strategy == ScheduleStrategy.DOUBLE_BUFFER
        self.allocator = MemoryAllocator(self.hw, self.double_buffer)
        
    def schedule(self, graph: Graph) -> List[ScheduleEntry]:
        """
        Generate execution schedule for the graph
        
        Returns: List of ScheduleEntry in execution order
        """
        schedule = []
        order = 0
        
        # Get topological order
        sorted_nodes = graph.topological_sort()
        
        # Apply operator fusion if enabled
        if self.config.enable_fusion:
            sorted_nodes = self._apply_fusion(graph, sorted_nodes)
        
        # Track when tensors are last used (for memory reuse)
        tensor_last_use = self._compute_tensor_lifetimes(graph, sorted_nodes)
        
        # Track which tensors are currently in SRAM
        tensors_in_sram: Set[str] = set()
        
        # DDR address assignment (simple: sequential)
        ddr_next_addr = 0
        ddr_addrs: Dict[str, int] = {}
        
        # Double-buffering state
        buffer_id = 0
        
        for node in sorted_nodes:
            # Assign TPC (simple round-robin for now)
            tpc_id = order % self.hw.num_tpcs
            node.tpc_id = tpc_id
            node.schedule_order = order
            
            if node.tiles:
                # Tiled operation: schedule each tile with double-buffering
                for tile_idx, tile in enumerate(node.tiles):
                    # Determine buffer for this tile
                    if self.double_buffer:
                        tile_buffer = tile_idx % 2
                        self.allocator.current_buffer = tile_buffer
                    
                    entry = self._schedule_tile(
                        graph, node, tile, tpc_id, order,
                        tensors_in_sram, ddr_addrs, ddr_next_addr
                    )
                    
                    # Double-buffering: async DMA for next tile
                    if self.double_buffer and tile_idx > 0:
                        entry.async_dma = True
                        entry.wait_for_dma = (tile_idx % 2 == 1)
                    
                    entry.buffer_id = tile_buffer if self.double_buffer else 0
                    schedule.append(entry)
                    order += 1
            else:
                # Non-tiled operation
                entry = self._schedule_node(
                    graph, node, tpc_id, order,
                    tensors_in_sram, ddr_addrs, ddr_next_addr
                )
                entry.buffer_id = buffer_id
                schedule.append(entry)
                order += 1
                
                # Swap buffers for double-buffering
                if self.double_buffer:
                    buffer_id = 1 - buffer_id
                    self.allocator.swap_buffers()
            
            # Update DDR addresses for new tensors
            for out_name in node.outputs:
                if out_name not in ddr_addrs:
                    tensor = graph.get_tensor(out_name)
                    if tensor:
                        ddr_addrs[out_name] = ddr_next_addr
                        ddr_next_addr += tensor.size_bytes
                        # Align to 64 bytes
                        ddr_next_addr = (ddr_next_addr + 63) & ~63
        
        # Post-process: add DMA prefetch hints
        if self.config.strategy == ScheduleStrategy.DOUBLE_BUFFER:
            schedule = self._add_prefetch_hints(schedule)
        
        return schedule
    
    def _apply_fusion(self, graph: Graph, nodes: List[Node]) -> List[Node]:
        """
        Apply operator fusion to reduce memory traffic
        
        Fusable patterns:
        - GEMM + Bias + ReLU
        - Conv2D + BatchNorm + ReLU
        - LayerNorm + GELU
        """
        fused_nodes = []
        skip_next = 0
        
        fusable_activations = {OpType.RELU, OpType.RELU6, OpType.GELU, OpType.SIGMOID, OpType.TANH}
        
        for i, node in enumerate(nodes):
            if skip_next > 0:
                skip_next -= 1
                continue
            
            # Try to fuse with following operations
            fused_with = []
            
            if node.op_type in [OpType.GEMM, OpType.MATMUL, OpType.CONV2D]:
                # Look ahead for fusable ops
                j = i + 1
                while j < len(nodes):
                    next_node = nodes[j]
                    
                    # Check if next node uses our output
                    if node.outputs[0] in next_node.inputs:
                        if next_node.op_type in fusable_activations:
                            fused_with.append(next_node.name)
                            skip_next += 1
                            j += 1
                        elif next_node.op_type == OpType.ADD and len(next_node.inputs) == 2:
                            # Bias add
                            fused_with.append(next_node.name)
                            skip_next += 1
                            j += 1
                        else:
                            break
                    else:
                        break
            
            if fused_with:
                node.attrs['fused_ops'] = fused_with
            
            fused_nodes.append(node)
        
        return fused_nodes
    
    def _add_prefetch_hints(self, schedule: List[ScheduleEntry]) -> List[ScheduleEntry]:
        """Add DMA prefetch hints for double-buffering"""
        prefetch_depth = self.config.dma_prefetch_depth
        
        for i, entry in enumerate(schedule):
            # Mark DMA loads that can be async
            if i < len(schedule) - prefetch_depth:
                # Look ahead to see if we can start this DMA early
                can_prefetch = True
                for j in range(1, prefetch_depth + 1):
                    if i + j < len(schedule):
                        future_entry = schedule[i + j]
                        # Check for dependencies
                        if entry.order in future_entry.depends_on:
                            can_prefetch = False
                            break
                
                entry.async_dma = can_prefetch
        
        return schedule
    
    def _schedule_node(self, graph: Graph, node: Node, tpc_id: int, order: int,
                       tensors_in_sram: Set[str], ddr_addrs: Dict[str, int],
                       ddr_next_addr: int) -> ScheduleEntry:
        """Schedule a non-tiled node"""
        entry = ScheduleEntry(
            order=order,
            node_name=node.name,
            tile_id=None,
            tpc_id=tpc_id
        )
        
        # Reset allocator for this operation
        self.allocator.reset()
        
        # Load inputs
        for inp_name in node.inputs:
            tensor = graph.get_tensor(inp_name)
            if not tensor:
                continue
            
            # Determine region based on tensor type
            if tensor.data is not None:
                region = 'weight'
            else:
                region = 'act_a'
            
            # Allocate SRAM space
            sram_addr = self.allocator.allocate(inp_name, tensor.size_bytes, region)
            entry.input_addrs[inp_name] = sram_addr
            
            # Generate DMA load if not already in SRAM
            if inp_name not in tensors_in_sram:
                ddr_addr = ddr_addrs.get(inp_name, 0)
                if tensor.data is not None:
                    # Weights have fixed DDR addresses
                    ddr_addr = ddr_addrs.get(inp_name, ddr_next_addr)
                    if inp_name not in ddr_addrs:
                        ddr_addrs[inp_name] = ddr_addr
                
                entry.dma_loads.append((inp_name, ddr_addr, sram_addr, tensor.size_bytes))
                tensors_in_sram.add(inp_name)
        
        # Allocate output
        for out_name in node.outputs:
            tensor = graph.get_tensor(out_name)
            if tensor:
                sram_addr = self.allocator.allocate(out_name, tensor.size_bytes, 'output')
                entry.output_addr = sram_addr
        
        return entry
    
    def _schedule_tile(self, graph: Graph, node: Node, tile: TileInfo,
                       tpc_id: int, order: int,
                       tensors_in_sram: Set[str], ddr_addrs: Dict[str, int],
                       ddr_next_addr: int) -> ScheduleEntry:
        """Schedule a single tile of a tiled operation"""
        entry = ScheduleEntry(
            order=order,
            node_name=node.name,
            tile_id=tile.tile_id,
            tpc_id=tpc_id
        )
        
        # Add tile dependencies
        for dep_tile_id in tile.depends_on:
            # Find the schedule order of the dependent tile
            dep_order = order - (tile.tile_id - dep_tile_id)
            entry.depends_on.append(dep_order)
        
        # For tiles, we use pre-computed addresses from tiler
        # (simplified: just copy from tile info)
        entry.input_addrs = tile.input_addrs.copy()
        entry.output_addr = tile.output_addr
        
        return entry
    
    def _compute_tensor_lifetimes(self, graph: Graph, 
                                   sorted_nodes: List[Node]) -> Dict[str, int]:
        """Compute when each tensor is last used (for memory reuse)"""
        last_use = {}
        
        for order, node in enumerate(sorted_nodes):
            for inp_name in node.inputs:
                last_use[inp_name] = order
        
        return last_use
    
    def estimate_memory_usage(self, graph: Graph) -> Dict[str, int]:
        """Estimate memory usage for the graph"""
        weight_mem = 0
        activation_mem = 0
        
        for tensor in graph.tensors.values():
            if tensor.data is not None:
                weight_mem += tensor.size_bytes
            else:
                activation_mem += tensor.size_bytes
        
        return {
            'weights': weight_mem,
            'activations': activation_mem,
            'total': weight_mem + activation_mem,
            'fits_in_sram': (weight_mem + activation_mem) < self.hw.sram_size
        }
    
    def estimate_execution_time(self, schedule: List[ScheduleEntry], 
                                 graph: Graph) -> Dict[str, float]:
        """
        Rough estimate of execution time
        
        Assumes:
        - GEMM: M*N*K MACs, 8x8 systolic does 64 MACs/cycle
        - DMA: 16 bytes/cycle
        - 100MHz clock
        """
        clock_mhz = 100
        macs_per_cycle = 64  # 8x8 systolic array
        dma_bytes_per_cycle = 16
        
        total_macs = 0
        total_dma_bytes = 0
        
        for entry in schedule:
            node = graph.get_node(entry.node_name)
            if not node:
                continue
            
            if node.op_type in [OpType.GEMM, OpType.MATMUL]:
                # Get dimensions
                inp = node.inputs[0]
                tensor = graph.get_tensor(inp)
                if tensor and len(tensor.shape) >= 2:
                    M = tensor.shape[-2]
                    K = tensor.shape[-1]
                    # Get N from weight
                    if len(node.inputs) > 1:
                        w_tensor = graph.get_tensor(node.inputs[1])
                        if w_tensor:
                            N = w_tensor.shape[-1]
                            total_macs += M * N * K
            
            # DMA
            for _, _, _, size in entry.dma_loads:
                total_dma_bytes += size
            for _, _, _, size in entry.dma_stores:
                total_dma_bytes += size
        
        compute_cycles = total_macs / macs_per_cycle
        dma_cycles = total_dma_bytes / dma_bytes_per_cycle
        
        # Assume some overlap
        total_cycles = max(compute_cycles, dma_cycles) + min(compute_cycles, dma_cycles) * 0.2
        
        time_us = total_cycles / clock_mhz
        
        return {
            'total_macs': total_macs,
            'total_dma_bytes': total_dma_bytes,
            'compute_cycles': compute_cycles,
            'dma_cycles': dma_cycles,
            'total_cycles': total_cycles,
            'time_us': time_us,
            'ops_per_sec': total_macs / (time_us / 1e6) if time_us > 0 else 0
        }


def schedule_graph(graph: Graph, hw_config: Optional[HardwareConfig] = None) -> List[ScheduleEntry]:
    """Convenience function to schedule a graph"""
    scheduler = Scheduler(hw_config)
    return scheduler.schedule(graph)


# Export from module
__all__ = ['Scheduler', 'ScheduleEntry', 'MemoryAllocator', 'schedule_graph', 
           'ScheduleConfig', 'ScheduleStrategy']


if __name__ == "__main__":
    # Test scheduler
    from ir.graph import create_simple_test_graph
    from tiler.tiler import TilingEngine
    
    print("Testing Scheduler")
    print("=" * 60)
    
    # Create and tile test graph
    graph = create_simple_test_graph()
    
    tiler = TilingEngine()
    tiler.tile_graph(graph)
    
    scheduler = Scheduler()
    schedule = scheduler.schedule(graph)
    
    print(f"\nSchedule ({len(schedule)} entries):")
    for entry in schedule:
        tile_str = f" tile {entry.tile_id}" if entry.tile_id is not None else ""
        deps_str = f" (deps: {entry.depends_on})" if entry.depends_on else ""
        print(f"  {entry.order}: {entry.node_name}{tile_str} on TPC{entry.tpc_id}{deps_str}")
        if entry.dma_loads:
            for name, ddr, sram, size in entry.dma_loads:
                print(f"       DMA_LOAD {name}: DDR[{ddr:#x}] -> SRAM[{sram:#x}] ({size} bytes)")
    
    print("\nMemory estimate:")
    mem = scheduler.estimate_memory_usage(graph)
    for k, v in mem.items():
        if isinstance(v, bool):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:,} bytes")
    
    print("\nExecution estimate:")
    timing = scheduler.estimate_execution_time(schedule, graph)
    for k, v in timing.items():
        if isinstance(v, float):
            print(f"  {k}: {v:,.2f}")
        else:
            print(f"  {k}: {v:,}")

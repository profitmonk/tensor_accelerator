"""
Coverage Collection Framework

Provides functional, protocol, and cross coverage for verification.
Compatible with cocotb-coverage but also works standalone.

Author: Tensor Accelerator Project
Date: December 2025
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Callable, Any, Tuple
from enum import Enum, auto
from collections import defaultdict
import time


class CoverageType(Enum):
    """Types of coverage"""
    FUNCTIONAL = auto()
    PROTOCOL = auto()
    CROSS = auto()


@dataclass
class CoverageBin:
    """Single coverage bin"""
    name: str
    hit_count: int = 0
    target_count: int = 1  # How many hits needed
    
    @property
    def covered(self) -> bool:
        return self.hit_count >= self.target_count
    
    def hit(self):
        self.hit_count += 1


@dataclass
class CoverPoint:
    """Coverage point with multiple bins"""
    name: str
    description: str = ""
    bins: Dict[str, CoverageBin] = field(default_factory=dict)
    ignore_bins: Set[str] = field(default_factory=set)
    illegal_bins: Set[str] = field(default_factory=set)
    
    def add_bin(self, name: str, target: int = 1):
        """Add a coverage bin"""
        self.bins[name] = CoverageBin(name=name, target_count=target)
        
    def add_bins_range(self, name_prefix: str, values: List[Any]):
        """Add bins for a range of values"""
        for v in values:
            self.add_bin(f"{name_prefix}_{v}")
            
    def sample(self, value: Any):
        """Sample a value"""
        bin_name = str(value)
        
        if bin_name in self.illegal_bins:
            raise ValueError(f"Illegal value sampled: {value}")
            
        if bin_name in self.ignore_bins:
            return
            
        if bin_name in self.bins:
            self.bins[bin_name].hit()
        else:
            # Auto-create bin
            self.bins[bin_name] = CoverageBin(name=bin_name, hit_count=1)
            
    @property
    def coverage_percent(self) -> float:
        """Calculate coverage percentage"""
        if not self.bins:
            return 0.0
        covered = sum(1 for b in self.bins.values() if b.covered)
        return 100.0 * covered / len(self.bins)
    
    @property
    def hits(self) -> int:
        """Total hits across all bins"""
        return sum(b.hit_count for b in self.bins.values())


@dataclass
class CrossCoverage:
    """Cross coverage between multiple cover points"""
    name: str
    coverpoints: List[str]  # Names of coverpoints to cross
    bins: Dict[Tuple, CoverageBin] = field(default_factory=dict)
    
    def sample(self, values: Tuple):
        """Sample cross values"""
        if values in self.bins:
            self.bins[values].hit()
        else:
            self.bins[values] = CoverageBin(name=str(values), hit_count=1)
            
    @property
    def coverage_percent(self) -> float:
        if not self.bins:
            return 0.0
        covered = sum(1 for b in self.bins.values() if b.covered)
        return 100.0 * covered / len(self.bins)


class CoverageCollector:
    """
    Main coverage collection and reporting class
    """
    
    def __init__(self, name: str = "coverage"):
        self.name = name
        self.coverpoints: Dict[str, CoverPoint] = {}
        self.crosses: Dict[str, CrossCoverage] = {}
        self.start_time = time.time()
        
    def add_coverpoint(self, name: str, description: str = "",
                       bins: List[str] = None) -> CoverPoint:
        """Add a coverage point"""
        cp = CoverPoint(name=name, description=description)
        if bins:
            for b in bins:
                cp.add_bin(b)
        self.coverpoints[name] = cp
        return cp
        
    def add_cross(self, name: str, coverpoints: List[str]) -> CrossCoverage:
        """Add cross coverage"""
        cross = CrossCoverage(name=name, coverpoints=coverpoints)
        self.crosses[name] = cross
        return cross
        
    def sample(self, coverpoint: str, value: Any):
        """Sample a coverpoint"""
        if coverpoint in self.coverpoints:
            self.coverpoints[coverpoint].sample(value)
            
    def sample_cross(self, cross_name: str, values: Tuple):
        """Sample cross coverage"""
        if cross_name in self.crosses:
            self.crosses[cross_name].sample(values)
            
    @property
    def total_coverage(self) -> float:
        """Overall coverage percentage"""
        total_bins = 0
        covered_bins = 0
        
        for cp in self.coverpoints.values():
            total_bins += len(cp.bins)
            covered_bins += sum(1 for b in cp.bins.values() if b.covered)
            
        for cross in self.crosses.values():
            total_bins += len(cross.bins)
            covered_bins += sum(1 for b in cross.bins.values() if b.covered)
            
        if total_bins == 0:
            return 0.0
        return 100.0 * covered_bins / total_bins
    
    def get_uncovered(self) -> Dict[str, List[str]]:
        """Get list of uncovered bins"""
        uncovered = {}
        
        for name, cp in self.coverpoints.items():
            uc = [b.name for b in cp.bins.values() if not b.covered]
            if uc:
                uncovered[name] = uc
                
        return uncovered
    
    def report(self) -> str:
        """Generate coverage report"""
        lines = [
            "=" * 70,
            f"Coverage Report: {self.name}",
            "=" * 70,
            f"Total Coverage: {self.total_coverage:.1f}%",
            f"Elapsed Time: {time.time() - self.start_time:.1f}s",
            "",
            "Coverpoints:",
        ]
        
        for name, cp in sorted(self.coverpoints.items()):
            lines.append(f"  {name}: {cp.coverage_percent:.1f}% ({cp.hits} hits)")
            
            # Show uncovered bins
            uncovered = [b.name for b in cp.bins.values() if not b.covered]
            if uncovered and len(uncovered) <= 5:
                lines.append(f"    Uncovered: {', '.join(uncovered)}")
            elif uncovered:
                lines.append(f"    Uncovered: {len(uncovered)} bins")
                
        if self.crosses:
            lines.append("")
            lines.append("Cross Coverage:")
            for name, cross in sorted(self.crosses.items()):
                lines.append(f"  {name}: {cross.coverage_percent:.1f}%")
                
        lines.append("=" * 70)
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """Export coverage data as JSON"""
        data = {
            'name': self.name,
            'total_coverage': self.total_coverage,
            'coverpoints': {},
            'crosses': {}
        }
        
        for name, cp in self.coverpoints.items():
            data['coverpoints'][name] = {
                'coverage': cp.coverage_percent,
                'bins': {
                    b.name: {'hits': b.hit_count, 'covered': b.covered}
                    for b in cp.bins.values()
                }
            }
            
        for name, cross in self.crosses.items():
            data['crosses'][name] = {
                'coverage': cross.coverage_percent,
                'coverpoints': cross.coverpoints,
                'bins_hit': len([b for b in cross.bins.values() if b.covered])
            }
            
        return json.dumps(data, indent=2)
    
    def save_report(self, filename: str):
        """Save coverage report to file"""
        with open(filename + ".txt", 'w') as f:
            f.write(self.report())
        with open(filename + ".json", 'w') as f:
            f.write(self.to_json())


class AXICoverageCollector(CoverageCollector):
    """
    Specialized coverage collector for AXI protocol
    """
    
    def __init__(self, name: str = "axi_coverage"):
        super().__init__(name)
        self._setup_coverpoints()
        
    def _setup_coverpoints(self):
        """Setup AXI-specific coverpoints"""
        
        # Transaction types
        self.add_coverpoint(
            "txn_type",
            "Transaction type coverage",
            bins=["read", "write"]
        )
        
        # Burst lengths
        burst_lens = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        self.add_coverpoint(
            "burst_length", 
            "Burst length coverage",
            bins=[str(x) for x in burst_lens]
        )
        
        # Burst types
        self.add_coverpoint(
            "burst_type",
            "Burst type coverage", 
            bins=["FIXED", "INCR", "WRAP"]
        )
        
        # Transfer sizes
        sizes = ["1B", "2B", "4B", "8B", "16B", "32B"]
        self.add_coverpoint(
            "transfer_size",
            "Transfer size coverage",
            bins=sizes
        )
        
        # Response types
        self.add_coverpoint(
            "response",
            "Response type coverage",
            bins=["OKAY", "EXOKAY", "SLVERR", "DECERR"]
        )
        
        # Address alignment
        self.add_coverpoint(
            "address_align",
            "Address alignment coverage",
            bins=["aligned", "unaligned"]
        )
        
        # Outstanding transactions
        outstanding = [str(x) for x in range(1, 17)]
        self.add_coverpoint(
            "outstanding",
            "Outstanding transaction coverage",
            bins=outstanding
        )
        
        # Handshake patterns
        self.add_coverpoint(
            "handshake",
            "Handshake pattern coverage",
            bins=["valid_first", "ready_first", "simultaneous"]
        )
        
        # Back-to-back transactions
        self.add_coverpoint(
            "back_to_back",
            "Back-to-back transaction coverage",
            bins=["yes", "no"]
        )
        
        # Cross coverage: burst_length x burst_type
        self.add_cross("burst_cross", ["burst_length", "burst_type"])
        
        # Cross coverage: txn_type x outstanding
        self.add_cross("outstanding_cross", ["txn_type", "outstanding"])
        
    def sample_transaction(self, txn_type: str, burst_len: int, 
                           burst_type: str, size_bytes: int,
                           response: str, aligned: bool,
                           outstanding: int, back_to_back: bool):
        """Sample a complete AXI transaction"""
        self.sample("txn_type", txn_type)
        self.sample("burst_length", str(burst_len))
        self.sample("burst_type", burst_type)
        self.sample("transfer_size", f"{size_bytes}B")
        self.sample("response", response)
        self.sample("address_align", "aligned" if aligned else "unaligned")
        self.sample("outstanding", str(min(outstanding, 16)))
        self.sample("back_to_back", "yes" if back_to_back else "no")
        
        # Cross coverage
        self.sample_cross("burst_cross", (str(burst_len), burst_type))
        self.sample_cross("outstanding_cross", (txn_type, str(min(outstanding, 16))))
        
    def sample_handshake(self, valid_first: bool, ready_first: bool):
        """Sample handshake pattern"""
        if valid_first and not ready_first:
            self.sample("handshake", "valid_first")
        elif ready_first and not valid_first:
            self.sample("handshake", "ready_first")
        else:
            self.sample("handshake", "simultaneous")


class FunctionalCoverageCollector(CoverageCollector):
    """
    Specialized coverage collector for tensor accelerator functionality
    """
    
    def __init__(self, name: str = "functional_coverage"):
        super().__init__(name)
        self._setup_coverpoints()
        
    def _setup_coverpoints(self):
        """Setup functional coverpoints"""
        
        # Operation types
        self.add_coverpoint(
            "operation",
            "Operation type coverage",
            bins=["GEMM", "GEMM_ACC", "GEMM_RELU", "GEMM_BIAS",
                  "CONV2D", "MAXPOOL", "AVGPOOL", "RELU", "GELU",
                  "LAYERNORM", "SOFTMAX", "ATTENTION", "DMA_LOAD", "DMA_STORE"]
        )
        
        # GEMM dimensions (M)
        m_bins = ["1", "8", "16", "32", "64", "128", "256", "512", "1024"]
        self.add_coverpoint("gemm_m", "GEMM M dimension", bins=m_bins)
        
        # GEMM dimensions (N)
        n_bins = ["1", "8", "16", "32", "64", "128", "256"]
        self.add_coverpoint("gemm_n", "GEMM N dimension", bins=n_bins)
        
        # GEMM dimensions (K)
        k_bins = ["1", "8", "16", "32", "64", "128", "256", "512"]
        self.add_coverpoint("gemm_k", "GEMM K dimension", bins=k_bins)
        
        # Data values (INT8)
        self.add_coverpoint(
            "int8_values",
            "INT8 value coverage",
            bins=["-128", "-64", "-1", "0", "1", "64", "127"]
        )
        
        # Accumulator values
        self.add_coverpoint(
            "accumulator",
            "Accumulator coverage",
            bins=["zero", "positive_small", "positive_large", 
                  "negative_small", "negative_large", "overflow"]
        )
        
        # TPC usage
        self.add_coverpoint(
            "tpc_usage",
            "TPC utilization coverage",
            bins=["1_tpc", "2_tpc", "3_tpc", "4_tpc"]
        )
        
        # Tiling patterns
        self.add_coverpoint(
            "tiling",
            "Tiling pattern coverage",
            bins=["no_tiling", "m_tiled", "n_tiled", "k_tiled", "all_tiled"]
        )
        
        # Memory access patterns
        self.add_coverpoint(
            "memory_pattern",
            "Memory access pattern coverage",
            bins=["sequential", "strided", "random"]
        )
        
        # Cross: operation x tpc_usage
        self.add_cross("op_tpc_cross", ["operation", "tpc_usage"])
        
        # Cross: gemm_m x gemm_n
        self.add_cross("gemm_mn_cross", ["gemm_m", "gemm_n"])
        
    def sample_gemm(self, m: int, n: int, k: int, 
                    operation: str = "GEMM", num_tpc: int = 1):
        """Sample GEMM operation"""
        self.sample("operation", operation)
        
        # Bin dimensions
        m_bin = self._find_bin(m, [1, 8, 16, 32, 64, 128, 256, 512, 1024])
        n_bin = self._find_bin(n, [1, 8, 16, 32, 64, 128, 256])
        k_bin = self._find_bin(k, [1, 8, 16, 32, 64, 128, 256, 512])
        
        self.sample("gemm_m", str(m_bin))
        self.sample("gemm_n", str(n_bin))
        self.sample("gemm_k", str(k_bin))
        
        self.sample("tpc_usage", f"{num_tpc}_tpc")
        
        # Cross coverage
        self.sample_cross("op_tpc_cross", (operation, f"{num_tpc}_tpc"))
        self.sample_cross("gemm_mn_cross", (str(m_bin), str(n_bin)))
        
    def sample_int8_value(self, value: int):
        """Sample INT8 data value"""
        if value == -128:
            self.sample("int8_values", "-128")
        elif value < -64:
            self.sample("int8_values", "-64")
        elif value < 0:
            self.sample("int8_values", "-1")
        elif value == 0:
            self.sample("int8_values", "0")
        elif value < 64:
            self.sample("int8_values", "1")
        elif value < 127:
            self.sample("int8_values", "64")
        else:
            self.sample("int8_values", "127")
            
    def sample_accumulator(self, value: int):
        """Sample accumulator value"""
        if value == 0:
            self.sample("accumulator", "zero")
        elif 0 < value < 10000:
            self.sample("accumulator", "positive_small")
        elif value >= 10000:
            self.sample("accumulator", "positive_large")
        elif -10000 < value < 0:
            self.sample("accumulator", "negative_small")
        else:
            self.sample("accumulator", "negative_large")
            
    @staticmethod
    def _find_bin(value: int, bins: List[int]) -> int:
        """Find appropriate bin for value"""
        for b in reversed(bins):
            if value >= b:
                return b
        return bins[0]


# Global coverage database
_coverage_db: Dict[str, CoverageCollector] = {}


def get_coverage(name: str) -> Optional[CoverageCollector]:
    """Get coverage collector by name"""
    return _coverage_db.get(name)


def register_coverage(collector: CoverageCollector):
    """Register coverage collector"""
    _coverage_db[collector.name] = collector
    

def report_all_coverage() -> str:
    """Generate report for all coverage collectors"""
    lines = ["=" * 70, "COMPLETE COVERAGE REPORT", "=" * 70, ""]
    
    for name, collector in sorted(_coverage_db.items()):
        lines.append(collector.report())
        lines.append("")
        
    # Overall summary
    total_points = sum(len(c.coverpoints) for c in _coverage_db.values())
    total_crosses = sum(len(c.crosses) for c in _coverage_db.values())
    avg_coverage = sum(c.total_coverage for c in _coverage_db.values()) / len(_coverage_db) if _coverage_db else 0
    
    lines.extend([
        "SUMMARY",
        "-" * 70,
        f"Coverage Groups: {len(_coverage_db)}",
        f"Total Coverpoints: {total_points}",
        f"Total Crosses: {total_crosses}",
        f"Average Coverage: {avg_coverage:.1f}%",
        "=" * 70
    ])
    
    return "\n".join(lines)

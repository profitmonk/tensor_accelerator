"""
Coverage Collection Framework

This package provides coverage collection and reporting for verification.
"""

from .coverage_collector import (
    CoverageCollector,
    CoverPoint,
    CoverageBin,
    CrossCoverage,
    AXICoverageCollector,
    FunctionalCoverageCollector,
    get_coverage,
    register_coverage,
    report_all_coverage
)

__all__ = [
    'CoverageCollector',
    'CoverPoint',
    'CoverageBin',
    'CrossCoverage',
    'AXICoverageCollector',
    'FunctionalCoverageCollector',
    'get_coverage',
    'register_coverage',
    'report_all_coverage'
]

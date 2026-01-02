"""
AXI Protocol Verification

This package provides protocol checkers and scoreboards for AXI4.
"""

from .protocol_checker import (
    AXI4ProtocolChecker,
    AXI4Scoreboard,
    ViolationType,
    Violation
)

__all__ = [
    'AXI4ProtocolChecker',
    'AXI4Scoreboard',
    'ViolationType',
    'Violation'
]

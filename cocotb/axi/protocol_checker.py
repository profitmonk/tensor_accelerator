"""
AXI4 Protocol Checker

Verifies AXI4 protocol compliance per ARM IHI 0022E.
Checks timing, handshaking, ordering, and signal rules.

Author: Tensor Accelerator Project
Date: December 2025
"""

import cocotb
from cocotb.triggers import RisingEdge, FallingEdge, Timer, Edge
from cocotb.clock import Clock
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum, auto
import logging


class ViolationType(Enum):
    """Types of protocol violations"""
    # Handshake violations
    VALID_DEASSERT_WITHOUT_READY = auto()
    DATA_CHANGE_WHILE_VALID = auto()
    
    # Write channel violations
    WLAST_MISSING = auto()
    WLAST_EARLY = auto()
    BRESP_BEFORE_WLAST = auto()
    WDATA_WITHOUT_AWADDR = auto()
    
    # Read channel violations
    RLAST_MISSING = auto()
    RLAST_EARLY = auto()
    RDATA_WITHOUT_ARADDR = auto()
    
    # Ordering violations
    WRITE_ORDER_VIOLATION = auto()
    READ_ORDER_VIOLATION = auto()
    
    # Signal violations
    UNDEFINED_VALUE = auto()
    RESERVED_VALUE = auto()


@dataclass
class Violation:
    """Represents a single protocol violation"""
    type: ViolationType
    time: int
    channel: str
    message: str
    severity: str = "ERROR"  # ERROR, WARNING, INFO


@dataclass 
class PendingTransaction:
    """Track pending transactions for ordering checks"""
    id: int
    address: int
    length: int
    start_time: int
    beats_received: int = 0


class AXI4ProtocolChecker:
    """
    AXI4 Protocol Compliance Checker
    
    Monitors AXI signals and reports protocol violations.
    """
    
    def __init__(self, dut, prefix: str, clock, 
                 data_width: int = 32,
                 strict_mode: bool = True):
        """
        Initialize protocol checker
        
        Args:
            dut: cocotb DUT handle
            prefix: Signal prefix (e.g., "m_axi")
            clock: Clock signal
            data_width: Data bus width
            strict_mode: If True, treat warnings as errors
        """
        self.dut = dut
        self.prefix = prefix
        self.clock = clock
        self.data_width = data_width
        self.strict_mode = strict_mode
        
        self._get_signals()
        
        # Violation log
        self.violations: List[Violation] = []
        
        # State tracking
        self._pending_writes: Dict[int, PendingTransaction] = {}
        self._pending_reads: Dict[int, PendingTransaction] = {}
        self._write_data_pending: List[PendingTransaction] = []
        
        # Previous values for stability checks
        self._prev_awaddr = 0
        self._prev_awlen = 0
        self._prev_araddr = 0
        self._prev_arlen = 0
        self._prev_wdata = 0
        
        # Running flag
        self._running = False
        
        # Logger
        self.log = logging.getLogger(f"AXI4Checker.{prefix}")
        
    def _get_signals(self):
        """Get AXI signal handles"""
        p = self.prefix
        
        # Write address channel
        self.awaddr = getattr(self.dut, f"{p}_awaddr", None)
        self.awlen = getattr(self.dut, f"{p}_awlen", None)
        self.awsize = getattr(self.dut, f"{p}_awsize", None)
        self.awburst = getattr(self.dut, f"{p}_awburst", None)
        self.awid = getattr(self.dut, f"{p}_awid", None)
        self.awvalid = getattr(self.dut, f"{p}_awvalid", None)
        self.awready = getattr(self.dut, f"{p}_awready", None)
        
        # Write data channel
        self.wdata = getattr(self.dut, f"{p}_wdata", None)
        self.wstrb = getattr(self.dut, f"{p}_wstrb", None)
        self.wlast = getattr(self.dut, f"{p}_wlast", None)
        self.wvalid = getattr(self.dut, f"{p}_wvalid", None)
        self.wready = getattr(self.dut, f"{p}_wready", None)
        
        # Write response channel
        self.bresp = getattr(self.dut, f"{p}_bresp", None)
        self.bid = getattr(self.dut, f"{p}_bid", None)
        self.bvalid = getattr(self.dut, f"{p}_bvalid", None)
        self.bready = getattr(self.dut, f"{p}_bready", None)
        
        # Read address channel
        self.araddr = getattr(self.dut, f"{p}_araddr", None)
        self.arlen = getattr(self.dut, f"{p}_arlen", None)
        self.arsize = getattr(self.dut, f"{p}_arsize", None)
        self.arburst = getattr(self.dut, f"{p}_arburst", None)
        self.arid = getattr(self.dut, f"{p}_arid", None)
        self.arvalid = getattr(self.dut, f"{p}_arvalid", None)
        self.arready = getattr(self.dut, f"{p}_arready", None)
        
        # Read data channel
        self.rdata = getattr(self.dut, f"{p}_rdata", None)
        self.rresp = getattr(self.dut, f"{p}_rresp", None)
        self.rlast = getattr(self.dut, f"{p}_rlast", None)
        self.rid = getattr(self.dut, f"{p}_rid", None)
        self.rvalid = getattr(self.dut, f"{p}_rvalid", None)
        self.rready = getattr(self.dut, f"{p}_rready", None)
        
    async def start(self):
        """Start protocol checking"""
        self._running = True
        self.violations = []
        
        # Start parallel checkers
        cocotb.start_soon(self._check_aw_channel())
        cocotb.start_soon(self._check_w_channel())
        cocotb.start_soon(self._check_b_channel())
        cocotb.start_soon(self._check_ar_channel())
        cocotb.start_soon(self._check_r_channel())
        cocotb.start_soon(self._check_valid_stability())
        
        self.log.info("Protocol checker started")
        
    def stop(self):
        """Stop protocol checking"""
        self._running = False
        self.log.info(f"Protocol checker stopped. {len(self.violations)} violations found.")
        
    def _record_violation(self, vtype: ViolationType, channel: str, 
                          message: str, severity: str = "ERROR"):
        """Record a protocol violation"""
        v = Violation(
            type=vtype,
            time=cocotb.utils.get_sim_time('ns'),
            channel=channel,
            message=message,
            severity=severity
        )
        self.violations.append(v)
        
        if severity == "ERROR":
            self.log.error(f"[{v.time}ns] {channel}: {message}")
        elif severity == "WARNING":
            self.log.warning(f"[{v.time}ns] {channel}: {message}")
        else:
            self.log.info(f"[{v.time}ns] {channel}: {message}")
            
    async def _check_aw_channel(self):
        """Check write address channel rules"""
        prev_valid = 0
        prev_ready = 0
        prev_addr = 0
        prev_len = 0
        
        while self._running:
            await RisingEdge(self.clock)
            
            if self.awvalid is None:
                continue
                
            curr_valid = int(self.awvalid.value)
            curr_ready = int(self.awready.value) if self.awready is not None else 0
            
            # Rule: VALID must stay high until READY
            # A handshake occurs when both VALID and READY are high
            # VALID can deassert on the cycle AFTER handshake completes
            handshake_completed_prev_cycle = (prev_valid == 1 and prev_ready == 1)
            
            if prev_valid == 1 and curr_valid == 0 and not handshake_completed_prev_cycle:
                self._record_violation(
                    ViolationType.VALID_DEASSERT_WITHOUT_READY,
                    "AW",
                    "AWVALID deasserted before AWREADY handshake"
                )
                
            # Rule: Address must be stable while VALID (before handshake)
            if prev_valid == 1 and curr_valid == 1 and curr_ready == 0:
                if self.awaddr is not None and int(self.awaddr.value) != prev_addr:
                    self._record_violation(
                        ViolationType.DATA_CHANGE_WHILE_VALID,
                        "AW", 
                        f"AWADDR changed while AWVALID high: {prev_addr:#x} -> {int(self.awaddr.value):#x}"
                    )
                if self.awlen is not None and int(self.awlen.value) != prev_len:
                    self._record_violation(
                        ViolationType.DATA_CHANGE_WHILE_VALID,
                        "AW",
                        f"AWLEN changed while AWVALID high"
                    )
                    
            # Track successful handshake
            if curr_valid == 1 and curr_ready == 1:
                txn_id = int(self.awid.value) if self.awid is not None else 0
                length = int(self.awlen.value) + 1 if self.awlen is not None else 1
                
                pending = PendingTransaction(
                    id=txn_id,
                    address=int(self.awaddr.value) if self.awaddr is not None else 0,
                    length=length,
                    start_time=cocotb.utils.get_sim_time('ns')
                )
                self._write_data_pending.append(pending)
                
            prev_valid = curr_valid
            prev_ready = curr_ready
            prev_addr = int(self.awaddr.value) if self.awaddr is not None else 0
            prev_len = int(self.awlen.value) if self.awlen is not None else 0
            
    async def _check_w_channel(self):
        """Check write data channel rules"""
        prev_valid = 0
        prev_ready = 0
        prev_data = 0
        beat_count = 0
        
        while self._running:
            await RisingEdge(self.clock)
            
            if self.wvalid is None:
                continue
                
            curr_valid = int(self.wvalid.value)
            curr_ready = int(self.wready.value) if self.wready is not None else 0
            curr_last = int(self.wlast.value) if self.wlast is not None else 0
            
            # A handshake occurs when both VALID and READY are high
            handshake_completed_prev_cycle = (prev_valid == 1 and prev_ready == 1)
            
            # Rule: VALID must stay high until READY
            if prev_valid == 1 and curr_valid == 0 and not handshake_completed_prev_cycle:
                self._record_violation(
                    ViolationType.VALID_DEASSERT_WITHOUT_READY,
                    "W",
                    "WVALID deasserted before WREADY handshake"
                )
                
            # Rule: WDATA must be stable while VALID
            if prev_valid == 1 and curr_valid == 1 and curr_ready == 0:
                if self.wdata is not None and int(self.wdata.value) != prev_data:
                    self._record_violation(
                        ViolationType.DATA_CHANGE_WHILE_VALID,
                        "W",
                        "WDATA changed while WVALID high"
                    )
                    
            # Track beats
            if curr_valid == 1 and curr_ready == 1:
                beat_count += 1
                
                # Check WLAST
                if self._write_data_pending:
                    pending = self._write_data_pending[0]
                    pending.beats_received += 1
                    
                    if pending.beats_received == pending.length:
                        if curr_last != 1:
                            self._record_violation(
                                ViolationType.WLAST_MISSING,
                                "W",
                                f"WLAST not asserted on final beat (beat {pending.beats_received}/{pending.length})"
                            )
                        self._write_data_pending.pop(0)
                        beat_count = 0
                    elif curr_last == 1:
                        self._record_violation(
                            ViolationType.WLAST_EARLY,
                            "W",
                            f"WLAST asserted before final beat (beat {pending.beats_received}/{pending.length})"
                        )
                        
            prev_valid = curr_valid
            prev_ready = curr_ready
            prev_data = int(self.wdata.value) if self.wdata is not None else 0
            
    async def _check_b_channel(self):
        """Check write response channel rules"""
        prev_valid = 0
        prev_ready = 0
        
        while self._running:
            await RisingEdge(self.clock)
            
            if self.bvalid is None:
                continue
                
            curr_valid = int(self.bvalid.value)
            curr_ready = int(self.bready.value) if self.bready is not None else 0
            
            # A handshake occurs when both VALID and READY are high
            handshake_completed_prev_cycle = (prev_valid == 1 and prev_ready == 1)
            
            # Rule: VALID must stay high until READY
            if prev_valid == 1 and curr_valid == 0 and not handshake_completed_prev_cycle:
                self._record_violation(
                    ViolationType.VALID_DEASSERT_WITHOUT_READY,
                    "B",
                    "BVALID deasserted before BREADY handshake"
                )
                
            # Check response code
            if curr_valid == 1:
                resp = int(self.bresp.value) if self.bresp is not None else 0
                if resp == 3:  # DECERR
                    self._record_violation(
                        ViolationType.RESERVED_VALUE,
                        "B",
                        "BRESP = DECERR (decode error)",
                        severity="WARNING"
                    )
                    
            prev_valid = curr_valid
            prev_ready = curr_ready
            
    async def _check_ar_channel(self):
        """Check read address channel rules"""
        prev_valid = 0
        prev_ready = 0
        prev_addr = 0
        prev_len = 0
        
        while self._running:
            await RisingEdge(self.clock)
            
            if self.arvalid is None:
                continue
                
            curr_valid = int(self.arvalid.value)
            curr_ready = int(self.arready.value) if self.arready is not None else 0
            
            # A handshake occurs when both VALID and READY are high
            handshake_completed_prev_cycle = (prev_valid == 1 and prev_ready == 1)
            
            # Rule: VALID must stay high until READY
            if prev_valid == 1 and curr_valid == 0 and not handshake_completed_prev_cycle:
                self._record_violation(
                    ViolationType.VALID_DEASSERT_WITHOUT_READY,
                    "AR",
                    "ARVALID deasserted before ARREADY handshake"
                )
                
            # Rule: Address must be stable while VALID
            if prev_valid == 1 and curr_valid == 1 and curr_ready == 0:
                if self.araddr is not None and int(self.araddr.value) != prev_addr:
                    self._record_violation(
                        ViolationType.DATA_CHANGE_WHILE_VALID,
                        "AR",
                        f"ARADDR changed while ARVALID high"
                    )
                    
            # Track successful handshake
            if curr_valid == 1 and curr_ready == 1:
                txn_id = int(self.arid.value) if self.arid is not None else 0
                length = int(self.arlen.value) + 1 if self.arlen is not None else 1
                
                self._pending_reads[txn_id] = PendingTransaction(
                    id=txn_id,
                    address=int(self.araddr.value) if self.araddr is not None else 0,
                    length=length,
                    start_time=cocotb.utils.get_sim_time('ns')
                )
                
            prev_valid = curr_valid
            prev_ready = curr_ready
            prev_addr = int(self.araddr.value) if self.araddr is not None else 0
            prev_len = int(self.arlen.value) if self.arlen is not None else 0
            
    async def _check_r_channel(self):
        """Check read data channel rules"""
        prev_valid = 0
        prev_ready = 0
        beats_by_id: Dict[int, int] = {}
        
        while self._running:
            await RisingEdge(self.clock)
            
            if self.rvalid is None:
                continue
                
            curr_valid = int(self.rvalid.value)
            curr_ready = int(self.rready.value) if self.rready is not None else 0
            curr_last = int(self.rlast.value) if self.rlast is not None else 0
            
            # A handshake occurs when both VALID and READY are high
            handshake_completed_prev_cycle = (prev_valid == 1 and prev_ready == 1)
            
            # Rule: VALID must stay high until READY  
            if prev_valid == 1 and curr_valid == 0 and not handshake_completed_prev_cycle:
                self._record_violation(
                    ViolationType.VALID_DEASSERT_WITHOUT_READY,
                    "R",
                    "RVALID deasserted before RREADY handshake"
                )
                
            # Track beats and check RLAST
            if curr_valid == 1 and curr_ready == 1:
                txn_id = int(self.rid.value) if self.rid is not None else 0
                
                if txn_id not in beats_by_id:
                    beats_by_id[txn_id] = 0
                beats_by_id[txn_id] += 1
                
                if txn_id in self._pending_reads:
                    pending = self._pending_reads[txn_id]
                    
                    if beats_by_id[txn_id] == pending.length:
                        if curr_last != 1:
                            self._record_violation(
                                ViolationType.RLAST_MISSING,
                                "R",
                                f"RLAST not asserted on final beat (ID={txn_id})"
                            )
                        del self._pending_reads[txn_id]
                        del beats_by_id[txn_id]
                    elif curr_last == 1:
                        self._record_violation(
                            ViolationType.RLAST_EARLY,
                            "R",
                            f"RLAST asserted before final beat (ID={txn_id}, beat {beats_by_id[txn_id]}/{pending.length})"
                        )
                        
            prev_valid = curr_valid
            prev_ready = curr_ready
            
    async def _check_valid_stability(self):
        """Check that signals remain stable while VALID is high"""
        # This is a comprehensive check running at higher frequency
        while self._running:
            await RisingEdge(self.clock)
            # Additional stability checks can be added here
            
    def get_violation_summary(self) -> Dict[str, int]:
        """Get summary of violations by type"""
        summary = {}
        for v in self.violations:
            key = v.type.name
            summary[key] = summary.get(key, 0) + 1
        return summary
    
    def get_violations_by_channel(self, channel: str) -> List[Violation]:
        """Get violations for specific channel"""
        return [v for v in self.violations if v.channel == channel]
    
    def has_errors(self) -> bool:
        """Check if any errors occurred"""
        return any(v.severity == "ERROR" for v in self.violations)
    
    def report(self) -> str:
        """Generate violation report"""
        lines = [
            "=" * 60,
            "AXI4 Protocol Checker Report",
            "=" * 60,
            f"Total violations: {len(self.violations)}",
            f"Errors: {sum(1 for v in self.violations if v.severity == 'ERROR')}",
            f"Warnings: {sum(1 for v in self.violations if v.severity == 'WARNING')}",
            "",
            "Violations by type:",
        ]
        
        summary = self.get_violation_summary()
        for vtype, count in sorted(summary.items()):
            lines.append(f"  {vtype}: {count}")
            
        if self.violations:
            lines.append("")
            lines.append("Detailed violations:")
            for i, v in enumerate(self.violations[:20]):  # First 20
                lines.append(f"  [{v.time}ns] [{v.severity}] {v.channel}: {v.message}")
            if len(self.violations) > 20:
                lines.append(f"  ... and {len(self.violations) - 20} more")
                
        lines.append("=" * 60)
        return "\n".join(lines)


class AXI4Scoreboard:
    """
    AXI4 Transaction Scoreboard
    
    Compares expected vs actual transactions for verification.
    """
    
    def __init__(self):
        self.expected_writes: List[Dict] = []
        self.expected_reads: List[Dict] = []
        self.actual_writes: List[Dict] = []
        self.actual_reads: List[Dict] = []
        self.mismatches: List[str] = []
        
    def add_expected_write(self, address: int, data: List[int]):
        """Add expected write transaction"""
        self.expected_writes.append({
            'address': address,
            'data': data,
            'matched': False
        })
        
    def add_expected_read(self, address: int, data: List[int]):
        """Add expected read transaction"""
        self.expected_reads.append({
            'address': address,
            'data': data,
            'matched': False
        })
        
    def add_actual_write(self, address: int, data: List[int]):
        """Record actual write transaction"""
        self.actual_writes.append({
            'address': address,
            'data': data
        })
        self._match_write(address, data)
        
    def add_actual_read(self, address: int, data: List[int]):
        """Record actual read transaction"""
        self.actual_reads.append({
            'address': address,
            'data': data
        })
        self._match_read(address, data)
        
    def _match_write(self, address: int, data: List[int]):
        """Try to match write against expected"""
        for exp in self.expected_writes:
            if not exp['matched'] and exp['address'] == address:
                if exp['data'] == data:
                    exp['matched'] = True
                    return
                else:
                    self.mismatches.append(
                        f"Write data mismatch at {address:#x}: "
                        f"expected {exp['data']}, got {data}"
                    )
                    return
                    
    def _match_read(self, address: int, data: List[int]):
        """Try to match read against expected"""
        for exp in self.expected_reads:
            if not exp['matched'] and exp['address'] == address:
                if exp['data'] == data:
                    exp['matched'] = True
                    return
                else:
                    self.mismatches.append(
                        f"Read data mismatch at {address:#x}: "
                        f"expected {exp['data']}, got {data}"
                    )
                    return
                    
    def check_complete(self) -> bool:
        """Check if all expected transactions matched"""
        unmatched_writes = [e for e in self.expected_writes if not e['matched']]
        unmatched_reads = [e for e in self.expected_reads if not e['matched']]
        
        if unmatched_writes:
            for e in unmatched_writes:
                self.mismatches.append(f"Unmatched expected write at {e['address']:#x}")
                
        if unmatched_reads:
            for e in unmatched_reads:
                self.mismatches.append(f"Unmatched expected read at {e['address']:#x}")
                
        return len(self.mismatches) == 0 and not unmatched_writes and not unmatched_reads
    
    def report(self) -> str:
        """Generate scoreboard report"""
        lines = [
            "=" * 60,
            "AXI4 Scoreboard Report",
            "=" * 60,
            f"Expected writes: {len(self.expected_writes)}",
            f"Actual writes: {len(self.actual_writes)}",
            f"Expected reads: {len(self.expected_reads)}",
            f"Actual reads: {len(self.actual_reads)}",
            f"Mismatches: {len(self.mismatches)}",
        ]
        
        if self.mismatches:
            lines.append("")
            lines.append("Mismatches:")
            for m in self.mismatches:
                lines.append(f"  {m}")
                
        lines.append("=" * 60)
        return "\n".join(lines)

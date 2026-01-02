"""
AXI4 Bus Functional Model (BFM) for cocotb

Provides master and slave models for AXI4 protocol verification.
Implements full AXI4 specification per ARM IHI 0022E.

Author: Tensor Accelerator Project
Date: December 2025
"""

import cocotb
from cocotb.triggers import RisingEdge, FallingEdge, Timer, Combine, First
from cocotb.clock import Clock
from cocotb.queue import Queue
from cocotb.handle import SimHandleBase
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Any
from enum import IntEnum
import random


class AXIBurst(IntEnum):
    """AXI Burst Types"""
    FIXED = 0
    INCR = 1
    WRAP = 2


class AXISize(IntEnum):
    """AXI Transfer Size (bytes)"""
    SIZE_1 = 0    # 1 byte
    SIZE_2 = 1    # 2 bytes
    SIZE_4 = 2    # 4 bytes
    SIZE_8 = 3    # 8 bytes
    SIZE_16 = 4   # 16 bytes
    SIZE_32 = 5   # 32 bytes
    SIZE_64 = 6   # 64 bytes
    SIZE_128 = 7  # 128 bytes


class AXIResp(IntEnum):
    """AXI Response Types"""
    OKAY = 0
    EXOKAY = 1
    SLVERR = 2
    DECERR = 3


@dataclass
class AXITransaction:
    """Represents a single AXI transaction"""
    address: int
    data: List[int] = field(default_factory=list)
    length: int = 1          # Number of beats (1-256)
    size: AXISize = AXISize.SIZE_4
    burst: AXIBurst = AXIBurst.INCR
    id: int = 0
    strobe: List[int] = field(default_factory=list)  # Write strobes
    response: AXIResp = AXIResp.OKAY
    
    # Timing info (for analysis)
    start_time: int = 0
    end_time: int = 0
    
    def __post_init__(self):
        if not self.strobe:
            # Default: all bytes valid
            bytes_per_beat = 1 << self.size
            self.strobe = [((1 << bytes_per_beat) - 1)] * self.length


class AXI4Master:
    """
    AXI4 Master BFM
    
    Drives AXI4 transactions to a slave device.
    Supports read, write, burst transfers.
    """
    
    def __init__(self, dut, prefix: str, clock, reset=None, 
                 data_width: int = 32, addr_width: int = 32, id_width: int = 4):
        """
        Initialize AXI4 Master
        
        Args:
            dut: cocotb DUT handle
            prefix: Signal prefix (e.g., "m_axi" for m_axi_awaddr)
            clock: Clock signal
            reset: Optional reset signal (active low)
            data_width: Data bus width in bits
            addr_width: Address bus width in bits
            id_width: ID field width in bits
        """
        self.dut = dut
        self.prefix = prefix
        self.clock = clock
        self.reset = reset
        self.data_width = data_width
        self.addr_width = addr_width
        self.id_width = id_width
        self.bytes_per_beat = data_width // 8
        
        # Get signal handles
        self._get_signals()
        
        # Transaction queues
        self.write_queue = Queue()
        self.read_queue = Queue()
        
        # Outstanding transactions (for out-of-order completion)
        self.outstanding_writes: Dict[int, AXITransaction] = {}
        self.outstanding_reads: Dict[int, AXITransaction] = {}
        
        # Statistics
        self.stats = {
            'writes': 0,
            'reads': 0,
            'write_bytes': 0,
            'read_bytes': 0,
            'errors': 0
        }
        
        # Configuration
        self.max_outstanding = 16
        self.random_ready_delay = False
        self.max_ready_delay = 5
        
    def _get_signals(self):
        """Get AXI signal handles from DUT"""
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
        
    def _init_signals(self):
        """Initialize all output signals to default values"""
        # Write address channel
        if self.awaddr: self.awaddr.value = 0
        if self.awlen: self.awlen.value = 0
        if self.awsize: self.awsize.value = 0
        if self.awburst: self.awburst.value = 0
        if self.awid is not None: self.awid.value = 0
        if self.awvalid: self.awvalid.value = 0
        
        # Write data channel
        if self.wdata: self.wdata.value = 0
        if self.wstrb: self.wstrb.value = 0
        if self.wlast: self.wlast.value = 0
        if self.wvalid: self.wvalid.value = 0
        
        # Write response channel
        if self.bready: self.bready.value = 1
        
        # Read address channel
        if self.araddr: self.araddr.value = 0
        if self.arlen: self.arlen.value = 0
        if self.arsize: self.arsize.value = 0
        if self.arburst: self.arburst.value = 0
        if self.arid is not None: self.arid.value = 0
        if self.arvalid: self.arvalid.value = 0
        
        # Read data channel
        if self.rready: self.rready.value = 1
        
    async def reset_bus(self):
        """Reset the AXI bus"""
        self._init_signals()
        await RisingEdge(self.clock)
        
    async def write(self, address: int, data: List[int], 
                    burst: AXIBurst = AXIBurst.INCR,
                    size: AXISize = None,
                    id: int = 0) -> AXITransaction:
        """
        Perform AXI write transaction
        
        Args:
            address: Start address
            data: List of data words to write
            burst: Burst type
            size: Transfer size (auto-detect if None)
            id: Transaction ID
            
        Returns:
            AXITransaction with response
        """
        if size is None:
            size = AXISize(min(self.data_width // 8 - 1, 2).bit_length())
            
        txn = AXITransaction(
            address=address,
            data=data,
            length=len(data),
            size=size,
            burst=burst,
            id=id
        )
        
        txn.start_time = cocotb.utils.get_sim_time('ns')
        
        # Write address phase
        await self._write_address_phase(txn)
        
        # Write data phase
        await self._write_data_phase(txn)
        
        # Write response phase
        await self._write_response_phase(txn)
        
        txn.end_time = cocotb.utils.get_sim_time('ns')
        
        # Update stats
        self.stats['writes'] += 1
        self.stats['write_bytes'] += len(data) * (1 << size)
        if txn.response != AXIResp.OKAY:
            self.stats['errors'] += 1
            
        return txn
    
    async def _write_address_phase(self, txn: AXITransaction):
        """Execute write address phase"""
        self.awaddr.value = txn.address
        self.awlen.value = txn.length - 1  # AXI: 0 = 1 beat
        self.awsize.value = txn.size
        self.awburst.value = txn.burst
        if self.awid is not None:
            self.awid.value = txn.id
        self.awvalid.value = 1
        
        # Wait for ready
        while True:
            await RisingEdge(self.clock)
            if self.awready.value == 1:
                break
                
        self.awvalid.value = 0
        
    async def _write_data_phase(self, txn: AXITransaction):
        """Execute write data phase"""
        for i, word in enumerate(txn.data):
            self.wdata.value = word
            self.wstrb.value = txn.strobe[i] if i < len(txn.strobe) else ((1 << self.bytes_per_beat) - 1)
            self.wlast.value = 1 if i == len(txn.data) - 1 else 0
            self.wvalid.value = 1
            
            # Wait for ready
            while True:
                await RisingEdge(self.clock)
                if self.wready.value == 1:
                    break
                    
        self.wvalid.value = 0
        self.wlast.value = 0
        
    async def _write_response_phase(self, txn: AXITransaction):
        """Execute write response phase"""
        self.bready.value = 1
        
        # Wait for valid response
        while True:
            await RisingEdge(self.clock)
            if self.bvalid.value == 1:
                txn.response = AXIResp(int(self.bresp.value))
                break
                
    async def read(self, address: int, length: int = 1,
                   burst: AXIBurst = AXIBurst.INCR,
                   size: AXISize = None,
                   id: int = 0) -> AXITransaction:
        """
        Perform AXI read transaction
        
        Args:
            address: Start address
            length: Number of beats to read
            burst: Burst type
            size: Transfer size (auto-detect if None)
            id: Transaction ID
            
        Returns:
            AXITransaction with data and response
        """
        if size is None:
            size = AXISize(min(self.data_width // 8 - 1, 2).bit_length())
            
        txn = AXITransaction(
            address=address,
            length=length,
            size=size,
            burst=burst,
            id=id
        )
        
        txn.start_time = cocotb.utils.get_sim_time('ns')
        
        # Read address phase
        await self._read_address_phase(txn)
        
        # Read data phase
        await self._read_data_phase(txn)
        
        txn.end_time = cocotb.utils.get_sim_time('ns')
        
        # Update stats
        self.stats['reads'] += 1
        self.stats['read_bytes'] += length * (1 << size)
        if txn.response != AXIResp.OKAY:
            self.stats['errors'] += 1
            
        return txn
    
    async def _read_address_phase(self, txn: AXITransaction):
        """Execute read address phase"""
        self.araddr.value = txn.address
        self.arlen.value = txn.length - 1
        self.arsize.value = txn.size
        self.arburst.value = txn.burst
        if self.arid is not None:
            self.arid.value = txn.id
        self.arvalid.value = 1
        
        # Wait for ready
        while True:
            await RisingEdge(self.clock)
            if self.arready.value == 1:
                break
                
        self.arvalid.value = 0
        
    async def _read_data_phase(self, txn: AXITransaction):
        """Execute read data phase"""
        txn.data = []
        self.rready.value = 1
        
        for i in range(txn.length):
            # Optional random delay
            if self.random_ready_delay:
                delay = random.randint(0, self.max_ready_delay)
                for _ in range(delay):
                    self.rready.value = 0
                    await RisingEdge(self.clock)
                self.rready.value = 1
            
            # Wait for valid data
            while True:
                await RisingEdge(self.clock)
                if self.rvalid.value == 1:
                    txn.data.append(int(self.rdata.value))
                    txn.response = AXIResp(int(self.rresp.value))
                    
                    # Check RLAST on final beat
                    if i == txn.length - 1:
                        if self.rlast.value != 1:
                            cocotb.log.warning(f"RLAST not asserted on final beat")
                    break
                    
    async def write_words(self, address: int, data: List[int]) -> bool:
        """Convenience method: write 32-bit words"""
        txn = await self.write(address, data, size=AXISize.SIZE_4)
        return txn.response == AXIResp.OKAY
    
    async def read_words(self, address: int, count: int) -> List[int]:
        """Convenience method: read 32-bit words"""
        txn = await self.read(address, count, size=AXISize.SIZE_4)
        return txn.data
    
    async def write_byte(self, address: int, data: int) -> bool:
        """Write single byte"""
        txn = await self.write(address, [data], size=AXISize.SIZE_1)
        return txn.response == AXIResp.OKAY
    
    async def read_byte(self, address: int) -> int:
        """Read single byte"""
        txn = await self.read(address, 1, size=AXISize.SIZE_1)
        return txn.data[0] if txn.data else 0


class AXI4Slave:
    """
    AXI4 Slave BFM
    
    Responds to AXI4 transactions from a master.
    Includes configurable memory model.
    """
    
    def __init__(self, dut, prefix: str, clock, 
                 data_width: int = 32, addr_width: int = 32,
                 memory_size: int = 1024 * 1024):  # 1MB default
        """
        Initialize AXI4 Slave
        
        Args:
            dut: cocotb DUT handle
            prefix: Signal prefix (e.g., "s_axi")
            clock: Clock signal
            data_width: Data bus width in bits
            addr_width: Address bus width in bits
            memory_size: Size of backing memory
        """
        self.dut = dut
        self.prefix = prefix
        self.clock = clock
        self.data_width = data_width
        self.addr_width = addr_width
        self.bytes_per_beat = data_width // 8
        
        # Backing memory
        self.memory = bytearray(memory_size)
        self.memory_size = memory_size
        
        # Get signal handles
        self._get_signals()
        
        # Configuration
        self.response_delay = 0
        self.error_injection_rate = 0.0
        self.read_callback: Optional[Callable] = None
        self.write_callback: Optional[Callable] = None
        
        # Running flag
        self._running = False
        
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
        
    def _init_signals(self):
        """Initialize output signals"""
        if self.awready: self.awready.value = 0
        if self.wready: self.wready.value = 0
        if self.bvalid: self.bvalid.value = 0
        if self.bresp: self.bresp.value = 0
        if self.arready: self.arready.value = 0
        if self.rvalid: self.rvalid.value = 0
        if self.rdata: self.rdata.value = 0
        if self.rresp: self.rresp.value = 0
        if self.rlast: self.rlast.value = 0
        
    async def start(self):
        """Start slave responders"""
        self._init_signals()
        self._running = True
        
        # Start parallel handlers
        cocotb.start_soon(self._write_handler())
        cocotb.start_soon(self._read_handler())
        
    def stop(self):
        """Stop slave responders"""
        self._running = False
        
    async def _write_handler(self):
        """Handle write transactions"""
        while self._running:
            # Wait for write address
            self.awready.value = 1
            while self._running:
                await RisingEdge(self.clock)
                if self.awvalid.value == 1:
                    addr = int(self.awaddr.value)
                    length = int(self.awlen.value) + 1
                    size = int(self.awsize.value)
                    burst = int(self.awburst.value)
                    break
            
            if not self._running:
                break
                
            self.awready.value = 0
            
            # Receive write data
            self.wready.value = 1
            bytes_per_beat = 1 << size
            
            for i in range(length):
                while self._running:
                    await RisingEdge(self.clock)
                    if self.wvalid.value == 1:
                        data = int(self.wdata.value)
                        strb = int(self.wstrb.value)
                        
                        # Calculate address for this beat
                        if burst == AXIBurst.INCR:
                            beat_addr = addr + i * bytes_per_beat
                        else:
                            beat_addr = addr
                            
                        # Write to memory with strobe
                        for b in range(bytes_per_beat):
                            if strb & (1 << b):
                                if beat_addr + b < self.memory_size:
                                    self.memory[beat_addr + b] = (data >> (b * 8)) & 0xFF
                        
                        # Callback
                        if self.write_callback:
                            self.write_callback(beat_addr, data, strb)
                            
                        break
                        
            self.wready.value = 0
            
            # Send response
            await self._delay(self.response_delay)
            
            self.bvalid.value = 1
            if random.random() < self.error_injection_rate:
                self.bresp.value = AXIResp.SLVERR
            else:
                self.bresp.value = AXIResp.OKAY
                
            while self._running:
                await RisingEdge(self.clock)
                if self.bready.value == 1:
                    break
                    
            self.bvalid.value = 0
            
    async def _read_handler(self):
        """Handle read transactions"""
        while self._running:
            # Wait for read address
            self.arready.value = 1
            while self._running:
                await RisingEdge(self.clock)
                if self.arvalid.value == 1:
                    addr = int(self.araddr.value)
                    length = int(self.arlen.value) + 1
                    size = int(self.arsize.value)
                    burst = int(self.arburst.value)
                    break
                    
            if not self._running:
                break
                
            self.arready.value = 0
            
            # Send read data
            bytes_per_beat = 1 << size
            
            for i in range(length):
                await self._delay(self.response_delay)
                
                # Calculate address for this beat
                if burst == AXIBurst.INCR:
                    beat_addr = addr + i * bytes_per_beat
                else:
                    beat_addr = addr
                    
                # Read from memory
                data = 0
                for b in range(bytes_per_beat):
                    if beat_addr + b < self.memory_size:
                        data |= self.memory[beat_addr + b] << (b * 8)
                        
                # Callback
                if self.read_callback:
                    data = self.read_callback(beat_addr, data)
                    
                self.rdata.value = data
                self.rlast.value = 1 if i == length - 1 else 0
                
                if random.random() < self.error_injection_rate:
                    self.rresp.value = AXIResp.SLVERR
                else:
                    self.rresp.value = AXIResp.OKAY
                    
                self.rvalid.value = 1
                
                while self._running:
                    await RisingEdge(self.clock)
                    if self.rready.value == 1:
                        break
                        
            self.rvalid.value = 0
            self.rlast.value = 0
            
    async def _delay(self, cycles: int):
        """Wait for specified clock cycles"""
        for _ in range(cycles):
            await RisingEdge(self.clock)
            
    def write_memory(self, address: int, data: bytes):
        """Direct memory write"""
        for i, byte in enumerate(data):
            if address + i < self.memory_size:
                self.memory[address + i] = byte
                
    def read_memory(self, address: int, length: int) -> bytes:
        """Direct memory read"""
        return bytes(self.memory[address:address + length])
    
    def fill_memory(self, value: int = 0):
        """Fill memory with value"""
        self.memory = bytearray([value] * self.memory_size)


class AXI4Monitor:
    """
    AXI4 Protocol Monitor
    
    Passively monitors AXI transactions for analysis and coverage.
    """
    
    def __init__(self, dut, prefix: str, clock, data_width: int = 32):
        self.dut = dut
        self.prefix = prefix
        self.clock = clock
        self.data_width = data_width
        
        self._get_signals()
        
        # Transaction logs
        self.write_transactions: List[AXITransaction] = []
        self.read_transactions: List[AXITransaction] = []
        
        # Callbacks
        self.on_write: Optional[Callable[[AXITransaction], None]] = None
        self.on_read: Optional[Callable[[AXITransaction], None]] = None
        
        self._running = False
        
    def _get_signals(self):
        """Get signal handles for monitoring"""
        p = self.prefix
        
        # Write channels
        self.awaddr = getattr(self.dut, f"{p}_awaddr", None)
        self.awlen = getattr(self.dut, f"{p}_awlen", None)
        self.awvalid = getattr(self.dut, f"{p}_awvalid", None)
        self.awready = getattr(self.dut, f"{p}_awready", None)
        self.wdata = getattr(self.dut, f"{p}_wdata", None)
        self.wlast = getattr(self.dut, f"{p}_wlast", None)
        self.wvalid = getattr(self.dut, f"{p}_wvalid", None)
        self.wready = getattr(self.dut, f"{p}_wready", None)
        self.bresp = getattr(self.dut, f"{p}_bresp", None)
        self.bvalid = getattr(self.dut, f"{p}_bvalid", None)
        self.bready = getattr(self.dut, f"{p}_bready", None)
        
        # Read channels
        self.araddr = getattr(self.dut, f"{p}_araddr", None)
        self.arlen = getattr(self.dut, f"{p}_arlen", None)
        self.arvalid = getattr(self.dut, f"{p}_arvalid", None)
        self.arready = getattr(self.dut, f"{p}_arready", None)
        self.rdata = getattr(self.dut, f"{p}_rdata", None)
        self.rlast = getattr(self.dut, f"{p}_rlast", None)
        self.rvalid = getattr(self.dut, f"{p}_rvalid", None)
        self.rready = getattr(self.dut, f"{p}_rready", None)
        
    async def start(self):
        """Start monitoring"""
        self._running = True
        cocotb.start_soon(self._monitor_writes())
        cocotb.start_soon(self._monitor_reads())
        
    def stop(self):
        """Stop monitoring"""
        self._running = False
        
    async def _monitor_writes(self):
        """Monitor write transactions"""
        while self._running:
            await RisingEdge(self.clock)
            
            # Detect write address handshake
            if self.awvalid.value == 1 and self.awready.value == 1:
                txn = AXITransaction(
                    address=int(self.awaddr.value),
                    length=int(self.awlen.value) + 1
                )
                txn.start_time = cocotb.utils.get_sim_time('ns')
                
                # Collect write data
                while len(txn.data) < txn.length:
                    await RisingEdge(self.clock)
                    if self.wvalid.value == 1 and self.wready.value == 1:
                        txn.data.append(int(self.wdata.value))
                        
                # Wait for response
                while True:
                    await RisingEdge(self.clock)
                    if self.bvalid.value == 1 and self.bready.value == 1:
                        txn.response = AXIResp(int(self.bresp.value))
                        break
                        
                txn.end_time = cocotb.utils.get_sim_time('ns')
                self.write_transactions.append(txn)
                
                if self.on_write:
                    self.on_write(txn)
                    
    async def _monitor_reads(self):
        """Monitor read transactions"""
        while self._running:
            await RisingEdge(self.clock)
            
            # Detect read address handshake
            if self.arvalid.value == 1 and self.arready.value == 1:
                txn = AXITransaction(
                    address=int(self.araddr.value),
                    length=int(self.arlen.value) + 1
                )
                txn.start_time = cocotb.utils.get_sim_time('ns')
                
                # Collect read data
                while len(txn.data) < txn.length:
                    await RisingEdge(self.clock)
                    if self.rvalid.value == 1 and self.rready.value == 1:
                        txn.data.append(int(self.rdata.value))
                        
                txn.end_time = cocotb.utils.get_sim_time('ns')
                self.read_transactions.append(txn)
                
                if self.on_read:
                    self.on_read(txn)

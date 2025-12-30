#!/usr/bin/env python3
"""
Cycle-Accurate DMA Engine Model

Matches RTL behavior exactly using proper registered outputs.
"""

from dataclasses import dataclass
from enum import IntEnum
import numpy as np


class DMAState(IntEnum):
    IDLE = 0
    DECODE = 1
    LOAD_ADDR = 2
    LOAD_DATA = 3
    LOAD_WRITE = 4
    STORE_REQ = 5
    STORE_WAIT = 6
    STORE_CAP = 13
    STORE_ADDR = 7
    STORE_DATA = 8
    STORE_RESP = 9
    NEXT_COL = 10
    NEXT_ROW = 11
    DONE = 12


class DMAOp(IntEnum):
    LOAD = 0x01
    STORE = 0x02


@dataclass
class DMACommand:
    subop: int = 0
    ext_addr: int = 0
    int_addr: int = 0
    rows: int = 1
    cols: int = 1
    ext_stride: int = 32
    int_stride: int = 32


class DMAEngine:
    """Cycle-accurate DMA engine model with proper registered outputs."""
    
    BYTES_PER_WORD = 32
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.reset()
    
    def reset(self):
        # State
        self.state = DMAState.IDLE
        self.state_next = DMAState.IDLE
        
        # Registers
        self.cmd_reg = DMACommand()
        self.op_type = 0
        self.row_count = 0
        self.col_count = 0
        self.ext_ptr = 0
        self.int_ptr = 0
        self.data_buf = 0
        
        # Registered outputs (directly driven by state)
        self.sram_addr = 0
        self.sram_wdata = 0
        self.sram_we = False
        self.sram_re = False
        self.axi_araddr = 0
        self.axi_arvalid = False
        self.axi_rready = False
        self.axi_awaddr = 0
        self.axi_awvalid = False
        self.axi_wdata = 0
        self.axi_wvalid = False
        self.axi_wlast = False
        self.cmd_ready = True
        self.cmd_done = False
        
        self.cycle = 0
    
    def log(self, msg: str):
        if self.verbose:
            print(f"[DMA @{self.cycle:3d}] {self.state.name:12s} | {msg}")
    
    def posedge(self, cmd_valid=False, cmd=None, sram_rdata=0, sram_ready=True,
                axi_arready=True, axi_rvalid=False, axi_rdata=0,
                axi_awready=True, axi_wready=True, axi_bvalid=False):
        """Execute one clock cycle using registered outputs."""
        self.cycle += 1
        
        # Update state from previous cycle's next state
        self.state = self.state_next
        
        # Default: clear single-cycle pulses
        self.sram_we = False
        self.sram_re = False
        self.cmd_done = False
        
        # Combinational: compute next state and outputs based on current state
        if self.state == DMAState.IDLE:
            self.cmd_ready = True
            self.axi_arvalid = False
            self.axi_awvalid = False
            self.axi_wvalid = False
            self.axi_rready = False
            
            if cmd_valid and cmd:
                self.cmd_reg = cmd
                self.op_type = cmd.subop
                self.ext_ptr = cmd.ext_addr
                self.int_ptr = cmd.int_addr
                self.row_count = 0
                self.col_count = 0
                self.log(f"CMD: {'LOAD' if cmd.subop == DMAOp.LOAD else 'STORE'}")
                self.state_next = DMAState.DECODE
                self.cmd_ready = False
            else:
                self.state_next = DMAState.IDLE
        
        elif self.state == DMAState.DECODE:
            if self.op_type == DMAOp.LOAD:
                self.state_next = DMAState.LOAD_ADDR
            else:
                self.state_next = DMAState.STORE_REQ
        
        # ===== LOAD Path =====
        elif self.state == DMAState.LOAD_ADDR:
            self.axi_araddr = self.ext_ptr
            self.axi_arvalid = True
            self.axi_rready = False
            
            if axi_arready:
                self.log(f"AR addr=0x{self.ext_ptr:x}")
                self.state_next = DMAState.LOAD_DATA
            else:
                self.state_next = DMAState.LOAD_ADDR
        
        elif self.state == DMAState.LOAD_DATA:
            self.axi_arvalid = False
            self.axi_rready = True
            
            if axi_rvalid:
                self.data_buf = axi_rdata
                self.log(f"R data=0x{axi_rdata:x}")
                self.state_next = DMAState.LOAD_WRITE
            else:
                self.state_next = DMAState.LOAD_DATA
        
        elif self.state == DMAState.LOAD_WRITE:
            self.axi_rready = False
            self.sram_addr = self.int_ptr
            self.sram_wdata = self.data_buf
            self.sram_we = True
            
            if sram_ready:
                self.log(f"SRAM WR addr=0x{self.int_ptr:x}")
                self.state_next = DMAState.NEXT_COL
            else:
                self.state_next = DMAState.LOAD_WRITE
        
        # ===== STORE Path =====
        elif self.state == DMAState.STORE_REQ:
            self.sram_addr = self.int_ptr
            self.sram_re = True
            
            if sram_ready:
                self.log(f"SRAM RD addr=0x{self.int_ptr:x}")
                self.state_next = DMAState.STORE_WAIT
            else:
                self.state_next = DMAState.STORE_REQ
        
        elif self.state == DMAState.STORE_WAIT:
            # Wait 1 cycle for SRAM latency
            self.state_next = DMAState.STORE_CAP
        
        elif self.state == DMAState.STORE_CAP:
            # Capture SRAM read data
            self.data_buf = sram_rdata
            self.log(f"Captured=0x{sram_rdata:x}")
            self.state_next = DMAState.STORE_ADDR
        
        elif self.state == DMAState.STORE_ADDR:
            self.axi_awaddr = self.ext_ptr
            self.axi_awvalid = True
            
            if axi_awready:
                self.log(f"AW addr=0x{self.ext_ptr:x}")
                self.state_next = DMAState.STORE_DATA
            else:
                self.state_next = DMAState.STORE_ADDR
        
        elif self.state == DMAState.STORE_DATA:
            self.axi_awvalid = False
            self.axi_wdata = self.data_buf
            self.axi_wlast = True
            self.axi_wvalid = True
            
            if axi_wready:
                self.log(f"W data=0x{self.data_buf:x}")
                self.state_next = DMAState.STORE_RESP
            else:
                self.state_next = DMAState.STORE_DATA
        
        elif self.state == DMAState.STORE_RESP:
            self.axi_wvalid = False
            self.axi_wlast = False
            
            if axi_bvalid:
                self.log("B response")
                self.state_next = DMAState.NEXT_COL
            else:
                self.state_next = DMAState.STORE_RESP
        
        # ===== Column/Row Advancement =====
        elif self.state == DMAState.NEXT_COL:
            self.col_count += 1
            self.ext_ptr += self.BYTES_PER_WORD
            self.int_ptr += self.BYTES_PER_WORD
            
            if self.col_count >= self.cmd_reg.cols:
                self.state_next = DMAState.NEXT_ROW
            elif self.op_type == DMAOp.LOAD:
                self.state_next = DMAState.LOAD_ADDR
            else:
                self.state_next = DMAState.STORE_REQ
        
        elif self.state == DMAState.NEXT_ROW:
            self.row_count += 1
            self.col_count = 0
            
            if self.row_count >= self.cmd_reg.rows:
                self.state_next = DMAState.DONE
            else:
                self.ext_ptr = self.cmd_reg.ext_addr + self.row_count * self.cmd_reg.ext_stride
                self.int_ptr = self.cmd_reg.int_addr + self.row_count * self.cmd_reg.int_stride
                if self.op_type == DMAOp.LOAD:
                    self.state_next = DMAState.LOAD_ADDR
                else:
                    self.state_next = DMAState.STORE_REQ
        
        elif self.state == DMAState.DONE:
            self.cmd_done = True
            self.cmd_ready = True
            self.log("DONE")
            self.state_next = DMAState.IDLE


class AXIMemory:
    """Simple AXI memory model."""
    
    def __init__(self, size_words=1024, read_latency=2, verbose=False):
        self.mem = np.zeros(size_words, dtype=np.uint64)
        self.read_latency = read_latency
        self.verbose = verbose
        
        self.arready = True
        self.awready = True
        self.wready = True
        
        self.r_state = 'IDLE'
        self.r_addr = 0
        self.r_delay = 0
        self.rvalid = False
        self.rdata = 0
        
        self.w_state = 'IDLE'
        self.w_addr = 0
        self.bvalid = False
        self.cycle = 0
    
    def word_addr(self, byte_addr):
        return (byte_addr >> 5) % len(self.mem)
    
    def posedge(self, arvalid=False, araddr=0, rready=False,
                awvalid=False, awaddr=0, wvalid=False, wdata=0, wlast=False, bready=True):
        self.cycle += 1
        
        # Read channel
        if self.r_state == 'IDLE':
            self.rvalid = False
            if arvalid:
                self.r_addr = araddr
                self.r_delay = self.read_latency
                self.r_state = 'DELAY'
                if self.verbose:
                    print(f"[AXI @{self.cycle}] AR addr=0x{araddr:x}")
        
        elif self.r_state == 'DELAY':
            self.rvalid = False
            self.r_delay -= 1
            if self.r_delay <= 0:
                self.r_state = 'DATA'
        
        elif self.r_state == 'DATA':
            self.rdata = int(self.mem[self.word_addr(self.r_addr)])
            self.rvalid = True
            if rready:
                if self.verbose:
                    print(f"[AXI @{self.cycle}] R data=0x{self.rdata:x}")
                self.r_state = 'IDLE'
        
        # Write channel
        if self.w_state == 'IDLE':
            self.bvalid = False
            if awvalid:
                self.w_addr = awaddr
                self.w_state = 'DATA'
                if self.verbose:
                    print(f"[AXI @{self.cycle}] AW addr=0x{awaddr:x}")
        
        elif self.w_state == 'DATA':
            if wvalid:
                word = self.word_addr(self.w_addr)
                self.mem[word] = wdata & 0xFFFFFFFFFFFFFFFF
                if self.verbose:
                    print(f"[AXI @{self.cycle}] W data=0x{wdata:x}")
                if wlast:
                    self.w_state = 'RESP'
        
        elif self.w_state == 'RESP':
            self.bvalid = True
            if bready:
                self.w_state = 'IDLE'


class SRAMModel:
    """SRAM model with 1-cycle read latency."""
    
    def __init__(self, num_words=256, verbose=False):
        self.mem = np.zeros(num_words, dtype=np.uint64)
        self.verbose = verbose
        self.rdata = 0
        self._rdata_pipe = 0
        self.cycle = 0
    
    def word_addr(self, byte_addr):
        return (byte_addr >> 5) % len(self.mem)
    
    def posedge(self, addr=0, wdata=0, we=False, re=False):
        self.cycle += 1
        
        # Registered read output (1-cycle latency)
        self.rdata = self._rdata_pipe
        
        word = self.word_addr(addr)
        if we:
            self.mem[word] = wdata & 0xFFFFFFFFFFFFFFFF
            if self.verbose:
                print(f"[SRAM @{self.cycle}] WR [{word}]=0x{wdata:x}")
        if re:
            self._rdata_pipe = int(self.mem[word])
            if self.verbose:
                print(f"[SRAM @{self.cycle}] RD [{word}] (pipe=0x{self._rdata_pipe:x})")


def run_cycles(dma, sram, axi, cmd, max_cycles=100):
    """Run DMA command to completion."""
    first = True
    for _ in range(max_cycles):
        # All components see signals from previous cycle
        sram.posedge(addr=dma.sram_addr, wdata=dma.sram_wdata,
                     we=dma.sram_we, re=dma.sram_re)
        axi.posedge(arvalid=dma.axi_arvalid, araddr=dma.axi_araddr, rready=dma.axi_rready,
                    awvalid=dma.axi_awvalid, awaddr=dma.axi_awaddr,
                    wvalid=dma.axi_wvalid, wdata=dma.axi_wdata, wlast=dma.axi_wlast)
        dma.posedge(
            cmd_valid=first, cmd=cmd if first else None,
            sram_rdata=sram.rdata, sram_ready=True,
            axi_arready=axi.arready, axi_rvalid=axi.rvalid, axi_rdata=axi.rdata,
            axi_awready=axi.awready, axi_wready=axi.wready, axi_bvalid=axi.bvalid
        )
        first = False
        if dma.cmd_done:
            return True
    return False


def test_dma():
    print("=" * 60)
    print("DMA ENGINE MODEL TEST")
    print("=" * 60)
    
    dma = DMAEngine(verbose=True)
    sram = SRAMModel(num_words=256, verbose=False)
    axi = AXIMemory(size_words=1024, read_latency=2, verbose=False)
    
    # Initialize AXI memory
    for i in range(16):
        axi.mem[i] = 0xDEAD0000 + i
    
    # TEST 1: LOAD
    print("\n--- TEST 1: DMA LOAD (1 word) ---")
    cmd = DMACommand(subop=DMAOp.LOAD, ext_addr=0x00, int_addr=0x100, rows=1, cols=1)
    assert run_cycles(dma, sram, axi, cmd), "Timeout!"
    
    expected, actual = 0xDEAD0000, int(sram.mem[sram.word_addr(0x100)])
    print(f"Result: SRAM[0x100]=0x{actual:x} (expected 0x{expected:x})")
    assert actual == expected
    print(">>> LOAD TEST PASSED <<<\n")
    
    # TEST 2: STORE
    print("--- TEST 2: DMA STORE (1 word) ---")
    sram.mem[sram.word_addr(0x200)] = 0xBEEF1234
    cmd = DMACommand(subop=DMAOp.STORE, ext_addr=0x400, int_addr=0x200, rows=1, cols=1)
    assert run_cycles(dma, sram, axi, cmd), "Timeout!"
    
    expected, actual = 0xBEEF1234, int(axi.mem[axi.word_addr(0x400)])
    print(f"Result: AXI[0x400]=0x{actual:x} (expected 0x{expected:x})")
    assert actual == expected
    print(">>> STORE TEST PASSED <<<\n")
    
    # TEST 3: Multi-word LOAD
    print("--- TEST 3: DMA LOAD (4 words) ---")
    cmd = DMACommand(subop=DMAOp.LOAD, ext_addr=0x00, int_addr=0x300, rows=1, cols=4)
    assert run_cycles(dma, sram, axi, cmd), "Timeout!"
    
    for i in range(4):
        expected = 0xDEAD0000 + i
        actual = int(sram.mem[sram.word_addr(0x300 + i*32)])
        assert actual == expected, f"Word {i}: got 0x{actual:x}"
    print(">>> 4-WORD LOAD PASSED <<<\n")
    
    # TEST 4: 2D transfer
    print("--- TEST 4: 2D LOAD (2x2) ---")
    axi.mem[0] = 0xAA000000
    axi.mem[1] = 0xAA000001
    axi.mem[2] = 0xAA000010
    axi.mem[3] = 0xAA000011
    cmd = DMACommand(subop=DMAOp.LOAD, ext_addr=0x00, int_addr=0x400, 
                     rows=2, cols=2, ext_stride=64, int_stride=64)
    assert run_cycles(dma, sram, axi, cmd), "Timeout!"
    
    assert int(sram.mem[sram.word_addr(0x400)]) == 0xAA000000
    assert int(sram.mem[sram.word_addr(0x420)]) == 0xAA000001
    assert int(sram.mem[sram.word_addr(0x440)]) == 0xAA000010
    assert int(sram.mem[sram.word_addr(0x460)]) == 0xAA000011
    print(">>> 2D LOAD PASSED <<<\n")
    
    print("=" * 60)
    print("ALL DMA MODEL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_dma()

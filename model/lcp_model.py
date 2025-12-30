#!/usr/bin/env python3
"""
Cycle-Accurate Local Command Processor (LCP) Model

Decodes instructions and dispatches to functional units:
- MXU (Matrix Unit)
- VPU (Vector Unit)  
- DMA (Data Movement)

Handles:
- Instruction fetch
- Operand decoding
- Unit dispatch and synchronization
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple
import numpy as np


class Opcode(IntEnum):
    """Instruction opcodes - matches RTL"""
    NOP = 0x00
    TENSOR = 0x01    # MXU operation
    VECTOR = 0x02    # VPU operation
    DMA = 0x03       # Memory transfer
    SYNC = 0x04      # Wait for unit
    LOOP = 0x05      # Begin loop
    ENDLOOP = 0x06   # End loop
    BARRIER = 0x07   # Global sync
    HALT = 0xFF      # Stop execution


class LCPState(IntEnum):
    IDLE = 0
    FETCH = 1
    DECODE = 2
    DISPATCH_MXU = 3
    DISPATCH_VPU = 4
    DISPATCH_DMA = 5
    WAIT_MXU = 6
    WAIT_VPU = 7
    WAIT_DMA = 8
    SYNC = 9
    HALTED = 10


@dataclass
class Instruction:
    """Decoded instruction"""
    opcode: int = 0
    subop: int = 0
    operands: Tuple[int, ...] = (0, 0, 0, 0, 0, 0)
    raw: int = 0
    
    @classmethod
    def from_int(cls, raw: int) -> 'Instruction':
        return cls(
            opcode=(raw >> 120) & 0xFF,
            subop=(raw >> 112) & 0xFF,
            operands=(
                (raw >> 92) & 0xFFFFF,   # op0: 20 bits
                (raw >> 72) & 0xFFFFF,   # op1: 20 bits
                (raw >> 52) & 0xFFFFF,   # op2: 20 bits
                (raw >> 36) & 0xFFFF,    # op3: 16 bits
                (raw >> 20) & 0xFFFF,    # op4: 16 bits
                (raw >> 4) & 0xFFFF,     # op5: 16 bits
            ),
            raw=raw
        )


class LocalCommandProcessor:
    """
    Cycle-accurate LCP model.
    
    Fetches instructions from IMEM, decodes, and dispatches to units.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.reset()
    
    def reset(self):
        self.state = LCPState.IDLE
        self.state_next = LCPState.IDLE
        
        self.pc = 0  # Program counter
        self.instr = Instruction()
        
        # Loop stack (simple: one level)
        self.loop_start = 0
        self.loop_count = 0
        self.loop_iter = 0
        
        # Unit commands
        self.mxu_cmd = 0
        self.mxu_valid = False
        self.vpu_cmd = 0
        self.vpu_valid = False
        self.dma_cmd = 0
        self.dma_valid = False
        
        # Status
        self.running = False
        self.halted = False
        
        # IMEM interface
        self.imem_addr = 0
        self.imem_re = False
        
        self.cycle = 0
    
    def log(self, msg: str):
        if self.verbose:
            print(f"[LCP @{self.cycle:3d}] {self.state.name:12s} PC={self.pc:3d} | {msg}")
    
    def start(self, start_pc: int = 0):
        """Start execution from given PC"""
        self.pc = start_pc
        self.running = True
        self.halted = False
        self.state_next = LCPState.FETCH
    
    def posedge(self, imem_rdata: int = 0, imem_valid: bool = False,
                mxu_ready: bool = True, mxu_done: bool = False,
                vpu_ready: bool = True, vpu_done: bool = False,
                dma_ready: bool = True, dma_done: bool = False,
                barrier_grant: bool = True):
        """Execute one clock cycle."""
        self.cycle += 1
        self.state = self.state_next
        
        # Clear command valids
        self.mxu_valid = False
        self.vpu_valid = False
        self.dma_valid = False
        self.imem_re = False
        
        if self.state == LCPState.IDLE:
            if self.running and not self.halted:
                self.state_next = LCPState.FETCH
            else:
                self.state_next = LCPState.IDLE
        
        elif self.state == LCPState.FETCH:
            self.imem_addr = self.pc
            self.imem_re = True
            self.log("Fetching")
            self.state_next = LCPState.DECODE
        
        elif self.state == LCPState.DECODE:
            if imem_valid:
                self.instr = Instruction.from_int(imem_rdata)
                self.log(f"Decoded: op=0x{self.instr.opcode:02x} subop=0x{self.instr.subop:02x}")
                
                op = self.instr.opcode
                if op == Opcode.NOP:
                    self.pc += 1
                    self.state_next = LCPState.FETCH
                elif op == Opcode.TENSOR:
                    self.state_next = LCPState.DISPATCH_MXU
                elif op == Opcode.VECTOR:
                    self.state_next = LCPState.DISPATCH_VPU
                elif op == Opcode.DMA:
                    self.state_next = LCPState.DISPATCH_DMA
                elif op == Opcode.SYNC:
                    self.state_next = LCPState.SYNC
                elif op == Opcode.LOOP:
                    self.loop_start = self.pc + 1
                    self.loop_count = self.instr.operands[0]
                    self.loop_iter = 0
                    self.log(f"LOOP start, count={self.loop_count}")
                    self.pc += 1
                    self.state_next = LCPState.FETCH
                elif op == Opcode.ENDLOOP:
                    self.loop_iter += 1
                    if self.loop_iter < self.loop_count:
                        self.pc = self.loop_start
                        self.log(f"ENDLOOP iter={self.loop_iter}/{self.loop_count}")
                    else:
                        self.pc += 1
                        self.log("ENDLOOP done")
                    self.state_next = LCPState.FETCH
                elif op == Opcode.BARRIER:
                    self.log("BARRIER")
                    self.state_next = LCPState.SYNC
                elif op == Opcode.HALT:
                    self.log("HALT")
                    self.halted = True
                    self.running = False
                    self.state_next = LCPState.HALTED
                else:
                    self.log(f"Unknown opcode 0x{op:02x}")
                    self.pc += 1
                    self.state_next = LCPState.FETCH
            else:
                self.state_next = LCPState.DECODE
        
        elif self.state == LCPState.DISPATCH_MXU:
            if mxu_ready:
                self.mxu_cmd = self.instr.raw
                self.mxu_valid = True
                self.log("Dispatch MXU")
                self.state_next = LCPState.WAIT_MXU
            else:
                self.state_next = LCPState.DISPATCH_MXU
        
        elif self.state == LCPState.DISPATCH_VPU:
            if vpu_ready:
                self.vpu_cmd = self.instr.raw
                self.vpu_valid = True
                self.log("Dispatch VPU")
                self.state_next = LCPState.WAIT_VPU
            else:
                self.state_next = LCPState.DISPATCH_VPU
        
        elif self.state == LCPState.DISPATCH_DMA:
            if dma_ready:
                self.dma_cmd = self.instr.raw
                self.dma_valid = True
                self.log("Dispatch DMA")
                self.state_next = LCPState.WAIT_DMA
            else:
                self.state_next = LCPState.DISPATCH_DMA
        
        elif self.state == LCPState.WAIT_MXU:
            if mxu_done:
                self.log("MXU done")
                self.pc += 1
                self.state_next = LCPState.FETCH
            else:
                self.state_next = LCPState.WAIT_MXU
        
        elif self.state == LCPState.WAIT_VPU:
            if vpu_done:
                self.log("VPU done")
                self.pc += 1
                self.state_next = LCPState.FETCH
            else:
                self.state_next = LCPState.WAIT_VPU
        
        elif self.state == LCPState.WAIT_DMA:
            if dma_done:
                self.log("DMA done")
                self.pc += 1
                self.state_next = LCPState.FETCH
            else:
                self.state_next = LCPState.WAIT_DMA
        
        elif self.state == LCPState.SYNC:
            # Wait for specified unit or barrier
            sync_unit = self.instr.subop
            done = False
            if sync_unit == 0x01:  # MXU
                done = mxu_done or mxu_ready
            elif sync_unit == 0x02:  # VPU
                done = vpu_done or vpu_ready
            elif sync_unit == 0x03:  # DMA
                done = dma_done or dma_ready
            else:  # Barrier
                done = barrier_grant
            
            if done:
                self.log("SYNC complete")
                self.pc += 1
                self.state_next = LCPState.FETCH
            else:
                self.state_next = LCPState.SYNC
        
        elif self.state == LCPState.HALTED:
            self.state_next = LCPState.HALTED


class IMEMModel:
    """Simple instruction memory model"""
    
    def __init__(self):
        self.mem = {}  # Sparse: PC -> 128-bit instruction
        self.rdata = 0
        self.valid = False
        self._pending_addr = None
    
    def posedge(self, addr: int = 0, re: bool = False):
        if self._pending_addr is not None:
            self.rdata = self.mem.get(self._pending_addr, 0)
            self.valid = True
            self._pending_addr = None
        else:
            self.valid = False
        
        if re:
            self._pending_addr = addr


def make_instr(opcode: int, subop: int = 0, ops: Tuple[int, ...] = (0,0,0,0,0,0)) -> int:
    """Build 128-bit instruction"""
    return ((opcode & 0xFF) << 120 |
            (subop & 0xFF) << 112 |
            (ops[0] & 0xFFFFF) << 92 |
            (ops[1] & 0xFFFFF) << 72 |
            (ops[2] & 0xFFFFF) << 52 |
            (ops[3] & 0xFFFF) << 36 |
            (ops[4] & 0xFFFF) << 20 |
            (ops[5] & 0xFFFF) << 4)


def test_lcp():
    print("=" * 60)
    print("LCP MODEL TEST")
    print("=" * 60)
    
    lcp = LocalCommandProcessor(verbose=True)
    imem = IMEMModel()
    
    # Simple program: NOP, TENSOR, HALT
    imem.mem[0] = make_instr(Opcode.NOP)
    imem.mem[1] = make_instr(Opcode.TENSOR, 0x01)
    imem.mem[2] = make_instr(Opcode.HALT)
    
    print("\n--- TEST 1: Simple Program ---")
    lcp.start(0)
    
    mxu_busy = False
    mxu_done_cycle = 0
    
    for cycle in range(50):
        imem.posedge(addr=lcp.imem_addr, re=lcp.imem_re)
        
        # Simulate MXU taking 3 cycles
        mxu_done = False
        if lcp.mxu_valid:
            mxu_busy = True
            mxu_done_cycle = cycle + 3
        if mxu_busy and cycle >= mxu_done_cycle:
            mxu_done = True
            mxu_busy = False
        
        lcp.posedge(imem_rdata=imem.rdata, imem_valid=imem.valid,
                    mxu_ready=not mxu_busy, mxu_done=mxu_done)
        
        if lcp.halted:
            break
    
    assert lcp.halted, "Should be halted"
    print(">>> SIMPLE PROGRAM PASSED <<<\n")
    
    # Test with loop
    print("--- TEST 2: Loop Program ---")
    lcp.reset()
    imem.mem.clear()
    
    # LOOP 3, NOP, ENDLOOP, HALT
    imem.mem[0] = make_instr(Opcode.LOOP, 0, (3,0,0,0,0,0))
    imem.mem[1] = make_instr(Opcode.NOP)
    imem.mem[2] = make_instr(Opcode.ENDLOOP)
    imem.mem[3] = make_instr(Opcode.HALT)
    
    lcp.start(0)
    nop_count = 0
    
    for _ in range(100):
        imem.posedge(addr=lcp.imem_addr, re=lcp.imem_re)
        lcp.posedge(imem_rdata=imem.rdata, imem_valid=imem.valid)
        
        # Count NOP executions
        if lcp.state == LCPState.DECODE and imem.valid:
            instr = Instruction.from_int(imem.rdata)
            if instr.opcode == Opcode.NOP:
                nop_count += 1
        
        if lcp.halted:
            break
    
    print(f"NOP executed {nop_count} times")
    assert nop_count == 3, f"Expected 3 NOPs, got {nop_count}"
    assert lcp.halted
    print(">>> LOOP PROGRAM PASSED <<<\n")
    
    print("=" * 60)
    print("ALL LCP TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_lcp()

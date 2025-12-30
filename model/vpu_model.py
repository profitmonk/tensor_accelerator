#!/usr/bin/env python3
"""
Cycle-Accurate Vector Processing Unit (VPU) Model
"""

from dataclasses import dataclass
from enum import IntEnum
import numpy as np


class VPUOp(IntEnum):
    NOP = 0x00
    RELU = 0x01
    ADD = 0x02
    SUB = 0x04
    MAX = 0x05
    MIN = 0x06
    ABS = 0x07
    SUM_REDUCE = 0x10
    MAX_REDUCE = 0x11
    COPY = 0x30


class VPUState(IntEnum):
    IDLE = 0
    FETCH_A = 1
    WAIT_A1 = 2
    WAIT_A2 = 3
    FETCH_B = 4
    WAIT_B1 = 5
    WAIT_B2 = 6
    COMPUTE = 7
    WRITE = 8
    NEXT = 9
    DONE = 10


@dataclass 
class VPUCommand:
    opcode: int = VPUOp.NOP
    src_a_addr: int = 0
    src_b_addr: int = 0
    dst_addr: int = 0
    length: int = 1


class VectorUnit:
    ELEMENTS_PER_WORD = 32
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.reset()
    
    def reset(self):
        self.state = VPUState.IDLE
        self.state_next = VPUState.IDLE
        self.cmd_reg = VPUCommand()
        self.elem_count = 0
        self.accumulator = 0
        self.data_a = np.zeros(self.ELEMENTS_PER_WORD, dtype=np.int8)
        self.data_b = np.zeros(self.ELEMENTS_PER_WORD, dtype=np.int8)
        self.data_out = np.zeros(self.ELEMENTS_PER_WORD, dtype=np.int8)
        self.sram_addr = 0
        self.sram_wdata = 0
        self.sram_we = False
        self.sram_re = False
        self.cmd_ready = True
        self.cmd_done = False
        self.cycle = 0
    
    def log(self, msg):
        if self.verbose:
            print(f"[VPU @{self.cycle:3d}] {self.state.name:10s} | {msg}")
    
    def _pack(self, arr):
        result = 0
        for i, val in enumerate(arr):
            result |= (int(val) & 0xFF) << (i * 8)
        return result
    
    def _unpack(self, val):
        arr = np.zeros(self.ELEMENTS_PER_WORD, dtype=np.int8)
        for i in range(self.ELEMENTS_PER_WORD):
            b = (val >> (i * 8)) & 0xFF
            arr[i] = np.int8(b if b < 128 else b - 256)
        return arr
    
    def _needs_b(self, op):
        return op in (VPUOp.ADD, VPUOp.SUB, VPUOp.MAX, VPUOp.MIN)
    
    def _is_reduce(self, op):
        return op in (VPUOp.SUM_REDUCE, VPUOp.MAX_REDUCE)
    
    def _compute(self, op, a, b):
        if op == VPUOp.RELU:
            return np.maximum(a, 0).astype(np.int8)
        elif op == VPUOp.ADD:
            return np.clip(a.astype(np.int16) + b.astype(np.int16), -128, 127).astype(np.int8)
        elif op == VPUOp.SUB:
            return np.clip(a.astype(np.int16) - b.astype(np.int16), -128, 127).astype(np.int8)
        elif op == VPUOp.MAX:
            return np.maximum(a, b)
        elif op == VPUOp.MIN:
            return np.minimum(a, b)
        elif op == VPUOp.ABS:
            return np.abs(a).astype(np.int8)
        elif op == VPUOp.COPY:
            return a.copy()
        return a
    
    def posedge(self, cmd_valid=False, cmd=None, sram_rdata=0, sram_ready=True):
        self.cycle += 1
        self.state = self.state_next
        self.sram_we = False
        self.sram_re = False
        self.cmd_done = False
        
        if self.state == VPUState.IDLE:
            self.cmd_ready = True
            if cmd_valid and cmd:
                self.cmd_reg = cmd
                self.elem_count = 0
                self.accumulator = 0
                self.log(f"CMD: {VPUOp(cmd.opcode).name}")
                self.state_next = VPUState.FETCH_A
                self.cmd_ready = False
            else:
                self.state_next = VPUState.IDLE
        
        elif self.state == VPUState.FETCH_A:
            self.sram_addr = self.cmd_reg.src_a_addr + self.elem_count * 32
            self.sram_re = True
            self.state_next = VPUState.WAIT_A1
        
        elif self.state == VPUState.WAIT_A1:
            self.state_next = VPUState.WAIT_A2
        
        elif self.state == VPUState.WAIT_A2:
            self.data_a = self._unpack(sram_rdata)
            self.log(f"Got A: {self.data_a[:4]}...")
            if self._needs_b(self.cmd_reg.opcode):
                self.state_next = VPUState.FETCH_B
            else:
                self.state_next = VPUState.COMPUTE
        
        elif self.state == VPUState.FETCH_B:
            self.sram_addr = self.cmd_reg.src_b_addr + self.elem_count * 32
            self.sram_re = True
            self.state_next = VPUState.WAIT_B1
        
        elif self.state == VPUState.WAIT_B1:
            self.state_next = VPUState.WAIT_B2
        
        elif self.state == VPUState.WAIT_B2:
            self.data_b = self._unpack(sram_rdata)
            self.log(f"Got B: {self.data_b[:4]}...")
            self.state_next = VPUState.COMPUTE
        
        elif self.state == VPUState.COMPUTE:
            if self._is_reduce(self.cmd_reg.opcode):
                if self.cmd_reg.opcode == VPUOp.SUM_REDUCE:
                    self.accumulator += int(np.sum(self.data_a.astype(np.int32)))
                elif self.cmd_reg.opcode == VPUOp.MAX_REDUCE:
                    m = int(np.max(self.data_a))
                    self.accumulator = max(self.accumulator, m) if self.elem_count > 0 else m
                self.log(f"Reduce acc={self.accumulator}")
            else:
                self.data_out = self._compute(self.cmd_reg.opcode, self.data_a, self.data_b)
                self.log(f"Compute out: {self.data_out[:4]}...")
            self.state_next = VPUState.WRITE
        
        elif self.state == VPUState.WRITE:
            if self._is_reduce(self.cmd_reg.opcode):
                out = np.zeros(self.ELEMENTS_PER_WORD, dtype=np.int8)
                for i in range(4):
                    out[i] = np.int8((self.accumulator >> (i*8)) & 0xFF)
                self.sram_wdata = self._pack(out)
            else:
                self.sram_wdata = self._pack(self.data_out)
            self.sram_addr = self.cmd_reg.dst_addr + self.elem_count * 32
            self.sram_we = True
            self.state_next = VPUState.NEXT
        
        elif self.state == VPUState.NEXT:
            self.elem_count += 1
            if self.elem_count >= self.cmd_reg.length:
                self.state_next = VPUState.DONE
            else:
                self.state_next = VPUState.FETCH_A
        
        elif self.state == VPUState.DONE:
            self.cmd_done = True
            self.cmd_ready = True
            self.log("DONE")
            self.state_next = VPUState.IDLE


class SRAMModel:
    def __init__(self):
        self.mem = {}
        self.rdata = 0
        self._next = 0
    
    def posedge(self, addr=0, wdata=0, we=False, re=False):
        self.rdata = self._next
        word = addr >> 5
        if we:
            self.mem[word] = wdata
        if re:
            self._next = self.mem.get(word, 0)


def test_vpu():
    print("=" * 60)
    print("VPU MODEL TEST")
    print("=" * 60)
    
    vpu = VectorUnit(verbose=True)
    sram = SRAMModel()
    
    def run(cmd, max_cycles=100):
        first = True
        for _ in range(max_cycles):
            sram.posedge(addr=vpu.sram_addr, wdata=vpu.sram_wdata,
                         we=vpu.sram_we, re=vpu.sram_re)
            vpu.posedge(cmd_valid=first, cmd=cmd if first else None,
                        sram_rdata=sram.rdata, sram_ready=True)
            first = False
            if vpu.cmd_done:
                return True
        return False
    
    print("\n--- TEST 1: ReLU ---")
    data = np.array([-5, -1, 0, 1, 5, 10, -128, 127] + [0]*24, dtype=np.int8)
    sram.mem[0] = vpu._pack(data)
    cmd = VPUCommand(opcode=VPUOp.RELU, src_a_addr=0, dst_addr=0x100, length=1)
    assert run(cmd)
    result = vpu._unpack(sram.mem.get(0x100 >> 5, 0))
    expected = np.maximum(data, 0)
    print(f"In:  {data[:8]}")
    print(f"Out: {result[:8]}")
    assert np.array_equal(result, expected)
    print(">>> RELU PASSED <<<\n")
    
    print("--- TEST 2: Vector Add ---")
    a = np.array([1, 2, 3, 4, 100, -100, 50, -50] + [0]*24, dtype=np.int8)
    b = np.array([10, 20, 30, 40, 50, -50, 100, 100] + [0]*24, dtype=np.int8)
    sram.mem[0x200 >> 5] = vpu._pack(a)
    sram.mem[0x300 >> 5] = vpu._pack(b)
    cmd = VPUCommand(opcode=VPUOp.ADD, src_a_addr=0x200, src_b_addr=0x300, dst_addr=0x400, length=1)
    assert run(cmd)
    result = vpu._unpack(sram.mem.get(0x400 >> 5, 0))
    expected = np.clip(a.astype(np.int16) + b.astype(np.int16), -128, 127).astype(np.int8)
    print(f"A:   {a[:8]}")
    print(f"B:   {b[:8]}")
    print(f"A+B: {result[:8]}")
    assert np.array_equal(result, expected)
    print(">>> ADD PASSED <<<\n")
    
    print("--- TEST 3: Sum Reduction ---")
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8] + [0]*24, dtype=np.int8)
    sram.mem[0x500 >> 5] = vpu._pack(data)
    cmd = VPUCommand(opcode=VPUOp.SUM_REDUCE, src_a_addr=0x500, dst_addr=0x600, length=1)
    assert run(cmd)
    result = sram.mem.get(0x600 >> 5, 0) & 0xFFFFFFFF
    expected = int(np.sum(data))
    print(f"Data: {data[:8]}")
    print(f"Sum:  {result} (expected {expected})")
    assert result == expected
    print(">>> SUM REDUCE PASSED <<<\n")
    
    print("=" * 60)
    print("ALL VPU TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_vpu()

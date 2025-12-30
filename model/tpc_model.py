#!/usr/bin/env python3
"""
Cycle-Accurate Tensor Processing Cluster (TPC) Model

Integrates: LCP, MXU, VPU, DMA, SRAM, AXI
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from systolic_array_model import SystolicArray
from dma_model import DMAEngine, DMACommand, DMAOp, AXIMemory, SRAMModel
from vpu_model import VectorUnit, VPUCommand, VPUOp
from lcp_model import LocalCommandProcessor, Opcode, IMEMModel


def make_dma_instr(subop, ext_addr, int_addr, rows=1, cols=1, ext_stride=32, int_stride=32):
    """Build DMA instruction matching RTL format"""
    return ((Opcode.DMA & 0xFF) << 120 |
            (subop & 0xFF) << 112 |
            (ext_addr & 0xFFFFFFFFFF) << 72 |
            (int_addr & 0xFFFFF) << 52 |
            (rows & 0xFFF) << 40 |
            (cols & 0xFFF) << 28 |
            (ext_stride & 0xFFF) << 16 |
            (int_stride & 0xFFF) << 4)


def make_vpu_instr(opcode, src_a, src_b, dst, length=1):
    """Build VPU instruction"""
    return ((Opcode.VECTOR & 0xFF) << 120 |
            (opcode & 0xFF) << 112 |
            (src_a & 0xFFFFF) << 92 |
            (src_b & 0xFFFFF) << 72 |
            (dst & 0xFFFFF) << 52 |
            (length & 0xFFFF) << 36)


def make_mxu_instr(weight_addr, act_addr, out_addr, M, K, N):
    """Build MXU instruction"""
    return ((Opcode.TENSOR & 0xFF) << 120 |
            (0x01 & 0xFF) << 112 |  # GEMM subop
            (weight_addr & 0xFFFFF) << 92 |
            (act_addr & 0xFFFFF) << 72 |
            (out_addr & 0xFFFFF) << 52 |
            (M & 0xFFFF) << 36 |
            (K & 0xFFFF) << 20 |
            (N & 0xFFFF) << 4)


def make_halt():
    return Opcode.HALT << 120


class TPCModel:
    """Full TPC model."""
    
    def __init__(self, array_size=4, verbose=False):
        self.verbose = verbose
        self.array_size = array_size
        
        self.lcp = LocalCommandProcessor(verbose=verbose)
        self.mxu = SystolicArray(array_size=array_size)
        self.vpu = VectorUnit(verbose=verbose)
        self.dma = DMAEngine(verbose=verbose)
        
        self.imem = IMEMModel()
        self.sram = SRAMModel(num_words=1024, verbose=False)
        self.axi_mem = AXIMemory(size_words=4096, read_latency=2, verbose=False)
        
        self.mxu_busy = False
        self.mxu_cycles_left = 0
        self.cycle = 0
    
    def log(self, msg):
        if self.verbose:
            print(f"[TPC @{self.cycle:4d}] {msg}")
    
    def load_program(self, program):
        for i, instr in enumerate(program):
            self.imem.mem[i] = instr
    
    def _pack(self, arr):
        result = 0
        for i, v in enumerate(arr.flatten()[:32]):
            result |= (int(v) & 0xFF) << (i * 8)
        return result
    
    def _unpack(self, val, n=32):
        arr = np.zeros(n, dtype=np.int8)
        for i in range(n):
            b = (val >> (i * 8)) & 0xFF
            arr[i] = np.int8(b if b < 128 else b - 256)
        return arr
    
    def load_sram(self, addr, data):
        """Load array into SRAM"""
        flat = data.flatten().astype(np.int8)
        word = addr >> 5
        for i in range(0, len(flat), 32):
            chunk = flat[i:i+32]
            self.sram.mem[word] = self._pack(np.pad(chunk, (0, max(0, 32-len(chunk)))))
            word += 1
    
    def read_sram(self, addr, size):
        """Read array from SRAM"""
        result = []
        word = addr >> 5
        while len(result) < size:
            packed = int(self.sram.mem[word]) if word < len(self.sram.mem) else 0
            result.extend(self._unpack(packed))
            word += 1
        return np.array(result[:size], dtype=np.int8)
    
    def _mxu_compute(self, cmd):
        """Execute MXU GEMM"""
        weight_addr = (cmd >> 92) & 0xFFFFF
        act_addr = (cmd >> 72) & 0xFFFFF
        out_addr = (cmd >> 52) & 0xFFFFF
        M = ((cmd >> 36) & 0xFFFF) or self.array_size
        K = ((cmd >> 20) & 0xFFFF) or self.array_size
        N = ((cmd >> 4) & 0xFFFF) or self.array_size
        
        W = self.read_sram(weight_addr, K * N).reshape(K, N)
        A = self.read_sram(act_addr, M * K).reshape(M, K)
        C = np.dot(A.astype(np.int32), W.astype(np.int32))
        self.load_sram(out_addr, C.astype(np.int8))
        
        self.mxu_cycles_left = K + M + N + 10
        self.mxu_busy = True
        self.log(f"MXU: {M}x{K} @ {K}x{N}")
    
    def run(self, max_cycles=10000):
        self.lcp.start(0)
        
        for _ in range(max_cycles):
            self.cycle += 1
            
            # IMEM
            self.imem.posedge(addr=self.lcp.imem_addr, re=self.lcp.imem_re)
            
            # MXU
            mxu_done = False
            if self.mxu_busy:
                self.mxu_cycles_left -= 1
                if self.mxu_cycles_left <= 0:
                    mxu_done = True
                    self.mxu_busy = False
            if self.lcp.mxu_valid:
                self._mxu_compute(self.lcp.mxu_cmd)
            
            # SRAM arbitration: VPU has priority when DMA is idle
            if self.vpu.sram_re or self.vpu.sram_we:
                sram_addr = self.vpu.sram_addr
                sram_wdata = self.vpu.sram_wdata
                sram_we = self.vpu.sram_we
                sram_re = self.vpu.sram_re
            else:
                sram_addr = self.dma.sram_addr
                sram_wdata = self.dma.sram_wdata
                sram_we = self.dma.sram_we
                sram_re = self.dma.sram_re
            
            self.sram.posedge(addr=sram_addr, wdata=sram_wdata, we=sram_we, re=sram_re)
            
            self.axi_mem.posedge(
                arvalid=self.dma.axi_arvalid, araddr=self.dma.axi_araddr, rready=self.dma.axi_rready,
                awvalid=self.dma.axi_awvalid, awaddr=self.dma.axi_awaddr,
                wvalid=self.dma.axi_wvalid, wdata=self.dma.axi_wdata, wlast=self.dma.axi_wlast
            )
            
            dma_cmd = None
            if self.lcp.dma_valid:
                raw = self.lcp.dma_cmd
                dma_cmd = DMACommand(
                    subop=(raw >> 112) & 0xFF,
                    ext_addr=(raw >> 72) & 0xFFFFFFFFFF,
                    int_addr=(raw >> 52) & 0xFFFFF,
                    rows=((raw >> 40) & 0xFFF) or 1,
                    cols=((raw >> 28) & 0xFFF) or 1,
                    ext_stride=((raw >> 16) & 0xFFF) or 32,
                    int_stride=((raw >> 4) & 0xFFF) or 32
                )
            
            self.dma.posedge(
                cmd_valid=self.lcp.dma_valid, cmd=dma_cmd,
                sram_rdata=self.sram.rdata, sram_ready=True,
                axi_arready=self.axi_mem.arready, axi_rvalid=self.axi_mem.rvalid, axi_rdata=self.axi_mem.rdata,
                axi_awready=self.axi_mem.awready, axi_wready=self.axi_mem.wready, axi_bvalid=self.axi_mem.bvalid
            )
            
            # VPU
            vpu_cmd = None
            if self.lcp.vpu_valid:
                raw = self.lcp.vpu_cmd
                vpu_cmd = VPUCommand(
                    opcode=(raw >> 112) & 0xFF,
                    src_a_addr=(raw >> 92) & 0xFFFFF,
                    src_b_addr=(raw >> 72) & 0xFFFFF,
                    dst_addr=(raw >> 52) & 0xFFFFF,
                    length=((raw >> 36) & 0xFFFF) or 1
                )
            
            self.vpu.posedge(cmd_valid=self.lcp.vpu_valid, cmd=vpu_cmd,
                            sram_rdata=self.sram.rdata, sram_ready=True)
            
            # LCP
            self.lcp.posedge(
                imem_rdata=self.imem.rdata, imem_valid=self.imem.valid,
                mxu_ready=not self.mxu_busy, mxu_done=mxu_done,
                vpu_ready=self.vpu.cmd_ready, vpu_done=self.vpu.cmd_done,
                dma_ready=self.dma.cmd_ready, dma_done=self.dma.cmd_done
            )
            
            if self.lcp.halted:
                self.log(f"Done in {self.cycle} cycles")
                return True
        
        self.log("TIMEOUT")
        return False


def test_tpc():
    print("=" * 70)
    print("TPC INTEGRATED MODEL TEST")
    print("=" * 70)
    
    # TEST 1: VPU ReLU only
    print("\n--- TEST 1: VPU ReLU ---")
    tpc = TPCModel(verbose=True)
    
    data = np.array([-5, -1, 0, 1, 5, 10, -128, 127] + [0]*24, dtype=np.int8)
    tpc.load_sram(0x100, data)
    
    program = [
        make_vpu_instr(VPUOp.RELU, src_a=0x100, src_b=0, dst=0x200, length=1),
        make_halt()
    ]
    tpc.load_program(program)
    
    assert tpc.run(max_cycles=100)
    
    result = tpc.read_sram(0x200, 32)[:8]
    expected = np.maximum(data[:8], 0)
    print(f"In:  {data[:8]}")
    print(f"Out: {result}")
    assert np.array_equal(result, expected)
    print(">>> VPU RELU PASSED <<<\n")
    
    # TEST 2: GEMM
    print("--- TEST 2: GEMM 2x2 ---")
    tpc = TPCModel(verbose=True)
    
    A = np.array([[1, 2], [3, 4]], dtype=np.int8)
    B = np.array([[5, 6], [7, 8]], dtype=np.int8)
    
    tpc.load_sram(0x000, B)  # Weights
    tpc.load_sram(0x100, A)  # Activations
    
    program = [
        make_mxu_instr(0x000, 0x100, 0x200, M=2, K=2, N=2),
        make_halt()
    ]
    tpc.load_program(program)
    
    assert tpc.run(max_cycles=100)
    
    C = tpc.read_sram(0x200, 4).reshape(2, 2)
    expected = np.dot(A.astype(np.int32), B.astype(np.int32)).astype(np.int8)
    print(f"A:\n{A}")
    print(f"B:\n{B}")
    print(f"C:\n{C}")
    print(f"Expected:\n{expected}")
    assert np.array_equal(C, expected)
    print(">>> GEMM PASSED <<<\n")
    
    # TEST 3: DMA + VPU
    print("--- TEST 3: DMA LOAD + VPU RELU ---")
    tpc = TPCModel(verbose=True)
    
    # Put data in external memory
    ext_data = np.array([-3, -1, 0, 2, 4, 6, -100, 100] + [0]*24, dtype=np.int8)
    tpc.axi_mem.mem[0] = tpc._pack(ext_data)
    
    program = [
        make_dma_instr(DMAOp.LOAD, ext_addr=0x00, int_addr=0x100, rows=1, cols=1),
        make_vpu_instr(VPUOp.RELU, src_a=0x100, src_b=0, dst=0x200, length=1),
        make_halt()
    ]
    tpc.load_program(program)
    
    assert tpc.run(max_cycles=200)
    
    result = tpc.read_sram(0x200, 32)[:8]
    expected = np.maximum(ext_data[:8], 0)
    print(f"Ext: {ext_data[:8]}")
    print(f"Out: {result}")
    assert np.array_equal(result, expected)
    print(">>> DMA+VPU PASSED <<<\n")
    
    print("=" * 70)
    print("ALL TPC INTEGRATED TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_tpc()

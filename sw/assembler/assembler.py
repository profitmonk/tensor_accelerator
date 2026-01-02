#!/usr/bin/env python3
"""
LCP Assembler - Tensor Accelerator Instruction Assembler

Generates binary instruction streams for the Local Command Processor.
Supports TENSOR, VECTOR, DMA, SYNC, LOOP, and control operations.

Usage:
    python assembler.py input.asm -o output.hex
"""

import argparse
import struct
import re
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import IntEnum

#==============================================================================
# Instruction Encoding
#==============================================================================

class Opcode(IntEnum):
    NOP      = 0x00
    TENSOR   = 0x01
    VECTOR   = 0x02
    DMA      = 0x03
    SYNC     = 0x04
    LOOP     = 0x05
    ENDLOOP  = 0x06
    BARRIER  = 0x07
    HALT     = 0xFF

class TensorSubop(IntEnum):
    GEMM         = 0x01
    GEMM_ACC     = 0x02
    GEMM_RELU    = 0x03
    GEMM_BIAS    = 0x04
    IM2COL       = 0x10  # im2col transform for conv
    MAXPOOL      = 0x20  # max pooling
    AVGPOOL      = 0x21  # average pooling
    TRANSPOSE    = 0x30  # 2D matrix transpose
    TRANSPOSE_ND = 0x31  # N-dimensional transpose

class VectorSubop(IntEnum):
    # Elementwise arithmetic
    ADD        = 0x01
    SUB        = 0x02
    MUL        = 0x03
    MADD       = 0x04
    DIV        = 0x05
    ADD_BCAST  = 0x06  # Add with broadcasting
    BIAS_ADD   = 0x07  # Add bias (broadcast over spatial)
    
    # Activations
    RELU       = 0x10
    GELU       = 0x11
    SILU       = 0x12
    SIGMOID    = 0x13
    TANH       = 0x14
    EXP        = 0x15
    RSQRT      = 0x16
    
    # Reductions
    SUM        = 0x20
    MAX        = 0x21
    MIN        = 0x22
    GLOBAL_AVG = 0x23  # Global average pooling
    
    # Memory/data movement
    LOAD       = 0x30
    STORE      = 0x31
    BCAST      = 0x32
    MOV        = 0x33
    ZERO       = 0x34
    SCALE      = 0x35
    SCALE_SHIFT = 0x36  # y = x * scale + shift
    
    # Softmax passes
    SOFTMAX_P1 = 0x40  # Pass 1: compute max
    SOFTMAX_P2 = 0x41  # Pass 2: exp and sum
    SOFTMAX_P3 = 0x42  # Pass 3: normalize
    
    # Normalization
    BATCHNORM      = 0x50  # BatchNorm: y = x * scale + bias (per channel)
    LAYERNORM_MEAN = 0x51  # LayerNorm pass 1: compute mean
    LAYERNORM_VAR  = 0x52  # LayerNorm pass 2: compute variance
    LAYERNORM_NORM = 0x53  # LayerNorm pass 3: normalize

class DMASubop(IntEnum):
    LOAD_2D   = 0x01
    STORE_2D  = 0x02
    COPY      = 0x03
    LOAD_1D   = 0x04
    STORE_1D  = 0x05

class SyncSubop(IntEnum):
    WAIT_MXU  = 0x01
    WAIT_VPU  = 0x02
    WAIT_DMA  = 0x03
    WAIT_ALL  = 0xFF

#==============================================================================
# Instruction Format
#==============================================================================

@dataclass
class Instruction:
    """
    128-bit instruction format:
    [127:120] opcode     (8 bits)
    [119:112] subop      (8 bits)
    [111:96]  dst        (16 bits)
    [95:80]   src0       (16 bits)
    [79:64]   src1       (16 bits)
    [63:48]   dim_m      (16 bits)
    [47:32]   dim_n      (16 bits)
    [31:16]   dim_k      (16 bits)
    [15:0]    flags      (16 bits)
    """
    opcode: int = 0
    subop: int = 0
    dst: int = 0
    src0: int = 0
    src1: int = 0
    dim_m: int = 0
    dim_n: int = 0
    dim_k: int = 0
    flags: int = 0
    
    def encode(self) -> bytes:
        """Encode to 16 bytes (128 bits), big-endian"""
        return struct.pack('>BBHHHHHHH',
            self.opcode & 0xFF,
            self.subop & 0xFF,
            self.dst & 0xFFFF,
            self.src0 & 0xFFFF,
            self.src1 & 0xFFFF,
            self.dim_m & 0xFFFF,
            self.dim_n & 0xFFFF,
            self.dim_k & 0xFFFF,
            self.flags & 0xFFFF
        )
    
    def to_hex(self) -> str:
        """Return as hex string for Verilog $readmemh"""
        return self.encode().hex()

#==============================================================================
# Assembler
#==============================================================================

class Assembler:
    """LCP Instruction Assembler"""
    
    # Predefined SRAM addresses
    SRAM_ADDRS = {
        'SRAM_ACT_A':    0x0000,
        'SRAM_ACT_B':    0x1000,
        'SRAM_WT_A':     0x2000,
        'SRAM_WT_B':     0x4000,
        'SRAM_OUT':      0x6000,
        'SRAM_SCRATCH':  0x7000,
        'SRAM_VEC':      0x8000,
    }
    
    # Vector register aliases
    VREG_ALIASES = {f'v{i}': i for i in range(32)}
    
    def __init__(self):
        self.instructions: List[Instruction] = []
        self.labels: Dict[str, int] = {}
        self.constants: Dict[str, int] = {}
        self.line_num = 0
        
    def parse_value(self, s: str) -> int:
        """Parse an integer value (decimal, hex, or symbol)"""
        s = s.strip()
        
        # Check for predefined addresses
        if s in self.SRAM_ADDRS:
            return self.SRAM_ADDRS[s]
        
        # Check for vector registers
        if s in self.VREG_ALIASES:
            return self.VREG_ALIASES[s]
        
        # Check for user-defined constants
        if s in self.constants:
            return self.constants[s]
        
        # Check for labels
        if s in self.labels:
            return self.labels[s]
        
        # Parse as number
        if s.startswith('0x') or s.startswith('0X'):
            return int(s, 16)
        elif s.startswith('0b') or s.startswith('0B'):
            return int(s, 2)
        else:
            return int(s)
    
    def parse_operands(self, operands: str) -> List[str]:
        """Split operands by comma, handling nested expressions"""
        result = []
        current = ""
        paren_depth = 0
        
        for char in operands:
            if char == '(':
                paren_depth += 1
                current += char
            elif char == ')':
                paren_depth -= 1
                current += char
            elif char == ',' and paren_depth == 0:
                result.append(current.strip())
                current = ""
            else:
                current += char
        
        if current.strip():
            result.append(current.strip())
        
        return result
    
    def assemble_line(self, line: str) -> Optional[Instruction]:
        """Assemble a single line"""
        # Remove comments
        if '#' in line:
            line = line[:line.index('#')]
        if '//' in line:
            line = line[:line.index('//')]
        
        line = line.strip()
        if not line:
            return None
        
        # Handle labels
        if line.endswith(':'):
            label = line[:-1].strip()
            self.labels[label] = len(self.instructions)
            return None
        
        # Handle .equ directives
        if line.startswith('.equ') or line.startswith('.set'):
            parts = line.split(None, 2)
            if len(parts) >= 3:
                name = parts[1].rstrip(',')
                value = self.parse_value(parts[2])
                self.constants[name] = value
            return None
        
        # Handle .config block (ignore for now)
        if line.startswith('.'):
            return None
        
        # Parse instruction
        parts = line.split(None, 1)
        mnemonic = parts[0].upper()
        operands = parts[1] if len(parts) > 1 else ""
        ops = self.parse_operands(operands)
        
        instr = Instruction()
        
        # Handle compound mnemonics (OP.SUBOP)
        if '.' in mnemonic:
            op, subop = mnemonic.split('.', 1)
        else:
            op, subop = mnemonic, ""
        
        #----------------------------------------------------------------------
        # TENSOR operations
        #----------------------------------------------------------------------
        if op == 'TENSOR':
            instr.opcode = Opcode.TENSOR
            
            tensor_subop_map = {
                'GEMM': TensorSubop.GEMM,
                'GEMM_ACC': TensorSubop.GEMM_ACC,
                'GEMM_RELU': TensorSubop.GEMM_RELU,
                'GEMM_BIAS': TensorSubop.GEMM_BIAS,
                'IM2COL': TensorSubop.IM2COL,
                'MAXPOOL': TensorSubop.MAXPOOL,
                'AVGPOOL': TensorSubop.AVGPOOL,
                'TRANSPOSE': TensorSubop.TRANSPOSE,
                'TRANSPOSE_ND': TensorSubop.TRANSPOSE_ND,
            }
            
            instr.subop = tensor_subop_map.get(subop, TensorSubop.GEMM)
            
            # Format varies by operation
            if subop in ['GEMM', 'GEMM_ACC', 'GEMM_RELU', 'GEMM_BIAS']:
                # TENSOR.GEMM dst, src_act, src_wt, M, N, K [, flags]
                if len(ops) >= 6:
                    instr.dst = self.parse_value(ops[0])
                    instr.src0 = self.parse_value(ops[1])
                    instr.src1 = self.parse_value(ops[2])
                    instr.dim_m = self.parse_value(ops[3])
                    instr.dim_n = self.parse_value(ops[4])
                    instr.dim_k = self.parse_value(ops[5])
                    if len(ops) > 6:
                        instr.flags = self.parse_value(ops[6])
            elif subop == 'IM2COL':
                # TENSOR.IM2COL dst, src, C, H, W, kH, kW, stride, pad
                if len(ops) >= 9:
                    instr.dst = self.parse_value(ops[0])
                    instr.src0 = self.parse_value(ops[1])
                    # Pack C, H, W into dim fields
                    instr.dim_m = self.parse_value(ops[2])  # C
                    instr.dim_n = self.parse_value(ops[3])  # H
                    instr.dim_k = self.parse_value(ops[4])  # W
                    # Pack kH, kW, stride, pad into src1 and flags
                    kH = self.parse_value(ops[5])
                    kW = self.parse_value(ops[6])
                    stride = self.parse_value(ops[7])
                    pad = self.parse_value(ops[8])
                    instr.src1 = (kH << 8) | kW
                    instr.flags = (stride << 8) | pad
            elif subop in ['MAXPOOL', 'AVGPOOL']:
                # TENSOR.MAXPOOL dst, src, C, H, W, kH, kW, sH, sW
                if len(ops) >= 9:
                    instr.dst = self.parse_value(ops[0])
                    instr.src0 = self.parse_value(ops[1])
                    instr.dim_m = self.parse_value(ops[2])  # C
                    instr.dim_n = self.parse_value(ops[3])  # H
                    instr.dim_k = self.parse_value(ops[4])  # W
                    kH = self.parse_value(ops[5])
                    kW = self.parse_value(ops[6])
                    sH = self.parse_value(ops[7])
                    sW = self.parse_value(ops[8])
                    instr.src1 = (kH << 8) | kW
                    instr.flags = (sH << 8) | sW
            elif subop == 'TRANSPOSE':
                # TENSOR.TRANSPOSE dst, src, M, N
                if len(ops) >= 4:
                    instr.dst = self.parse_value(ops[0])
                    instr.src0 = self.parse_value(ops[1])
                    instr.dim_m = self.parse_value(ops[2])
                    instr.dim_n = self.parse_value(ops[3])
            elif subop == 'TRANSPOSE_ND':
                # TENSOR.TRANSPOSE_ND dst, src, shape..., perm...
                # Complex - just store dst, src for now
                if len(ops) >= 2:
                    instr.dst = self.parse_value(ops[0])
                    instr.src0 = self.parse_value(ops[1])
        
        #----------------------------------------------------------------------
        # VECTOR operations
        #----------------------------------------------------------------------
        elif op == 'VEC' or op == 'VECTOR':
            instr.opcode = Opcode.VECTOR
            
            subop_map = {
                # Elementwise
                'ADD': VectorSubop.ADD,
                'SUB': VectorSubop.SUB,
                'MUL': VectorSubop.MUL,
                'MADD': VectorSubop.MADD,
                'DIV': VectorSubop.DIV,
                'ADD_BCAST': VectorSubop.ADD_BCAST,
                'BIAS_ADD': VectorSubop.BIAS_ADD,
                # Activations
                'RELU': VectorSubop.RELU,
                'GELU': VectorSubop.GELU,
                'SILU': VectorSubop.SILU,
                'SIGMOID': VectorSubop.SIGMOID,
                'TANH': VectorSubop.TANH,
                'EXP': VectorSubop.EXP,
                'RSQRT': VectorSubop.RSQRT,
                # Reductions
                'SUM': VectorSubop.SUM,
                'MAX': VectorSubop.MAX,
                'MIN': VectorSubop.MIN,
                'GLOBAL_AVG': VectorSubop.GLOBAL_AVG,
                # Memory/data
                'LOAD': VectorSubop.LOAD,
                'STORE': VectorSubop.STORE,
                'BCAST': VectorSubop.BCAST,
                'MOV': VectorSubop.MOV,
                'ZERO': VectorSubop.ZERO,
                'SCALE': VectorSubop.SCALE,
                'SCALE_SHIFT': VectorSubop.SCALE_SHIFT,
                # Softmax
                'SOFTMAX_P1': VectorSubop.SOFTMAX_P1,
                'SOFTMAX_P2': VectorSubop.SOFTMAX_P2,
                'SOFTMAX_P3': VectorSubop.SOFTMAX_P3,
                # Normalization
                'BATCHNORM': VectorSubop.BATCHNORM,
                'LAYERNORM_MEAN': VectorSubop.LAYERNORM_MEAN,
                'LAYERNORM_VAR': VectorSubop.LAYERNORM_VAR,
                'LAYERNORM_NORM': VectorSubop.LAYERNORM_NORM,
            }
            
            instr.subop = subop_map.get(subop, 0)
            
            # Format varies by operation
            if subop in ['LOAD', 'STORE']:
                # VEC.LOAD vd, addr, count
                if len(ops) >= 2:
                    instr.dst = self.parse_value(ops[0])
                    instr.src0 = self.parse_value(ops[1])
                    if len(ops) > 2:
                        instr.dim_m = self.parse_value(ops[2])
            elif subop in ['ADD', 'SUB', 'MUL', 'DIV']:
                # VEC.ADD dst, src1, src2, count
                if len(ops) >= 4:
                    instr.dst = self.parse_value(ops[0])
                    instr.src0 = self.parse_value(ops[1])
                    instr.src1 = self.parse_value(ops[2])
                    instr.dim_m = self.parse_value(ops[3])
                elif len(ops) >= 3:
                    instr.dst = self.parse_value(ops[0])
                    instr.src0 = self.parse_value(ops[1])
                    instr.src1 = self.parse_value(ops[2])
            elif subop == 'ADD_BCAST':
                # VEC.ADD_BCAST dst, src1, src2, count1, count2
                if len(ops) >= 5:
                    instr.dst = self.parse_value(ops[0])
                    instr.src0 = self.parse_value(ops[1])
                    instr.src1 = self.parse_value(ops[2])
                    instr.dim_m = self.parse_value(ops[3])
                    instr.dim_n = self.parse_value(ops[4])
            elif subop == 'BIAS_ADD':
                # VEC.BIAS_ADD dst, src, bias, M, N
                if len(ops) >= 5:
                    instr.dst = self.parse_value(ops[0])
                    instr.src0 = self.parse_value(ops[1])
                    instr.src1 = self.parse_value(ops[2])
                    instr.dim_m = self.parse_value(ops[3])
                    instr.dim_n = self.parse_value(ops[4])
            elif subop in ['RELU', 'GELU', 'SILU', 'EXP', 'RSQRT', 'TANH', 'SIGMOID']:
                # VEC.RELU dst, src, count
                if len(ops) >= 2:
                    instr.dst = self.parse_value(ops[0])
                    instr.src0 = self.parse_value(ops[1])
                    if len(ops) > 2:
                        instr.src1 = self.parse_value(ops[2])  # Count in src1
            elif subop in ['SUM', 'MAX', 'MIN']:
                # VEC.SUM vd, vs1
                if len(ops) >= 2:
                    instr.dst = self.parse_value(ops[0])
                    instr.src0 = self.parse_value(ops[1])
            elif subop == 'GLOBAL_AVG':
                # VEC.GLOBAL_AVG dst, src, channels, spatial_size
                if len(ops) >= 4:
                    instr.dst = self.parse_value(ops[0])
                    instr.src0 = self.parse_value(ops[1])
                    instr.dim_m = self.parse_value(ops[2])  # channels
                    instr.dim_n = self.parse_value(ops[3])  # spatial
            elif subop == 'BCAST':
                # VEC.BCAST vd, imm
                if len(ops) >= 2:
                    instr.dst = self.parse_value(ops[0])
                    instr.dim_n = self.parse_value(ops[1])
            elif subop == 'SCALE':
                # VEC.SCALE dst, src, scale, count
                if len(ops) >= 4:
                    instr.dst = self.parse_value(ops[0])
                    instr.src0 = self.parse_value(ops[1])
                    instr.dim_n = self.parse_value(ops[2])  # scale
                    instr.dim_m = self.parse_value(ops[3])  # count
                elif len(ops) >= 3:
                    instr.dst = self.parse_value(ops[0])
                    instr.src0 = self.parse_value(ops[1])
                    instr.dim_n = self.parse_value(ops[2])
            elif subop == 'SCALE_SHIFT':
                # VEC.SCALE_SHIFT dst, src, scale, bias, size, count
                if len(ops) >= 6:
                    instr.dst = self.parse_value(ops[0])
                    instr.src0 = self.parse_value(ops[1])
                    instr.src1 = self.parse_value(ops[2])  # scale addr
                    instr.dim_m = self.parse_value(ops[3])  # bias addr (packed)
                    instr.dim_n = self.parse_value(ops[4])  # size
                    instr.dim_k = self.parse_value(ops[5])  # count
            elif subop == 'ZERO':
                # VEC.ZERO vd
                if len(ops) >= 1:
                    instr.dst = self.parse_value(ops[0])
            elif subop in ['SOFTMAX_P1', 'SOFTMAX_P2', 'SOFTMAX_P3']:
                # SOFTMAX passes: dst, src, [scratch], size, vectors
                if len(ops) >= 4:
                    instr.dst = self.parse_value(ops[0])
                    instr.src0 = self.parse_value(ops[1])
                    if subop == 'SOFTMAX_P1':
                        instr.dim_m = self.parse_value(ops[2])
                        instr.dim_n = self.parse_value(ops[3])
                    else:
                        instr.src1 = self.parse_value(ops[2])  # scratch
                        instr.dim_m = self.parse_value(ops[3])
                        if len(ops) > 4:
                            instr.dim_n = self.parse_value(ops[4])
            elif subop == 'BATCHNORM':
                # VEC.BATCHNORM dst, src, scale, bias, channels, spatial
                if len(ops) >= 6:
                    instr.dst = self.parse_value(ops[0])
                    instr.src0 = self.parse_value(ops[1])
                    instr.src1 = self.parse_value(ops[2])  # scale addr
                    instr.dim_m = self.parse_value(ops[3])  # bias addr
                    instr.dim_n = self.parse_value(ops[4])  # channels
                    instr.dim_k = self.parse_value(ops[5])  # spatial
            elif subop in ['LAYERNORM_MEAN', 'LAYERNORM_VAR', 'LAYERNORM_NORM']:
                # Various LayerNorm passes
                if len(ops) >= 4:
                    instr.dst = self.parse_value(ops[0])
                    instr.src0 = self.parse_value(ops[1])
                    if subop == 'LAYERNORM_MEAN':
                        instr.dim_m = self.parse_value(ops[2])
                        instr.dim_n = self.parse_value(ops[3])
                    else:
                        instr.src1 = self.parse_value(ops[2])
                        instr.dim_m = self.parse_value(ops[3])
                        if len(ops) > 4:
                            instr.dim_n = self.parse_value(ops[4])
                        if len(ops) > 5:
                            instr.dim_k = self.parse_value(ops[5])
        
        #----------------------------------------------------------------------
        # DMA operations
        #----------------------------------------------------------------------
        elif op == 'DMA':
            instr.opcode = Opcode.DMA
            
            if subop == 'LOAD_2D':
                instr.subop = DMASubop.LOAD_2D
            elif subop == 'STORE_2D':
                instr.subop = DMASubop.STORE_2D
            elif subop == 'COPY':
                instr.subop = DMASubop.COPY
            elif subop == 'LOAD_1D':
                instr.subop = DMASubop.LOAD_1D
            elif subop == 'STORE_1D':
                instr.subop = DMASubop.STORE_1D
            else:
                instr.subop = DMASubop.LOAD_2D
            
            # Format: DMA.LOAD_2D dst, src, rows, cols, src_stride, dst_stride
            if len(ops) >= 4:
                instr.dst = self.parse_value(ops[0])
                instr.src0 = self.parse_value(ops[1])
                instr.dim_m = self.parse_value(ops[2])  # rows
                instr.dim_n = self.parse_value(ops[3])  # cols
                if len(ops) > 4:
                    instr.dim_k = self.parse_value(ops[4])  # src_stride
                if len(ops) > 5:
                    instr.src1 = self.parse_value(ops[5])   # dst_stride
        
        #----------------------------------------------------------------------
        # SYNC operations
        #----------------------------------------------------------------------
        elif op == 'SYNC':
            instr.opcode = Opcode.SYNC
            
            if subop == 'WAIT_MXU':
                instr.subop = SyncSubop.WAIT_MXU
            elif subop == 'WAIT_VPU':
                instr.subop = SyncSubop.WAIT_VPU
            elif subop == 'WAIT_DMA':
                instr.subop = SyncSubop.WAIT_DMA
            elif subop == 'WAIT_ALL':
                instr.subop = SyncSubop.WAIT_ALL
            else:
                instr.subop = SyncSubop.WAIT_ALL
        
        #----------------------------------------------------------------------
        # LOOP operations
        #----------------------------------------------------------------------
        elif op == 'LOOP':
            instr.opcode = Opcode.LOOP
            if len(ops) >= 1:
                instr.dim_n = self.parse_value(ops[0])  # Loop count
        
        elif op == 'ENDLOOP':
            instr.opcode = Opcode.ENDLOOP
        
        #----------------------------------------------------------------------
        # Control operations
        #----------------------------------------------------------------------
        elif op == 'BARRIER':
            instr.opcode = Opcode.BARRIER
        
        elif op == 'HALT':
            instr.opcode = Opcode.HALT
        
        elif op == 'NOP':
            instr.opcode = Opcode.NOP
        
        else:
            raise ValueError(f"Unknown mnemonic: {mnemonic} at line {self.line_num}")
        
        return instr
    
    def assemble(self, source: str) -> List[Instruction]:
        """Assemble source code"""
        self.instructions = []
        self.labels = {}
        self.line_num = 0
        
        # First pass: collect labels
        for line in source.split('\n'):
            self.line_num += 1
            line = line.strip()
            if line.endswith(':'):
                label = line[:-1].strip()
                self.labels[label] = len([l for l in source.split('\n')[:self.line_num] 
                                         if l.strip() and not l.strip().startswith('#') 
                                         and not l.strip().startswith('.') 
                                         and not l.strip().endswith(':')])
        
        # Second pass: assemble instructions
        self.line_num = 0
        for line in source.split('\n'):
            self.line_num += 1
            try:
                instr = self.assemble_line(line)
                if instr is not None:
                    self.instructions.append(instr)
            except Exception as e:
                print(f"Error at line {self.line_num}: {line}")
                raise e
        
        return self.instructions
    
    def write_hex(self, filename: str):
        """Write instructions as hex file for Verilog $readmemh"""
        with open(filename, 'w') as f:
            for i, instr in enumerate(self.instructions):
                f.write(f"{instr.to_hex()}  // {i:04d}\n")
    
    def write_bin(self, filename: str):
        """Write instructions as binary file"""
        with open(filename, 'wb') as f:
            for instr in self.instructions:
                f.write(instr.encode())
    
    def write_coe(self, filename: str):
        """Write instructions as Xilinx COE file"""
        with open(filename, 'w') as f:
            f.write("memory_initialization_radix=16;\n")
            f.write("memory_initialization_vector=\n")
            for i, instr in enumerate(self.instructions):
                suffix = ";" if i == len(self.instructions) - 1 else ","
                f.write(f"{instr.to_hex()}{suffix}\n")


#==============================================================================
# Main
#==============================================================================

def main():
    parser = argparse.ArgumentParser(description='LCP Assembler')
    parser.add_argument('input', help='Input assembly file')
    parser.add_argument('-o', '--output', default='output.hex', help='Output file')
    parser.add_argument('-f', '--format', choices=['hex', 'bin', 'coe'], default='hex',
                       help='Output format')
    args = parser.parse_args()
    
    with open(args.input, 'r') as f:
        source = f.read()
    
    asm = Assembler()
    asm.assemble(source)
    
    print(f"Assembled {len(asm.instructions)} instructions")
    
    if args.format == 'hex':
        asm.write_hex(args.output)
    elif args.format == 'bin':
        asm.write_bin(args.output)
    elif args.format == 'coe':
        asm.write_coe(args.output)
    
    print(f"Output written to {args.output}")


if __name__ == '__main__':
    main()

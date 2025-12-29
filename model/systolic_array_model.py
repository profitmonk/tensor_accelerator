#!/usr/bin/env python3
"""
Cycle-Accurate Functional Model for Weight-Stationary Systolic Array

This model exactly mimics RTL behavior:
- All signals are modeled
- Registered (sequential) vs combinational logic is explicit
- Clock edges are simulated with posedge() calls
- Can generate VCD-like traces for comparison with RTL

Architecture:
- Weight-stationary: weights loaded once, held in PE registers
- Activations flow horizontally (left to right) with input skewing
- Partial sums flow vertically (top to bottom)
- Output de-skewing aligns results

Author: Tensor Accelerator Project
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class Signal:
    """A wire/register with current and next values for proper sequential modeling"""
    name: str
    width: int = 32
    value: int = 0
    next_value: int = 0
    is_reg: bool = True  # True for registers, False for wires
    
    def set(self, val: int):
        """Set next value (will take effect on posedge)"""
        if self.is_reg:
            self.next_value = val
        else:
            self.value = val  # Wires update immediately
            self.next_value = val
    
    def get(self) -> int:
        """Get current value"""
        return self.value
    
    def posedge(self):
        """Clock edge: transfer next to current for registers"""
        if self.is_reg:
            self.value = self.next_value
    
    def reset(self, val: int = 0):
        """Synchronous reset"""
        self.value = val
        self.next_value = val


class MAC_PE:
    """
    MAC Processing Element - exactly models rtl/core/mac_pe.v
    
    Weight-stationary dataflow:
    - weight_reg: loaded once, held stationary
    - act_in -> act_reg -> act_out (1 cycle delay for activation flow)
    - psum_out = psum_in + (act_reg * weight_reg)
    """
    
    def __init__(self, row: int, col: int, data_width: int = 8, acc_width: int = 32):
        self.row = row
        self.col = col
        self.data_width = data_width
        self.acc_width = acc_width
        
        # Registers (directly matching RTL)
        self.weight_reg = Signal(f"pe[{row}][{col}].weight_reg", data_width)
        self.act_reg = Signal(f"pe[{row}][{col}].act_reg", data_width)
        self.act_out = Signal(f"pe[{row}][{col}].act_out", data_width)
        self.psum_out = Signal(f"pe[{row}][{col}].psum_out", acc_width)
        
        # Combinational signals (for debug)
        self.product = Signal(f"pe[{row}][{col}].product", 2*data_width, is_reg=False)
    
    def reset(self):
        """Reset all registers to 0"""
        self.weight_reg.reset(0)
        self.act_reg.reset(0)
        self.act_out.reset(0)
        self.psum_out.reset(0)
        self.product.reset(0)
    
    def comb_logic(self, act_in: int, psum_in: int, enable: bool, 
                   load_weight: bool, weight_in: int, clear_acc: bool):
        """
        Combinational logic - compute next values
        Called BEFORE posedge
        """
        # Product is combinational (uses registered act_reg)
        self.product.set(self.act_reg.get() * self.weight_reg.get())
        
        # Weight loading (independent of enable)
        if load_weight:
            self.weight_reg.set(weight_in)
        
        # Main datapath (when enabled)
        if enable:
            # Register activation
            self.act_reg.set(act_in)
            
            # Pass activation to right neighbor (from registered value)
            self.act_out.set(self.act_reg.get())
            
            # Compute psum_out = psum_in + product
            if clear_acc:
                self.psum_out.set(self.product.get())
            else:
                self.psum_out.set(psum_in + self.product.get())
    
    def posedge(self):
        """Clock edge - update all registers"""
        self.weight_reg.posedge()
        self.act_reg.posedge()
        self.act_out.posedge()
        self.psum_out.posedge()
    
    def get_state(self) -> Dict:
        """Return current state for debugging"""
        return {
            'weight_reg': self.weight_reg.get(),
            'act_reg': self.act_reg.get(),
            'act_out': self.act_out.get(),
            'psum_out': self.psum_out.get(),
            'product': self.product.get()
        }


class InputSkewRegister:
    """
    Input skewing for one row.
    Row i needs i+1 stages total (i delay stages + 1 output register).
    
    This matches the RTL fix where we added an output register stage.
    """
    
    def __init__(self, row: int, data_width: int = 8):
        self.row = row
        self.data_width = data_width
        self.num_delay_stages = row  # Row 0 has 0 delay stages, row 1 has 1, etc.
        
        # Delay stages (shift register)
        self.stages: List[Signal] = [
            Signal(f"skew[{row}].stage[{i}]", data_width)
            for i in range(self.num_delay_stages)
        ]
        
        # Output register (always present)
        self.output = Signal(f"skew[{row}].output", data_width)
    
    def reset(self):
        for stage in self.stages:
            stage.reset(0)
        self.output.reset(0)
    
    def comb_logic(self, input_val: int, enable: bool):
        """Compute next values for shift register"""
        if not enable:
            return
        
        if self.num_delay_stages == 0:
            # Row 0: input goes directly to output register
            self.output.set(input_val)
        else:
            # Stage 0 gets input
            self.stages[0].set(input_val)
            
            # Shift through delay stages
            for i in range(1, self.num_delay_stages):
                self.stages[i].set(self.stages[i-1].get())
            
            # Output register gets from last delay stage
            self.output.set(self.stages[self.num_delay_stages - 1].get())
    
    def posedge(self):
        for stage in self.stages:
            stage.posedge()
        self.output.posedge()
    
    def get_output(self) -> int:
        return self.output.get()


class OutputDeskewRegister:
    """
    Output de-skewing for one column.
    Column j arrives 2*j cycles after column 0 (input skew + horizontal prop).
    So column j needs 2*(ARRAY_SIZE-1-j) delay stages to align with last column.
    """
    
    def __init__(self, col: int, array_size: int, acc_width: int = 32):
        self.col = col
        self.acc_width = acc_width
        self.num_stages = 2 * (array_size - 1 - col)
        
        # Delay stages
        self.stages: List[Signal] = [
            Signal(f"deskew[{col}].stage[{i}]", acc_width)
            for i in range(self.num_stages)
        ]
        
        # Output (either from last stage or direct)
        self.output = Signal(f"deskew[{col}].output", acc_width, is_reg=False)
    
    def reset(self):
        for stage in self.stages:
            stage.reset(0)
        self.output.reset(0)
    
    def comb_logic(self, input_val: int, enable: bool):
        """Compute next values"""
        if self.num_stages == 0:
            # Direct passthrough (rightmost column)
            self.output.set(input_val)
        else:
            if enable:
                # Shift in new value
                self.stages[0].set(input_val)
                for i in range(1, self.num_stages):
                    self.stages[i].set(self.stages[i-1].get())
            
            # Output from last stage
            self.output.set(self.stages[self.num_stages - 1].get())
    
    def posedge(self):
        for stage in self.stages:
            stage.posedge()
    
    def get_output(self) -> int:
        return self.output.get()


class SystolicArray:
    """
    Complete systolic array model - matches rtl/core/systolic_array.v
    
    State machine: IDLE -> LOAD -> COMPUTE -> DRAIN -> DONE
    """
    
    # State encoding
    S_IDLE = 0
    S_LOAD = 1
    S_COMPUTE = 2
    S_DRAIN = 3
    S_DONE = 4
    
    STATE_NAMES = ['IDLE', 'LOAD', 'COMPUTE', 'DRAIN', 'DONE']
    
    def __init__(self, array_size: int = 4, data_width: int = 8, acc_width: int = 32):
        self.array_size = array_size
        self.data_width = data_width
        self.acc_width = acc_width
        
        # State machine registers
        self.state = Signal("state", 3)
        self.cycle_count = Signal("cycle_count", 16)
        
        # Configuration
        self.cfg_k_tiles = Signal("cfg_k_tiles", 16, is_reg=False)
        
        # PE array
        self.pes: List[List[MAC_PE]] = [
            [MAC_PE(row, col, data_width, acc_width) 
             for col in range(array_size)]
            for row in range(array_size)
        ]
        
        # Input skewing (one per row)
        self.input_skew: List[InputSkewRegister] = [
            InputSkewRegister(row, data_width)
            for row in range(array_size)
        ]
        
        # Output de-skewing (one per column)
        self.output_deskew: List[OutputDeskewRegister] = [
            OutputDeskewRegister(col, array_size, acc_width)
            for col in range(array_size)
        ]
        
        # Inter-PE wiring (combinational)
        # act_h[row][col] = horizontal activation at PE[row][col] input
        self.act_h: List[List[Signal]] = [
            [Signal(f"act_h[{row}][{col}]", data_width, is_reg=False) 
             for col in range(array_size + 1)]
            for row in range(array_size)
        ]
        
        # psum_v[row][col] = vertical psum at PE[row][col] input (row 0 = top input)
        self.psum_v: List[List[Signal]] = [
            [Signal(f"psum_v[{row}][{col}]", acc_width, is_reg=False) 
             for col in range(array_size)]
            for row in range(array_size + 1)
        ]
        
        # Output signals
        self.result_valid = Signal("result_valid", 1, is_reg=False)
        self.result_data: List[Signal] = [
            Signal(f"result_data[{col}]", acc_width, is_reg=False)
            for col in range(array_size)
        ]
        self.busy = Signal("busy", 1, is_reg=False)
        self.done = Signal("done", 1, is_reg=False)
        
        # Internal control signals
        self.pe_enable = Signal("pe_enable", 1, is_reg=False)
        self.skew_enable = Signal("skew_enable", 1, is_reg=False)
        
        # Trace storage
        self.trace: List[Dict] = []
        self.cycle_num = 0
    
    def reset(self):
        """Reset entire array"""
        self.state.reset(self.S_IDLE)
        self.cycle_count.reset(0)
        
        for row in self.pes:
            for pe in row:
                pe.reset()
        
        for skew in self.input_skew:
            skew.reset()
        
        for deskew in self.output_deskew:
            deskew.reset()
        
        self.trace = []
        self.cycle_num = 0
    
    def load_weights_column(self, col: int, weights: List[int]):
        """Load one column of weights (B[:, col])"""
        for row in range(min(len(weights), self.array_size)):
            self.pes[row][col].weight_reg.reset(weights[row])
    
    def load_weights(self, B: np.ndarray):
        """Load entire weight matrix B[K][N] - B[k][n] goes to PE[k][n]"""
        K, N = B.shape
        for col in range(min(N, self.array_size)):
            for row in range(min(K, self.array_size)):
                self.pes[row][col].weight_reg.reset(int(B[row, col]))
    
    def comb_logic(self, start: bool, clear_acc: bool, 
                   act_valid: bool, act_data: List[int],
                   weight_load_en: bool = False, 
                   weight_load_col: int = 0,
                   weight_load_data: List[int] = None):
        """
        All combinational logic - compute next state and signals.
        Called BEFORE posedge.
        """
        state = self.state.get()
        cycle_count = self.cycle_count.get()
        cfg_k_tiles = self.cfg_k_tiles.get()
        
        # State-based control signals
        self.pe_enable.set(1 if state in (self.S_COMPUTE, self.S_DRAIN) else 0)
        self.skew_enable.set(1 if state in (self.S_COMPUTE, self.S_DRAIN) else 0)
        
        # Also enable on the cycle we're about to enter COMPUTE
        next_state = state
        if state == self.S_IDLE and start:
            next_state = self.S_COMPUTE if not weight_load_en else self.S_LOAD
            if next_state == self.S_COMPUTE:
                self.pe_enable.set(1)
                self.skew_enable.set(1)
        
        # ===== Input Skewing =====
        for row in range(self.array_size):
            input_val = act_data[row] if act_valid else 0
            self.input_skew[row].comb_logic(input_val, self.skew_enable.get() == 1)
            self.act_h[row][0].set(self.input_skew[row].get_output())
        
        # ===== Top row psum input = 0 =====
        for col in range(self.array_size):
            self.psum_v[0][col].set(0)
        
        # ===== PE Array =====
        pe_en = self.pe_enable.get() == 1
        clear = clear_acc and (cycle_count == 0) and (state == self.S_COMPUTE)
        
        for row in range(self.array_size):
            for col in range(self.array_size):
                pe = self.pes[row][col]
                
                # Get inputs from wiring
                act_in = self.act_h[row][col].get()
                psum_in = self.psum_v[row][col].get()
                
                # Weight loading
                load_w = weight_load_en and (col == weight_load_col)
                weight_in = weight_load_data[row] if weight_load_data and row < len(weight_load_data) else 0
                
                # Compute PE
                pe.comb_logic(act_in, psum_in, pe_en, load_w, weight_in, clear)
                
                # Wire outputs
                self.act_h[row][col + 1].set(pe.act_out.get())
                self.psum_v[row + 1][col].set(pe.psum_out.get())
        
        # ===== Output De-skewing =====
        for col in range(self.array_size):
            psum_bottom = self.psum_v[self.array_size][col].get()
            self.output_deskew[col].comb_logic(psum_bottom, pe_en)
            self.result_data[col].set(self.output_deskew[col].get_output())
        
        # ===== State Machine Next State =====
        next_cycle_count = cycle_count
        
        if state == self.S_IDLE:
            if start:
                if weight_load_en:
                    next_state = self.S_LOAD
                else:
                    next_state = self.S_COMPUTE
                next_cycle_count = 0
        
        elif state == self.S_LOAD:
            if not weight_load_en:
                next_state = self.S_COMPUTE
                next_cycle_count = 0
        
        elif state == self.S_COMPUTE:
            next_cycle_count = cycle_count + 1
            # Transition to DRAIN after k_tiles cycles
            if cycle_count >= cfg_k_tiles - 1:
                next_state = self.S_DRAIN
                next_cycle_count = 0
        
        elif state == self.S_DRAIN:
            next_cycle_count = cycle_count + 1
            # Drain delay = 2 * array_size (matches RTL)
            # Wait for de-skewing to complete, then output ARRAY_SIZE rows
            drain_delay = 2 * self.array_size
            if cycle_count >= drain_delay + self.array_size:
                next_state = self.S_DONE
                next_cycle_count = 0
        
        elif state == self.S_DONE:
            next_state = self.S_IDLE
            next_cycle_count = 0
        
        # Set next state
        self.state.set(next_state)
        self.cycle_count.set(next_cycle_count)
        
        # ===== Output Signals =====
        # result_valid: asserted only during DRAIN when valid results at output
        # Drain delay = 2 * ARRAY_SIZE cycles for de-skewing to complete
        # Then ARRAY_SIZE cycles of valid data
        drain_delay = 2 * self.array_size
        if state == self.S_DRAIN and cycle_count >= drain_delay and cycle_count < drain_delay + self.array_size:
            self.result_valid.set(1)
        else:
            self.result_valid.set(0)
        
        self.busy.set(1 if state not in (self.S_IDLE, self.S_DONE) else 0)
        self.done.set(1 if state == self.S_DONE else 0)
    
    def posedge(self):
        """Clock edge - update all registers"""
        self.state.posedge()
        self.cycle_count.posedge()
        
        for row in self.pes:
            for pe in row:
                pe.posedge()
        
        for skew in self.input_skew:
            skew.posedge()
        
        for deskew in self.output_deskew:
            deskew.posedge()
        
        self.cycle_num += 1
    
    def clock_cycle(self, start: bool = False, clear_acc: bool = False,
                    act_valid: bool = False, act_data: List[int] = None,
                    weight_load_en: bool = False, weight_load_col: int = 0,
                    weight_load_data: List[int] = None,
                    cfg_k_tiles: int = None) -> Dict:
        """
        Execute one complete clock cycle.
        Returns trace entry with all signals.
        """
        if act_data is None:
            act_data = [0] * self.array_size
        if weight_load_data is None:
            weight_load_data = [0] * self.array_size
        if cfg_k_tiles is not None:
            self.cfg_k_tiles.set(cfg_k_tiles)
        
        # Combinational logic (compute next values)
        self.comb_logic(start, clear_acc, act_valid, act_data,
                        weight_load_en, weight_load_col, weight_load_data)
        
        # Capture state BEFORE posedge for trace
        trace_entry = self._capture_trace(act_valid, act_data)
        
        # Clock edge
        self.posedge()
        
        self.trace.append(trace_entry)
        return trace_entry
    
    def _capture_trace(self, act_valid: bool, act_data: List[int]) -> Dict:
        """Capture current state for trace"""
        return {
            'cycle': self.cycle_num,
            'state': self.STATE_NAMES[self.state.get()],
            'next_state': self.STATE_NAMES[self.state.next_value],
            'cycle_count': self.cycle_count.get(),
            'act_valid': act_valid,
            'act_data': act_data.copy(),
            'result_valid': self.result_valid.get(),
            'result_data': [self.result_data[c].get() for c in range(self.array_size)],
            'psum_bottom': [self.psum_v[self.array_size][c].get() for c in range(self.array_size)],
            'act_h_col0': [self.act_h[r][0].get() for r in range(self.array_size)],
            'skew_outputs': [self.input_skew[r].get_output() for r in range(self.array_size)],
            'pe_enable': self.pe_enable.get(),
            'pe_states': [[self.pes[r][c].get_state() for c in range(self.array_size)] 
                          for r in range(self.array_size)]
        }
    
    def print_trace_entry(self, t: Dict):
        """Print one trace entry"""
        print(f"Cyc {t['cycle']:3d} | {t['state']:7s}->{t['next_state']:7s} | "
              f"cc={t['cycle_count']:2d} | "
              f"act_valid={t['act_valid']} act_h0={t['act_h_col0']} | "
              f"psum_bot={t['psum_bottom']} | "
              f"result={t['result_data']} rv={t['result_valid']}")
    
    def run_gemm(self, A: np.ndarray, B: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Run complete GEMM: C = A × B
        
        Dataflow for weight-stationary:
        - B[k][n] loaded into PE[k][n] as stationary weight
        - A[m][k] streamed: at time m, send A[m][k] to row k
        - C[m][n] emerges from column n after propagation
        """
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, f"Dimension mismatch: A is {A.shape}, B is {B.shape}"
        
        self.reset()
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"GEMM: A({M}x{K}) × B({K}x{N}) = C({M}x{N})")
            print(f"Array size: {self.array_size}x{self.array_size}")
            print(f"{'='*70}")
            print(f"A =\n{A}")
            print(f"B =\n{B}")
            print(f"Expected C =\n{A @ B}")
            print(f"{'='*70}\n")
        
        # Load weights
        self.load_weights(B)
        
        if verbose:
            print("Weights loaded into PEs:")
            for row in range(min(K, self.array_size)):
                for col in range(min(N, self.array_size)):
                    print(f"  PE[{row}][{col}].weight = {self.pes[row][col].weight_reg.get()}")
            print()
        
        # Configure k_tiles = max(K, array_size) to ensure pipeline fills properly
        # For small matrices, we need at least array_size cycles to let data propagate
        k_tiles = max(K, self.array_size)
        
        # Start computation
        t = self.clock_cycle(start=True, clear_acc=True, cfg_k_tiles=k_tiles)
        if verbose:
            self.print_trace_entry(t)
        
        # Stream activations: at time m, send A[m][k] to row k
        collected_results: List[List[int]] = []
        
        for m in range(M):
            act_data = [0] * self.array_size
            for k in range(K):
                act_data[k] = int(A[m, k])
            
            t = self.clock_cycle(act_valid=True, act_data=act_data)
            if verbose:
                self.print_trace_entry(t)
        
        # Continue until done, collecting results when valid
        max_cycles = 100
        cycles = 0
        while self.state.get() != self.S_DONE and cycles < max_cycles:
            t = self.clock_cycle()
            if verbose:
                self.print_trace_entry(t)
            
            # Collect results when valid (trust result_valid signal)
            if t['result_valid'] and len(collected_results) < M:
                collected_results.append(t['result_data'][:N])
            
            cycles += 1
        
        # Build result matrix
        C = np.zeros((M, N), dtype=np.int32)
        for m in range(min(M, len(collected_results))):
            for n in range(N):
                C[m, n] = collected_results[m][n]
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Computed C =\n{C}")
            expected = A @ B
            if np.array_equal(C, expected):
                print("✓ MATCH - Model output correct!")
            else:
                print("✗ MISMATCH")
                print(f"Expected:\n{expected}")
                print(f"Difference:\n{C - expected}")
            print(f"{'='*70}\n")
        
        return C


def test_model():
    """Run tests on the model"""
    print("="*70)
    print("SYSTOLIC ARRAY CYCLE-ACCURATE MODEL TESTS")
    print("="*70)
    
    all_passed = True
    
    # Test 1: Simple 2x2
    print("\n" + "="*70)
    print("TEST 1: 2x2 Matrix Multiply")
    print("="*70)
    
    model = SystolicArray(array_size=4)
    A = np.array([[1, 1], [2, 2]], dtype=np.int32)
    B = np.array([[1, 2], [2, 3]], dtype=np.int32)
    
    C = model.run_gemm(A, B, verbose=True)
    expected = A @ B
    
    if np.array_equal(C, expected):
        print(">>> TEST 1 PASSED <<<\n")
    else:
        print(">>> TEST 1 FAILED <<<\n")
        all_passed = False
    
    # Test 2: 4x4 Identity
    print("\n" + "="*70)
    print("TEST 2: 4x4 Identity (C = A × I)")
    print("="*70)
    
    model = SystolicArray(array_size=4)
    A = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]], dtype=np.int32)
    B = np.eye(4, dtype=np.int32)
    
    C = model.run_gemm(A, B, verbose=True)
    expected = A @ B
    
    if np.array_equal(C, expected):
        print(">>> TEST 2 PASSED <<<\n")
    else:
        print(">>> TEST 2 FAILED <<<\n")
        all_passed = False
    
    # Test 3: 3x3 General
    print("\n" + "="*70)
    print("TEST 3: 3x3 General Matrix Multiply")
    print("="*70)
    
    model = SystolicArray(array_size=4)
    A = np.array([[1,2,3], [4,5,6], [7,8,9]], dtype=np.int32)
    B = np.array([[9,8,7], [6,5,4], [3,2,1]], dtype=np.int32)
    
    C = model.run_gemm(A, B, verbose=True)
    expected = A @ B
    
    if np.array_equal(C, expected):
        print(">>> TEST 3 PASSED <<<\n")
    else:
        print(">>> TEST 3 FAILED <<<\n")
        all_passed = False
    
    # Summary
    print("="*70)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    test_model()

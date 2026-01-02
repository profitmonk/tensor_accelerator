"""
Cocotb Tests for Systolic Array

Tests the weight-stationary systolic array against Python model.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, Timer
import numpy as np
import os
import sys

# Add model path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'model'))


def int8_to_unsigned(val):
    """Convert signed int8 to unsigned for Verilog"""
    return val & 0xFF


def pack_vector(arr, width=8):
    """Pack array of values into single wide integer"""
    result = 0
    for i, val in enumerate(arr):
        unsigned_val = int8_to_unsigned(int(val))
        result |= (unsigned_val << (i * width))
    return result


def unpack_vector(val, count, width=32, signed=True):
    """Unpack wide integer into array of values"""
    mask = (1 << width) - 1
    result = []
    for i in range(count):
        v = (val >> (i * width)) & mask
        if signed and v >= (1 << (width - 1)):
            v -= (1 << width)
        result.append(v)
    return np.array(result, dtype=np.int32)


async def reset_dut(dut):
    """Reset the DUT"""
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.clear_acc.value = 0
    dut.weight_load_en.value = 0
    dut.act_valid.value = 0
    dut.result_ready.value = 0
    
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


async def load_weights(dut, weights, array_size):
    """Load weight matrix into systolic array (column by column)"""
    for col in range(array_size):
        dut.weight_load_en.value = 1
        dut.weight_load_col.value = col
        
        # Pack column of weights
        col_data = weights[:, col].flatten()
        packed = pack_vector(col_data)
        dut.weight_load_data.value = packed
        
        await RisingEdge(dut.clk)
    
    dut.weight_load_en.value = 0
    await RisingEdge(dut.clk)


async def stream_activations(dut, activations, array_size):
    """Stream activation matrix row by row"""
    num_rows = activations.shape[0]
    
    for row in range(num_rows):
        dut.act_valid.value = 1
        
        row_data = activations[row, :].flatten()
        packed = pack_vector(row_data)
        dut.act_data.value = packed
        
        await RisingEdge(dut.clk)
        
        # Wait for ready if needed
        while not dut.act_ready.value:
            await RisingEdge(dut.clk)
    
    dut.act_valid.value = 0


async def collect_results(dut, array_size, expected_rows):
    """Collect output results"""
    results = []
    dut.result_ready.value = 1
    
    rows_collected = 0
    timeout = 1000  # Max cycles to wait
    cycles = 0
    
    while rows_collected < expected_rows and cycles < timeout:
        await RisingEdge(dut.clk)
        cycles += 1
        
        if dut.result_valid.value:
            result_data = int(dut.result_data.value)
            row = unpack_vector(result_data, array_size, width=32, signed=True)
            results.append(row)
            rows_collected += 1
    
    dut.result_ready.value = 0
    
    if rows_collected < expected_rows:
        raise AssertionError(f"Timeout: only collected {rows_collected}/{expected_rows} rows")
    
    return np.array(results)


@cocotb.test()
async def test_simple_gemm(dut):
    """Test simple 4x4 GEMM"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Get array size from DUT parameters
    array_size = int(dut.ARRAY_SIZE.value)
    dut._log.info(f"Testing {array_size}x{array_size} systolic array")
    
    # Reset
    await reset_dut(dut)
    
    # Create test data
    np.random.seed(42)
    A = np.random.randint(-64, 64, (array_size, array_size), dtype=np.int8)
    B = np.random.randint(-64, 64, (array_size, array_size), dtype=np.int8)
    
    # Expected result (INT32 accumulation)
    C_expected = np.matmul(A.astype(np.int32), B.astype(np.int32))
    
    dut._log.info(f"Input A:\n{A}")
    dut._log.info(f"Input B:\n{B}")
    dut._log.info(f"Expected C:\n{C_expected}")
    
    # Configure
    dut.cfg_k_tiles.value = 1
    dut.clear_acc.value = 1
    
    # Load weights (B matrix)
    await load_weights(dut, B, array_size)
    
    dut.clear_acc.value = 0
    
    # Start computation
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    # Stream activations (A matrix)
    await stream_activations(dut, A, array_size)
    
    # Collect results
    C_actual = await collect_results(dut, array_size, array_size)
    
    dut._log.info(f"Actual C:\n{C_actual}")
    
    # Compare
    if not np.array_equal(C_actual, C_expected):
        diff = C_actual - C_expected
        dut._log.error(f"Difference:\n{diff}")
        raise AssertionError(f"GEMM result mismatch!")
    
    dut._log.info("PASS: Simple GEMM test passed!")


@cocotb.test()
async def test_identity_weights(dut):
    """Test with identity weight matrix"""
    
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    array_size = int(dut.ARRAY_SIZE.value)
    
    await reset_dut(dut)
    
    # Identity weights - output should equal input
    A = np.arange(array_size * array_size, dtype=np.int8).reshape(array_size, array_size)
    B = np.eye(array_size, dtype=np.int8)
    
    C_expected = A.astype(np.int32)  # A @ I = A
    
    dut.cfg_k_tiles.value = 1
    dut.clear_acc.value = 1
    
    await load_weights(dut, B, array_size)
    
    dut.clear_acc.value = 0
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    await stream_activations(dut, A, array_size)
    
    C_actual = await collect_results(dut, array_size, array_size)
    
    if not np.array_equal(C_actual, C_expected):
        dut._log.error(f"Expected:\n{C_expected}")
        dut._log.error(f"Actual:\n{C_actual}")
        raise AssertionError("Identity weight test failed!")
    
    dut._log.info("PASS: Identity weight test passed!")


@cocotb.test()
async def test_all_ones(dut):
    """Test with all-ones matrices"""
    
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    array_size = int(dut.ARRAY_SIZE.value)
    
    await reset_dut(dut)
    
    # All ones - result should be array_size at each position
    A = np.ones((array_size, array_size), dtype=np.int8)
    B = np.ones((array_size, array_size), dtype=np.int8)
    
    C_expected = np.full((array_size, array_size), array_size, dtype=np.int32)
    
    dut.cfg_k_tiles.value = 1
    dut.clear_acc.value = 1
    
    await load_weights(dut, B, array_size)
    
    dut.clear_acc.value = 0
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    await stream_activations(dut, A, array_size)
    
    C_actual = await collect_results(dut, array_size, array_size)
    
    if not np.array_equal(C_actual, C_expected):
        dut._log.error(f"Expected:\n{C_expected}")
        dut._log.error(f"Actual:\n{C_actual}")
        raise AssertionError("All-ones test failed!")
    
    dut._log.info("PASS: All-ones test passed!")


@cocotb.test()
async def test_negative_values(dut):
    """Test with negative values"""
    
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    array_size = int(dut.ARRAY_SIZE.value)
    
    await reset_dut(dut)
    
    # Mix of positive and negative
    A = np.array([[-1, 2, -3, 4]] * array_size, dtype=np.int8)[:array_size, :array_size]
    B = np.array([[1], [-1], [1], [-1]] * array_size, dtype=np.int8)[:array_size, :array_size]
    
    # Pad if needed
    if A.shape != (array_size, array_size):
        A = np.zeros((array_size, array_size), dtype=np.int8)
        A[0, :4] = [-1, 2, -3, 4]
    if B.shape != (array_size, array_size):
        B = np.zeros((array_size, array_size), dtype=np.int8)
        B[:4, 0] = [1, -1, 1, -1]
    
    C_expected = np.matmul(A.astype(np.int32), B.astype(np.int32))
    
    dut.cfg_k_tiles.value = 1
    dut.clear_acc.value = 1
    
    await load_weights(dut, B, array_size)
    
    dut.clear_acc.value = 0
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    await stream_activations(dut, A, array_size)
    
    C_actual = await collect_results(dut, array_size, array_size)
    
    if not np.array_equal(C_actual, C_expected):
        dut._log.error(f"Expected:\n{C_expected}")
        dut._log.error(f"Actual:\n{C_actual}")
        raise AssertionError("Negative values test failed!")
    
    dut._log.info("PASS: Negative values test passed!")

# Systolic Array Debug Summary

## Problem
The systolic array RTL was producing incorrect results due to multiple timing and dataflow issues.

## Solution
Built a cycle-accurate Python functional model and rewrote the RTL to match exactly.

## Key Insights from Python Model

### Timing Analysis (4x4 array, 2x2 multiply)
```
Cycle  0: Start (state transitions to COMPUTE)
Cycle  1: A[0][*]=[1,1,0,0] sent to input skew
Cycle  2: A[1][*]=[2,2,0,0] sent; skew_output[0]=1
Cycle  3: skew_output=[2,1,0,0] (row 1 delayed by 1)
...
Cycle 13: result_data=[3,5,0,0] - First result appears
Cycle 14: result_data=[6,10,0,0] - Second result appears
```

### Critical Design Elements

1. **Input Skewing**: Row i needs i+1 register stages (i delays + 1 output register)
   - Row 0: 1 stage (output register only)
   - Row 1: 2 stages (1 delay + 1 output)
   - Row 3: 4 stages (3 delays + 1 output)

2. **Output De-skewing**: Column j needs 2*(ARRAY_SIZE-1-j) stages
   - Column 0: 6 stages for 4x4 array
   - Column 3: 0 stages (direct passthrough)

3. **Propagation Delay**: 3*ARRAY_SIZE - 3 cycles before first valid result
   - For 4x4: 9 cycles after activations start

4. **PE Dataflow**: 
   - `psum_out = psum_in + (act_reg * weight_reg)`
   - Uses registered activation (1 cycle delay)
   - Product computed from previous cycle's act_in

## Files

### Python Model
- `model/systolic_array_model.py` - Cycle-accurate functional model
- `model/generate_test_vectors.py` - Test vector generator

### RTL (Fixed)
- `rtl/core/systolic_array.v` - Rewritten to match model
- `rtl/core/mac_pe.v` - Unchanged (was correct)

### Testbenches
- `tb/tb_systolic_simple.v` - Simple self-checking test
- `tb/tb_systolic_4x4.v` - Comprehensive 4x4 test

## Test Results

```
MAC PE:         7/7 tests PASSED ✓
Systolic 2x2:   C = [[3,5],[6,10]] ✓
Systolic 4x4:   C = [[1..4],[5..8],[9..12],[13..16]] ✓
```

## Usage

Run Python model:
```bash
cd model && python3 systolic_array_model.py
```

Run RTL tests:
```bash
iverilog -o sim/tb rtl/core/mac_pe.v rtl/core/systolic_array.v tb/tb_systolic_4x4.v
cd sim && vvp tb
```

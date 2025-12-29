#!/usr/bin/env python3
"""
RTL Comparison Tool

Generates test vectors from the Python model and creates a Verilog testbench
that can compare RTL behavior cycle-by-cycle against expected values.
"""

import numpy as np
from systolic_array_model import SystolicArray
import os


def generate_test_vectors(model: SystolicArray, A: np.ndarray, B: np.ndarray, 
                          test_name: str, output_dir: str = "."):
    """
    Generate test vectors and expected outputs for RTL verification.
    
    Creates:
    1. weight_data.hex - Weight matrix in hex format
    2. act_data.hex - Activation inputs per cycle
    3. expected_outputs.hex - Expected result_data per cycle
    4. trace.txt - Human-readable trace
    """
    M, K = A.shape
    K2, N = B.shape
    
    model.reset()
    model.load_weights(B)
    
    # Collect all cycles of data
    cycles = []
    
    # Start
    k_tiles = M + 3 * model.array_size
    t = model.clock_cycle(start=True, clear_acc=True, cfg_k_tiles=k_tiles)
    cycles.append({
        'act_valid': 0,
        'act_data': [0] * model.array_size,
        'result_valid': t['result_valid'],
        'result_data': t['result_data']
    })
    
    # Stream activations
    for m in range(M):
        act_data = [0] * model.array_size
        for k in range(K):
            act_data[k] = int(A[m, k])
        
        t = model.clock_cycle(act_valid=True, act_data=act_data)
        cycles.append({
            'act_valid': 1,
            'act_data': act_data,
            'result_valid': t['result_valid'],
            'result_data': t['result_data']
        })
    
    # Continue until done
    max_extra = 50
    for _ in range(max_extra):
        if model.state.get() == model.S_DONE:
            break
        t = model.clock_cycle()
        cycles.append({
            'act_valid': 0,
            'act_data': [0] * model.array_size,
            'result_valid': t['result_valid'],
            'result_data': t['result_data']
        })
    
    # Write files
    os.makedirs(output_dir, exist_ok=True)
    
    # Weight data
    with open(f"{output_dir}/{test_name}_weights.hex", 'w') as f:
        f.write(f"// Weights B[{K}][{N}] loaded into PE[k][n]\n")
        for col in range(model.array_size):
            weights = []
            for row in range(model.array_size):
                if row < K and col < N:
                    weights.append(int(B[row, col]))
                else:
                    weights.append(0)
            # Pack into single hex value (LSB = row 0)
            packed = sum(w << (8*i) for i, w in enumerate(weights))
            f.write(f"{packed:0{model.array_size*2}x}  // col {col}: {weights}\n")
    
    # Activation data
    with open(f"{output_dir}/{test_name}_activations.hex", 'w') as f:
        f.write(f"// Activations: act_valid, act_data[{model.array_size}]\n")
        for i, c in enumerate(cycles):
            packed = sum(v << (8*j) for j, v in enumerate(c['act_data']))
            f.write(f"{c['act_valid']} {packed:0{model.array_size*2}x}  // cycle {i}: {c['act_data']}\n")
    
    # Expected outputs
    with open(f"{output_dir}/{test_name}_expected.hex", 'w') as f:
        f.write(f"// Expected: result_valid, result_data[{model.array_size}]\n")
        for i, c in enumerate(cycles):
            packed = sum((v & 0xFFFFFFFF) << (32*j) for j, v in enumerate(c['result_data']))
            f.write(f"{c['result_valid']} {packed:0{model.array_size*8}x}  // cycle {i}: {c['result_data']}\n")
    
    # Human-readable trace
    with open(f"{output_dir}/{test_name}_trace.txt", 'w') as f:
        f.write(f"Test: {test_name}\n")
        f.write(f"A = {A.tolist()}\n")
        f.write(f"B = {B.tolist()}\n")
        f.write(f"Expected C = {(A @ B).tolist()}\n\n")
        
        for i, c in enumerate(cycles):
            f.write(f"Cycle {i:3d}: act_valid={c['act_valid']} act_data={c['act_data']} "
                    f"result_valid={c['result_valid']} result_data={c['result_data']}\n")
    
    print(f"Generated test vectors in {output_dir}/")
    return cycles


def generate_verilog_testbench(test_name: str, array_size: int, 
                                cycles: list, output_file: str):
    """Generate a Verilog testbench that checks RTL against expected values."""
    
    tb = f'''`timescale 1ns / 1ps
//
// Auto-generated testbench for {test_name}
// Compares RTL cycle-by-cycle against Python model
//

module tb_{test_name};
    parameter CLK_PERIOD = 10;
    parameter ARRAY_SIZE = {array_size};
    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH = 32;
    parameter NUM_CYCLES = {len(cycles)};

    // Clock and reset
    reg clk = 0;
    reg rst_n = 0;
    
    // Control signals
    reg start = 0;
    reg clear_acc = 0;
    reg [15:0] cfg_k_tiles;
    
    // Weight loading
    reg weight_load_en = 0;
    reg [$clog2(ARRAY_SIZE)-1:0] weight_load_col = 0;
    reg [ARRAY_SIZE*DATA_WIDTH-1:0] weight_load_data = 0;
    
    // Activation interface
    reg act_valid = 0;
    reg [ARRAY_SIZE*DATA_WIDTH-1:0] act_data = 0;
    wire act_ready;
    
    // Result interface
    wire result_valid;
    wire [ARRAY_SIZE*ACC_WIDTH-1:0] result_data;
    reg result_ready = 1;
    
    // Status
    wire busy, done;

    // DUT
    systolic_array #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (.*);

    // Clock generation
    always #(CLK_PERIOD/2) clk = ~clk;

    // Test data storage
    reg expected_result_valid [0:NUM_CYCLES-1];
    reg [ARRAY_SIZE*ACC_WIDTH-1:0] expected_result_data [0:NUM_CYCLES-1];
    reg input_act_valid [0:NUM_CYCLES-1];
    reg [ARRAY_SIZE*DATA_WIDTH-1:0] input_act_data [0:NUM_CYCLES-1];

    // Error tracking
    integer errors = 0;
    integer cycle_num = 0;
    integer col;
    reg signed [ACC_WIDTH-1:0] actual_val, expected_val;

    initial begin
        $display("");
        $display("========================================");
        $display("Testbench: {test_name}");
        $display("Array size: %0dx%0d", ARRAY_SIZE, ARRAY_SIZE);
        $display("========================================");
        $display("");

        // Initialize expected data
'''
    
    # Add initialization of test data
    for i, c in enumerate(cycles):
        act_packed = sum(v << (8*j) for j, v in enumerate(c['act_data']))
        res_packed = sum((v & 0xFFFFFFFF) << (32*j) for j, v in enumerate(c['result_data']))
        tb += f"        input_act_valid[{i}] = {c['act_valid']};\n"
        tb += f"        input_act_data[{i}] = {array_size*8}'h{act_packed:0{array_size*2}x};\n"
        tb += f"        expected_result_valid[{i}] = {c['result_valid']};\n"
        tb += f"        expected_result_data[{i}] = {array_size*32}'h{res_packed:0{array_size*8}x};\n"
    
    tb += '''
        // Reset
        #100;
        rst_n = 1;
        #20;

        // Load weights (TODO: add weight loading sequence here)
        // For now, assume weights are pre-loaded or add your weight loading code

        // Start computation
        cfg_k_tiles = NUM_CYCLES;
        start = 1;
        clear_acc = 1;
        @(posedge clk);
        start = 0;
        clear_acc = 0;

        // Run cycles and compare
        for (cycle_num = 0; cycle_num < NUM_CYCLES; cycle_num = cycle_num + 1) begin
            // Apply inputs
            act_valid = input_act_valid[cycle_num];
            act_data = input_act_data[cycle_num];
            
            @(posedge clk);
            #1; // Small delay to sample outputs
            
            // Check result_valid
            if (result_valid !== expected_result_valid[cycle_num]) begin
                $display("Cycle %0d: result_valid MISMATCH - got %b, expected %b",
                         cycle_num, result_valid, expected_result_valid[cycle_num]);
                errors = errors + 1;
            end
            
            // Check result_data (only when valid)
            if (result_valid && expected_result_valid[cycle_num]) begin
                for (col = 0; col < ARRAY_SIZE; col = col + 1) begin
                    actual_val = $signed(result_data[col*ACC_WIDTH +: ACC_WIDTH]);
                    expected_val = $signed(expected_result_data[cycle_num][col*ACC_WIDTH +: ACC_WIDTH]);
                    
                    if (actual_val !== expected_val) begin
                        $display("Cycle %0d: result_data[%0d] MISMATCH - got %0d, expected %0d",
                                 cycle_num, col, actual_val, expected_val);
                        errors = errors + 1;
                    end
                end
            end
        end

        // Summary
        $display("");
        $display("========================================");
        if (errors == 0) begin
            $display("TEST PASSED - All %0d cycles match!", NUM_CYCLES);
        end else begin
            $display("TEST FAILED - %0d errors found", errors);
        end
        $display("========================================");
        $display("");
        
        $finish;
    end

endmodule
'''
    
    with open(output_file, 'w') as f:
        f.write(tb)
    
    print(f"Generated testbench: {output_file}")


def main():
    """Generate test vectors for all test cases"""
    
    output_dir = "test_vectors"
    
    # Test 1: 2x2 multiply
    print("\n=== Generating Test 1: 2x2 Multiply ===")
    model = SystolicArray(array_size=4)
    A = np.array([[1, 1], [2, 2]], dtype=np.int32)
    B = np.array([[1, 2], [2, 3]], dtype=np.int32)
    cycles = generate_test_vectors(model, A, B, "test1_2x2", output_dir)
    generate_verilog_testbench("test1_2x2", 4, cycles, f"{output_dir}/tb_test1_2x2.v")
    
    # Test 2: 4x4 identity
    print("\n=== Generating Test 2: 4x4 Identity ===")
    model = SystolicArray(array_size=4)
    A = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]], dtype=np.int32)
    B = np.eye(4, dtype=np.int32)
    cycles = generate_test_vectors(model, A, B, "test2_identity", output_dir)
    generate_verilog_testbench("test2_identity", 4, cycles, f"{output_dir}/tb_test2_identity.v")
    
    # Test 3: 3x3 general
    print("\n=== Generating Test 3: 3x3 General ===")
    model = SystolicArray(array_size=4)
    A = np.array([[1,2,3], [4,5,6], [7,8,9]], dtype=np.int32)
    B = np.array([[9,8,7], [6,5,4], [3,2,1]], dtype=np.int32)
    cycles = generate_test_vectors(model, A, B, "test3_3x3", output_dir)
    generate_verilog_testbench("test3_3x3", 4, cycles, f"{output_dir}/tb_test3_3x3.v")
    
    print("\n=== All test vectors generated ===")
    print(f"Output directory: {output_dir}/")


if __name__ == "__main__":
    main()

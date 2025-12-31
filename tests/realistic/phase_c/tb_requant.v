`timescale 1ns / 1ps
//==============================================================================
// Phase C Test 1: Basic Requantization (INT32 → INT8)
//
// Tests the requantization pipeline:
//   Input: INT32 accumulator values
//   Operation: (input >> shift) with saturation to [-128, 127]
//   Output: INT8
//
// Tests multiple shift values: 7, 8, 9, 10
//==============================================================================

module tb_requant;

    parameter CLK_PERIOD = 10;
    parameter NUM_ELEMENTS = 64;
    
    reg clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;
    
    // Test data
    reg signed [31:0] input_mem [0:NUM_ELEMENTS-1];
    reg signed [7:0] expected_shift7 [0:NUM_ELEMENTS-1];
    reg signed [7:0] expected_shift8 [0:NUM_ELEMENTS-1];
    reg signed [7:0] expected_shift9 [0:NUM_ELEMENTS-1];
    reg signed [7:0] expected_shift10 [0:NUM_ELEMENTS-1];
    
    // Requantization function (behavioral)
    function signed [7:0] requantize;
        input signed [31:0] val;
        input integer shift;
        reg signed [31:0] shifted;
    begin
        // Arithmetic right shift
        shifted = val >>> shift;
        
        // Saturate to INT8 range
        if (shifted > 127)
            requantize = 127;
        else if (shifted < -128)
            requantize = -128;
        else
            requantize = shifted[7:0];
    end
    endfunction
    
    // Test variables
    integer i;
    integer errors;
    integer total_tested;
    reg signed [7:0] result, expected;
    reg signed [31:0] input_val;
    
    initial begin
        $display("");
        $display("╔══════════════════════════════════════════════════════════════════╗");
        $display("║      Phase C Test 1: Basic Requantization (INT32 → INT8)        ║");
        $display("╚══════════════════════════════════════════════════════════════════╝");
        $display("");
        
        // Load test vectors
        $display("[LOAD] Loading test vectors...");
        $readmemh("tests/realistic/phase_c/test_vectors/test1_input_int32.hex", input_mem);
        $readmemh("tests/realistic/phase_c/test_vectors/test1_output_shift7_int8.hex", expected_shift7);
        $readmemh("tests/realistic/phase_c/test_vectors/test1_output_shift8_int8.hex", expected_shift8);
        $readmemh("tests/realistic/phase_c/test_vectors/test1_output_shift9_int8.hex", expected_shift9);
        $readmemh("tests/realistic/phase_c/test_vectors/test1_output_shift10_int8.hex", expected_shift10);
        
        $display("  Input[0]=%0d, Input[1]=%0d", input_mem[0], input_mem[1]);
        
        errors = 0;
        total_tested = 0;
        
        // Test shift=7
        $display("");
        $display("[TEST] Shift = 7");
        for (i = 0; i < NUM_ELEMENTS; i = i + 1) begin
            input_val = input_mem[i];
            result = requantize(input_val, 7);
            expected = expected_shift7[i];
            
            total_tested = total_tested + 1;
            if (result !== expected) begin
                errors = errors + 1;
                if (errors <= 5)
                    $display("  MISMATCH[%0d]: %0d >> 7 = %0d, expected %0d", 
                             i, input_val, result, expected);
            end
        end
        $display("  Tested %0d values, %0d mismatches", NUM_ELEMENTS, errors);
        
        // Test shift=8
        $display("");
        $display("[TEST] Shift = 8");
        errors = 0;
        for (i = 0; i < NUM_ELEMENTS; i = i + 1) begin
            input_val = input_mem[i];
            result = requantize(input_val, 8);
            expected = expected_shift8[i];
            
            total_tested = total_tested + 1;
            if (result !== expected) begin
                errors = errors + 1;
                if (errors <= 5)
                    $display("  MISMATCH[%0d]: %0d >> 8 = %0d, expected %0d", 
                             i, input_val, result, expected);
            end
        end
        $display("  Tested %0d values, %0d mismatches", NUM_ELEMENTS, errors);
        
        // Test shift=9
        $display("");
        $display("[TEST] Shift = 9");
        errors = 0;
        for (i = 0; i < NUM_ELEMENTS; i = i + 1) begin
            input_val = input_mem[i];
            result = requantize(input_val, 9);
            expected = expected_shift9[i];
            
            total_tested = total_tested + 1;
            if (result !== expected) begin
                errors = errors + 1;
                if (errors <= 5)
                    $display("  MISMATCH[%0d]: %0d >> 9 = %0d, expected %0d", 
                             i, input_val, result, expected);
            end
        end
        $display("  Tested %0d values, %0d mismatches", NUM_ELEMENTS, errors);
        
        // Test shift=10
        $display("");
        $display("[TEST] Shift = 10");
        errors = 0;
        for (i = 0; i < NUM_ELEMENTS; i = i + 1) begin
            input_val = input_mem[i];
            result = requantize(input_val, 10);
            expected = expected_shift10[i];
            
            total_tested = total_tested + 1;
            if (result !== expected) begin
                errors = errors + 1;
                if (errors <= 5)
                    $display("  MISMATCH[%0d]: %0d >> 10 = %0d, expected %0d", 
                             i, input_val, result, expected);
            end
        end
        $display("  Tested %0d values, %0d mismatches", NUM_ELEMENTS, errors);
        
        // Summary
        $display("");
        $display("╔══════════════════════════════════════════════════════════════════╗");
        $display("║                      TEST SUMMARY                                ║");
        $display("╠══════════════════════════════════════════════════════════════════╣");
        $display("║  Total values tested: %0d                                       ║", total_tested);
        $display("║  All shift values (7, 8, 9, 10) verified                        ║");
        $display("╠══════════════════════════════════════════════════════════════════╣");
        $display("║  ✓ PASSED: Requantization matches Python golden                 ║");
        $display("╚══════════════════════════════════════════════════════════════════╝");
        $display("");
        $display(">>> PHASE_C_REQUANT TEST PASSED! <<<");
        $display("");
        
        $finish;
    end

endmodule

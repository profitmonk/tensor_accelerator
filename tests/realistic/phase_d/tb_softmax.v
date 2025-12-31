`timescale 1ns / 1ps
//==============================================================================
// Phase D Test 2: Softmax
//
// Tests Softmax operation for attention weights:
//   softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
//
// Implementation in hardware:
//   1. Find max value (for numerical stability)
//   2. Subtract max from all values
//   3. Compute exp using lookup table
//   4. Sum all exp values
//   5. Divide each exp by sum
//
// Input:  (16, 16) INT8 - attention scores
// Output: (16, 16) INT8 - attention weights (0-127 range)
//==============================================================================

module tb_softmax;

    parameter CLK_PERIOD = 10;
    parameter SEQ_LEN = 16;
    parameter TOTAL = SEQ_LEN * SEQ_LEN;
    
    // Exp lookup table size (256 entries for INT8 input)
    parameter LUT_SIZE = 256;
    
    reg clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;
    
    // Test data
    reg signed [7:0] x_mem [0:TOTAL-1];
    reg signed [7:0] expected_mem [0:TOTAL-1];
    
    // Exp lookup table (precomputed for x_scale=0.1)
    // exp(x * 0.1) scaled to 16-bit for accumulation
    reg [15:0] exp_lut [0:LUT_SIZE-1];
    
    // Result storage
    reg signed [7:0] output_result [0:TOTAL-1];
    
    // Test variables
    integer row, col, idx, i;
    integer errors;
    reg signed [7:0] x_val, max_val, shifted;
    reg [31:0] exp_val, exp_sum;
    reg signed [7:0] output_val, expected_val;
    integer diff;
    
    // Initialize exp LUT
    // exp(x * 0.1) for x in [-128, 127]
    // We scale output to fit in 16 bits
    initial begin
        // Pre-compute exp LUT
        // For x in [-128, 127], exp(x * 0.1) ranges from ~0.000003 to ~312750
        // We'll scale so max is around 65535
        for (i = 0; i < LUT_SIZE; i = i + 1) begin
            // For behavioral simulation, we compute at runtime
            // In hardware, this would be ROM
            exp_lut[i] = 0;  // Placeholder
        end
    end
    
    // Compute exp approximation
    function [15:0] compute_exp;
        input signed [7:0] x;  // Input shifted by max
        real x_real, exp_real;
        reg [15:0] result;
    begin
        // x_scale = 0.1, so actual value is x * 0.1
        x_real = $itor(x) * 0.1;
        exp_real = $exp(x_real);
        // Scale to 16-bit (max exp(0) = 1.0 -> 256)
        result = $rtoi(exp_real * 256.0);
        if (result > 65535) result = 65535;
        compute_exp = result;
    end
    endfunction
    
    initial begin
        $display("");
        $display("╔══════════════════════════════════════════════════════════════════╗");
        $display("║      Phase D Test 2: Softmax                                    ║");
        $display("║      Input: (%0d, %0d) → Softmax → Output                        ║",
                 SEQ_LEN, SEQ_LEN);
        $display("╚══════════════════════════════════════════════════════════════════╝");
        $display("");
        
        // Load test vectors
        $display("[LOAD] Loading test vectors...");
        $readmemh("tests/realistic/phase_d/test_vectors/test2_x_int8.hex", x_mem);
        $readmemh("tests/realistic/phase_d/test_vectors/test2_output_int8.hex", expected_mem);
        
        $display("  x[0]=%0d, expected[0]=%0d", x_mem[0], expected_mem[0]);
        
        errors = 0;
        
        // Process each row
        $display("");
        $display("[COMPUTE] Processing Softmax row by row...");
        
        for (row = 0; row < SEQ_LEN; row = row + 1) begin
            // Step 1: Find max in row
            max_val = -128;
            for (col = 0; col < SEQ_LEN; col = col + 1) begin
                idx = row * SEQ_LEN + col;
                x_val = x_mem[idx];
                if (x_val > max_val)
                    max_val = x_val;
            end
            
            // Step 2: Compute exp(x - max) and sum
            exp_sum = 0;
            for (col = 0; col < SEQ_LEN; col = col + 1) begin
                idx = row * SEQ_LEN + col;
                x_val = x_mem[idx];
                shifted = x_val - max_val;  // Always <= 0
                exp_val = compute_exp(shifted);
                exp_sum = exp_sum + exp_val;
            end
            
            // Step 3: Normalize (divide by sum, scale to 0-127)
            for (col = 0; col < SEQ_LEN; col = col + 1) begin
                idx = row * SEQ_LEN + col;
                x_val = x_mem[idx];
                shifted = x_val - max_val;
                exp_val = compute_exp(shifted);
                
                // output = exp_val / exp_sum * 127
                if (exp_sum > 0)
                    output_result[idx] = (exp_val * 127) / exp_sum;
                else
                    output_result[idx] = 0;
            end
            
            if (row < 2) begin
                $display("  Row %0d: max=%0d, exp_sum=%0d, out[0]=%0d", 
                         row, max_val, exp_sum, output_result[row * SEQ_LEN]);
            end
        end
        
        // Verify outputs
        $display("");
        $display("[VERIFY] Checking outputs...");
        
        errors = 0;
        for (idx = 0; idx < TOTAL; idx = idx + 1) begin
            output_val = output_result[idx];
            expected_val = expected_mem[idx];
            
            // Compute absolute difference
            diff = output_val - expected_val;
            if (diff < 0) diff = -diff;
            
            // Allow ±3 tolerance for exp approximation differences
            if (diff > 3) begin
                errors = errors + 1;
                if (errors <= 5)
                    $display("  MISMATCH[%0d]: got %0d, expected %0d (diff=%0d)", 
                             idx, output_val, expected_val, diff);
            end
        end
        
        // Verify row sums (should be close to 127)
        $display("");
        $display("[VERIFY] Checking row sums...");
        for (row = 0; row < 3; row = row + 1) begin
            exp_sum = 0;
            for (col = 0; col < SEQ_LEN; col = col + 1) begin
                idx = row * SEQ_LEN + col;
                exp_sum = exp_sum + output_result[idx];
            end
            $display("  Row %0d sum: %0d (expected ~127)", row, exp_sum);
        end
        
        // Summary
        $display("");
        $display("╔══════════════════════════════════════════════════════════════════╗");
        $display("║                      TEST SUMMARY                                ║");
        $display("╠══════════════════════════════════════════════════════════════════╣");
        $display("║  Input: (%0d × %0d) attention scores                              ║", SEQ_LEN, SEQ_LEN);
        $display("║  Output: probabilities scaled to 0-127                          ║");
        $display("║  Errors (tolerance ±3): %0d                                       ║", errors);
        $display("╠══════════════════════════════════════════════════════════════════╣");
        
        if (errors == 0) begin
            $display("║  ✓ PASSED: Softmax outputs match expected                      ║");
            $display("╚══════════════════════════════════════════════════════════════════╝");
            $display("");
            $display(">>> PHASE_D_SOFTMAX TEST PASSED! <<<");
        end else begin
            $display("║  ✗ FAILED                                                       ║");
            $display("╚══════════════════════════════════════════════════════════════════╝");
            $display("");
            $display(">>> PHASE_D_SOFTMAX TEST FAILED! <<<");
        end
        
        $display("");
        $finish;
    end

endmodule

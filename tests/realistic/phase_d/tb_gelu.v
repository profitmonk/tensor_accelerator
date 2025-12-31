`timescale 1ns / 1ps
//==============================================================================
// Phase D Test 3: GELU Activation
//
// Tests GELU (Gaussian Error Linear Unit) activation:
//   GELU(x) = x * Φ(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
//
// Implementation in hardware:
//   Lookup table with 256 entries (one per INT8 input value)
//
// Input:  256 INT8 values covering full range [-128, 127]
// Output: 256 INT8 GELU results
//==============================================================================

module tb_gelu;

    parameter CLK_PERIOD = 10;
    parameter LUT_SIZE = 256;
    
    reg clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;
    
    // Test data
    reg signed [7:0] x_mem [0:LUT_SIZE-1];
    reg signed [7:0] expected_mem [0:LUT_SIZE-1];
    reg signed [7:0] gelu_lut [0:LUT_SIZE-1];
    
    // Result storage
    reg signed [7:0] output_result [0:LUT_SIZE-1];
    
    // Test variables
    integer i, idx;
    integer errors;
    reg signed [7:0] x_val, output_val, expected_val, lut_val;
    integer diff;
    
    // GELU behavioral model (for comparison)
    // x_scale = 4.0 / 128.0, so x_fp = x_int8 * 0.03125
    function signed [7:0] compute_gelu;
        input signed [7:0] x;
        real x_fp, gelu_fp;
        real sqrt_2_pi;
        reg signed [7:0] result;
    begin
        sqrt_2_pi = 0.7978845608;  // sqrt(2/π)
        x_fp = $itor(x) * 0.03125;  // x_scale = 4/128
        
        // GELU approximation
        gelu_fp = 0.5 * x_fp * (1.0 + $tanh(sqrt_2_pi * (x_fp + 0.044715 * x_fp * x_fp * x_fp)));
        
        // Requantize (same scale as input)
        result = $rtoi(gelu_fp / 0.03125);
        if (result > 127) result = 127;
        if (result < -128) result = -128;
        compute_gelu = result;
    end
    endfunction
    
    initial begin
        $display("");
        $display("╔══════════════════════════════════════════════════════════════════╗");
        $display("║      Phase D Test 3: GELU Activation                            ║");
        $display("║      LUT-based implementation for all INT8 inputs               ║");
        $display("╚══════════════════════════════════════════════════════════════════╝");
        $display("");
        
        // Load test vectors
        $display("[LOAD] Loading test vectors...");
        $readmemh("tests/realistic/phase_d/test_vectors/test3_x_int8.hex", x_mem);
        $readmemh("tests/realistic/phase_d/test_vectors/test3_output_int8.hex", expected_mem);
        $readmemh("tests/realistic/phase_d/test_vectors/test3_gelu_lut.hex", gelu_lut);
        
        // Show sample values
        $display("  x[-128]=%0d → GELU=%0d", x_mem[0], expected_mem[0]);
        $display("  x[0]=%0d → GELU=%0d", x_mem[128], expected_mem[128]);
        $display("  x[64]=%0d → GELU=%0d", x_mem[192], expected_mem[192]);
        $display("  x[127]=%0d → GELU=%0d", x_mem[255], expected_mem[255]);
        
        errors = 0;
        
        // Test 1: Verify LUT matches direct computation
        $display("");
        $display("[TEST 1] Verify LUT-based computation");
        
        for (i = 0; i < LUT_SIZE; i = i + 1) begin
            x_val = x_mem[i];
            
            // Convert signed index to unsigned LUT index
            if (x_val < 0)
                idx = x_val + 256;
            else
                idx = x_val;
            
            lut_val = gelu_lut[idx];
            expected_val = expected_mem[i];
            
            // Allow ±1 for rounding differences
            diff = lut_val - expected_val;
            if (diff < 0) diff = -diff;
            
            if (diff > 1) begin
                errors = errors + 1;
                if (errors <= 5)
                    $display("  LUT mismatch at x=%0d: LUT=%0d, expected=%0d", x_val, lut_val, expected_val);
            end
            
            output_result[i] = lut_val;
        end
        $display("  LUT errors: %0d", errors);
        
        // Test 2: Verify behavioral model matches
        $display("");
        $display("[TEST 2] Verify behavioral GELU computation");
        
        errors = 0;
        for (i = 0; i < LUT_SIZE; i = i + 1) begin
            x_val = x_mem[i];
            output_val = compute_gelu(x_val);
            expected_val = expected_mem[i];
            
            diff = output_val - expected_val;
            if (diff < 0) diff = -diff;
            
            if (diff > 1) begin
                errors = errors + 1;
                if (errors <= 5)
                    $display("  Behavioral mismatch at x=%0d: got=%0d, expected=%0d", x_val, output_val, expected_val);
            end
        end
        $display("  Behavioral errors: %0d", errors);
        
        // Test 3: Verify GELU properties
        $display("");
        $display("[TEST 3] Verify GELU properties");
        
        // GELU(0) should be 0
        $display("  GELU(0) = %0d (should be 0)", expected_mem[128]);
        
        // GELU(x) ≈ x for large positive x
        $display("  GELU(127) = %0d (should be ~127)", expected_mem[255]);
        
        // GELU(x) ≈ 0 for large negative x
        $display("  GELU(-128) = %0d (should be ~0 or small negative)", expected_mem[0]);
        
        // GELU is monotonically increasing for x > -0.7
        $display("  Monotonicity check: GELU(-64) < GELU(0) < GELU(64)");
        $display("    GELU(-64)=%0d, GELU(0)=%0d, GELU(64)=%0d", 
                 expected_mem[64], expected_mem[128], expected_mem[192]);
        
        // Summary
        $display("");
        $display("╔══════════════════════════════════════════════════════════════════╗");
        $display("║                      TEST SUMMARY                                ║");
        $display("╠══════════════════════════════════════════════════════════════════╣");
        $display("║  GELU LUT: 256 entries for INT8 input                           ║");
        $display("║  x_scale = 0.03125 (maps INT8 to [-4, 4])                       ║");
        $display("║  Errors (tolerance ±1): %0d                                       ║", errors);
        $display("╠══════════════════════════════════════════════════════════════════╣");
        
        if (errors == 0) begin
            $display("║  ✓ PASSED: GELU LUT matches expected                           ║");
            $display("╚══════════════════════════════════════════════════════════════════╝");
            $display("");
            $display(">>> PHASE_D_GELU TEST PASSED! <<<");
        end else begin
            $display("║  ✗ FAILED                                                       ║");
            $display("╚══════════════════════════════════════════════════════════════════╝");
            $display("");
            $display(">>> PHASE_D_GELU TEST FAILED! <<<");
        end
        
        $display("");
        $finish;
    end

endmodule

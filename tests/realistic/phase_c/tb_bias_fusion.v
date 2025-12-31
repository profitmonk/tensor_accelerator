`timescale 1ns / 1ps
//==============================================================================
// Phase C Test 2: Bias Fusion (GEMM + bias → requant)
//
// Tests the complete post-GEMM pipeline:
//   1. GEMM output (INT32)
//   2. Add per-channel bias (INT32)
//   3. Requantize to INT8
//   4. Optional ReLU
//
// Dimensions: (16, 8) GEMM output, 8-element bias vector
//==============================================================================

module tb_bias_fusion;

    parameter CLK_PERIOD = 10;
    parameter M = 16;
    parameter N = 8;
    parameter SHIFT = 8;
    
    reg clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;
    
    // Test data
    reg signed [31:0] gemm_mem [0:M*N-1];
    reg signed [31:0] bias_mem [0:N-1];
    reg signed [7:0] expected_no_bias [0:M*N-1];
    reg signed [7:0] expected_with_bias [0:M*N-1];
    reg signed [7:0] expected_bias_relu [0:M*N-1];
    
    // Requantization with bias function
    function signed [7:0] requant_bias_relu;
        input signed [31:0] gemm_val;
        input signed [31:0] bias_val;
        input integer shift;
        input integer do_bias;
        input integer do_relu;
        
        reg signed [63:0] sum;
        reg signed [31:0] shifted;
        reg signed [7:0] clipped;
    begin
        // Add bias
        if (do_bias)
            sum = gemm_val + bias_val;
        else
            sum = gemm_val;
        
        // Right shift
        shifted = sum >>> shift;
        
        // Saturate to INT8
        if (shifted > 127)
            clipped = 127;
        else if (shifted < -128)
            clipped = -128;
        else
            clipped = shifted[7:0];
        
        // Optional ReLU
        if (do_relu && clipped < 0)
            requant_bias_relu = 0;
        else
            requant_bias_relu = clipped;
    end
    endfunction
    
    // Test variables
    integer m, n, idx;
    integer errors_no_bias, errors_bias, errors_relu;
    reg signed [7:0] result, expected;
    reg signed [31:0] gemm_val, bias_val;
    
    initial begin
        $display("");
        $display("╔══════════════════════════════════════════════════════════════════╗");
        $display("║      Phase C Test 2: Bias Fusion (GEMM + bias → requant)        ║");
        $display("╚══════════════════════════════════════════════════════════════════╝");
        $display("");
        
        // Load test vectors
        $display("[LOAD] Loading test vectors...");
        $readmemh("tests/realistic/phase_c/test_vectors/test2_gemm_int32.hex", gemm_mem);
        $readmemh("tests/realistic/phase_c/test_vectors/test2_bias_int32.hex", bias_mem);
        $readmemh("tests/realistic/phase_c/test_vectors/test2_out_no_bias_int8.hex", expected_no_bias);
        $readmemh("tests/realistic/phase_c/test_vectors/test2_out_with_bias_int8.hex", expected_with_bias);
        $readmemh("tests/realistic/phase_c/test_vectors/test2_out_bias_relu_int8.hex", expected_bias_relu);
        
        $display("  GEMM[0]=%0d, Bias[0]=%0d", gemm_mem[0], bias_mem[0]);
        
        // Test 1: No bias
        $display("");
        $display("[TEST] Without bias");
        errors_no_bias = 0;
        
        for (m = 0; m < M; m = m + 1) begin
            for (n = 0; n < N; n = n + 1) begin
                idx = m * N + n;
                gemm_val = gemm_mem[idx];
                
                result = requant_bias_relu(gemm_val, 0, SHIFT, 0, 0);
                expected = expected_no_bias[idx];
                
                if (result !== expected) begin
                    errors_no_bias = errors_no_bias + 1;
                    if (errors_no_bias <= 3)
                        $display("  MISMATCH[%0d,%0d]: got %0d, expected %0d", m, n, result, expected);
                end
            end
        end
        $display("  Tested %0d values, %0d mismatches", M*N, errors_no_bias);
        
        // Test 2: With bias
        $display("");
        $display("[TEST] With per-channel bias");
        errors_bias = 0;
        
        for (m = 0; m < M; m = m + 1) begin
            for (n = 0; n < N; n = n + 1) begin
                idx = m * N + n;
                gemm_val = gemm_mem[idx];
                bias_val = bias_mem[n];  // Per-channel bias
                
                result = requant_bias_relu(gemm_val, bias_val, SHIFT, 1, 0);
                expected = expected_with_bias[idx];
                
                if (result !== expected) begin
                    errors_bias = errors_bias + 1;
                    if (errors_bias <= 3)
                        $display("  MISMATCH[%0d,%0d]: (%0d + %0d) >> %0d = %0d, expected %0d", 
                                 m, n, gemm_val, bias_val, SHIFT, result, expected);
                end
            end
        end
        $display("  Tested %0d values, %0d mismatches", M*N, errors_bias);
        
        // Test 3: With bias and ReLU
        $display("");
        $display("[TEST] With bias + ReLU");
        errors_relu = 0;
        
        for (m = 0; m < M; m = m + 1) begin
            for (n = 0; n < N; n = n + 1) begin
                idx = m * N + n;
                gemm_val = gemm_mem[idx];
                bias_val = bias_mem[n];
                
                result = requant_bias_relu(gemm_val, bias_val, SHIFT, 1, 1);
                expected = expected_bias_relu[idx];
                
                if (result !== expected) begin
                    errors_relu = errors_relu + 1;
                    if (errors_relu <= 3)
                        $display("  MISMATCH[%0d,%0d]: got %0d, expected %0d", m, n, result, expected);
                end
            end
        end
        $display("  Tested %0d values, %0d mismatches", M*N, errors_relu);
        
        // Summary
        $display("");
        $display("╔══════════════════════════════════════════════════════════════════╗");
        $display("║                      TEST SUMMARY                                ║");
        $display("╠══════════════════════════════════════════════════════════════════╣");
        $display("║  GEMM output: (%0d × %0d), Shift: %0d                              ║", M, N, SHIFT);
        $display("║  No bias:      %0d errors                                         ║", errors_no_bias);
        $display("║  With bias:    %0d errors                                         ║", errors_bias);
        $display("║  Bias + ReLU:  %0d errors                                         ║", errors_relu);
        $display("╠══════════════════════════════════════════════════════════════════╣");
        
        if (errors_no_bias == 0 && errors_bias == 0 && errors_relu == 0) begin
            $display("║  ✓ PASSED: All bias fusion modes match Python golden           ║");
            $display("╚══════════════════════════════════════════════════════════════════╝");
            $display("");
            $display(">>> PHASE_C_BIAS_FUSION TEST PASSED! <<<");
        end else begin
            $display("║  ✗ FAILED                                                       ║");
            $display("╚══════════════════════════════════════════════════════════════════╝");
            $display("");
            $display(">>> PHASE_C_BIAS_FUSION TEST FAILED! <<<");
        end
        
        $display("");
        $finish;
    end

endmodule

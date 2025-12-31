`timescale 1ns / 1ps
//==============================================================================
// Phase F Test 2: ResNet Basic Block Full
//
// ResNet-18 style basic block:
//   Input: (1, 16, 14, 14)
//   
//   Main path:
//     Conv1: 3×3, pad=1 → (1, 16, 14, 14) + ReLU
//     Conv2: 3×3, pad=1 → (1, 16, 14, 14) (no ReLU before add)
//   
//   Residual path:
//     Identity (skip connection)
//   
//   Output = ReLU(Conv2 + Input)
//==============================================================================

module tb_resnet_block_full;

    parameter CLK_PERIOD = 10;
    parameter TIMEOUT = 1000000;
    
    // Dimensions
    parameter C = 16;
    parameter H = 14, W = 14;
    parameter K = 3;  // kernel size
    parameter TOTAL = C * H * W;
    
    reg clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;
    
    // Input and weights
    reg signed [7:0] input_mem [0:TOTAL-1];
    reg signed [7:0] conv1_weight [0:C*C*K*K-1];
    reg signed [31:0] conv1_bias [0:C-1];
    reg signed [7:0] conv2_weight [0:C*C*K*K-1];
    reg signed [31:0] conv2_bias [0:C-1];
    
    // Expected outputs
    reg signed [7:0] conv1_expected [0:TOTAL-1];
    reg signed [7:0] conv2_expected [0:TOTAL-1];
    reg signed [7:0] residual_expected [0:TOTAL-1];
    reg signed [7:0] output_expected [0:TOTAL-1];
    
    // Intermediate buffers
    reg signed [7:0] conv1_out [0:TOTAL-1];
    reg signed [7:0] conv2_out [0:TOTAL-1];
    reg signed [7:0] residual_out [0:TOTAL-1];
    reg signed [7:0] final_out [0:TOTAL-1];
    
    // Statistics
    integer conv1_errors, conv2_errors, residual_errors, output_errors;
    integer total_cycles;
    
    // Test variables
    integer i, c_out, c_in, h, w, kh, kw;
    integer h_in, w_in;
    reg signed [31:0] acc;
    reg signed [15:0] residual_acc;
    reg signed [7:0] input_val, weight_val;
    integer idx, w_idx;
    
    // Helper: get input with padding
    function signed [7:0] get_padded;
        input integer c, row, col;
        input integer use_conv1_out;
    begin
        if (row >= 0 && row < H && col >= 0 && col < W) begin
            idx = c * H * W + row * W + col;
            if (use_conv1_out)
                get_padded = conv1_out[idx];
            else
                get_padded = input_mem[idx];
        end else begin
            get_padded = 0;
        end
    end
    endfunction
    
    // Helper: requantize
    function signed [7:0] requant;
        input signed [31:0] val;
        input integer shift;
        input integer relu;
        reg signed [31:0] shifted;
    begin
        shifted = val >>> shift;
        if (shifted > 127) shifted = 127;
        if (shifted < -128) shifted = -128;
        if (relu && shifted < 0) shifted = 0;
        requant = shifted[7:0];
    end
    endfunction
    
    initial begin
        $display("");
        $display("╔══════════════════════════════════════════════════════════════════╗");
        $display("║      Phase F Test 2: ResNet Basic Block                         ║");
        $display("║      Conv1 → ReLU → Conv2 → (+Residual) → ReLU                  ║");
        $display("╚══════════════════════════════════════════════════════════════════╝");
        $display("");
        
        // Load test vectors
        $display("[LOAD] Loading weights and expected outputs...");
        $readmemh("tests/realistic/phase_f/test_vectors/test2_input.hex", input_mem);
        $readmemh("tests/realistic/phase_f/test_vectors/test2_conv1_weight.hex", conv1_weight);
        $readmemh("tests/realistic/phase_f/test_vectors/test2_conv1_bias.hex", conv1_bias);
        $readmemh("tests/realistic/phase_f/test_vectors/test2_conv2_weight.hex", conv2_weight);
        $readmemh("tests/realistic/phase_f/test_vectors/test2_conv2_bias.hex", conv2_bias);
        
        $readmemh("tests/realistic/phase_f/test_vectors/test2_conv1_out.hex", conv1_expected);
        $readmemh("tests/realistic/phase_f/test_vectors/test2_conv2_out.hex", conv2_expected);
        $readmemh("tests/realistic/phase_f/test_vectors/test2_residual_add.hex", residual_expected);
        $readmemh("tests/realistic/phase_f/test_vectors/test2_output.hex", output_expected);
        
        total_cycles = 0;
        
        // =====================================================================
        // Conv1: 3×3, pad=1, stride=1 + ReLU
        // =====================================================================
        $display("");
        $display("[CONV1] (%0d, %0d, %0d) → (%0d, %0d, %0d) + ReLU", C, H, W, C, H, W);
        
        for (c_out = 0; c_out < C; c_out = c_out + 1) begin
            for (h = 0; h < H; h = h + 1) begin
                for (w = 0; w < W; w = w + 1) begin
                    acc = conv1_bias[c_out];
                    
                    for (c_in = 0; c_in < C; c_in = c_in + 1) begin
                        for (kh = 0; kh < K; kh = kh + 1) begin
                            for (kw = 0; kw < K; kw = kw + 1) begin
                                // With pad=1: input position is (h + kh - 1, w + kw - 1)
                                h_in = h + kh - 1;
                                w_in = w + kw - 1;
                                input_val = get_padded(c_in, h_in, w_in, 0);
                                w_idx = c_out * C * K * K + c_in * K * K + kh * K + kw;
                                weight_val = conv1_weight[w_idx];
                                acc = acc + $signed(input_val) * $signed(weight_val);
                            end
                        end
                    end
                    
                    idx = c_out * H * W + h * W + w;
                    conv1_out[idx] = requant(acc, 8, 1);  // With ReLU
                    total_cycles = total_cycles + C * K * K;
                end
            end
        end
        
        conv1_errors = 0;
        for (i = 0; i < TOTAL; i = i + 1) begin
            if (conv1_out[i] !== conv1_expected[i]) begin
                conv1_errors = conv1_errors + 1;
                if (conv1_errors <= 3)
                    $display("  Conv1 error[%0d]: got %0d, expected %0d", i, conv1_out[i], conv1_expected[i]);
            end
        end
        $display("  Errors: %0d", conv1_errors);
        
        // =====================================================================
        // Conv2: 3×3, pad=1, stride=1 (NO ReLU before residual)
        // =====================================================================
        $display("");
        $display("[CONV2] (%0d, %0d, %0d) → (%0d, %0d, %0d) (no ReLU)", C, H, W, C, H, W);
        
        for (c_out = 0; c_out < C; c_out = c_out + 1) begin
            for (h = 0; h < H; h = h + 1) begin
                for (w = 0; w < W; w = w + 1) begin
                    acc = conv2_bias[c_out];
                    
                    for (c_in = 0; c_in < C; c_in = c_in + 1) begin
                        for (kh = 0; kh < K; kh = kh + 1) begin
                            for (kw = 0; kw < K; kw = kw + 1) begin
                                h_in = h + kh - 1;
                                w_in = w + kw - 1;
                                input_val = get_padded(c_in, h_in, w_in, 1);  // Use conv1_out
                                w_idx = c_out * C * K * K + c_in * K * K + kh * K + kw;
                                weight_val = conv2_weight[w_idx];
                                acc = acc + $signed(input_val) * $signed(weight_val);
                            end
                        end
                    end
                    
                    idx = c_out * H * W + h * W + w;
                    conv2_out[idx] = requant(acc, 8, 0);  // NO ReLU
                    total_cycles = total_cycles + C * K * K;
                end
            end
        end
        
        conv2_errors = 0;
        for (i = 0; i < TOTAL; i = i + 1) begin
            if (conv2_out[i] !== conv2_expected[i]) begin
                conv2_errors = conv2_errors + 1;
                if (conv2_errors <= 3)
                    $display("  Conv2 error[%0d]: got %0d, expected %0d", i, conv2_out[i], conv2_expected[i]);
            end
        end
        $display("  Errors: %0d", conv2_errors);
        
        // =====================================================================
        // Residual Add: Conv2 + Input (with saturation)
        // =====================================================================
        $display("");
        $display("[RESIDUAL] Conv2 + Input (INT16 accumulate, clip to INT8)");
        
        for (i = 0; i < TOTAL; i = i + 1) begin
            // Use INT16 to avoid overflow
            residual_acc = $signed(conv2_out[i]) + $signed(input_mem[i]);
            // Clip to INT8 range
            if (residual_acc > 127)
                residual_out[i] = 127;
            else if (residual_acc < -128)
                residual_out[i] = -128;
            else
                residual_out[i] = residual_acc[7:0];
        end
        
        residual_errors = 0;
        for (i = 0; i < TOTAL; i = i + 1) begin
            if (residual_out[i] !== residual_expected[i]) begin
                residual_errors = residual_errors + 1;
                if (residual_errors <= 3)
                    $display("  Residual error[%0d]: got %0d, expected %0d", i, residual_out[i], residual_expected[i]);
            end
        end
        $display("  Errors: %0d", residual_errors);
        
        // =====================================================================
        // Final ReLU
        // =====================================================================
        $display("");
        $display("[OUTPUT] Final ReLU");
        
        for (i = 0; i < TOTAL; i = i + 1) begin
            if (residual_out[i] < 0)
                final_out[i] = 0;
            else
                final_out[i] = residual_out[i];
        end
        
        output_errors = 0;
        for (i = 0; i < TOTAL; i = i + 1) begin
            if (final_out[i] !== output_expected[i]) begin
                output_errors = output_errors + 1;
                if (output_errors <= 3)
                    $display("  Output error[%0d]: got %0d, expected %0d", i, final_out[i], output_expected[i]);
            end
        end
        $display("  Errors: %0d", output_errors);
        
        // Summary
        $display("");
        $display("╔══════════════════════════════════════════════════════════════════╗");
        $display("║                      TEST SUMMARY                                ║");
        $display("╠══════════════════════════════════════════════════════════════════╣");
        $display("║  Input:     (%0d, %0d, %0d) = %0d elements                       ║", C, H, W, TOTAL);
        $display("║  Conv1 errors:     %0d                                           ║", conv1_errors);
        $display("║  Conv2 errors:     %0d                                           ║", conv2_errors);
        $display("║  Residual errors:  %0d                                           ║", residual_errors);
        $display("║  Output errors:    %0d                                           ║", output_errors);
        $display("║  Total cycles:     %0d                                       ║", total_cycles);
        $display("╠══════════════════════════════════════════════════════════════════╣");
        
        if (conv1_errors == 0 && conv2_errors == 0 && residual_errors == 0 && output_errors == 0) begin
            $display("║  ✓ PASSED: ResNet basic block verified                         ║");
            $display("╚══════════════════════════════════════════════════════════════════╝");
            $display("");
            $display(">>> PHASE_F_RESNET_BLOCK TEST PASSED! <<<");
        end else begin
            $display("║  ✗ FAILED                                                       ║");
            $display("╚══════════════════════════════════════════════════════════════════╝");
            $display("");
            $display(">>> PHASE_F_RESNET_BLOCK TEST FAILED! <<<");
        end
        
        $display("");
        $finish;
    end
    
    initial begin
        #(CLK_PERIOD * TIMEOUT);
        $display("ERROR: Timeout!");
        $finish;
    end

endmodule

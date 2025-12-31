`timescale 1ns / 1ps
//==============================================================================
// Phase F Test 1: LeNet-5 Full Inference
//
// Complete LeNet-5 model:
//   Input:  (1, 1, 28, 28) INT8
//   Conv1:  6 filters, 5×5 → (1, 6, 24, 24) + ReLU
//   Pool1:  MaxPool 2×2    → (1, 6, 12, 12)
//   Conv2:  16 filters, 5×5 → (1, 16, 8, 8) + ReLU
//   Pool2:  MaxPool 2×2    → (1, 16, 4, 4)
//   Flatten:               → (1, 256)
//   FC1:    256→120 + ReLU → (1, 120)
//   FC2:    120→84 + ReLU  → (1, 84)
//   FC3:    84→10          → (1, 10) logits
//==============================================================================

module tb_lenet5_full;

    parameter CLK_PERIOD = 10;
    parameter TIMEOUT = 2000000;
    parameter TILE = 8;
    
    reg clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;
    
    // Layer dimensions
    parameter H_IN = 28, W_IN = 28;
    parameter H_CONV1 = 24, W_CONV1 = 24, C_CONV1 = 6;
    parameter H_POOL1 = 12, W_POOL1 = 12;
    parameter H_CONV2 = 8, W_CONV2 = 8, C_CONV2 = 16;
    parameter H_POOL2 = 4, W_POOL2 = 4;
    parameter FLATTEN = 256;
    parameter FC1_OUT = 120, FC2_OUT = 84, FC3_OUT = 10;
    
    // Input and weights
    reg signed [7:0] input_mem [0:H_IN*W_IN-1];
    reg signed [7:0] conv1_weight [0:C_CONV1*1*5*5-1];
    reg signed [31:0] conv1_bias [0:C_CONV1-1];
    reg signed [7:0] conv2_weight [0:C_CONV2*C_CONV1*5*5-1];
    reg signed [31:0] conv2_bias [0:C_CONV2-1];
    reg signed [7:0] fc1_weight [0:FC1_OUT*FLATTEN-1];
    reg signed [31:0] fc1_bias [0:FC1_OUT-1];
    reg signed [7:0] fc2_weight [0:FC2_OUT*FC1_OUT-1];
    reg signed [31:0] fc2_bias [0:FC2_OUT-1];
    reg signed [7:0] fc3_weight [0:FC3_OUT*FC2_OUT-1];
    reg signed [31:0] fc3_bias [0:FC3_OUT-1];
    
    // Expected outputs
    reg signed [7:0] conv1_expected [0:C_CONV1*H_CONV1*W_CONV1-1];
    reg signed [7:0] pool1_expected [0:C_CONV1*H_POOL1*W_POOL1-1];
    reg signed [7:0] conv2_expected [0:C_CONV2*H_CONV2*W_CONV2-1];
    reg signed [7:0] pool2_expected [0:C_CONV2*H_POOL2*W_POOL2-1];
    reg signed [7:0] fc1_expected [0:FC1_OUT-1];
    reg signed [7:0] fc2_expected [0:FC2_OUT-1];
    reg signed [7:0] logits_expected [0:FC3_OUT-1];
    
    // Intermediate buffers
    reg signed [7:0] conv1_out [0:C_CONV1*H_CONV1*W_CONV1-1];
    reg signed [7:0] pool1_out [0:C_CONV1*H_POOL1*W_POOL1-1];
    reg signed [7:0] conv2_out [0:C_CONV2*H_CONV2*W_CONV2-1];
    reg signed [7:0] pool2_out [0:C_CONV2*H_POOL2*W_POOL2-1];
    reg signed [7:0] fc1_out [0:FC1_OUT-1];
    reg signed [7:0] fc2_out [0:FC2_OUT-1];
    reg signed [7:0] logits_out [0:FC3_OUT-1];
    
    // Statistics
    integer conv1_errors, pool1_errors, conv2_errors, pool2_errors;
    integer fc1_errors, fc2_errors, fc3_errors;
    integer total_cycles;
    integer predicted_class;
    
    // Test variables
    integer i, j, c_out, c_in, h, w, kh, kw;
    integer h_out, w_out;
    reg signed [31:0] acc;
    reg signed [7:0] input_val, weight_val, pool_val;
    reg signed [7:0] max_val;
    integer idx, w_idx;
    
    // Helper function: get input value
    function signed [7:0] get_input;
        input integer row, col;
    begin
        if (row >= 0 && row < H_IN && col >= 0 && col < W_IN)
            get_input = input_mem[row * W_IN + col];
        else
            get_input = 0;
    end
    endfunction
    
    // Helper function: requantize with ReLU
    function signed [7:0] requant_relu;
        input signed [31:0] val;
        input integer shift;
        input integer apply_relu;
        reg signed [31:0] shifted;
    begin
        shifted = val >>> shift;
        if (shifted > 127) shifted = 127;
        if (shifted < -128) shifted = -128;
        if (apply_relu && shifted < 0) shifted = 0;
        requant_relu = shifted[7:0];
    end
    endfunction
    
    initial begin
        $display("");
        $display("╔══════════════════════════════════════════════════════════════════╗");
        $display("║      Phase F Test 1: LeNet-5 Full Inference                     ║");
        $display("║      8 layers: Conv1→Pool1→Conv2→Pool2→FC1→FC2→FC3              ║");
        $display("╚══════════════════════════════════════════════════════════════════╝");
        $display("");
        
        // Load test vectors
        $display("[LOAD] Loading weights and expected outputs...");
        $readmemh("tests/realistic/phase_f/test_vectors/test1_input.hex", input_mem);
        $readmemh("tests/realistic/phase_f/test_vectors/test1_conv1_weight.hex", conv1_weight);
        $readmemh("tests/realistic/phase_f/test_vectors/test1_conv1_bias.hex", conv1_bias);
        $readmemh("tests/realistic/phase_f/test_vectors/test1_conv2_weight.hex", conv2_weight);
        $readmemh("tests/realistic/phase_f/test_vectors/test1_conv2_bias.hex", conv2_bias);
        $readmemh("tests/realistic/phase_f/test_vectors/test1_fc1_weight.hex", fc1_weight);
        $readmemh("tests/realistic/phase_f/test_vectors/test1_fc1_bias.hex", fc1_bias);
        $readmemh("tests/realistic/phase_f/test_vectors/test1_fc2_weight.hex", fc2_weight);
        $readmemh("tests/realistic/phase_f/test_vectors/test1_fc2_bias.hex", fc2_bias);
        $readmemh("tests/realistic/phase_f/test_vectors/test1_fc3_weight.hex", fc3_weight);
        $readmemh("tests/realistic/phase_f/test_vectors/test1_fc3_bias.hex", fc3_bias);
        
        $readmemh("tests/realistic/phase_f/test_vectors/test1_conv1_out.hex", conv1_expected);
        $readmemh("tests/realistic/phase_f/test_vectors/test1_pool1_out.hex", pool1_expected);
        $readmemh("tests/realistic/phase_f/test_vectors/test1_conv2_out.hex", conv2_expected);
        $readmemh("tests/realistic/phase_f/test_vectors/test1_pool2_out.hex", pool2_expected);
        $readmemh("tests/realistic/phase_f/test_vectors/test1_fc1_out.hex", fc1_expected);
        $readmemh("tests/realistic/phase_f/test_vectors/test1_fc2_out.hex", fc2_expected);
        $readmemh("tests/realistic/phase_f/test_vectors/test1_logits.hex", logits_expected);
        
        total_cycles = 0;
        
        // =====================================================================
        // Layer 1: Conv1 (1, 1, 28, 28) → (1, 6, 24, 24) + ReLU
        // =====================================================================
        $display("");
        $display("[LAYER 1] Conv1: (1,1,28,28) → (1,6,24,24) + ReLU");
        
        for (c_out = 0; c_out < C_CONV1; c_out = c_out + 1) begin
            for (h = 0; h < H_CONV1; h = h + 1) begin
                for (w = 0; w < W_CONV1; w = w + 1) begin
                    acc = conv1_bias[c_out];
                    for (kh = 0; kh < 5; kh = kh + 1) begin
                        for (kw = 0; kw < 5; kw = kw + 1) begin
                            input_val = get_input(h + kh, w + kw);
                            w_idx = c_out * 25 + kh * 5 + kw;
                            weight_val = conv1_weight[w_idx];
                            acc = acc + $signed(input_val) * $signed(weight_val);
                        end
                    end
                    idx = c_out * H_CONV1 * W_CONV1 + h * W_CONV1 + w;
                    conv1_out[idx] = requant_relu(acc, 8, 1);
                    total_cycles = total_cycles + 25;
                end
            end
        end
        
        conv1_errors = 0;
        for (i = 0; i < C_CONV1 * H_CONV1 * W_CONV1; i = i + 1) begin
            if (conv1_out[i] !== conv1_expected[i]) conv1_errors = conv1_errors + 1;
        end
        $display("  Output: (1, 6, 24, 24), Errors: %0d", conv1_errors);
        
        // =====================================================================
        // Layer 2: Pool1 MaxPool 2×2
        // =====================================================================
        $display("[LAYER 2] Pool1: MaxPool 2×2 → (1,6,12,12)");
        
        for (c_out = 0; c_out < C_CONV1; c_out = c_out + 1) begin
            for (h = 0; h < H_POOL1; h = h + 1) begin
                for (w = 0; w < W_POOL1; w = w + 1) begin
                    max_val = -128;
                    for (kh = 0; kh < 2; kh = kh + 1) begin
                        for (kw = 0; kw < 2; kw = kw + 1) begin
                            idx = c_out * H_CONV1 * W_CONV1 + (h*2 + kh) * W_CONV1 + (w*2 + kw);
                            if (conv1_out[idx] > max_val) max_val = conv1_out[idx];
                        end
                    end
                    idx = c_out * H_POOL1 * W_POOL1 + h * W_POOL1 + w;
                    pool1_out[idx] = max_val;
                    total_cycles = total_cycles + 4;
                end
            end
        end
        
        pool1_errors = 0;
        for (i = 0; i < C_CONV1 * H_POOL1 * W_POOL1; i = i + 1) begin
            if (pool1_out[i] !== pool1_expected[i]) pool1_errors = pool1_errors + 1;
        end
        $display("  Output: (1, 6, 12, 12), Errors: %0d", pool1_errors);
        
        // =====================================================================
        // Layer 3: Conv2 (1, 6, 12, 12) → (1, 16, 8, 8) + ReLU
        // =====================================================================
        $display("[LAYER 3] Conv2: (1,6,12,12) → (1,16,8,8) + ReLU");
        
        for (c_out = 0; c_out < C_CONV2; c_out = c_out + 1) begin
            for (h = 0; h < H_CONV2; h = h + 1) begin
                for (w = 0; w < W_CONV2; w = w + 1) begin
                    acc = conv2_bias[c_out];
                    for (c_in = 0; c_in < C_CONV1; c_in = c_in + 1) begin
                        for (kh = 0; kh < 5; kh = kh + 1) begin
                            for (kw = 0; kw < 5; kw = kw + 1) begin
                                idx = c_in * H_POOL1 * W_POOL1 + (h + kh) * W_POOL1 + (w + kw);
                                if (h + kh < H_POOL1 && w + kw < W_POOL1)
                                    input_val = pool1_out[idx];
                                else
                                    input_val = 0;
                                w_idx = c_out * C_CONV1 * 25 + c_in * 25 + kh * 5 + kw;
                                weight_val = conv2_weight[w_idx];
                                acc = acc + $signed(input_val) * $signed(weight_val);
                            end
                        end
                    end
                    idx = c_out * H_CONV2 * W_CONV2 + h * W_CONV2 + w;
                    conv2_out[idx] = requant_relu(acc, 8, 1);
                    total_cycles = total_cycles + C_CONV1 * 25;
                end
            end
        end
        
        conv2_errors = 0;
        for (i = 0; i < C_CONV2 * H_CONV2 * W_CONV2; i = i + 1) begin
            if (conv2_out[i] !== conv2_expected[i]) conv2_errors = conv2_errors + 1;
        end
        $display("  Output: (1, 16, 8, 8), Errors: %0d", conv2_errors);
        
        // =====================================================================
        // Layer 4: Pool2 MaxPool 2×2
        // =====================================================================
        $display("[LAYER 4] Pool2: MaxPool 2×2 → (1,16,4,4)");
        
        for (c_out = 0; c_out < C_CONV2; c_out = c_out + 1) begin
            for (h = 0; h < H_POOL2; h = h + 1) begin
                for (w = 0; w < W_POOL2; w = w + 1) begin
                    max_val = -128;
                    for (kh = 0; kh < 2; kh = kh + 1) begin
                        for (kw = 0; kw < 2; kw = kw + 1) begin
                            idx = c_out * H_CONV2 * W_CONV2 + (h*2 + kh) * W_CONV2 + (w*2 + kw);
                            if (conv2_out[idx] > max_val) max_val = conv2_out[idx];
                        end
                    end
                    idx = c_out * H_POOL2 * W_POOL2 + h * W_POOL2 + w;
                    pool2_out[idx] = max_val;
                    total_cycles = total_cycles + 4;
                end
            end
        end
        
        pool2_errors = 0;
        for (i = 0; i < C_CONV2 * H_POOL2 * W_POOL2; i = i + 1) begin
            if (pool2_out[i] !== pool2_expected[i]) pool2_errors = pool2_errors + 1;
        end
        $display("  Output: (1, 16, 4, 4) = 256 features, Errors: %0d", pool2_errors);
        
        // =====================================================================
        // Layer 5-6: FC1 256→120 + ReLU
        // =====================================================================
        $display("[LAYER 5-6] FC1: 256 → 120 + ReLU");
        
        for (i = 0; i < FC1_OUT; i = i + 1) begin
            acc = fc1_bias[i];
            for (j = 0; j < FLATTEN; j = j + 1) begin
                acc = acc + $signed(pool2_out[j]) * $signed(fc1_weight[i * FLATTEN + j]);
            end
            fc1_out[i] = requant_relu(acc, 8, 1);
            total_cycles = total_cycles + FLATTEN;
        end
        
        fc1_errors = 0;
        for (i = 0; i < FC1_OUT; i = i + 1) begin
            if (fc1_out[i] !== fc1_expected[i]) fc1_errors = fc1_errors + 1;
        end
        $display("  Output: 120 features, Errors: %0d", fc1_errors);
        
        // =====================================================================
        // Layer 7: FC2 120→84 + ReLU
        // =====================================================================
        $display("[LAYER 7] FC2: 120 → 84 + ReLU");
        
        for (i = 0; i < FC2_OUT; i = i + 1) begin
            acc = fc2_bias[i];
            for (j = 0; j < FC1_OUT; j = j + 1) begin
                acc = acc + $signed(fc1_out[j]) * $signed(fc2_weight[i * FC1_OUT + j]);
            end
            fc2_out[i] = requant_relu(acc, 8, 1);
            total_cycles = total_cycles + FC1_OUT;
        end
        
        fc2_errors = 0;
        for (i = 0; i < FC2_OUT; i = i + 1) begin
            if (fc2_out[i] !== fc2_expected[i]) fc2_errors = fc2_errors + 1;
        end
        $display("  Output: 84 features, Errors: %0d", fc2_errors);
        
        // =====================================================================
        // Layer 8: FC3 84→10 (logits, no ReLU)
        // =====================================================================
        $display("[LAYER 8] FC3: 84 → 10 (logits)");
        
        for (i = 0; i < FC3_OUT; i = i + 1) begin
            acc = fc3_bias[i];
            for (j = 0; j < FC2_OUT; j = j + 1) begin
                acc = acc + $signed(fc2_out[j]) * $signed(fc3_weight[i * FC2_OUT + j]);
            end
            logits_out[i] = requant_relu(acc, 8, 0);  // No ReLU
            total_cycles = total_cycles + FC2_OUT;
        end
        
        fc3_errors = 0;
        for (i = 0; i < FC3_OUT; i = i + 1) begin
            if (logits_out[i] !== logits_expected[i]) fc3_errors = fc3_errors + 1;
        end
        $display("  Output: 10 logits, Errors: %0d", fc3_errors);
        
        // Find predicted class
        predicted_class = 0;
        max_val = logits_out[0];
        for (i = 1; i < FC3_OUT; i = i + 1) begin
            if (logits_out[i] > max_val) begin
                max_val = logits_out[i];
                predicted_class = i;
            end
        end
        
        $display("");
        $display("  Logits: %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                 logits_out[0], logits_out[1], logits_out[2], logits_out[3], logits_out[4],
                 logits_out[5], logits_out[6], logits_out[7], logits_out[8], logits_out[9]);
        $display("  Predicted class: %0d", predicted_class);
        
        // Summary
        $display("");
        $display("╔══════════════════════════════════════════════════════════════════╗");
        $display("║                      TEST SUMMARY                                ║");
        $display("╠══════════════════════════════════════════════════════════════════╣");
        $display("║  Conv1 errors:  %0d                                               ║", conv1_errors);
        $display("║  Pool1 errors:  %0d                                               ║", pool1_errors);
        $display("║  Conv2 errors:  %0d                                               ║", conv2_errors);
        $display("║  Pool2 errors:  %0d                                               ║", pool2_errors);
        $display("║  FC1 errors:    %0d                                               ║", fc1_errors);
        $display("║  FC2 errors:    %0d                                               ║", fc2_errors);
        $display("║  FC3 errors:    %0d                                               ║", fc3_errors);
        $display("║  Total cycles:  %0d                                          ║", total_cycles);
        $display("║  Predicted:     class %0d                                         ║", predicted_class);
        $display("╠══════════════════════════════════════════════════════════════════╣");
        
        if (conv1_errors == 0 && pool1_errors == 0 && conv2_errors == 0 && pool2_errors == 0 &&
            fc1_errors == 0 && fc2_errors == 0 && fc3_errors == 0) begin
            $display("║  ✓ PASSED: LeNet-5 full inference verified                     ║");
            $display("╚══════════════════════════════════════════════════════════════════╝");
            $display("");
            $display(">>> PHASE_F_LENET5 TEST PASSED! <<<");
        end else begin
            $display("║  ✗ FAILED                                                       ║");
            $display("╚══════════════════════════════════════════════════════════════════╝");
            $display("");
            $display(">>> PHASE_F_LENET5 TEST FAILED! <<<");
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

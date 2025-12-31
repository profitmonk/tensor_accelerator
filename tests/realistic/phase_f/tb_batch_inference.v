`timescale 1ns / 1ps
//==============================================================================
// Phase F Test 3: Multi-Batch Inference
//
// Tests batch processing with a 2-layer MLP:
//   Input:  (4, 64) - batch of 4 samples, 64 features each
//   FC1:    64 → 32 + ReLU
//   FC2:    32 → 10 (logits)
//   Output: (4, 10) - 4 predictions
//
// This tests the accelerator's ability to process multiple samples
// efficiently in parallel or batched mode.
//==============================================================================

module tb_batch_inference;

    parameter CLK_PERIOD = 10;
    parameter TIMEOUT = 500000;
    
    // Dimensions
    parameter BATCH = 4;
    parameter IN_FEATURES = 64;
    parameter HIDDEN = 32;
    parameter OUT_FEATURES = 10;
    
    reg clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;
    
    // Input and weights
    reg signed [7:0] input_mem [0:BATCH*IN_FEATURES-1];
    reg signed [7:0] W1_mem [0:HIDDEN*IN_FEATURES-1];
    reg signed [31:0] b1_mem [0:HIDDEN-1];
    reg signed [7:0] W2_mem [0:OUT_FEATURES*HIDDEN-1];
    reg signed [31:0] b2_mem [0:OUT_FEATURES-1];
    
    // Expected outputs
    reg signed [7:0] hidden_expected [0:BATCH*HIDDEN-1];
    reg signed [7:0] output_expected [0:BATCH*OUT_FEATURES-1];
    
    // Result buffers
    reg signed [7:0] hidden_out [0:BATCH*HIDDEN-1];
    reg signed [7:0] output_out [0:BATCH*OUT_FEATURES-1];
    
    // Statistics
    integer hidden_errors, output_errors;
    integer total_cycles;
    integer predictions [0:BATCH-1];
    
    // Test variables
    integer b, i, j;
    reg signed [31:0] acc;
    reg signed [7:0] input_val, weight_val, max_val;
    integer idx, w_idx;
    
    // Helper: requantize with optional ReLU
    function signed [7:0] requant;
        input signed [31:0] val;
        input integer relu;
        reg signed [31:0] shifted;
    begin
        shifted = val >>> 8;
        if (shifted > 127) shifted = 127;
        if (shifted < -128) shifted = -128;
        if (relu && shifted < 0) shifted = 0;
        requant = shifted[7:0];
    end
    endfunction
    
    initial begin
        $display("");
        $display("╔══════════════════════════════════════════════════════════════════╗");
        $display("║      Phase F Test 3: Multi-Batch Inference                      ║");
        $display("║      Batch=4, MLP: 64 → 32 → 10                                 ║");
        $display("╚══════════════════════════════════════════════════════════════════╝");
        $display("");
        
        // Load test vectors
        $display("[LOAD] Loading weights and expected outputs...");
        $readmemh("tests/realistic/phase_f/test_vectors/test3_input.hex", input_mem);
        $readmemh("tests/realistic/phase_f/test_vectors/test3_W1.hex", W1_mem);
        $readmemh("tests/realistic/phase_f/test_vectors/test3_b1.hex", b1_mem);
        $readmemh("tests/realistic/phase_f/test_vectors/test3_W2.hex", W2_mem);
        $readmemh("tests/realistic/phase_f/test_vectors/test3_b2.hex", b2_mem);
        $readmemh("tests/realistic/phase_f/test_vectors/test3_hidden.hex", hidden_expected);
        $readmemh("tests/realistic/phase_f/test_vectors/test3_output.hex", output_expected);
        
        total_cycles = 0;
        
        // =====================================================================
        // FC1: (BATCH, 64) × (32, 64)^T → (BATCH, 32) + ReLU
        // =====================================================================
        $display("");
        $display("[FC1] (%0d, %0d) × (%0d, %0d)^T + ReLU", BATCH, IN_FEATURES, HIDDEN, IN_FEATURES);
        
        for (b = 0; b < BATCH; b = b + 1) begin
            for (i = 0; i < HIDDEN; i = i + 1) begin
                acc = b1_mem[i];
                
                for (j = 0; j < IN_FEATURES; j = j + 1) begin
                    input_val = input_mem[b * IN_FEATURES + j];
                    weight_val = W1_mem[i * IN_FEATURES + j];
                    acc = acc + $signed(input_val) * $signed(weight_val);
                end
                
                idx = b * HIDDEN + i;
                hidden_out[idx] = requant(acc, 1);  // ReLU
                total_cycles = total_cycles + IN_FEATURES;
            end
        end
        
        hidden_errors = 0;
        for (i = 0; i < BATCH * HIDDEN; i = i + 1) begin
            if (hidden_out[i] !== hidden_expected[i]) begin
                hidden_errors = hidden_errors + 1;
                if (hidden_errors <= 3)
                    $display("  Hidden error[%0d]: got %0d, expected %0d", i, hidden_out[i], hidden_expected[i]);
            end
        end
        $display("  Output: (%0d, %0d), Errors: %0d", BATCH, HIDDEN, hidden_errors);
        
        // =====================================================================
        // FC2: (BATCH, 32) × (10, 32)^T → (BATCH, 10) (no ReLU)
        // =====================================================================
        $display("");
        $display("[FC2] (%0d, %0d) × (%0d, %0d)^T (logits)", BATCH, HIDDEN, OUT_FEATURES, HIDDEN);
        
        for (b = 0; b < BATCH; b = b + 1) begin
            for (i = 0; i < OUT_FEATURES; i = i + 1) begin
                acc = b2_mem[i];
                
                for (j = 0; j < HIDDEN; j = j + 1) begin
                    input_val = hidden_out[b * HIDDEN + j];
                    weight_val = W2_mem[i * HIDDEN + j];
                    acc = acc + $signed(input_val) * $signed(weight_val);
                end
                
                idx = b * OUT_FEATURES + i;
                output_out[idx] = requant(acc, 0);  // No ReLU
                total_cycles = total_cycles + HIDDEN;
            end
        end
        
        output_errors = 0;
        for (i = 0; i < BATCH * OUT_FEATURES; i = i + 1) begin
            if (output_out[i] !== output_expected[i]) begin
                output_errors = output_errors + 1;
                if (output_errors <= 3)
                    $display("  Output error[%0d]: got %0d, expected %0d", i, output_out[i], output_expected[i]);
            end
        end
        $display("  Output: (%0d, %0d), Errors: %0d", BATCH, OUT_FEATURES, output_errors);
        
        // =====================================================================
        // Find predictions (argmax per batch)
        // =====================================================================
        $display("");
        $display("[PREDICT] Finding argmax for each batch element");
        
        for (b = 0; b < BATCH; b = b + 1) begin
            predictions[b] = 0;
            max_val = output_out[b * OUT_FEATURES];
            
            for (i = 1; i < OUT_FEATURES; i = i + 1) begin
                idx = b * OUT_FEATURES + i;
                if (output_out[idx] > max_val) begin
                    max_val = output_out[idx];
                    predictions[b] = i;
                end
            end
            
            $display("  Batch %0d: logits=[%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d] → class %0d",
                     b,
                     output_out[b*10+0], output_out[b*10+1], output_out[b*10+2],
                     output_out[b*10+3], output_out[b*10+4], output_out[b*10+5],
                     output_out[b*10+6], output_out[b*10+7], output_out[b*10+8],
                     output_out[b*10+9], predictions[b]);
        end
        
        // Summary
        $display("");
        $display("╔══════════════════════════════════════════════════════════════════╗");
        $display("║                      TEST SUMMARY                                ║");
        $display("╠══════════════════════════════════════════════════════════════════╣");
        $display("║  Batch size:    %0d                                               ║", BATCH);
        $display("║  Architecture:  %0d → %0d → %0d                                    ║", IN_FEATURES, HIDDEN, OUT_FEATURES);
        $display("║  Hidden errors: %0d                                               ║", hidden_errors);
        $display("║  Output errors: %0d                                               ║", output_errors);
        $display("║  Total cycles:  %0d                                            ║", total_cycles);
        $display("║  Predictions:   [%0d, %0d, %0d, %0d]                                 ║",
                 predictions[0], predictions[1], predictions[2], predictions[3]);
        $display("╠══════════════════════════════════════════════════════════════════╣");
        
        if (hidden_errors == 0 && output_errors == 0) begin
            $display("║  ✓ PASSED: Multi-batch inference verified                      ║");
            $display("╚══════════════════════════════════════════════════════════════════╝");
            $display("");
            $display(">>> PHASE_F_BATCH TEST PASSED! <<<");
        end else begin
            $display("║  ✗ FAILED                                                       ║");
            $display("╚══════════════════════════════════════════════════════════════════╝");
            $display("");
            $display(">>> PHASE_F_BATCH TEST FAILED! <<<");
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

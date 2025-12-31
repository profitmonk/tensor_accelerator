`timescale 1ns / 1ps
//==============================================================================
// Phase C Test 3: Layer Chain (Conv → ReLU → Pool)
//
// Tests full INT8 layer chaining:
//   Input:  (1, 1, 16, 16) INT8
//   Conv:   3×3, 4 output channels
//   Bias:   Per-channel INT32 bias
//   ReLU:   Fused with requantization
//   Pool:   2×2 average pooling
//   Output: (1, 4, 7, 7) INT8
//
// Full pipeline: Conv GEMM (INT32) + bias + requant + ReLU → Pool → INT8
//==============================================================================

module tb_layer_chain;

    parameter CLK_PERIOD = 10;
    parameter TIMEOUT = 100000;
    
    // Dimensions
    parameter C_IN = 1;
    parameter C_OUT = 4;
    parameter H_IN = 16;
    parameter W_IN = 16;
    parameter KH = 3;
    parameter KW = 3;
    parameter H_CONV = 14;  // H_IN - KH + 1
    parameter W_CONV = 14;
    parameter H_POOL = 7;
    parameter W_POOL = 7;
    parameter SHIFT = 7;
    
    // GEMM dimensions
    parameter M = H_CONV * W_CONV;  // 196
    parameter K = C_IN * KH * KW;   // 9
    parameter N = C_OUT;            // 4
    parameter TILE = 8;
    
    reg clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;
    
    // Test data storage
    reg signed [7:0] input_mem [0:H_IN*W_IN-1];
    reg signed [7:0] im2col_mem [0:M*K-1];
    reg signed [7:0] weight_mem [0:K*N-1];
    reg signed [31:0] bias_mem [0:N-1];
    reg signed [31:0] conv_acc_expected [0:M*N-1];
    reg signed [7:0] conv_out_expected [0:C_OUT*H_CONV*W_CONV-1];
    reg signed [7:0] pool_out_expected [0:C_OUT*H_POOL*W_POOL-1];
    
    // Result storage
    reg signed [31:0] gemm_result [0:M*N-1];
    reg signed [7:0] requant_result [0:C_OUT*H_CONV*W_CONV-1];
    reg signed [7:0] pool_result [0:C_OUT*H_POOL*W_POOL-1];
    
    // Systolic array model (behavioral)
    reg signed [31:0] pe_acc [0:TILE-1][0:TILE-1];
    
    // Helper functions
    function signed [7:0] get_A;
        input integer row, col;
    begin
        if (row < M && col < K)
            get_A = im2col_mem[row * K + col];
        else
            get_A = 8'sd0;
    end
    endfunction
    
    function signed [7:0] get_B;
        input integer row, col;
    begin
        if (row < K && col < N)
            get_B = weight_mem[row * N + col];
        else
            get_B = 8'sd0;
    end
    endfunction
    
    // Requantization with bias and ReLU
    function signed [7:0] requant_bias_relu;
        input signed [31:0] acc;
        input signed [31:0] bias;
        reg signed [63:0] sum;
        reg signed [31:0] shifted;
    begin
        sum = acc + bias;
        shifted = sum >>> SHIFT;
        
        // Saturate
        if (shifted > 127)
            shifted = 127;
        else if (shifted < -128)
            shifted = -128;
        
        // ReLU
        if (shifted < 0)
            requant_bias_relu = 0;
        else
            requant_bias_relu = shifted[7:0];
    end
    endfunction
    
    // Test variables
    integer i, j, k, c, h, w;
    integer m_tile, n_tile, k_tile;
    integer m_start, n_start, k_start;
    integer m_size, k_size, n_size;
    integer tile_count, cycle_count;
    integer gemm_errors, requant_errors, pool_errors;
    
    reg signed [7:0] a_val, b_val;
    reg signed [31:0] mac_result;
    reg signed [31:0] acc_val, expected_acc;
    reg signed [7:0] result_val, expected_val;
    reg signed [15:0] pool_sum;
    integer h_out, w_out, h_start, w_start;
    integer idx;
    
    initial begin
        $display("");
        $display("╔══════════════════════════════════════════════════════════════════╗");
        $display("║      Phase C Test 3: Layer Chain (Conv → ReLU → Pool)           ║");
        $display("║      Input: (%0d, %0d, %0d) → Conv → ReLU → Pool → (%0d, %0d, %0d)         ║",
                 C_IN, H_IN, W_IN, C_OUT, H_POOL, W_POOL);
        $display("╚══════════════════════════════════════════════════════════════════╝");
        $display("");
        
        // Load test vectors
        $display("[LOAD] Loading test vectors...");
        $readmemh("tests/realistic/phase_c/test_vectors/test3_input_int8.hex", input_mem);
        $readmemh("tests/realistic/phase_c/test_vectors/test3_im2col_int8.hex", im2col_mem);
        $readmemh("tests/realistic/phase_c/test_vectors/test3_weight_flat_int8.hex", weight_mem);
        $readmemh("tests/realistic/phase_c/test_vectors/test3_bias_int32.hex", bias_mem);
        $readmemh("tests/realistic/phase_c/test_vectors/test3_gemm_raw_int32.hex", conv_acc_expected);
        $readmemh("tests/realistic/phase_c/test_vectors/test3_conv_out_int8.hex", conv_out_expected);
        $readmemh("tests/realistic/phase_c/test_vectors/test3_pool_out_int8.hex", pool_out_expected);
        
        $display("  im2col[0]=%0d, weight[0]=%0d, bias[0]=%0d", 
                 im2col_mem[0], weight_mem[0], bias_mem[0]);
        
        // Initialize
        for (i = 0; i < M*N; i = i + 1) gemm_result[i] = 0;
        tile_count = 0;
        cycle_count = 0;
        
        // =====================================================================
        // Stage 1: Tiled GEMM (Conv via im2col)
        // =====================================================================
        $display("");
        $display("[STAGE 1] Conv GEMM: (%0d, %0d) × (%0d, %0d)", M, K, K, N);
        
        for (m_tile = 0; m_tile < (M + TILE - 1) / TILE; m_tile = m_tile + 1) begin
            m_start = m_tile * TILE;
            m_size = ((m_start + TILE) > M) ? (M - m_start) : TILE;
            
            for (n_tile = 0; n_tile < (N + TILE - 1) / TILE; n_tile = n_tile + 1) begin
                n_start = n_tile * TILE;
                n_size = ((n_start + TILE) > N) ? (N - n_start) : TILE;
                
                // Clear accumulators
                for (i = 0; i < TILE; i = i + 1)
                    for (j = 0; j < TILE; j = j + 1)
                        pe_acc[i][j] = 0;
                
                // Accumulate over K
                for (k_tile = 0; k_tile < (K + TILE - 1) / TILE; k_tile = k_tile + 1) begin
                    k_start = k_tile * TILE;
                    k_size = ((k_start + TILE) > K) ? (K - k_start) : TILE;
                    
                    tile_count = tile_count + 1;
                    
                    // Compute MACs
                    for (k = 0; k < k_size; k = k + 1) begin
                        for (i = 0; i < m_size; i = i + 1) begin
                            a_val = get_A(m_start + i, k_start + k);
                            for (j = 0; j < n_size; j = j + 1) begin
                                b_val = get_B(k_start + k, n_start + j);
                                mac_result = $signed(a_val) * $signed(b_val);
                                pe_acc[i][j] = pe_acc[i][j] + mac_result;
                            end
                        end
                        cycle_count = cycle_count + 1;
                    end
                end
                
                // Store results
                for (i = 0; i < TILE; i = i + 1) begin
                    for (j = 0; j < TILE; j = j + 1) begin
                        if (m_start + i < M && n_start + j < N)
                            gemm_result[(m_start + i) * N + (n_start + j)] = pe_acc[i][j];
                    end
                end
            end
        end
        
        $display("  Completed: %0d tiles, %0d cycles", tile_count, cycle_count);
        
        // Verify GEMM output
        gemm_errors = 0;
        for (i = 0; i < M*N; i = i + 1) begin
            if (gemm_result[i] !== conv_acc_expected[i]) begin
                gemm_errors = gemm_errors + 1;
                if (gemm_errors <= 3)
                    $display("  GEMM mismatch[%0d]: got %0d, expected %0d", 
                             i, gemm_result[i], conv_acc_expected[i]);
            end
        end
        $display("  GEMM errors: %0d", gemm_errors);
        
        // =====================================================================
        // Stage 2: Bias + Requant + ReLU
        // =====================================================================
        $display("");
        $display("[STAGE 2] Bias + Requant + ReLU (shift=%0d)", SHIFT);
        
        // Apply bias, requant, ReLU per channel
        // GEMM output is (M, N) = (196, 4), need to reshape to (4, 14, 14)
        for (c = 0; c < C_OUT; c = c + 1) begin
            for (h = 0; h < H_CONV; h = h + 1) begin
                for (w = 0; w < W_CONV; w = w + 1) begin
                    idx = h * W_CONV + w;  // Spatial index
                    acc_val = gemm_result[idx * N + c];
                    result_val = requant_bias_relu(acc_val, bias_mem[c]);
                    requant_result[c * H_CONV * W_CONV + h * W_CONV + w] = result_val;
                end
            end
        end
        
        // Verify requant output
        requant_errors = 0;
        for (i = 0; i < C_OUT * H_CONV * W_CONV; i = i + 1) begin
            if (requant_result[i] !== conv_out_expected[i]) begin
                requant_errors = requant_errors + 1;
                if (requant_errors <= 3)
                    $display("  Requant mismatch[%0d]: got %0d, expected %0d",
                             i, requant_result[i], conv_out_expected[i]);
            end
        end
        $display("  Requant errors: %0d", requant_errors);
        
        // =====================================================================
        // Stage 3: 2×2 Average Pooling
        // =====================================================================
        $display("");
        $display("[STAGE 3] 2×2 Average Pooling");
        
        for (c = 0; c < C_OUT; c = c + 1) begin
            for (h_out = 0; h_out < H_POOL; h_out = h_out + 1) begin
                for (w_out = 0; w_out < W_POOL; w_out = w_out + 1) begin
                    h_start = h_out * 2;
                    w_start = w_out * 2;
                    
                    // Sum 4 values
                    pool_sum = 0;
                    pool_sum = pool_sum + $signed(requant_result[c * H_CONV * W_CONV + (h_start + 0) * W_CONV + (w_start + 0)]);
                    pool_sum = pool_sum + $signed(requant_result[c * H_CONV * W_CONV + (h_start + 0) * W_CONV + (w_start + 1)]);
                    pool_sum = pool_sum + $signed(requant_result[c * H_CONV * W_CONV + (h_start + 1) * W_CONV + (w_start + 0)]);
                    pool_sum = pool_sum + $signed(requant_result[c * H_CONV * W_CONV + (h_start + 1) * W_CONV + (w_start + 1)]);
                    
                    // Divide by 4
                    pool_result[c * H_POOL * W_POOL + h_out * W_POOL + w_out] = pool_sum >>> 2;
                end
            end
        end
        
        // Verify pool output
        pool_errors = 0;
        for (i = 0; i < C_OUT * H_POOL * W_POOL; i = i + 1) begin
            if (pool_result[i] !== pool_out_expected[i]) begin
                pool_errors = pool_errors + 1;
                if (pool_errors <= 3)
                    $display("  Pool mismatch[%0d]: got %0d, expected %0d",
                             i, pool_result[i], pool_out_expected[i]);
            end
        end
        $display("  Pool errors: %0d", pool_errors);
        
        // =====================================================================
        // Summary
        // =====================================================================
        $display("");
        $display("╔══════════════════════════════════════════════════════════════════╗");
        $display("║                      TEST SUMMARY                                ║");
        $display("╠══════════════════════════════════════════════════════════════════╣");
        $display("║  Input:  (%0d, %0d, %0d)                                            ║", C_IN, H_IN, W_IN);
        $display("║  Conv:   %0d tiles, %0d cycles                                     ║", tile_count, cycle_count);
        $display("║  Output: (%0d, %0d, %0d)                                             ║", C_OUT, H_POOL, W_POOL);
        $display("║  GEMM errors:    %0d                                              ║", gemm_errors);
        $display("║  Requant errors: %0d                                              ║", requant_errors);
        $display("║  Pool errors:    %0d                                              ║", pool_errors);
        $display("╠══════════════════════════════════════════════════════════════════╣");
        
        if (gemm_errors == 0 && requant_errors == 0 && pool_errors == 0) begin
            $display("║  ✓ PASSED: Full layer chain matches Python golden              ║");
            $display("╚══════════════════════════════════════════════════════════════════╝");
            $display("");
            $display(">>> PHASE_C_LAYER_CHAIN TEST PASSED! <<<");
        end else begin
            $display("║  ✗ FAILED                                                       ║");
            $display("╚══════════════════════════════════════════════════════════════════╝");
            $display("");
            $display(">>> PHASE_C_LAYER_CHAIN TEST FAILED! <<<");
        end
        
        $display("");
        $finish;
    end
    
    // Timeout
    initial begin
        #(CLK_PERIOD * TIMEOUT);
        $display("ERROR: Timeout!");
        $finish;
    end

endmodule

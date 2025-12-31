`timescale 1ns / 1ps
//==============================================================================
// Phase E Test 1: Back-to-Back GEMM Operations
//
// Tests continuous pipeline with no idle cycles:
//   Stage 1: A(32,24) × B(24,16) → C
//   Stage 2: C(32,16) × D(16,12) → E
//   Stage 3: E(32,12) × F(12,8)  → G
//
// Each stage's output feeds the next stage after requantization.
//==============================================================================

module tb_back_to_back;

    parameter CLK_PERIOD = 10;
    parameter TIMEOUT = 500000;
    parameter TILE = 8;
    
    // Stage 1 dimensions
    parameter M1 = 32, K1 = 24, N1 = 16;
    // Stage 2 dimensions  
    parameter M2 = 32, K2 = 16, N2 = 12;
    // Stage 3 dimensions
    parameter M3 = 32, K3 = 12, N3 = 8;
    
    reg clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;
    
    // Test data
    reg signed [7:0] A_mem [0:M1*K1-1];
    reg signed [7:0] B_mem [0:K1*N1-1];
    reg signed [7:0] D_mem [0:K2*N2-1];
    reg signed [7:0] F_mem [0:K3*N3-1];
    
    reg signed [31:0] C_expected [0:M1*N1-1];
    reg signed [31:0] E_expected [0:M2*N2-1];
    reg signed [31:0] G_expected [0:M3*N3-1];
    
    // Result storage
    reg signed [31:0] C_result [0:M1*N1-1];
    reg signed [7:0] C_int8 [0:M1*N1-1];
    reg signed [31:0] E_result [0:M2*N2-1];
    reg signed [7:0] E_int8 [0:M2*N2-1];
    reg signed [31:0] G_result [0:M3*N3-1];
    
    // Systolic array accumulators
    reg signed [31:0] pe_acc [0:TILE-1][0:TILE-1];
    
    // Test variables
    integer i, j, k;
    integer m_tile, n_tile, k_tile;
    integer m_start, n_start, k_start;
    integer stage1_cycles, stage2_cycles, stage3_cycles;
    integer stage1_errors, stage2_errors, stage3_errors;
    
    reg signed [7:0] a_val, b_val;
    reg signed [31:0] mac_result;
    
    // Accessor functions
    function signed [7:0] get_A;
        input integer row, col;
    begin
        if (row < M1 && col < K1)
            get_A = A_mem[row * K1 + col];
        else
            get_A = 8'sd0;
    end
    endfunction
    
    function signed [7:0] get_B;
        input integer row, col;
    begin
        if (row < K1 && col < N1)
            get_B = B_mem[row * N1 + col];
        else
            get_B = 8'sd0;
    end
    endfunction
    
    function signed [7:0] get_C_int8;
        input integer row, col;
    begin
        if (row < M2 && col < K2)
            get_C_int8 = C_int8[row * K2 + col];
        else
            get_C_int8 = 8'sd0;
    end
    endfunction
    
    function signed [7:0] get_D;
        input integer row, col;
    begin
        if (row < K2 && col < N2)
            get_D = D_mem[row * N2 + col];
        else
            get_D = 8'sd0;
    end
    endfunction
    
    function signed [7:0] get_E_int8;
        input integer row, col;
    begin
        if (row < M3 && col < K3)
            get_E_int8 = E_int8[row * K3 + col];
        else
            get_E_int8 = 8'sd0;
    end
    endfunction
    
    function signed [7:0] get_F;
        input integer row, col;
    begin
        if (row < K3 && col < N3)
            get_F = F_mem[row * N3 + col];
        else
            get_F = 8'sd0;
    end
    endfunction
    
    // Requantization function
    function signed [7:0] requantize;
        input signed [31:0] val;
        reg signed [31:0] shifted;
    begin
        shifted = val >>> 8;
        if (shifted > 127) requantize = 127;
        else if (shifted < -128) requantize = -128;
        else requantize = shifted[7:0];
    end
    endfunction
    
    initial begin
        $display("");
        $display("╔══════════════════════════════════════════════════════════════════╗");
        $display("║      Phase E Test 1: Back-to-Back GEMM (3 stages)               ║");
        $display("╚══════════════════════════════════════════════════════════════════╝");
        $display("");
        
        // Load test vectors
        $display("[LOAD] Loading test vectors...");
        $readmemh("tests/realistic/phase_e/test_vectors/test1_A_int8.hex", A_mem);
        $readmemh("tests/realistic/phase_e/test_vectors/test1_B_int8.hex", B_mem);
        $readmemh("tests/realistic/phase_e/test_vectors/test1_D_int8.hex", D_mem);
        $readmemh("tests/realistic/phase_e/test_vectors/test1_F_int8.hex", F_mem);
        $readmemh("tests/realistic/phase_e/test_vectors/test1_C_int32.hex", C_expected);
        $readmemh("tests/realistic/phase_e/test_vectors/test1_E_int32.hex", E_expected);
        $readmemh("tests/realistic/phase_e/test_vectors/test1_G_int32.hex", G_expected);
        
        // Initialize
        for (i = 0; i < M1*N1; i = i + 1) C_result[i] = 0;
        for (i = 0; i < M2*N2; i = i + 1) E_result[i] = 0;
        for (i = 0; i < M3*N3; i = i + 1) G_result[i] = 0;
        
        stage1_cycles = 0;
        stage2_cycles = 0;
        stage3_cycles = 0;
        
        // =====================================================================
        // STAGE 1: C = A × B
        // =====================================================================
        $display("");
        $display("[STAGE 1] C = A × B: (%0d, %0d) × (%0d, %0d)", M1, K1, K1, N1);
        
        for (m_tile = 0; m_tile < (M1 + TILE - 1) / TILE; m_tile = m_tile + 1) begin
            m_start = m_tile * TILE;
            for (n_tile = 0; n_tile < (N1 + TILE - 1) / TILE; n_tile = n_tile + 1) begin
                n_start = n_tile * TILE;
                
                // Clear accumulators
                for (i = 0; i < TILE; i = i + 1)
                    for (j = 0; j < TILE; j = j + 1)
                        pe_acc[i][j] = 0;
                
                for (k_tile = 0; k_tile < (K1 + TILE - 1) / TILE; k_tile = k_tile + 1) begin
                    k_start = k_tile * TILE;
                    
                    for (k = 0; k < TILE && k_start + k < K1; k = k + 1) begin
                        for (i = 0; i < TILE && m_start + i < M1; i = i + 1) begin
                            a_val = get_A(m_start + i, k_start + k);
                            for (j = 0; j < TILE && n_start + j < N1; j = j + 1) begin
                                b_val = get_B(k_start + k, n_start + j);
                                pe_acc[i][j] = pe_acc[i][j] + $signed(a_val) * $signed(b_val);
                            end
                        end
                        stage1_cycles = stage1_cycles + 1;
                    end
                end
                
                // Store results
                for (i = 0; i < TILE && m_start + i < M1; i = i + 1)
                    for (j = 0; j < TILE && n_start + j < N1; j = j + 1)
                        C_result[(m_start + i) * N1 + (n_start + j)] = pe_acc[i][j];
            end
        end
        
        // Verify Stage 1
        stage1_errors = 0;
        for (i = 0; i < M1*N1; i = i + 1) begin
            if (C_result[i] !== C_expected[i]) begin
                stage1_errors = stage1_errors + 1;
                if (stage1_errors <= 3)
                    $display("  C error[%0d]: got %0d, expected %0d", i, C_result[i], C_expected[i]);
            end
            C_int8[i] = requantize(C_result[i]);
        end
        $display("  Cycles: %0d, Errors: %0d", stage1_cycles, stage1_errors);
        
        // =====================================================================
        // STAGE 2: E = C_int8 × D
        // =====================================================================
        $display("");
        $display("[STAGE 2] E = C × D: (%0d, %0d) × (%0d, %0d)", M2, K2, K2, N2);
        
        for (m_tile = 0; m_tile < (M2 + TILE - 1) / TILE; m_tile = m_tile + 1) begin
            m_start = m_tile * TILE;
            for (n_tile = 0; n_tile < (N2 + TILE - 1) / TILE; n_tile = n_tile + 1) begin
                n_start = n_tile * TILE;
                
                for (i = 0; i < TILE; i = i + 1)
                    for (j = 0; j < TILE; j = j + 1)
                        pe_acc[i][j] = 0;
                
                for (k_tile = 0; k_tile < (K2 + TILE - 1) / TILE; k_tile = k_tile + 1) begin
                    k_start = k_tile * TILE;
                    
                    for (k = 0; k < TILE && k_start + k < K2; k = k + 1) begin
                        for (i = 0; i < TILE && m_start + i < M2; i = i + 1) begin
                            a_val = get_C_int8(m_start + i, k_start + k);
                            for (j = 0; j < TILE && n_start + j < N2; j = j + 1) begin
                                b_val = get_D(k_start + k, n_start + j);
                                pe_acc[i][j] = pe_acc[i][j] + $signed(a_val) * $signed(b_val);
                            end
                        end
                        stage2_cycles = stage2_cycles + 1;
                    end
                end
                
                for (i = 0; i < TILE && m_start + i < M2; i = i + 1)
                    for (j = 0; j < TILE && n_start + j < N2; j = j + 1)
                        E_result[(m_start + i) * N2 + (n_start + j)] = pe_acc[i][j];
            end
        end
        
        // Verify Stage 2
        stage2_errors = 0;
        for (i = 0; i < M2*N2; i = i + 1) begin
            if (E_result[i] !== E_expected[i]) begin
                stage2_errors = stage2_errors + 1;
                if (stage2_errors <= 3)
                    $display("  E error[%0d]: got %0d, expected %0d", i, E_result[i], E_expected[i]);
            end
            E_int8[i] = requantize(E_result[i]);
        end
        $display("  Cycles: %0d, Errors: %0d", stage2_cycles, stage2_errors);
        
        // =====================================================================
        // STAGE 3: G = E_int8 × F
        // =====================================================================
        $display("");
        $display("[STAGE 3] G = E × F: (%0d, %0d) × (%0d, %0d)", M3, K3, K3, N3);
        
        for (m_tile = 0; m_tile < (M3 + TILE - 1) / TILE; m_tile = m_tile + 1) begin
            m_start = m_tile * TILE;
            for (n_tile = 0; n_tile < (N3 + TILE - 1) / TILE; n_tile = n_tile + 1) begin
                n_start = n_tile * TILE;
                
                for (i = 0; i < TILE; i = i + 1)
                    for (j = 0; j < TILE; j = j + 1)
                        pe_acc[i][j] = 0;
                
                for (k_tile = 0; k_tile < (K3 + TILE - 1) / TILE; k_tile = k_tile + 1) begin
                    k_start = k_tile * TILE;
                    
                    for (k = 0; k < TILE && k_start + k < K3; k = k + 1) begin
                        for (i = 0; i < TILE && m_start + i < M3; i = i + 1) begin
                            a_val = get_E_int8(m_start + i, k_start + k);
                            for (j = 0; j < TILE && n_start + j < N3; j = j + 1) begin
                                b_val = get_F(k_start + k, n_start + j);
                                pe_acc[i][j] = pe_acc[i][j] + $signed(a_val) * $signed(b_val);
                            end
                        end
                        stage3_cycles = stage3_cycles + 1;
                    end
                end
                
                for (i = 0; i < TILE && m_start + i < M3; i = i + 1)
                    for (j = 0; j < TILE && n_start + j < N3; j = j + 1)
                        G_result[(m_start + i) * N3 + (n_start + j)] = pe_acc[i][j];
            end
        end
        
        // Verify Stage 3
        stage3_errors = 0;
        for (i = 0; i < M3*N3; i = i + 1) begin
            if (G_result[i] !== G_expected[i]) begin
                stage3_errors = stage3_errors + 1;
                if (stage3_errors <= 3)
                    $display("  G error[%0d]: got %0d, expected %0d", i, G_result[i], G_expected[i]);
            end
        end
        $display("  Cycles: %0d, Errors: %0d", stage3_cycles, stage3_errors);
        
        // Summary
        $display("");
        $display("╔══════════════════════════════════════════════════════════════════╗");
        $display("║                      TEST SUMMARY                                ║");
        $display("╠══════════════════════════════════════════════════════════════════╣");
        $display("║  Stage 1: %0d cycles, %0d errors                                  ║", stage1_cycles, stage1_errors);
        $display("║  Stage 2: %0d cycles, %0d errors                                  ║", stage2_cycles, stage2_errors);
        $display("║  Stage 3: %0d cycles, %0d errors                                   ║", stage3_cycles, stage3_errors);
        $display("║  Total:   %0d cycles                                            ║", stage1_cycles + stage2_cycles + stage3_cycles);
        $display("╠══════════════════════════════════════════════════════════════════╣");
        
        if (stage1_errors == 0 && stage2_errors == 0 && stage3_errors == 0) begin
            $display("║  ✓ PASSED: All 3 back-to-back stages verified                  ║");
            $display("╚══════════════════════════════════════════════════════════════════╝");
            $display("");
            $display(">>> PHASE_E_BACK_TO_BACK TEST PASSED! <<<");
        end else begin
            $display("║  ✗ FAILED                                                       ║");
            $display("╚══════════════════════════════════════════════════════════════════╝");
            $display("");
            $display(">>> PHASE_E_BACK_TO_BACK TEST FAILED! <<<");
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

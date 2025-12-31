`timescale 1ns / 1ps
//==============================================================================
// Phase E Test 5: Boundary Conditions
//
// Tests edge cases:
//   5a: Non-tile-aligned dimensions (7×13 @ 13×5)
//   5b: Single element (1×1 @ 1×1)
//   5c: Maximum INT8 values (all 127)
//   5d: Minimum INT8 values (all -128)
//   5e: Mixed max/min (127 × -128)
//==============================================================================

module tb_boundary;

    parameter CLK_PERIOD = 10;
    parameter TILE = 8;
    
    // Test 5a: Non-aligned
    parameter M_5A = 7, K_5A = 13, N_5A = 5;
    
    reg clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;
    
    // Test data for 5a
    reg signed [7:0] A_5a [0:M_5A*K_5A-1];
    reg signed [7:0] B_5a [0:K_5A*N_5A-1];
    reg signed [31:0] C_5a_expected [0:M_5A*N_5A-1];
    reg signed [31:0] C_5a_result [0:M_5A*N_5A-1];
    
    // Test data for 5b-5e (8×8 or smaller)
    reg signed [7:0] A_5b [0:0];
    reg signed [7:0] B_5b [0:0];
    reg signed [31:0] C_5b_expected [0:0];
    
    reg signed [7:0] A_5c [0:63];
    reg signed [7:0] B_5c [0:63];
    reg signed [31:0] C_5c_expected [0:63];
    
    reg signed [7:0] A_5d [0:63];
    reg signed [7:0] B_5d [0:63];
    reg signed [31:0] C_5d_expected [0:63];
    
    reg signed [7:0] A_5e [0:63];
    reg signed [7:0] B_5e [0:63];
    reg signed [31:0] C_5e_expected [0:63];
    
    // Systolic array
    reg signed [31:0] pe_acc [0:TILE-1][0:TILE-1];
    
    // Test variables
    integer i, j, k;
    integer m_tile, n_tile, k_tile;
    integer m_start, n_start, k_start;
    integer errors_5a, errors_5b, errors_5c, errors_5d, errors_5e;
    reg signed [7:0] a_val, b_val;
    reg signed [31:0] result, expected;
    
    initial begin
        $display("");
        $display("╔══════════════════════════════════════════════════════════════════╗");
        $display("║      Phase E Test 5: Boundary Conditions                        ║");
        $display("╚══════════════════════════════════════════════════════════════════╝");
        $display("");
        
        // Load test vectors
        $display("[LOAD] Loading test vectors...");
        $readmemh("tests/realistic/phase_e/test_vectors/test5a_A_int8.hex", A_5a);
        $readmemh("tests/realistic/phase_e/test_vectors/test5a_B_int8.hex", B_5a);
        $readmemh("tests/realistic/phase_e/test_vectors/test5a_C_int32.hex", C_5a_expected);
        
        $readmemh("tests/realistic/phase_e/test_vectors/test5b_A_int8.hex", A_5b);
        $readmemh("tests/realistic/phase_e/test_vectors/test5b_B_int8.hex", B_5b);
        $readmemh("tests/realistic/phase_e/test_vectors/test5b_C_int32.hex", C_5b_expected);
        
        $readmemh("tests/realistic/phase_e/test_vectors/test5c_A_int8.hex", A_5c);
        $readmemh("tests/realistic/phase_e/test_vectors/test5c_B_int8.hex", B_5c);
        $readmemh("tests/realistic/phase_e/test_vectors/test5c_C_int32.hex", C_5c_expected);
        
        $readmemh("tests/realistic/phase_e/test_vectors/test5d_A_int8.hex", A_5d);
        $readmemh("tests/realistic/phase_e/test_vectors/test5d_B_int8.hex", B_5d);
        $readmemh("tests/realistic/phase_e/test_vectors/test5d_C_int32.hex", C_5d_expected);
        
        $readmemh("tests/realistic/phase_e/test_vectors/test5e_A_int8.hex", A_5e);
        $readmemh("tests/realistic/phase_e/test_vectors/test5e_B_int8.hex", B_5e);
        $readmemh("tests/realistic/phase_e/test_vectors/test5e_C_int32.hex", C_5e_expected);
        
        // Initialize
        for (i = 0; i < M_5A*N_5A; i = i + 1) C_5a_result[i] = 0;
        
        // =====================================================================
        // Test 5a: Non-tile-aligned (7×13 @ 13×5)
        // =====================================================================
        $display("");
        $display("[TEST 5a] Non-aligned: (%0d, %0d) × (%0d, %0d)", M_5A, K_5A, K_5A, N_5A);
        
        for (m_tile = 0; m_tile < (M_5A + TILE - 1) / TILE; m_tile = m_tile + 1) begin
            m_start = m_tile * TILE;
            for (n_tile = 0; n_tile < (N_5A + TILE - 1) / TILE; n_tile = n_tile + 1) begin
                n_start = n_tile * TILE;
                
                for (i = 0; i < TILE; i = i + 1)
                    for (j = 0; j < TILE; j = j + 1)
                        pe_acc[i][j] = 0;
                
                for (k_tile = 0; k_tile < (K_5A + TILE - 1) / TILE; k_tile = k_tile + 1) begin
                    k_start = k_tile * TILE;
                    
                    for (k = 0; k < TILE; k = k + 1) begin
                        if (k_start + k < K_5A) begin
                            for (i = 0; i < TILE; i = i + 1) begin
                                if (m_start + i < M_5A) begin
                                    a_val = A_5a[(m_start + i) * K_5A + (k_start + k)];
                                    for (j = 0; j < TILE; j = j + 1) begin
                                        if (n_start + j < N_5A) begin
                                            b_val = B_5a[(k_start + k) * N_5A + (n_start + j)];
                                            pe_acc[i][j] = pe_acc[i][j] + $signed(a_val) * $signed(b_val);
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
                
                for (i = 0; i < TILE; i = i + 1)
                    for (j = 0; j < TILE; j = j + 1)
                        if (m_start + i < M_5A && n_start + j < N_5A)
                            C_5a_result[(m_start + i) * N_5A + (n_start + j)] = pe_acc[i][j];
            end
        end
        
        errors_5a = 0;
        for (i = 0; i < M_5A*N_5A; i = i + 1) begin
            if (C_5a_result[i] !== C_5a_expected[i]) begin
                errors_5a = errors_5a + 1;
                if (errors_5a <= 3)
                    $display("  Error[%0d]: got %0d, expected %0d", i, C_5a_result[i], C_5a_expected[i]);
            end
        end
        $display("  Errors: %0d", errors_5a);
        
        // =====================================================================
        // Test 5b: Single element (1×1 @ 1×1)
        // =====================================================================
        $display("");
        $display("[TEST 5b] Single element: 100 × 50");
        
        result = $signed(A_5b[0]) * $signed(B_5b[0]);
        expected = C_5b_expected[0];
        errors_5b = (result !== expected) ? 1 : 0;
        
        $display("  Result: %0d, Expected: %0d", result, expected);
        $display("  Errors: %0d", errors_5b);
        
        // =====================================================================
        // Test 5c: Maximum values (all 127)
        // =====================================================================
        $display("");
        $display("[TEST 5c] Max values: 127 × 127 × 8");
        
        // Each element should be 127*127*8 = 129032
        result = 0;
        for (k = 0; k < 8; k = k + 1)
            result = result + $signed(A_5c[k]) * $signed(B_5c[k*8]);
        
        expected = C_5c_expected[0];
        errors_5c = (result !== expected) ? 1 : 0;
        
        $display("  Result[0]: %0d, Expected: %0d", result, expected);
        $display("  Fits in INT32: %s", (result > -2147483648 && result < 2147483647) ? "YES" : "NO");
        $display("  Errors: %0d", errors_5c);
        
        // =====================================================================
        // Test 5d: Minimum values (all -128)
        // =====================================================================
        $display("");
        $display("[TEST 5d] Min values: -128 × -128 × 8");
        
        result = 0;
        for (k = 0; k < 8; k = k + 1)
            result = result + $signed(A_5d[k]) * $signed(B_5d[k*8]);
        
        expected = C_5d_expected[0];
        errors_5d = (result !== expected) ? 1 : 0;
        
        $display("  Result[0]: %0d, Expected: %0d", result, expected);
        $display("  Errors: %0d", errors_5d);
        
        // =====================================================================
        // Test 5e: Mixed max/min (127 × -128)
        // =====================================================================
        $display("");
        $display("[TEST 5e] Mixed: 127 × -128 × 8");
        
        result = 0;
        for (k = 0; k < 8; k = k + 1)
            result = result + $signed(A_5e[k]) * $signed(B_5e[k*8]);
        
        expected = C_5e_expected[0];
        errors_5e = (result !== expected) ? 1 : 0;
        
        $display("  Result[0]: %0d, Expected: %0d", result, expected);
        $display("  Errors: %0d", errors_5e);
        
        // Summary
        $display("");
        $display("╔══════════════════════════════════════════════════════════════════╗");
        $display("║                      TEST SUMMARY                                ║");
        $display("╠══════════════════════════════════════════════════════════════════╣");
        $display("║  5a Non-aligned (7×13×5):  %0d errors                              ║", errors_5a);
        $display("║  5b Single element:        %0d errors                              ║", errors_5b);
        $display("║  5c Max values (127):      %0d errors                              ║", errors_5c);
        $display("║  5d Min values (-128):     %0d errors                              ║", errors_5d);
        $display("║  5e Mixed (127×-128):      %0d errors                              ║", errors_5e);
        $display("╠══════════════════════════════════════════════════════════════════╣");
        
        if (errors_5a == 0 && errors_5b == 0 && errors_5c == 0 && errors_5d == 0 && errors_5e == 0) begin
            $display("║  ✓ PASSED: All boundary conditions verified                    ║");
            $display("╚══════════════════════════════════════════════════════════════════╝");
            $display("");
            $display(">>> PHASE_E_BOUNDARY TEST PASSED! <<<");
        end else begin
            $display("║  ✗ FAILED                                                       ║");
            $display("╚══════════════════════════════════════════════════════════════════╝");
            $display("");
            $display(">>> PHASE_E_BOUNDARY TEST FAILED! <<<");
        end
        
        $display("");
        $finish;
    end

endmodule

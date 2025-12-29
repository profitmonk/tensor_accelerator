//==============================================================================
// Systolic Array Unit Testbench - Fixed Version
//
// Tests:
// 1. Weight loading
// 2. Simple 2x2 GEMM (manually verified)
// 3. 4x4 Identity matrix test
//==============================================================================

`timescale 1ns / 1ps

module tb_systolic_array;

    //==========================================================================
    // Parameters
    //==========================================================================
    
    parameter CLK_PERIOD = 10;
    parameter ARRAY_SIZE = 16;
    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH  = 32;

    //==========================================================================
    // Signals
    //==========================================================================
    
    reg  clk;
    reg  rst_n;
    
    // Control
    reg  start;
    reg  clear_acc;
    wire busy;
    wire done;
    reg  [15:0] cfg_k_tiles;
    
    // Weight loading
    reg  weight_load_en;
    reg  [$clog2(ARRAY_SIZE)-1:0] weight_load_col;
    reg  [ARRAY_SIZE*DATA_WIDTH-1:0] weight_load_data;
    
    // Activation input
    reg  act_valid;
    reg  [ARRAY_SIZE*DATA_WIDTH-1:0] act_data;
    wire act_ready;
    
    // Result output
    wire result_valid;
    wire [ARRAY_SIZE*ACC_WIDTH-1:0] result_data;
    reg  result_ready;

    //==========================================================================
    // DUT
    //==========================================================================
    
    systolic_array #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk              (clk),
        .rst_n            (rst_n),
        .start            (start),
        .clear_acc        (clear_acc),
        .busy             (busy),
        .done             (done),
        .cfg_k_tiles      (cfg_k_tiles),
        .weight_load_en   (weight_load_en),
        .weight_load_col  (weight_load_col),
        .weight_load_data (weight_load_data),
        .act_valid        (act_valid),
        .act_data         (act_data),
        .act_ready        (act_ready),
        .result_valid     (result_valid),
        .result_data      (result_data),
        .result_ready     (result_ready)
    );

    //==========================================================================
    // Clock Generation
    //==========================================================================
    
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    //==========================================================================
    // Test Data Storage
    //==========================================================================
    
    reg signed [DATA_WIDTH-1:0] matrix_A [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    reg signed [DATA_WIDTH-1:0] matrix_B [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    reg signed [ACC_WIDTH-1:0]  matrix_C_expected [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    reg signed [ACC_WIDTH-1:0]  matrix_C_actual [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    
    // Test tracking
    integer total_tests;
    integer passed_tests;

    //==========================================================================
    // Helper Tasks
    //==========================================================================
    
    // Clear all matrices to zero
    task clear_matrices;
        integer i, j;
        begin
            for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
                for (j = 0; j < ARRAY_SIZE; j = j + 1) begin
                    matrix_A[i][j] = 0;
                    matrix_B[i][j] = 0;
                    matrix_C_expected[i][j] = 0;
                    matrix_C_actual[i][j] = 32'hDEADBEEF; // Mark as "not collected"
                end
            end
        end
    endtask
    
    // Setup simple 2x2 multiply test
    // A = [1 1]  B = [1 2]  C = [3  5]
    //     [2 2]      [2 3]      [6 10]
    task init_simple_2x2_test;
        begin
            clear_matrices();
            
            matrix_A[0][0] = 1; matrix_A[0][1] = 1;
            matrix_A[1][0] = 2; matrix_A[1][1] = 2;
            
            matrix_B[0][0] = 1; matrix_B[0][1] = 2;
            matrix_B[1][0] = 2; matrix_B[1][1] = 3;
            
            matrix_C_expected[0][0] = 3;  matrix_C_expected[0][1] = 5;
            matrix_C_expected[1][0] = 6;  matrix_C_expected[1][1] = 10;
            
            $display("  A = [1 1]  B = [1 2]");
            $display("      [2 2]      [2 3]");
            $display("  Expected C = [3  5]");
            $display("               [6 10]");
        end
    endtask
    
    // Setup 4x4 identity test: C = A × I = A
    task init_identity_4x4_test;
        integer i;
        begin
            clear_matrices();
            
            // A = arbitrary 4x4
            matrix_A[0][0] = 1; matrix_A[0][1] = 2; matrix_A[0][2] = 3; matrix_A[0][3] = 4;
            matrix_A[1][0] = 5; matrix_A[1][1] = 6; matrix_A[1][2] = 7; matrix_A[1][3] = 8;
            matrix_A[2][0] = 9; matrix_A[2][1] = 10; matrix_A[2][2] = 11; matrix_A[2][3] = 12;
            matrix_A[3][0] = 13; matrix_A[3][1] = 14; matrix_A[3][2] = 15; matrix_A[3][3] = 16;
            
            // B = Identity
            for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
                matrix_B[i][i] = 1;
            end
            
            // Expected: C = A
            matrix_C_expected[0][0] = 1; matrix_C_expected[0][1] = 2; matrix_C_expected[0][2] = 3; matrix_C_expected[0][3] = 4;
            matrix_C_expected[1][0] = 5; matrix_C_expected[1][1] = 6; matrix_C_expected[1][2] = 7; matrix_C_expected[1][3] = 8;
            matrix_C_expected[2][0] = 9; matrix_C_expected[2][1] = 10; matrix_C_expected[2][2] = 11; matrix_C_expected[2][3] = 12;
            matrix_C_expected[3][0] = 13; matrix_C_expected[3][1] = 14; matrix_C_expected[3][2] = 15; matrix_C_expected[3][3] = 16;
            
            $display("  A = [ 1  2  3  4]  B = Identity");
            $display("      [ 5  6  7  8]");
            $display("      [ 9 10 11 12]");
            $display("      [13 14 15 16]");
            $display("  Expected C = A");
        end
    endtask

    //==========================================================================
    // Systolic Array Operation Tasks
    //==========================================================================
    
    // Load weights (B matrix) column by column
    task load_weights;
        integer col, row;
        begin
            $display("[%0t] Loading weights...", $time);
            
            for (col = 0; col < ARRAY_SIZE; col = col + 1) begin
                weight_load_en = 1;
                weight_load_col = col;
                
                // Pack column of B: weight_load_data[row] = B[row][col]
                for (row = 0; row < ARRAY_SIZE; row = row + 1) begin
                    weight_load_data[row*DATA_WIDTH +: DATA_WIDTH] = matrix_B[row][col];
                end
                
                @(posedge clk);
            end
            
            weight_load_en = 0;
            weight_load_data = 0;
            @(posedge clk);
            
            $display("[%0t] Weights loaded", $time);
        end
    endtask
    
    // Run computation by streaming activations
    // Also collects results as they emerge from the array
    task run_computation;
        integer k, row, col;
        integer row_idx;
        integer wait_cycles;
        begin
            $display("[%0t] Running computation...", $time);
            
            // Initialize result collection
            row_idx = 0;
            for (row = 0; row < ARRAY_SIZE; row = row + 1) begin
                for (col = 0; col < ARRAY_SIZE; col = col + 1) begin
                    matrix_C_actual[row][col] = 32'hDEADBEEF;
                end
            end
            
            // Start the array
            start = 1;
            clear_acc = 1;
            cfg_k_tiles = ARRAY_SIZE;  // K = ARRAY_SIZE for square multiply
            result_ready = 1;
            @(posedge clk);
            start = 0;
            clear_acc = 0;
            
            // Stream A row by row (m = output row index)
            // At time m, send A[m][k] to row k for all k
            // This produces C[m][n] at output column n
            for (row = 0; row < ARRAY_SIZE; row = row + 1) begin
                act_valid = 1;
                for (k = 0; k < ARRAY_SIZE; k = k + 1) begin
                    // A[row][k] goes to systolic row k
                    act_data[k*DATA_WIDTH +: DATA_WIDTH] = matrix_A[row][k];
                end
                @(posedge clk);
            end
            
            act_valid = 0;
            act_data = 0;
            
            // Collect results - result_valid is now properly timed
            wait_cycles = 0;
            while (row_idx < ARRAY_SIZE && wait_cycles < ARRAY_SIZE * 6) begin
                @(posedge clk);
                wait_cycles = wait_cycles + 1;
                
                if (result_valid) begin
                    // Capture this row's results
                    for (col = 0; col < ARRAY_SIZE; col = col + 1) begin
                        matrix_C_actual[row_idx][col] = $signed(result_data[col*ACC_WIDTH +: ACC_WIDTH]);
                    end
                    if (row_idx < 4) begin
                        $display("[%0t]   Row %0d: C[%0d][0]=%0d C[%0d][1]=%0d", 
                                 $time, row_idx, 
                                 row_idx, matrix_C_actual[row_idx][0],
                                 row_idx, matrix_C_actual[row_idx][1]);
                    end
                    row_idx = row_idx + 1;
                end
            end
            
            result_ready = 0;
            $display("[%0t] Computation complete, collected %0d result rows", $time, row_idx);
        end
    endtask
    
    // Collect results (now mostly handled by run_computation, kept for compatibility)
    task collect_results;
        begin
            // Results are now collected inline during run_computation
        end
    endtask
    
    // Complete GEMM flow
    task run_gemm;
        begin
            load_weights();
            run_computation();
            collect_results();
        end
    endtask

    //==========================================================================
    // Verification
    //==========================================================================
    
    task verify_results;
        input integer size;  // Check size x size portion
        integer i, j;
        integer errors;
        begin
            errors = 0;
            
            $display("");
            $display("Verification (%0dx%0d):", size, size);
            $display("  Expected vs Actual:");
            
            for (i = 0; i < size; i = i + 1) begin
                for (j = 0; j < size; j = j + 1) begin
                    if (matrix_C_actual[i][j] === 32'hDEADBEEF) begin
                        $display("    C[%0d][%0d]: expected=%0d, actual=NOT_COLLECTED", 
                                 i, j, matrix_C_expected[i][j]);
                        errors = errors + 1;
                    end else if (matrix_C_expected[i][j] !== matrix_C_actual[i][j]) begin
                        $display("    C[%0d][%0d]: expected=%0d, actual=%0d  <-- MISMATCH", 
                                 i, j, matrix_C_expected[i][j], matrix_C_actual[i][j]);
                        errors = errors + 1;
                    end else begin
                        $display("    C[%0d][%0d]: expected=%0d, actual=%0d  OK", 
                                 i, j, matrix_C_expected[i][j], matrix_C_actual[i][j]);
                    end
                end
            end
            
            $display("");
            total_tests = total_tests + 1;
            if (errors == 0) begin
                $display("  >>> TEST PASSED <<<");
                passed_tests = passed_tests + 1;
            end else begin
                $display("  >>> TEST FAILED: %0d errors <<<", errors);
            end
            $display("");
        end
    endtask

    //==========================================================================
    // Main Test Sequence
    //==========================================================================
    
    initial begin
        // Initialize
        rst_n = 0;
        start = 0;
        clear_acc = 0;
        cfg_k_tiles = 0;
        weight_load_en = 0;
        weight_load_col = 0;
        weight_load_data = 0;
        act_valid = 0;
        act_data = 0;
        result_ready = 0;
        total_tests = 0;
        passed_tests = 0;
        
        // Reset
        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 5);
        
        $display("");
        $display("╔════════════════════════════════════════════════════════════╗");
        $display("║         Systolic Array Testbench                           ║");
        $display("║         Array: %0dx%0d, Data: %0d-bit, Acc: %0d-bit             ║", 
                 ARRAY_SIZE, ARRAY_SIZE, DATA_WIDTH, ACC_WIDTH);
        $display("╚════════════════════════════════════════════════════════════╝");
        
        //----------------------------------------------------------------------
        // Test 1: Simple 2x2 multiply
        //----------------------------------------------------------------------
        $display("");
        $display("────────────────────────────────────────────────────────────────");
        $display("[TEST 1] Simple 2x2 Matrix Multiply");
        $display("────────────────────────────────────────────────────────────────");
        
        init_simple_2x2_test();
        run_gemm();
        verify_results(2);
        
        // Reset between tests
        rst_n = 0;
        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 5);
        
        //----------------------------------------------------------------------
        // Test 2: 4x4 Identity
        //----------------------------------------------------------------------
        $display("");
        $display("────────────────────────────────────────────────────────────────");
        $display("[TEST 2] 4x4 Identity Matrix Test (C = A × I = A)");
        $display("────────────────────────────────────────────────────────────────");
        
        init_identity_4x4_test();
        run_gemm();
        verify_results(4);
        
        //----------------------------------------------------------------------
        // Summary
        //----------------------------------------------------------------------
        #(CLK_PERIOD * 10);
        $display("");
        $display("╔════════════════════════════════════════════════════════════╗");
        $display("║                    TEST SUMMARY                            ║");
        $display("╠════════════════════════════════════════════════════════════╣");
        $display("║   Passed: %0d / %0d                                          ║", passed_tests, total_tests);
        $display("╚════════════════════════════════════════════════════════════╝");
        
        if (passed_tests == total_tests) begin
            $display("   >>> ALL TESTS PASSED! <<<");
        end else begin
            $display("   >>> SOME TESTS FAILED <<<");
        end
        $display("");
        
        $finish;
    end

    //==========================================================================
    // Waveform Dump
    //==========================================================================
    
    initial begin
        $dumpfile("systolic_array.vcd");
        $dumpvars(0, tb_systolic_array);
    end

    //==========================================================================
    // Timeout Watchdog
    //==========================================================================
    
    initial begin
        #(CLK_PERIOD * 20000);
        $display("ERROR: Simulation timeout!");
        $finish;
    end

    //==========================================================================
    // State Monitor
    //==========================================================================
    
    always @(posedge clk) begin
        if (result_valid) begin
            $display("[%0t] result_valid asserted", $time);
        end
        if (done && !dut.done) begin
            $display("[%0t] done deasserted", $time);
        end
    end

endmodule

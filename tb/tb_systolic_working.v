//==============================================================================
// Working Systolic Array Testbench
// Uses 4x4 array with correct dataflow verification
//==============================================================================
`timescale 1ns / 1ps

module tb_systolic_working;
    parameter CLK_PERIOD = 10;
    parameter ARRAY_SIZE = 4;
    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH  = 32;

    reg  clk = 0;
    reg  rst_n = 0;
    reg  start = 0;
    reg  clear_acc = 0;
    wire busy, done;
    reg  [15:0] cfg_k_tiles;
    reg  weight_load_en = 0;
    reg  [$clog2(ARRAY_SIZE)-1:0] weight_load_col = 0;
    reg  [ARRAY_SIZE*DATA_WIDTH-1:0] weight_load_data = 0;
    reg  act_valid = 0;
    reg  [ARRAY_SIZE*DATA_WIDTH-1:0] act_data = 0;
    wire act_ready;
    wire result_valid;
    wire [ARRAY_SIZE*ACC_WIDTH-1:0] result_data;
    reg  result_ready = 1;

    systolic_array #(.ARRAY_SIZE(ARRAY_SIZE), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH))
    dut (.*);

    always #(CLK_PERIOD/2) clk = ~clk;

    // Test matrices
    reg signed [7:0] A [0:3][0:3];
    reg signed [7:0] B [0:3][0:3];
    reg signed [31:0] C_expected [0:3][0:3];
    reg signed [31:0] C_actual [0:3][0:3];
    
    integer i, j, k, m, n, col, row_count;
    integer errors, total_tests, passed_tests;

    // Capture results
    always @(posedge clk) begin
        if (result_valid && row_count < ARRAY_SIZE) begin
            for (col = 0; col < ARRAY_SIZE; col = col + 1) begin
                C_actual[row_count][col] = $signed(result_data[col*ACC_WIDTH +: ACC_WIDTH]);
            end
            row_count = row_count + 1;
        end
    end

    // Compute expected result
    task compute_expected;
        integer ii, jj, kk;
        begin
            for (ii = 0; ii < ARRAY_SIZE; ii = ii + 1) begin
                for (jj = 0; jj < ARRAY_SIZE; jj = jj + 1) begin
                    C_expected[ii][jj] = 0;
                    for (kk = 0; kk < ARRAY_SIZE; kk = kk + 1) begin
                        C_expected[ii][jj] = C_expected[ii][jj] + A[ii][kk] * B[kk][jj];
                    end
                end
            end
        end
    endtask

    // Initialize matrices for test
    task init_matrices;
        input integer test_num;
        begin
            case (test_num)
                1: begin  // 2x2 multiply (rest zeros)
                    for (i = 0; i < 4; i = i + 1) begin
                        for (j = 0; j < 4; j = j + 1) begin
                            A[i][j] = 0; B[i][j] = 0;
                        end
                    end
                    A[0][0] = 1; A[0][1] = 1;
                    A[1][0] = 2; A[1][1] = 2;
                    B[0][0] = 1; B[0][1] = 2;
                    B[1][0] = 2; B[1][1] = 3;
                end
                2: begin  // Identity test
                    for (i = 0; i < 4; i = i + 1) begin
                        for (j = 0; j < 4; j = j + 1) begin
                            A[i][j] = i * 4 + j + 1;
                            B[i][j] = (i == j) ? 1 : 0;
                        end
                    end
                end
                3: begin  // Full 4x4 multiply
                    for (i = 0; i < 4; i = i + 1) begin
                        for (j = 0; j < 4; j = j + 1) begin
                            A[i][j] = i + j + 1;
                            B[i][j] = i * 2 + j;
                        end
                    end
                end
            endcase
            compute_expected();
        end
    endtask

    // Load weights (B matrix)
    task load_weights;
        begin
            for (col = 0; col < ARRAY_SIZE; col = col + 1) begin
                weight_load_en = 1;
                weight_load_col = col;
                weight_load_data = 0;
                for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
                    weight_load_data[i*DATA_WIDTH +: DATA_WIDTH] = B[i][col];
                end
                @(posedge clk);
            end
            weight_load_en = 0;
            @(posedge clk);
        end
    endtask

    // Run computation
    task run_computation;
        begin
            start = 1; clear_acc = 1;
            @(posedge clk);
            start = 0; clear_acc = 0;
            
            // Stream activations: A[m][k] to row k at each cycle
            for (m = 0; m < ARRAY_SIZE; m = m + 1) begin
                act_valid = 1;
                act_data = 0;
                for (k = 0; k < ARRAY_SIZE; k = k + 1) begin
                    act_data[k*DATA_WIDTH +: DATA_WIDTH] = A[m][k];
                end
                @(posedge clk);
            end
            act_valid = 0;
            act_data = 0;
            
            // Wait for completion
            while (!done) @(posedge clk);
        end
    endtask

    // Verify results
    task verify;
        input integer test_num;
        begin
            errors = 0;
            $display("  Expected vs Actual:");
            for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
                for (j = 0; j < ARRAY_SIZE; j = j + 1) begin
                    if (C_expected[i][j] != C_actual[i][j]) begin
                        $display("    C[%0d][%0d]: expected=%0d, actual=%0d  <-- MISMATCH",
                                 i, j, C_expected[i][j], C_actual[i][j]);
                        errors = errors + 1;
                    end
                end
            end
            
            total_tests = total_tests + 1;
            if (errors == 0) begin
                $display("  >>> TEST %0d PASSED <<<", test_num);
                passed_tests = passed_tests + 1;
            end else begin
                $display("  >>> TEST %0d FAILED: %0d errors <<<", test_num, errors);
            end
        end
    endtask

    initial begin
        $dumpfile("systolic_working.vcd");
        $dumpvars(0, tb_systolic_working);
        
        $display("");
        $display("╔═══════════════════════════════════════════════════════════════╗");
        $display("║  SYSTOLIC ARRAY WORKING TEST (4x4 Array)                      ║");
        $display("╚═══════════════════════════════════════════════════════════════╝");
        
        total_tests = 0;
        passed_tests = 0;
        cfg_k_tiles = ARRAY_SIZE;
        
        #100 rst_n = 1;
        @(posedge clk);

        // Test 1: 2x2 multiply
        $display("\n[TEST 1] 2x2 Matrix Multiply");
        $display("  A = [1 1 0 0]  B = [1 2 0 0]");
        $display("      [2 2 0 0]      [2 3 0 0]");
        $display("      [0 0 0 0]      [0 0 0 0]");
        $display("      [0 0 0 0]      [0 0 0 0]");
        init_matrices(1);
        row_count = 0;
        for (i = 0; i < 4; i = i + 1) for (j = 0; j < 4; j = j + 1) C_actual[i][j] = 32'hDEAD;
        load_weights();
        run_computation();
        verify(1);

        // Test 2: Identity multiply
        $display("\n[TEST 2] Identity Matrix Test (C = A × I = A)");
        init_matrices(2);
        row_count = 0;
        for (i = 0; i < 4; i = i + 1) for (j = 0; j < 4; j = j + 1) C_actual[i][j] = 32'hDEAD;
        load_weights();
        run_computation();
        verify(2);

        // Test 3: Full 4x4 multiply
        $display("\n[TEST 3] Full 4x4 Matrix Multiply");
        init_matrices(3);
        row_count = 0;
        for (i = 0; i < 4; i = i + 1) for (j = 0; j < 4; j = j + 1) C_actual[i][j] = 32'hDEAD;
        load_weights();
        run_computation();
        verify(3);

        // Summary
        $display("");
        $display("╔═══════════════════════════════════════════════════════════════╗");
        $display("║  TEST SUMMARY: %0d / %0d Passed                                  ║", passed_tests, total_tests);
        if (passed_tests == total_tests)
            $display("║  >>> ALL TESTS PASSED <<<                                     ║");
        else
            $display("║  >>> SOME TESTS FAILED <<<                                    ║");
        $display("╚═══════════════════════════════════════════════════════════════╝");
        $display("");
        
        $finish;
    end
endmodule

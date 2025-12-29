`timescale 1ns / 1ps

module tb_systolic_4x4;
    parameter CLK_PERIOD = 10;
    parameter ARRAY_SIZE = 4;  // 4x4 array for 4x4 data
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

    integer i, j, k, col, row_idx, wait_cycles, errors;
    integer first_nonzero;
    reg signed [31:0] C_actual [0:3][0:3];
    reg signed [31:0] C_expected [0:3][0:3];
    reg [31:0] row_sum;

    initial begin
        $display("\n╔════════════════════════════════════════════════════════════╗");
        $display("║         4x4 Systolic Array Test                            ║");
        $display("╚════════════════════════════════════════════════════════════╝\n");

        // Test 1: 2x2 multiply
        $display("TEST 1: 2x2 Matrix Multiply");
        $display("  A = [1 1], B = [1 2], Expected C = [3  5]");
        $display("      [2 2]      [2 3]              [6 10]");
        
        C_expected[0][0] = 3;  C_expected[0][1] = 5;
        C_expected[1][0] = 6;  C_expected[1][1] = 10;
        for (i = 2; i < 4; i = i + 1) for (j = 0; j < 4; j = j + 1) C_expected[i][j] = 0;
        for (i = 0; i < 2; i = i + 1) for (j = 2; j < 4; j = j + 1) C_expected[i][j] = 0;

        cfg_k_tiles = 20;
        #100 rst_n = 1;
        @(posedge clk);

        // Load weights
        for (col = 0; col < ARRAY_SIZE; col = col + 1) begin
            weight_load_en = 1;
            weight_load_col = col;
            weight_load_data = 0;
            if (col == 0) begin
                weight_load_data[0*8 +: 8] = 1;  // B[0][0]
                weight_load_data[1*8 +: 8] = 2;  // B[1][0]
            end else if (col == 1) begin
                weight_load_data[0*8 +: 8] = 2;  // B[0][1]
                weight_load_data[1*8 +: 8] = 3;  // B[1][1]
            end
            @(posedge clk);
        end
        weight_load_en = 0;
        @(posedge clk);

        // Start
        start = 1; clear_acc = 1;
        result_ready = 1;
        @(posedge clk);
        start = 0; clear_acc = 0;

        // Stream activations
        act_valid = 1;
        act_data = 0;
        act_data[0*8 +: 8] = 1; act_data[1*8 +: 8] = 1;  // A[0][*]
        @(posedge clk);
        act_data = 0;
        act_data[0*8 +: 8] = 2; act_data[1*8 +: 8] = 2;  // A[1][*]
        @(posedge clk);
        act_valid = 0;
        act_data = 0;

        // Collect results
        row_idx = 0;
        first_nonzero = 0;
        for (i = 0; i < 4; i = i + 1) for (j = 0; j < 4; j = j + 1) C_actual[i][j] = 32'hDEADBEEF;
        
        wait_cycles = 0;
        while (row_idx < 4 && wait_cycles < 50) begin
            @(posedge clk);
            wait_cycles = wait_cycles + 1;
            if (result_valid) begin
                row_sum = 0;
                for (col = 0; col < ARRAY_SIZE; col = col + 1)
                    row_sum = row_sum | result_data[col*ACC_WIDTH +: ACC_WIDTH];
                if (row_sum != 0 || first_nonzero) begin
                    first_nonzero = 1;
                    for (col = 0; col < ARRAY_SIZE; col = col + 1)
                        C_actual[row_idx][col] = $signed(result_data[col*ACC_WIDTH +: ACC_WIDTH]);
                    $display("  Captured Row %0d: [%0d, %0d, %0d, %0d]",
                             row_idx, C_actual[row_idx][0], C_actual[row_idx][1],
                             C_actual[row_idx][2], C_actual[row_idx][3]);
                    row_idx = row_idx + 1;
                end
            end
        end

        // Verify
        errors = 0;
        $display("\nVerification (2x2):");
        for (i = 0; i < 2; i = i + 1) begin
            for (j = 0; j < 2; j = j + 1) begin
                if (C_actual[i][j] == C_expected[i][j])
                    $display("  C[%0d][%0d]: expected=%0d, actual=%0d OK", i, j, C_expected[i][j], C_actual[i][j]);
                else begin
                    $display("  C[%0d][%0d]: expected=%0d, actual=%0d MISMATCH", i, j, C_expected[i][j], C_actual[i][j]);
                    errors = errors + 1;
                end
            end
        end
        
        if (errors == 0) $display("\n>>> TEST 1 PASSED <<<\n");
        else $display("\n>>> TEST 1 FAILED: %0d errors <<<\n", errors);

        // Test 2: 4x4 Identity
        $display("\nTEST 2: 4x4 Identity (C = A × I = A)");
        
        // Reset
        rst_n = 0; #20; rst_n = 1;
        @(posedge clk);

        // Expected
        for (i = 0; i < 4; i = i + 1) for (j = 0; j < 4; j = j + 1) 
            C_expected[i][j] = i*4 + j + 1;

        // Load identity weights
        for (col = 0; col < ARRAY_SIZE; col = col + 1) begin
            weight_load_en = 1;
            weight_load_col = col;
            weight_load_data = 0;
            weight_load_data[col*8 +: 8] = 1;  // Diagonal = 1
            @(posedge clk);
        end
        weight_load_en = 0;
        @(posedge clk);

        // Start
        start = 1; clear_acc = 1;
        @(posedge clk);
        start = 0; clear_acc = 0;

        // Stream A = [1,2,3,4; 5,6,7,8; 9,10,11,12; 13,14,15,16]
        act_valid = 1;
        for (k = 0; k < 4; k = k + 1) begin
            act_data = 0;
            for (i = 0; i < 4; i = i + 1)
                act_data[i*8 +: 8] = k*4 + i + 1;  // A[k][i]
            @(posedge clk);
        end
        act_valid = 0;
        act_data = 0;

        // Collect
        row_idx = 0;
        first_nonzero = 0;
        for (i = 0; i < 4; i = i + 1) for (j = 0; j < 4; j = j + 1) C_actual[i][j] = 32'hDEADBEEF;
        
        wait_cycles = 0;
        while (row_idx < 4 && wait_cycles < 50) begin
            @(posedge clk);
            wait_cycles = wait_cycles + 1;
            if (result_valid) begin
                row_sum = 0;
                for (col = 0; col < ARRAY_SIZE; col = col + 1)
                    row_sum = row_sum | result_data[col*ACC_WIDTH +: ACC_WIDTH];
                if (row_sum != 0 || first_nonzero) begin
                    first_nonzero = 1;
                    for (col = 0; col < ARRAY_SIZE; col = col + 1)
                        C_actual[row_idx][col] = $signed(result_data[col*ACC_WIDTH +: ACC_WIDTH]);
                    $display("  Captured Row %0d: [%0d, %0d, %0d, %0d]",
                             row_idx, C_actual[row_idx][0], C_actual[row_idx][1],
                             C_actual[row_idx][2], C_actual[row_idx][3]);
                    row_idx = row_idx + 1;
                end
            end
        end

        // Verify
        errors = 0;
        $display("\nVerification (4x4):");
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                if (C_actual[i][j] == C_expected[i][j])
                    $display("  C[%0d][%0d]: expected=%0d, actual=%0d OK", i, j, C_expected[i][j], C_actual[i][j]);
                else begin
                    $display("  C[%0d][%0d]: expected=%0d, actual=%0d MISMATCH", i, j, C_expected[i][j], C_actual[i][j]);
                    errors = errors + 1;
                end
            end
        end
        
        if (errors == 0) $display("\n>>> TEST 2 PASSED <<<\n");
        else $display("\n>>> TEST 2 FAILED: %0d errors <<<\n", errors);

        $display("\n╔════════════════════════════════════════════════════════════╗");
        $display("║                    TEST COMPLETE                           ║");
        $display("╚════════════════════════════════════════════════════════════╝\n");
        $finish;
    end
endmodule

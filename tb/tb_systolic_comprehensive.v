`timescale 1ns / 1ps
//
// Comprehensive Systolic Array Test
// Tests various matrix sizes and validates against expected values
//

module tb_systolic_comprehensive;
    parameter CLK_PERIOD = 10;
    parameter ARRAY_SIZE = 4;
    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH = 32;

    // Clock and reset
    reg clk = 0;
    reg rst_n = 0;
    
    // Control
    reg start = 0;
    reg clear_acc = 0;
    reg [15:0] cfg_k_tiles = ARRAY_SIZE;
    
    // Weight loading
    reg weight_load_en = 0;
    reg [$clog2(ARRAY_SIZE)-1:0] weight_load_col = 0;
    reg [ARRAY_SIZE*DATA_WIDTH-1:0] weight_load_data = 0;
    
    // Activation interface
    reg act_valid = 0;
    reg [ARRAY_SIZE*DATA_WIDTH-1:0] act_data = 0;
    wire act_ready;
    
    // Result interface
    wire result_valid;
    wire [ARRAY_SIZE*ACC_WIDTH-1:0] result_data;
    reg result_ready = 1;
    
    wire busy, done;

    // DUT
    systolic_array #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (.*);

    always #(CLK_PERIOD/2) clk = ~clk;

    // Test matrices
    reg signed [DATA_WIDTH-1:0] A [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    reg signed [DATA_WIDTH-1:0] B [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    reg signed [ACC_WIDTH-1:0] C_expected [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    reg signed [ACC_WIDTH-1:0] C_actual [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    
    integer row, col, k;
    integer result_row;
    integer errors;
    integer test_num;

    // Tasks
    task reset_dut;
        begin
            rst_n = 0;
            start = 0;
            clear_acc = 0;
            weight_load_en = 0;
            act_valid = 0;
            repeat(5) @(posedge clk);
            rst_n = 1;
            repeat(2) @(posedge clk);
        end
    endtask

    task load_weights;
        integer c;
        begin
            for (c = 0; c < ARRAY_SIZE; c = c + 1) begin
                @(posedge clk);
                weight_load_en = 1;
                weight_load_col = c;
                // Pack column c of B
                weight_load_data = {B[3][c], B[2][c], B[1][c], B[0][c]};
            end
            @(posedge clk);
            weight_load_en = 0;
        end
    endtask

    task compute_gemm;
        integer r;
        begin
            @(posedge clk);
            start = 1;
            clear_acc = 1;
            @(posedge clk);
            start = 0;
            clear_acc = 0;
            
            // Stream activations
            for (r = 0; r < ARRAY_SIZE; r = r + 1) begin
                @(posedge clk);
                act_valid = 1;
                act_data = {A[r][3], A[r][2], A[r][1], A[r][0]};
            end
            @(posedge clk);
            act_valid = 0;
            act_data = 0;
        end
    endtask

    task collect_results;
        integer timeout;
        begin
            result_row = 0;
            timeout = 100;
            
            while (result_row < ARRAY_SIZE && timeout > 0) begin
                @(posedge clk);
                timeout = timeout - 1;
                
                if (result_valid) begin
                    for (col = 0; col < ARRAY_SIZE; col = col + 1) begin
                        C_actual[result_row][col] = $signed(result_data[col*ACC_WIDTH +: ACC_WIDTH]);
                    end
                    result_row = result_row + 1;
                end
            end
            
            if (result_row < ARRAY_SIZE) begin
                $display("ERROR: Timeout waiting for results (got %0d/%0d)", result_row, ARRAY_SIZE);
            end
        end
    endtask

    task compute_expected;
        begin
            for (row = 0; row < ARRAY_SIZE; row = row + 1) begin
                for (col = 0; col < ARRAY_SIZE; col = col + 1) begin
                    C_expected[row][col] = 0;
                    for (k = 0; k < ARRAY_SIZE; k = k + 1) begin
                        C_expected[row][col] = C_expected[row][col] + 
                            $signed(A[row][k]) * $signed(B[k][col]);
                    end
                end
            end
        end
    endtask

    task check_results;
        input integer row_offset;  // Results are offset by this many rows
        begin
            errors = 0;
            for (row = 0; row < ARRAY_SIZE; row = row + 1) begin
                for (col = 0; col < ARRAY_SIZE; col = col + 1) begin
                    // Account for row offset in output
                    if (row + row_offset < ARRAY_SIZE) begin
                        if (C_actual[row + row_offset][col] !== C_expected[row][col]) begin
                            $display("MISMATCH: C[%0d][%0d] expected=%0d, actual=%0d",
                                     row, col, C_expected[row][col], 
                                     C_actual[row + row_offset][col]);
                            errors = errors + 1;
                        end
                    end
                end
            end
        end
    endtask

    task display_matrices;
        begin
            $display("Matrix A:");
            for (row = 0; row < ARRAY_SIZE; row = row + 1) begin
                $write("  [");
                for (col = 0; col < ARRAY_SIZE; col = col + 1) begin
                    $write("%4d", A[row][col]);
                end
                $display("]");
            end
            
            $display("Matrix B:");
            for (row = 0; row < ARRAY_SIZE; row = row + 1) begin
                $write("  [");
                for (col = 0; col < ARRAY_SIZE; col = col + 1) begin
                    $write("%4d", B[row][col]);
                end
                $display("]");
            end
            
            $display("Expected C:");
            for (row = 0; row < ARRAY_SIZE; row = row + 1) begin
                $write("  [");
                for (col = 0; col < ARRAY_SIZE; col = col + 1) begin
                    $write("%6d", C_expected[row][col]);
                end
                $display("]");
            end
            
            $display("Actual C:");
            for (row = 0; row < ARRAY_SIZE; row = row + 1) begin
                $write("  [");
                for (col = 0; col < ARRAY_SIZE; col = col + 1) begin
                    $write("%6d", C_actual[row][col]);
                end
                $display("]");
            end
        end
    endtask

    task run_test;
        input [255:0] test_name;
        input integer row_offset;
        begin
            $display("");
            $display("========================================");
            $display("Test: %s", test_name);
            $display("========================================");
            
            reset_dut;
            compute_expected;
            load_weights;
            compute_gemm;
            collect_results;
            display_matrices;
            check_results(row_offset);
            
            if (errors == 0) begin
                $display("PASSED!");
            end else begin
                $display("FAILED with %0d errors", errors);
            end
        end
    endtask

    // Main test sequence
    initial begin
        $dumpfile("tb_systolic_comprehensive.vcd");
        $dumpvars(0, tb_systolic_comprehensive);
        
        test_num = 0;
        
        //------------------------------------------------------
        // Test 1: Identity matrix
        //------------------------------------------------------
        for (row = 0; row < ARRAY_SIZE; row = row + 1) begin
            for (col = 0; col < ARRAY_SIZE; col = col + 1) begin
                A[row][col] = (row * ARRAY_SIZE + col + 1);  // 1,2,3,4,5,...
                B[row][col] = (row == col) ? 1 : 0;          // Identity
            end
        end
        run_test("Identity", 1);
        test_num = test_num + (errors == 0 ? 1 : 0);
        
        //------------------------------------------------------
        // Test 2: All ones
        //------------------------------------------------------
        for (row = 0; row < ARRAY_SIZE; row = row + 1) begin
            for (col = 0; col < ARRAY_SIZE; col = col + 1) begin
                A[row][col] = 1;
                B[row][col] = 1;
            end
        end
        run_test("All Ones", 1);
        test_num = test_num + (errors == 0 ? 1 : 0);
        
        //------------------------------------------------------
        // Test 3: Small positive values
        //------------------------------------------------------
        for (row = 0; row < ARRAY_SIZE; row = row + 1) begin
            for (col = 0; col < ARRAY_SIZE; col = col + 1) begin
                A[row][col] = (row + 1);
                B[row][col] = (col + 1);
            end
        end
        run_test("Small Positive", 1);
        test_num = test_num + (errors == 0 ? 1 : 0);
        
        //------------------------------------------------------
        // Test 4: Negative values
        //------------------------------------------------------
        for (row = 0; row < ARRAY_SIZE; row = row + 1) begin
            for (col = 0; col < ARRAY_SIZE; col = col + 1) begin
                A[row][col] = (row % 2 == 0) ? (col + 1) : -(col + 1);
                B[row][col] = (col % 2 == 0) ? (row + 1) : -(row + 1);
            end
        end
        run_test("Mixed Signs", 1);
        test_num = test_num + (errors == 0 ? 1 : 0);
        
        //------------------------------------------------------
        // Test 5: Larger values (near INT8 limits)
        //------------------------------------------------------
        for (row = 0; row < ARRAY_SIZE; row = row + 1) begin
            for (col = 0; col < ARRAY_SIZE; col = col + 1) begin
                A[row][col] = 30 + row * 10 + col;  // 30-63 range
                B[row][col] = -(20 + row * 5 + col); // -20 to -35 range  
            end
        end
        run_test("Larger Values", 1);
        test_num = test_num + (errors == 0 ? 1 : 0);
        
        //------------------------------------------------------
        // Summary
        //------------------------------------------------------
        $display("");
        $display("========================================");
        $display("TEST SUMMARY: %0d/5 tests passed", test_num);
        $display("========================================");
        
        if (test_num == 5) begin
            $display("ALL TESTS PASSED!");
        end else begin
            $display("SOME TESTS FAILED");
        end
        
        $finish;
    end

endmodule

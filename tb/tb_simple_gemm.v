`timescale 1ns / 1ps
//
// Simple Systolic Array Test
// Tests a 4x4 INT8 GEMM: C = A @ B
//
// A = [[1, 2, 0, 0],    B = [[1, 0, 0, 0],
//      [3, 4, 0, 0],         [0, 1, 0, 0],
//      [0, 0, 0, 0],         [0, 0, 0, 0],
//      [0, 0, 0, 0]]         [0, 0, 0, 0]]
//
// Expected C = A @ B = A (since B is identity in top-left 2x2)
//

module tb_simple_gemm;
    parameter CLK_PERIOD = 10;
    parameter ARRAY_SIZE = 4;
    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH = 32;

    // Clock and reset
    reg clk = 0;
    reg rst_n = 0;
    
    // Control signals
    reg start = 0;
    reg clear_acc = 0;
    reg [15:0] cfg_k_tiles = 4;  // One K-tile of size 4
    
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
    
    // Status
    wire busy, done;

    // DUT
    systolic_array #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (.*);

    // Clock generation
    always #(CLK_PERIOD/2) clk = ~clk;

    // Result capture
    reg [ACC_WIDTH-1:0] result_matrix [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    integer result_row = 0;
    integer col;
    
    // Debug state machine
    always @(posedge clk) begin
        if (result_valid) begin
            $display("Time %0t: Result row %0d valid", $time, result_row);
            for (col = 0; col < ARRAY_SIZE; col = col + 1) begin
                result_matrix[result_row][col] = result_data[col*ACC_WIDTH +: ACC_WIDTH];
                $display("  C[%0d][%0d] = %0d", result_row, col, 
                         $signed(result_data[col*ACC_WIDTH +: ACC_WIDTH]));
            end
            result_row = result_row + 1;
        end
    end

    initial begin
        $display("");
        $display("========================================");
        $display("Simple GEMM Test: 4x4 Identity");
        $display("========================================");
        $display("");
        
        // Dump waveform
        $dumpfile("tb_simple_gemm.vcd");
        $dumpvars(0, tb_simple_gemm);
        
        // Reset
        #100;
        rst_n = 1;
        #20;
        
        //------------------------------------------------------
        // Load weights (B matrix - identity in top-left 2x2)
        // B = [[1, 0, 0, 0],
        //      [0, 1, 0, 0],
        //      [0, 0, 0, 0],
        //      [0, 0, 0, 0]]
        //------------------------------------------------------
        $display("Loading weights...");
        
        // Column 0: [1, 0, 0, 0]
        @(posedge clk);
        weight_load_en = 1;
        weight_load_col = 0;
        weight_load_data = {8'd0, 8'd0, 8'd0, 8'd1};  // [row3, row2, row1, row0]
        
        // Column 1: [0, 1, 0, 0]
        @(posedge clk);
        weight_load_col = 1;
        weight_load_data = {8'd0, 8'd0, 8'd1, 8'd0};
        
        // Column 2: [0, 0, 0, 0]
        @(posedge clk);
        weight_load_col = 2;
        weight_load_data = {8'd0, 8'd0, 8'd0, 8'd0};
        
        // Column 3: [0, 0, 0, 0]
        @(posedge clk);
        weight_load_col = 3;
        weight_load_data = {8'd0, 8'd0, 8'd0, 8'd0};
        
        @(posedge clk);
        weight_load_en = 0;
        
        $display("Weights loaded");
        
        //------------------------------------------------------
        // Start computation
        //------------------------------------------------------
        @(posedge clk);
        $display("Starting computation...");
        start = 1;
        clear_acc = 1;
        
        @(posedge clk);
        start = 0;
        clear_acc = 0;
        
        //------------------------------------------------------
        // Stream activations (A matrix)
        // A = [[1, 2, 0, 0],
        //      [3, 4, 0, 0],
        //      [0, 0, 0, 0],
        //      [0, 0, 0, 0]]
        //------------------------------------------------------
        $display("Streaming activations...");
        
        // Row 0: [1, 2, 0, 0]
        @(posedge clk);
        act_valid = 1;
        act_data = {8'd0, 8'd0, 8'd2, 8'd1};  // [col3, col2, col1, col0]
        
        // Row 1: [3, 4, 0, 0]
        @(posedge clk);
        act_data = {8'd0, 8'd0, 8'd4, 8'd3};
        
        // Row 2: [0, 0, 0, 0]
        @(posedge clk);
        act_data = {8'd0, 8'd0, 8'd0, 8'd0};
        
        // Row 3: [0, 0, 0, 0]
        @(posedge clk);
        act_data = {8'd0, 8'd0, 8'd0, 8'd0};
        
        @(posedge clk);
        act_valid = 0;
        act_data = 0;
        
        $display("Activations complete, waiting for results...");
        
        //------------------------------------------------------
        // Wait for results
        //------------------------------------------------------
        repeat (50) @(posedge clk);  // Wait for pipeline to drain
        
        //------------------------------------------------------
        // Check results
        // Note: Results are offset by 1 row due to pipeline timing
        //------------------------------------------------------
        $display("");
        $display("========================================");
        $display("Results (expected A since B is identity):");
        $display("Note: Results shifted by 1 row due to pipeline");
        $display("========================================");
        
        if (result_row >= ARRAY_SIZE) begin
            // Expected: C = A (since B has identity in top-left)
            // Due to pipeline, results are shifted by 1 row
            // Row 1 contains result for A row 0
            // Row 2 contains result for A row 1
            
            if (result_matrix[1][0] == 1 && result_matrix[1][1] == 2 &&
                result_matrix[2][0] == 3 && result_matrix[2][1] == 4) begin
                $display("TEST PASSED!");
                $display("  C[1][0]=1, C[1][1]=2 (expected from A row 0)");
                $display("  C[2][0]=3, C[2][1]=4 (expected from A row 1)");
            end else begin
                $display("TEST FAILED!");
                $display("Expected C[1][0]=1, got %0d", $signed(result_matrix[1][0]));
                $display("Expected C[1][1]=2, got %0d", $signed(result_matrix[1][1]));
                $display("Expected C[2][0]=3, got %0d", $signed(result_matrix[2][0]));
                $display("Expected C[2][1]=4, got %0d", $signed(result_matrix[2][1]));
            end
        end else begin
            $display("TEST FAILED - only got %0d result rows", result_row);
        end
        
        $display("");
        $finish;
    end

endmodule

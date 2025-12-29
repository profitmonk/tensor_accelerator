`timescale 1ns / 1ps
//==============================================================================
// Simple Self-Checking Testbench for Systolic Array
// Matches Python model cycle-by-cycle
//==============================================================================

module tb_systolic_simple;
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
    reg [15:0] cfg_k_tiles;
    wire busy, done;
    
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

    // DUT - using v2 file
    systolic_array #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (.*);

    // Clock
    always #(CLK_PERIOD/2) clk = ~clk;

    // Test variables
    integer cycle;
    integer errors;
    integer col;
    reg signed [31:0] r0, r1, r2, r3;
    
    // Helper task to display results
    task show_cycle;
        begin
            #1;
            r0 = $signed(result_data[0*32 +: 32]);
            r1 = $signed(result_data[1*32 +: 32]);
            r2 = $signed(result_data[2*32 +: 32]);
            r3 = $signed(result_data[3*32 +: 32]);
            $display("Cycle %3d: state=%d act_valid=%b result_valid=%b result=[%4d,%4d,%4d,%4d]",
                     cycle, dut.state, act_valid, result_valid, r0, r1, r2, r3);
        end
    endtask

    initial begin
        $display("");
        $display("==============================================");
        $display("Simple Systolic Array Test");
        $display("==============================================");
        
        errors = 0;
        
        // Reset
        #100;
        rst_n = 1;
        @(posedge clk);
        
        //======================================================================
        // TEST 1: 2x2 Matrix Multiply
        // A = [1 1; 2 2], B = [1 2; 2 3]
        // Expected C = [3 5; 6 10]
        //======================================================================
        $display("");
        $display("=== TEST 1: 2x2 Multiply ===");
        $display("A = [1 1; 2 2], B = [1 2; 2 3]");
        $display("Expected C = [3 5; 6 10]");
        $display("");
        
        // Load weights: B[k][n] -> PE[k][n]
        // Column 0: B[0][0]=1, B[1][0]=2
        weight_load_en = 1;
        weight_load_col = 0;
        weight_load_data = 0;
        weight_load_data[0*8 +: 8] = 8'd1;  // B[0][0]
        weight_load_data[1*8 +: 8] = 8'd2;  // B[1][0]
        @(posedge clk);
        
        // Column 1: B[0][1]=2, B[1][1]=3
        weight_load_col = 1;
        weight_load_data = 0;
        weight_load_data[0*8 +: 8] = 8'd2;  // B[0][1]
        weight_load_data[1*8 +: 8] = 8'd3;  // B[1][1]
        @(posedge clk);
        
        // Columns 2,3: zeros
        weight_load_col = 2;
        weight_load_data = 0;
        @(posedge clk);
        weight_load_col = 3;
        @(posedge clk);
        weight_load_en = 0;
        @(posedge clk);
        
        $display("Weights loaded. Starting computation...");
        $display("");
        
        // Configure and start
        cfg_k_tiles = 16'd16;  // Enough cycles for computation
        
        cycle = 0;
        
        // Cycle 0: Start
        start = 1;
        clear_acc = 1;
        show_cycle();
        @(posedge clk);
        cycle = cycle + 1;
        start = 0;
        clear_acc = 0;
        
        // Cycle 1: Send A[0][*] = [1, 1, 0, 0] to rows
        act_valid = 1;
        act_data = 0;
        act_data[0*8 +: 8] = 8'd1;  // A[0][0] to row 0
        act_data[1*8 +: 8] = 8'd1;  // A[0][1] to row 1
        show_cycle();
        @(posedge clk);
        cycle = cycle + 1;
        
        // Cycle 2: Send A[1][*] = [2, 2, 0, 0]
        act_data = 0;
        act_data[0*8 +: 8] = 8'd2;  // A[1][0]
        act_data[1*8 +: 8] = 8'd2;  // A[1][1]
        show_cycle();
        @(posedge clk);
        cycle = cycle + 1;
        
        // Cycle 3+: No more activations, wait for results
        act_valid = 0;
        act_data = 0;
        
        // Run until we see results or timeout
        repeat(25) begin
            show_cycle();
            
            // Check for expected results
            if (result_valid) begin
                r0 = $signed(result_data[0*32 +: 32]);
                r1 = $signed(result_data[1*32 +: 32]);
                
                // First result row: [3, 5]
                if (r0 == 3 && r1 == 5) begin
                    $display("  >>> Found C[0] = [3, 5] - CORRECT!");
                end
                // Second result row: [6, 10]
                if (r0 == 6 && r1 == 10) begin
                    $display("  >>> Found C[1] = [6, 10] - CORRECT!");
                end
            end
            
            @(posedge clk);
            cycle = cycle + 1;
        end
        
        //======================================================================
        // TEST 2: 4x4 Identity
        // A = [1..16], B = I
        // Expected C = A
        //======================================================================
        $display("");
        $display("=== TEST 2: 4x4 Identity ===");
        $display("A = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]");
        $display("B = Identity, Expected C = A");
        $display("");
        
        // Reset for new test
        rst_n = 0;
        @(posedge clk);
        rst_n = 1;
        @(posedge clk);
        
        // Load identity weights
        for (col = 0; col < 4; col = col + 1) begin
            weight_load_en = 1;
            weight_load_col = col;
            weight_load_data = 0;
            weight_load_data[col*8 +: 8] = 8'd1;  // B[col][col] = 1
            @(posedge clk);
        end
        weight_load_en = 0;
        @(posedge clk);
        
        $display("Identity weights loaded. Starting computation...");
        $display("");
        
        cfg_k_tiles = 16'd20;
        cycle = 0;
        
        // Start
        start = 1;
        clear_acc = 1;
        show_cycle();
        @(posedge clk);
        cycle = cycle + 1;
        start = 0;
        clear_acc = 0;
        
        // Send rows of A
        // A[m][k] goes to row k at time m
        act_valid = 1;
        
        // m=0: A[0][*] = [1,2,3,4]
        act_data = {8'd4, 8'd3, 8'd2, 8'd1};
        show_cycle();
        @(posedge clk); cycle = cycle + 1;
        
        // m=1: A[1][*] = [5,6,7,8]
        act_data = {8'd8, 8'd7, 8'd6, 8'd5};
        show_cycle();
        @(posedge clk); cycle = cycle + 1;
        
        // m=2: A[2][*] = [9,10,11,12]
        act_data = {8'd12, 8'd11, 8'd10, 8'd9};
        show_cycle();
        @(posedge clk); cycle = cycle + 1;
        
        // m=3: A[3][*] = [13,14,15,16]
        act_data = {8'd16, 8'd15, 8'd14, 8'd13};
        show_cycle();
        @(posedge clk); cycle = cycle + 1;
        
        act_valid = 0;
        act_data = 0;
        
        // Wait for results
        repeat(25) begin
            show_cycle();
            
            if (result_valid) begin
                r0 = $signed(result_data[0*32 +: 32]);
                r1 = $signed(result_data[1*32 +: 32]);
                r2 = $signed(result_data[2*32 +: 32]);
                r3 = $signed(result_data[3*32 +: 32]);
                
                if (r0 == 1 && r1 == 2 && r2 == 3 && r3 == 4) 
                    $display("  >>> Found C[0] = [1,2,3,4] - CORRECT!");
                if (r0 == 5 && r1 == 6 && r2 == 7 && r3 == 8) 
                    $display("  >>> Found C[1] = [5,6,7,8] - CORRECT!");
                if (r0 == 9 && r1 == 10 && r2 == 11 && r3 == 12) 
                    $display("  >>> Found C[2] = [9,10,11,12] - CORRECT!");
                if (r0 == 13 && r1 == 14 && r2 == 15 && r3 == 16) 
                    $display("  >>> Found C[3] = [13,14,15,16] - CORRECT!");
            end
            
            @(posedge clk);
            cycle = cycle + 1;
        end
        
        $display("");
        $display("==============================================");
        $display("Test Complete");
        $display("==============================================");
        $finish;
    end

endmodule

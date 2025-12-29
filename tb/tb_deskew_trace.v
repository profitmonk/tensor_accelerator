`timescale 1ns / 1ps

module tb_deskew_trace;
    parameter CLK_PERIOD = 10;
    parameter ARRAY_SIZE = 4;
    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH  = 32;

    reg  clk = 0;
    reg  rst_n = 0;
    reg  start = 0;
    reg  clear_acc = 0;
    wire busy, done;
    reg  [15:0] cfg_k_tiles = 20;
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
    
    integer cycle, col;
    always @(posedge clk) begin
        #1;
        if (dut.state >= 2 && cycle < 20) begin
            $display("Cyc %2d | psum_bottom=[%2d,%2d,%2d,%2d] | result_data=[%2d,%2d,%2d,%2d] | rvalid=%b",
                     cycle,
                     $signed(dut.psum_bottom[0]),
                     $signed(dut.psum_bottom[1]),
                     $signed(dut.psum_bottom[2]),
                     $signed(dut.psum_bottom[3]),
                     $signed(result_data[0*32 +: 32]),
                     $signed(result_data[1*32 +: 32]),
                     $signed(result_data[2*32 +: 32]),
                     $signed(result_data[3*32 +: 32]),
                     result_valid);
            cycle = cycle + 1;
        end
    end

    initial begin
        $display("\n=== DE-SKEW TRACE (4x4 Identity: C = A) ===\n");
        $display("Expected: psum_bottom shows raw (skewed), result_data shows de-skewed");
        $display("");
        
        cycle = 0;
        
        #100 rst_n = 1;
        @(posedge clk);
        
        // Load identity weights: B[k][k] = 1
        for (col = 0; col < ARRAY_SIZE; col = col + 1) begin
            weight_load_en = 1;
            weight_load_col = col;
            weight_load_data = 0;
            weight_load_data[col*8 +: 8] = 8'd1;
            @(posedge clk);
        end
        weight_load_en = 0;
        @(posedge clk);
        
        // Start
        start = 1; clear_acc = 1;
        @(posedge clk);
        start = 0; clear_acc = 0;
        
        // Send A = [1,2,3,4; 5,6,7,8; 9,10,11,12; 13,14,15,16]
        // A[m][k] goes to row k
        act_valid = 1;
        // Row 0 gets A[*][0] = [1,5,9,13]
        // Row k gets A[*][k]
        // So act_data[k] = A[m][k] at time m
        act_data = 0;
        act_data[0*8 +: 8] = 1;  // A[0][0]
        act_data[1*8 +: 8] = 2;  // A[0][1]
        act_data[2*8 +: 8] = 3;  // A[0][2]
        act_data[3*8 +: 8] = 4;  // A[0][3]
        @(posedge clk);
        
        act_data = 0;
        act_data[0*8 +: 8] = 5;  // A[1][0]
        act_data[1*8 +: 8] = 6;  // A[1][1]
        act_data[2*8 +: 8] = 7;  // A[1][2]
        act_data[3*8 +: 8] = 8;  // A[1][3]
        @(posedge clk);
        
        act_data = 0;
        act_data[0*8 +: 8] = 9;
        act_data[1*8 +: 8] = 10;
        act_data[2*8 +: 8] = 11;
        act_data[3*8 +: 8] = 12;
        @(posedge clk);
        
        act_data = 0;
        act_data[0*8 +: 8] = 13;
        act_data[1*8 +: 8] = 14;
        act_data[2*8 +: 8] = 15;
        act_data[3*8 +: 8] = 16;
        @(posedge clk);
        
        act_valid = 0;
        act_data = 0;
        
        repeat(20) @(posedge clk);
        
        $display("\n=== Analysis ===");
        $display("If de-skew works: result_data should show aligned rows [1,2,3,4], [5,6,7,8], etc.");
        $finish;
    end
endmodule

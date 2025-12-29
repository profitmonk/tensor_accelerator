`timescale 1ns / 1ps

module tb_trace_16x16;
    parameter CLK_PERIOD = 10;
    parameter ARRAY_SIZE = 16;
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
    
    integer cycle, col;
    always @(posedge clk) begin
        #1;
        if (dut.state >= 2 && cycle < 50) begin
            $display("Cyc %2d | st=%d | psum_v[2][0]=%3d psum_v[8][0]=%3d psum_v[16][0]=%3d | rvalid=%b res[0]=%d",
                     cycle, dut.state,
                     $signed(dut.psum_v[2][0]),
                     $signed(dut.psum_v[8][0]),
                     $signed(dut.psum_v[16][0]),
                     result_valid,
                     $signed(result_data[0*ACC_WIDTH +: ACC_WIDTH]));
            cycle = cycle + 1;
        end
    end

    initial begin
        $display("\n=== 16x16 ARRAY TRACE (2x2 multiply) ===\n");
        $display("Watching psum at rows 2, 8, 16 (output)");
        $display("C[0][0]=3 should propagate: row2 -> row8 -> row16\n");
        
        cycle = 0;
        cfg_k_tiles = 32;  // K + propagation
        
        #100 rst_n = 1;
        @(posedge clk);
        
        // Load 2x2 weights
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
        @(posedge clk);
        start = 0; clear_acc = 0;
        
        // Send 2 activations for 2x2
        act_valid = 1;
        act_data = 0;
        act_data[0*8 +: 8] = 1;  // A[0][0]
        act_data[1*8 +: 8] = 1;  // A[0][1]
        @(posedge clk);
        
        act_data = 0;
        act_data[0*8 +: 8] = 2;  // A[1][0]
        act_data[1*8 +: 8] = 2;  // A[1][1]
        @(posedge clk);
        
        act_valid = 0;
        act_data = 0;
        
        // Wait and observe
        repeat(50) @(posedge clk);
        
        $display("\nDone=%b, Final result[0]=%d", done, $signed(result_data[0*ACC_WIDTH +: ACC_WIDTH]));
        $finish;
    end
endmodule

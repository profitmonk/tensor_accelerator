`timescale 1ns / 1ps

module tb_systolic_debug;
    parameter CLK_PERIOD = 10;
    parameter ARRAY_SIZE = 16;
    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH  = 32;

    reg  clk;
    reg  rst_n;
    reg  start;
    reg  clear_acc;
    wire busy;
    wire done;
    reg  [15:0] cfg_k_tiles;
    reg  weight_load_en;
    reg  [$clog2(ARRAY_SIZE)-1:0] weight_load_col;
    reg  [ARRAY_SIZE*DATA_WIDTH-1:0] weight_load_data;
    reg  act_valid;
    reg  [ARRAY_SIZE*DATA_WIDTH-1:0] act_data;
    wire act_ready;
    wire result_valid;
    wire [ARRAY_SIZE*ACC_WIDTH-1:0] result_data;
    reg  result_ready;

    systolic_array #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .clear_acc(clear_acc),
        .busy(busy), .done(done), .cfg_k_tiles(cfg_k_tiles),
        .weight_load_en(weight_load_en), .weight_load_col(weight_load_col),
        .weight_load_data(weight_load_data),
        .act_valid(act_valid), .act_data(act_data), .act_ready(act_ready),
        .result_valid(result_valid), .result_data(result_data), .result_ready(result_ready)
    );

    initial begin clk = 0; forever #(CLK_PERIOD/2) clk = ~clk; end

    // Monitor state
    always @(posedge clk) begin
        $display("[%0t] state=%0d cycle_count=%0d busy=%b done=%b act_ready=%b result_valid=%b",
                 $time, dut.state, dut.cycle_count, busy, done, act_ready, result_valid);
    end

    initial begin
        $display("=== Debug Testbench ===");
        
        // Init
        rst_n = 0; start = 0; clear_acc = 0;
        weight_load_en = 0; weight_load_col = 0; weight_load_data = 0;
        act_valid = 0; act_data = 0; result_ready = 1;
        cfg_k_tiles = 2;
        
        #100 rst_n = 1;
        #20;
        
        // Load weights for 2x2 in corner
        $display("--- Loading weights ---");
        weight_load_en = 1;
        weight_load_col = 0;
        weight_load_data = 0;
        weight_load_data[0*8 +: 8] = 1; // B[0][0] = 1
        weight_load_data[1*8 +: 8] = 2; // B[1][0] = 2
        #10;
        
        weight_load_col = 1;
        weight_load_data = 0;
        weight_load_data[0*8 +: 8] = 2; // B[0][1] = 2
        weight_load_data[1*8 +: 8] = 3; // B[1][1] = 3
        #10;
        
        weight_load_en = 0;
        #10;
        
        // Start
        $display("--- Starting computation ---");
        start = 1;
        clear_acc = 1;
        #10;
        start = 0;
        clear_acc = 0;
        
        // Wait to see state transitions
        repeat(10) @(posedge clk);
        
        // Stream activations
        $display("--- Streaming activations ---");
        act_valid = 1;
        act_data = 0;
        act_data[0*8 +: 8] = 1; // A[0][0] = 1
        act_data[1*8 +: 8] = 2; // A[1][0] = 2
        #10;
        
        act_data = 0;
        act_data[0*8 +: 8] = 1; // A[0][1] = 1
        act_data[1*8 +: 8] = 2; // A[1][1] = 2
        #10;
        
        act_valid = 0;
        
        // Wait for results
        $display("--- Waiting for results ---");
        repeat(50) @(posedge clk);
        
        $display("=== Test Complete ===");
        $finish;
    end
endmodule

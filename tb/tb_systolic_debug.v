`timescale 1ns / 1ps
module tb_systolic_debug;
    parameter CLK = 10;
    parameter ARRAY_SIZE = 4;
    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH = 32;

    reg clk = 0;
    reg rst_n = 0;
    always #(CLK/2) clk = ~clk;

    reg start = 0;
    reg clear_acc = 1;
    wire busy, done;
    reg [15:0] cfg_k_tiles = ARRAY_SIZE;
    
    reg weight_load_en = 0;
    reg [$clog2(ARRAY_SIZE)-1:0] weight_load_col = 0;
    reg [ARRAY_SIZE*DATA_WIDTH-1:0] weight_load_data = 0;
    
    reg act_valid = 0;
    reg [ARRAY_SIZE*DATA_WIDTH-1:0] act_data = 0;
    wire act_ready;
    
    wire result_valid;
    wire [ARRAY_SIZE*ACC_WIDTH-1:0] result_data;
    reg result_ready = 1;

    systolic_array #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .clear_acc(clear_acc),
        .busy(busy), .done(done),
        .cfg_k_tiles(cfg_k_tiles),
        .weight_load_en(weight_load_en),
        .weight_load_col(weight_load_col),
        .weight_load_data(weight_load_data),
        .act_valid(act_valid),
        .act_data(act_data),
        .act_ready(act_ready),
        .result_valid(result_valid),
        .result_data(result_data),
        .result_ready(result_ready)
    );

    integer i, cycle;
    
    initial begin
        $display("Systolic Array Timing with drain_delay = 2*4-1 = 7");
        
        #(CLK*3); rst_n = 1; #(CLK*3);

        // Load identity weights
        for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
            weight_load_en = 1;
            weight_load_col = i;
            weight_load_data = 0;
            weight_load_data[i*DATA_WIDTH +: DATA_WIDTH] = 8'd1;
            @(posedge clk);
        end
        weight_load_en = 0;
        @(posedge clk);
        
        start = 1;
        @(posedge clk);
        start = 0;
        cycle = 0;
        
        // Stream activations
        act_valid = 1;
        act_data = {8'd4, 8'd3, 8'd2, 8'd1}; @(posedge clk); cycle = cycle + 1;
        act_data = {8'd8, 8'd7, 8'd6, 8'd5}; @(posedge clk); cycle = cycle + 1;
        act_data = {8'd12, 8'd11, 8'd10, 8'd9}; @(posedge clk); cycle = cycle + 1;
        act_data = {8'd16, 8'd15, 8'd14, 8'd13}; @(posedge clk); cycle = cycle + 1;
        act_valid = 0;
        
        // Trace
        for (i = 0; i < 20; i = i + 1) begin
            @(posedge clk);
            cycle = cycle + 1;
            $display("Cyc %2d: state=%d drain_cnt=%2d valid=%b result=[%2d,%2d,%2d,%2d]",
                cycle, dut.state, dut.cycle_count, result_valid,
                $signed(result_data[31:0]), $signed(result_data[63:32]),
                $signed(result_data[95:64]), $signed(result_data[127:96]));
            if (done) i = 999;
        end
        $finish;
    end
endmodule

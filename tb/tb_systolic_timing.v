`timescale 1ns / 1ps
module tb_systolic_timing;
    parameter CLK = 10;
    parameter ARRAY_SIZE = 16;
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

    integer i, j, cycle, row_count;
    reg [7:0] fill_val;
    
    initial begin
        $display("Systolic Array Timing Debug (16x16)");
        $display("drain_delay = 2*%0d = %0d", ARRAY_SIZE, 2*ARRAY_SIZE);
        
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
        
        // Stream activations: row i has value (i+1) in all columns
        for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
            act_valid = 1;
            fill_val = i + 1;
            for (j = 0; j < ARRAY_SIZE; j = j + 1) begin
                act_data[j*DATA_WIDTH +: DATA_WIDTH] = fill_val;
            end
            @(posedge clk);
            cycle = cycle + 1;
        end
        act_valid = 0;
        
        // Wait for results
        row_count = 0;
        for (i = 0; i < 80; i = i + 1) begin
            @(posedge clk);
            cycle = cycle + 1;
            if (result_valid) begin
                row_count = row_count + 1;
                $display("[Cycle %0d] drain_cnt=%0d row=%0d first_val=%0d",
                    cycle, dut.cycle_count, row_count,
                    $signed(result_data[31:0]));
            end
            if (done) begin
                $display("[Cycle %0d] DONE - captured %0d rows", cycle, row_count);
                i = 999;
            end
        end
        $finish;
    end
endmodule

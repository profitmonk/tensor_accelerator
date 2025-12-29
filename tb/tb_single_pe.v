`timescale 1ns / 1ps

module tb_single_pe;
    parameter CLK_PERIOD = 10;
    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH  = 32;

    reg clk = 0;
    reg rst_n = 0;
    reg enable = 0;
    reg load_weight = 0;
    reg clear_acc = 0;
    reg [DATA_WIDTH-1:0] weight_in = 0;
    reg [DATA_WIDTH-1:0] act_in = 0;
    wire [DATA_WIDTH-1:0] act_out;
    reg [ACC_WIDTH-1:0] psum_in = 0;
    wire [ACC_WIDTH-1:0] psum_out;

    mac_pe #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) dut (.*);

    always #(CLK_PERIOD/2) clk = ~clk;
    
    // Monitor with delay to see post-update values
    always @(posedge clk) begin
        #1;  // Small delay to see updated values
        $display("[%0t] load_weight=%b enable=%b weight_reg=%0d act_reg=%0d product=%0d psum_out=%0d",
                 $time, load_weight, enable, dut.weight_reg, dut.act_reg, dut.product, psum_out);
    end

    initial begin
        $display("\n=== MAC PE Debug ===\n");
        
        #100 rst_n = 1;
        @(posedge clk);  // Wait for first clean edge after reset
        
        // Load weight = 2
        $display("Setting load_weight=1, weight_in=2...");
        load_weight = 1;
        weight_in = 8'd2;
        @(posedge clk);
        load_weight = 0;
        
        $display("After load: weight_reg = %0d (expect 2)", dut.weight_reg);
        
        // Enable and send data
        enable = 1;
        psum_in = 32'd5;
        act_in = 8'd3;
        
        repeat(4) @(posedge clk);
        
        $display("\nFinal psum_out = %0d (expect 11 = 5 + 3*2)", psum_out);
        $finish;
    end
endmodule

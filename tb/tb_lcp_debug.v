`timescale 1ns / 1ps
module tb_lcp_debug;
    parameter CLK = 10;
    
    reg clk = 0;
    reg rst_n = 0;
    reg start = 0;
    reg [19:0] start_pc = 0;
    wire busy, done, error;
    
    wire [19:0] imem_addr;
    reg [127:0] imem_data;
    wire imem_re;
    reg imem_valid = 0;
    
    wire [127:0] mxu_cmd;
    wire mxu_valid;
    reg mxu_ready = 1;
    reg mxu_done = 0;
    
    wire [127:0] vpu_cmd, dma_cmd;
    wire vpu_valid, dma_valid;
    reg vpu_ready = 1, vpu_done = 0;
    reg dma_ready = 1, dma_done = 0;
    
    reg global_sync_in = 0;
    wire sync_request;
    reg sync_grant = 0;
    
    local_cmd_processor dut (.*);
    
    always #(CLK/2) clk = ~clk;
    
    // Instruction memory
    reg [127:0] imem [0:15];
    
    always @(posedge clk) begin
        imem_valid <= imem_re;
        if (imem_re) begin
            imem_data <= imem[imem_addr[3:0]];
            $display("  [%0t] FETCH addr=%0d data=%h", $time, imem_addr[3:0], imem[imem_addr[3:0]]);
        end
    end
    
    // Monitor state
    always @(posedge clk) begin
        $display("  [%0t] state=%0d pc=%0d busy=%b done=%b imem_re=%b imem_valid=%b", 
                 $time, dut.state, dut.pc, busy, done, imem_re, imem_valid);
    end
    
    initial begin
        $display("LCP Debug Test");
        
        // NOP=00, HALT=FF
        imem[0] = {8'h00, 120'd0};  // NOP
        imem[1] = {8'hFF, 120'd0};  // HALT
        
        #50 rst_n = 1;
        #20;
        
        $display("Starting LCP...");
        start_pc = 0;
        start = 1;
        @(posedge clk);
        start = 0;
        
        repeat(50) @(posedge clk);
        
        $display("Final: done=%b error=%b state=%0d", done, error, dut.state);
        $finish;
    end
endmodule

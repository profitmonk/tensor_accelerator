`timescale 1ns / 1ps
module tb_lcp_debug2;
    parameter CLK = 10;
    
    reg clk = 0;
    reg rst_n = 0;
    reg start = 0;
    reg [19:0] start_pc = 0;
    wire busy, done, error;
    
    wire [19:0] imem_addr;
    reg [127:0] imem_data = 0;
    wire imem_re;
    reg imem_valid = 0;
    
    wire [127:0] mxu_cmd, vpu_cmd, dma_cmd;
    wire mxu_valid, vpu_valid, dma_valid;
    reg mxu_ready = 1, vpu_ready = 1, dma_ready = 1;
    reg mxu_done = 0, vpu_done = 0, dma_done = 0;
    
    reg global_sync_in = 0;
    wire sync_request;
    reg sync_grant = 0;
    
    local_cmd_processor dut (.*);
    
    always #(CLK/2) clk = ~clk;
    
    // Instruction memory with HALT at address 0
    always @(posedge clk) begin
        imem_valid <= imem_re;
        if (imem_re) begin
            imem_data <= {8'hFF, 120'd0};  // Always return HALT
        end
    end
    
    initial begin
        $display("LCP Debug Test 2 - Detailed Timing");
        $display("Clock period = %0d ns", CLK);
        
        // Reset sequence
        $display("[%0t] Reset asserted", $time);
        #(CLK * 5);
        rst_n = 1;
        $display("[%0t] Reset released, state=%0d", $time, dut.state);
        
        #(CLK * 2);
        $display("[%0t] Before start, state=%0d", $time, dut.state);
        
        // Assert start for one full clock cycle
        @(negedge clk);  // Align to negedge
        $display("[%0t] At negedge, asserting start", $time);
        start_pc = 0;
        start = 1;
        
        @(posedge clk);
        $display("[%0t] At posedge with start=1, state=%0d", $time, dut.state);
        
        @(negedge clk);
        start = 0;
        $display("[%0t] Start deasserted", $time);
        
        // Watch for 20 cycles
        repeat(20) begin
            @(posedge clk);
            $display("[%0t] state=%0d pc=%0d imem_re=%b imem_valid=%b done=%b", 
                     $time, dut.state, dut.pc, imem_re, imem_valid, done);
            if (done) begin
                $display(">>> LCP DONE! <<<");
                $finish;
            end
        end
        
        $display("LCP did not complete in 20 cycles");
        $finish;
    end
endmodule

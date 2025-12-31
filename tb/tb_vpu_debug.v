`timescale 1ns/1ps
module tb_vpu_debug;
    parameter LANES = 64;
    parameter DATA_WIDTH = 16;
    parameter SRAM_ADDR_W = 20;
    
    reg clk, rst_n;
    reg [127:0] cmd;
    reg cmd_valid;
    wire cmd_ready, cmd_done;
    wire [SRAM_ADDR_W-1:0] sram_addr;
    wire [LANES*DATA_WIDTH-1:0] sram_wdata;
    reg [LANES*DATA_WIDTH-1:0] sram_rdata;
    wire sram_we, sram_re;
    reg sram_ready;
    
    vector_unit #(
        .LANES(LANES), .DATA_WIDTH(DATA_WIDTH), .SRAM_ADDR_W(SRAM_ADDR_W)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .cmd(cmd), .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_done(cmd_done),
        .sram_addr(sram_addr), .sram_wdata(sram_wdata), .sram_rdata(sram_rdata),
        .sram_we(sram_we), .sram_re(sram_re), .sram_ready(sram_ready)
    );
    
    initial clk = 0;
    always #5 clk = ~clk;
    
    always @(posedge clk) begin
        if (cmd_valid && cmd_ready)
            $display("[%0t] CMD ACCEPTED: subop=%h (input cmd[119:112]=%h)", 
                     $time, dut.subop_reg, cmd[119:112]);
        $display("[%0t] state=%0d subop_reg=%h", $time, dut.state, dut.subop_reg);
    end
    
    localparam VOP_LOAD = 8'h30;
    
    function [127:0] vpu_cmd;
        input [7:0] subop;
        input [4:0] vd, vs1;
        input [19:0] addr;
    begin
        vpu_cmd = 0;
        vpu_cmd[127:120] = 8'h02;
        vpu_cmd[119:112] = subop;
        vpu_cmd[116:112] = vd;
        vpu_cmd[111:107] = vs1;
        vpu_cmd[95:76] = addr;
        vpu_cmd[63:48] = 16'd64;
    end
    endfunction
    
    initial begin
        rst_n = 0; cmd = 0; cmd_valid = 0; sram_ready = 1;
        sram_rdata = 0;
        
        repeat(3) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        $display("\n=== Test cmd encoding ===");
        cmd = vpu_cmd(VOP_LOAD, 5'd0, 5'd0, 20'h0);
        $display("cmd = %h", cmd);
        $display("cmd[127:120] = %h (opcode)", cmd[127:120]);
        $display("cmd[119:112] = %h (subop, expect 0x30)", cmd[119:112]);
        
        cmd_valid = 1;
        repeat(10) @(posedge clk);
        
        $finish;
    end
    
    initial #2000 begin $display("TIMEOUT"); $finish; end
endmodule

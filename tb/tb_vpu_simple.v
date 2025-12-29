`timescale 1ns / 1ps
module tb_vpu_simple;
    parameter CLK = 10;
    reg clk = 0;
    reg rst_n = 0;
    always #(CLK/2) clk = ~clk;

    reg [127:0] cmd = 0;
    reg cmd_valid = 0;
    wire cmd_ready, cmd_done;
    
    wire [19:0] sram_addr;
    wire [127:0] sram_wdata;
    reg [127:0] sram_rdata = 0;
    wire sram_we, sram_re;
    reg sram_ready = 1;

    vector_unit #(.LANES(8), .DATA_WIDTH(16)) dut (
        .clk(clk), .rst_n(rst_n),
        .cmd(cmd), .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_done(cmd_done),
        .sram_addr(sram_addr), .sram_wdata(sram_wdata), .sram_rdata(sram_rdata),
        .sram_we(sram_we), .sram_re(sram_re), .sram_ready(sram_ready)
    );

    integer i;
    reg done_seen = 0;
    reg cmd_accepted = 0;
    
    initial begin
        $display("VPU Simple Debug");
        
        #(CLK*3); rst_n = 1; #(CLK*2);
        
        dut.vrf[0] = {16'd8, 16'd7, 16'd6, 16'd5, 16'd4, 16'd3, 16'd2, 16'd1};
        
        $display("Before: state=%0d, cmd_ready=%b", dut.state, cmd_ready);
        
        // VOP_ZERO = 0x34
        cmd = 0;
        cmd[127:120] = 8'h02;
        cmd[119:112] = 8'h34;
        cmd[111:107] = 5'd0;
        
        // Assert cmd_valid BEFORE the clock edge
        cmd_valid = 1;
        
        for (i = 0; i < 20 && !done_seen; i = i + 1) begin
            @(posedge clk);
            #1;  // Small delay to let combinational logic settle
            $display("Cyc %0d: st=%0d rdy=%b done=%b valid=%b", i, dut.state, cmd_ready, cmd_done, cmd_valid);
            
            // Deassert cmd_valid ONE cycle AFTER it's accepted
            if (cmd_accepted) begin
                cmd_valid = 0;
                cmd_accepted = 0;
            end
            if (cmd_ready && cmd_valid && !cmd_accepted) begin
                cmd_accepted = 1;
                $display("  -> Command accepted, will deassert next cycle");
            end
            if (cmd_done) done_seen = 1;
        end
        
        $display("V20: %h", dut.vrf[20]);
        if (dut.vrf[20] == 0) $display("PASS: ZERO worked");
        else $display("FAIL: V20 not zero");
        $finish;
    end
endmodule

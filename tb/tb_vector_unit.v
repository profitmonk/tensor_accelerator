`timescale 1ns/1ps
/*
 * Vector Unit Testbench
 * Tests basic VPU operations with corrected command encoding
 */

module tb_vector_unit;

    parameter LANES = 8;
    parameter DATA_WIDTH = 16;
    parameter VREG_COUNT = 32;
    parameter SRAM_ADDR_W = 20;
    parameter CLK = 10;
    
    reg clk, rst_n;
    reg [127:0] cmd;
    reg cmd_valid;
    wire cmd_ready, cmd_done;
    wire [SRAM_ADDR_W-1:0] sram_addr;
    wire [LANES*DATA_WIDTH-1:0] sram_wdata;
    reg [LANES*DATA_WIDTH-1:0] sram_rdata;
    wire sram_we, sram_re;
    reg sram_ready;
    
    // VPU opcodes
    localparam VOP_ADD  = 8'h01;
    localparam VOP_RELU = 8'h10;
    localparam VOP_SUM  = 8'h20;
    localparam VOP_ZERO = 8'h34;
    
    vector_unit #(
        .LANES(LANES), .DATA_WIDTH(DATA_WIDTH), .VREG_COUNT(VREG_COUNT), .SRAM_ADDR_W(SRAM_ADDR_W)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .cmd(cmd), .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_done(cmd_done),
        .sram_addr(sram_addr), .sram_wdata(sram_wdata), .sram_rdata(sram_rdata),
        .sram_we(sram_we), .sram_re(sram_re), .sram_ready(sram_ready)
    );
    
    initial clk = 0;
    always #(CLK/2) clk = ~clk;
    
    // Command builder: [127:120]=opcode, [119:112]=subop, [111:107]=vd, [106:102]=vs1, [101:97]=vs2
    function [127:0] make_cmd;
        input [7:0] subop;
        input [4:0] vd, vs1, vs2;
    begin
        make_cmd = 0;
        make_cmd[127:120] = 8'h02;    // VPU opcode
        make_cmd[119:112] = subop;
        make_cmd[111:107] = vd;
        make_cmd[106:102] = vs1;
        make_cmd[101:97]  = vs2;
    end
    endfunction
    
    task issue_cmd;
        input [127:0] c;
    begin
        cmd = c; cmd_valid = 1;
        @(posedge clk);
        while (!cmd_ready) @(posedge clk);
        @(posedge clk);
        cmd_valid = 0;
        while (!cmd_done) @(posedge clk);
        @(posedge clk);
    end
    endtask
    
    task init_vrf;
        integer k;
    begin
        for (k = 0; k < VREG_COUNT; k = k + 1) dut.vrf[k] = 0;
    end
    endtask
    
    task print_vreg;
        input [4:0] idx;
    begin
        $display("    V%0d = [%d, %d, %d, %d, %d, %d, %d, %d]", idx,
            $signed(dut.vrf[idx][7*DATA_WIDTH +: DATA_WIDTH]),
            $signed(dut.vrf[idx][6*DATA_WIDTH +: DATA_WIDTH]),
            $signed(dut.vrf[idx][5*DATA_WIDTH +: DATA_WIDTH]),
            $signed(dut.vrf[idx][4*DATA_WIDTH +: DATA_WIDTH]),
            $signed(dut.vrf[idx][3*DATA_WIDTH +: DATA_WIDTH]),
            $signed(dut.vrf[idx][2*DATA_WIDTH +: DATA_WIDTH]),
            $signed(dut.vrf[idx][1*DATA_WIDTH +: DATA_WIDTH]),
            $signed(dut.vrf[idx][0*DATA_WIDTH +: DATA_WIDTH]));
    end
    endtask
    
    integer errors;
    reg signed [DATA_WIDTH-1:0] val;
    
    initial begin
        $display("");
        $display("╔══════════════════════════════════════════════════════════════╗");
        $display("║              Vector Unit Testbench                           ║");
        $display("╚══════════════════════════════════════════════════════════════╝");
        
        rst_n = 0; cmd = 0; cmd_valid = 0; sram_ready = 1; sram_rdata = 0;
        errors = 0;
        
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(3) @(posedge clk);
        init_vrf();
        
        //======================================================================
        // TEST 1: Vector ADD
        //======================================================================
        $display("");
        $display("[TEST 1] Vector ADD: V2 = V0 + V1");
        
        dut.vrf[0] = {16'd8, 16'd7, 16'd6, 16'd5, 16'd4, 16'd3, 16'd2, 16'd1};
        dut.vrf[1] = {16'd80, 16'd70, 16'd60, 16'd50, 16'd40, 16'd30, 16'd20, 16'd10};
        print_vreg(0);
        print_vreg(1);
        
        issue_cmd(make_cmd(VOP_ADD, 5'd2, 5'd0, 5'd1));  // vd=2, vs1=0, vs2=1
        
        print_vreg(2);
        val = $signed(dut.vrf[2][0 +: DATA_WIDTH]);
        if (val == 11) $display("  PASS: V2[0] = %0d (1+10)", val);
        else begin $display("  FAIL: V2[0] = %0d, expected 11", val); errors = errors + 1; end
        
        val = $signed(dut.vrf[2][7*DATA_WIDTH +: DATA_WIDTH]);
        if (val == 88) $display("  PASS: V2[7] = %0d (8+80)", val);
        else begin $display("  FAIL: V2[7] = %0d, expected 88", val); errors = errors + 1; end

        #(CLK * 5);

        //======================================================================
        // TEST 2: ReLU
        //======================================================================
        $display("");
        $display("[TEST 2] ReLU: V16 = relu(V5)");
        
        init_vrf();
        dut.vrf[5] = {16'sh0007, 16'sh0005, 16'sh0003, 16'sh0001, 16'sh0000, 16'shFFFF, 16'shFFFD, 16'shFFFB};
        print_vreg(5);
        
        issue_cmd(make_cmd(VOP_RELU, 5'd16, 5'd5, 5'd0));  // vd=16, vs1=5
        
        print_vreg(16);
        val = $signed(dut.vrf[16][0 +: DATA_WIDTH]);
        if (val == 0) $display("  PASS: relu(-5) = 0");
        else begin $display("  FAIL: relu(-5) = %0d, expected 0", val); errors = errors + 1; end
        
        val = $signed(dut.vrf[16][7*DATA_WIDTH +: DATA_WIDTH]);
        if (val == 7) $display("  PASS: relu(7) = 7");
        else begin $display("  FAIL: relu(7) = %0d, expected 7", val); errors = errors + 1; end

        #(CLK * 5);

        //======================================================================
        // TEST 3: Reduction SUM
        //======================================================================
        $display("");
        $display("[TEST 3] Reduction SUM: V0 = sum(V3)");
        
        init_vrf();
        dut.vrf[3] = {16'd8, 16'd7, 16'd6, 16'd5, 16'd4, 16'd3, 16'd2, 16'd1};
        print_vreg(3);
        
        issue_cmd(make_cmd(VOP_SUM, 5'd0, 5'd3, 5'd0));  // vd=0, vs1=3
        
        print_vreg(0);
        val = $signed(dut.vrf[0][0 +: DATA_WIDTH]);
        if (val == 36) $display("  PASS: sum([1..8]) = %0d", val);
        else begin $display("  FAIL: sum = %0d, expected 36", val); errors = errors + 1; end

        #(CLK * 5);

        //======================================================================
        // TEST 4: ZERO
        //======================================================================
        $display("");
        $display("[TEST 4] Vector ZERO: V20 = 0");
        
        dut.vrf[20] = {LANES{16'hFFFF}};
        $display("    Before: V20 = %h", dut.vrf[20]);
        
        issue_cmd(make_cmd(VOP_ZERO, 5'd20, 5'd0, 5'd0));  // vd=20
        
        $display("    After:  V20 = %h", dut.vrf[20]);
        if (dut.vrf[20] == 0) $display("  PASS: V20 zeroed");
        else begin $display("  FAIL: V20 not zero"); errors = errors + 1; end

        //======================================================================
        // Summary
        //======================================================================
        #(CLK * 10);
        $display("");
        $display("════════════════════════════════════════════════════════════");
        if (errors == 0) begin
            $display(">>> ALL TESTS PASSED! <<<");
        end else begin
            $display(">>> SOME TESTS FAILED <<<");
            $display("    Errors: %0d", errors);
        end
        $display("════════════════════════════════════════════════════════════");
        $finish;
    end
    
    initial #50000 begin $display("TIMEOUT"); $finish; end

endmodule

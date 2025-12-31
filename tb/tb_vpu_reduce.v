`timescale 1ns / 1ps
//==============================================================================
// VPU Reduction Tests: SUM, MAX, MIN
//
// Tests reduction operations used for:
//   - SUM: GlobalAveragePool, ReduceSum, LayerNorm (mean)
//   - MAX: MaxPool, ReduceMax, Softmax (max for stability)
//   - MIN: Clip, ReduceMin
//
// ONNX ops: ReduceSum, ReduceMax, ReduceMin, GlobalAveragePool, MaxPool
//==============================================================================

module tb_vpu_reduce;

    parameter LANES = 8;
    parameter DATA_WIDTH = 16;
    parameter VREG_COUNT = 32;
    parameter SRAM_ADDR_W = 20;
    parameter CLK = 10;

    reg clk = 0;
    reg rst_n = 0;
    always #(CLK/2) clk = ~clk;

    // VPU interface
    reg [127:0] cmd;
    reg cmd_valid;
    wire cmd_ready, cmd_done;
    wire [SRAM_ADDR_W-1:0] sram_addr;
    wire [LANES*DATA_WIDTH-1:0] sram_wdata;
    reg [LANES*DATA_WIDTH-1:0] sram_rdata;
    wire sram_we, sram_re;
    reg sram_ready = 1;

    // DUT
    vector_unit #(
        .LANES(LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .VREG_COUNT(VREG_COUNT),
        .SRAM_ADDR_W(SRAM_ADDR_W)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .cmd(cmd),
        .cmd_valid(cmd_valid),
        .cmd_ready(cmd_ready),
        .cmd_done(cmd_done),
        .sram_addr(sram_addr),
        .sram_wdata(sram_wdata),
        .sram_rdata(sram_rdata),
        .sram_we(sram_we),
        .sram_re(sram_re),
        .sram_ready(sram_ready)
    );

    // Opcodes
    localparam VOP_SUM  = 8'h20;
    localparam VOP_MAX  = 8'h21;
    localparam VOP_MIN  = 8'h22;

    // Command builder
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

    integer i, errors;
    reg signed [DATA_WIDTH-1:0] val, expected;

    initial begin
        $display("");
        $display("╔══════════════════════════════════════════════════════════════╗");
        $display("║         VPU Reduction Tests: SUM, MAX, MIN                   ║");
        $display("╚══════════════════════════════════════════════════════════════╝");
        $display("");

        errors = 0;
        cmd = 0;
        cmd_valid = 0;

        rst_n = 0;
        #(CLK*5);
        rst_n = 1;
        #(CLK*5);

        init_vrf();

        //======================================================================
        // Test 1: SUM reduction [1,2,3,4,5,6,7,8] → 36
        //======================================================================
        $display("[TEST 1] SUM reduction");
        
        // V0 = [1,2,3,4,5,6,7,8]
        for (i = 0; i < LANES; i = i + 1)
            dut.vrf[0][i*DATA_WIDTH +: DATA_WIDTH] = i + 1;
        
        // V1 = SUM(V0)
        issue_cmd(make_cmd(VOP_SUM, 5'd1, 5'd0, 5'd0));

        // Result should be 1+2+3+4+5+6+7+8 = 36
        // Reduction result typically stored in lane 0
        val = $signed(dut.vrf[1][0*DATA_WIDTH +: DATA_WIDTH]);
        if (val == 36) begin
            $display("  PASS: SUM([1..8]) = 36");
        end else begin
            $display("  FAIL: SUM([1..8]) = %0d (expected 36)", val);
            errors = errors + 1;
        end

        //======================================================================
        // Test 2: SUM with negative values [-4,-3,-2,-1,1,2,3,4] → 0
        //======================================================================
        $display("");
        $display("[TEST 2] SUM with negatives");

        dut.vrf[2][0*DATA_WIDTH +: DATA_WIDTH] = -16'd4;
        dut.vrf[2][1*DATA_WIDTH +: DATA_WIDTH] = -16'd3;
        dut.vrf[2][2*DATA_WIDTH +: DATA_WIDTH] = -16'd2;
        dut.vrf[2][3*DATA_WIDTH +: DATA_WIDTH] = -16'd1;
        dut.vrf[2][4*DATA_WIDTH +: DATA_WIDTH] = 16'd1;
        dut.vrf[2][5*DATA_WIDTH +: DATA_WIDTH] = 16'd2;
        dut.vrf[2][6*DATA_WIDTH +: DATA_WIDTH] = 16'd3;
        dut.vrf[2][7*DATA_WIDTH +: DATA_WIDTH] = 16'd4;

        issue_cmd(make_cmd(VOP_SUM, 5'd3, 5'd2, 5'd2));

        val = $signed(dut.vrf[3][0*DATA_WIDTH +: DATA_WIDTH]);
        if (val == 0) begin
            $display("  PASS: SUM([-4..4]) = 0");
        end else begin
            $display("  FAIL: SUM([-4..4]) = %0d (expected 0)", val);
            errors = errors + 1;
        end

        //======================================================================
        // Test 3: MAX reduction [3,7,2,9,1,8,4,6] → 9
        //======================================================================
        $display("");
        $display("[TEST 3] MAX reduction");

        dut.vrf[4][0*DATA_WIDTH +: DATA_WIDTH] = 16'd3;
        dut.vrf[4][1*DATA_WIDTH +: DATA_WIDTH] = 16'd7;
        dut.vrf[4][2*DATA_WIDTH +: DATA_WIDTH] = 16'd2;
        dut.vrf[4][3*DATA_WIDTH +: DATA_WIDTH] = 16'd9;
        dut.vrf[4][4*DATA_WIDTH +: DATA_WIDTH] = 16'd1;
        dut.vrf[4][5*DATA_WIDTH +: DATA_WIDTH] = 16'd8;
        dut.vrf[4][6*DATA_WIDTH +: DATA_WIDTH] = 16'd4;
        dut.vrf[4][7*DATA_WIDTH +: DATA_WIDTH] = 16'd6;

        issue_cmd(make_cmd(VOP_MAX, 5'd5, 5'd4, 5'd4));

        val = $signed(dut.vrf[5][0*DATA_WIDTH +: DATA_WIDTH]);
        if (val == 9) begin
            $display("  PASS: MAX([3,7,2,9,1,8,4,6]) = 9");
        end else begin
            $display("  FAIL: MAX = %0d (expected 9)", val);
            errors = errors + 1;
        end

        //======================================================================
        // Test 4: MAX with negative values [-5,2,-8,1,-3,4,-7,0] → 4
        //======================================================================
        $display("");
        $display("[TEST 4] MAX with negatives");

        dut.vrf[6][0*DATA_WIDTH +: DATA_WIDTH] = -16'd5;
        dut.vrf[6][1*DATA_WIDTH +: DATA_WIDTH] = 16'd2;
        dut.vrf[6][2*DATA_WIDTH +: DATA_WIDTH] = -16'd8;
        dut.vrf[6][3*DATA_WIDTH +: DATA_WIDTH] = 16'd1;
        dut.vrf[6][4*DATA_WIDTH +: DATA_WIDTH] = -16'd3;
        dut.vrf[6][5*DATA_WIDTH +: DATA_WIDTH] = 16'd4;
        dut.vrf[6][6*DATA_WIDTH +: DATA_WIDTH] = -16'd7;
        dut.vrf[6][7*DATA_WIDTH +: DATA_WIDTH] = 16'd0;

        issue_cmd(make_cmd(VOP_MAX, 5'd7, 5'd6, 5'd6));

        val = $signed(dut.vrf[7][0*DATA_WIDTH +: DATA_WIDTH]);
        if (val == 4) begin
            $display("  PASS: MAX([-5,2,-8,1,-3,4,-7,0]) = 4");
        end else begin
            $display("  FAIL: MAX = %0d (expected 4)", val);
            errors = errors + 1;
        end

        //======================================================================
        // Test 5: MIN reduction [3,7,2,9,1,8,4,6] → 1
        //======================================================================
        $display("");
        $display("[TEST 5] MIN reduction");

        // Reuse V4 data
        issue_cmd(make_cmd(VOP_MIN, 5'd8, 5'd4, 5'd4));

        val = $signed(dut.vrf[8][0*DATA_WIDTH +: DATA_WIDTH]);
        if (val == 1) begin
            $display("  PASS: MIN([3,7,2,9,1,8,4,6]) = 1");
        end else begin
            $display("  FAIL: MIN = %0d (expected 1)", val);
            errors = errors + 1;
        end

        //======================================================================
        // Test 6: MIN with negative values → -8
        //======================================================================
        $display("");
        $display("[TEST 6] MIN with negatives");

        // Reuse V6 data
        issue_cmd(make_cmd(VOP_MIN, 5'd9, 5'd6, 5'd6));

        val = $signed(dut.vrf[9][0*DATA_WIDTH +: DATA_WIDTH]);
        if (val == -8) begin
            $display("  PASS: MIN([-5,2,-8,1,-3,4,-7,0]) = -8");
        end else begin
            $display("  FAIL: MIN = %0d (expected -8)", val);
            errors = errors + 1;
        end

        //======================================================================
        // Test 7: All same values [5,5,5,5,5,5,5,5]
        //======================================================================
        $display("");
        $display("[TEST 7] All same values");

        for (i = 0; i < LANES; i = i + 1)
            dut.vrf[10][i*DATA_WIDTH +: DATA_WIDTH] = 16'd5;

        issue_cmd(make_cmd(VOP_SUM, 5'd11, 5'd10, 5'd10));
        val = $signed(dut.vrf[11][0*DATA_WIDTH +: DATA_WIDTH]);
        if (val == 40) begin  // 5 * 8 = 40
            $display("  PASS: SUM([5,5,5,5,5,5,5,5]) = 40");
        end else begin
            $display("  FAIL: SUM = %0d (expected 40)", val);
            errors = errors + 1;
        end

        issue_cmd(make_cmd(VOP_MAX, 5'd12, 5'd10, 5'd10));
        val = $signed(dut.vrf[12][0*DATA_WIDTH +: DATA_WIDTH]);
        if (val == 5) begin
            $display("  PASS: MAX([5,5,...]) = 5");
        end else begin
            $display("  FAIL: MAX = %0d (expected 5)", val);
            errors = errors + 1;
        end

        issue_cmd(make_cmd(VOP_MIN, 5'd13, 5'd10, 5'd10));
        val = $signed(dut.vrf[13][0*DATA_WIDTH +: DATA_WIDTH]);
        if (val == 5) begin
            $display("  PASS: MIN([5,5,...]) = 5");
        end else begin
            $display("  FAIL: MIN = %0d (expected 5)", val);
            errors = errors + 1;
        end

        //======================================================================
        // Summary
        //======================================================================
        $display("");
        $display("╔══════════════════════════════════════════════════════════════╗");
        $display("║                    TEST SUMMARY                              ║");
        $display("╠══════════════════════════════════════════════════════════════╣");
        if (errors == 0) begin
            $display("║   PASSED: All VPU reduction tests (SUM, MAX, MIN)          ║");
            $display("╚══════════════════════════════════════════════════════════════╝");
            $display(">>> VPU REDUCE TEST PASSED! <<<");
        end else begin
            $display("║   FAILED: %0d errors                                        ║", errors);
            $display("╚══════════════════════════════════════════════════════════════╝");
            $display(">>> VPU REDUCE TEST FAILED <<<");
        end

        #(CLK * 10);
        $finish;
    end

    initial begin
        #(CLK * 10000);
        $display("TIMEOUT!");
        $finish;
    end

endmodule

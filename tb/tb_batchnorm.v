`timescale 1ns / 1ps
//==============================================================================
// BatchNorm Test: Fused y = scale * x + bias
//
// In inference mode, BatchNorm is fused into:
//   y = γ * (x - μ) / σ + β
//     = (γ/σ) * x + (β - γμ/σ)
//     = scale * x + bias
//
// This is a simple VPU MUL + VPU ADD operation.
//
// ONNX op: BatchNormalization (inference mode)
//==============================================================================

module tb_batchnorm;

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
    localparam VOP_ADD  = 8'h01;
    localparam VOP_MUL  = 8'h03;

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
        $display("║       BatchNorm Test: y = scale * x + bias                   ║");
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
        // Test 1: Simple BatchNorm
        // x = [10, 20, 30, 40, 50, 60, 70, 80]
        // scale = 2 (per-channel, but broadcast here)
        // bias = 5
        // y = 2*x + 5 = [25, 45, 65, 85, 105, 125, 145, 165]
        //======================================================================
        $display("[TEST 1] BatchNorm: y = 2*x + 5");

        // V0 = x
        for (i = 0; i < LANES; i = i + 1)
            dut.vrf[0][i*DATA_WIDTH +: DATA_WIDTH] = (i + 1) * 10;

        // V1 = scale (broadcast 2)
        for (i = 0; i < LANES; i = i + 1)
            dut.vrf[1][i*DATA_WIDTH +: DATA_WIDTH] = 16'd2;

        // V2 = bias (broadcast 5)
        for (i = 0; i < LANES; i = i + 1)
            dut.vrf[2][i*DATA_WIDTH +: DATA_WIDTH] = 16'd5;

        // V3 = V0 * V1 (scale * x)
        issue_cmd(make_cmd(VOP_MUL, 5'd3, 5'd0, 5'd1));

        // V4 = V3 + V2 (scaled + bias)
        issue_cmd(make_cmd(VOP_ADD, 5'd4, 5'd3, 5'd2));

        // Verify: y[i] = 2 * (i+1)*10 + 5 = 20*(i+1) + 5
        for (i = 0; i < LANES; i = i + 1) begin
            val = $signed(dut.vrf[4][i*DATA_WIDTH +: DATA_WIDTH]);
            expected = 20 * (i + 1) + 5;
            if (val == expected) begin
                $display("  PASS: y[%0d] = %0d (2*%0d + 5)", i, val, (i+1)*10);
            end else begin
                $display("  FAIL: y[%0d] = %0d (expected %0d)", i, val, expected);
                errors = errors + 1;
            end
        end

        //======================================================================
        // Test 2: Per-channel scale (different scale per lane)
        // x = [100, 100, 100, 100, 100, 100, 100, 100]
        // scale = [1, 2, 3, 4, 5, 6, 7, 8]
        // bias = 0
        // y = [100, 200, 300, 400, 500, 600, 700, 800]
        //======================================================================
        $display("");
        $display("[TEST 2] Per-channel scale");

        // V5 = x (all 100)
        for (i = 0; i < LANES; i = i + 1)
            dut.vrf[5][i*DATA_WIDTH +: DATA_WIDTH] = 16'd100;

        // V6 = per-channel scale [1,2,3,4,5,6,7,8]
        for (i = 0; i < LANES; i = i + 1)
            dut.vrf[6][i*DATA_WIDTH +: DATA_WIDTH] = i + 1;

        // V7 = bias (all 0)
        for (i = 0; i < LANES; i = i + 1)
            dut.vrf[7][i*DATA_WIDTH +: DATA_WIDTH] = 16'd0;

        // V8 = V5 * V6
        issue_cmd(make_cmd(VOP_MUL, 5'd8, 5'd5, 5'd6));

        // V9 = V8 + V7 (add zero bias - tests the full BN path)
        issue_cmd(make_cmd(VOP_ADD, 5'd9, 5'd8, 5'd7));

        // Verify
        for (i = 0; i < LANES; i = i + 1) begin
            val = $signed(dut.vrf[9][i*DATA_WIDTH +: DATA_WIDTH]);
            expected = 100 * (i + 1);
            if (val == expected) begin
                if (i == 0 || i == 7)  // Print first and last
                    $display("  PASS: y[%0d] = %0d (100 * %0d)", i, val, i+1);
            end else begin
                $display("  FAIL: y[%0d] = %0d (expected %0d)", i, val, expected);
                errors = errors + 1;
            end
        end
        if (errors == 0)
            $display("  PASS: All 8 per-channel scales correct");

        //======================================================================
        // Test 3: Negative scale (for channels that need inversion)
        // x = [10, 20, 30, 40]
        // scale = [-1, 1, -1, 1]
        // bias = [100, 0, 100, 0]
        // y = [-10+100, 20+0, -30+100, 40+0] = [90, 20, 70, 40]
        //======================================================================
        $display("");
        $display("[TEST 3] Negative scales");

        dut.vrf[10][0*DATA_WIDTH +: DATA_WIDTH] = 16'd10;
        dut.vrf[10][1*DATA_WIDTH +: DATA_WIDTH] = 16'd20;
        dut.vrf[10][2*DATA_WIDTH +: DATA_WIDTH] = 16'd30;
        dut.vrf[10][3*DATA_WIDTH +: DATA_WIDTH] = 16'd40;

        dut.vrf[11][0*DATA_WIDTH +: DATA_WIDTH] = -16'd1;
        dut.vrf[11][1*DATA_WIDTH +: DATA_WIDTH] = 16'd1;
        dut.vrf[11][2*DATA_WIDTH +: DATA_WIDTH] = -16'd1;
        dut.vrf[11][3*DATA_WIDTH +: DATA_WIDTH] = 16'd1;

        dut.vrf[12][0*DATA_WIDTH +: DATA_WIDTH] = 16'd100;
        dut.vrf[12][1*DATA_WIDTH +: DATA_WIDTH] = 16'd0;
        dut.vrf[12][2*DATA_WIDTH +: DATA_WIDTH] = 16'd100;
        dut.vrf[12][3*DATA_WIDTH +: DATA_WIDTH] = 16'd0;

        issue_cmd(make_cmd(VOP_MUL, 5'd13, 5'd10, 5'd11));
        issue_cmd(make_cmd(VOP_ADD, 5'd14, 5'd13, 5'd12));

        // Expected: [90, 20, 70, 40]
        val = $signed(dut.vrf[14][0*DATA_WIDTH +: DATA_WIDTH]);
        if (val == 90) $display("  PASS: y[0] = 90 (-1*10 + 100)");
        else begin $display("  FAIL: y[0] = %0d (expected 90)", val); errors = errors + 1; end

        val = $signed(dut.vrf[14][1*DATA_WIDTH +: DATA_WIDTH]);
        if (val == 20) $display("  PASS: y[1] = 20 (1*20 + 0)");
        else begin $display("  FAIL: y[1] = %0d (expected 20)", val); errors = errors + 1; end

        val = $signed(dut.vrf[14][2*DATA_WIDTH +: DATA_WIDTH]);
        if (val == 70) $display("  PASS: y[2] = 70 (-1*30 + 100)");
        else begin $display("  FAIL: y[2] = %0d (expected 70)", val); errors = errors + 1; end

        val = $signed(dut.vrf[14][3*DATA_WIDTH +: DATA_WIDTH]);
        if (val == 40) $display("  PASS: y[3] = 40 (1*40 + 0)");
        else begin $display("  FAIL: y[3] = %0d (expected 40)", val); errors = errors + 1; end

        //======================================================================
        // Summary
        //======================================================================
        $display("");
        $display("╔══════════════════════════════════════════════════════════════╗");
        $display("║                    TEST SUMMARY                              ║");
        $display("╠══════════════════════════════════════════════════════════════╣");
        if (errors == 0) begin
            $display("║   PASSED: BatchNorm (y = scale*x + bias)                   ║");
            $display("╚══════════════════════════════════════════════════════════════╝");
            $display(">>> BATCHNORM TEST PASSED! <<<");
        end else begin
            $display("║   FAILED: %0d errors                                        ║", errors);
            $display("╚══════════════════════════════════════════════════════════════╝");
            $display(">>> BATCHNORM TEST FAILED <<<");
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

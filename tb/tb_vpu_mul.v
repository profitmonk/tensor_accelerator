`timescale 1ns / 1ps
//==============================================================================
// VPU MUL Test: Element-wise multiplication
//
// Tests VPU multiply operation used for:
//   - Scaling (BatchNorm, LayerNorm)
//   - Attention weights
//   - Quantization scale factors
//
// ONNX ops: Mul, BatchNormalization (fused)
//==============================================================================

module tb_vpu_mul;

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
    localparam VOP_MUL  = 8'h03;
    localparam VOP_ZERO = 8'h34;

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

    integer i, errors;
    reg signed [DATA_WIDTH-1:0] val;

    initial begin
        $display("");
        $display("╔══════════════════════════════════════════════════════════════╗");
        $display("║           VPU MUL Test: Element-wise Multiply                ║");
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
        // Test 1: Simple multiply [2,3,4,5,6,7,8,9] × [1,2,3,4,5,6,7,8]
        //======================================================================
        $display("[TEST 1] Simple multiply");
        
        // Load V0 = [2,3,4,5,6,7,8,9]
        for (i = 0; i < LANES; i = i + 1)
            dut.vrf[0][i*DATA_WIDTH +: DATA_WIDTH] = i + 2;
        
        // Load V1 = [1,2,3,4,5,6,7,8]
        for (i = 0; i < LANES; i = i + 1)
            dut.vrf[1][i*DATA_WIDTH +: DATA_WIDTH] = i + 1;

        // V2 = V0 * V1
        issue_cmd(make_cmd(VOP_MUL, 5'd2, 5'd0, 5'd1));

        // Check results: (i+2) * (i+1) for i=0..7
        // i=0: 2*1=2, i=1: 3*2=6, i=2: 4*3=12, i=3: 5*4=20
        // i=4: 6*5=30, i=5: 7*6=42, i=6: 8*7=56, i=7: 9*8=72
        
        val = $signed(dut.vrf[2][0*DATA_WIDTH +: DATA_WIDTH]);
        if (val == 2) begin
            $display("  PASS: 2 × 1 = 2");
        end else begin
            $display("  FAIL: 2 × 1 = %0d (expected 2)", val);
            errors = errors + 1;
        end

        val = $signed(dut.vrf[2][1*DATA_WIDTH +: DATA_WIDTH]);
        if (val == 6) begin
            $display("  PASS: 3 × 2 = 6");
        end else begin
            $display("  FAIL: 3 × 2 = %0d (expected 6)", val);
            errors = errors + 1;
        end

        val = $signed(dut.vrf[2][2*DATA_WIDTH +: DATA_WIDTH]);
        if (val == 12) begin
            $display("  PASS: 4 × 3 = 12");
        end else begin
            $display("  FAIL: 4 × 3 = %0d (expected 12)", val);
            errors = errors + 1;
        end

        val = $signed(dut.vrf[2][3*DATA_WIDTH +: DATA_WIDTH]);
        if (val == 20) begin
            $display("  PASS: 5 × 4 = 20");
        end else begin
            $display("  FAIL: 5 × 4 = %0d (expected 20)", val);
            errors = errors + 1;
        end

        //======================================================================
        // Test 2: Signed multiply [-3, 4, -5, 6, -7, 8, -9, 10] × [2, -2, 2, -2, ...]
        //======================================================================
        $display("");
        $display("[TEST 2] Signed multiply");

        // V3 = alternating signs
        dut.vrf[3][0*DATA_WIDTH +: DATA_WIDTH] = -16'd3;
        dut.vrf[3][1*DATA_WIDTH +: DATA_WIDTH] = 16'd4;
        dut.vrf[3][2*DATA_WIDTH +: DATA_WIDTH] = -16'd5;
        dut.vrf[3][3*DATA_WIDTH +: DATA_WIDTH] = 16'd6;
        dut.vrf[3][4*DATA_WIDTH +: DATA_WIDTH] = -16'd7;
        dut.vrf[3][5*DATA_WIDTH +: DATA_WIDTH] = 16'd8;
        dut.vrf[3][6*DATA_WIDTH +: DATA_WIDTH] = -16'd9;
        dut.vrf[3][7*DATA_WIDTH +: DATA_WIDTH] = 16'd10;

        // V4 = [2, -2, 2, -2, 2, -2, 2, -2]
        dut.vrf[4][0*DATA_WIDTH +: DATA_WIDTH] = 16'd2;
        dut.vrf[4][1*DATA_WIDTH +: DATA_WIDTH] = -16'd2;
        dut.vrf[4][2*DATA_WIDTH +: DATA_WIDTH] = 16'd2;
        dut.vrf[4][3*DATA_WIDTH +: DATA_WIDTH] = -16'd2;
        dut.vrf[4][4*DATA_WIDTH +: DATA_WIDTH] = 16'd2;
        dut.vrf[4][5*DATA_WIDTH +: DATA_WIDTH] = -16'd2;
        dut.vrf[4][6*DATA_WIDTH +: DATA_WIDTH] = 16'd2;
        dut.vrf[4][7*DATA_WIDTH +: DATA_WIDTH] = -16'd2;

        // V5 = V3 * V4
        issue_cmd(make_cmd(VOP_MUL, 5'd5, 5'd3, 5'd4));

        // Expected: [-6, -8, -10, -12, -14, -16, -18, -20]
        val = $signed(dut.vrf[5][0*DATA_WIDTH +: DATA_WIDTH]);
        if (val == -6) begin
            $display("  PASS: -3 × 2 = -6");
        end else begin
            $display("  FAIL: -3 × 2 = %0d (expected -6)", val);
            errors = errors + 1;
        end

        val = $signed(dut.vrf[5][1*DATA_WIDTH +: DATA_WIDTH]);
        if (val == -8) begin
            $display("  PASS: 4 × -2 = -8");
        end else begin
            $display("  FAIL: 4 × -2 = %0d (expected -8)", val);
            errors = errors + 1;
        end

        val = $signed(dut.vrf[5][2*DATA_WIDTH +: DATA_WIDTH]);
        if (val == -10) begin
            $display("  PASS: -5 × 2 = -10");
        end else begin
            $display("  FAIL: -5 × 2 = %0d (expected -10)", val);
            errors = errors + 1;
        end

        val = $signed(dut.vrf[5][3*DATA_WIDTH +: DATA_WIDTH]);
        if (val == -12) begin
            $display("  PASS: 6 × -2 = -12");
        end else begin
            $display("  FAIL: 6 × -2 = %0d (expected -12)", val);
            errors = errors + 1;
        end

        //======================================================================
        // Test 3: Scale factor (BatchNorm-like): x × scale
        //======================================================================
        $display("");
        $display("[TEST 3] Scaling (BatchNorm-like)");

        // x = [100, 200, 300, 400, 500, 600, 700, 800]
        for (i = 0; i < LANES; i = i + 1)
            dut.vrf[6][i*DATA_WIDTH +: DATA_WIDTH] = (i + 1) * 100;

        // scale = [3, 3, 3, 3, 3, 3, 3, 3] (broadcast)
        for (i = 0; i < LANES; i = i + 1)
            dut.vrf[7][i*DATA_WIDTH +: DATA_WIDTH] = 3;

        // V8 = V6 * V7
        issue_cmd(make_cmd(VOP_MUL, 5'd8, 5'd6, 5'd7));

        val = $signed(dut.vrf[8][0*DATA_WIDTH +: DATA_WIDTH]);
        if (val == 300) begin
            $display("  PASS: 100 × 3 = 300");
        end else begin
            $display("  FAIL: 100 × 3 = %0d (expected 300)", val);
            errors = errors + 1;
        end

        val = $signed(dut.vrf[8][3*DATA_WIDTH +: DATA_WIDTH]);
        if (val == 1200) begin
            $display("  PASS: 400 × 3 = 1200");
        end else begin
            $display("  FAIL: 400 × 3 = %0d (expected 1200)", val);
            errors = errors + 1;
        end

        //======================================================================
        // Test 4: Zero multiply
        //======================================================================
        $display("");
        $display("[TEST 4] Zero multiply");

        // V9 = all zeros
        for (i = 0; i < LANES; i = i + 1)
            dut.vrf[9][i*DATA_WIDTH +: DATA_WIDTH] = 0;

        // V10 = V6 * V9 (anything × 0 = 0)
        issue_cmd(make_cmd(VOP_MUL, 5'd10, 5'd6, 5'd9));

        if (dut.vrf[10] == 0) begin
            $display("  PASS: x × 0 = 0");
        end else begin
            $display("  FAIL: x × 0 != 0");
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
            $display("║   PASSED: All VPU MUL tests                                ║");
            $display("╚══════════════════════════════════════════════════════════════╝");
            $display(">>> VPU MUL TEST PASSED! <<<");
        end else begin
            $display("║   FAILED: %0d errors                                        ║", errors);
            $display("╚══════════════════════════════════════════════════════════════╝");
            $display(">>> VPU MUL TEST FAILED <<<");
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

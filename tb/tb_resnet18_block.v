`timescale 1ns / 1ps
//==============================================================================
// ResNet-18 Basic Block Test (VPU-focused)
//
// Tests the key ResNet operations:
//   x → BN1 → ReLU → BN2 → (+x) → ReLU → y
//
// Convolutions are tested separately (tb_e2e_conv2d.v).
// This test focuses on the VPU operations that implement:
//   - BatchNorm (fused scale*x + bias)
//   - Residual (skip) connection
//   - ReLU activations
//
// These patterns are what differentiate ResNet from earlier CNNs.
//==============================================================================

module tb_resnet18_block;

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
    localparam VOP_RELU = 8'h10;

    // Command builder
    function [127:0] make_cmd;
        input [7:0] subop;
        input [4:0] vd, vs1, vs2;
    begin
        make_cmd = 0;
        make_cmd[127:120] = 8'h02;
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
        $display("║              ResNet-18 Basic Block Test                      ║");
        $display("║      x → BN1 → ReLU → BN2 → (+x) → ReLU → y                 ║");
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
        // Test: Complete ResNet Basic Block
        //
        // Input X = [1, 2, 3, 4, 5, 6, 7, 8]
        // 
        // In real ResNet: x → Conv1 → BN1 → ReLU → Conv2 → BN2 → (+x) → ReLU → y
        // We simulate Conv outputs as identity (y = x) for verification.
        //
        // BN1: scale=2, bias=0 → y = 2x
        // BN2: scale=1, bias=1 → y = x + 1
        //
        // Expected flow:
        //   After Conv1 (identity): [1, 2, 3, 4, 5, 6, 7, 8]
        //   After BN1 (2x):         [2, 4, 6, 8, 10, 12, 14, 16]
        //   After ReLU1:            [2, 4, 6, 8, 10, 12, 14, 16] (all positive)
        //   After Conv2 (identity): [2, 4, 6, 8, 10, 12, 14, 16]
        //   After BN2 (x+1):        [3, 5, 7, 9, 11, 13, 15, 17]
        //   After +X residual:      [4, 7, 10, 13, 16, 19, 22, 25]
        //   After ReLU2:            [4, 7, 10, 13, 16, 19, 22, 25]
        //======================================================================

        $display("[TEST] ResNet Basic Block: BN1 → ReLU → BN2 → (+skip) → ReLU");
        $display("");
        $display("  Input X = [1, 2, 3, 4, 5, 6, 7, 8]");
        $display("  BN1: scale=2, bias=0 → 2*x");
        $display("  BN2: scale=1, bias=1 → x+1");
        $display("  Expected output: 2*x + 1 + x = 3*x + 1");
        $display("  Expected[0] = 3*1 + 1 = 4");
        $display("");

        // V0 = Input X = [1, 2, 3, 4, 5, 6, 7, 8]
        for (i = 0; i < LANES; i = i + 1)
            dut.vrf[0][i*DATA_WIDTH +: DATA_WIDTH] = i + 1;

        // V1 = BN1 scale = [2, 2, 2, 2, 2, 2, 2, 2]
        for (i = 0; i < LANES; i = i + 1)
            dut.vrf[1][i*DATA_WIDTH +: DATA_WIDTH] = 16'd2;

        // V2 = BN1 bias = [0, 0, 0, 0, 0, 0, 0, 0]
        for (i = 0; i < LANES; i = i + 1)
            dut.vrf[2][i*DATA_WIDTH +: DATA_WIDTH] = 16'd0;

        // V10 = BN2 scale = [1, 1, 1, 1, 1, 1, 1, 1]
        for (i = 0; i < LANES; i = i + 1)
            dut.vrf[10][i*DATA_WIDTH +: DATA_WIDTH] = 16'd1;

        // V11 = BN2 bias = [1, 1, 1, 1, 1, 1, 1, 1]
        for (i = 0; i < LANES; i = i + 1)
            dut.vrf[11][i*DATA_WIDTH +: DATA_WIDTH] = 16'd1;

        //======================================================================
        // Stage 1: BN1 = scale * x + bias = 2*x + 0
        //======================================================================
        $display("[Stage 1] BN1: V3 = 2 * V0 + 0");
        
        // V3 = V0 * V1 (scale)
        issue_cmd(make_cmd(VOP_MUL, 5'd3, 5'd0, 5'd1));
        // V4 = V3 + V2 (add bias, which is 0)
        issue_cmd(make_cmd(VOP_ADD, 5'd4, 5'd3, 5'd2));

        $display("  BN1 output V4[0:3] = %d, %d, %d, %d",
            $signed(dut.vrf[4][0*DATA_WIDTH +: DATA_WIDTH]),
            $signed(dut.vrf[4][1*DATA_WIDTH +: DATA_WIDTH]),
            $signed(dut.vrf[4][2*DATA_WIDTH +: DATA_WIDTH]),
            $signed(dut.vrf[4][3*DATA_WIDTH +: DATA_WIDTH]));

        val = $signed(dut.vrf[4][0*DATA_WIDTH +: DATA_WIDTH]);
        if (val == 2) $display("  PASS: BN1[0] = 2 (2*1)");
        else begin $display("  FAIL: BN1[0] = %d (expected 2)", val); errors = errors + 1; end

        //======================================================================
        // Stage 2: ReLU1
        //======================================================================
        $display("");
        $display("[Stage 2] ReLU1: V5 = ReLU(V4)");
        
        issue_cmd(make_cmd(VOP_RELU, 5'd5, 5'd4, 5'd0));

        $display("  ReLU1 output V5[0:3] = %d, %d, %d, %d",
            $signed(dut.vrf[5][0*DATA_WIDTH +: DATA_WIDTH]),
            $signed(dut.vrf[5][1*DATA_WIDTH +: DATA_WIDTH]),
            $signed(dut.vrf[5][2*DATA_WIDTH +: DATA_WIDTH]),
            $signed(dut.vrf[5][3*DATA_WIDTH +: DATA_WIDTH]));

        //======================================================================
        // Stage 3: BN2 = scale * x + bias = 1*x + 1
        //======================================================================
        $display("");
        $display("[Stage 3] BN2: V7 = 1 * V5 + 1");
        
        // V6 = V5 * V10 (scale by 1 = identity)
        issue_cmd(make_cmd(VOP_MUL, 5'd6, 5'd5, 5'd10));
        // V7 = V6 + V11 (add bias = 1)
        issue_cmd(make_cmd(VOP_ADD, 5'd7, 5'd6, 5'd11));

        $display("  BN2 output V7[0:3] = %d, %d, %d, %d",
            $signed(dut.vrf[7][0*DATA_WIDTH +: DATA_WIDTH]),
            $signed(dut.vrf[7][1*DATA_WIDTH +: DATA_WIDTH]),
            $signed(dut.vrf[7][2*DATA_WIDTH +: DATA_WIDTH]),
            $signed(dut.vrf[7][3*DATA_WIDTH +: DATA_WIDTH]));

        val = $signed(dut.vrf[7][0*DATA_WIDTH +: DATA_WIDTH]);
        if (val == 3) $display("  PASS: BN2[0] = 3 (2+1)");
        else begin $display("  FAIL: BN2[0] = %d (expected 3)", val); errors = errors + 1; end

        //======================================================================
        // Stage 4: Residual Add = BN2_output + original_input
        //======================================================================
        $display("");
        $display("[Stage 4] Residual: V8 = V7 + V0 (skip connection!)");
        
        // V8 = V7 + V0 (BN2 output + original input)
        issue_cmd(make_cmd(VOP_ADD, 5'd8, 5'd7, 5'd0));

        $display("  Residual output V8[0:3] = %d, %d, %d, %d",
            $signed(dut.vrf[8][0*DATA_WIDTH +: DATA_WIDTH]),
            $signed(dut.vrf[8][1*DATA_WIDTH +: DATA_WIDTH]),
            $signed(dut.vrf[8][2*DATA_WIDTH +: DATA_WIDTH]),
            $signed(dut.vrf[8][3*DATA_WIDTH +: DATA_WIDTH]));

        // Expected: (2*x + 1) + x = 3*x + 1
        // For x=1: 3*1 + 1 = 4
        val = $signed(dut.vrf[8][0*DATA_WIDTH +: DATA_WIDTH]);
        expected = 4;  // 3 + 1
        if (val == expected) begin
            $display("  PASS: Residual[0] = 4 (BN2 + skip = 3 + 1)");
        end else begin
            $display("  FAIL: Residual[0] = %d (expected %d)", val, expected);
            errors = errors + 1;
        end

        // For x=2: 3*2 + 1 = 7
        val = $signed(dut.vrf[8][1*DATA_WIDTH +: DATA_WIDTH]);
        expected = 7;  // 5 + 2
        if (val == expected) begin
            $display("  PASS: Residual[1] = 7 (BN2 + skip = 5 + 2)");
        end else begin
            $display("  FAIL: Residual[1] = %d (expected %d)", val, expected);
            errors = errors + 1;
        end

        //======================================================================
        // Stage 5: Final ReLU
        //======================================================================
        $display("");
        $display("[Stage 5] ReLU2: V9 = ReLU(V8)");
        
        issue_cmd(make_cmd(VOP_RELU, 5'd9, 5'd8, 5'd0));

        $display("  Final output V9[0:7]:");
        $display("    [%d, %d, %d, %d, %d, %d, %d, %d]",
            $signed(dut.vrf[9][0*DATA_WIDTH +: DATA_WIDTH]),
            $signed(dut.vrf[9][1*DATA_WIDTH +: DATA_WIDTH]),
            $signed(dut.vrf[9][2*DATA_WIDTH +: DATA_WIDTH]),
            $signed(dut.vrf[9][3*DATA_WIDTH +: DATA_WIDTH]),
            $signed(dut.vrf[9][4*DATA_WIDTH +: DATA_WIDTH]),
            $signed(dut.vrf[9][5*DATA_WIDTH +: DATA_WIDTH]),
            $signed(dut.vrf[9][6*DATA_WIDTH +: DATA_WIDTH]),
            $signed(dut.vrf[9][7*DATA_WIDTH +: DATA_WIDTH]));

        // Verify all 8 outputs: 3*x + 1 for x = 1..8
        // Expected: [4, 7, 10, 13, 16, 19, 22, 25]
        $display("");
        $display("[VERIFY] Checking all 8 outputs: expected 3*x + 1");
        
        for (i = 0; i < LANES; i = i + 1) begin
            val = $signed(dut.vrf[9][i*DATA_WIDTH +: DATA_WIDTH]);
            expected = 3 * (i + 1) + 1;
            if (val == expected) begin
                if (i == 0 || i == 7)
                    $display("  PASS: output[%d] = %d (3*%d + 1)", i, val, i+1);
            end else begin
                $display("  FAIL: output[%d] = %d (expected %d)", i, val, expected);
                errors = errors + 1;
            end
        end

        if (errors == 0)
            $display("  PASS: All 8 outputs correct!");

        //======================================================================
        // Summary
        //======================================================================
        $display("");
        $display("╔══════════════════════════════════════════════════════════════╗");
        $display("║                    TEST SUMMARY                              ║");
        $display("╠══════════════════════════════════════════════════════════════╣");
        if (errors == 0) begin
            $display("║   PASSED: ResNet-18 Basic Block                            ║");
            $display("║   Verified stages:                                         ║");
            $display("║     1. BatchNorm1 (scale*x + bias)                         ║");
            $display("║     2. ReLU1                                               ║");
            $display("║     3. BatchNorm2 (scale*x + bias)                         ║");
            $display("║     4. Residual Addition (+skip connection)                ║");
            $display("║     5. ReLU2                                               ║");
            $display("╚══════════════════════════════════════════════════════════════╝");
            $display(">>> RESNET-18 BLOCK TEST PASSED! <<<");
        end else begin
            $display("║   FAILED: %0d errors                                        ║", errors);
            $display("╚══════════════════════════════════════════════════════════════╝");
            $display(">>> RESNET-18 BLOCK TEST FAILED <<<");
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

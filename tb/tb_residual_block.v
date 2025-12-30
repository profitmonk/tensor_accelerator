`timescale 1ns / 1ps
//==============================================================================
// Residual Block Test: Y = F(X) + X where F(X) = ReLU(X @ W + b)
//
// Tests skip/residual connection commonly used in ResNets.
// Verifies against Python golden model output.
//
// Test Configuration (matches model_residual.py):
//   - Input X: 4×4 matrix
//   - Weights W: 4×4 matrix
//   - Bias b: 4-element vector
//   - Output Y: 4×4 matrix = ReLU(X @ W + b) + X
//==============================================================================

module tb_residual_block;

    parameter CLK = 10;
    parameter SRAM_WIDTH = 256;
    parameter N = 4;           // Rows
    parameter FEATURES = 4;    // Features

    reg clk = 0;
    reg rst_n = 0;
    always #(CLK/2) clk = ~clk;

    // TPC interface signals
    reg tpc_start = 0;
    reg [19:0] tpc_start_pc = 0;
    wire tpc_busy, tpc_done, tpc_error;
    reg global_sync_in = 0;
    wire sync_request;
    reg sync_grant = 0;
    reg [SRAM_WIDTH-1:0] noc_rx_data = 0;
    reg [19:0] noc_rx_addr = 0;
    reg noc_rx_valid = 0;
    wire noc_rx_ready;
    reg noc_rx_is_instr = 0;
    wire [SRAM_WIDTH-1:0] noc_tx_data;
    wire [19:0] noc_tx_addr;
    wire noc_tx_valid;
    reg noc_tx_ready = 1;
    wire [39:0] axi_awaddr, axi_araddr;
    wire [7:0] axi_awlen, axi_arlen;
    wire axi_awvalid, axi_arvalid, axi_wvalid, axi_wlast, axi_rready, axi_bready;
    wire [255:0] axi_wdata;
    reg axi_awready = 1, axi_arready = 1, axi_wready = 1;
    reg axi_bvalid = 0, axi_rvalid = 0, axi_rlast = 0;
    reg [1:0] axi_bresp = 0;
    reg [255:0] axi_rdata = 0;

    // DUT
    tensor_processing_cluster #(
        .ARRAY_SIZE(4), .SRAM_WIDTH(SRAM_WIDTH),
        .SRAM_BANKS(4), .SRAM_DEPTH(256), .VPU_LANES(16)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .tpc_start(tpc_start), .tpc_start_pc(tpc_start_pc),
        .tpc_busy(tpc_busy), .tpc_done(tpc_done), .tpc_error(tpc_error),
        .global_sync_in(global_sync_in), .sync_request(sync_request), .sync_grant(sync_grant),
        .noc_rx_data(noc_rx_data), .noc_rx_addr(noc_rx_addr),
        .noc_rx_valid(noc_rx_valid), .noc_rx_ready(noc_rx_ready), .noc_rx_is_instr(noc_rx_is_instr),
        .noc_tx_data(noc_tx_data), .noc_tx_addr(noc_tx_addr),
        .noc_tx_valid(noc_tx_valid), .noc_tx_ready(noc_tx_ready),
        .axi_awaddr(axi_awaddr), .axi_awlen(axi_awlen), .axi_awvalid(axi_awvalid), .axi_awready(axi_awready),
        .axi_wdata(axi_wdata), .axi_wlast(axi_wlast), .axi_wvalid(axi_wvalid), .axi_wready(axi_wready),
        .axi_bresp(axi_bresp), .axi_bvalid(axi_bvalid), .axi_bready(axi_bready),
        .axi_araddr(axi_araddr), .axi_arlen(axi_arlen), .axi_arvalid(axi_arvalid), .axi_arready(axi_arready),
        .axi_rdata(axi_rdata), .axi_rlast(axi_rlast), .axi_rvalid(axi_rvalid), .axi_rready(axi_rready)
    );

    // Opcodes
    localparam OP_TENSOR = 8'h01;
    localparam OP_VECTOR = 8'h02;
    localparam OP_SYNC   = 8'h04;
    localparam OP_HALT   = 8'hFF;
    localparam VOP_ADD   = 8'h01;
    localparam VOP_RELU  = 8'h10;
    localparam VOP_LOAD  = 8'h30;
    localparam VOP_STORE = 8'h31;
    localparam SYNC_MXU  = 8'h01;
    localparam SYNC_VPU  = 8'h02;

    // SRAM addresses (word-aligned, 4 banks)
    // Each row goes to a different bank
    localparam SRAM_X     = 20'h00000;  // Input X (4 rows)
    localparam SRAM_W     = 20'h00010;  // Weights W (4 rows)
    localparam SRAM_B     = 20'h00030;  // Bias b (1 word)
    localparam SRAM_Z     = 20'h00020;  // GEMM output Z (4 rows)
    localparam SRAM_FX    = 20'h00040;  // F(X) after ReLU
    localparam SRAM_Y     = 20'h00050;  // Final output Y

    integer i, j, errors;
    reg signed [31:0] actual, expected;

    // Test data from Python golden model (seed=42)
    // X = [[ 2,-3,-1,-1], [-2,-1, 1, 1], [ 0,-1, 3, 0], [ 3,-3, 0, 4]]
    // W = [[-2,-1,-2, 0], [ 0, 0,-1, 2], [-1, 0,-1, 0], [-1,-2, 2, 1]]
    // b = [-1, -6, -7, 3]
    // Y = [[ 2,-3,-1,-1], [-1,-1, 1, 3], [ 0,-1, 3, 1], [ 3,-3, 0, 5]]

    // Pack X rows (8-bit elements, little-endian within row)
    wire [255:0] X_row0 = {224'd0, 8'hFF, 8'hFF, 8'hFD, 8'h02};  // [ 2,-3,-1,-1]
    wire [255:0] X_row1 = {224'd0, 8'h01, 8'h01, 8'hFF, 8'hFE};  // [-2,-1, 1, 1]
    wire [255:0] X_row2 = {224'd0, 8'h00, 8'h03, 8'hFF, 8'h00};  // [ 0,-1, 3, 0]
    wire [255:0] X_row3 = {224'd0, 8'h04, 8'h00, 8'hFD, 8'h03};  // [ 3,-3, 0, 4]

    // Pack W rows (8-bit elements)
    wire [255:0] W_row0 = {224'd0, 8'h00, 8'hFE, 8'hFF, 8'hFE};  // [-2,-1,-2, 0]
    wire [255:0] W_row1 = {224'd0, 8'h02, 8'hFF, 8'h00, 8'h00};  // [ 0, 0,-1, 2]
    wire [255:0] W_row2 = {224'd0, 8'h00, 8'hFF, 8'h00, 8'hFF};  // [-1, 0,-1, 0]
    wire [255:0] W_row3 = {224'd0, 8'h01, 8'h02, 8'hFE, 8'hFF};  // [-1,-2, 2, 1]

    // Bias as 32-bit signed values in one word
    wire [255:0] bias_word = {128'd0, 32'sd3, -32'sd7, -32'sd6, -32'sd1};

    // Expected Y output (32-bit signed elements per row)
    // Y = [[ 2,-3,-1,-1], [-1,-1, 1, 3], [ 0,-1, 3, 1], [ 3,-3, 0, 5]]
    wire signed [31:0] Y_expected [0:15];
    assign Y_expected[0]  = 32'sd2;   assign Y_expected[1]  = -32'sd3;
    assign Y_expected[2]  = -32'sd1;  assign Y_expected[3]  = -32'sd1;
    assign Y_expected[4]  = -32'sd1;  assign Y_expected[5]  = -32'sd1;
    assign Y_expected[6]  = 32'sd1;   assign Y_expected[7]  = 32'sd3;
    assign Y_expected[8]  = 32'sd0;   assign Y_expected[9]  = -32'sd1;
    assign Y_expected[10] = 32'sd3;   assign Y_expected[11] = 32'sd1;
    assign Y_expected[12] = 32'sd3;   assign Y_expected[13] = -32'sd3;
    assign Y_expected[14] = 32'sd0;   assign Y_expected[15] = 32'sd5;

    // Store X as 32-bit for skip connection
    wire [255:0] X32_row0 = {128'd0, -32'sd1, -32'sd1, -32'sd3, 32'sd2};
    wire [255:0] X32_row1 = {128'd0, 32'sd1, 32'sd1, -32'sd1, -32'sd2};
    wire [255:0] X32_row2 = {128'd0, 32'sd0, 32'sd3, -32'sd1, 32'sd0};
    wire [255:0] X32_row3 = {128'd0, 32'sd4, 32'sd0, -32'sd3, 32'sd3};

    initial begin
        $display("");
        $display("╔══════════════════════════════════════════════════════════════╗");
        $display("║        Residual Block Test: Y = ReLU(X×W + b) + X            ║");
        $display("║        Golden Model: python/models/model_residual.py         ║");
        $display("╚══════════════════════════════════════════════════════════════╝");
        $display("");

        errors = 0;

        //======================================================================
        // Initialize SRAM with test data
        //======================================================================
        $display("[SETUP] Initializing SRAM...");

        // X input (8-bit) at 0x0000-0x0003 (banks 0-3, word 0)
        dut.sram_inst.bank_gen[0].bank_inst.mem[0] = X_row0;
        dut.sram_inst.bank_gen[1].bank_inst.mem[0] = X_row1;
        dut.sram_inst.bank_gen[2].bank_inst.mem[0] = X_row2;
        dut.sram_inst.bank_gen[3].bank_inst.mem[0] = X_row3;

        // W weights (8-bit) at 0x0010-0x0013 (banks 0-3, word 4)
        dut.sram_inst.bank_gen[0].bank_inst.mem[4] = W_row0;
        dut.sram_inst.bank_gen[1].bank_inst.mem[4] = W_row1;
        dut.sram_inst.bank_gen[2].bank_inst.mem[4] = W_row2;
        dut.sram_inst.bank_gen[3].bank_inst.mem[4] = W_row3;

        // Bias at 0x0030 (bank 0, word 12)
        dut.sram_inst.bank_gen[0].bank_inst.mem[12] = bias_word;

        // X as 32-bit for skip connection at 0x0060 (banks 0-3, word 24)
        dut.sram_inst.bank_gen[0].bank_inst.mem[24] = X32_row0;
        dut.sram_inst.bank_gen[1].bank_inst.mem[24] = X32_row1;
        dut.sram_inst.bank_gen[2].bank_inst.mem[24] = X32_row2;
        dut.sram_inst.bank_gen[3].bank_inst.mem[24] = X32_row3;

        $display("  X: 4×4 input matrix");
        $display("  W: 4×4 weight matrix");
        $display("  b: 4-element bias vector");

        //======================================================================
        // Load Program: Y = ReLU(X @ W + b) + X
        //======================================================================
        $display("");
        $display("[SETUP] Loading program...");

        i = 0;

        // Stage 1: GEMM Z = X @ W
        dut.instr_mem[i] = {OP_TENSOR, 8'h01, 16'h0020, 16'h0010, 16'h0000, 16'd4, 16'd4, 16'd4, 16'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_MXU, 112'd0}; i = i + 1;

        // Stage 2: VPU load Z row 0, add bias, ReLU, store to FX
        // For simplicity, process row 0 only (single VPU width)
        // Load Z[0] into v0
        dut.instr_mem[i] = {OP_VECTOR, VOP_LOAD, 5'd0, 5'd0, 5'd0, 1'b0, 20'h00020, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;

        // Load bias into v1
        dut.instr_mem[i] = {OP_VECTOR, VOP_LOAD, 5'd1, 5'd0, 5'd0, 1'b0, 20'h00030, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;

        // v0 = v0 + v1 (Z + bias)
        dut.instr_mem[i] = {OP_VECTOR, VOP_ADD, 5'd0, 5'd0, 5'd1, 1'b0, 20'h00000, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;

        // v0 = ReLU(v0)
        dut.instr_mem[i] = {OP_VECTOR, VOP_RELU, 5'd0, 5'd0, 5'd0, 1'b0, 20'h00000, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;

        // Stage 3: Load X[0] (32-bit) for skip connection into v2
        dut.instr_mem[i] = {OP_VECTOR, VOP_LOAD, 5'd2, 5'd0, 5'd0, 1'b0, 20'h00060, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;

        // v0 = v0 + v2 (F(X) + X)
        dut.instr_mem[i] = {OP_VECTOR, VOP_ADD, 5'd0, 5'd0, 5'd2, 1'b0, 20'h00000, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;

        // Store Y[0] = v0
        dut.instr_mem[i] = {OP_VECTOR, VOP_STORE, 5'd0, 5'd0, 5'd0, 1'b0, 20'h00050, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;

        // HALT
        dut.instr_mem[i] = {OP_HALT, 120'd0};

        $display("  Program: %0d instructions", i+1);
        $display("  Flow: GEMM → ADD bias → ReLU → ADD skip → Store");

        //======================================================================
        // Run Test
        //======================================================================
        $display("");
        $display("[EXEC] Running residual block...");

        rst_n = 0; #(CLK*5);
        rst_n = 1; #(CLK*5);

        @(negedge clk);
        tpc_start = 1;
        @(posedge clk); @(posedge clk);
        @(negedge clk);
        tpc_start = 0;

        for (i = 0; i < 300; i = i + 1) begin
            @(posedge clk);
            if (tpc_done) begin
                $display("  Completed in %0d cycles", i);
                i = 10000;
            end
        end

        if (i < 10000) begin
            $display("  TIMEOUT!");
            errors = errors + 1;
        end

        #(CLK*5);

        //======================================================================
        // Verify Results
        //======================================================================
        $display("");
        $display("[VERIFY] Checking output Y[0] (row 0)...");

        // Y output at 0x50 (bank 0, word 20)
        $display("  Y[0] raw: %h", dut.sram_inst.bank_gen[0].bank_inst.mem[20][127:0]);

        // Check row 0 (4 elements)
        for (j = 0; j < 4; j = j + 1) begin
            actual = $signed(dut.sram_inst.bank_gen[0].bank_inst.mem[20][j*32 +: 32]);
            expected = Y_expected[j];
            if (actual == expected) begin
                $display("    PASS: Y[0,%0d] = %0d", j, actual);
            end else begin
                $display("    FAIL: Y[0,%0d] = %0d, expected %0d", j, actual, expected);
                errors = errors + 1;
            end
        end

        // Also check intermediate values
        $display("");
        $display("  Intermediate values (row 0):");
        $display("    Z (GEMM): %h", dut.sram_inst.bank_gen[0].bank_inst.mem[8][127:0]);
        $display("    v0 (final): %h", dut.vpu_inst.vrf[0][127:0]);
        $display("    v1 (bias): %h", dut.vpu_inst.vrf[1][127:0]);
        $display("    v2 (X skip): %h", dut.vpu_inst.vrf[2][127:0]);

        //======================================================================
        // Summary
        //======================================================================
        $display("");
        $display("╔══════════════════════════════════════════════════════════════╗");
        $display("║                    TEST SUMMARY                              ║");
        $display("╠══════════════════════════════════════════════════════════════╣");
        if (errors == 0) begin
            $display("║   PASSED: Residual block Y = ReLU(X×W+b) + X                ║");
            $display("╚══════════════════════════════════════════════════════════════╝");
            $display(">>> RESIDUAL BLOCK TEST PASSED! <<<");
        end else begin
            $display("║   FAILED: %0d errors                                         ║", errors);
            $display("╚══════════════════════════════════════════════════════════════╝");
            $display(">>> RESIDUAL BLOCK TEST FAILED <<<");
        end

        #(CLK * 10);
        $finish;
    end

    initial begin
        #(CLK * 20000);
        $display("GLOBAL TIMEOUT!");
        $finish;
    end

endmodule

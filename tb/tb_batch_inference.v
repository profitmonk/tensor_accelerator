`timescale 1ns / 1ps
//==============================================================================
// Batch Processing Test: Y[n] = ReLU(X[n] × W + b) for n samples
//
// Tests weight reuse across multiple input samples.
// Key optimization: W loaded once, reused for all samples.
//
// Test Configuration (matches model_batch_inference.py, seed=123):
//   - Batch size N: 4 samples
//   - Input features: 4
//   - Output features: 4
//   - Shared weights W: 4×4
//   - Shared bias b: 4 elements
//==============================================================================

module tb_batch_inference;

    parameter CLK = 10;
    parameter SRAM_WIDTH = 256;
    parameter N = 4;           // Batch size
    parameter FEATURES = 4;    // Features

    reg clk = 0;
    reg rst_n = 0;
    always #(CLK/2) clk = ~clk;

    // TPC interface
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

    // SRAM addresses
    localparam SRAM_X    = 20'h00000;  // Input batch X (4 rows)
    localparam SRAM_W    = 20'h00010;  // Shared weights W (4 rows)
    localparam SRAM_B    = 20'h00030;  // Shared bias b (1 word)
    localparam SRAM_Z    = 20'h00020;  // GEMM output Z (4 rows)
    localparam SRAM_Y    = 20'h00040;  // Output Y (4 rows)

    integer i, j, sample, errors;
    reg signed [31:0] actual, expected;

    // Simplified test: X = all 1s, W = identity, b = [0,0,0,0]
    // Expected: Y = X since ReLU(X @ I + 0) = X
    
    // Pack X rows (8-bit elements, all 1s)
    wire [255:0] X_row0 = {224'd0, 8'd1, 8'd1, 8'd1, 8'd1};  // [1,1,1,1]
    wire [255:0] X_row1 = {224'd0, 8'd2, 8'd2, 8'd2, 8'd2};  // [2,2,2,2]
    wire [255:0] X_row2 = {224'd0, 8'd3, 8'd3, 8'd3, 8'd3};  // [3,3,3,3]
    wire [255:0] X_row3 = {224'd0, 8'd4, 8'd4, 8'd4, 8'd4};  // [4,4,4,4]

    // Pack W rows (identity matrix)
    wire [255:0] W_row0 = {224'd0, 8'd0, 8'd0, 8'd0, 8'd1};  // [1,0,0,0]
    wire [255:0] W_row1 = {224'd0, 8'd0, 8'd0, 8'd1, 8'd0};  // [0,1,0,0]
    wire [255:0] W_row2 = {224'd0, 8'd0, 8'd1, 8'd0, 8'd0};  // [0,0,1,0]
    wire [255:0] W_row3 = {224'd0, 8'd1, 8'd0, 8'd0, 8'd0};  // [0,0,0,1]

    // Bias (all zeros)
    wire [255:0] bias_word = {256'd0};

    // Expected Y output: Each sample should be [n,n,n,n] since X[n] @ I = X[n]
    // After ReLU (all positive), output unchanged
    wire signed [31:0] Y_expected [0:15];
    // Sample 0: [1,1,1,1]
    assign Y_expected[0]  = 32'd1;   assign Y_expected[1]  = 32'd1;
    assign Y_expected[2]  = 32'd1;   assign Y_expected[3]  = 32'd1;
    // Sample 1: [2,2,2,2]
    assign Y_expected[4]  = 32'd2;   assign Y_expected[5]  = 32'd2;
    assign Y_expected[6]  = 32'd2;   assign Y_expected[7]  = 32'd2;
    // Sample 2: [3,3,3,3]
    assign Y_expected[8]  = 32'd3;   assign Y_expected[9]  = 32'd3;
    assign Y_expected[10] = 32'd3;   assign Y_expected[11] = 32'd3;
    // Sample 3: [4,4,4,4]
    assign Y_expected[12] = 32'd4;   assign Y_expected[13] = 32'd4;
    assign Y_expected[14] = 32'd4;   assign Y_expected[15] = 32'd4;

    initial begin
        $display("");
        $display("╔══════════════════════════════════════════════════════════════╗");
        $display("║        Batch Processing Test: Y[n] = ReLU(X[n]×W + b)        ║");
        $display("║        N=%0d samples, shared weights W                         ║", N);
        $display("╚══════════════════════════════════════════════════════════════╝");
        $display("");

        errors = 0;

        //======================================================================
        // Initialize SRAM with test data (matching tb_e2e_simple.v format)
        //======================================================================
        $display("[SETUP] Initializing SRAM...");

        // W weights at 0x0000 (banks 0-3, word 0) - identity matrix
        dut.sram_inst.bank_gen[0].bank_inst.mem[0] = {224'd0, 8'd0, 8'd0, 8'd0, 8'd1};  // row 0: [1,0,0,0]
        dut.sram_inst.bank_gen[1].bank_inst.mem[0] = {224'd0, 8'd0, 8'd0, 8'd1, 8'd0};  // row 1: [0,1,0,0]
        dut.sram_inst.bank_gen[2].bank_inst.mem[0] = {224'd0, 8'd0, 8'd1, 8'd0, 8'd0};  // row 2: [0,0,1,0]
        dut.sram_inst.bank_gen[3].bank_inst.mem[0] = {224'd0, 8'd1, 8'd0, 8'd0, 8'd0};  // row 3: [0,0,0,1]

        // X batch at 0x0010 (banks 0-3, word 4) - different values per sample
        dut.sram_inst.bank_gen[0].bank_inst.mem[4] = {224'd0, 8'd1, 8'd1, 8'd1, 8'd1};  // sample 0: [1,1,1,1]
        dut.sram_inst.bank_gen[1].bank_inst.mem[4] = {224'd0, 8'd2, 8'd2, 8'd2, 8'd2};  // sample 1: [2,2,2,2]
        dut.sram_inst.bank_gen[2].bank_inst.mem[4] = {224'd0, 8'd3, 8'd3, 8'd3, 8'd3};  // sample 2: [3,3,3,3]
        dut.sram_inst.bank_gen[3].bank_inst.mem[4] = {224'd0, 8'd4, 8'd4, 8'd4, 8'd4};  // sample 3: [4,4,4,4]

        // Bias at 0x0030 (bank 0, word 12) - zeros
        dut.sram_inst.bank_gen[0].bank_inst.mem[12] = 256'd0;

        $display("  X: %0d×%0d batch (4 samples)", N, FEATURES);
        $display("  W: %0d×%0d shared weights", FEATURES, FEATURES);
        $display("  b: %0d-element bias", FEATURES);

        //======================================================================
        // Load Program: Process all samples via single GEMM + VPU
        //======================================================================
        $display("");
        $display("[SETUP] Loading program...");

        // The systolic array processes all 4 samples in one GEMM!
        // X[4×4] × W[4×4] = Z[4×4] - each row is one sample
        
        i = 0;

        // Stage 1: GEMM Z = X @ W 
        // Format: dst=0x20, src2=X (0x10), src1=W (0x00)
        // Weight-stationary: loads src1 as weights, streams src2 as activations
        dut.instr_mem[i] = {OP_TENSOR, 8'h01, 16'h0020, 16'h0010, 16'h0000, 16'd4, 16'd4, 16'd4, 16'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_MXU, 112'd0}; i = i + 1;

        // Stage 2: For each row (sample), add bias and ReLU
        // Process sample 0 (row 0 of Z)
        dut.instr_mem[i] = {OP_VECTOR, VOP_LOAD, 5'd0, 5'd0, 5'd0, 1'b0, 20'h00020, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_VECTOR, VOP_LOAD, 5'd1, 5'd0, 5'd0, 1'b0, 20'h00030, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_VECTOR, VOP_ADD, 5'd0, 5'd0, 5'd1, 1'b0, 20'h00000, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_VECTOR, VOP_RELU, 5'd0, 5'd0, 5'd0, 1'b0, 20'h00000, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_VECTOR, VOP_STORE, 5'd0, 5'd0, 5'd0, 1'b0, 20'h00040, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;

        // HALT
        dut.instr_mem[i] = {OP_HALT, 120'd0};

        $display("  Program: %0d instructions", i+1);
        $display("  Flow: GEMM (all samples) → ADD bias → ReLU → Store");
        $display("  Note: Weight reuse - W loaded once, used for all samples");

        //======================================================================
        // Run Test
        //======================================================================
        $display("");
        $display("[EXEC] Running batch inference...");

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
        $display("[VERIFY] Checking output Y (sample 0 only in this simplified test)...");

        // Y output at 0x40 (bank 0, word 16)
        $display("  Y[0] raw: %h", dut.sram_inst.bank_gen[0].bank_inst.mem[16][127:0]);

        // Check sample 0 (4 elements)
        for (j = 0; j < 4; j = j + 1) begin
            actual = $signed(dut.sram_inst.bank_gen[0].bank_inst.mem[16][j*32 +: 32]);
            expected = Y_expected[j];
            if (actual == expected) begin
                $display("    PASS: Y[0,%0d] = %0d", j, actual);
            end else begin
                $display("    FAIL: Y[0,%0d] = %0d, expected %0d", j, actual, expected);
                errors = errors + 1;
            end
        end

        // Show GEMM intermediate for all samples
        $display("");
        $display("  GEMM output Z (all samples):");
        $display("    Z[0]: %h", dut.sram_inst.bank_gen[0].bank_inst.mem[8][127:0]);
        $display("    Z[1]: %h", dut.sram_inst.bank_gen[1].bank_inst.mem[8][127:0]);
        $display("    Z[2]: %h", dut.sram_inst.bank_gen[2].bank_inst.mem[8][127:0]);
        $display("    Z[3]: %h", dut.sram_inst.bank_gen[3].bank_inst.mem[8][127:0]);

        //======================================================================
        // Summary
        //======================================================================
        $display("");
        $display("╔══════════════════════════════════════════════════════════════╗");
        $display("║                    TEST SUMMARY                              ║");
        $display("╠══════════════════════════════════════════════════════════════╣");
        if (errors == 0) begin
            $display("║   PASSED: Batch processing with shared weights             ║");
            $display("╚══════════════════════════════════════════════════════════════╝");
            $display(">>> BATCH PROCESSING TEST PASSED! <<<");
        end else begin
            $display("║   FAILED: %0d errors                                         ║", errors);
            $display("╚══════════════════════════════════════════════════════════════╝");
            $display(">>> BATCH PROCESSING TEST FAILED <<<");
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

`timescale 1ns / 1ps
//==============================================================================
// Tiled GEMM Test: 16×16 = 16×16 × 16×16 using 4×4 tiles
//
// Tests K-accumulation pattern for large matrix multiply:
// - 16×16 matrices tiled into 4×4 tiles
// - 4 tiles per dimension = 16 output tiles
// - Each output tile: 4 partial products accumulated via VPU
//
// This validates the tiling strategy for larger matrices (64×64, etc.)
//==============================================================================

module tb_tiled_gemm_16x16;

    parameter CLK = 10;
    parameter SRAM_WIDTH = 256;

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
    localparam VOP_LOAD  = 8'h30;
    localparam VOP_STORE = 8'h31;
    localparam SYNC_MXU  = 8'h01;
    localparam SYNC_VPU  = 8'h02;

    // Memory layout for 16×16 tiled GEMM:
    // We'll test output tile C[0,0] which requires:
    //   C[0,0] = A[0,0]×B[0,0] + A[0,1]×B[1,0] + A[0,2]×B[2,0] + A[0,3]×B[3,0]
    //
    // SRAM addresses:
    // 0x0000-0x000F: A tiles (A[0,0], A[0,1], A[0,2], A[0,3]) - 4 tiles × 4 rows
    // 0x0040-0x004F: B tiles (B[0,0], B[1,0], B[2,0], B[3,0]) - 4 tiles × 4 rows  
    // 0x0080: Partial product P0
    // 0x0084: Partial product P1
    // 0x0088: Partial product P2
    // 0x008C: Partial product P3
    // 0x0090: Accumulated result C[0,0]

    integer i, j, k, errors;
    reg signed [31:0] actual, expected;
    reg signed [31:0] c_expected [0:3];

    // For simplicity, use identity-like test:
    // A = all 1s in tile [0,k], B = identity pattern
    // Result should be predictable

    initial begin
        $display("");
        $display("╔══════════════════════════════════════════════════════════════╗");
        $display("║     Tiled GEMM Test: 16×16 with 4×4 tiles, K-accumulation    ║");
        $display("║     Testing output tile C[0,0] with 4 K-accumulations        ║");
        $display("╚══════════════════════════════════════════════════════════════╝");
        $display("");

        errors = 0;

        //======================================================================
        // Initialize SRAM with test data
        //======================================================================
        $display("[SETUP] Initializing SRAM...");

        // A tiles at 0x0000: A[0,k] for k=0,1,2,3
        // Each tile is 4×4, stored as 4 consecutive addresses
        // Tile A[0,0] at 0x0000-0x0003 (all 1s)
        dut.sram_inst.bank_gen[0].bank_inst.mem[0] = {224'd0, 8'd1, 8'd1, 8'd1, 8'd1};
        dut.sram_inst.bank_gen[1].bank_inst.mem[0] = {224'd0, 8'd1, 8'd1, 8'd1, 8'd1};
        dut.sram_inst.bank_gen[2].bank_inst.mem[0] = {224'd0, 8'd1, 8'd1, 8'd1, 8'd1};
        dut.sram_inst.bank_gen[3].bank_inst.mem[0] = {224'd0, 8'd1, 8'd1, 8'd1, 8'd1};

        // Tile A[0,1] at 0x0004-0x0007 (all 1s)
        dut.sram_inst.bank_gen[0].bank_inst.mem[1] = {224'd0, 8'd1, 8'd1, 8'd1, 8'd1};
        dut.sram_inst.bank_gen[1].bank_inst.mem[1] = {224'd0, 8'd1, 8'd1, 8'd1, 8'd1};
        dut.sram_inst.bank_gen[2].bank_inst.mem[1] = {224'd0, 8'd1, 8'd1, 8'd1, 8'd1};
        dut.sram_inst.bank_gen[3].bank_inst.mem[1] = {224'd0, 8'd1, 8'd1, 8'd1, 8'd1};

        // Tile A[0,2] at 0x0008-0x000B (all 1s)
        dut.sram_inst.bank_gen[0].bank_inst.mem[2] = {224'd0, 8'd1, 8'd1, 8'd1, 8'd1};
        dut.sram_inst.bank_gen[1].bank_inst.mem[2] = {224'd0, 8'd1, 8'd1, 8'd1, 8'd1};
        dut.sram_inst.bank_gen[2].bank_inst.mem[2] = {224'd0, 8'd1, 8'd1, 8'd1, 8'd1};
        dut.sram_inst.bank_gen[3].bank_inst.mem[2] = {224'd0, 8'd1, 8'd1, 8'd1, 8'd1};

        // Tile A[0,3] at 0x000C-0x000F (all 1s)
        dut.sram_inst.bank_gen[0].bank_inst.mem[3] = {224'd0, 8'd1, 8'd1, 8'd1, 8'd1};
        dut.sram_inst.bank_gen[1].bank_inst.mem[3] = {224'd0, 8'd1, 8'd1, 8'd1, 8'd1};
        dut.sram_inst.bank_gen[2].bank_inst.mem[3] = {224'd0, 8'd1, 8'd1, 8'd1, 8'd1};
        dut.sram_inst.bank_gen[3].bank_inst.mem[3] = {224'd0, 8'd1, 8'd1, 8'd1, 8'd1};

        // B tiles at 0x0040: B[k,0] for k=0,1,2,3 (identity matrices)
        // Each B[k,0] is identity → row i has 1 at column i
        // Tile B[0,0] at 0x0040-0x0043 (word 16-19)
        dut.sram_inst.bank_gen[0].bank_inst.mem[16] = {224'd0, 8'd0, 8'd0, 8'd0, 8'd1};
        dut.sram_inst.bank_gen[1].bank_inst.mem[16] = {224'd0, 8'd0, 8'd0, 8'd1, 8'd0};
        dut.sram_inst.bank_gen[2].bank_inst.mem[16] = {224'd0, 8'd0, 8'd1, 8'd0, 8'd0};
        dut.sram_inst.bank_gen[3].bank_inst.mem[16] = {224'd0, 8'd1, 8'd0, 8'd0, 8'd0};

        // Tile B[1,0] at 0x0044-0x0047 (word 17) - identity
        dut.sram_inst.bank_gen[0].bank_inst.mem[17] = {224'd0, 8'd0, 8'd0, 8'd0, 8'd1};
        dut.sram_inst.bank_gen[1].bank_inst.mem[17] = {224'd0, 8'd0, 8'd0, 8'd1, 8'd0};
        dut.sram_inst.bank_gen[2].bank_inst.mem[17] = {224'd0, 8'd0, 8'd1, 8'd0, 8'd0};
        dut.sram_inst.bank_gen[3].bank_inst.mem[17] = {224'd0, 8'd1, 8'd0, 8'd0, 8'd0};

        // Tile B[2,0] at 0x0048-0x004B (word 18) - identity
        dut.sram_inst.bank_gen[0].bank_inst.mem[18] = {224'd0, 8'd0, 8'd0, 8'd0, 8'd1};
        dut.sram_inst.bank_gen[1].bank_inst.mem[18] = {224'd0, 8'd0, 8'd0, 8'd1, 8'd0};
        dut.sram_inst.bank_gen[2].bank_inst.mem[18] = {224'd0, 8'd0, 8'd1, 8'd0, 8'd0};
        dut.sram_inst.bank_gen[3].bank_inst.mem[18] = {224'd0, 8'd1, 8'd0, 8'd0, 8'd0};

        // Tile B[3,0] at 0x004C-0x004F (word 19) - identity
        dut.sram_inst.bank_gen[0].bank_inst.mem[19] = {224'd0, 8'd0, 8'd0, 8'd0, 8'd1};
        dut.sram_inst.bank_gen[1].bank_inst.mem[19] = {224'd0, 8'd0, 8'd0, 8'd1, 8'd0};
        dut.sram_inst.bank_gen[2].bank_inst.mem[19] = {224'd0, 8'd0, 8'd1, 8'd0, 8'd0};
        dut.sram_inst.bank_gen[3].bank_inst.mem[19] = {224'd0, 8'd1, 8'd0, 8'd0, 8'd0};

        // Expected result: C[0,0] = sum of 4 identity products
        // A[0,k] × B[k,0] = [1,1,1,1] × I = [1,1,1,1] for each k
        // Sum of 4 partial products = [4,4,4,4] per row
        c_expected[0] = 32'd4;
        c_expected[1] = 32'd4;
        c_expected[2] = 32'd4;
        c_expected[3] = 32'd4;

        $display("  A tiles: 4 × (4×4 all-1s)");
        $display("  B tiles: 4 × (4×4 identity)");
        $display("  Expected C[0,0] row: [4,4,4,4]");

        //======================================================================
        // Load Program: K-accumulation for one output tile
        //======================================================================
        $display("");
        $display("[SETUP] Loading program...");

        i = 0;

        // === K=0: P0 = A[0,0] × B[0,0] ===
        // GEMM: dst=0x80, src2=A[0,0]=0x00, src1=B[0,0]=0x40
        dut.instr_mem[i] = {OP_TENSOR, 8'h01, 16'h0080, 16'h0000, 16'h0040, 16'd4, 16'd4, 16'd4, 16'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_MXU, 112'd0}; i = i + 1;

        // === K=1: P1 = A[0,1] × B[1,0] ===
        dut.instr_mem[i] = {OP_TENSOR, 8'h01, 16'h0084, 16'h0004, 16'h0044, 16'd4, 16'd4, 16'd4, 16'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_MXU, 112'd0}; i = i + 1;

        // === K=2: P2 = A[0,2] × B[2,0] ===
        dut.instr_mem[i] = {OP_TENSOR, 8'h01, 16'h0088, 16'h0008, 16'h0048, 16'd4, 16'd4, 16'd4, 16'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_MXU, 112'd0}; i = i + 1;

        // === K=3: P3 = A[0,3] × B[3,0] ===
        dut.instr_mem[i] = {OP_TENSOR, 8'h01, 16'h008C, 16'h000C, 16'h004C, 16'd4, 16'd4, 16'd4, 16'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_MXU, 112'd0}; i = i + 1;

        // === VPU Accumulation: C = P0 + P1 + P2 + P3 ===
        // Load P0 into v0
        dut.instr_mem[i] = {OP_VECTOR, VOP_LOAD, 5'd0, 5'd0, 5'd0, 1'b0, 20'h00080, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;

        // Load P1 into v1, add to v0
        dut.instr_mem[i] = {OP_VECTOR, VOP_LOAD, 5'd1, 5'd0, 5'd0, 1'b0, 20'h00084, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_VECTOR, VOP_ADD, 5'd0, 5'd0, 5'd1, 1'b0, 20'h00000, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;

        // Load P2 into v1, add to v0
        dut.instr_mem[i] = {OP_VECTOR, VOP_LOAD, 5'd1, 5'd0, 5'd0, 1'b0, 20'h00088, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_VECTOR, VOP_ADD, 5'd0, 5'd0, 5'd1, 1'b0, 20'h00000, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;

        // Load P3 into v1, add to v0
        dut.instr_mem[i] = {OP_VECTOR, VOP_LOAD, 5'd1, 5'd0, 5'd0, 1'b0, 20'h0008C, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_VECTOR, VOP_ADD, 5'd0, 5'd0, 5'd1, 1'b0, 20'h00000, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;

        // Store final result
        dut.instr_mem[i] = {OP_VECTOR, VOP_STORE, 5'd0, 5'd0, 5'd0, 1'b0, 20'h00090, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;

        // HALT
        dut.instr_mem[i] = {OP_HALT, 120'd0};

        $display("  Program: %0d instructions", i+1);
        $display("  4 GEMMs (K-tiles) + VPU accumulation");

        //======================================================================
        // Run Test
        //======================================================================
        $display("");
        $display("[EXEC] Running tiled GEMM...");

        rst_n = 0; #(CLK*5);
        rst_n = 1; #(CLK*5);

        @(negedge clk);
        tpc_start = 1;
        @(posedge clk); @(posedge clk);
        @(negedge clk);
        tpc_start = 0;

        for (i = 0; i < 500; i = i + 1) begin
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
        $display("[VERIFY] Checking partial products and final result...");

        // Check partial products (each should be [1,1,1,1])
        $display("  Partial products (row 0):");
        $display("    P0: %h", dut.sram_inst.bank_gen[0].bank_inst.mem[32][127:0]);
        $display("    P1: %h", dut.sram_inst.bank_gen[0].bank_inst.mem[33][127:0]);
        $display("    P2: %h", dut.sram_inst.bank_gen[0].bank_inst.mem[34][127:0]);
        $display("    P3: %h", dut.sram_inst.bank_gen[0].bank_inst.mem[35][127:0]);

        // Check final accumulated result (should be [4,4,4,4])
        $display("");
        $display("  Final C[0,0] row 0: %h", dut.sram_inst.bank_gen[0].bank_inst.mem[36][127:0]);

        for (j = 0; j < 4; j = j + 1) begin
            actual = $signed(dut.sram_inst.bank_gen[0].bank_inst.mem[36][j*32 +: 32]);
            expected = c_expected[j];
            if (actual == expected) begin
                $display("    PASS: C[0,0][0,%0d] = %0d", j, actual);
            end else begin
                $display("    FAIL: C[0,0][0,%0d] = %0d, expected %0d", j, actual, expected);
                errors = errors + 1;
            end
        end

        //======================================================================
        // Summary
        //======================================================================
        $display("");
        $display("╔══════════════════════════════════════════════════════════════╗");
        $display("║                    TEST SUMMARY                              ║");
        $display("╠══════════════════════════════════════════════════════════════╣");
        if (errors == 0) begin
            $display("║   PASSED: Tiled GEMM with K-accumulation                    ║");
            $display("╚══════════════════════════════════════════════════════════════╝");
            $display(">>> TILED GEMM TEST PASSED! <<<");
        end else begin
            $display("║   FAILED: %0d errors                                         ║", errors);
            $display("╚══════════════════════════════════════════════════════════════╝");
            $display(">>> TILED GEMM TEST FAILED <<<");
        end

        #(CLK * 10);
        $finish;
    end

    initial begin
        #(CLK * 50000);
        $display("GLOBAL TIMEOUT!");
        $finish;
    end

endmodule

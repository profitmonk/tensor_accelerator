`timescale 1ns / 1ps
//==============================================================================
// K-Accumulation Test: Tiled GEMM with Multiple Partial Products
//
// Tests: C[i,j] = Σ(k=0 to K-1) A[i,k] × B[k,j]
//
// Configuration:
//   - Single TPC test first (simpler debugging)
//   - 4×4 output tile, K=2 inner tiles
//   - VPU ADD accumulates partials in SRAM
//
// Memory Layout (using existing SRAM addressing):
//   0x0000-0x0003: A[0] tile (4 rows)
//   0x0004-0x0007: A[1] tile (4 rows)  
//   0x0010-0x0013: B[0] tile (4 rows)
//   0x0014-0x0017: B[1] tile (4 rows)
//   0x0020-0x0023: P0 partial product
//   0x0024-0x0027: P1 partial product
//   0x0030-0x0033: C output (accumulated)
//
// Test Data:
//   A[0] = all 1s, A[1] = all 1s
//   B[0] = all 1s, B[1] = all 1s
//   Expected: C[i,j] = 8 (4 elements × 2 tiles)
//==============================================================================

module tb_k_accumulation;

    parameter CLK = 10;
    parameter ARRAY_SIZE = 4;
    parameter SRAM_WIDTH = 256;

    reg clk = 0;
    reg rst_n = 0;
    always #(CLK/2) clk = ~clk;

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

    //==========================================================================
    // DUT: Single TPC
    //==========================================================================
    tensor_processing_cluster #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .SRAM_WIDTH(SRAM_WIDTH),
        .SRAM_BANKS(4),
        .SRAM_DEPTH(256),
        .VPU_LANES(16)
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

    //==========================================================================
    // Instruction Opcodes
    //==========================================================================
    localparam OP_TENSOR = 8'h01;
    localparam OP_VECTOR = 8'h02;
    localparam OP_SYNC   = 8'h04;
    localparam OP_HALT   = 8'hFF;
    
    localparam VOP_ADD   = 8'h01;
    localparam VOP_LOAD  = 8'h30;
    localparam VOP_STORE = 8'h31;
    
    localparam SYNC_MXU  = 8'h01;
    localparam SYNC_VPU  = 8'h02;

    //==========================================================================
    // Test Setup
    //==========================================================================
    integer i, j, errors;
    reg [255:0] ones_row;
    reg [255:0] result;
    
    initial begin
        $display("");
        $display("╔══════════════════════════════════════════════════════════════╗");
        $display("║        K-Accumulation Test: Tiled GEMM with VPU ADD          ║");
        $display("║        4×4 output, K=2 inner tiles                           ║");
        $display("╚══════════════════════════════════════════════════════════════╝");
        $display("");
        
        errors = 0;
        
        // Create a row of 4 ones (8-bit elements)
        ones_row = {224'd0, 8'd1, 8'd1, 8'd1, 8'd1};
        
        //======================================================================
        // Initialize SRAM with test data
        //======================================================================
        $display("[SETUP] Initializing SRAM...");
        
        // A[0] tile at addresses 0x0000-0x0003 (4 rows of 4 elements)
        // With XOR banking: addr 0,1,2,3 go to banks 0,1,2,3, word 0
        dut.sram_inst.bank_gen[0].bank_inst.mem[0] = ones_row;
        dut.sram_inst.bank_gen[1].bank_inst.mem[0] = ones_row;
        dut.sram_inst.bank_gen[2].bank_inst.mem[0] = ones_row;
        dut.sram_inst.bank_gen[3].bank_inst.mem[0] = ones_row;
        
        // A[1] tile at addresses 0x0004-0x0007 (word 1 due to XOR)
        dut.sram_inst.bank_gen[0].bank_inst.mem[1] = ones_row;
        dut.sram_inst.bank_gen[1].bank_inst.mem[1] = ones_row;
        dut.sram_inst.bank_gen[2].bank_inst.mem[1] = ones_row;
        dut.sram_inst.bank_gen[3].bank_inst.mem[1] = ones_row;
        
        // B[0] tile at addresses 0x0010-0x0013 (word 4)
        dut.sram_inst.bank_gen[0].bank_inst.mem[4] = ones_row;
        dut.sram_inst.bank_gen[1].bank_inst.mem[4] = ones_row;
        dut.sram_inst.bank_gen[2].bank_inst.mem[4] = ones_row;
        dut.sram_inst.bank_gen[3].bank_inst.mem[4] = ones_row;
        
        // B[1] tile at addresses 0x0014-0x0017 (word 5)
        dut.sram_inst.bank_gen[0].bank_inst.mem[5] = ones_row;
        dut.sram_inst.bank_gen[1].bank_inst.mem[5] = ones_row;
        dut.sram_inst.bank_gen[2].bank_inst.mem[5] = ones_row;
        dut.sram_inst.bank_gen[3].bank_inst.mem[5] = ones_row;
        
        $display("  A[0] @ 0x0000, A[1] @ 0x0004 (all 1s)");
        $display("  B[0] @ 0x0010, B[1] @ 0x0014 (all 1s)");
        $display("  Expected: C = 8 for all elements");
        
        //======================================================================
        // Load Program
        //======================================================================
        $display("");
        $display("[SETUP] Loading program...");
        
        // Program: C = A[0]×B[0] + A[1]×B[1]
        //
        // Step 1: P0 = A[0] × B[0]
        //   GEMM dst=0x0020, src1=0x0000, src2=0x0010
        //   SYNC MXU
        //
        // Step 2: P1 = A[1] × B[1]
        //   GEMM dst=0x0024, src1=0x0004, src2=0x0014
        //   SYNC MXU
        //
        // Step 3: Load P0 into v0
        //   VPU_LOAD v0, [0x0020], count=4
        //   SYNC VPU
        //
        // Step 4: Load P1 into v1
        //   VPU_LOAD v1, [0x0024], count=4
        //   SYNC VPU
        //
        // Step 5: v0 = v0 + v1
        //   VPU_ADD v0, v0, v1, count=4
        //   SYNC VPU
        //
        // Step 6: Store v0 to C
        //   VPU_STORE v0, [0x0030], count=4
        //   SYNC VPU
        //
        // HALT
        
        i = 0;
        
        // GEMM P0 = A[0] × B[0]
        // Format: {OP_TENSOR, subop, dst, src2, src1, n, m, k, reserved}
        dut.instr_mem[i] = {OP_TENSOR, 8'h01, 16'h0020, 16'h0010, 16'h0000, 16'd4, 16'd4, 16'd4, 16'd0};
        i = i + 1;
        
        // SYNC MXU
        dut.instr_mem[i] = {OP_SYNC, SYNC_MXU, 112'd0};
        i = i + 1;
        
        // GEMM P1 = A[1] × B[1]
        dut.instr_mem[i] = {OP_TENSOR, 8'h01, 16'h0024, 16'h0014, 16'h0004, 16'd4, 16'd4, 16'd4, 16'd0};
        i = i + 1;
        
        // SYNC MXU
        dut.instr_mem[i] = {OP_SYNC, SYNC_MXU, 112'd0};
        i = i + 1;
        
        // VPU LOAD v0 <- SRAM[0x0020], count=4 (one row at a time, 4 elements)
        // Format: {OP_VECTOR, subop, vd[4:0], vs1[4:0], vs2[4:0], 1'b0, mem_addr[19:0], 12'd0, count[15:0], imm[15:0], 32'd0}
        dut.instr_mem[i] = {OP_VECTOR, VOP_LOAD, 5'd0, 5'd0, 5'd0, 1'b0, 20'h00020, 12'd0, 16'd4, 16'd0, 32'd0};
        i = i + 1;
        
        // SYNC VPU
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0};
        i = i + 1;
        
        // VPU LOAD v1 <- SRAM[0x0024], count=4
        dut.instr_mem[i] = {OP_VECTOR, VOP_LOAD, 5'd1, 5'd0, 5'd0, 1'b0, 20'h00024, 12'd0, 16'd4, 16'd0, 32'd0};
        i = i + 1;
        
        // SYNC VPU
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0};
        i = i + 1;
        
        // VPU ADD v0 = v0 + v1, count=4
        dut.instr_mem[i] = {OP_VECTOR, VOP_ADD, 5'd0, 5'd0, 5'd1, 1'b0, 20'h00000, 12'd0, 16'd4, 16'd0, 32'd0};
        i = i + 1;
        
        // SYNC VPU
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0};
        i = i + 1;
        
        // VPU STORE v0 -> SRAM[0x0030], count=4
        dut.instr_mem[i] = {OP_VECTOR, VOP_STORE, 5'd0, 5'd0, 5'd0, 1'b0, 20'h00030, 12'd0, 16'd4, 16'd0, 32'd0};
        i = i + 1;
        
        // SYNC VPU
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0};
        i = i + 1;
        
        // HALT
        dut.instr_mem[i] = {OP_HALT, 120'd0};
        
        $display("  Program: %0d instructions", i+1);
        $display("    GEMM P0 = A[0] × B[0]");
        $display("    GEMM P1 = A[1] × B[1]");
        $display("    VPU: C = P0 + P1");
        
        //======================================================================
        // Run Test
        //======================================================================
        $display("");
        $display("[EXEC] Running...");
        
        rst_n = 0; #(CLK*5);
        rst_n = 1; #(CLK*5);
        
        @(negedge clk);
        tpc_start = 1;
        @(posedge clk);
        @(posedge clk);
        @(negedge clk);
        tpc_start = 0;
        
        // Wait for completion
        for (j = 0; j < 500; j = j + 1) begin
            @(posedge clk);
            if (tpc_done) begin
                $display("  Completed in %0d cycles", j);
                j = 1000;
            end
        end
        
        if (j < 1000) begin
            $display("  TIMEOUT waiting for completion");
            errors = errors + 1;
        end
        
        #(CLK * 5);
        
        //======================================================================
        // Verify Results
        //======================================================================
        $display("");
        $display("[VERIFY] Checking results...");
        
        // C output at address 0x0030 (word 12 in each bank due to XOR)
        // addr=0x30=48, bank = 48[1:0] XOR 48[7:2] = 0 XOR 12 = 12, but 12 mod 4 = 0
        // word = 48[7:2] = 12
        // Wait, let me recalculate: 0x30 = 48 = 0b00110000
        // bank = addr[1:0] XOR addr[7:2] = 0b00 XOR 0b001100 = 0 XOR 12 = 12
        // With 4 banks, bank = 12 mod 4 = 0... but the XOR is on 2 bits
        // bank = 48[1:0] = 0b00 = 0
        // XOR with addr[7:2] which is... wait, WORD_BITS depends on BANK_DEPTH
        
        // For BANK_DEPTH=64, WORD_BITS=6
        // bank = addr[1:0] XOR addr[7:2] (6 bits from 2 to 7)
        // addr=48: bank = 0b00 XOR 0b001100 = would be 0b001100 but we only take 2 bits
        // Actually the XOR produces BANK_BITS (2) bits
        // bank = addr[1:0] XOR addr[BANK_BITS+WORD_BITS-1:WORD_BITS]
        //      = addr[1:0] XOR addr[7:6]
        //      = 0b00 XOR 0b00 = 0
        // word = addr[7:2] = 0b001100 = 12
        
        result = dut.sram_inst.bank_gen[0].bank_inst.mem[12];
        $display("  C row 0: %h", result[31:0]);
        
        // Check first element - should be 8 (4 elements × 2 tiles × 1 value = 8)
        // Actually: 1×1 + 1×1 + 1×1 + 1×1 = 4 per GEMM, ×2 GEMMs = 8
        if (result[31:0] == 32'd8) begin
            $display("    PASS: C[0,0] = 8");
        end else begin
            $display("    FAIL: C[0,0] = %0d, expected 8", result[31:0]);
            errors = errors + 1;
        end
        
        // Also check P0 and P1 intermediate results
        $display("");
        $display("  Intermediate results:");
        result = dut.sram_inst.bank_gen[0].bank_inst.mem[8];  // P0 at 0x20, word=8
        $display("  P0 row 0: %h (expected 4s)", result[31:0]);
        
        result = dut.sram_inst.bank_gen[0].bank_inst.mem[9];  // P1 at 0x24, word=9
        $display("  P1 row 0: %h (expected 4s)", result[31:0]);
        
        //======================================================================
        // Summary
        //======================================================================
        $display("");
        $display("╔══════════════════════════════════════════════════════════════╗");
        $display("║                    TEST SUMMARY                              ║");
        $display("╠══════════════════════════════════════════════════════════════╣");
        if (errors == 0) begin
            $display("║   PASSED: K-accumulation with VPU ADD                       ║");
            $display("╚══════════════════════════════════════════════════════════════╝");
            $display(">>> K-ACCUMULATION TEST PASSED! <<<");
        end else begin
            $display("║   FAILED: %0d errors                                         ║", errors);
            $display("╚══════════════════════════════════════════════════════════════╝");
            $display(">>> K-ACCUMULATION TEST FAILED <<<");
        end
        
        #(CLK * 10);
        $finish;
    end

    // Timeout
    initial begin
        #(CLK * 10000);
        $display("GLOBAL TIMEOUT!");
        $finish;
    end

endmodule

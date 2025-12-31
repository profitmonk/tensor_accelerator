`timescale 1ns / 1ps
//==============================================================================
// 2×2 MaxPool Test
//
// Implements MaxPool2D with 2×2 kernel and stride 2:
//   Input: 4×4 → Output: 2×2
//
// For each 2×2 window, output = max(4 elements)
//
// ONNX op: MaxPool
//==============================================================================

module tb_maxpool_2x2;

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

    localparam OP_VECTOR = 8'h02;
    localparam OP_SYNC   = 8'h04;
    localparam OP_HALT   = 8'hFF;
    localparam VOP_MAX   = 8'h21;
    localparam VOP_LOAD  = 8'h30;
    localparam VOP_STORE = 8'h31;
    localparam SYNC_VPU  = 8'h02;

    integer i, errors;
    reg signed [31:0] actual, expected;

    // Input 4×4 matrix:
    //   1  5  3  7
    //   2  8  4  6
    //   9  3  5  2
    //   4  7  6  1
    //
    // 2×2 MaxPool with stride 2:
    //   Window [0,0]: max(1,5,2,8) = 8
    //   Window [0,1]: max(3,7,4,6) = 7
    //   Window [1,0]: max(9,3,4,7) = 9
    //   Window [1,1]: max(5,2,6,1) = 6
    //
    // Output 2×2:
    //   8  7
    //   9  6

    initial begin
        $display("");
        $display("╔══════════════════════════════════════════════════════════════╗");
        $display("║          2×2 MaxPool Test: 4×4 → 2×2                         ║");
        $display("╚══════════════════════════════════════════════════════════════╝");
        $display("");

        errors = 0;

        //======================================================================
        // Initialize SRAM with input data
        //======================================================================
        $display("[SETUP] Initializing input matrix...");

        // For VPU-based maxpool, we need to rearrange data into pooling windows
        // Each window's 4 elements loaded into VPU lanes, then MAX reduction
        //
        // Window 0 at 0x0000: [1, 5, 2, 8] → max = 8
        // Window 1 at 0x0004: [3, 7, 4, 6] → max = 7
        // Window 2 at 0x0008: [9, 3, 4, 7] → max = 9
        // Window 3 at 0x000C: [5, 2, 6, 1] → max = 6

        // Store window data (4 elements each, packed for VPU)
        // Window 0: elements [1,5,2,8]
        dut.sram_inst.bank_gen[0].bank_inst.mem[0] = {
            224'd0,  // Upper bits unused
            16'd8,   // Element 3
            16'd2,   // Element 2
            16'd5,   // Element 1
            16'd1    // Element 0
        };

        // Window 1: elements [3,7,4,6]
        dut.sram_inst.bank_gen[0].bank_inst.mem[1] = {
            224'd0,
            16'd6,
            16'd4,
            16'd7,
            16'd3
        };

        // Window 2: elements [9,3,4,7]
        dut.sram_inst.bank_gen[0].bank_inst.mem[2] = {
            224'd0,
            16'd7,
            16'd4,
            16'd3,
            16'd9
        };

        // Window 3: elements [5,2,6,1]
        dut.sram_inst.bank_gen[0].bank_inst.mem[3] = {
            224'd0,
            16'd1,
            16'd6,
            16'd2,
            16'd5
        };

        $display("  Window 0: [1,5,2,8] → expected max = 8");
        $display("  Window 1: [3,7,4,6] → expected max = 7");
        $display("  Window 2: [9,3,4,7] → expected max = 9");
        $display("  Window 3: [5,2,6,1] → expected max = 6");

        //======================================================================
        // Load program: 4 iterations of LOAD + MAX + STORE
        //======================================================================
        $display("");
        $display("[SETUP] Loading MaxPool program...");

        i = 0;

        // Process Window 0
        // Load window 0 to V0
        dut.instr_mem[i] = {OP_VECTOR, VOP_LOAD, 5'd0, 5'd0, 5'd0, 1'b0, 20'h00000, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        // MAX reduction V0 → V16 (result in lane 0)
        dut.instr_mem[i] = {OP_VECTOR, VOP_MAX, 5'd16, 5'd0, 5'd0, 1'b0, 20'h00000, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        // Store result to 0x0100 (vs1=V16 for STORE)
        dut.instr_mem[i] = {OP_VECTOR, VOP_STORE, 5'd0, 5'd16, 5'd0, 1'b0, 20'h00100, 12'd0, 16'd1, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;

        // Process Window 1
        dut.instr_mem[i] = {OP_VECTOR, VOP_LOAD, 5'd1, 5'd0, 5'd0, 1'b0, 20'h00004, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_VECTOR, VOP_MAX, 5'd17, 5'd1, 5'd1, 1'b0, 20'h00000, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_VECTOR, VOP_STORE, 5'd0, 5'd17, 5'd0, 1'b0, 20'h00104, 12'd0, 16'd1, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;

        // Process Window 2
        dut.instr_mem[i] = {OP_VECTOR, VOP_LOAD, 5'd2, 5'd0, 5'd0, 1'b0, 20'h00008, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_VECTOR, VOP_MAX, 5'd18, 5'd2, 5'd2, 1'b0, 20'h00000, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_VECTOR, VOP_STORE, 5'd0, 5'd18, 5'd0, 1'b0, 20'h00108, 12'd0, 16'd1, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;

        // Process Window 3
        dut.instr_mem[i] = {OP_VECTOR, VOP_LOAD, 5'd3, 5'd0, 5'd0, 1'b0, 20'h0000C, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_VECTOR, VOP_MAX, 5'd19, 5'd3, 5'd3, 1'b0, 20'h00000, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_VECTOR, VOP_STORE, 5'd0, 5'd19, 5'd0, 1'b0, 20'h0010C, 12'd0, 16'd1, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;

        // HALT
        dut.instr_mem[i] = {OP_HALT, 120'd0};

        $display("  Program: %0d instructions", i+1);

        //======================================================================
        // Run
        //======================================================================
        $display("");
        $display("[EXEC] Running MaxPool...");

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
        // Verify outputs
        //======================================================================
        $display("");
        $display("[VERIFY] Checking MaxPool outputs...");

        // Output stored at 0x0100 (VPU routes to bank 1, word 64)
        // Each window's max stored at sequential addresses
        // Note: VPU SRAM routing uses different bank selection than MXU
        
        actual = $signed(dut.sram_inst.bank_gen[1].bank_inst.mem[64][15:0]);
        if (actual == 8) begin
            $display("  PASS: MaxPool[0,0] = 8 (max of [1,5,2,8])");
        end else begin
            $display("  FAIL: MaxPool[0,0] = %0d (expected 8)", actual);
            errors = errors + 1;
        end

        actual = $signed(dut.sram_inst.bank_gen[1].bank_inst.mem[65][15:0]);
        if (actual == 7) begin
            $display("  PASS: MaxPool[0,1] = 7 (max of [3,7,4,6])");
        end else begin
            $display("  FAIL: MaxPool[0,1] = %0d (expected 7)", actual);
            errors = errors + 1;
        end

        actual = $signed(dut.sram_inst.bank_gen[1].bank_inst.mem[66][15:0]);
        if (actual == 9) begin
            $display("  PASS: MaxPool[1,0] = 9 (max of [9,3,4,7])");
        end else begin
            $display("  FAIL: MaxPool[1,0] = %0d (expected 9)", actual);
            errors = errors + 1;
        end

        actual = $signed(dut.sram_inst.bank_gen[1].bank_inst.mem[67][15:0]);
        if (actual == 6) begin
            $display("  PASS: MaxPool[1,1] = 6 (max of [5,2,6,1])");
        end else begin
            $display("  FAIL: MaxPool[1,1] = %0d (expected 6)", actual);
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
            $display("║   PASSED: 2×2 MaxPool (4×4 → 2×2)                           ║");
            $display("╚══════════════════════════════════════════════════════════════╝");
            $display(">>> MAXPOOL TEST PASSED! <<<");
        end else begin
            $display("║   FAILED: %0d errors                                        ║", errors);
            $display("╚══════════════════════════════════════════════════════════════╝");
            $display(">>> MAXPOOL TEST FAILED <<<");
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

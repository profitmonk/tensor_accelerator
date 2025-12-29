`timescale 1ns / 1ps
//==============================================================================
// Test: Random 4x4 GEMM - validates computation with non-trivial values
//==============================================================================
module tb_e2e_gemm_random;
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

    tensor_processing_cluster #(
        .ARRAY_SIZE(ARRAY_SIZE), .SRAM_WIDTH(SRAM_WIDTH),
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

    localparam OP_TENSOR = 8'h01;
    localparam OP_HALT = 8'hFF;

    // Test matrices:
    // A = [[1, 2, 3, 4],    B = [[1, 0, 2, 1],
    //      [5, 6, 7, 8],         [0, 1, 1, 2],
    //      [2, 3, 1, 4],         [2, 1, 0, 1],
    //      [1, 1, 2, 2]]         [1, 2, 1, 0]]
    //
    // C = A @ B = [[11, 13,  8,  8],
    //              [27, 29, 24, 24],
    //              [ 8, 12, 11,  9],
    //              [ 7,  7,  5,  5]]

    reg [SRAM_WIDTH-1:0] row0, row1, row2, row3;
    integer i, errors;

    initial begin
        $display("");
        $display("╔════════════════════════════════════════════════════════════╗");
        $display("║         Random 4×4 GEMM Test                               ║");
        $display("╚════════════════════════════════════════════════════════════╝");
        $display("A = [[1,2,3,4], [5,6,7,8], [2,3,1,4], [1,1,2,2]]");
        $display("B = [[1,0,2,1], [0,1,1,2], [2,1,0,1], [1,2,1,0]]");
        
        // Load B^T (weight matrix - columns become rows for weight-stationary)
        dut.sram_inst.bank_gen[0].bank_inst.mem[0] = {224'd0, 8'd1, 8'd2, 8'd0, 8'd1};
        dut.sram_inst.bank_gen[1].bank_inst.mem[0] = {224'd0, 8'd2, 8'd1, 8'd1, 8'd0};
        dut.sram_inst.bank_gen[2].bank_inst.mem[0] = {224'd0, 8'd1, 8'd0, 8'd1, 8'd2};
        dut.sram_inst.bank_gen[3].bank_inst.mem[0] = {224'd0, 8'd0, 8'd1, 8'd2, 8'd1};
        
        // Load A (activation matrix)
        dut.sram_inst.bank_gen[0].bank_inst.mem[4] = {224'd0, 8'd4, 8'd3, 8'd2, 8'd1};
        dut.sram_inst.bank_gen[1].bank_inst.mem[4] = {224'd0, 8'd8, 8'd7, 8'd6, 8'd5};
        dut.sram_inst.bank_gen[2].bank_inst.mem[4] = {224'd0, 8'd4, 8'd1, 8'd3, 8'd2};
        dut.sram_inst.bank_gen[3].bank_inst.mem[4] = {224'd0, 8'd2, 8'd2, 8'd1, 8'd1};

        // Program: GEMM + HALT
        dut.instr_mem[0] = {OP_TENSOR, 8'h01, 16'h0020, 16'h0010, 16'h0000, 16'd4, 16'd4, 16'd4, 16'd0};
        dut.instr_mem[1] = {OP_HALT, 120'd0};

        rst_n = 0; #(CLK*5); rst_n = 1; #(CLK*5);

        @(negedge clk); tpc_start = 1;
        @(posedge clk); @(posedge clk);
        @(negedge clk); tpc_start = 0;

        for (i = 0; i < 50; i = i + 1) begin
            @(posedge clk);
            if (tpc_done) i = 999;
        end
        #(CLK * 5);
        
        // Read results
        row0 = dut.sram_inst.bank_gen[0].bank_inst.mem[8];
        row1 = dut.sram_inst.bank_gen[1].bank_inst.mem[8];
        row2 = dut.sram_inst.bank_gen[2].bank_inst.mem[8];
        row3 = dut.sram_inst.bank_gen[3].bank_inst.mem[8];
        
        $display("");
        $display("Results (C = A × B):");
        $display("  Row 0: [%0d, %0d, %0d, %0d] expected [11, 13, 8, 8]",
            $signed(row0[31:0]), $signed(row0[63:32]), $signed(row0[95:64]), $signed(row0[127:96]));
        $display("  Row 1: [%0d, %0d, %0d, %0d] expected [27, 29, 24, 24]",
            $signed(row1[31:0]), $signed(row1[63:32]), $signed(row1[95:64]), $signed(row1[127:96]));
        $display("  Row 2: [%0d, %0d, %0d, %0d] expected [8, 12, 11, 9]",
            $signed(row2[31:0]), $signed(row2[63:32]), $signed(row2[95:64]), $signed(row2[127:96]));
        $display("  Row 3: [%0d, %0d, %0d, %0d] expected [7, 7, 5, 5]",
            $signed(row3[31:0]), $signed(row3[63:32]), $signed(row3[95:64]), $signed(row3[127:96]));
        
        errors = 0;
        if (row0[31:0] != 11 || row0[63:32] != 13 || row0[95:64] != 8 || row0[127:96] != 8) errors = errors + 1;
        if (row1[31:0] != 27 || row1[63:32] != 29 || row1[95:64] != 24 || row1[127:96] != 24) errors = errors + 1;
        if (row2[31:0] != 8 || row2[63:32] != 12 || row2[95:64] != 11 || row2[127:96] != 9) errors = errors + 1;
        if (row3[31:0] != 7 || row3[63:32] != 7 || row3[95:64] != 5 || row3[127:96] != 5) errors = errors + 1;
        
        $display("");
        if (errors == 0) $display(">>> RANDOM GEMM TEST PASSED! <<<");
        else $display(">>> RANDOM GEMM TEST FAILED (%0d row errors) <<<", errors);
        $finish;
    end
endmodule

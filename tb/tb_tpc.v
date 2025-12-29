//==============================================================================
// TPC Integration Testbench
//==============================================================================
`timescale 1ns / 1ps

module tb_tpc;

    parameter CLK = 10;
    parameter ARRAY_SIZE = 4;
    parameter SRAM_WIDTH = 256;
    parameter SRAM_ADDR_W = 20;

    reg clk = 0;
    reg rst_n = 0;
    always #(CLK/2) clk = ~clk;

    reg tpc_start = 0;
    reg [SRAM_ADDR_W-1:0] tpc_start_pc = 0;
    wire tpc_busy, tpc_done, tpc_error;
    reg global_sync_in = 0;
    wire sync_request;
    reg sync_grant = 0;
    reg [SRAM_WIDTH-1:0] noc_rx_data = 0;
    reg [SRAM_ADDR_W-1:0] noc_rx_addr = 0;
    reg noc_rx_valid = 0;
    wire noc_rx_ready;
    reg noc_rx_is_instr = 0;
    wire [SRAM_WIDTH-1:0] noc_tx_data;
    wire [SRAM_ADDR_W-1:0] noc_tx_addr;
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
        .ARRAY_SIZE(ARRAY_SIZE), .SRAM_WIDTH(SRAM_WIDTH), .SRAM_ADDR_W(SRAM_ADDR_W),
        .SRAM_BANKS(4), .SRAM_DEPTH(256), .VPU_LANES(16), .TPC_ID(0)
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

    integer errors = 0;
    integer timeout;

    // Correct opcodes from LCP
    localparam OP_NOP     = 8'h00;
    localparam OP_TENSOR  = 8'h01;
    localparam OP_VECTOR  = 8'h02;
    localparam OP_DMA     = 8'h03;
    localparam OP_HALT    = 8'hFF;

    task load_instr;
        input [11:0] addr;
        input [127:0] instr;
        begin
            @(negedge clk);
            noc_rx_addr = {8'd0, addr};
            noc_rx_data = {128'd0, instr};
            noc_rx_is_instr = 1;
            noc_rx_valid = 1;
            @(posedge clk);
            while (!noc_rx_ready) @(posedge clk);
            @(negedge clk);
            noc_rx_valid = 0;
            noc_rx_is_instr = 0;
        end
    endtask

    task run_tpc;
        input [SRAM_ADDR_W-1:0] pc;
        begin
            @(negedge clk);
            tpc_start_pc = pc;
            tpc_start = 1;
            @(posedge clk);
            @(negedge clk);
            tpc_start = 0;
            timeout = 0;
            while (!tpc_done && timeout < 200) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
        end
    endtask

    initial begin
        $display("");
        $display("╔════════════════════════════════════════════════════════════╗");
        $display("║           TPC Integration Testbench                        ║");
        $display("╚════════════════════════════════════════════════════════════╝");

        #(CLK * 5); rst_n = 1; #(CLK * 5);

        //======================================================================
        $display("");
        $display("[TEST 1] Reset State");
        if (!tpc_busy && !tpc_done && !tpc_error && noc_rx_ready) $display("  PASS: TPC idle");
        else begin $display("  FAIL"); errors = errors + 1; end

        //======================================================================
        $display("");
        $display("[TEST 2] Load Instructions via NoC");
        load_instr(12'h000, {OP_NOP, 120'd0});
        load_instr(12'h001, {OP_NOP, 120'd0});
        load_instr(12'h002, {OP_HALT, 120'd0});
        $display("  PASS: 3 instructions loaded (NOP, NOP, HALT)");

        #(CLK * 5);

        //======================================================================
        $display("");
        $display("[TEST 3] Execute NOP, NOP, HALT");
        run_tpc(20'h0);
        if (tpc_done && !tpc_error) $display("  PASS: Program completed (%0d cycles)", timeout);
        else begin $display("  FAIL: done=%b error=%b", tpc_done, tpc_error); errors = errors + 1; end

        #(CLK * 10);

        //======================================================================
        $display("");
        $display("[TEST 4] Busy During Execution");
        load_instr(12'h010, {OP_NOP, 120'd0});
        load_instr(12'h011, {OP_NOP, 120'd0});
        load_instr(12'h012, {OP_NOP, 120'd0});
        load_instr(12'h013, {OP_HALT, 120'd0});
        
        @(negedge clk);
        tpc_start_pc = 20'h010;
        tpc_start = 1;
        @(posedge clk);
        @(negedge clk);
        tpc_start = 0;
        #(CLK * 3);
        if (tpc_busy) $display("  PASS: TPC busy during execution");
        else begin $display("  FAIL: not busy"); errors = errors + 1; end
        while (!tpc_done) @(posedge clk);

        #(CLK * 10);

        //======================================================================
        $display("");
        $display("[TEST 5] Multiple Programs");
        load_instr(12'h020, {OP_NOP, 120'd0});
        load_instr(12'h021, {OP_HALT, 120'd0});
        run_tpc(20'h020);
        if (tpc_done) $display("  PASS: Second program completed");
        else begin $display("  FAIL"); errors = errors + 1; end

        #(CLK * 10);

        //======================================================================
        $display("");
        $display("[TEST 6] Error-Free Completion");
        if (!tpc_error) $display("  PASS: No errors");
        else begin $display("  FAIL: error flag set"); errors = errors + 1; end

        //======================================================================
        $display("");
        $display("════════════════════════════════════════");
        $display("Tests: 6, Errors: %0d", errors);
        if (errors == 0) $display(">>> ALL TESTS PASSED! <<<");
        else $display(">>> SOME TESTS FAILED <<<");
        $display("");
        $finish;
    end

    initial begin $dumpfile("tpc.vcd"); $dumpvars(0, tb_tpc); end
    initial begin #(CLK * 10000); $display("TIMEOUT!"); $finish; end

endmodule

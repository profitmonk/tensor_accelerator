`timescale 1ns / 1ps
module tb_tpc_debug;
    parameter CLK = 10;
    reg clk = 0;
    reg rst_n = 0;
    always #(CLK/2) clk = ~clk;

    reg tpc_start = 0;
    reg [19:0] tpc_start_pc = 0;
    wire tpc_busy, tpc_done, tpc_error;
    reg global_sync_in = 0;
    wire sync_request;
    reg sync_grant = 0;
    reg [255:0] noc_rx_data = 0;
    reg [19:0] noc_rx_addr = 0;
    reg noc_rx_valid = 0;
    wire noc_rx_ready;
    reg noc_rx_is_instr = 0;
    wire [255:0] noc_tx_data;
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
        .ARRAY_SIZE(4), .SRAM_WIDTH(256), .SRAM_ADDR_W(20),
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

    integer i;
    initial begin
        $display("TPC Debug");
        #(CLK*3); rst_n = 1; #(CLK*2);
        
        $display("Loading HALT instruction at addr 0...");
        noc_rx_addr = 20'd0;
        noc_rx_data = {128'd0, 8'h07, 8'h00, 112'd0};  // CTRL.HALT
        noc_rx_is_instr = 1;
        noc_rx_valid = 1;
        @(posedge clk);
        while (!noc_rx_ready) @(posedge clk);
        @(negedge clk);
        noc_rx_valid = 0;
        noc_rx_is_instr = 0;
        
        #(CLK * 5);
        
        $display("Starting TPC at PC=0...");
        tpc_start_pc = 20'd0;
        tpc_start = 1;
        @(posedge clk);
        @(negedge clk);
        tpc_start = 0;
        
        for (i = 0; i < 50; i = i + 1) begin
            @(posedge clk);
            #1;
            $display("Cyc %2d: busy=%b done=%b err=%b lcp_st=%0d imem_re=%b imem_valid=%b",
                     i, tpc_busy, tpc_done, tpc_error, 
                     dut.lcp_inst.state, dut.lcp_imem_re, dut.lcp_imem_valid);
            if (tpc_done) begin
                $display("DONE!");
                $finish;
            end
        end
        $display("No completion");
        $finish;
    end
endmodule

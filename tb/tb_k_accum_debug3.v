`timescale 1ns / 1ps
module tb_k_accum_debug3;
    parameter CLK = 10;
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
    localparam VOP_LOAD  = 8'h30;
    localparam SYNC_VPU  = 8'h02;

    integer i;

    initial begin
        $display("VPU LOAD Debug Test - Check SRAM Read Path");

        // Put test data in SRAM at address 0x0020
        // addr=0x20=32: bank = 32[1:0] XOR 32[9:8] = 0 XOR 0 = 0, word = 32[9:2] = 8
        dut.sram_inst.bank_gen[0].bank_inst.mem[8] = 256'hDEADBEEF;

        // Program: Just VPU LOAD v0 from 0x0020
        dut.instr_mem[0] = {OP_VECTOR, VOP_LOAD, 5'd0, 5'd0, 5'd0, 1'b0, 20'h00020, 12'd0, 16'd4, 16'd0, 32'd0};
        dut.instr_mem[1] = {OP_SYNC, SYNC_VPU, 112'd0};
        dut.instr_mem[2] = {OP_HALT, 120'd0};

        rst_n = 0; #(CLK*5); rst_n = 1; #(CLK*5);

        @(negedge clk);
        tpc_start = 1;
        @(posedge clk); @(posedge clk);
        @(negedge clk);
        tpc_start = 0;

        // Monitor VPU and SRAM
        for (i = 0; i < 40; i = i + 1) begin
            @(posedge clk);
            $display("[%0d] vpu_state=%d vpu_addr=%h vpu_re=%b vpu_ready=%b vpu_rdata=%h", 
                i, dut.vpu_inst.state, dut.vpu_sram_addr, dut.vpu_sram_re, dut.vpu_sram_ready,
                dut.vpu_sram_rdata[31:0]);
            if (tpc_done) begin
                $display("Done");
                i = 1000;
            end
        end

        #(CLK*5);
        $display("VRF v0 = %h", dut.vpu_inst.vrf[0]);
        
        $finish;
    end

    initial begin #(CLK*5000); $display("TIMEOUT"); $finish; end
endmodule

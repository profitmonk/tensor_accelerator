`timescale 1ns / 1ps
module tb_k_accum_debug2;
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
    localparam VOP_ADD   = 8'h01;
    localparam VOP_LOAD  = 8'h30;
    localparam VOP_STORE = 8'h31;
    localparam SYNC_VPU  = 8'h02;

    integer i;
    reg [255:0] ones_row;

    initial begin
        $display("VPU LOAD/ADD/STORE Debug Test");

        ones_row = {224'd0, 8'd1, 8'd1, 8'd1, 8'd1};
        
        // Put test data in SRAM at address 0x0020 (P0) and 0x0024 (P1)
        // addr=0x20=32: bank = 32[1:0] XOR 32[9:8] = 0 XOR 0 = 0, word = 32[9:2] = 8
        dut.sram_inst.bank_gen[0].bank_inst.mem[8] = {224'd0, 32'd4};  // P0 row 0
        dut.sram_inst.bank_gen[1].bank_inst.mem[8] = {224'd0, 32'd4};
        dut.sram_inst.bank_gen[2].bank_inst.mem[8] = {224'd0, 32'd4};
        dut.sram_inst.bank_gen[3].bank_inst.mem[8] = {224'd0, 32'd4};
        
        // addr=0x24=36: bank = 36[1:0] XOR 36[9:8] = 0 XOR 0 = 0, word = 36[9:2] = 9
        dut.sram_inst.bank_gen[0].bank_inst.mem[9] = {224'd0, 32'd4};  // P1 row 0
        dut.sram_inst.bank_gen[1].bank_inst.mem[9] = {224'd0, 32'd4};
        dut.sram_inst.bank_gen[2].bank_inst.mem[9] = {224'd0, 32'd4};
        dut.sram_inst.bank_gen[3].bank_inst.mem[9] = {224'd0, 32'd4};

        // Program: VPU LOAD v0, VPU LOAD v1, VPU ADD, VPU STORE
        i = 0;
        dut.instr_mem[i] = {OP_VECTOR, VOP_LOAD, 5'd0, 5'd0, 5'd0, 1'b0, 20'h00020, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_VECTOR, VOP_LOAD, 5'd1, 5'd0, 5'd0, 1'b0, 20'h00024, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_VECTOR, VOP_ADD, 5'd0, 5'd0, 5'd1, 1'b0, 20'h00000, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_VECTOR, VOP_STORE, 5'd0, 5'd0, 5'd0, 1'b0, 20'h00030, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_HALT, 120'd0};

        rst_n = 0; #(CLK*5); rst_n = 1; #(CLK*5);

        @(negedge clk);
        tpc_start = 1;
        @(posedge clk); @(posedge clk);
        @(negedge clk);
        tpc_start = 0;

        // Monitor VPU SRAM interface
        for (i = 0; i < 100; i = i + 1) begin
            @(posedge clk);
            if (dut.vpu_sram_we || dut.vpu_sram_re) begin
                $display("[%0d] VPU SRAM: addr=%h we=%b re=%b wdata=%h", 
                    i, dut.vpu_sram_addr, dut.vpu_sram_we, dut.vpu_sram_re, dut.vpu_sram_wdata[31:0]);
            end
            if (tpc_done) begin
                $display("Done at cycle %d", i);
                i = 1000;
            end
        end

        #(CLK*5);
        
        // Check result at 0x0030, word 12
        $display("Result at SRAM[0x30] (bank0, word12): %h", dut.sram_inst.bank_gen[0].bank_inst.mem[12][31:0]);
        $display("VRF v0: %h", dut.vpu_inst.vrf[0][31:0]);
        
        $finish;
    end

    initial begin #(CLK*5000); $display("TIMEOUT"); $finish; end
endmodule

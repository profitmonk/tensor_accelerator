`timescale 1ns / 1ps
module tb_e2e_debug;
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

    integer i;

    initial begin
        $display("E2E Debug - Write Verification");
        
        dut.sram_inst.bank_gen[0].bank_inst.mem[0] = {224'd0, 8'd0, 8'd0, 8'd0, 8'd1};
        dut.sram_inst.bank_gen[1].bank_inst.mem[0] = {224'd0, 8'd0, 8'd0, 8'd1, 8'd0};
        dut.sram_inst.bank_gen[2].bank_inst.mem[0] = {224'd0, 8'd0, 8'd1, 8'd0, 8'd0};
        dut.sram_inst.bank_gen[3].bank_inst.mem[0] = {224'd0, 8'd1, 8'd0, 8'd0, 8'd0};
        dut.sram_inst.bank_gen[0].bank_inst.mem[4] = {224'd0, 8'd4, 8'd3, 8'd2, 8'd1};
        dut.sram_inst.bank_gen[1].bank_inst.mem[4] = {224'd0, 8'd8, 8'd7, 8'd6, 8'd5};
        dut.sram_inst.bank_gen[2].bank_inst.mem[4] = {224'd0, 8'd12,8'd11,8'd10,8'd9};
        dut.sram_inst.bank_gen[3].bank_inst.mem[4] = {224'd0, 8'd16,8'd15,8'd14,8'd13};

        dut.instr_mem[0] = {OP_TENSOR, 8'h01, 16'h0020, 16'h0010, 16'h0000, 16'd4, 16'd4, 16'd4, 16'd0};
        dut.instr_mem[1] = {OP_HALT, 120'd0};

        rst_n = 0; #(CLK*5); rst_n = 1; #(CLK*5);

        @(negedge clk);
        tpc_start = 1;
        @(posedge clk); @(posedge clk);
        @(negedge clk);
        tpc_start = 0;

        for (i = 0; i < 40; i = i + 1) begin
            @(posedge clk);
            #1;
            if (dut.mxu_o_we) begin
                $display("Cyc %2d: WRITE addr=%h data=[%d,%d,%d,%d]", i, dut.mxu_o_addr,
                    $signed(dut.mxu_o_wdata[31:0]),
                    $signed(dut.mxu_o_wdata[63:32]),
                    $signed(dut.mxu_o_wdata[95:64]),
                    $signed(dut.mxu_o_wdata[127:96]));
            end
            if (dut.systolic_result_valid && !dut.mxu_o_we) begin
                $display("Cyc %2d: VALID but NOT WRITING! mxu_st=%d out_cnt=%d", 
                    i, dut.mxu_state, dut.mxu_out_cnt);
            end
            if (tpc_done) begin
                $display("DONE at cycle %d", i);
                i = 999;
            end
        end
        
        $display("\nMemory:");
        $display("  bank0[8]=%h", dut.sram_inst.bank_gen[0].bank_inst.mem[8]);
        $display("  bank1[8]=%h", dut.sram_inst.bank_gen[1].bank_inst.mem[8]);
        $display("  bank2[8]=%h", dut.sram_inst.bank_gen[2].bank_inst.mem[8]);
        $display("  bank3[8]=%h", dut.sram_inst.bank_gen[3].bank_inst.mem[8]);
        $finish;
    end
endmodule

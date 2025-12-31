`timescale 1ns / 1ps
module tb_single_tpc_dma_debug;

    parameter CLK = 10;
    reg clk = 0;
    reg rst_n = 0;
    always #(CLK/2) clk = ~clk;

    // Stub signals for simple AXI-Lite control
    reg [11:0] s_axi_ctrl_awaddr = 0;
    reg s_axi_ctrl_awvalid = 0;
    wire s_axi_ctrl_awready;
    reg [31:0] s_axi_ctrl_wdata = 0;
    reg [3:0] s_axi_ctrl_wstrb = 4'hF;
    reg s_axi_ctrl_wvalid = 0;
    wire s_axi_ctrl_wready;
    wire [1:0] s_axi_ctrl_bresp;
    wire s_axi_ctrl_bvalid;
    reg s_axi_ctrl_bready = 1;
    reg [11:0] s_axi_ctrl_araddr = 0;
    reg s_axi_ctrl_arvalid = 0;
    wire s_axi_ctrl_arready;
    wire [31:0] s_axi_ctrl_rdata;
    wire [1:0] s_axi_ctrl_rresp;
    wire s_axi_ctrl_rvalid;
    reg s_axi_ctrl_rready = 1;

    // AXI Memory
    wire [3:0] m_axi_awid, m_axi_arid;
    wire [39:0] m_axi_awaddr, m_axi_araddr;
    wire [7:0] m_axi_awlen, m_axi_arlen;
    wire [2:0] m_axi_awsize, m_axi_arsize;
    wire [1:0] m_axi_awburst, m_axi_arburst;
    wire m_axi_awvalid, m_axi_arvalid;
    wire m_axi_awready, m_axi_arready;
    wire [255:0] m_axi_wdata, m_axi_rdata;
    wire [31:0] m_axi_wstrb;
    wire m_axi_wlast, m_axi_wvalid, m_axi_wready;
    wire [3:0] m_axi_bid, m_axi_rid;
    wire [1:0] m_axi_bresp, m_axi_rresp;
    wire m_axi_bvalid, m_axi_bready;
    wire m_axi_rlast, m_axi_rvalid, m_axi_rready;
    wire irq;

    tensor_accelerator_top #(
        .GRID_X(2), .GRID_Y(2), .ARRAY_SIZE(4),
        .SRAM_BANKS(4), .SRAM_DEPTH(256), .VPU_LANES(16)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .s_axi_ctrl_awaddr(s_axi_ctrl_awaddr), .s_axi_ctrl_awvalid(s_axi_ctrl_awvalid),
        .s_axi_ctrl_awready(s_axi_ctrl_awready),
        .s_axi_ctrl_wdata(s_axi_ctrl_wdata), .s_axi_ctrl_wstrb(s_axi_ctrl_wstrb),
        .s_axi_ctrl_wvalid(s_axi_ctrl_wvalid), .s_axi_ctrl_wready(s_axi_ctrl_wready),
        .s_axi_ctrl_bresp(s_axi_ctrl_bresp), .s_axi_ctrl_bvalid(s_axi_ctrl_bvalid),
        .s_axi_ctrl_bready(s_axi_ctrl_bready),
        .s_axi_ctrl_araddr(s_axi_ctrl_araddr), .s_axi_ctrl_arvalid(s_axi_ctrl_arvalid),
        .s_axi_ctrl_arready(s_axi_ctrl_arready),
        .s_axi_ctrl_rdata(s_axi_ctrl_rdata), .s_axi_ctrl_rresp(s_axi_ctrl_rresp),
        .s_axi_ctrl_rvalid(s_axi_ctrl_rvalid), .s_axi_ctrl_rready(s_axi_ctrl_rready),
        .m_axi_awid(m_axi_awid), .m_axi_awaddr(m_axi_awaddr), .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize), .m_axi_awburst(m_axi_awburst),
        .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata), .m_axi_wstrb(m_axi_wstrb), .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid), .m_axi_bresp(m_axi_bresp), .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready),
        .m_axi_arid(m_axi_arid), .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst),
        .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_rid(m_axi_rid), .m_axi_rdata(m_axi_rdata), .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast), .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready),
        .irq(irq)
    );

    axi_memory_model #(
        .AXI_ADDR_WIDTH(40), .AXI_DATA_WIDTH(256), .AXI_ID_WIDTH(4),
        .MEM_SIZE_MB(1), .READ_LATENCY(2), .WRITE_LATENCY(1)
    ) axi_mem (
        .clk(clk), .rst_n(rst_n),
        .s_axi_awid(m_axi_awid), .s_axi_awaddr(m_axi_awaddr), .s_axi_awlen(m_axi_awlen),
        .s_axi_awsize(m_axi_awsize), .s_axi_awburst(m_axi_awburst),
        .s_axi_awvalid(m_axi_awvalid), .s_axi_awready(m_axi_awready),
        .s_axi_wdata(m_axi_wdata), .s_axi_wstrb(m_axi_wstrb), .s_axi_wlast(m_axi_wlast),
        .s_axi_wvalid(m_axi_wvalid), .s_axi_wready(m_axi_wready),
        .s_axi_bid(m_axi_bid), .s_axi_bresp(m_axi_bresp), .s_axi_bvalid(m_axi_bvalid),
        .s_axi_bready(m_axi_bready),
        .s_axi_arid(m_axi_arid), .s_axi_araddr(m_axi_araddr), .s_axi_arlen(m_axi_arlen),
        .s_axi_arsize(m_axi_arsize), .s_axi_arburst(m_axi_arburst),
        .s_axi_arvalid(m_axi_arvalid), .s_axi_arready(m_axi_arready),
        .s_axi_rid(m_axi_rid), .s_axi_rdata(m_axi_rdata), .s_axi_rresp(m_axi_rresp),
        .s_axi_rlast(m_axi_rlast), .s_axi_rvalid(m_axi_rvalid), .s_axi_rready(m_axi_rready)
    );

    localparam OP_DMA = 8'h03, OP_SYNC = 8'h04, OP_HALT = 8'hFF;
    localparam DMA_LOAD = 8'h01, SYNC_DMA = 8'h03;
    
    function [127:0] make_dma_cmd;
        input [7:0] subop;
        input [39:0] ext_addr;
        input [19:0] int_addr;
        input [11:0] rows, cols, ext_stride, int_stride;
        begin
            make_dma_cmd = {OP_DMA, subop, ext_addr, int_addr, rows, cols, ext_stride, int_stride, 4'd0};
        end
    endfunction

    reg [31:0] rdata;
    integer i;

    task axi_write(input [11:0] addr, input [31:0] data);
        begin
            @(negedge clk);
            s_axi_ctrl_awaddr = addr; s_axi_ctrl_awvalid = 1;
            s_axi_ctrl_wdata = data; s_axi_ctrl_wvalid = 1;
            @(posedge clk);
            while (!s_axi_ctrl_awready || !s_axi_ctrl_wready) @(posedge clk);
            @(negedge clk);
            s_axi_ctrl_awvalid = 0; s_axi_ctrl_wvalid = 0;
            while (!s_axi_ctrl_bvalid) @(posedge clk);
            @(posedge clk);
        end
    endtask

    task axi_read(input [11:0] addr);
        begin
            @(negedge clk);
            s_axi_ctrl_araddr = addr; s_axi_ctrl_arvalid = 1;
            @(posedge clk);
            while (!s_axi_ctrl_arready) @(posedge clk);
            @(negedge clk); s_axi_ctrl_arvalid = 0;
            while (!s_axi_ctrl_rvalid) @(posedge clk);
            rdata = s_axi_ctrl_rdata;
            @(posedge clk);
        end
    endtask

    initial begin
        $display("Single TPC DMA Debug Test");
        axi_mem.mem[0] = 256'hDEADBEEF;
        
        rst_n = 0; #(CLK*5); rst_n = 1; #(CLK*5);

        // Load program
        dut.tpc_gen[0].tpc_inst.instr_mem[0] = make_dma_cmd(DMA_LOAD, 40'h0, 20'h100, 12'd1, 12'd1, 12'd32, 12'd32);
        dut.tpc_gen[0].tpc_inst.instr_mem[1] = {OP_SYNC, SYNC_DMA, 112'd0};
        dut.tpc_gen[0].tpc_inst.instr_mem[2] = {OP_HALT, 120'd0};
        dut.tpc_gen[1].tpc_inst.instr_mem[0] = {OP_HALT, 120'd0};
        dut.tpc_gen[2].tpc_inst.instr_mem[0] = {OP_HALT, 120'd0};
        dut.tpc_gen[3].tpc_inst.instr_mem[0] = {OP_HALT, 120'd0};

        // Start
        axi_write(12'h000, 32'h00000F00);
        axi_write(12'h000, 32'h00000F01);

        // Debug loop
        for (i = 0; i < 100; i = i + 1) begin
            @(posedge clk);
            if (i < 20 || i % 10 == 0) begin
                $display("[%0d] arvalid=%b arready=%b active=%b pending=%b active_tpc=%d", 
                    i, m_axi_arvalid, m_axi_arready, 
                    dut.axi_transaction_active, dut.pending, dut.active_tpc);
                $display("      TPC0_arvalid=%b dma_state=%d lcp_state=%d",
                    dut.tpc_gen[0].tpc_inst.dma_inst.axi_arvalid,
                    dut.tpc_gen[0].tpc_inst.dma_inst.state,
                    dut.tpc_gen[0].tpc_inst.lcp_inst.state);
            end
            if (dut.gcp_inst.tpc_done == 4'hF) begin
                $display("All done at cycle %d", i);
                i = 1000;
            end
        end
        
        $finish;
    end

    initial begin #(CLK*2000); $display("TIMEOUT"); $finish; end
endmodule

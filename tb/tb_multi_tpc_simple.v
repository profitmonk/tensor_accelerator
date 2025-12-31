`timescale 1ns / 1ps
//==============================================================================
// Simplified Multi-TPC Test - Just run HALT on all 4 TPCs
//==============================================================================

module tb_multi_tpc_simple;

    parameter CLK = 10;
    parameter ARRAY_SIZE = 4;
    parameter DATA_WIDTH = 256;
    parameter NUM_TPCS = 4;

    reg clk = 0;
    reg rst_n = 0;
    always #(CLK/2) clk = ~clk;

    // AXI-Lite Control
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

    // AXI Memory (stub - tie off)
    wire [3:0] m_axi_awid;
    wire [39:0] m_axi_awaddr;
    wire [7:0] m_axi_awlen;
    wire [2:0] m_axi_awsize;
    wire [1:0] m_axi_awburst;
    wire m_axi_awvalid;
    wire m_axi_awready = 1'b1;
    wire [255:0] m_axi_wdata;
    wire [31:0] m_axi_wstrb;
    wire m_axi_wlast;
    wire m_axi_wvalid;
    wire m_axi_wready = 1'b1;
    wire [3:0] m_axi_bid = 4'd0;
    wire [1:0] m_axi_bresp = 2'd0;
    wire m_axi_bvalid = 1'b0;
    wire m_axi_bready;
    wire [3:0] m_axi_arid;
    wire [39:0] m_axi_araddr;
    wire [7:0] m_axi_arlen;
    wire [2:0] m_axi_arsize;
    wire [1:0] m_axi_arburst;
    wire m_axi_arvalid;
    wire m_axi_arready = 1'b1;
    wire [3:0] m_axi_rid = 4'd0;
    wire [255:0] m_axi_rdata = 256'd0;
    wire [1:0] m_axi_rresp = 2'd0;
    wire m_axi_rlast = 1'b0;
    wire m_axi_rvalid = 1'b0;
    wire m_axi_rready;

    wire irq;

    tensor_accelerator_top #(
        .GRID_X(2), .GRID_Y(2),
        .ARRAY_SIZE(ARRAY_SIZE),
        .SRAM_BANKS(4), .SRAM_DEPTH(256),
        .VPU_LANES(16)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .s_axi_ctrl_awaddr(s_axi_ctrl_awaddr),
        .s_axi_ctrl_awvalid(s_axi_ctrl_awvalid),
        .s_axi_ctrl_awready(s_axi_ctrl_awready),
        .s_axi_ctrl_wdata(s_axi_ctrl_wdata),
        .s_axi_ctrl_wstrb(s_axi_ctrl_wstrb),
        .s_axi_ctrl_wvalid(s_axi_ctrl_wvalid),
        .s_axi_ctrl_wready(s_axi_ctrl_wready),
        .s_axi_ctrl_bresp(s_axi_ctrl_bresp),
        .s_axi_ctrl_bvalid(s_axi_ctrl_bvalid),
        .s_axi_ctrl_bready(s_axi_ctrl_bready),
        .s_axi_ctrl_araddr(s_axi_ctrl_araddr),
        .s_axi_ctrl_arvalid(s_axi_ctrl_arvalid),
        .s_axi_ctrl_arready(s_axi_ctrl_arready),
        .s_axi_ctrl_rdata(s_axi_ctrl_rdata),
        .s_axi_ctrl_rresp(s_axi_ctrl_rresp),
        .s_axi_ctrl_rvalid(s_axi_ctrl_rvalid),
        .s_axi_ctrl_rready(s_axi_ctrl_rready),
        .m_axi_awid(m_axi_awid),
        .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst),
        .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata),
        .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid),
        .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready),
        .m_axi_arid(m_axi_arid),
        .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst),
        .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rid(m_axi_rid),
        .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready),
        .irq(irq)
    );

    localparam ADDR_CTRL   = 12'h000;
    localparam ADDR_STATUS = 12'h004;
    localparam OP_HALT = 8'hFF;
    localparam OP_NOP = 8'h00;

    reg [31:0] rdata;
    integer timeout;

    task axi_write(input [11:0] addr, input [31:0] data);
        begin
            @(negedge clk);
            s_axi_ctrl_awaddr = addr;
            s_axi_ctrl_awvalid = 1;
            s_axi_ctrl_wdata = data;
            s_axi_ctrl_wvalid = 1;
            @(posedge clk);
            while (!s_axi_ctrl_awready || !s_axi_ctrl_wready) @(posedge clk);
            @(negedge clk);
            s_axi_ctrl_awvalid = 0;
            s_axi_ctrl_wvalid = 0;
            while (!s_axi_ctrl_bvalid) @(posedge clk);
            @(posedge clk);
        end
    endtask

    task axi_read(input [11:0] addr);
        begin
            @(negedge clk);
            s_axi_ctrl_araddr = addr;
            s_axi_ctrl_arvalid = 1;
            @(posedge clk);
            while (!s_axi_ctrl_arready) @(posedge clk);
            @(negedge clk);
            s_axi_ctrl_arvalid = 0;
            while (!s_axi_ctrl_rvalid) @(posedge clk);
            rdata = s_axi_ctrl_rdata;
            @(posedge clk);
        end
    endtask

    reg success;

    initial begin
        $display("");
        $display("╔══════════════════════════════════════════════════════════════╗");
        $display("║        Simplified Multi-TPC Test (NOP + HALT)                ║");
        $display("╚══════════════════════════════════════════════════════════════╝");
        $display("");

        rst_n = 0;
        #(CLK * 5);
        rst_n = 1;
        #(CLK * 5);

        // Load simple programs: NOP, NOP, HALT
        $display("[SETUP] Loading programs...");
        dut.tpc_gen[0].tpc_inst.instr_mem[0] = {OP_NOP, 120'd0};
        dut.tpc_gen[0].tpc_inst.instr_mem[1] = {OP_NOP, 120'd0};
        dut.tpc_gen[0].tpc_inst.instr_mem[2] = {OP_HALT, 120'd0};
        
        dut.tpc_gen[1].tpc_inst.instr_mem[0] = {OP_NOP, 120'd0};
        dut.tpc_gen[1].tpc_inst.instr_mem[1] = {OP_HALT, 120'd0};
        
        dut.tpc_gen[2].tpc_inst.instr_mem[0] = {OP_HALT, 120'd0};
        
        dut.tpc_gen[3].tpc_inst.instr_mem[0] = {OP_NOP, 120'd0};
        dut.tpc_gen[3].tpc_inst.instr_mem[1] = {OP_NOP, 120'd0};
        dut.tpc_gen[3].tpc_inst.instr_mem[2] = {OP_NOP, 120'd0};
        dut.tpc_gen[3].tpc_inst.instr_mem[3] = {OP_HALT, 120'd0};
        $display("  Programs loaded");

        // Start all TPCs
        $display("[EXEC] Starting all 4 TPCs...");
        axi_write(ADDR_CTRL, 32'h00000F00);  // Enable all
        axi_write(ADDR_CTRL, 32'h00000F01);  // Start

        // Wait for completion
        timeout = 0;
        success = 0;
        while (timeout < 100 && !success) begin
            @(posedge clk);
            axi_read(ADDR_STATUS);
            if (rdata[11:8] == 4'b1111) success = 1;
            timeout = timeout + 1;
        end

        if (success) begin
            $display("  All TPCs completed in %0d cycles", timeout);
            $display(">>> MULTI-TPC SIMPLE TEST PASSED! <<<");
        end else begin
            $display("  TIMEOUT: status=%h", rdata);
            $display(">>> MULTI-TPC SIMPLE TEST FAILED <<<");
        end

        #(CLK * 10);
        $finish;
    end

    initial begin
        #(CLK * 5000);
        $display("TIMEOUT!");
        $finish;
    end

endmodule

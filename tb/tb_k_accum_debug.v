`timescale 1ns / 1ps
module tb_k_accum_debug;
    parameter CLK = 10;
    reg clk = 0;
    reg rst_n = 0;
    always #(CLK/2) clk = ~clk;

    // Minimal AXI-Lite stub
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
    wire m_axi_awvalid, m_axi_arvalid, m_axi_awready, m_axi_arready;
    wire [255:0] m_axi_wdata, m_axi_rdata;
    wire [31:0] m_axi_wstrb;
    wire m_axi_wlast, m_axi_wvalid, m_axi_wready;
    wire [3:0] m_axi_bid, m_axi_rid;
    wire [1:0] m_axi_bresp, m_axi_rresp;
    wire m_axi_bvalid, m_axi_bready, m_axi_rlast, m_axi_rvalid, m_axi_rready;
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

    localparam OP_VECTOR = 8'h02;
    localparam OP_SYNC   = 8'h04;
    localparam OP_HALT   = 8'hFF;
    localparam VOP_ADD   = 8'h01;
    localparam VOP_LOAD  = 8'h30;
    localparam VOP_STORE = 8'h31;
    localparam SYNC_VPU  = 8'h02;

    function [127:0] make_vpu_cmd;
        input [7:0] subop;
        input [4:0] vd, vs1, vs2;
        input [19:0] mem_addr;
        input [15:0] count;
        begin
            make_vpu_cmd = {OP_VECTOR, subop, vd, vs1, vs2, 1'b0, mem_addr, 12'd0, count, 16'd0, 32'd0};
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
        $display("K-Accumulation Debug: Testing VPU LOAD/ADD/STORE");
        
        rst_n = 0; #(CLK*5); rst_n = 1; #(CLK*5);

        // Initialize SRAM with test data directly
        // Put values 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 at address 0x200
        dut.tpc_gen[0].tpc_inst.sram_inst.bank_gen[0].bank_inst.mem[16] = 
            {224'd0, 32'h04030201};  // Elements 0-3 in first bank word
        dut.tpc_gen[0].tpc_inst.sram_inst.bank_gen[1].bank_inst.mem[16] = 
            {224'd0, 32'h08070605};
        dut.tpc_gen[0].tpc_inst.sram_inst.bank_gen[2].bank_inst.mem[16] = 
            {224'd0, 32'h0c0b0a09};
        dut.tpc_gen[0].tpc_inst.sram_inst.bank_gen[3].bank_inst.mem[16] = 
            {224'd0, 32'h100f0e0d};
            
        // Put values 10,20,30,... at address 0x300
        dut.tpc_gen[0].tpc_inst.sram_inst.bank_gen[0].bank_inst.mem[24] = 
            {224'd0, 32'h28201e14};  // 20,30,32,40
        dut.tpc_gen[0].tpc_inst.sram_inst.bank_gen[1].bank_inst.mem[24] = 
            {224'd0, 32'h50483c32};
        dut.tpc_gen[0].tpc_inst.sram_inst.bank_gen[2].bank_inst.mem[24] = 
            {224'd0, 32'h78706450};
        dut.tpc_gen[0].tpc_inst.sram_inst.bank_gen[3].bank_inst.mem[24] = 
            {224'd0, 32'ha0988c64};

        // Simple program: VPU LOAD v0 from 0x200, VPU LOAD v1 from 0x300, VPU ADD, VPU STORE
        dut.tpc_gen[0].tpc_inst.instr_mem[0] = make_vpu_cmd(VOP_LOAD, 5'd0, 5'd0, 5'd0, 20'h0200, 16'd16);
        dut.tpc_gen[0].tpc_inst.instr_mem[1] = {OP_SYNC, SYNC_VPU, 112'd0};
        dut.tpc_gen[0].tpc_inst.instr_mem[2] = make_vpu_cmd(VOP_LOAD, 5'd1, 5'd0, 5'd0, 20'h0300, 16'd16);
        dut.tpc_gen[0].tpc_inst.instr_mem[3] = {OP_SYNC, SYNC_VPU, 112'd0};
        dut.tpc_gen[0].tpc_inst.instr_mem[4] = make_vpu_cmd(VOP_ADD, 5'd0, 5'd0, 5'd1, 20'h0000, 16'd16);
        dut.tpc_gen[0].tpc_inst.instr_mem[5] = {OP_SYNC, SYNC_VPU, 112'd0};
        dut.tpc_gen[0].tpc_inst.instr_mem[6] = make_vpu_cmd(VOP_STORE, 5'd0, 5'd0, 5'd0, 20'h0400, 16'd16);
        dut.tpc_gen[0].tpc_inst.instr_mem[7] = {OP_SYNC, SYNC_VPU, 112'd0};
        dut.tpc_gen[0].tpc_inst.instr_mem[8] = {OP_HALT, 120'd0};
        
        dut.tpc_gen[1].tpc_inst.instr_mem[0] = {OP_HALT, 120'd0};
        dut.tpc_gen[2].tpc_inst.instr_mem[0] = {OP_HALT, 120'd0};
        dut.tpc_gen[3].tpc_inst.instr_mem[0] = {OP_HALT, 120'd0};

        // Start
        axi_write(12'h000, 32'h00000F00);
        axi_write(12'h000, 32'h00000F01);

        // Debug loop
        for (i = 0; i < 100; i = i + 1) begin
            @(posedge clk);
            if (i < 30 || i % 10 == 0) begin
                $display("[%0d] lcp_state=%d vpu_state=%d vpu_done=%b", 
                    i, 
                    dut.tpc_gen[0].tpc_inst.lcp_inst.state,
                    dut.tpc_gen[0].tpc_inst.vpu_inst.state,
                    dut.tpc_gen[0].tpc_inst.vpu_inst.cmd_done);
            end
            if (dut.gcp_inst.tpc_done == 4'hF) begin
                $display("All done at cycle %d", i);
                i = 1000;
            end
        end
        
        // Check result at 0x400
        $display("Result at SRAM[0x400]: bank0=%h", 
            dut.tpc_gen[0].tpc_inst.sram_inst.bank_gen[0].bank_inst.mem[32]);
        
        $finish;
    end

    initial begin #(CLK*5000); $display("TIMEOUT"); $finish; end
endmodule

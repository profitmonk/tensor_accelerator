`timescale 1ns / 1ps
module tb_gcp_debug;
    parameter CLK = 10;
    reg clk = 0;
    reg rst_n = 0;
    always #(CLK/2) clk = ~clk;

    reg [11:0] s_axi_awaddr = 0, s_axi_araddr = 0;
    reg s_axi_awvalid = 0, s_axi_arvalid = 0;
    wire s_axi_awready, s_axi_arready;
    reg [31:0] s_axi_wdata = 0;
    reg [3:0] s_axi_wstrb = 4'hF;
    reg s_axi_wvalid = 0;
    wire s_axi_wready;
    wire [1:0] s_axi_bresp, s_axi_rresp;
    wire s_axi_bvalid, s_axi_rvalid;
    reg s_axi_bready = 1, s_axi_rready = 1;
    wire [31:0] s_axi_rdata;

    wire [3:0] tpc_start;
    wire [19:0] tpc_start_pc [0:3];
    reg [3:0] tpc_busy = 0, tpc_done = 0, tpc_error = 0;
    wire global_sync_out;
    reg [3:0] sync_request = 0;
    wire [3:0] sync_grant;
    wire irq;

    global_cmd_processor #(.NUM_TPCS(4)) dut (
        .clk(clk), .rst_n(rst_n),
        .s_axi_awaddr(s_axi_awaddr), .s_axi_awvalid(s_axi_awvalid), .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata), .s_axi_wstrb(s_axi_wstrb), .s_axi_wvalid(s_axi_wvalid), .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp), .s_axi_bvalid(s_axi_bvalid), .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr), .s_axi_arvalid(s_axi_arvalid), .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata), .s_axi_rresp(s_axi_rresp), .s_axi_rvalid(s_axi_rvalid), .s_axi_rready(s_axi_rready),
        .tpc_start(tpc_start), .tpc_start_pc(tpc_start_pc),
        .tpc_busy(tpc_busy), .tpc_done(tpc_done), .tpc_error(tpc_error),
        .global_sync_out(global_sync_out), .sync_request(sync_request), .sync_grant(sync_grant),
        .irq(irq)
    );

    integer i;
    initial begin
        $display("GCP Debug");
        #(CLK*3); rst_n = 1; #(CLK*2);
        
        $display("Initial: tpc_enable=%h, tpc_start=%b", dut.tpc_enable, tpc_start);
        
        // Write to control register: enable all + start
        $display("Writing CTRL = 0x0000FF01");
        s_axi_awaddr = 12'h000;
        s_axi_wdata = 32'h0000FF01;
        s_axi_awvalid = 1;
        s_axi_wvalid = 1;
        
        for (i = 0; i < 10; i = i + 1) begin
            @(posedge clk);
            #1;
            $display("Cyc %d: awready=%b wready=%b bvalid=%b state=%d start=%b pulse=%b enable=%h",
                     i, s_axi_awready, s_axi_wready, s_axi_bvalid, dut.axi_state,
                     tpc_start, dut.global_start_pulse, dut.tpc_enable);
            
            if (s_axi_awready && s_axi_wready) begin
                s_axi_awvalid <= 0;
                s_axi_wvalid <= 0;
            end
        end
        
        $finish;
    end
endmodule

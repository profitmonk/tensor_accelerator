`timescale 1ns / 1ps
module tb_both_debug;
    parameter CLK = 10;
    reg clk = 0;
    always #(CLK/2) clk = ~clk;
    
    // GCP Test
    reg rst_n = 0;
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

    global_cmd_processor #(.NUM_TPCS(4)) gcp (
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
    
    task axi_write;
        input [11:0] addr;
        input [31:0] data;
        begin
            @(negedge clk);
            s_axi_awaddr = addr; s_axi_wdata = data;
            s_axi_awvalid = 1; s_axi_wvalid = 1;
            @(posedge clk);
            while (!(s_axi_awready && s_axi_wready)) @(posedge clk);
            @(negedge clk);
            s_axi_awvalid = 0; s_axi_wvalid = 0;
            while (!s_axi_bvalid) @(posedge clk);
            @(posedge clk);
        end
    endtask

    initial begin
        $display("GCP Barrier Debug");
        #(CLK*3); rst_n = 1; #(CLK*2);
        
        // Enable all TPCs
        axi_write(12'h000, 32'h00000F00);
        
        $display("Testing barrier sync with tpc_enable=%h", gcp.tpc_enable);
        
        sync_request = 4'b1111;
        
        for (i = 0; i < 10; i = i + 1) begin
            @(posedge clk);
            #1;
            $display("Cyc %d: sync_req=%b all_sync=%b barrier=%b grant=%b sync_out=%b",
                     i, sync_request, gcp.all_sync_requested, gcp.barrier_active, sync_grant, global_sync_out);
        end
        
        $finish;
    end
endmodule

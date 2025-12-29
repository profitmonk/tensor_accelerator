`timescale 1ns / 1ps
module tb_dma_store_debug;
    parameter CLK = 10;
    reg clk = 0;
    reg rst_n = 0;
    always #(CLK/2) clk = ~clk;

    reg [127:0] cmd = 0;
    reg cmd_valid = 0;
    wire cmd_ready, cmd_done;
    wire [19:0] sram_addr;
    wire [255:0] sram_wdata;
    reg [255:0] sram_rdata = 0;
    wire sram_we, sram_re;
    reg sram_ready = 1;
    
    wire [39:0] axi_awaddr;
    wire [7:0] axi_awlen;
    wire axi_awvalid;
    reg axi_awready = 1;
    wire [255:0] axi_wdata;
    wire axi_wlast, axi_wvalid;
    reg axi_wready = 1;
    reg [1:0] axi_bresp = 0;
    reg axi_bvalid = 0;
    wire axi_bready;
    
    // Stub read channel
    wire [39:0] axi_araddr; wire [7:0] axi_arlen; wire axi_arvalid;
    reg axi_arready = 1;
    reg [255:0] axi_rdata = 0; reg axi_rlast = 0, axi_rvalid = 0;
    wire axi_rready;

    dma_engine dut (
        .clk(clk), .rst_n(rst_n),
        .cmd(cmd), .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_done(cmd_done),
        .sram_addr(sram_addr), .sram_wdata(sram_wdata), .sram_rdata(sram_rdata),
        .sram_we(sram_we), .sram_re(sram_re), .sram_ready(sram_ready),
        .axi_awaddr(axi_awaddr), .axi_awlen(axi_awlen), .axi_awvalid(axi_awvalid), .axi_awready(axi_awready),
        .axi_wdata(axi_wdata), .axi_wlast(axi_wlast), .axi_wvalid(axi_wvalid), .axi_wready(axi_wready),
        .axi_bresp(axi_bresp), .axi_bvalid(axi_bvalid), .axi_bready(axi_bready),
        .axi_araddr(axi_araddr), .axi_arlen(axi_arlen), .axi_arvalid(axi_arvalid), .axi_arready(axi_arready),
        .axi_rdata(axi_rdata), .axi_rlast(axi_rlast), .axi_rvalid(axi_rvalid), .axi_rready(axi_rready)
    );

    reg [255:0] sram_mem [0:63];
    reg [255:0] ext_mem [0:63];
    reg [39:0] captured_awaddr;
    reg [255:0] captured_wdata;
    
    // SRAM model
    always @(posedge clk) begin
        if (sram_re) begin
            sram_rdata <= sram_mem[sram_addr[7:0]];
            $display("  SRAM read[%d] = %h", sram_addr[7:0], sram_mem[sram_addr[7:0]][31:0]);
        end
    end
    
    // AXI write capture
    always @(posedge clk) begin
        if (axi_awvalid && axi_awready) begin
            captured_awaddr <= axi_awaddr;
            $display("  AXI AW: addr=%h", axi_awaddr);
        end
        if (axi_wvalid && axi_wready) begin
            captured_wdata <= axi_wdata;
            ext_mem[captured_awaddr[10:5]] <= axi_wdata;
            $display("  AXI W: data=%h, stored at ext[%d]", axi_wdata[31:0], captured_awaddr[10:5]);
            if (axi_wlast) axi_bvalid <= 1;
        end
        if (axi_bvalid && axi_bready) axi_bvalid <= 0;
    end

    integer i;
    initial begin
        $display("DMA STORE Debug");
        
        // Init SRAM with test data
        sram_mem[10] = 256'hCAFEBABE_DEADBEEF;
        for (i = 0; i < 64; i = i + 1) ext_mem[i] = 0;
        
        #(CLK*3); rst_n = 1; #(CLK*2);
        
        // Issue STORE: sram[10] -> ext[20]
        // ext_addr = 20*32 = 640 = 0x280
        // int_addr = 10*32 = 320 = 0x140
        cmd = 0;
        cmd[127:120] = 8'h03;    // DMA opcode
        cmd[119:112] = 8'h02;    // STORE
        cmd[111:72] = 40'h280;   // ext_addr
        cmd[71:52] = 20'h140;    // int_addr
        cmd[51:40] = 12'd1;      // rows
        cmd[39:28] = 12'd1;      // cols
        
        $display("Command: STORE ext_addr=%h int_addr=%h", cmd[111:72], cmd[71:52]);
        
        cmd_valid = 1;
        
        for (i = 0; i < 30; i = i + 1) begin
            @(posedge clk);
            #1;
            $display("Cyc %2d: st=%2d sram_addr=%h sram_re=%b awvalid=%b wvalid=%b done=%b",
                     i, dut.state, sram_addr, sram_re, axi_awvalid, axi_wvalid, cmd_done);
            
            if (cmd_ready && cmd_valid) cmd_valid = 0;
            if (cmd_done) begin
                $display("STORE complete");
                $display("  ext_mem[20] = %h", ext_mem[20][31:0]);
                $display("  sram_mem[10] = %h", sram_mem[10][31:0]);
                if (ext_mem[20] == sram_mem[10]) $display("PASS");
                else $display("FAIL: mismatch");
                $finish;
            end
        end
        $display("TIMEOUT");
        $finish;
    end
endmodule

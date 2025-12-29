`timescale 1ns / 1ps
module tb_dma_debug;
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
    
    wire [39:0] axi_araddr;
    wire [7:0] axi_arlen;
    wire axi_arvalid;
    reg axi_arready = 1;
    reg [255:0] axi_rdata = 256'hDEADBEEF;
    reg axi_rlast = 0, axi_rvalid = 0;
    wire axi_rready;
    
    // Stub write channel
    wire [39:0] axi_awaddr; wire [7:0] axi_awlen; wire axi_awvalid;
    reg axi_awready = 1; wire [255:0] axi_wdata; wire axi_wlast, axi_wvalid;
    reg axi_wready = 1; reg [1:0] axi_bresp = 0; reg axi_bvalid = 0; wire axi_bready;

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

    // AXI slave model with burst support
    reg [7:0] burst_len = 0;
    reg [7:0] beat_count = 0;
    reg burst_active = 0;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            burst_active <= 0; axi_rvalid <= 0; axi_rlast <= 0;
        end else begin
            // Accept AR
            if (axi_arvalid && axi_arready && !burst_active) begin
                burst_active <= 1;
                burst_len <= axi_arlen;
                beat_count <= 0;
                $display("     -> Burst started, len=%d", axi_arlen);
            end
            
            // Provide data
            if (burst_active && !axi_rvalid) begin
                axi_rvalid <= 1;
                axi_rdata <= {beat_count, 248'h0};
                axi_rlast <= (beat_count >= burst_len);
            end
            
            // Handshake
            if (axi_rvalid && axi_rready) begin
                if (beat_count >= burst_len) begin
                    burst_active <= 0;
                    axi_rvalid <= 0;
                    axi_rlast <= 0;
                    $display("     -> Burst complete");
                end else begin
                    beat_count <= beat_count + 1;
                    axi_rdata <= {beat_count + 8'd1, 248'h0};
                    axi_rlast <= (beat_count + 1 >= burst_len);
                end
            end
        end
    end

    integer i;
    initial begin
        $display("DMA Debug - Burst Transfer (cols=2)");
        #(CLK*3); rst_n = 1; #(CLK*2);
        
        // Issue LOAD command: 1 row, 2 cols (should be burst of 2)
        cmd = 0;
        cmd[127:120] = 8'h03;
        cmd[119:112] = 8'h01;    // LOAD
        cmd[111:72] = 40'h0;
        cmd[71:52] = 20'h0;
        cmd[51:40] = 12'd1;      // rows = 1
        cmd[39:28] = 12'd2;      // cols = 2 (burst of 2)
        
        cmd_valid = 1;
        
        for (i = 0; i < 50; i = i + 1) begin
            @(posedge clk);
            #1;
            $display("Cyc %2d: st=%2d rvalid=%b rready=%b rlast=%b burst_cnt=%d col_cnt=%d sram_we=%b done=%b",
                     i, dut.state, axi_rvalid, axi_rready, axi_rlast, dut.burst_count, dut.col_count, sram_we, cmd_done);
            
            if (cmd_ready && cmd_valid) cmd_valid = 0;
            if (cmd_done) begin
                $display("DONE at cycle %d", i);
                $finish;
            end
        end
        $display("TIMEOUT - state=%d", dut.state);
        $finish;
    end
endmodule

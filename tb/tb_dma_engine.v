//==============================================================================
// DMA Engine Unit Testbench - Simplified
//==============================================================================
`timescale 1ns / 1ps

module tb_dma_engine;
    parameter CLK = 10;
    parameter DATA_WIDTH = 256;

    reg clk = 0;
    reg rst_n = 0;
    always #(CLK/2) clk = ~clk;

    reg [127:0] cmd = 0;
    reg cmd_valid = 0;
    wire cmd_ready, cmd_done;

    wire [19:0] sram_addr;
    wire [DATA_WIDTH-1:0] sram_wdata;
    reg [DATA_WIDTH-1:0] sram_rdata = 0;
    wire sram_we, sram_re;
    reg sram_ready = 1;

    // AXI signals
    wire [39:0] axi_awaddr, axi_araddr;
    wire [7:0] axi_awlen, axi_arlen;
    wire axi_awvalid, axi_arvalid, axi_wvalid, axi_wlast, axi_rready;
    wire [DATA_WIDTH-1:0] axi_wdata;
    wire axi_bready;
    
    reg axi_awready = 1, axi_arready = 1, axi_wready = 1;
    reg [1:0] axi_bresp = 0;
    reg axi_bvalid = 0;
    reg [DATA_WIDTH-1:0] axi_rdata = 0;
    reg axi_rlast = 0, axi_rvalid = 0;

    dma_engine #(.DATA_WIDTH(DATA_WIDTH)) dut (
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

    localparam DMA_LOAD = 8'h01, DMA_STORE = 8'h02;

    // External memory model
    reg [DATA_WIDTH-1:0] ext_mem [0:15];
    reg [DATA_WIDTH-1:0] sram_mem [0:15];
    
    integer errors = 0, i, timeout;
    reg [7:0] beat_count;
    
    // AXI Read response - simplified single-beat
    always @(posedge clk) begin
        if (axi_arvalid && axi_arready) begin
            // Immediate response for simplicity
            axi_rvalid <= 1;
            axi_rdata <= ext_mem[axi_araddr[7:4]];
            axi_rlast <= 1;
        end else if (axi_rvalid && axi_rready) begin
            axi_rvalid <= 0;
            axi_rlast <= 0;
        end
    end
    
    // AXI Write response
    always @(posedge clk) begin
        if (axi_wvalid && axi_wready && axi_wlast) begin
            axi_bvalid <= 1;
        end else if (axi_bvalid && axi_bready) begin
            axi_bvalid <= 0;
        end
    end
    
    // SRAM model
    always @(posedge clk) begin
        if (sram_we) begin
            sram_mem[sram_addr[7:4]] <= sram_wdata;
            $display("  SRAM[%h] <= %h", sram_addr[7:4], sram_wdata[31:0]);
        end
        if (sram_re) begin
            sram_rdata <= sram_mem[sram_addr[7:4]];
        end
    end

    initial begin
        $display("");
        $display("╔════════════════════════════════════════════════════════════╗");
        $display("║           DMA Engine Unit Testbench                        ║");
        $display("╚════════════════════════════════════════════════════════════╝");

        // Init external memory with test data
        for (i = 0; i < 16; i = i + 1) begin
            ext_mem[i] = {8{i[31:0] + 32'hAA000000}};
            sram_mem[i] = 0;
        end

        #(CLK * 5); rst_n = 1; #(CLK * 5);

        //======================================================================
        // TEST 1: Command acceptance
        //======================================================================
        $display("");
        $display("[TEST 1] Command Interface");
        
        if (cmd_ready) $display("  PASS: DMA ready for commands");
        else begin $display("  FAIL: DMA not ready"); errors = errors + 1; end

        //======================================================================
        // TEST 2: LOAD command (1x1)
        //======================================================================
        $display("");
        $display("[TEST 2] LOAD: External[0] → SRAM[0]");
        
        // Command: opcode=03, subop=LOAD, ext_addr=0, int_addr=0, rows=1, cols=1
        cmd = 0;
        cmd[127:120] = 8'h03;       // DMA opcode
        cmd[119:112] = DMA_LOAD;
        cmd[111:72] = 40'h0;        // ext_addr
        cmd[71:52] = 20'h0;         // int_addr
        cmd[51:40] = 12'd1;         // rows
        cmd[39:28] = 12'd1;         // cols
        cmd[27:16] = 12'd32;        // src_stride
        cmd[15:4] = 12'd32;         // dst_stride
        
        cmd_valid = 1;
        @(posedge clk);
        while (!cmd_ready) @(posedge clk);
        cmd_valid = 0;
        
        timeout = 0;
        while (!cmd_done && timeout < 100) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        
        if (cmd_done) begin
            $display("  PASS: LOAD completed in %0d cycles", timeout);
        end else begin
            $display("  FAIL: LOAD timeout (state=%0d)", dut.state);
            errors = errors + 1;
        end

        //======================================================================
        // TEST 3: State machine returns to IDLE
        //======================================================================
        $display("");
        $display("[TEST 3] State Machine Reset");
        
        #(CLK * 5);
        if (dut.state == 0 && cmd_ready) $display("  PASS: Back to IDLE");
        else begin $display("  FAIL: state=%0d, ready=%b", dut.state, cmd_ready); errors = errors + 1; end

        //======================================================================
        // Summary
        //======================================================================
        #(CLK * 10);
        $display("");
        $display("════════════════════════════════════════");
        $display("Tests: 3, Errors: %0d", errors);
        if (errors == 0) $display(">>> ALL TESTS PASSED! <<<");
        else $display(">>> SOME TESTS FAILED <<<");
        $display("");
        $finish;
    end

    initial begin $dumpfile("dma.vcd"); $dumpvars(0, tb_dma_engine); end
    initial begin #(CLK * 5000); $display("TIMEOUT!"); $finish; end
endmodule

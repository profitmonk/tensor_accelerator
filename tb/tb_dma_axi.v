`timescale 1ns / 1ps
//==============================================================================
// DMA Engine Test with AXI Memory Model
// Tests LOAD and STORE paths with proper external memory simulation
//==============================================================================
module tb_dma_axi;
    parameter CLK = 10;
    parameter DATA_WIDTH = 256;
    parameter EXT_ADDR_W = 40;
    parameter INT_ADDR_W = 20;

    reg clk = 0;
    reg rst_n = 0;
    always #(CLK/2) clk = ~clk;

    //==========================================================================
    // DMA Command Interface
    //==========================================================================
    reg [127:0] cmd = 0;
    reg cmd_valid = 0;
    wire cmd_ready, cmd_done;

    //==========================================================================
    // SRAM Interface (behavioral model matching RTL timing)
    //==========================================================================
    wire [INT_ADDR_W-1:0] sram_addr;
    wire [DATA_WIDTH-1:0] sram_wdata;
    reg  [DATA_WIDTH-1:0] sram_rdata;
    wire sram_we, sram_re;
    wire sram_ready;
    
    // Simple SRAM model with 1-cycle read latency (matching RTL)
    reg [DATA_WIDTH-1:0] sram_mem [0:255];
    wire [7:0] sram_word = sram_addr[12:5];  // Word address
    
    // Registered read output (1-cycle latency, matches sram_bank)
    always @(posedge clk) begin
        if (sram_we) sram_mem[sram_word] <= sram_wdata;
        if (sram_re) sram_rdata <= sram_mem[sram_word];
    end
    
    assign sram_ready = 1'b1;  // Always ready (no arbitration in this test)

    //==========================================================================
    // AXI Interface
    //==========================================================================
    wire [EXT_ADDR_W-1:0] axi_awaddr, axi_araddr;
    wire [7:0] axi_awlen, axi_arlen;
    wire axi_awvalid, axi_arvalid;
    wire axi_awready, axi_arready;
    wire [DATA_WIDTH-1:0] axi_wdata;
    wire axi_wlast, axi_wvalid;
    wire axi_wready;
    wire [1:0] axi_bresp;
    wire axi_bvalid;
    wire axi_bready;
    wire [DATA_WIDTH-1:0] axi_rdata;
    wire axi_rlast, axi_rvalid;
    wire axi_rready;

    //==========================================================================
    // DUT: DMA Engine
    //==========================================================================
    dma_engine #(
        .EXT_ADDR_W(EXT_ADDR_W),
        .INT_ADDR_W(INT_ADDR_W),
        .DATA_WIDTH(DATA_WIDTH),
        .MAX_BURST(16)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .cmd(cmd), .cmd_valid(cmd_valid),
        .cmd_ready(cmd_ready), .cmd_done(cmd_done),
        .sram_addr(sram_addr), .sram_wdata(sram_wdata),
        .sram_rdata(sram_rdata), .sram_we(sram_we), .sram_re(sram_re),
        .sram_ready(sram_ready),
        .axi_awaddr(axi_awaddr), .axi_awlen(axi_awlen),
        .axi_awvalid(axi_awvalid), .axi_awready(axi_awready),
        .axi_wdata(axi_wdata), .axi_wlast(axi_wlast),
        .axi_wvalid(axi_wvalid), .axi_wready(axi_wready),
        .axi_bresp(axi_bresp), .axi_bvalid(axi_bvalid), .axi_bready(axi_bready),
        .axi_araddr(axi_araddr), .axi_arlen(axi_arlen),
        .axi_arvalid(axi_arvalid), .axi_arready(axi_arready),
        .axi_rdata(axi_rdata), .axi_rlast(axi_rlast),
        .axi_rvalid(axi_rvalid), .axi_rready(axi_rready)
    );

    //==========================================================================
    // AXI Memory Model
    //==========================================================================
    axi_memory_model #(
        .AXI_ADDR_WIDTH(EXT_ADDR_W),
        .AXI_DATA_WIDTH(DATA_WIDTH),
        .AXI_ID_WIDTH(4),
        .MEM_SIZE_MB(1),
        .READ_LATENCY(2),
        .WRITE_LATENCY(1)
    ) axi_mem (
        .clk(clk), .rst_n(rst_n),
        // Write address
        .s_axi_awid(4'd0), .s_axi_awaddr(axi_awaddr),
        .s_axi_awlen(axi_awlen), .s_axi_awsize(3'd5), // 32 bytes
        .s_axi_awburst(2'b01), .s_axi_awvalid(axi_awvalid),
        .s_axi_awready(axi_awready),
        // Write data
        .s_axi_wdata(axi_wdata), .s_axi_wstrb({32{1'b1}}),
        .s_axi_wlast(axi_wlast), .s_axi_wvalid(axi_wvalid),
        .s_axi_wready(axi_wready),
        // Write response
        .s_axi_bid(), .s_axi_bresp(axi_bresp),
        .s_axi_bvalid(axi_bvalid), .s_axi_bready(axi_bready),
        // Read address
        .s_axi_arid(4'd0), .s_axi_araddr(axi_araddr),
        .s_axi_arlen(axi_arlen), .s_axi_arsize(3'd5),
        .s_axi_arburst(2'b01), .s_axi_arvalid(axi_arvalid),
        .s_axi_arready(axi_arready),
        // Read data
        .s_axi_rid(), .s_axi_rdata(axi_rdata),
        .s_axi_rresp(), .s_axi_rlast(axi_rlast),
        .s_axi_rvalid(axi_rvalid), .s_axi_rready(axi_rready)
    );

    //==========================================================================
    // Test Helper Functions
    //==========================================================================
    localparam DMA_LOAD = 8'h01, DMA_STORE = 8'h02;
    localparam OP_DMA = 8'h03;

    function [127:0] make_dma_cmd;
        input [7:0] subop;
        input [39:0] ext_addr;
        input [19:0] int_addr;
        input [11:0] rows, cols;
        input [11:0] ext_stride, int_stride;
        begin
            make_dma_cmd = {OP_DMA, subop, ext_addr, int_addr, rows, cols, 
                           ext_stride, int_stride, 4'd0};
        end
    endfunction

    task issue_cmd(input [127:0] c);
        begin
            @(negedge clk);
            cmd = c;
            cmd_valid = 1;
            @(posedge clk);
            while (!cmd_ready) @(posedge clk);
            @(negedge clk);
            cmd_valid = 0;
            // Wait for completion
            while (!cmd_done) @(posedge clk);
            @(posedge clk);
        end
    endtask

    //==========================================================================
    // Main Test
    //==========================================================================
    integer i, errors, test_errors;
    reg [DATA_WIDTH-1:0] expected, actual;

    initial begin
        $display("");
        $display("╔════════════════════════════════════════════════════════════╗");
        $display("║            DMA Engine + AXI Memory Test                    ║");
        $display("╚════════════════════════════════════════════════════════════╝");
        
        errors = 0;

        // Initialize memories
        for (i = 0; i < 256; i = i + 1) begin
            sram_mem[i] = 256'd0;
        end

        // Initialize AXI memory with test patterns
        for (i = 0; i < 16; i = i + 1) begin
            axi_mem.mem[i] = {224'd0, 32'hDEAD0000 + i};
        end

        rst_n = 0; #(CLK*5); rst_n = 1; #(CLK*5);

        //======================================================================
        // TEST 1: Single-word DMA LOAD
        //======================================================================
        $display("");
        $display("[TEST 1] DMA LOAD: 1 word from external to SRAM");
        
        issue_cmd(make_dma_cmd(DMA_LOAD, 40'h0, 20'h100, 12'd1, 12'd1, 12'd32, 12'd32));
        
        expected = {224'd0, 32'hDEAD0000};
        actual = sram_mem[8];
        
        if (actual == expected) begin
            $display("  PASS: SRAM[8] = 0x%h", actual[31:0]);
        end else begin
            $display("  FAIL: SRAM[8] = 0x%h, expected 0x%h", actual[31:0], expected[31:0]);
            errors = errors + 1;
        end

        //======================================================================
        // TEST 2: Multi-word DMA LOAD
        //======================================================================
        $display("");
        $display("[TEST 2] DMA LOAD: 4 words from external to SRAM");
        
        issue_cmd(make_dma_cmd(DMA_LOAD, 40'h0, 20'h200, 12'd1, 12'd4, 12'd32, 12'd32));
        
        test_errors = 0;
        for (i = 0; i < 4; i = i + 1) begin
            expected = {224'd0, 32'hDEAD0000 + i};
            actual = sram_mem[16 + i];
            if (actual[31:0] != expected[31:0]) begin
                $display("  FAIL: SRAM[%0d] = 0x%h, expected 0x%h", 16+i, actual[31:0], expected[31:0]);
                test_errors = test_errors + 1;
            end
        end
        if (test_errors == 0) $display("  PASS: 4 words loaded correctly");
        else errors = errors + test_errors;

        //======================================================================
        // TEST 3: DMA STORE - Single word
        //======================================================================
        $display("");
        $display("[TEST 3] DMA STORE: 1 word from SRAM to external");
        
        sram_mem[32] = {224'd0, 32'hBEEF1234};
        
        issue_cmd(make_dma_cmd(DMA_STORE, 40'h1000, 20'h400, 12'd1, 12'd1, 12'd32, 12'd32));
        
        expected = {224'd0, 32'hBEEF1234};
        actual = axi_mem.mem[128];
        
        if (actual[31:0] == expected[31:0]) begin
            $display("  PASS: EXT_MEM[128] = 0x%h", actual[31:0]);
        end else begin
            $display("  FAIL: EXT_MEM[128] = 0x%h, expected 0x%h", actual[31:0], expected[31:0]);
            errors = errors + 1;
        end

        //======================================================================
        // TEST 4: DMA STORE - Multiple words
        //======================================================================
        $display("");
        $display("[TEST 4] DMA STORE: 4 words from SRAM to external");
        
        for (i = 0; i < 4; i = i + 1) begin
            sram_mem[40 + i] = {224'd0, 32'hCAFE0000 + i};
        end
        
        issue_cmd(make_dma_cmd(DMA_STORE, 40'h2000, 20'h500, 12'd1, 12'd4, 12'd32, 12'd32));
        
        test_errors = 0;
        for (i = 0; i < 4; i = i + 1) begin
            expected = {224'd0, 32'hCAFE0000 + i};
            actual = axi_mem.mem[256 + i];
            if (actual[31:0] != expected[31:0]) begin
                $display("  FAIL: EXT_MEM[%0d] = 0x%h, expected 0x%h", 256+i, actual[31:0], expected[31:0]);
                test_errors = test_errors + 1;
            end
        end
        if (test_errors == 0) $display("  PASS: 4 words stored correctly");
        else errors = errors + test_errors;

        //======================================================================
        // TEST 5: Round-trip (LOAD → STORE)
        //======================================================================
        $display("");
        $display("[TEST 5] Round-trip: LOAD from ext, then STORE back");
        
        axi_mem.mem[512] = 256'd0;
        
        issue_cmd(make_dma_cmd(DMA_LOAD, 40'h0, 20'h600, 12'd1, 12'd1, 12'd32, 12'd32));
        issue_cmd(make_dma_cmd(DMA_STORE, 40'h4000, 20'h600, 12'd1, 12'd1, 12'd32, 12'd32));
        
        expected = {224'd0, 32'hDEAD0000};
        actual = axi_mem.mem[512];
        
        if (actual[31:0] == expected[31:0]) begin
            $display("  PASS: Round-trip data = 0x%h", actual[31:0]);
        end else begin
            $display("  FAIL: Round-trip data = 0x%h, expected 0x%h", actual[31:0], expected[31:0]);
            errors = errors + 1;
        end

        //======================================================================
        // TEST 6: 2D Transfer (2 rows x 2 cols)
        //======================================================================
        $display("");
        $display("[TEST 6] 2D LOAD: 2 rows x 2 cols");
        
        axi_mem.mem[0] = {224'd0, 32'hAA000000};
        axi_mem.mem[1] = {224'd0, 32'hAA000001};
        axi_mem.mem[2] = {224'd0, 32'hAA000010};
        axi_mem.mem[3] = {224'd0, 32'hAA000011};
        
        issue_cmd(make_dma_cmd(DMA_LOAD, 40'h0, 20'h700, 12'd2, 12'd2, 12'd64, 12'd64));
        
        if (sram_mem[56][31:0] == 32'hAA000000 &&
            sram_mem[57][31:0] == 32'hAA000001 &&
            sram_mem[58][31:0] == 32'hAA000010 &&
            sram_mem[59][31:0] == 32'hAA000011) begin
            $display("  PASS: 2x2 data loaded correctly");
        end else begin
            $display("  FAIL: 2D load mismatch");
            $display("    SRAM[56]=0x%h (expect AA000000)", sram_mem[56][31:0]);
            $display("    SRAM[57]=0x%h (expect AA000001)", sram_mem[57][31:0]);
            $display("    SRAM[58]=0x%h (expect AA000010)", sram_mem[58][31:0]);
            $display("    SRAM[59]=0x%h (expect AA000011)", sram_mem[59][31:0]);
            errors = errors + 1;
        end

        //======================================================================
        // Summary
        //======================================================================
        $display("");
        $display("════════════════════════════════════════════════════════════");
        $display("DMA TEST SUMMARY: 6 tests, %0d errors", errors);
        $display("════════════════════════════════════════════════════════════");
        
        if (errors == 0) 
            $display(">>> ALL DMA TESTS PASSED! <<<");
        else 
            $display(">>> DMA TESTS FAILED <<<");
        
        $finish;
    end

    initial begin #(CLK * 50000); $display("TIMEOUT"); $finish; end

endmodule

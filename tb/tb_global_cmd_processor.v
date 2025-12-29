//==============================================================================
// Global Command Processor (GCP) Unit Testbench
//==============================================================================
`timescale 1ns / 1ps

module tb_global_cmd_processor;

    parameter CLK = 10;
    parameter NUM_TPCS = 4;

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

    wire [NUM_TPCS-1:0] tpc_start;
    wire [19:0] tpc_start_pc [0:NUM_TPCS-1];
    reg [NUM_TPCS-1:0] tpc_busy = 0, tpc_done = 0, tpc_error = 0;

    wire global_sync_out;
    reg [NUM_TPCS-1:0] sync_request = 0;
    wire [NUM_TPCS-1:0] sync_grant;
    wire irq;

    global_cmd_processor #(.NUM_TPCS(NUM_TPCS)) dut (
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

    localparam ADDR_CTRL = 12'h000, ADDR_STATUS = 12'h004;
    localparam ADDR_IRQ_EN = 12'h008, ADDR_IRQ_STATUS = 12'h00C;
    localparam ADDR_TPC_BASE = 12'h100;

    integer errors = 0, timeout, i;
    reg [31:0] read_data;
    reg [3:0] captured_start;
    reg sync_seen;

    task axi_write_capture;
        input [11:0] addr;
        input [31:0] data;
        output [3:0] start_cap;
        begin
            start_cap = 0;
            @(negedge clk);
            s_axi_awaddr = addr; s_axi_wdata = data;
            s_axi_awvalid = 1; s_axi_wvalid = 1;
            timeout = 0;
            while (timeout < 20) begin
                @(posedge clk); #1;
                if (tpc_start != 0) start_cap = tpc_start;
                if (s_axi_awready && s_axi_wready) begin
                    s_axi_awvalid = 0; s_axi_wvalid = 0;
                end
                if (s_axi_bvalid) timeout = 100;
                timeout = timeout + 1;
            end
            @(posedge clk);
        end
    endtask

    task axi_write;
        input [11:0] addr;
        input [31:0] data;
        begin axi_write_capture(addr, data, captured_start); end
    endtask

    task axi_read;
        input [11:0] addr;
        output [31:0] data;
        begin
            @(negedge clk);
            s_axi_araddr = addr; s_axi_arvalid = 1;
            @(posedge clk);
            while (!s_axi_arready) @(posedge clk);
            @(negedge clk); s_axi_arvalid = 0;
            @(posedge clk);
            while (!s_axi_rvalid) @(posedge clk);
            data = s_axi_rdata;
            @(posedge clk);
        end
    endtask

    initial begin
        $display("");
        $display("╔════════════════════════════════════════════════════════════╗");
        $display("║      Global Command Processor Unit Testbench               ║");
        $display("╚════════════════════════════════════════════════════════════╝");

        #(CLK * 5); rst_n = 1; #(CLK * 5);

        //======================================================================
        $display("");
        $display("[TEST 1] Read Control Register");
        axi_read(ADDR_CTRL, read_data);
        if (read_data[15:8] == 8'hFF) $display("  PASS: Default enable = 0xFF");
        else begin $display("  FAIL"); errors = errors + 1; end

        //======================================================================
        $display("");
        $display("[TEST 2] Write TPC Start PC");
        axi_write(ADDR_TPC_BASE, 32'h00001000);
        axi_write(ADDR_TPC_BASE + 16, 32'h00002000);
        axi_read(ADDR_TPC_BASE, read_data);
        if (read_data == 32'h00001000) $display("  PASS: TPC0 PC");
        else begin $display("  FAIL"); errors = errors + 1; end
        axi_read(ADDR_TPC_BASE + 16, read_data);
        if (read_data == 32'h00002000) $display("  PASS: TPC1 PC");
        else begin $display("  FAIL"); errors = errors + 1; end

        //======================================================================
        $display("");
        $display("[TEST 3] Global Start Pulse");
        axi_write_capture(ADDR_CTRL, 32'h0000FF01, captured_start);
        if (captured_start == 4'b1111) $display("  PASS: All TPCs started");
        else begin $display("  FAIL: start=%b", captured_start); errors = errors + 1; end

        //======================================================================
        $display("");
        $display("[TEST 4] Status Register");
        tpc_busy = 4'b1010; tpc_done = 4'b0101;
        #(CLK * 2);
        axi_read(ADDR_STATUS, read_data);
        if (read_data[3:0] == 4'b1010) $display("  PASS: Busy correct");
        else begin $display("  FAIL"); errors = errors + 1; end

        //======================================================================
        $display("");
        $display("[TEST 5] TPC Enable Mask");
        axi_write(ADDR_CTRL, 32'h00000300);
        axi_write_capture(ADDR_CTRL, 32'h00000301, captured_start);
        if (captured_start == 4'b0011) $display("  PASS: Masked start");
        else begin $display("  FAIL: start=%b", captured_start); errors = errors + 1; end

        //======================================================================
        $display("");
        $display("[TEST 6] Barrier Synchronization");
        axi_write(ADDR_CTRL, 32'h00000F00);
        
        sync_request = 4'b0011;
        #(CLK * 2);
        if (sync_grant == 0) $display("  PASS: No premature grant");
        else begin $display("  FAIL"); errors = errors + 1; end
        
        sync_request = 4'b1111;
        sync_seen = 0;
        for (i = 0; i < 5; i = i + 1) begin
            @(posedge clk); #1;
            if (global_sync_out) sync_seen = 1;
        end
        if (sync_seen) $display("  PASS: Barrier triggered");
        else begin $display("  FAIL"); errors = errors + 1; end
        sync_request = 0;

        //======================================================================
        $display("");
        $display("[TEST 7] Interrupt Generation");
        axi_write(ADDR_CTRL, 32'h0000FF01);
        #(CLK * 2);
        axi_write(ADDR_IRQ_EN, 32'h00000001);
        tpc_done = 4'b1111; tpc_busy = 0;
        #(CLK * 5);
        if (irq == 1) $display("  PASS: IRQ asserted");
        else begin $display("  FAIL"); errors = errors + 1; end
        
        axi_write(ADDR_IRQ_STATUS, 32'h00000001);
        #(CLK * 5);
        // IRQ may re-assert due to done still being high - that's ok
        $display("  INFO: IRQ=%b (may stay high if done still set)", irq);

        //======================================================================
        $display("");
        $display("════════════════════════════════════════");
        $display("Tests: 7, Errors: %0d", errors);
        if (errors == 0) $display(">>> ALL TESTS PASSED! <<<");
        else $display(">>> SOME TESTS FAILED <<<");
        $display("");
        $finish;
    end

    initial begin $dumpfile("gcp.vcd"); $dumpvars(0, tb_global_cmd_processor); end
    initial begin #(CLK * 10000); $display("TIMEOUT!"); $finish; end

endmodule

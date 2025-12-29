//==============================================================================
// Tensor Accelerator Top-Level Integration Testbench
//==============================================================================
`timescale 1ns / 1ps

module tb_top;

    parameter CLK = 10;
    parameter NUM_TPCS = 4;

    reg clk = 0;
    reg rst_n = 0;
    always #(CLK/2) clk = ~clk;

    // AXI-Lite Control Interface
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

    // AXI Memory Interface (stub)
    wire [3:0] m_axi_awid;
    wire [39:0] m_axi_awaddr;
    wire [7:0] m_axi_awlen;
    wire [2:0] m_axi_awsize;
    wire [1:0] m_axi_awburst;
    wire m_axi_awvalid;
    reg m_axi_awready = 1;
    wire [255:0] m_axi_wdata;
    wire [31:0] m_axi_wstrb;
    wire m_axi_wlast;
    wire m_axi_wvalid;
    reg m_axi_wready = 1;
    reg [3:0] m_axi_bid = 0;
    reg [1:0] m_axi_bresp = 0;
    reg m_axi_bvalid = 0;
    wire m_axi_bready;
    wire [3:0] m_axi_arid;
    wire [39:0] m_axi_araddr;
    wire [7:0] m_axi_arlen;
    wire [2:0] m_axi_arsize;
    wire [1:0] m_axi_arburst;
    wire m_axi_arvalid;
    reg m_axi_arready = 1;
    reg [3:0] m_axi_rid = 0;
    reg [255:0] m_axi_rdata = 0;
    reg [1:0] m_axi_rresp = 0;
    reg m_axi_rlast = 0;
    reg m_axi_rvalid = 0;
    wire m_axi_rready;

    wire irq;

    // DUT
    tensor_accelerator_top #(
        .GRID_X(2), .GRID_Y(2),
        .ARRAY_SIZE(4),
        .SRAM_BANKS(4), .SRAM_DEPTH(256),
        .VPU_LANES(16)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .s_axi_ctrl_awaddr(s_axi_ctrl_awaddr), .s_axi_ctrl_awvalid(s_axi_ctrl_awvalid), .s_axi_ctrl_awready(s_axi_ctrl_awready),
        .s_axi_ctrl_wdata(s_axi_ctrl_wdata), .s_axi_ctrl_wstrb(s_axi_ctrl_wstrb), .s_axi_ctrl_wvalid(s_axi_ctrl_wvalid), .s_axi_ctrl_wready(s_axi_ctrl_wready),
        .s_axi_ctrl_bresp(s_axi_ctrl_bresp), .s_axi_ctrl_bvalid(s_axi_ctrl_bvalid), .s_axi_ctrl_bready(s_axi_ctrl_bready),
        .s_axi_ctrl_araddr(s_axi_ctrl_araddr), .s_axi_ctrl_arvalid(s_axi_ctrl_arvalid), .s_axi_ctrl_arready(s_axi_ctrl_arready),
        .s_axi_ctrl_rdata(s_axi_ctrl_rdata), .s_axi_ctrl_rresp(s_axi_ctrl_rresp), .s_axi_ctrl_rvalid(s_axi_ctrl_rvalid), .s_axi_ctrl_rready(s_axi_ctrl_rready),
        .m_axi_awid(m_axi_awid), .m_axi_awaddr(m_axi_awaddr), .m_axi_awlen(m_axi_awlen), .m_axi_awsize(m_axi_awsize), .m_axi_awburst(m_axi_awburst), .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata), .m_axi_wstrb(m_axi_wstrb), .m_axi_wlast(m_axi_wlast), .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid), .m_axi_bresp(m_axi_bresp), .m_axi_bvalid(m_axi_bvalid), .m_axi_bready(m_axi_bready),
        .m_axi_arid(m_axi_arid), .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen), .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst), .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_rid(m_axi_rid), .m_axi_rdata(m_axi_rdata), .m_axi_rresp(m_axi_rresp), .m_axi_rlast(m_axi_rlast), .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready),
        .irq(irq)
    );

    integer errors = 0;
    integer timeout, i;
    reg [31:0] rdata;

    // GCP Register addresses (corrected from RTL)
    // CTRL[0] = start, CTRL[15:8] = tpc_enable
    // STATUS = {15'b0, all_done, 4'b0_error, 4'b0_done, 4'b0_busy}
    localparam ADDR_CTRL       = 12'h000;
    localparam ADDR_STATUS     = 12'h004;
    localparam ADDR_IRQ_EN     = 12'h008;
    localparam ADDR_IRQ_STATUS = 12'h00C;

    // Opcodes
    localparam OP_HALT = 8'hFF;

    //==========================================================================
    // AXI-Lite Tasks
    //==========================================================================
    task axi_write;
        input [11:0] addr;
        input [31:0] data;
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

    task axi_read;
        input [11:0] addr;
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

    //==========================================================================
    // Pre-load instructions into TPC instruction memories
    //==========================================================================
    task preload_instructions;
        begin
            // Load HALT instruction at address 0 for all TPCs
            dut.tpc_gen[0].tpc_inst.instr_mem[0] = {OP_HALT, 120'd0};
            dut.tpc_gen[1].tpc_inst.instr_mem[0] = {OP_HALT, 120'd0};
            dut.tpc_gen[2].tpc_inst.instr_mem[0] = {OP_HALT, 120'd0};
            dut.tpc_gen[3].tpc_inst.instr_mem[0] = {OP_HALT, 120'd0};
            $display("  INFO: HALT instruction preloaded to all TPCs");
        end
    endtask

    //==========================================================================
    // Wait for completion
    //==========================================================================
    task wait_done;
        input [3:0] mask;
        output success;
        begin
            timeout = 0;
            success = 0;
            while (timeout < 200 && !success) begin
                @(posedge clk);
                axi_read(ADDR_STATUS);
                // done is in bits [11:8]
                if ((rdata[11:8] & mask) == mask) success = 1;
                timeout = timeout + 1;
            end
        end
    endtask

    reg success;

    //==========================================================================
    // Test Sequence
    //==========================================================================
    initial begin
        $display("");
        $display("╔════════════════════════════════════════════════════════════╗");
        $display("║        Tensor Accelerator Top-Level Integration Test       ║");
        $display("╚════════════════════════════════════════════════════════════╝");

        #(CLK * 5); rst_n = 1; #(CLK * 5);

        //======================================================================
        $display("");
        $display("[TEST 1] Reset State");
        axi_read(ADDR_STATUS);
        // busy[3:0] should be 0
        if (rdata[3:0] == 4'b0000) $display("  PASS: All TPCs idle (busy=0)");
        else begin $display("  FAIL: status=%h", rdata); errors = errors + 1; end

        //======================================================================
        $display("");
        $display("[TEST 2] GCP CTRL Register");
        // Write tpc_enable = 0x0F, don't start yet
        axi_write(ADDR_CTRL, 32'h00000F00);
        axi_read(ADDR_CTRL);
        // CTRL read returns {16'b0, tpc_enable, 8'b0}
        if (rdata[15:8] == 8'h0F) $display("  PASS: TPC enable = 0x0F");
        else begin $display("  FAIL: expected 0F00, got %h", rdata); errors = errors + 1; end

        //======================================================================
        $display("");
        $display("[TEST 3] Pre-load Instructions");
        preload_instructions;
        $display("  PASS: Instructions loaded");

        //======================================================================
        $display("");
        $display("[TEST 4] Single TPC Execution (TPC0)");
        // Enable only TPC0, then start
        axi_write(ADDR_CTRL, 32'h00000100);  // Enable TPC0 only
        axi_write(ADDR_CTRL, 32'h00000101);  // Start + TPC0 enabled
        
        wait_done(4'b0001, success);
        if (success) $display("  PASS: TPC0 completed (%0d cycles)", timeout);
        else begin $display("  FAIL: TPC0 timeout, status=%h", rdata); errors = errors + 1; end

        #(CLK * 20);

        //======================================================================
        $display("");
        $display("[TEST 5] All TPCs Parallel Execution");
        // Enable all 4 TPCs
        axi_write(ADDR_CTRL, 32'h00000F00);  // Enable all
        axi_write(ADDR_CTRL, 32'h00000F01);  // Start all
        
        wait_done(4'b1111, success);
        if (success) $display("  PASS: All 4 TPCs completed (%0d cycles)", timeout);
        else begin $display("  FAIL: Not all done, status=%h", rdata); errors = errors + 1; end

        #(CLK * 20);

        //======================================================================
        $display("");
        $display("[TEST 6] IRQ Generation");
        axi_write(ADDR_IRQ_EN, 32'h00000001);  // Enable IRQ
        axi_write(ADDR_CTRL, 32'h00000101);    // Start TPC0
        
        timeout = 0;
        while (!irq && timeout < 200) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        if (irq) $display("  PASS: IRQ asserted");
        else begin $display("  FAIL: No IRQ"); errors = errors + 1; end

        // Clear IRQ
        axi_write(ADDR_IRQ_STATUS, 32'h00000001);
        #(CLK * 5);

        //======================================================================
        $display("");
        $display("[TEST 7] Error-Free Execution");
        axi_read(ADDR_STATUS);
        // error is in bits [19:16]
        if (rdata[19:16] == 4'h0) $display("  PASS: No TPC errors");
        else begin $display("  FAIL: Errors=%h", rdata[19:16]); errors = errors + 1; end

        //======================================================================
        $display("");
        $display("════════════════════════════════════════");
        $display("Tests: 7, Errors: %0d", errors);
        if (errors == 0) $display(">>> ALL TESTS PASSED! <<<");
        else $display(">>> SOME TESTS FAILED <<<");
        $display("");
        $finish;
    end

    initial begin $dumpfile("top.vcd"); $dumpvars(0, tb_top); end
    initial begin #(CLK * 50000); $display("TIMEOUT!"); $finish; end

endmodule

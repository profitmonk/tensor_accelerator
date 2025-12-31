`timescale 1ns / 1ps
module tb_batch_debug2;
    parameter CLK = 10;
    parameter SRAM_WIDTH = 256;

    reg clk = 0;
    reg rst_n = 0;
    always #(CLK/2) clk = ~clk;

    reg tpc_start = 0;
    reg [19:0] tpc_start_pc = 0;
    wire tpc_busy, tpc_done, tpc_error;
    reg global_sync_in = 0;
    wire sync_request;
    reg sync_grant = 0;
    reg [SRAM_WIDTH-1:0] noc_rx_data = 0;
    reg [19:0] noc_rx_addr = 0;
    reg noc_rx_valid = 0;
    wire noc_rx_ready;
    reg noc_rx_is_instr = 0;
    wire [SRAM_WIDTH-1:0] noc_tx_data;
    wire [19:0] noc_tx_addr;
    wire noc_tx_valid;
    reg noc_tx_ready = 1;
    wire [39:0] axi_awaddr, axi_araddr;
    wire [7:0] axi_awlen, axi_arlen;
    wire axi_awvalid, axi_arvalid, axi_wvalid, axi_wlast, axi_rready, axi_bready;
    wire [255:0] axi_wdata;
    reg axi_awready = 1, axi_arready = 1, axi_wready = 1;
    reg axi_bvalid = 0, axi_rvalid = 0, axi_rlast = 0;
    reg [1:0] axi_bresp = 0;
    reg [255:0] axi_rdata = 0;

    tensor_processing_cluster #(
        .ARRAY_SIZE(4), .SRAM_WIDTH(SRAM_WIDTH),
        .SRAM_BANKS(4), .SRAM_DEPTH(256), .VPU_LANES(16)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .tpc_start(tpc_start), .tpc_start_pc(tpc_start_pc),
        .tpc_busy(tpc_busy), .tpc_done(tpc_done), .tpc_error(tpc_error),
        .global_sync_in(global_sync_in), .sync_request(sync_request), .sync_grant(sync_grant),
        .noc_rx_data(noc_rx_data), .noc_rx_addr(noc_rx_addr),
        .noc_rx_valid(noc_rx_valid), .noc_rx_ready(noc_rx_ready), .noc_rx_is_instr(noc_rx_is_instr),
        .noc_tx_data(noc_tx_data), .noc_tx_addr(noc_tx_addr),
        .noc_tx_valid(noc_tx_valid), .noc_tx_ready(noc_tx_ready),
        .axi_awaddr(axi_awaddr), .axi_awlen(axi_awlen), .axi_awvalid(axi_awvalid), .axi_awready(axi_awready),
        .axi_wdata(axi_wdata), .axi_wlast(axi_wlast), .axi_wvalid(axi_wvalid), .axi_wready(axi_wready),
        .axi_bresp(axi_bresp), .axi_bvalid(axi_bvalid), .axi_bready(axi_bready),
        .axi_araddr(axi_araddr), .axi_arlen(axi_arlen), .axi_arvalid(axi_arvalid), .axi_arready(axi_arready),
        .axi_rdata(axi_rdata), .axi_rlast(axi_rlast), .axi_rvalid(axi_rvalid), .axi_rready(axi_rready)
    );

    localparam OP_TENSOR = 8'h01;
    localparam OP_HALT   = 8'hFF;

    integer i;

    initial begin
        $display("Batch Debug 2: Check SRAM initialization and MXU addressing");

        // Setup exactly like tb_e2e_simple.v which passes:
        // W at 0x0000, X at 0x0010, instruction: GEMM dst=0x20, src2=0x10, src1=0x00
        
        // Identity weights at 0x0000
        dut.sram_inst.bank_gen[0].bank_inst.mem[0] = {224'd0, 8'd0, 8'd0, 8'd0, 8'd1};
        dut.sram_inst.bank_gen[1].bank_inst.mem[0] = {224'd0, 8'd0, 8'd0, 8'd1, 8'd0};
        dut.sram_inst.bank_gen[2].bank_inst.mem[0] = {224'd0, 8'd0, 8'd1, 8'd0, 8'd0};
        dut.sram_inst.bank_gen[3].bank_inst.mem[0] = {224'd0, 8'd1, 8'd0, 8'd0, 8'd0};

        // Activations at 0x0010
        dut.sram_inst.bank_gen[0].bank_inst.mem[4] = {224'd0, 8'd1, 8'd1, 8'd1, 8'd1};
        dut.sram_inst.bank_gen[1].bank_inst.mem[4] = {224'd0, 8'd2, 8'd2, 8'd2, 8'd2};
        dut.sram_inst.bank_gen[2].bank_inst.mem[4] = {224'd0, 8'd3, 8'd3, 8'd3, 8'd3};
        dut.sram_inst.bank_gen[3].bank_inst.mem[4] = {224'd0, 8'd4, 8'd4, 8'd4, 8'd4};

        $display("");
        $display("SRAM initialized:");
        $display("  bank0[0] (W row0): %h", dut.sram_inst.bank_gen[0].bank_inst.mem[0][31:0]);
        $display("  bank1[0] (W row1): %h", dut.sram_inst.bank_gen[1].bank_inst.mem[0][31:0]);
        $display("  bank2[0] (W row2): %h", dut.sram_inst.bank_gen[2].bank_inst.mem[0][31:0]);
        $display("  bank3[0] (W row3): %h", dut.sram_inst.bank_gen[3].bank_inst.mem[0][31:0]);
        $display("  bank0[4] (X row0): %h", dut.sram_inst.bank_gen[0].bank_inst.mem[4][31:0]);
        $display("  bank1[4] (X row1): %h", dut.sram_inst.bank_gen[1].bank_inst.mem[4][31:0]);
        $display("  bank2[4] (X row2): %h", dut.sram_inst.bank_gen[2].bank_inst.mem[4][31:0]);
        $display("  bank3[4] (X row3): %h", dut.sram_inst.bank_gen[3].bank_inst.mem[4][31:0]);

        // Use EXACT format from tb_e2e_simple.v: dst=0x20, src2=0x10, src1=0x00
        // This means: C[0x20] = A[0x00] × B[0x10] where A=weights, B=activations
        dut.instr_mem[0] = {OP_TENSOR, 8'h01, 16'h0020, 16'h0010, 16'h0000, 16'd4, 16'd4, 16'd4, 16'd0};
        dut.instr_mem[1] = {OP_HALT, 120'd0};

        $display("");
        $display("Instruction: GEMM dst=0x20, src2=0x10 (X), src1=0x00 (W)");
        $display("Expected: X @ W (each sample row × identity = same row)");

        rst_n = 0; #(CLK*5); rst_n = 1; #(CLK*5);

        @(negedge clk);
        tpc_start = 1;
        @(posedge clk); @(posedge clk);
        @(negedge clk);
        tpc_start = 0;

        for (i = 0; i < 50; i = i + 1) begin
            @(posedge clk);
            if (tpc_done) begin
                $display("Done at cycle %0d", i);
                i = 1000;
            end
        end

        #(CLK*5);
        
        $display("");
        $display("Output Z (should be [1,1,1,1], [2,2,2,2], [3,3,3,3], [4,4,4,4]):");
        $display("  Z[0]: %h  -> [%d,%d,%d,%d]", dut.sram_inst.bank_gen[0].bank_inst.mem[8][127:0],
            $signed(dut.sram_inst.bank_gen[0].bank_inst.mem[8][31:0]),
            $signed(dut.sram_inst.bank_gen[0].bank_inst.mem[8][63:32]),
            $signed(dut.sram_inst.bank_gen[0].bank_inst.mem[8][95:64]),
            $signed(dut.sram_inst.bank_gen[0].bank_inst.mem[8][127:96]));
        $display("  Z[1]: %h  -> [%d,%d,%d,%d]", dut.sram_inst.bank_gen[1].bank_inst.mem[8][127:0],
            $signed(dut.sram_inst.bank_gen[1].bank_inst.mem[8][31:0]),
            $signed(dut.sram_inst.bank_gen[1].bank_inst.mem[8][63:32]),
            $signed(dut.sram_inst.bank_gen[1].bank_inst.mem[8][95:64]),
            $signed(dut.sram_inst.bank_gen[1].bank_inst.mem[8][127:96]));
        $display("  Z[2]: %h  -> [%d,%d,%d,%d]", dut.sram_inst.bank_gen[2].bank_inst.mem[8][127:0],
            $signed(dut.sram_inst.bank_gen[2].bank_inst.mem[8][31:0]),
            $signed(dut.sram_inst.bank_gen[2].bank_inst.mem[8][63:32]),
            $signed(dut.sram_inst.bank_gen[2].bank_inst.mem[8][95:64]),
            $signed(dut.sram_inst.bank_gen[2].bank_inst.mem[8][127:96]));
        $display("  Z[3]: %h  -> [%d,%d,%d,%d]", dut.sram_inst.bank_gen[3].bank_inst.mem[8][127:0],
            $signed(dut.sram_inst.bank_gen[3].bank_inst.mem[8][31:0]),
            $signed(dut.sram_inst.bank_gen[3].bank_inst.mem[8][63:32]),
            $signed(dut.sram_inst.bank_gen[3].bank_inst.mem[8][95:64]),
            $signed(dut.sram_inst.bank_gen[3].bank_inst.mem[8][127:96]));

        // Check if results match expected
        if (dut.sram_inst.bank_gen[0].bank_inst.mem[8][31:0] == 32'd1 &&
            dut.sram_inst.bank_gen[1].bank_inst.mem[8][31:0] == 32'd2 &&
            dut.sram_inst.bank_gen[2].bank_inst.mem[8][31:0] == 32'd3 &&
            dut.sram_inst.bank_gen[3].bank_inst.mem[8][31:0] == 32'd4) begin
            $display("");
            $display(">>> BATCH GEMM CORRECT! <<<");
        end else begin
            $display("");
            $display(">>> BATCH GEMM MISMATCH <<<");
        end
        
        $finish;
    end

    initial begin #(CLK*5000); $display("TIMEOUT"); $finish; end
endmodule

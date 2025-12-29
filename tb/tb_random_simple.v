`timescale 1ns / 1ps
module tb_random_simple;
    parameter CLK = 10;
    parameter ARRAY_SIZE = 4;
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
        .ARRAY_SIZE(ARRAY_SIZE), .SRAM_WIDTH(SRAM_WIDTH),
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
    localparam OP_HALT = 8'hFF;

    reg [SRAM_WIDTH-1:0] row0, row1, row2, row3;
    integer i, errors;

    initial begin
        $display("Random Matrix Test (numpy-verified)");
        
        // B^T columns
        dut.sram_inst.bank_gen[0].bank_inst.mem[0] = {224'd0, 8'sd17, -8'sd66, 8'sd94, -8'sd23};
        dut.sram_inst.bank_gen[1].bank_inst.mem[0] = {224'd0, -8'sd95, 8'sd48, -8'sd31, 8'sd67};
        dut.sram_inst.bank_gen[2].bank_inst.mem[0] = {224'd0, 8'sd33, 8'sd102, 8'sd7, -8'sd55};
        dut.sram_inst.bank_gen[3].bank_inst.mem[0] = {224'd0, 8'sd61, -8'sd89, -8'sd44, 8'sd82};
        
        // A rows
        dut.sram_inst.bank_gen[0].bank_inst.mem[4] = {224'd0, -8'sd99, 8'sd6, -8'sd15, 8'sd51};
        dut.sram_inst.bank_gen[1].bank_inst.mem[4] = {224'd0, 8'sd45, -8'sd53, 8'sd89, -8'sd65};
        dut.sram_inst.bank_gen[2].bank_inst.mem[4] = {224'd0, 8'sd127, 8'sd19, -8'sd82, 8'sd31};
        dut.sram_inst.bank_gen[3].bank_inst.mem[4] = {224'd0, -8'sd12, 8'sd100, -8'sd77, -8'sd38};

        dut.instr_mem[0] = {OP_TENSOR, 8'h01, 16'h0020, 16'h0010, 16'h0000, 16'd4, 16'd4, 16'd4, 16'd0};
        dut.instr_mem[1] = {OP_HALT, 120'd0};

        rst_n = 0; #50; rst_n = 1; #50;
        tpc_start = 1; #20; tpc_start = 0;

        #5000; // Wait for completion

        row0 = dut.sram_inst.bank_gen[0].bank_inst.mem[8];
        row1 = dut.sram_inst.bank_gen[1].bank_inst.mem[8];
        row2 = dut.sram_inst.bank_gen[2].bank_inst.mem[8];
        row3 = dut.sram_inst.bank_gen[3].bank_inst.mem[8];

        // Expected from numpy:
        // C[0] = [-4662, 13575, -5565, -1731]
        // C[1] = [14124, -13933, 277, -1784]
        // C[2] = [-7516, -6534, 3850, 12206]
        // C[3] = [-13168, 5781, 11355, -9360]

        $display("Results:");
        $display("  C[0]: [%0d, %0d, %0d, %0d] expect [-4662, 13575, -5565, -1731]",
            $signed(row0[31:0]), $signed(row0[63:32]), $signed(row0[95:64]), $signed(row0[127:96]));
        $display("  C[1]: [%0d, %0d, %0d, %0d] expect [14124, -13933, 277, -1784]",
            $signed(row1[31:0]), $signed(row1[63:32]), $signed(row1[95:64]), $signed(row1[127:96]));
        $display("  C[2]: [%0d, %0d, %0d, %0d] expect [-7516, -6534, 3850, 12206]",
            $signed(row2[31:0]), $signed(row2[63:32]), $signed(row2[95:64]), $signed(row2[127:96]));
        $display("  C[3]: [%0d, %0d, %0d, %0d] expect [-13168, 5781, 11355, -9360]",
            $signed(row3[31:0]), $signed(row3[63:32]), $signed(row3[95:64]), $signed(row3[127:96]));

        errors = 0;
        if ($signed(row0[31:0]) != -4662) errors = errors + 1;
        if ($signed(row0[63:32]) != 13575) errors = errors + 1;
        if ($signed(row0[95:64]) != -5565) errors = errors + 1;
        if ($signed(row0[127:96]) != -1731) errors = errors + 1;
        if ($signed(row1[31:0]) != 14124) errors = errors + 1;
        if ($signed(row1[63:32]) != -13933) errors = errors + 1;
        if ($signed(row3[127:96]) != -9360) errors = errors + 1;

        if (errors == 0) $display(">>> RANDOM MATRIX TEST PASSED! <<<");
        else $display(">>> RANDOM MATRIX TEST FAILED (%0d errors) <<<", errors);
        $finish;
    end
    
    initial begin #100000; $display("TIMEOUT"); $finish; end
endmodule

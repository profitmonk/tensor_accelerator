`timescale 1ns / 1ps
//==============================================================================
// Test: Conv2D via im2col transformation
// 
// Setup:
//   Input:  3×3 image (single channel)
//   Filter: 2×2 kernel with 4 output channels
//   Output: 2×2 × 4 channels
//
// im2col transforms this to GEMM:
//   - Activation matrix: 4 rows × 4 cols (4 patches, each 2×2 flattened)
//   - Weight matrix: 4 rows × 4 cols (4 filters, each 2×2 flattened)
//   - Output: 4 × 4 (4 spatial positions × 4 channels)
//==============================================================================
module tb_e2e_conv2d;
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

    // Input image (3×3):
    //   [1, 2, 3]
    //   [4, 5, 6]
    //   [7, 8, 9]
    //
    // 4 filters (2×2 each), generating 4 output channels:
    //   Filter 0: [[1,0],[0,0]] - top-left pixel detector
    //   Filter 1: [[0,1],[0,0]] - top-right pixel detector  
    //   Filter 2: [[0,0],[1,0]] - bottom-left pixel detector
    //   Filter 3: [[0,0],[0,1]] - bottom-right pixel detector
    //
    // im2col patches (2×2 windows, flattened row-major):
    //   Patch(0,0): [1,2,4,5] → position (0,0)
    //   Patch(0,1): [2,3,5,6] → position (0,1)  
    //   Patch(1,0): [4,5,7,8] → position (1,0)
    //   Patch(1,1): [5,6,8,9] → position (1,1)
    //
    // Activation matrix A (im2col, 4 patches × 4 elements):
    //   [[1,2,4,5],
    //    [2,3,5,6],
    //    [4,5,7,8],
    //    [5,6,8,9]]
    //
    // Weight matrix B (4 filters × 4 elements, transposed for weight-stationary):
    //   Filter elements in cols: [f0[0],f1[0],f2[0],f3[0]; f0[1],f1[1],f2[1],f3[1]; ...]
    //   B^T = [[1,0,0,0],   <- filter element [0,0]
    //          [0,1,0,0],   <- filter element [0,1]
    //          [0,0,1,0],   <- filter element [1,0]
    //          [0,0,0,1]]   <- filter element [1,1]
    //
    // Output C = A × B^T (4 positions × 4 channels):
    //   Each row = one spatial position, each col = one channel
    //   C[i][j] = patch[i] dot filter[j]
    //
    // Expected output (transposed due to weight-stationary dataflow):
    //   C = [[1,2,4,5],    <- pos(0,0): ch0=1, ch1=2, ch2=4, ch3=5
    //        [2,3,5,6],    <- pos(0,1): ch0=2, ch1=3, ch2=5, ch3=6
    //        [4,5,7,8],    <- pos(1,0): ch0=4, ch1=5, ch2=7, ch3=8
    //        [5,6,8,9]]    <- pos(1,1): ch0=5, ch1=6, ch2=8, ch3=9

    reg [SRAM_WIDTH-1:0] row0, row1, row2, row3;
    integer i, errors;

    initial begin
        $display("");
        $display("╔════════════════════════════════════════════════════════════╗");
        $display("║         Conv2D via im2col Test                             ║");
        $display("║         3×3 input, 2×2 kernel, 4 output channels           ║");
        $display("╚════════════════════════════════════════════════════════════╝");
        $display("");
        $display("Input (3×3):");
        $display("  [1, 2, 3]");
        $display("  [4, 5, 6]");
        $display("  [7, 8, 9]");
        $display("");
        $display("Filters (2×2 each): identity detectors for each corner");
        
        //======================================================================
        // Load weight matrix B^T (identity - each filter picks one corner)
        //======================================================================
        dut.sram_inst.bank_gen[0].bank_inst.mem[0] = {224'd0, 8'd0, 8'd0, 8'd0, 8'd1};
        dut.sram_inst.bank_gen[1].bank_inst.mem[0] = {224'd0, 8'd0, 8'd0, 8'd1, 8'd0};
        dut.sram_inst.bank_gen[2].bank_inst.mem[0] = {224'd0, 8'd0, 8'd1, 8'd0, 8'd0};
        dut.sram_inst.bank_gen[3].bank_inst.mem[0] = {224'd0, 8'd1, 8'd0, 8'd0, 8'd0};
        
        //======================================================================
        // Load im2col activation matrix (4 patches)
        //======================================================================
        // Patch(0,0): [1,2,4,5]
        dut.sram_inst.bank_gen[0].bank_inst.mem[4] = {224'd0, 8'd5, 8'd4, 8'd2, 8'd1};
        // Patch(0,1): [2,3,5,6]
        dut.sram_inst.bank_gen[1].bank_inst.mem[4] = {224'd0, 8'd6, 8'd5, 8'd3, 8'd2};
        // Patch(1,0): [4,5,7,8]
        dut.sram_inst.bank_gen[2].bank_inst.mem[4] = {224'd0, 8'd8, 8'd7, 8'd5, 8'd4};
        // Patch(1,1): [5,6,8,9]
        dut.sram_inst.bank_gen[3].bank_inst.mem[4] = {224'd0, 8'd9, 8'd8, 8'd6, 8'd5};

        //======================================================================
        // Program
        //======================================================================
        dut.instr_mem[0] = {OP_TENSOR, 8'h01, 16'h0020, 16'h0010, 16'h0000, 16'd4, 16'd4, 16'd4, 16'd0};
        dut.instr_mem[1] = {OP_HALT, 120'd0};

        rst_n = 0; #(CLK*5); rst_n = 1; #(CLK*5);

        @(negedge clk); tpc_start = 1;
        @(posedge clk); @(posedge clk);
        @(negedge clk); tpc_start = 0;

        for (i = 0; i < 50; i = i + 1) begin
            @(posedge clk);
            if (tpc_done) i = 999;
        end
        #(CLK * 5);
        
        // Read results
        row0 = dut.sram_inst.bank_gen[0].bank_inst.mem[8];
        row1 = dut.sram_inst.bank_gen[1].bank_inst.mem[8];
        row2 = dut.sram_inst.bank_gen[2].bank_inst.mem[8];
        row3 = dut.sram_inst.bank_gen[3].bank_inst.mem[8];
        
        $display("");
        $display("Output (2×2 spatial × 4 channels):");
        $display("  pos(0,0) channels: [%0d,%0d,%0d,%0d] expect [1,2,4,5]",
            $signed(row0[31:0]), $signed(row0[63:32]), $signed(row0[95:64]), $signed(row0[127:96]));
        $display("  pos(0,1) channels: [%0d,%0d,%0d,%0d] expect [2,3,5,6]",
            $signed(row1[31:0]), $signed(row1[63:32]), $signed(row1[95:64]), $signed(row1[127:96]));
        $display("  pos(1,0) channels: [%0d,%0d,%0d,%0d] expect [4,5,7,8]",
            $signed(row2[31:0]), $signed(row2[63:32]), $signed(row2[95:64]), $signed(row2[127:96]));
        $display("  pos(1,1) channels: [%0d,%0d,%0d,%0d] expect [5,6,8,9]",
            $signed(row3[31:0]), $signed(row3[63:32]), $signed(row3[95:64]), $signed(row3[127:96]));
        
        errors = 0;
        if (row0[31:0] != 1 || row0[63:32] != 2 || row0[95:64] != 4 || row0[127:96] != 5) errors = errors + 1;
        if (row1[31:0] != 2 || row1[63:32] != 3 || row1[95:64] != 5 || row1[127:96] != 6) errors = errors + 1;
        if (row2[31:0] != 4 || row2[63:32] != 5 || row2[95:64] != 7 || row2[127:96] != 8) errors = errors + 1;
        if (row3[31:0] != 5 || row3[63:32] != 6 || row3[95:64] != 8 || row3[127:96] != 9) errors = errors + 1;
        
        $display("");
        $display("Interpretation: Each output position contains the 4 corners of its");
        $display("corresponding input patch (identity filter extracts corners).");
        $display("");
        if (errors == 0) $display(">>> CONV2D TEST PASSED! <<<");
        else $display(">>> CONV2D TEST FAILED (%0d errors) <<<", errors);
        $finish;
    end
endmodule

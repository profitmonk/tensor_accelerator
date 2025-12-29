`timescale 1ns / 1ps
//==============================================================================
// Comprehensive Stress Test Suite
// Tests: signed values, large values, overflow, sparse, fixed-point interpretation
//==============================================================================
module tb_stress_test;
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
    integer test_num, total_errors;
    integer i;

    //==========================================================================
    // Helper tasks
    //==========================================================================
    task load_weights(
        input [7:0] w00, w01, w02, w03,
        input [7:0] w10, w11, w12, w13,
        input [7:0] w20, w21, w22, w23,
        input [7:0] w30, w31, w32, w33
    );
    begin
        // B^T format for weight-stationary
        dut.sram_inst.bank_gen[0].bank_inst.mem[0] = {224'd0, w30, w20, w10, w00};
        dut.sram_inst.bank_gen[1].bank_inst.mem[0] = {224'd0, w31, w21, w11, w01};
        dut.sram_inst.bank_gen[2].bank_inst.mem[0] = {224'd0, w32, w22, w12, w02};
        dut.sram_inst.bank_gen[3].bank_inst.mem[0] = {224'd0, w33, w23, w13, w03};
    end
    endtask

    task load_activations(
        input [7:0] a00, a01, a02, a03,
        input [7:0] a10, a11, a12, a13,
        input [7:0] a20, a21, a22, a23,
        input [7:0] a30, a31, a32, a33
    );
    begin
        dut.sram_inst.bank_gen[0].bank_inst.mem[4] = {224'd0, a03, a02, a01, a00};
        dut.sram_inst.bank_gen[1].bank_inst.mem[4] = {224'd0, a13, a12, a11, a10};
        dut.sram_inst.bank_gen[2].bank_inst.mem[4] = {224'd0, a23, a22, a21, a20};
        dut.sram_inst.bank_gen[3].bank_inst.mem[4] = {224'd0, a33, a32, a31, a30};
    end
    endtask

    task run_gemm;
    begin
        dut.instr_mem[0] = {OP_TENSOR, 8'h01, 16'h0020, 16'h0010, 16'h0000, 16'd4, 16'd4, 16'd4, 16'd0};
        dut.instr_mem[1] = {OP_HALT, 120'd0};
        
        rst_n = 0; #(CLK*5); rst_n = 1; #(CLK*5);
        @(negedge clk); tpc_start = 1;
        @(posedge clk); @(posedge clk);
        @(negedge clk); tpc_start = 0;
        
        for (i = 0; i < 60; i = i + 1) begin
            @(posedge clk);
            if (tpc_done) i = 999;
        end
        #(CLK * 3);
        
        row0 = dut.sram_inst.bank_gen[0].bank_inst.mem[8];
        row1 = dut.sram_inst.bank_gen[1].bank_inst.mem[8];
        row2 = dut.sram_inst.bank_gen[2].bank_inst.mem[8];
        row3 = dut.sram_inst.bank_gen[3].bank_inst.mem[8];
    end
    endtask

    function integer check_result(
        input signed [31:0] actual,
        input signed [31:0] expected,
        input [79:0] name
    );
    begin
        if (actual !== expected) begin
            $display("    FAIL %s: got %0d, expected %0d", name, actual, expected);
            check_result = 1;
        end else begin
            check_result = 0;
        end
    end
    endfunction

    //==========================================================================
    // Main test sequence
    //==========================================================================
    initial begin
        $display("");
        $display("╔════════════════════════════════════════════════════════════╗");
        $display("║              STRESS TEST SUITE                             ║");
        $display("║  Signed, Large Values, Overflow, Sparse, Fixed-Point       ║");
        $display("╚════════════════════════════════════════════════════════════╝");
        
        test_num = 0;
        total_errors = 0;

        //======================================================================
        // TEST 1: Negative numbers
        // A = [[-1,-2],[-3,-4]] padded, B = [[1,2],[3,4]] padded
        // C[0][0] = (-1)*1 + (-2)*3 = -7
        // C[0][1] = (-1)*2 + (-2)*4 = -10
        // C[1][0] = (-3)*1 + (-4)*3 = -15
        // C[1][1] = (-3)*2 + (-4)*4 = -22
        //======================================================================
        test_num = test_num + 1;
        $display("");
        $display("[TEST %0d] Negative Numbers", test_num);
        
        load_weights(
            8'd1, 8'd2, 8'd0, 8'd0,
            8'd3, 8'd4, 8'd0, 8'd0,
            8'd0, 8'd0, 8'd1, 8'd0,
            8'd0, 8'd0, 8'd0, 8'd1
        );
        load_activations(
            -8'sd1, -8'sd2, 8'd0, 8'd0,
            -8'sd3, -8'sd4, 8'd0, 8'd0,
            8'd0, 8'd0, 8'd1, 8'd0,
            8'd0, 8'd0, 8'd0, 8'd1
        );
        run_gemm;
        
        $display("  C[0] = [%0d, %0d, %0d, %0d]", $signed(row0[31:0]), $signed(row0[63:32]), $signed(row0[95:64]), $signed(row0[127:96]));
        $display("  C[1] = [%0d, %0d, %0d, %0d]", $signed(row1[31:0]), $signed(row1[63:32]), $signed(row1[95:64]), $signed(row1[127:96]));
        
        total_errors = total_errors + check_result($signed(row0[31:0]), -7, "C[0][0]");
        total_errors = total_errors + check_result($signed(row0[63:32]), -10, "C[0][1]");
        total_errors = total_errors + check_result($signed(row1[31:0]), -15, "C[1][0]");
        total_errors = total_errors + check_result($signed(row1[63:32]), -22, "C[1][1]");

        //======================================================================
        // TEST 2: Mixed positive/negative
        // A = [[10,-20],[30,-40]], B = [[-5,6],[7,-8]]
        // C[0][0] = 10*(-5) + (-20)*7 = -50 - 140 = -190
        // C[0][1] = 10*6 + (-20)*(-8) = 60 + 160 = 220
        // C[1][0] = 30*(-5) + (-40)*7 = -150 - 280 = -430
        // C[1][1] = 30*6 + (-40)*(-8) = 180 + 320 = 500
        //======================================================================
        test_num = test_num + 1;
        $display("");
        $display("[TEST %0d] Mixed Positive/Negative", test_num);
        
        load_weights(
            -8'sd5, 8'sd6, 8'd0, 8'd0,
            8'sd7, -8'sd8, 8'd0, 8'd0,
            8'd0, 8'd0, 8'd1, 8'd0,
            8'd0, 8'd0, 8'd0, 8'd1
        );
        load_activations(
            8'sd10, -8'sd20, 8'd0, 8'd0,
            8'sd30, -8'sd40, 8'd0, 8'd0,
            8'd0, 8'd0, 8'd1, 8'd0,
            8'd0, 8'd0, 8'd0, 8'd1
        );
        run_gemm;
        
        $display("  C[0] = [%0d, %0d, ...]", $signed(row0[31:0]), $signed(row0[63:32]));
        $display("  C[1] = [%0d, %0d, ...]", $signed(row1[31:0]), $signed(row1[63:32]));
        
        total_errors = total_errors + check_result($signed(row0[31:0]), -190, "C[0][0]");
        total_errors = total_errors + check_result($signed(row0[63:32]), 220, "C[0][1]");
        total_errors = total_errors + check_result($signed(row1[31:0]), -430, "C[1][0]");
        total_errors = total_errors + check_result($signed(row1[63:32]), 500, "C[1][1]");

        //======================================================================
        // TEST 3: Large values (near max INT8 = 127)
        // A = [[127,127,127,127],...], B = [[127,0,0,0],[0,127,0,0],...]
        // C[i][j] = 127 * 127 = 16129 (tests accumulator range)
        //======================================================================
        test_num = test_num + 1;
        $display("");
        $display("[TEST %0d] Large Positive Values (127)", test_num);
        
        load_weights(
            8'd127, 8'd0, 8'd0, 8'd0,
            8'd0, 8'd127, 8'd0, 8'd0,
            8'd0, 8'd0, 8'd127, 8'd0,
            8'd0, 8'd0, 8'd0, 8'd127
        );
        load_activations(
            8'd127, 8'd127, 8'd127, 8'd127,
            8'd127, 8'd127, 8'd127, 8'd127,
            8'd127, 8'd127, 8'd127, 8'd127,
            8'd127, 8'd127, 8'd127, 8'd127
        );
        run_gemm;
        
        $display("  C[0] = [%0d, %0d, %0d, %0d] (expect 16129 each)", 
            $signed(row0[31:0]), $signed(row0[63:32]), $signed(row0[95:64]), $signed(row0[127:96]));
        
        total_errors = total_errors + check_result($signed(row0[31:0]), 16129, "C[0][0]");
        total_errors = total_errors + check_result($signed(row0[63:32]), 16129, "C[0][1]");
        total_errors = total_errors + check_result($signed(row0[95:64]), 16129, "C[0][2]");
        total_errors = total_errors + check_result($signed(row0[127:96]), 16129, "C[0][3]");

        //======================================================================
        // TEST 4: Large negative values (near min INT8 = -128)
        // A = [[-128,-128,...]], B diagonal with -128
        // C[i][i] = (-128) * (-128) = 16384
        //======================================================================
        test_num = test_num + 1;
        $display("");
        $display("[TEST %0d] Large Negative Values (-128)", test_num);
        
        load_weights(
            -8'sd128, 8'd0, 8'd0, 8'd0,
            8'd0, -8'sd128, 8'd0, 8'd0,
            8'd0, 8'd0, -8'sd128, 8'd0,
            8'd0, 8'd0, 8'd0, -8'sd128
        );
        load_activations(
            -8'sd128, -8'sd128, -8'sd128, -8'sd128,
            -8'sd128, -8'sd128, -8'sd128, -8'sd128,
            -8'sd128, -8'sd128, -8'sd128, -8'sd128,
            -8'sd128, -8'sd128, -8'sd128, -8'sd128
        );
        run_gemm;
        
        $display("  C[0] = [%0d, %0d, %0d, %0d] (expect 16384 each)", 
            $signed(row0[31:0]), $signed(row0[63:32]), $signed(row0[95:64]), $signed(row0[127:96]));
        
        total_errors = total_errors + check_result($signed(row0[31:0]), 16384, "C[0][0]");
        total_errors = total_errors + check_result($signed(row0[63:32]), 16384, "C[0][1]");

        //======================================================================
        // TEST 5: Accumulator stress (sum of 4 large products)
        // A = [[127,127,127,127],...], B = [[127,127,127,127],...]
        // C[i][j] = 4 * (127*127) = 4 * 16129 = 64516
        //======================================================================
        test_num = test_num + 1;
        $display("");
        $display("[TEST %0d] Accumulator Stress (4 x 127*127 = 64516)", test_num);
        
        load_weights(
            8'd127, 8'd127, 8'd127, 8'd127,
            8'd127, 8'd127, 8'd127, 8'd127,
            8'd127, 8'd127, 8'd127, 8'd127,
            8'd127, 8'd127, 8'd127, 8'd127
        );
        load_activations(
            8'd127, 8'd127, 8'd127, 8'd127,
            8'd127, 8'd127, 8'd127, 8'd127,
            8'd127, 8'd127, 8'd127, 8'd127,
            8'd127, 8'd127, 8'd127, 8'd127
        );
        run_gemm;
        
        $display("  C[0][0] = %0d (expect 64516)", $signed(row0[31:0]));
        $display("  C[3][3] = %0d (expect 64516)", $signed(row3[127:96]));
        
        total_errors = total_errors + check_result($signed(row0[31:0]), 64516, "C[0][0]");
        total_errors = total_errors + check_result($signed(row3[127:96]), 64516, "C[3][3]");

        //======================================================================
        // TEST 6: Mixed sign accumulator stress
        // A = [[127,-128,127,-128],...], B = [[1,1,1,1],...]
        // C[i][j] = 127 - 128 + 127 - 128 = -2
        //======================================================================
        test_num = test_num + 1;
        $display("");
        $display("[TEST %0d] Mixed Sign Accumulator (127-128+127-128 = -2)", test_num);
        
        load_weights(
            8'd1, 8'd1, 8'd1, 8'd1,
            8'd1, 8'd1, 8'd1, 8'd1,
            8'd1, 8'd1, 8'd1, 8'd1,
            8'd1, 8'd1, 8'd1, 8'd1
        );
        load_activations(
            8'sd127, -8'sd128, 8'sd127, -8'sd128,
            8'sd127, -8'sd128, 8'sd127, -8'sd128,
            8'sd127, -8'sd128, 8'sd127, -8'sd128,
            8'sd127, -8'sd128, 8'sd127, -8'sd128
        );
        run_gemm;
        
        $display("  C[0] = [%0d, %0d, %0d, %0d] (expect -2 each)", 
            $signed(row0[31:0]), $signed(row0[63:32]), $signed(row0[95:64]), $signed(row0[127:96]));
        
        total_errors = total_errors + check_result($signed(row0[31:0]), -2, "C[0][0]");
        total_errors = total_errors + check_result($signed(row1[63:32]), -2, "C[1][1]");

        //======================================================================
        // TEST 7: Sparse matrix (mostly zeros)
        // A = [[1,0,0,0],[0,2,0,0],[0,0,3,0],[0,0,0,4]]
        // B = [[5,0,0,0],[0,6,0,0],[0,0,7,0],[0,0,0,8]]
        // C = [[5,0,0,0],[0,12,0,0],[0,0,21,0],[0,0,0,32]]
        //======================================================================
        test_num = test_num + 1;
        $display("");
        $display("[TEST %0d] Sparse Diagonal Matrices", test_num);
        
        load_weights(
            8'd5, 8'd0, 8'd0, 8'd0,
            8'd0, 8'd6, 8'd0, 8'd0,
            8'd0, 8'd0, 8'd7, 8'd0,
            8'd0, 8'd0, 8'd0, 8'd8
        );
        load_activations(
            8'd1, 8'd0, 8'd0, 8'd0,
            8'd0, 8'd2, 8'd0, 8'd0,
            8'd0, 8'd0, 8'd3, 8'd0,
            8'd0, 8'd0, 8'd0, 8'd4
        );
        run_gemm;
        
        $display("  Diagonal: [%0d, %0d, %0d, %0d] (expect [5, 12, 21, 32])", 
            $signed(row0[31:0]), $signed(row1[63:32]), $signed(row2[95:64]), $signed(row3[127:96]));
        $display("  Off-diag: [%0d, %0d] (expect 0)", $signed(row0[63:32]), $signed(row1[31:0]));
        
        total_errors = total_errors + check_result($signed(row0[31:0]), 5, "C[0][0]");
        total_errors = total_errors + check_result($signed(row1[63:32]), 12, "C[1][1]");
        total_errors = total_errors + check_result($signed(row2[95:64]), 21, "C[2][2]");
        total_errors = total_errors + check_result($signed(row3[127:96]), 32, "C[3][3]");
        total_errors = total_errors + check_result($signed(row0[63:32]), 0, "C[0][1]");
        total_errors = total_errors + check_result($signed(row1[31:0]), 0, "C[1][0]");

        //======================================================================
        // TEST 8: Checkerboard pattern
        // A = [[1,-1,1,-1],[-1,1,-1,1],...], B = same
        // Row sum of A = 0, so C should have specific pattern
        //======================================================================
        test_num = test_num + 1;
        $display("");
        $display("[TEST %0d] Checkerboard Pattern", test_num);
        
        load_weights(
            8'sd1, -8'sd1, 8'sd1, -8'sd1,
            -8'sd1, 8'sd1, -8'sd1, 8'sd1,
            8'sd1, -8'sd1, 8'sd1, -8'sd1,
            -8'sd1, 8'sd1, -8'sd1, 8'sd1
        );
        load_activations(
            8'sd1, -8'sd1, 8'sd1, -8'sd1,
            -8'sd1, 8'sd1, -8'sd1, 8'sd1,
            8'sd1, -8'sd1, 8'sd1, -8'sd1,
            -8'sd1, 8'sd1, -8'sd1, 8'sd1
        );
        run_gemm;
        
        // C[0][0] = 1*1 + (-1)*(-1) + 1*1 + (-1)*(-1) = 4
        // C[0][1] = 1*(-1) + (-1)*1 + 1*(-1) + (-1)*1 = -4
        $display("  C[0] = [%0d, %0d, %0d, %0d] (expect [4, -4, 4, -4])", 
            $signed(row0[31:0]), $signed(row0[63:32]), $signed(row0[95:64]), $signed(row0[127:96]));
        
        total_errors = total_errors + check_result($signed(row0[31:0]), 4, "C[0][0]");
        total_errors = total_errors + check_result($signed(row0[63:32]), -4, "C[0][1]");
        total_errors = total_errors + check_result($signed(row0[95:64]), 4, "C[0][2]");
        total_errors = total_errors + check_result($signed(row0[127:96]), -4, "C[0][3]");

        //======================================================================
        // TEST 9: Fixed-point Q4.4 interpretation
        // In Q4.4: 0x18 = 1.5, 0x20 = 2.0, 0x30 = 3.0
        // A = [[1.5, 2.0], [2.5, 3.0]] = [[0x18, 0x20], [0x28, 0x30]]
        // B = [[2.0, 0], [0, 2.0]] = [[0x20, 0], [0, 0x20]]
        // C = [[3.0, 4.0], [5.0, 6.0]] in Q8.8 = [[0x300, 0x400], [0x500, 0x600]]
        // Actually C[0][0] = 0x18 * 0x20 = 0x300 = 768 decimal (3.0 in Q8.8)
        //======================================================================
        test_num = test_num + 1;
        $display("");
        $display("[TEST %0d] Fixed-Point Q4.4 (1.5 * 2.0 = 3.0)", test_num);
        $display("  Q4.4 format: 0x18=1.5, 0x20=2.0, 0x28=2.5, 0x30=3.0");
        
        load_weights(
            8'h20, 8'h00, 8'd0, 8'd0,  // 2.0, 0
            8'h00, 8'h20, 8'd0, 8'd0,  // 0, 2.0
            8'd0, 8'd0, 8'h10, 8'd0,   // 1.0
            8'd0, 8'd0, 8'd0, 8'h10
        );
        load_activations(
            8'h18, 8'h20, 8'd0, 8'd0,  // 1.5, 2.0
            8'h28, 8'h30, 8'd0, 8'd0,  // 2.5, 3.0
            8'd0, 8'd0, 8'h10, 8'd0,
            8'd0, 8'd0, 8'd0, 8'h10
        );
        run_gemm;
        
        // Result in Q8.8: 0x300 = 768 = 3.0, 0x400 = 1024 = 4.0
        $display("  C[0] = [%0d, %0d] (expect [768, 1024] = [3.0, 4.0] in Q8.8)", 
            $signed(row0[31:0]), $signed(row0[63:32]));
        $display("  C[1] = [%0d, %0d] (expect [1280, 1536] = [5.0, 6.0] in Q8.8)", 
            $signed(row1[31:0]), $signed(row1[63:32]));
        
        total_errors = total_errors + check_result($signed(row0[31:0]), 768, "C[0][0]=3.0");
        total_errors = total_errors + check_result($signed(row0[63:32]), 1024, "C[0][1]=4.0");
        total_errors = total_errors + check_result($signed(row1[31:0]), 1280, "C[1][0]=5.0");
        total_errors = total_errors + check_result($signed(row1[63:32]), 1536, "C[1][1]=6.0");

        //======================================================================
        // TEST 10: All zeros (edge case)
        //======================================================================
        test_num = test_num + 1;
        $display("");
        $display("[TEST %0d] All Zeros", test_num);
        
        load_weights(0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0);
        load_activations(0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0);
        run_gemm;
        
        $display("  C[0][0] = %0d (expect 0)", $signed(row0[31:0]));
        total_errors = total_errors + check_result($signed(row0[31:0]), 0, "C[0][0]");
        total_errors = total_errors + check_result($signed(row3[127:96]), 0, "C[3][3]");

        //======================================================================
        // Summary
        //======================================================================
        $display("");
        $display("════════════════════════════════════════════════════════════");
        $display("STRESS TEST SUMMARY: %0d tests, %0d errors", test_num, total_errors);
        $display("════════════════════════════════════════════════════════════");
        
        if (total_errors == 0) 
            $display(">>> ALL STRESS TESTS PASSED! <<<");
        else 
            $display(">>> STRESS TESTS FAILED <<<");
        
        $finish;
    end
endmodule

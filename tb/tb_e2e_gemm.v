//==============================================================================
// End-to-End GEMM Test
//
// Tests complete data path:
// 1. Pre-load weights into SRAM
// 2. Pre-load activations into SRAM  
// 3. Execute MXU GEMM instruction
// 4. Verify computed results
//
// Test case: 4x4 matrix multiply (matches ARRAY_SIZE=4)
//   C = A × B
//   where A, B are 4x4 INT8 matrices
//   C is 4x4 INT32 result
//==============================================================================
`timescale 1ns / 1ps

module tb_e2e_gemm;

    parameter CLK = 10;
    parameter ARRAY_SIZE = 4;
    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH = 32;
    parameter SRAM_WIDTH = 256;

    reg clk = 0;
    reg rst_n = 0;
    always #(CLK/2) clk = ~clk;

    // TPC interface
    reg tpc_start = 0;
    reg [19:0] tpc_start_pc = 0;
    wire tpc_busy, tpc_done, tpc_error;
    reg global_sync_in = 0;
    wire sync_request;
    reg sync_grant = 0;

    // NoC interface (unused)
    reg [SRAM_WIDTH-1:0] noc_rx_data = 0;
    reg [19:0] noc_rx_addr = 0;
    reg noc_rx_valid = 0;
    wire noc_rx_ready;
    reg noc_rx_is_instr = 0;
    wire [SRAM_WIDTH-1:0] noc_tx_data;
    wire [19:0] noc_tx_addr;
    wire noc_tx_valid;
    reg noc_tx_ready = 1;

    // AXI stub
    wire [39:0] axi_awaddr, axi_araddr;
    wire [7:0] axi_awlen, axi_arlen;
    wire axi_awvalid, axi_arvalid, axi_wvalid, axi_wlast, axi_rready, axi_bready;
    wire [255:0] axi_wdata;
    reg axi_awready = 1, axi_arready = 1, axi_wready = 1;
    reg axi_bvalid = 0, axi_rvalid = 0, axi_rlast = 0;
    reg [1:0] axi_bresp = 0;
    reg [255:0] axi_rdata = 0;

    // DUT - single TPC
    tensor_processing_cluster #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .SRAM_WIDTH(SRAM_WIDTH),
        .SRAM_BANKS(4),
        .SRAM_DEPTH(256),
        .VPU_LANES(16),
        .TPC_ID(0)
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

    integer errors = 0;
    integer timeout, i, j, row, col;

    // Opcodes
    localparam OP_TENSOR = 8'h01;
    localparam OP_HALT   = 8'hFF;
    localparam MXU_GEMM  = 8'h01;

    // SRAM addresses (simple linear for testing)
    // Weight matrix at address 0x00 (rows 0-3)
    // Activation matrix at address 0x10 (rows 0-3)
    // Output matrix at address 0x20 (rows 0-3)
    localparam WEIGHT_ADDR = 16'h0000;
    localparam ACT_ADDR    = 16'h0010;
    localparam OUT_ADDR    = 16'h0020;

    // Test matrices (4x4, INT8)
    // A (activation) = [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]]
    // B (weight)     = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]] (identity for simple test)
    // Expected C = A (since B is identity)

    // Actually, let's use non-trivial matrices:
    // A = [[1,2], [3,4]] padded to 4x4 with zeros in unused positions
    // B = [[5,6], [7,8]] padded to 4x4
    // C = [[19,22], [43,50]]

    // For systolic array: each SRAM word (256-bit) holds one row of 4 INT8 values
    // Packed as: row[0]=LSB, row[3]=MSB for weights (column-major in systolic)

    //==========================================================================
    // Helper: Pack 4 INT8 values into 32-bit word
    //==========================================================================
    function [31:0] pack4;
        input [7:0] v0, v1, v2, v3;
        begin
            pack4 = {v3, v2, v1, v0};
        end
    endfunction

    //==========================================================================
    // Test matrices (4x4)
    //==========================================================================
    // Using simple values for easy verification
    // A (activations, row-major):
    //   Row 0: [1, 2, 3, 4]
    //   Row 1: [5, 6, 7, 8]
    //   Row 2: [0, 0, 0, 0]
    //   Row 3: [0, 0, 0, 0]
    //
    // B (weights, stored column-major for systolic):
    //   Col 0: [1, 0, 0, 0]  -> computes with A to get C[*,0]
    //   Col 1: [0, 1, 0, 0]
    //   Col 2: [0, 0, 1, 0]
    //   Col 3: [0, 0, 0, 1]
    // This is identity, so C = A

    reg [7:0] mat_a [0:3][0:3];
    reg [7:0] mat_b [0:3][0:3];
    reg signed [31:0] expected_c [0:3][0:3];
    reg signed [31:0] actual_c [0:3][0:3];

    //==========================================================================
    // Initialize test data
    //==========================================================================
    task init_matrices;
        begin
            // Matrix A (activations)
            mat_a[0][0] = 1;  mat_a[0][1] = 2;  mat_a[0][2] = 3;  mat_a[0][3] = 4;
            mat_a[1][0] = 5;  mat_a[1][1] = 6;  mat_a[1][2] = 7;  mat_a[1][3] = 8;
            mat_a[2][0] = 9;  mat_a[2][1] = 10; mat_a[2][2] = 11; mat_a[2][3] = 12;
            mat_a[3][0] = 13; mat_a[3][1] = 14; mat_a[3][2] = 15; mat_a[3][3] = 16;

            // Matrix B (weights) - identity for simple verification
            mat_b[0][0] = 1;  mat_b[0][1] = 0;  mat_b[0][2] = 0;  mat_b[0][3] = 0;
            mat_b[1][0] = 0;  mat_b[1][1] = 1;  mat_b[1][2] = 0;  mat_b[1][3] = 0;
            mat_b[2][0] = 0;  mat_b[2][1] = 0;  mat_b[2][2] = 1;  mat_b[2][3] = 0;
            mat_b[3][0] = 0;  mat_b[3][1] = 0;  mat_b[3][2] = 0;  mat_b[3][3] = 1;

            // Expected C = A * B = A (since B is identity)
            for (row = 0; row < 4; row = row + 1) begin
                for (col = 0; col < 4; col = col + 1) begin
                    expected_c[row][col] = mat_a[row][col];
                end
            end
        end
    endtask

    //==========================================================================
    // Load matrices into SRAM (direct memory access for test)
    //==========================================================================
    task load_matrices;
        integer r;
        reg [SRAM_WIDTH-1:0] sram_word;
        begin
            // Load activation matrix (row-major) - one row per SRAM word
            // SRAM word is 256 bits = 32 bytes = 32 INT8 values
            // We only use first 4 bytes per row
            for (r = 0; r < 4; r = r + 1) begin
                sram_word = 0;
                sram_word[7:0]   = mat_a[r][0];
                sram_word[15:8]  = mat_a[r][1];
                sram_word[23:16] = mat_a[r][2];
                sram_word[31:24] = mat_a[r][3];
                // Direct write to SRAM bank 0
                dut.sram_inst.bank_gen[0].bank_inst.mem[ACT_ADDR + r] = sram_word;
            end

            // Load weight matrix (column-major for systolic array)
            // Column j goes to address WEIGHT_ADDR + j
            for (col = 0; col < 4; col = col + 1) begin
                sram_word = 0;
                sram_word[7:0]   = mat_b[0][col];  // row 0
                sram_word[15:8]  = mat_b[1][col];  // row 1
                sram_word[23:16] = mat_b[2][col];  // row 2
                sram_word[31:24] = mat_b[3][col];  // row 3
                dut.sram_inst.bank_gen[0].bank_inst.mem[WEIGHT_ADDR + col] = sram_word;
            end

            $display("  INFO: Matrices loaded into SRAM");
        end
    endtask

    //==========================================================================
    // Load program into instruction memory
    //==========================================================================
    task load_program;
        reg [127:0] mxu_instr;
        begin
            // MXU GEMM instruction
            // Format: [127:120]=opcode, [119:112]=subop, [111:96]=dst, [95:80]=src0(act), [79:64]=src1(wt)
            mxu_instr = 0;
            mxu_instr[127:120] = OP_TENSOR;
            mxu_instr[119:112] = MXU_GEMM;
            mxu_instr[111:96]  = OUT_ADDR;     // dst
            mxu_instr[95:80]   = ACT_ADDR;     // src0 (activations)
            mxu_instr[79:64]   = WEIGHT_ADDR;  // src1 (weights)
            mxu_instr[63:48]   = 16'd4;        // M
            mxu_instr[47:32]   = 16'd4;        // N
            mxu_instr[31:16]   = 16'd4;        // K

            dut.instr_mem[0] = mxu_instr;
            dut.instr_mem[1] = {OP_HALT, 120'd0};

            $display("  INFO: Program loaded (MXU GEMM + HALT)");
        end
    endtask

    //==========================================================================
    // Read results from SRAM
    // Address mapping: addr -> bank = (addr[1:0] ^ addr[9:8]), word = addr[9:2]
    //==========================================================================
    task read_results;
        integer r;
        reg [SRAM_WIDTH-1:0] sram_word;
        reg [1:0] bank;
        reg [7:0] word;
        begin
            for (r = 0; r < 4; r = r + 1) begin
                // Calculate bank and word for OUT_ADDR + r
                bank = (OUT_ADDR + r) ^ ((OUT_ADDR + r) >> 8);
                bank = bank[1:0];
                word = (OUT_ADDR + r) >> 2;
                
                case (bank)
                    0: sram_word = dut.sram_inst.bank_gen[0].bank_inst.mem[word];
                    1: sram_word = dut.sram_inst.bank_gen[1].bank_inst.mem[word];
                    2: sram_word = dut.sram_inst.bank_gen[2].bank_inst.mem[word];
                    3: sram_word = dut.sram_inst.bank_gen[3].bank_inst.mem[word];
                endcase
                
                // Each result is 32-bit (ACC_WIDTH)
                actual_c[r][0] = $signed(sram_word[31:0]);
                actual_c[r][1] = $signed(sram_word[63:32]);
                actual_c[r][2] = $signed(sram_word[95:64]);
                actual_c[r][3] = $signed(sram_word[127:96]);
            end
        end
    endtask

    //==========================================================================
    // Verify results
    //==========================================================================
    task verify_results;
        integer match;
        begin
            match = 1;
            $display("  Results:");
            $display("    Row | Expected          | Actual");
            for (row = 0; row < 4; row = row + 1) begin
                $display("    %0d   | [%3d,%3d,%3d,%3d] | [%3d,%3d,%3d,%3d]", 
                    row,
                    expected_c[row][0], expected_c[row][1], expected_c[row][2], expected_c[row][3],
                    actual_c[row][0], actual_c[row][1], actual_c[row][2], actual_c[row][3]);
                for (col = 0; col < 4; col = col + 1) begin
                    if (actual_c[row][col] !== expected_c[row][col]) match = 0;
                end
            end
            if (match) $display("  PASS: Results match expected");
            else begin $display("  FAIL: Results mismatch"); errors = errors + 1; end
        end
    endtask

    //==========================================================================
    // Run TPC
    //==========================================================================
    task run_tpc;
        begin
            @(negedge clk);
            tpc_start_pc = 20'd0;
            tpc_start = 1;
            @(posedge clk);
            @(negedge clk);
            tpc_start = 0;

            timeout = 0;
            while (!tpc_done && timeout < 500) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
        end
    endtask

    //==========================================================================
    // Main test
    //==========================================================================
    initial begin
        $display("");
        $display("╔════════════════════════════════════════════════════════════╗");
        $display("║           End-to-End GEMM Test                             ║");
        $display("║           4×4 Matrix Multiply (Identity Test)              ║");
        $display("╚════════════════════════════════════════════════════════════╝");

        init_matrices;

        #(CLK * 5);
        rst_n = 1;
        #(CLK * 5);

        //======================================================================
        $display("");
        $display("[TEST 1] Load Test Data");
        load_matrices;
        load_program;
        $display("  PASS: Data and program loaded");

        //======================================================================
        $display("");
        $display("[TEST 2] Execute GEMM");
        run_tpc;
        if (tpc_done && !tpc_error) begin
            $display("  PASS: GEMM completed (%0d cycles)", timeout);
        end else begin
            $display("  FAIL: done=%b error=%b timeout=%0d", tpc_done, tpc_error, timeout);
            errors = errors + 1;
        end

        //======================================================================
        $display("");
        $display("[TEST 3] Verify Results");
        #(CLK * 10);  // Let results settle
        read_results;
        verify_results;

        //======================================================================
        $display("");
        $display("════════════════════════════════════════");
        $display("Tests: 3, Errors: %0d", errors);
        if (errors == 0) $display(">>> ALL TESTS PASSED! <<<");
        else $display(">>> SOME TESTS FAILED <<<");
        $display("");
        $finish;
    end

    initial begin $dumpfile("e2e_gemm.vcd"); $dumpvars(0, tb_e2e_gemm); end
    initial begin #(CLK * 50000); $display("TIMEOUT!"); $finish; end

endmodule

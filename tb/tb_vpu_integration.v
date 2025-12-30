`timescale 1ns/1ps
/*
 * VPU Integration Testbench
 * 
 * Tests VPU with proper command interface:
 * 1. LOAD data from SRAM to vector register
 * 2. Apply ReLU activation
 * 3. STORE result back to SRAM
 * 4. Verify negative values are zeroed
 */

module tb_vpu_integration;

    // Parameters - match TPC defaults
    parameter LANES       = 64;
    parameter DATA_WIDTH  = 16;
    parameter VREG_COUNT  = 32;
    parameter SRAM_ADDR_W = 20;
    
    // Clock and reset
    reg clk;
    reg rst_n;
    
    // VPU Command Interface
    reg [127:0]                  cmd;
    reg                          cmd_valid;
    wire                         cmd_ready;
    wire                         cmd_done;
    
    // SRAM Interface
    wire [SRAM_ADDR_W-1:0]       sram_addr;
    wire [LANES*DATA_WIDTH-1:0]  sram_wdata;
    reg  [LANES*DATA_WIDTH-1:0]  sram_rdata;
    wire                         sram_we;
    wire                         sram_re;
    reg                          sram_ready;
    
    // =========================================================================
    // Behavioral SRAM Model
    // =========================================================================
    reg [LANES*DATA_WIDTH-1:0] sram_mem [0:255];
    reg [LANES*DATA_WIDTH-1:0] sram_rdata_next;
    
    always @(posedge clk) begin
        sram_rdata <= sram_rdata_next;
        
        if (sram_re) begin
            sram_rdata_next <= sram_mem[sram_addr[7:0]];
        end
        
        if (sram_we) begin
            sram_mem[sram_addr[7:0]] <= sram_wdata;
        end
    end
    
    // =========================================================================
    // VPU Instance
    // =========================================================================
    vector_unit #(
        .LANES      (LANES),
        .DATA_WIDTH (DATA_WIDTH),
        .VREG_COUNT (VREG_COUNT),
        .SRAM_ADDR_W(SRAM_ADDR_W)
    ) u_vpu (
        .clk        (clk),
        .rst_n      (rst_n),
        .cmd        (cmd),
        .cmd_valid  (cmd_valid),
        .cmd_ready  (cmd_ready),
        .cmd_done   (cmd_done),
        .sram_addr  (sram_addr),
        .sram_wdata (sram_wdata),
        .sram_rdata (sram_rdata),
        .sram_we    (sram_we),
        .sram_re    (sram_re),
        .sram_ready (sram_ready)
    );
    
    // =========================================================================
    // Clock Generation
    // =========================================================================
    initial clk = 0;
    always #5 clk = ~clk;
    
    // =========================================================================
    // VPU Command Encoding
    // =========================================================================
    // Suboperation codes (from vector_unit.v)
    localparam VOP_RELU  = 8'h10;
    localparam VOP_LOAD  = 8'h30;
    localparam VOP_STORE = 8'h31;
    localparam OPCODE_VPU = 8'h02;
    
    // Build VPU command - field layout:
    //   [127:120] opcode (8 bits)
    //   [119:112] subop (8 bits)
    //   [111:107] vd - destination register (5 bits)
    //   [106:102] vs1 - source 1 register (5 bits)
    //   [101:97]  vs2 - source 2 register (5 bits)
    //   [95:76]   mem_addr (20 bits)
    //   [63:48]   count (16 bits)
    function [127:0] make_vpu_cmd;
        input [7:0]  subop;
        input [4:0]  vd;        // Destination register
        input [4:0]  vs1;       // Source 1 register
        input [4:0]  vs2;       // Source 2 register
        input [19:0] mem_addr;  // Memory address
        input [15:0] count;     // Element count
    begin
        make_vpu_cmd = 128'b0;
        make_vpu_cmd[127:120] = OPCODE_VPU;  // Opcode
        make_vpu_cmd[119:112] = subop;       // Subop
        make_vpu_cmd[111:107] = vd;          // Dest reg (fixed position)
        make_vpu_cmd[106:102] = vs1;         // Src1 reg (fixed position)
        make_vpu_cmd[101:97]  = vs2;         // Src2 reg (fixed position)
        make_vpu_cmd[95:76]   = mem_addr;    // Memory address
        make_vpu_cmd[63:48]   = count;       // Count
    end
    endfunction
    
    // =========================================================================
    // Test Tasks
    // =========================================================================
    task issue_cmd;
        input [127:0] c;
    begin
        cmd = c;
        cmd_valid = 1;
        @(posedge clk);
        while (!cmd_ready) @(posedge clk);
        @(posedge clk);
        cmd_valid = 0;
        // Wait for completion
        while (!cmd_done) @(posedge clk);
        @(posedge clk);
    end
    endtask
    
    // =========================================================================
    // Test Variables
    // =========================================================================
    integer i, errors, test_num, total_errors;
    reg signed [DATA_WIDTH-1:0] test_data [0:LANES-1];
    reg signed [DATA_WIDTH-1:0] expected [0:LANES-1];
    reg signed [DATA_WIDTH-1:0] actual [0:LANES-1];
    
    // =========================================================================
    // Main Test
    // =========================================================================
    initial begin
        $display("");
        $display("╔══════════════════════════════════════════════════════════════╗");
        $display("║           VPU Integration Testbench                          ║");
        $display("║           LOAD → ReLU → STORE Pipeline                       ║");
        $display("╚══════════════════════════════════════════════════════════════╝");
        $display("");
        
        // Initialize
        rst_n = 0;
        cmd = 0;
        cmd_valid = 0;
        sram_ready = 1;
        sram_rdata = 0;
        sram_rdata_next = 0;
        total_errors = 0;
        
        // Clear SRAM
        for (i = 0; i < 256; i = i + 1) begin
            sram_mem[i] = 0;
        end
        
        // Reset
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);
        
        // =====================================================================
        // TEST 1: Basic ReLU (LOAD → RELU → STORE)
        // =====================================================================
        test_num = 1;
        errors = 0;
        $display("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        $display("[TEST %0d] LOAD → ReLU → STORE", test_num);
        $display("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        // Initialize test data with mix of positive and negative values
        for (i = 0; i < LANES; i = i + 1) begin
            // Create pattern: -32, -31, ..., -1, 0, 1, ..., 31
            test_data[i] = i - (LANES/2);
            expected[i] = (test_data[i] < 0) ? 0 : test_data[i];
        end
        
        // Pack test data into SRAM (address 0x00)
        for (i = 0; i < LANES; i = i + 1) begin
            sram_mem[0][i*DATA_WIDTH +: DATA_WIDTH] = test_data[i];
        end
        
        $display("  Input data (first 8): %d %d %d %d %d %d %d %d", 
                 test_data[0], test_data[1], test_data[2], test_data[3],
                 test_data[4], test_data[5], test_data[6], test_data[7]);
        $display("  Expected (first 8):   %d %d %d %d %d %d %d %d",
                 expected[0], expected[1], expected[2], expected[3],
                 expected[4], expected[5], expected[6], expected[7]);
        
        // LOAD from SRAM[0x00] to V0
        $display("  Issuing LOAD command (SRAM[0x00] → V0)...");
        issue_cmd(make_vpu_cmd(VOP_LOAD, 5'd0, 5'd0, 5'd0, 20'h00, 16'd64));
        
        // RELU V0 → V1
        $display("  Issuing ReLU command (V0 → V1)...");
        issue_cmd(make_vpu_cmd(VOP_RELU, 5'd1, 5'd0, 5'd0, 20'h00, 16'd64));
        
        // STORE V1 to SRAM[0x10] - STORE reads from vs1, not vd
        $display("  Issuing STORE command (V1 → SRAM[0x10])...");
        issue_cmd(make_vpu_cmd(VOP_STORE, 5'd0, 5'd1, 5'd0, 20'h10, 16'd64));
        
        // Read result from SRAM
        for (i = 0; i < LANES; i = i + 1) begin
            actual[i] = $signed(sram_mem[16][i*DATA_WIDTH +: DATA_WIDTH]);
        end
        
        $display("  Actual (first 8):     %d %d %d %d %d %d %d %d",
                 actual[0], actual[1], actual[2], actual[3],
                 actual[4], actual[5], actual[6], actual[7]);
        
        // Verify
        for (i = 0; i < LANES; i = i + 1) begin
            if (actual[i] !== expected[i]) begin
                errors = errors + 1;
                if (errors <= 5) begin
                    $display("  ERROR: lane[%0d] = %d, expected %d", i, actual[i], expected[i]);
                end
            end
            if (actual[i] < 0) begin
                errors = errors + 1;
                $display("  ERROR: lane[%0d] = %d is negative after ReLU!", i, actual[i]);
            end
        end
        
        if (errors == 0) begin
            $display(">>> TEST %0d PASSED <<<", test_num);
        end else begin
            $display(">>> TEST %0d FAILED (%0d errors) <<<", test_num, errors);
            total_errors = total_errors + errors;
        end
        
        repeat(10) @(posedge clk);
        
        // =====================================================================
        // TEST 2: All negative values
        // =====================================================================
        test_num = 2;
        errors = 0;
        $display("");
        $display("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        $display("[TEST %0d] ReLU on all negative values", test_num);
        $display("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        // All negative values
        for (i = 0; i < LANES; i = i + 1) begin
            test_data[i] = -(i + 1);
            expected[i] = 0;  // All should become 0
        end
        
        for (i = 0; i < LANES; i = i + 1) begin
            sram_mem[1][i*DATA_WIDTH +: DATA_WIDTH] = test_data[i];
        end
        
        $display("  Input (first 8): %d %d %d %d %d %d %d %d",
                 test_data[0], test_data[1], test_data[2], test_data[3],
                 test_data[4], test_data[5], test_data[6], test_data[7]);
        
        // LOAD → RELU → STORE (STORE reads from vs1)
        issue_cmd(make_vpu_cmd(VOP_LOAD, 5'd2, 5'd0, 5'd0, 20'h01, 16'd64));
        issue_cmd(make_vpu_cmd(VOP_RELU, 5'd3, 5'd2, 5'd0, 20'h00, 16'd64));
        issue_cmd(make_vpu_cmd(VOP_STORE, 5'd0, 5'd3, 5'd0, 20'h20, 16'd64));
        
        for (i = 0; i < LANES; i = i + 1) begin
            actual[i] = $signed(sram_mem[32][i*DATA_WIDTH +: DATA_WIDTH]);
        end
        
        $display("  Output (first 8): %d %d %d %d %d %d %d %d",
                 actual[0], actual[1], actual[2], actual[3],
                 actual[4], actual[5], actual[6], actual[7]);
        
        // Verify all zeros
        for (i = 0; i < LANES; i = i + 1) begin
            if (actual[i] !== 0) begin
                errors = errors + 1;
            end
        end
        
        if (errors == 0) begin
            $display(">>> TEST %0d PASSED <<<", test_num);
        end else begin
            $display(">>> TEST %0d FAILED (%0d non-zero outputs) <<<", test_num, errors);
            total_errors = total_errors + errors;
        end
        
        repeat(10) @(posedge clk);
        
        // =====================================================================
        // TEST 3: All positive values (should pass through unchanged)
        // =====================================================================
        test_num = 3;
        errors = 0;
        $display("");
        $display("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        $display("[TEST %0d] ReLU on all positive values (passthrough)", test_num);
        $display("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        for (i = 0; i < LANES; i = i + 1) begin
            test_data[i] = i + 1;
            expected[i] = i + 1;  // Should pass through
        end
        
        for (i = 0; i < LANES; i = i + 1) begin
            sram_mem[2][i*DATA_WIDTH +: DATA_WIDTH] = test_data[i];
        end
        
        $display("  Input (first 8): %d %d %d %d %d %d %d %d",
                 test_data[0], test_data[1], test_data[2], test_data[3],
                 test_data[4], test_data[5], test_data[6], test_data[7]);
        
        issue_cmd(make_vpu_cmd(VOP_LOAD, 5'd4, 5'd0, 5'd0, 20'h02, 16'd64));
        issue_cmd(make_vpu_cmd(VOP_RELU, 5'd5, 5'd4, 5'd0, 20'h00, 16'd64));
        issue_cmd(make_vpu_cmd(VOP_STORE, 5'd0, 5'd5, 5'd0, 20'h30, 16'd64));
        
        for (i = 0; i < LANES; i = i + 1) begin
            actual[i] = $signed(sram_mem[48][i*DATA_WIDTH +: DATA_WIDTH]);
        end
        
        $display("  Output (first 8): %d %d %d %d %d %d %d %d",
                 actual[0], actual[1], actual[2], actual[3],
                 actual[4], actual[5], actual[6], actual[7]);
        
        for (i = 0; i < LANES; i = i + 1) begin
            if (actual[i] !== expected[i]) begin
                errors = errors + 1;
            end
        end
        
        if (errors == 0) begin
            $display(">>> TEST %0d PASSED <<<", test_num);
        end else begin
            $display(">>> TEST %0d FAILED (%0d errors) <<<", test_num, errors);
            total_errors = total_errors + errors;
        end
        
        // =====================================================================
        // Summary
        // =====================================================================
        repeat(10) @(posedge clk);
        
        $display("");
        $display("╔══════════════════════════════════════════════════════════════╗");
        $display("║                    TEST SUMMARY                              ║");
        $display("╠══════════════════════════════════════════════════════════════╣");
        if (total_errors == 0) begin
            $display("║   All %0d tests PASSED                                       ║", test_num);
            $display("╚══════════════════════════════════════════════════════════════╝");
            $display(">>> ALL VPU INTEGRATION TESTS PASSED! <<<");
        end else begin
            $display("║   FAILED: %0d total errors                                   ║", total_errors);
            $display("╚══════════════════════════════════════════════════════════════╝");
            $display(">>> SOME TESTS FAILED <<<");
        end
        $display("");
        
        $finish;
    end
    
    // Timeout
    initial begin
        #50000;
        $display("ERROR: Testbench timeout!");
        $finish;
    end

endmodule

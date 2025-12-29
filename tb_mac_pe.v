//==============================================================================
// MAC Processing Element Testbench
//
// Tests:
// 1. Weight loading
// 2. Simple multiply: 3 × 4 = 12
// 3. Accumulation: 12 + (3 × 5) = 27
// 4. Signed multiply: 3 × (-2) = -6
//==============================================================================

`timescale 1ns / 1ps

module tb_mac_pe;

    //==========================================================================
    // Parameters
    //==========================================================================
    
    parameter CLK_PERIOD = 10;
    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH = 32;

    //==========================================================================
    // Signals
    //==========================================================================
    
    reg  clk;
    reg  rst_n;
    reg  enable;
    reg  load_weight;
    reg  clear_acc;
    reg  [DATA_WIDTH-1:0] weight_in;
    reg  [DATA_WIDTH-1:0] act_in;
    wire [DATA_WIDTH-1:0] act_out;
    reg  [ACC_WIDTH-1:0]  psum_in;
    wire [ACC_WIDTH-1:0]  psum_out;

    //==========================================================================
    // DUT
    //==========================================================================
    
    mac_pe #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .enable     (enable),
        .load_weight(load_weight),
        .clear_acc  (clear_acc),
        .weight_in  (weight_in),
        .act_in     (act_in),
        .act_out    (act_out),
        .psum_in    (psum_in),
        .psum_out   (psum_out)
    );

    //==========================================================================
    // Clock Generation
    //==========================================================================
    
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    //==========================================================================
    // Test Variables
    //==========================================================================
    
    integer tests_passed;
    integer tests_failed;

    //==========================================================================
    // Test Sequence
    //==========================================================================
    
    initial begin
        // Initialize
        rst_n = 0;
        enable = 0;
        load_weight = 0;
        clear_acc = 0;
        weight_in = 0;
        act_in = 0;
        psum_in = 0;
        tests_passed = 0;
        tests_failed = 0;
        
        $display("");
        $display("╔════════════════════════════════════════════════════════════╗");
        $display("║           MAC Processing Element Testbench                 ║");
        $display("║           Data Width: %0d bits, Acc Width: %0d bits           ║", DATA_WIDTH, ACC_WIDTH);
        $display("╚════════════════════════════════════════════════════════════╝");
        $display("");
        
        // Reset
        #(CLK_PERIOD * 3);
        rst_n = 1;
        #(CLK_PERIOD * 2);
        
        //----------------------------------------------------------------------
        // Test 1: Load weight = 3
        //----------------------------------------------------------------------
        $display("────────────────────────────────────────────────────────────────");
        $display("[TEST 1] Loading weight = 3");
        $display("────────────────────────────────────────────────────────────────");
        
        load_weight = 1;
        weight_in = 8'd3;
        #CLK_PERIOD;
        load_weight = 0;
        #CLK_PERIOD;
        
        // Check internal weight register (via hierarchical access)
        if (dut.weight_reg == 8'd3) begin
            $display("  PASS: weight_reg = %0d (expected 3)", dut.weight_reg);
            tests_passed = tests_passed + 1;
        end else begin
            $display("  FAIL: weight_reg = %0d (expected 3)", dut.weight_reg);
            tests_failed = tests_failed + 1;
        end
        
        //----------------------------------------------------------------------
        // Test 2: Multiply 3 × 4 = 12
        //----------------------------------------------------------------------
        $display("");
        $display("────────────────────────────────────────────────────────────────");
        $display("[TEST 2] Computing 3 × 4 + 0 = 12");
        $display("────────────────────────────────────────────────────────────────");
        
        enable = 1;
        clear_acc = 1;
        act_in = 8'd4;
        psum_in = 32'd0;
        #CLK_PERIOD;
        clear_acc = 0;
        #CLK_PERIOD;
        
        if (psum_out == 32'd12) begin
            $display("  PASS: psum_out = %0d (expected 12)", psum_out);
            tests_passed = tests_passed + 1;
        end else begin
            $display("  FAIL: psum_out = %0d (expected 12)", psum_out);
            tests_failed = tests_failed + 1;
        end
        
        // Also check activation passthrough
        if (act_out == 8'd4) begin
            $display("  PASS: act_out = %0d (activation passed through)", act_out);
            tests_passed = tests_passed + 1;
        end else begin
            $display("  FAIL: act_out = %0d (expected 4)", act_out);
            tests_failed = tests_failed + 1;
        end
        
        //----------------------------------------------------------------------
        // Test 3: Accumulate: 12 + (3 × 5) = 12 + 15 = 27
        //----------------------------------------------------------------------
        $display("");
        $display("────────────────────────────────────────────────────────────────");
        $display("[TEST 3] Accumulating: 12 + (3 × 5) = 27");
        $display("────────────────────────────────────────────────────────────────");
        
        act_in = 8'd5;
        psum_in = 32'd12;
        #(CLK_PERIOD * 2);
        
        if (psum_out == 32'd27) begin
            $display("  PASS: psum_out = %0d (expected 27)", psum_out);
            tests_passed = tests_passed + 1;
        end else begin
            $display("  FAIL: psum_out = %0d (expected 27)", psum_out);
            tests_failed = tests_failed + 1;
        end
        
        //----------------------------------------------------------------------
        // Test 4: Signed multiplication: 3 × (-2) = -6
        //----------------------------------------------------------------------
        $display("");
        $display("────────────────────────────────────────────────────────────────");
        $display("[TEST 4] Signed multiply: 3 × (-2) = -6");
        $display("────────────────────────────────────────────────────────────────");
        
        clear_acc = 1;
        act_in = -8'sd2;  // -2 in signed 8-bit
        psum_in = 32'd0;
        #CLK_PERIOD;
        clear_acc = 0;
        #CLK_PERIOD;
        
        if ($signed(psum_out) == -32'sd6) begin
            $display("  PASS: psum_out = %0d (expected -6)", $signed(psum_out));
            tests_passed = tests_passed + 1;
        end else begin
            $display("  FAIL: psum_out = %0d (expected -6)", $signed(psum_out));
            tests_failed = tests_failed + 1;
        end
        
        //----------------------------------------------------------------------
        // Test 5: Larger values: 127 × 127 = 16129
        //----------------------------------------------------------------------
        $display("");
        $display("────────────────────────────────────────────────────────────────");
        $display("[TEST 5] Larger values: 127 × 127 = 16129");
        $display("────────────────────────────────────────────────────────────────");
        
        // Load new weight
        load_weight = 1;
        weight_in = 8'd127;
        #CLK_PERIOD;
        load_weight = 0;
        
        clear_acc = 1;
        act_in = 8'd127;
        psum_in = 32'd0;
        #CLK_PERIOD;
        clear_acc = 0;
        #CLK_PERIOD;
        
        if (psum_out == 32'd16129) begin
            $display("  PASS: psum_out = %0d (expected 16129)", psum_out);
            tests_passed = tests_passed + 1;
        end else begin
            $display("  FAIL: psum_out = %0d (expected 16129)", psum_out);
            tests_failed = tests_failed + 1;
        end
        
        //----------------------------------------------------------------------
        // Test 6: Negative × Negative: (-5) × (-7) = 35
        //----------------------------------------------------------------------
        $display("");
        $display("────────────────────────────────────────────────────────────────");
        $display("[TEST 6] Negative × Negative: (-5) × (-7) = 35");
        $display("────────────────────────────────────────────────────────────────");
        
        // Load negative weight
        load_weight = 1;
        weight_in = -8'sd5;  // -5
        #CLK_PERIOD;
        load_weight = 0;
        
        clear_acc = 1;
        act_in = -8'sd7;  // -7
        psum_in = 32'd0;
        #CLK_PERIOD;
        clear_acc = 0;
        #CLK_PERIOD;
        
        if (psum_out == 32'd35) begin
            $display("  PASS: psum_out = %0d (expected 35)", psum_out);
            tests_passed = tests_passed + 1;
        end else begin
            $display("  FAIL: psum_out = %0d (expected 35)", psum_out);
            tests_failed = tests_failed + 1;
        end
        
        //----------------------------------------------------------------------
        // Summary
        //----------------------------------------------------------------------
        $display("");
        $display("╔════════════════════════════════════════════════════════════╗");
        $display("║                      TEST SUMMARY                          ║");
        $display("╠════════════════════════════════════════════════════════════╣");
        $display("║   Passed: %0d                                                ║", tests_passed);
        $display("║   Failed: %0d                                                ║", tests_failed);
        $display("╚════════════════════════════════════════════════════════════╝");
        
        if (tests_failed == 0) begin
            $display("");
            $display("   >>> ALL TESTS PASSED! <<<");
        end else begin
            $display("");
            $display("   >>> SOME TESTS FAILED <<<");
        end
        $display("");
        
        #(CLK_PERIOD * 5);
        $finish;
    end

    //==========================================================================
    // Waveform Dump
    //==========================================================================
    
    initial begin
        $dumpfile("mac_pe.vcd");
        $dumpvars(0, tb_mac_pe);
    end

    //==========================================================================
    // Timeout Watchdog
    //==========================================================================
    
    initial begin
        #(CLK_PERIOD * 1000);
        $display("ERROR: Simulation timeout!");
        $finish;
    end

endmodule

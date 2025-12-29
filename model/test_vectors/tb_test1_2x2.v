`timescale 1ns / 1ps
//
// Auto-generated testbench for test1_2x2
// Compares RTL cycle-by-cycle against Python model
//

module tb_test1_2x2;
    parameter CLK_PERIOD = 10;
    parameter ARRAY_SIZE = 4;
    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH = 32;
    parameter NUM_CYCLES = 25;

    // Clock and reset
    reg clk = 0;
    reg rst_n = 0;
    
    // Control signals
    reg start = 0;
    reg clear_acc = 0;
    reg [15:0] cfg_k_tiles;
    
    // Weight loading
    reg weight_load_en = 0;
    reg [$clog2(ARRAY_SIZE)-1:0] weight_load_col = 0;
    reg [ARRAY_SIZE*DATA_WIDTH-1:0] weight_load_data = 0;
    
    // Activation interface
    reg act_valid = 0;
    reg [ARRAY_SIZE*DATA_WIDTH-1:0] act_data = 0;
    wire act_ready;
    
    // Result interface
    wire result_valid;
    wire [ARRAY_SIZE*ACC_WIDTH-1:0] result_data;
    reg result_ready = 1;
    
    // Status
    wire busy, done;

    // DUT
    systolic_array #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (.*);

    // Clock generation
    always #(CLK_PERIOD/2) clk = ~clk;

    // Test data storage
    reg expected_result_valid [0:NUM_CYCLES-1];
    reg [ARRAY_SIZE*ACC_WIDTH-1:0] expected_result_data [0:NUM_CYCLES-1];
    reg input_act_valid [0:NUM_CYCLES-1];
    reg [ARRAY_SIZE*DATA_WIDTH-1:0] input_act_data [0:NUM_CYCLES-1];

    // Error tracking
    integer errors = 0;
    integer cycle_num = 0;
    integer col;
    reg signed [ACC_WIDTH-1:0] actual_val, expected_val;

    initial begin
        $display("");
        $display("========================================");
        $display("Testbench: test1_2x2");
        $display("Array size: %0dx%0d", ARRAY_SIZE, ARRAY_SIZE);
        $display("========================================");
        $display("");

        // Initialize expected data
        input_act_valid[0] = 0;
        input_act_data[0] = 32'h00000000;
        expected_result_valid[0] = 0;
        expected_result_data[0] = 128'h00000000000000000000000000000000;
        input_act_valid[1] = 1;
        input_act_data[1] = 32'h00000101;
        expected_result_valid[1] = 0;
        expected_result_data[1] = 128'h00000000000000000000000000000000;
        input_act_valid[2] = 1;
        input_act_data[2] = 32'h00000202;
        expected_result_valid[2] = 0;
        expected_result_data[2] = 128'h00000000000000000000000000000000;
        input_act_valid[3] = 0;
        input_act_data[3] = 32'h00000000;
        expected_result_valid[3] = 0;
        expected_result_data[3] = 128'h00000000000000000000000000000000;
        input_act_valid[4] = 0;
        input_act_data[4] = 32'h00000000;
        expected_result_valid[4] = 0;
        expected_result_data[4] = 128'h00000000000000000000000000000000;
        input_act_valid[5] = 0;
        input_act_data[5] = 32'h00000000;
        expected_result_valid[5] = 0;
        expected_result_data[5] = 128'h00000000000000000000000000000000;
        input_act_valid[6] = 0;
        input_act_data[6] = 32'h00000000;
        expected_result_valid[6] = 0;
        expected_result_data[6] = 128'h00000000000000000000000000000000;
        input_act_valid[7] = 0;
        input_act_data[7] = 32'h00000000;
        expected_result_valid[7] = 0;
        expected_result_data[7] = 128'h00000000000000000000000000000000;
        input_act_valid[8] = 0;
        input_act_data[8] = 32'h00000000;
        expected_result_valid[8] = 0;
        expected_result_data[8] = 128'h00000000000000000000000000000000;
        input_act_valid[9] = 0;
        input_act_data[9] = 32'h00000000;
        expected_result_valid[9] = 0;
        expected_result_data[9] = 128'h00000000000000000000000000000000;
        input_act_valid[10] = 0;
        input_act_data[10] = 32'h00000000;
        expected_result_valid[10] = 1;
        expected_result_data[10] = 128'h00000000000000000000000000000000;
        input_act_valid[11] = 0;
        input_act_data[11] = 32'h00000000;
        expected_result_valid[11] = 1;
        expected_result_data[11] = 128'h00000000000000000000000000000000;
        input_act_valid[12] = 0;
        input_act_data[12] = 32'h00000000;
        expected_result_valid[12] = 1;
        expected_result_data[12] = 128'h00000000000000000000000000000000;
        input_act_valid[13] = 0;
        input_act_data[13] = 32'h00000000;
        expected_result_valid[13] = 1;
        expected_result_data[13] = 128'h00000000000000000000000500000003;
        input_act_valid[14] = 0;
        input_act_data[14] = 32'h00000000;
        expected_result_valid[14] = 1;
        expected_result_data[14] = 128'h00000000000000000000000a00000006;
        input_act_valid[15] = 0;
        input_act_data[15] = 32'h00000000;
        expected_result_valid[15] = 1;
        expected_result_data[15] = 128'h00000000000000000000000000000000;
        input_act_valid[16] = 0;
        input_act_data[16] = 32'h00000000;
        expected_result_valid[16] = 1;
        expected_result_data[16] = 128'h00000000000000000000000000000000;
        input_act_valid[17] = 0;
        input_act_data[17] = 32'h00000000;
        expected_result_valid[17] = 1;
        expected_result_data[17] = 128'h00000000000000000000000000000000;
        input_act_valid[18] = 0;
        input_act_data[18] = 32'h00000000;
        expected_result_valid[18] = 1;
        expected_result_data[18] = 128'h00000000000000000000000000000000;
        input_act_valid[19] = 0;
        input_act_data[19] = 32'h00000000;
        expected_result_valid[19] = 1;
        expected_result_data[19] = 128'h00000000000000000000000000000000;
        input_act_valid[20] = 0;
        input_act_data[20] = 32'h00000000;
        expected_result_valid[20] = 1;
        expected_result_data[20] = 128'h00000000000000000000000000000000;
        input_act_valid[21] = 0;
        input_act_data[21] = 32'h00000000;
        expected_result_valid[21] = 1;
        expected_result_data[21] = 128'h00000000000000000000000000000000;
        input_act_valid[22] = 0;
        input_act_data[22] = 32'h00000000;
        expected_result_valid[22] = 1;
        expected_result_data[22] = 128'h00000000000000000000000000000000;
        input_act_valid[23] = 0;
        input_act_data[23] = 32'h00000000;
        expected_result_valid[23] = 1;
        expected_result_data[23] = 128'h00000000000000000000000000000000;
        input_act_valid[24] = 0;
        input_act_data[24] = 32'h00000000;
        expected_result_valid[24] = 1;
        expected_result_data[24] = 128'h00000000000000000000000000000000;

        // Reset
        #100;
        rst_n = 1;
        #20;

        // Load weights (TODO: add weight loading sequence here)
        // For now, assume weights are pre-loaded or add your weight loading code

        // Start computation
        cfg_k_tiles = NUM_CYCLES;
        start = 1;
        clear_acc = 1;
        @(posedge clk);
        start = 0;
        clear_acc = 0;

        // Run cycles and compare
        for (cycle_num = 0; cycle_num < NUM_CYCLES; cycle_num = cycle_num + 1) begin
            // Apply inputs
            act_valid = input_act_valid[cycle_num];
            act_data = input_act_data[cycle_num];
            
            @(posedge clk);
            #1; // Small delay to sample outputs
            
            // Check result_valid
            if (result_valid !== expected_result_valid[cycle_num]) begin
                $display("Cycle %0d: result_valid MISMATCH - got %b, expected %b",
                         cycle_num, result_valid, expected_result_valid[cycle_num]);
                errors = errors + 1;
            end
            
            // Check result_data (only when valid)
            if (result_valid && expected_result_valid[cycle_num]) begin
                for (col = 0; col < ARRAY_SIZE; col = col + 1) begin
                    actual_val = $signed(result_data[col*ACC_WIDTH +: ACC_WIDTH]);
                    expected_val = $signed(expected_result_data[cycle_num][col*ACC_WIDTH +: ACC_WIDTH]);
                    
                    if (actual_val !== expected_val) begin
                        $display("Cycle %0d: result_data[%0d] MISMATCH - got %0d, expected %0d",
                                 cycle_num, col, actual_val, expected_val);
                        errors = errors + 1;
                    end
                end
            end
        end

        // Summary
        $display("");
        $display("========================================");
        if (errors == 0) begin
            $display("TEST PASSED - All %0d cycles match!", NUM_CYCLES);
        end else begin
            $display("TEST FAILED - %0d errors found", errors);
        end
        $display("========================================");
        $display("");
        
        $finish;
    end

endmodule

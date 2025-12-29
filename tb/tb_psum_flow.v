`timescale 1ns / 1ps

module tb_psum_flow;
    parameter CLK_PERIOD = 10;
    parameter ARRAY_SIZE = 4;  // Small array to see full flow
    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH  = 32;

    reg  clk = 0;
    reg  rst_n = 0;
    reg  start = 0;
    reg  clear_acc = 0;
    wire busy, done;
    reg  [15:0] cfg_k_tiles = 2;
    reg  weight_load_en = 0;
    reg  [$clog2(ARRAY_SIZE)-1:0] weight_load_col = 0;
    reg  [ARRAY_SIZE*DATA_WIDTH-1:0] weight_load_data = 0;
    reg  act_valid = 0;
    reg  [ARRAY_SIZE*DATA_WIDTH-1:0] act_data = 0;
    wire act_ready;
    wire result_valid;
    wire [ARRAY_SIZE*ACC_WIDTH-1:0] result_data;
    reg  result_ready = 1;

    systolic_array #(.ARRAY_SIZE(ARRAY_SIZE), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH))
    dut (.*);

    always #(CLK_PERIOD/2) clk = ~clk;
    
    integer cycle;
    always @(posedge clk) begin
        #1;
        if (dut.state >= 2) begin
            $display("Cycle %2d | st=%d | psum_v[1][0]=%3d psum_v[2][0]=%3d psum_v[3][0]=%3d psum_v[4][0]=%3d | rvalid=%b result[0]=%d",
                     cycle, dut.state,
                     $signed(dut.psum_v[1][0]), 
                     $signed(dut.psum_v[2][0]),
                     $signed(dut.psum_v[3][0]),
                     $signed(dut.psum_v[4][0]),
                     result_valid,
                     $signed(result_data[0*ACC_WIDTH +: ACC_WIDTH]));
            cycle = cycle + 1;
        end
    end

    initial begin
        $display("\n=== PSUM FLOW TRACE (4x4 array, 2x2 multiply) ===\n");
        $display("Expected: C[0][0]=3, C[1][0]=6 should appear at psum_v[4][0]\n");
        
        cycle = 0;
        
        #100 rst_n = 1;
        @(posedge clk);
        
        // Load 2x2 weights into top-left corner
        weight_load_en = 1;
        weight_load_col = 0;
        weight_load_data = 0;
        weight_load_data[0*8 +: 8] = 1;  // B[0][0]=1
        weight_load_data[1*8 +: 8] = 2;  // B[1][0]=2
        @(posedge clk);
        weight_load_col = 1; weight_load_data = 0;
        weight_load_data[0*8 +: 8] = 2;  // B[0][1]=2
        weight_load_data[1*8 +: 8] = 3;  // B[1][1]=3
        @(posedge clk);
        weight_load_col = 2; weight_load_data = 0; @(posedge clk);
        weight_load_col = 3; weight_load_data = 0; @(posedge clk);
        weight_load_en = 0;
        @(posedge clk);
        
        // Start
        start = 1; clear_acc = 1;
        @(posedge clk);
        start = 0; clear_acc = 0;
        
        // Send 2 activations (for 2x2)
        act_valid = 1;
        act_data = 0;
        act_data[0*8 +: 8] = 1;  // A[0][0]=1
        act_data[1*8 +: 8] = 1;  // A[0][1]=1
        @(posedge clk);
        
        act_data = 0;
        act_data[0*8 +: 8] = 2;  // A[1][0]=2
        act_data[1*8 +: 8] = 2;  // A[1][1]=2
        @(posedge clk);
        
        act_valid = 0;
        act_data = 0;
        
        // Wait for propagation
        repeat(20) @(posedge clk);
        
        $display("\nFinal result_data[0] = %d (expect 3 or 6 at different times)", 
                 $signed(result_data[0*ACC_WIDTH +: ACC_WIDTH]));
        $finish;
    end
endmodule

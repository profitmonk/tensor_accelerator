`timescale 1ns / 1ps

module tb_trace_all;
    parameter CLK_PERIOD = 10;
    parameter ARRAY_SIZE = 4;
    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH  = 32;

    reg  clk = 0;
    reg  rst_n = 0;
    reg  start = 0;
    reg  clear_acc = 0;
    wire busy, done;
    reg  [15:0] cfg_k_tiles;
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
    
    // Trace PE outputs
    always @(posedge clk) begin
        if (dut.state >= 2) begin  // COMPUTE or later
            $display("Cycle %2d | State=%d | psum_v[2][0:1]=[%4d,%4d] | psum_v[4][0:1]=[%4d,%4d] | result_valid=%b",
                     cycle, dut.state,
                     $signed(dut.psum_v[2][0]), $signed(dut.psum_v[2][1]),
                     $signed(dut.psum_v[4][0]), $signed(dut.psum_v[4][1]),
                     result_valid);
            cycle = cycle + 1;
        end
    end

    initial begin
        $display("\n=== Tracing psum flow through array ===\n");
        $display("Testing C = A × B where:");
        $display("A = [1 1], B = [1 2]");
        $display("    [2 2]      [2 3]");
        $display("\nExpected: C[0][0]=3, C[0][1]=5, C[1][0]=6, C[1][1]=10\n");
        $display("psum_v[2] = after 2 rows of PEs (rows 0,1 = our 2x2 data)");
        $display("psum_v[4] = after 4 rows of PEs (bottom = output)");
        $display("");
        
        cycle = 0;
        cfg_k_tiles = 2;  // K=2
        
        #100 rst_n = 1;
        #20;
        
        // Load weights
        weight_load_en = 1;
        weight_load_col = 0;
        weight_load_data = 0;
        weight_load_data[0*8 +: 8] = 1;  // B[0][0]
        weight_load_data[1*8 +: 8] = 2;  // B[1][0]
        @(posedge clk);
        
        weight_load_col = 1;
        weight_load_data = 0;
        weight_load_data[0*8 +: 8] = 2;  // B[0][1]
        weight_load_data[1*8 +: 8] = 3;  // B[1][1]
        @(posedge clk);
        
        weight_load_col = 2; weight_load_data = 0; @(posedge clk);
        weight_load_col = 3; weight_load_data = 0; @(posedge clk);
        weight_load_en = 0; @(posedge clk);
        
        // Start
        $display("Starting computation...\n");
        start = 1; clear_acc = 1; @(posedge clk);
        start = 0; clear_acc = 0;
        
        // Send activations: A[m][k] to row k at time m
        // Time 0: m=0, send A[0][0]=1 to row 0, A[0][1]=1 to row 1
        act_valid = 1;
        act_data = 0;
        act_data[0*8 +: 8] = 1;  // A[0][0]
        act_data[1*8 +: 8] = 1;  // A[0][1]
        @(posedge clk);
        
        // Time 1: m=1, send A[1][0]=2 to row 0, A[1][1]=2 to row 1
        act_data = 0;
        act_data[0*8 +: 8] = 2;  // A[1][0]
        act_data[1*8 +: 8] = 2;  // A[1][1]
        @(posedge clk);
        
        act_valid = 0;
        act_data = 0;
        
        // Wait and observe
        repeat(20) @(posedge clk);
        
        $display("\n=== Analysis ===");
        $display("Look for psum_v[2][0]=3 (C[0][0]) and psum_v[2][1]=5 (C[0][1]) appearing");
        $display("Then psum_v[2][0]=6 (C[1][0]) and psum_v[2][1]=10 (C[1][1])");
        $display("These should propagate to psum_v[4] (output) after 2 more cycles");
        
        $finish;
    end
endmodule

`timescale 1ns / 1ps

module tb_systolic_data;
    parameter CLK_PERIOD = 10;
    parameter ARRAY_SIZE = 4;  // Smaller array for debugging
    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH  = 32;

    reg  clk;
    reg  rst_n;
    reg  start;
    reg  clear_acc;
    wire busy;
    wire done;
    reg  [15:0] cfg_k_tiles;
    reg  weight_load_en;
    reg  [$clog2(ARRAY_SIZE)-1:0] weight_load_col;
    reg  [ARRAY_SIZE*DATA_WIDTH-1:0] weight_load_data;
    reg  act_valid;
    reg  [ARRAY_SIZE*DATA_WIDTH-1:0] act_data;
    wire act_ready;
    wire result_valid;
    wire [ARRAY_SIZE*ACC_WIDTH-1:0] result_data;
    reg  result_ready;

    systolic_array #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .clear_acc(clear_acc),
        .busy(busy), .done(done), .cfg_k_tiles(cfg_k_tiles),
        .weight_load_en(weight_load_en), .weight_load_col(weight_load_col),
        .weight_load_data(weight_load_data),
        .act_valid(act_valid), .act_data(act_data), .act_ready(act_ready),
        .result_valid(result_valid), .result_data(result_data), .result_ready(result_ready)
    );

    initial begin clk = 0; forever #(CLK_PERIOD/2) clk = ~clk; end

    integer i;
    
    // Monitor PE[0][0] weight and result
    always @(posedge clk) begin
        if (weight_load_en) begin
            $display("[%0t] LOAD col=%0d data=%h", $time, weight_load_col, weight_load_data);
        end
        if (act_valid) begin
            $display("[%0t] ACT data=%h", $time, act_data);
        end
        if (result_valid) begin
            $display("[%0t] RESULT[0]=%0d RESULT[1]=%0d RESULT[2]=%0d RESULT[3]=%0d", 
                     $time, 
                     $signed(result_data[0*ACC_WIDTH +: ACC_WIDTH]),
                     $signed(result_data[1*ACC_WIDTH +: ACC_WIDTH]),
                     $signed(result_data[2*ACC_WIDTH +: ACC_WIDTH]),
                     $signed(result_data[3*ACC_WIDTH +: ACC_WIDTH]));
        end
    end

    initial begin
        $display("=== 4x4 Systolic Array Data Flow Test ===");
        $display("Computing C = A x B where:");
        $display("  A = [1 2]    B = [1 0]");
        $display("      [3 4]        [0 1]");
        $display("Expected C = [1 2]");
        $display("             [3 4]");
        
        rst_n = 0; start = 0; clear_acc = 0;
        weight_load_en = 0; weight_load_col = 0; weight_load_data = 0;
        act_valid = 0; act_data = 0; result_ready = 1;
        cfg_k_tiles = 2;
        
        #100 rst_n = 1;
        #20;
        
        // Load identity-like weights
        // B[0][0]=1, B[0][1]=0 (col 0)
        // B[1][0]=0, B[1][1]=1 (col 1)
        $display("--- Loading weights (identity matrix) ---");
        
        // Column 0: B[*][0] = [1, 0, 0, 0]
        weight_load_en = 1;
        weight_load_col = 0;
        weight_load_data = 0;
        weight_load_data[0*8 +: 8] = 8'd1;  // B[0][0] = 1
        weight_load_data[1*8 +: 8] = 8'd0;  // B[1][0] = 0
        @(posedge clk);
        
        // Column 1: B[*][1] = [0, 1, 0, 0]
        weight_load_col = 1;
        weight_load_data = 0;
        weight_load_data[0*8 +: 8] = 8'd0;  // B[0][1] = 0
        weight_load_data[1*8 +: 8] = 8'd1;  // B[1][1] = 1
        @(posedge clk);
        
        // Columns 2, 3: zeros
        weight_load_col = 2;
        weight_load_data = 0;
        @(posedge clk);
        
        weight_load_col = 3;
        weight_load_data = 0;
        @(posedge clk);
        
        weight_load_en = 0;
        @(posedge clk);
        
        // Verify PE[0][0] weight
        $display("PE[0][0].weight_reg = %0d (expected 1)", 
                 dut.pe_row[0].pe_col[0].pe_inst.weight_reg);
        $display("PE[0][1].weight_reg = %0d (expected 0)", 
                 dut.pe_row[0].pe_col[1].pe_inst.weight_reg);
        $display("PE[1][0].weight_reg = %0d (expected 0)", 
                 dut.pe_row[1].pe_col[0].pe_inst.weight_reg);
        $display("PE[1][1].weight_reg = %0d (expected 1)", 
                 dut.pe_row[1].pe_col[1].pe_inst.weight_reg);
        
        // Start
        $display("--- Starting computation ---");
        start = 1;
        clear_acc = 1;
        @(posedge clk);
        start = 0;
        clear_acc = 0;
        
        $display("state=%0d", dut.state);
        
        // Stream activations
        // A[0][*] = [1, 2, 0, 0] -> row 0
        // A[1][*] = [3, 4, 0, 0] -> row 1
        $display("--- Streaming activations (k=0) ---");
        act_valid = 1;
        act_data = 0;
        act_data[0*8 +: 8] = 8'd1;  // A[0][0] = 1
        act_data[1*8 +: 8] = 8'd3;  // A[1][0] = 3
        @(posedge clk);
        
        $display("--- Streaming activations (k=1) ---");
        act_data = 0;
        act_data[0*8 +: 8] = 8'd2;  // A[0][1] = 2
        act_data[1*8 +: 8] = 8'd4;  // A[1][1] = 4
        @(posedge clk);
        
        act_valid = 0;
        act_data = 0;
        
        // Wait for array to process
        $display("--- Waiting for array ---");
        repeat(20) begin
            @(posedge clk);
            $display("[%0t] state=%0d cycle=%0d pe[0][0].psum_out=%0d", 
                     $time, dut.state, dut.cycle_count,
                     $signed(dut.pe_row[0].pe_col[0].pe_inst.psum_out));
        end
        
        $display("=== Test Complete ===");
        $finish;
    end
endmodule

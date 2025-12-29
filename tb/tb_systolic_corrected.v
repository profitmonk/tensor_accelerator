`timescale 1ns / 1ps

// Corrected systolic array testbench with proper dataflow
module tb_systolic_corrected;
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

    integer m, k, col, row;
    integer errors;
    
    // 2x2 test matrices
    reg signed [7:0] A [0:1][0:1];
    reg signed [7:0] B [0:1][0:1];
    reg signed [31:0] C_expected [0:1][0:1];
    reg signed [31:0] C_actual [0:1];
    reg signed [31:0] result_log [0:3][0:3];  // Log results over time
    integer result_time;
    
    // Capture results over time
    always @(posedge clk) begin
        if (result_valid && result_time < 4) begin
            for (col = 0; col < ARRAY_SIZE; col = col + 1) begin
                result_log[result_time][col] = $signed(result_data[col*ACC_WIDTH +: ACC_WIDTH]);
            end
            $display("[%0t] Result row %0d: [%0d, %0d, %0d, %0d]", 
                     $time, result_time,
                     result_log[result_time][0], result_log[result_time][1],
                     result_log[result_time][2], result_log[result_time][3]);
            result_time = result_time + 1;
        end
    end

    initial begin
        $display("\n");
        $display("╔═══════════════════════════════════════════════════════════════╗");
        $display("║  CORRECTED DATAFLOW TEST                                      ║");
        $display("║                                                               ║");
        $display("║  Weight-stationary systolic array:                            ║");
        $display("║  - B[k][n] stored in PE[k][n]                                 ║");
        $display("║  - A[m][k] sent to row k at time step m                       ║");
        $display("║  - C[m][n] = sum_k A[m][k] * B[k][n] emerges at bottom col n  ║");
        $display("╚═══════════════════════════════════════════════════════════════╝\n");
        
        // Test matrices
        // A = [1 1]    B = [1 2]
        //     [2 2]        [2 3]
        A[0][0] = 1; A[0][1] = 1;
        A[1][0] = 2; A[1][1] = 2;
        
        B[0][0] = 1; B[0][1] = 2;
        B[1][0] = 2; B[1][1] = 3;
        
        // Expected C = A × B
        // C[0][0] = 1*1 + 1*2 = 3    C[0][1] = 1*2 + 1*3 = 5
        // C[1][0] = 2*1 + 2*2 = 6    C[1][1] = 2*2 + 2*3 = 10
        C_expected[0][0] = 3;  C_expected[0][1] = 5;
        C_expected[1][0] = 6;  C_expected[1][1] = 10;
        
        $display("A = [%0d %0d]    B = [%0d %0d]", A[0][0], A[0][1], B[0][0], B[0][1]);
        $display("    [%0d %0d]        [%0d %0d]", A[1][0], A[1][1], B[1][0], B[1][1]);
        $display("");
        $display("Expected C = [%0d %0d]", C_expected[0][0], C_expected[0][1]);
        $display("             [%0d %0d]", C_expected[1][0], C_expected[1][1]);
        $display("");
        
        result_time = 0;
        cfg_k_tiles = 2;  // K dimension = 2
        
        #100 rst_n = 1;
        #20;
        
        //----------------------------------------------------------------------
        // Load weights: B[k][n] -> PE[k][n]
        //----------------------------------------------------------------------
        $display("=== Loading Weights ===");
        
        // Column 0: B[*][0] = [B[0][0], B[1][0], 0, 0] = [1, 2, 0, 0]
        weight_load_en = 1;
        weight_load_col = 0;
        weight_load_data = 0;
        weight_load_data[0*8 +: 8] = B[0][0];  // PE[0][0] = 1
        weight_load_data[1*8 +: 8] = B[1][0];  // PE[1][0] = 2
        @(posedge clk);
        
        // Column 1: B[*][1] = [B[0][1], B[1][1], 0, 0] = [2, 3, 0, 0]
        weight_load_col = 1;
        weight_load_data = 0;
        weight_load_data[0*8 +: 8] = B[0][1];  // PE[0][1] = 2
        weight_load_data[1*8 +: 8] = B[1][1];  // PE[1][1] = 3
        @(posedge clk);
        
        // Columns 2, 3: zeros
        weight_load_col = 2; weight_load_data = 0; @(posedge clk);
        weight_load_col = 3; weight_load_data = 0; @(posedge clk);
        
        weight_load_en = 0;
        @(posedge clk);
        
        $display("  PE[0][0].weight = %0d, PE[0][1].weight = %0d", 
                 dut.pe_row[0].pe_col[0].pe_inst.weight_reg,
                 dut.pe_row[0].pe_col[1].pe_inst.weight_reg);
        $display("  PE[1][0].weight = %0d, PE[1][1].weight = %0d",
                 dut.pe_row[1].pe_col[0].pe_inst.weight_reg,
                 dut.pe_row[1].pe_col[1].pe_inst.weight_reg);
        
        //----------------------------------------------------------------------
        // Start computation
        //----------------------------------------------------------------------
        $display("\n=== Starting Computation ===");
        start = 1;
        clear_acc = 1;
        @(posedge clk);
        start = 0;
        clear_acc = 0;
        
        //----------------------------------------------------------------------
        // Stream activations: A[m][k] to row k at time step m
        // This is the CORRECTED dataflow!
        //----------------------------------------------------------------------
        $display("\n=== Streaming Activations (corrected dataflow) ===");
        
        // Time step 0: compute output row m=0
        // Send A[0][k] to row k: A[0][0]=1 to row 0, A[0][1]=1 to row 1
        $display("m=0: Sending A[0][*] = [%0d, %0d] to rows [0, 1]", A[0][0], A[0][1]);
        act_valid = 1;
        act_data = 0;
        act_data[0*8 +: 8] = A[0][0];  // A[0][0] = 1 to row 0
        act_data[1*8 +: 8] = A[0][1];  // A[0][1] = 1 to row 1
        @(posedge clk);
        
        // Time step 1: compute output row m=1
        // Send A[1][k] to row k: A[1][0]=2 to row 0, A[1][1]=2 to row 1
        $display("m=1: Sending A[1][*] = [%0d, %0d] to rows [0, 1]", A[1][0], A[1][1]);
        act_data = 0;
        act_data[0*8 +: 8] = A[1][0];  // A[1][0] = 2 to row 0
        act_data[1*8 +: 8] = A[1][1];  // A[1][1] = 2 to row 1
        @(posedge clk);
        
        act_valid = 0;
        act_data = 0;
        
        //----------------------------------------------------------------------
        // Wait for results
        //----------------------------------------------------------------------
        $display("\n=== Waiting for Results ===");
        repeat(20) @(posedge clk);
        
        //----------------------------------------------------------------------
        // Verify
        //----------------------------------------------------------------------
        $display("\n=== Verification ===");
        errors = 0;
        
        if (result_time >= 2) begin
            // First result row should be C[0][*]
            if (result_log[0][0] == C_expected[0][0] && result_log[0][1] == C_expected[0][1]) begin
                $display("  C[0][*]: [%0d, %0d] - PASS", result_log[0][0], result_log[0][1]);
            end else begin
                $display("  C[0][*]: [%0d, %0d] expected [%0d, %0d] - FAIL", 
                         result_log[0][0], result_log[0][1], 
                         C_expected[0][0], C_expected[0][1]);
                errors = errors + 1;
            end
            
            // Second result row should be C[1][*]
            if (result_log[1][0] == C_expected[1][0] && result_log[1][1] == C_expected[1][1]) begin
                $display("  C[1][*]: [%0d, %0d] - PASS", result_log[1][0], result_log[1][1]);
            end else begin
                $display("  C[1][*]: [%0d, %0d] expected [%0d, %0d] - FAIL",
                         result_log[1][0], result_log[1][1],
                         C_expected[1][0], C_expected[1][1]);
                errors = errors + 1;
            end
        end else begin
            $display("  ERROR: Only captured %0d result rows, expected 2", result_time);
            errors = errors + 1;
        end
        
        $display("\n╔═══════════════════════════════════════════════════════════════╗");
        if (errors == 0)
            $display("║                     TEST PASSED                               ║");
        else
            $display("║                     TEST FAILED                               ║");
        $display("╚═══════════════════════════════════════════════════════════════╝\n");
        
        $finish;
    end
endmodule

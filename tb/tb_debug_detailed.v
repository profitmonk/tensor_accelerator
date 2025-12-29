`timescale 1ns / 1ps

// Detailed debug testbench - traces data through the systolic array
module tb_debug_detailed;
    parameter CLK_PERIOD = 10;
    parameter ARRAY_SIZE = 4;  // Use small array for visibility
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
    
    // Detailed monitoring of internal signals
    always @(posedge clk) begin
        if (dut.state == 2 || dut.state == 3) begin  // COMPUTE or DRAIN
            $display("\n=== Cycle %0d (state=%0d) ===", cycle, dut.state);
            cycle = cycle + 1;
            
            // Show activation inputs to each row
            $display("Activation inputs (act_h[row][0]):");
            $display("  Row 0: %0d", $signed(dut.act_h[0][0]));
            $display("  Row 1: %0d", $signed(dut.act_h[1][0]));
            $display("  Row 2: %0d", $signed(dut.act_h[2][0]));
            $display("  Row 3: %0d", $signed(dut.act_h[3][0]));
            
            // Show weights in first column
            $display("Weights in column 0:");
            $display("  PE[0][0].weight = %0d", $signed(dut.pe_row[0].pe_col[0].pe_inst.weight_reg));
            $display("  PE[1][0].weight = %0d", $signed(dut.pe_row[1].pe_col[0].pe_inst.weight_reg));
            $display("  PE[2][0].weight = %0d", $signed(dut.pe_row[2].pe_col[0].pe_inst.weight_reg));
            $display("  PE[3][0].weight = %0d", $signed(dut.pe_row[3].pe_col[0].pe_inst.weight_reg));
            
            // Show psum outputs from bottom row (this is what becomes result_data)
            $display("Bottom row psum (psum_v[4][col] = result_data):");
            $display("  Col 0: %0d", $signed(dut.psum_v[4][0]));
            $display("  Col 1: %0d", $signed(dut.psum_v[4][1]));
            $display("  Col 2: %0d", $signed(dut.psum_v[4][2]));
            $display("  Col 3: %0d", $signed(dut.psum_v[4][3]));
            
            // Show intermediate psum at row 1 (after first row of PEs)
            $display("Row 1 psum inputs (psum_v[1][col]):");
            $display("  Col 0: %0d", $signed(dut.psum_v[1][0]));
            $display("  Col 1: %0d", $signed(dut.psum_v[1][1]));
            
            if (result_valid) begin
                $display(">>> result_valid! <<<");
            end
        end
    end

    initial begin
        $display("\n");
        $display("╔═══════════════════════════════════════════════════════════════╗");
        $display("║  DETAILED SYSTOLIC ARRAY DEBUG                                ║");
        $display("║  Testing 2x2 matrix multiply: C = A × B                       ║");
        $display("║                                                               ║");
        $display("║  A = [1 1]    B = [1 2]    Expected C = [3  5]                ║");
        $display("║      [2 2]        [2 3]                 [6 10]               ║");
        $display("╚═══════════════════════════════════════════════════════════════╝");
        $display("");
        
        cycle = 0;
        cfg_k_tiles = 2;  // K=2 for 2x2 multiply
        
        #100 rst_n = 1;
        #20;
        
        // Load weights (B matrix) column by column
        // B = [1 2]
        //     [2 3]
        // Column 0 of B: [1, 2] goes to PE[0][0], PE[1][0]
        // Column 1 of B: [2, 3] goes to PE[0][1], PE[1][1]
        
        $display("=== LOADING WEIGHTS ===");
        $display("Loading B column 0: [1, 2, 0, 0]");
        weight_load_en = 1;
        weight_load_col = 0;
        weight_load_data = 0;
        weight_load_data[0*8 +: 8] = 8'd1;  // B[0][0] = 1
        weight_load_data[1*8 +: 8] = 8'd2;  // B[1][0] = 2
        @(posedge clk);
        
        $display("Loading B column 1: [2, 3, 0, 0]");
        weight_load_col = 1;
        weight_load_data = 0;
        weight_load_data[0*8 +: 8] = 8'd2;  // B[0][1] = 2
        weight_load_data[1*8 +: 8] = 8'd3;  // B[1][1] = 3
        @(posedge clk);
        
        $display("Loading B column 2: [0, 0, 0, 0]");
        weight_load_col = 2;
        weight_load_data = 0;
        @(posedge clk);
        
        $display("Loading B column 3: [0, 0, 0, 0]");
        weight_load_col = 3;
        weight_load_data = 0;
        @(posedge clk);
        
        weight_load_en = 0;
        @(posedge clk);
        
        // Verify weights loaded correctly
        $display("\n=== VERIFY WEIGHT LOADING ===");
        $display("PE[0][0].weight = %0d (expect 1)", dut.pe_row[0].pe_col[0].pe_inst.weight_reg);
        $display("PE[1][0].weight = %0d (expect 2)", dut.pe_row[1].pe_col[0].pe_inst.weight_reg);
        $display("PE[0][1].weight = %0d (expect 2)", dut.pe_row[0].pe_col[1].pe_inst.weight_reg);
        $display("PE[1][1].weight = %0d (expect 3)", dut.pe_row[1].pe_col[1].pe_inst.weight_reg);
        
        // Start computation
        $display("\n=== STARTING COMPUTATION ===");
        start = 1;
        clear_acc = 1;
        @(posedge clk);
        start = 0;
        clear_acc = 0;
        
        // Stream A matrix row by row (but as columns of K dimension)
        // A = [1 1]  -> Row 0 gets A[0][k], Row 1 gets A[1][k]
        //     [2 2]
        // K=0: send A[*][0] = [1, 2] to rows 0, 1
        // K=1: send A[*][1] = [1, 2] to rows 0, 1
        
        $display("\n=== STREAMING ACTIVATIONS ===");
        $display("K=0: Sending A[*][0] = [1, 2, 0, 0] to rows");
        act_valid = 1;
        act_data = 0;
        act_data[0*8 +: 8] = 8'd1;  // A[0][0] = 1
        act_data[1*8 +: 8] = 8'd2;  // A[1][0] = 2
        @(posedge clk);
        
        $display("K=1: Sending A[*][1] = [1, 2, 0, 0] to rows");
        act_data = 0;
        act_data[0*8 +: 8] = 8'd1;  // A[0][1] = 1
        act_data[1*8 +: 8] = 8'd2;  // A[1][1] = 2
        @(posedge clk);
        
        act_valid = 0;
        act_data = 0;
        
        // Let computation complete
        $display("\n=== WAITING FOR COMPLETION ===");
        repeat(15) @(posedge clk);
        
        // Final results
        $display("\n=== FINAL RESULTS ===");
        $display("result_data[col]:");
        $display("  Col 0: %0d (expect C[*][0])", $signed(result_data[0*ACC_WIDTH +: ACC_WIDTH]));
        $display("  Col 1: %0d (expect C[*][1])", $signed(result_data[1*ACC_WIDTH +: ACC_WIDTH]));
        $display("  Col 2: %0d", $signed(result_data[2*ACC_WIDTH +: ACC_WIDTH]));
        $display("  Col 3: %0d", $signed(result_data[3*ACC_WIDTH +: ACC_WIDTH]));
        
        $display("\n=== EXPECTED ===");
        $display("C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] = 1*1 + 1*2 = 3");
        $display("C[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1] = 1*2 + 1*3 = 5");
        $display("C[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0] = 2*1 + 2*2 = 6");
        $display("C[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1] = 2*2 + 2*3 = 10");
        
        $display("\n=== TEST COMPLETE ===\n");
        $finish;
    end
endmodule

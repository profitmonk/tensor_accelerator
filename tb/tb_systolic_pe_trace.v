`timescale 1ns / 1ps

module tb_systolic_pe_trace;
    parameter CLK_PERIOD = 10;
    parameter ARRAY_SIZE = 4;
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
    
    // Trace PE[0][0] and PE[1][0] internals
    always @(posedge clk) begin
        #1;
        if (dut.state >= 2) begin
            $display("Cycle | PE[0][0]: en=%b act_in=%0d w=%0d prod=%0d psum_out=%0d | PE[1][0]: en=%b act_in=%0d w=%0d prod=%0d psum_in=%0d psum_out=%0d",
                dut.pe_row[0].pe_col[0].pe_inst.enable,
                dut.pe_row[0].pe_col[0].pe_inst.act_in,
                dut.pe_row[0].pe_col[0].pe_inst.weight_reg,
                dut.pe_row[0].pe_col[0].pe_inst.product,
                dut.pe_row[0].pe_col[0].pe_inst.psum_out,
                dut.pe_row[1].pe_col[0].pe_inst.enable,
                dut.pe_row[1].pe_col[0].pe_inst.act_in,
                dut.pe_row[1].pe_col[0].pe_inst.weight_reg,
                dut.pe_row[1].pe_col[0].pe_inst.product,
                dut.pe_row[1].pe_col[0].pe_inst.psum_in,
                dut.pe_row[1].pe_col[0].pe_inst.psum_out);
        end
    end

    initial begin
        $display("\n=== Tracing PE[0][0] and PE[1][0] ===\n");
        $display("C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] = 1*1 + 1*2 = 3");
        $display("");
        
        #100 rst_n = 1;
        @(posedge clk);
        
        // Load weights: B[0][0]=1, B[1][0]=2
        weight_load_en = 1;
        weight_load_col = 0;
        weight_load_data = 0;
        weight_load_data[0*8 +: 8] = 8'd1;  // PE[0][0] weight
        weight_load_data[1*8 +: 8] = 8'd2;  // PE[1][0] weight
        @(posedge clk);
        weight_load_col = 1; weight_load_data = 0; @(posedge clk);
        weight_load_col = 2; @(posedge clk);
        weight_load_col = 3; @(posedge clk);
        weight_load_en = 0;
        @(posedge clk);
        
        $display("Weights loaded: PE[0][0]=%0d PE[1][0]=%0d",
                 dut.pe_row[0].pe_col[0].pe_inst.weight_reg,
                 dut.pe_row[1].pe_col[0].pe_inst.weight_reg);
        $display("");
        
        // Start
        start = 1; clear_acc = 1;
        @(posedge clk);
        start = 0; clear_acc = 0;
        
        // Send activations: A[0][0]=1 to row 0, A[0][1]=1 to row 1
        $display("Sending: A[0][0]=1 to row 0, A[0][1]=1 to row 1");
        act_valid = 1;
        act_data = 0;
        act_data[0*8 +: 8] = 8'd1;  // A[0][0] to row 0
        act_data[1*8 +: 8] = 8'd1;  // A[0][1] to row 1
        @(posedge clk);
        
        // Send: A[1][0]=2 to row 0, A[1][1]=2 to row 1
        $display("Sending: A[1][0]=2 to row 0, A[1][1]=2 to row 1");
        act_data = 0;
        act_data[0*8 +: 8] = 8'd2;
        act_data[1*8 +: 8] = 8'd2;
        @(posedge clk);
        
        act_valid = 0;
        act_data = 0;
        
        $display("\nWaiting for propagation...\n");
        repeat(15) @(posedge clk);
        
        $display("\npsum_v[2][0] = %0d (should be C[m][0] for various m)", 
                 $signed(dut.psum_v[2][0]));
        
        $finish;
    end
endmodule

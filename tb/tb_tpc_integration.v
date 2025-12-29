//==============================================================================
// Tensor Processing Cluster (TPC) Integration Test
//
// Tests the complete flow:
// 1. Load instructions into instruction memory
// 2. Load weights/activations into data SRAM
// 3. LCP fetches and executes instructions
// 4. MXU (systolic array) performs GEMM
// 5. Results written back to SRAM
// 6. Verify results
//==============================================================================
`timescale 1ns / 1ps

module tb_tpc_integration;

    parameter CLK = 10;
    parameter ARRAY_SIZE = 4;  // Use smaller array for faster simulation
    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH = 32;
    parameter SRAM_DEPTH = 256;
    
    reg clk = 0;
    reg rst_n = 0;
    
    always #(CLK/2) clk = ~clk;
    
    //==========================================================================
    // LCP Signals
    //==========================================================================
    reg lcp_start = 0;
    reg [19:0] lcp_start_pc = 0;
    wire lcp_busy, lcp_done, lcp_error;
    
    wire [19:0] imem_addr;
    reg [127:0] imem_data;
    wire imem_re;
    reg imem_valid = 0;
    
    wire [127:0] mxu_cmd;
    wire mxu_valid;
    reg mxu_ready = 0;
    reg mxu_done_to_lcp = 0;
    
    // Unused interfaces
    wire [127:0] vpu_cmd, dma_cmd;
    wire vpu_valid, dma_valid, sync_request;
    reg vpu_ready = 1, vpu_done = 0;
    reg dma_ready = 1, dma_done = 0;
    reg global_sync_in = 0, sync_grant = 0;
    
    //==========================================================================
    // MXU (Systolic Array) Signals
    //==========================================================================
    reg mxu_start = 0;
    reg mxu_clear_acc = 0;
    wire mxu_busy, mxu_done;
    reg [15:0] mxu_cfg_k_tiles;
    
    reg weight_load_en = 0;
    reg [$clog2(ARRAY_SIZE)-1:0] weight_load_col = 0;
    reg [ARRAY_SIZE*DATA_WIDTH-1:0] weight_load_data = 0;
    
    reg act_valid = 0;
    reg [ARRAY_SIZE*DATA_WIDTH-1:0] act_data = 0;
    wire act_ready;
    
    wire result_valid;
    wire [ARRAY_SIZE*ACC_WIDTH-1:0] result_data;
    
    //==========================================================================
    // Instruction Memory
    //==========================================================================
    reg [127:0] instr_mem [0:63];
    
    always @(posedge clk) begin
        imem_valid <= imem_re;
        if (imem_re) imem_data <= instr_mem[imem_addr[5:0]];
    end
    
    //==========================================================================
    // Data SRAM (simplified)
    //==========================================================================
    reg [ARRAY_SIZE*DATA_WIDTH-1:0] weight_sram [0:SRAM_DEPTH-1];
    reg [ARRAY_SIZE*DATA_WIDTH-1:0] act_sram [0:SRAM_DEPTH-1];
    reg [ARRAY_SIZE*ACC_WIDTH-1:0] result_sram [0:SRAM_DEPTH-1];
    
    //==========================================================================
    // LCP Instance
    //==========================================================================
    local_cmd_processor lcp (
        .clk            (clk),
        .rst_n          (rst_n),
        .start          (lcp_start),
        .start_pc       (lcp_start_pc),
        .busy           (lcp_busy),
        .done           (lcp_done),
        .error          (lcp_error),
        .imem_addr      (imem_addr),
        .imem_data      (imem_data),
        .imem_re        (imem_re),
        .imem_valid     (imem_valid),
        .mxu_cmd        (mxu_cmd),
        .mxu_valid      (mxu_valid),
        .mxu_ready      (mxu_ready),
        .mxu_done       (mxu_done_to_lcp),
        .vpu_cmd        (vpu_cmd),
        .vpu_valid      (vpu_valid),
        .vpu_ready      (vpu_ready),
        .vpu_done       (vpu_done),
        .dma_cmd        (dma_cmd),
        .dma_valid      (dma_valid),
        .dma_ready      (dma_ready),
        .dma_done       (dma_done),
        .global_sync_in (global_sync_in),
        .sync_request   (sync_request),
        .sync_grant     (sync_grant)
    );
    
    //==========================================================================
    // MXU (Systolic Array) Instance
    //==========================================================================
    systolic_array #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) mxu (
        .clk              (clk),
        .rst_n            (rst_n),
        .start            (mxu_start),
        .clear_acc        (mxu_clear_acc),
        .busy             (mxu_busy),
        .done             (mxu_done),
        .cfg_k_tiles      (mxu_cfg_k_tiles),
        .weight_load_en   (weight_load_en),
        .weight_load_col  (weight_load_col),
        .weight_load_data (weight_load_data),
        .act_valid        (act_valid),
        .act_data         (act_data),
        .act_ready        (act_ready),
        .result_valid     (result_valid),
        .result_data      (result_data),
        .result_ready     (1'b1)
    );
    
    //==========================================================================
    // MXU Controller (bridges LCP commands to systolic array)
    //==========================================================================
    localparam MXU_IDLE = 0, MXU_LOAD_W = 1, MXU_STREAM_A = 2, MXU_DRAIN = 3, MXU_DONE_ST = 4;
    reg [2:0] mxu_state = MXU_IDLE;
    reg [15:0] mxu_cycle = 0;
    reg [15:0] mxu_k_dim = 0;
    reg [15:0] weight_addr = 0, act_addr = 0, result_addr = 0;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mxu_state <= MXU_IDLE;
            mxu_ready <= 1;
            mxu_done_to_lcp <= 0;
            mxu_start <= 0;
            mxu_clear_acc <= 0;
            weight_load_en <= 0;
            act_valid <= 0;
            mxu_cycle <= 0;
        end else begin
            mxu_start <= 0;
            mxu_clear_acc <= 0;
            mxu_done_to_lcp <= 0;
            weight_load_en <= 0;
            act_valid <= 0;
            
            case (mxu_state)
                MXU_IDLE: begin
                    mxu_ready <= 1;
                    if (mxu_valid && mxu_ready) begin
                        mxu_ready <= 0;
                        // Decode command: k_tiles in [47:32], dst in [63:48], src1(W) in [79:64], src0(A) in [95:80]
                        mxu_k_dim <= mxu_cmd[47:32];
                        result_addr <= mxu_cmd[63:48];
                        weight_addr <= mxu_cmd[79:64];
                        act_addr <= mxu_cmd[95:80];
                        mxu_cycle <= 0;
                        mxu_state <= MXU_LOAD_W;
                        $display("  [MXU] Received command: k=%0d, A@%h, W@%h, C@%h", 
                                 mxu_cmd[47:32], mxu_cmd[95:80], mxu_cmd[79:64], mxu_cmd[63:48]);
                    end
                end
                
                MXU_LOAD_W: begin
                    weight_load_en <= 1;
                    weight_load_col <= mxu_cycle[$clog2(ARRAY_SIZE)-1:0];
                    weight_load_data <= weight_sram[weight_addr + mxu_cycle];
                    mxu_cycle <= mxu_cycle + 1;
                    
                    if (mxu_cycle >= ARRAY_SIZE - 1) begin
                        mxu_cycle <= 0;
                        mxu_start <= 1;
                        mxu_clear_acc <= 1;
                        mxu_cfg_k_tiles <= mxu_k_dim + 3*ARRAY_SIZE;
                        mxu_state <= MXU_STREAM_A;
                    end
                end
                
                MXU_STREAM_A: begin
                    if (mxu_cycle < mxu_k_dim) begin
                        act_valid <= 1;
                        act_data <= act_sram[act_addr + mxu_cycle];
                        mxu_cycle <= mxu_cycle + 1;
                    end else if (mxu_done) begin
                        mxu_cycle <= 0;
                        mxu_state <= MXU_DRAIN;
                    end
                end
                
                MXU_DRAIN: begin
                    if (result_valid) begin
                        result_sram[result_addr + mxu_cycle] <= result_data;
                        $display("  [MXU] Result[%0d] = %h", mxu_cycle, result_data);
                        mxu_cycle <= mxu_cycle + 1;
                    end
                    if (mxu_cycle >= ARRAY_SIZE) begin
                        mxu_state <= MXU_DONE_ST;
                    end
                end
                
                MXU_DONE_ST: begin
                    mxu_done_to_lcp <= 1;
                    mxu_state <= MXU_IDLE;
                end
            endcase
        end
    end
    
    //==========================================================================
    // Opcodes
    //==========================================================================
    localparam OP_NOP = 8'h00, OP_TENSOR = 8'h01, OP_SYNC = 8'h04, OP_HALT = 8'hFF;
    localparam SYNC_MXU = 8'h01;
    
    //==========================================================================
    // Test
    //==========================================================================
    integer i, j, errors;
    reg signed [ACC_WIDTH-1:0] expected [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    reg signed [ACC_WIDTH-1:0] actual_val;
    
    initial begin
        $display("");
        $display("╔════════════════════════════════════════════════════════════╗");
        $display("║     TPC Integration Test (LCP + Systolic Array)            ║");
        $display("╚════════════════════════════════════════════════════════════╝");
        $display("");
        
        errors = 0;
        
        // Initialize memories
        for (i = 0; i < SRAM_DEPTH; i = i + 1) begin
            weight_sram[i] = 0;
            act_sram[i] = 0;
            result_sram[i] = 0;
        end
        for (i = 0; i < 64; i = i + 1) instr_mem[i] = {OP_NOP, 120'd0};
        
        // Setup test data: 2x2 GEMM
        // A = [[1,1],[2,2]], B = [[1,2],[2,3]]
        // Expected C = [[3,5],[6,10]]
        
        // Weights B (column-packed for loading): B[k][n] -> weight_sram[n][k*8 +: 8]
        // Col 0: B[0][0]=1, B[1][0]=2
        weight_sram[0] = {24'd0, 8'd2, 8'd1};  // [-, -, B[1][0], B[0][0]]
        // Col 1: B[0][1]=2, B[1][1]=3  
        weight_sram[1] = {24'd0, 8'd3, 8'd2};  // [-, -, B[1][1], B[0][1]]
        weight_sram[2] = 0;
        weight_sram[3] = 0;
        
        // Activations A (row-packed): A[m][k] -> act_sram[m][k*8 +: 8]
        // Row 0: A[0][0]=1, A[0][1]=1 -> send to row 0 and row 1 at time 0
        act_sram[0] = {24'd0, 8'd1, 8'd1};  // [-, -, A[0][1], A[0][0]]
        // Row 1: A[1][0]=2, A[1][1]=2
        act_sram[1] = {24'd0, 8'd2, 8'd2};  // [-, -, A[1][1], A[1][0]]
        
        // Expected results
        expected[0][0] = 3;  expected[0][1] = 5;
        expected[1][0] = 6;  expected[1][1] = 10;
        
        // Program: TENSOR(A@0, W@0, C@16, k=2), SYNC_MXU, HALT
        // Format: {opcode[127:120], subop[119:112], unused[111:96], src0(A)[95:80], src1(W)[79:64], dst[63:48], k_tiles[47:32], unused[31:0]}
        instr_mem[0] = {OP_TENSOR, 8'h00, 16'd0, 16'd0, 16'd0, 16'd16, 16'd2, 32'd0};
        instr_mem[1] = {OP_SYNC, SYNC_MXU, 112'd0};
        instr_mem[2] = {OP_HALT, 120'd0};
        
        // Reset
        #(CLK * 5);
        rst_n = 1;
        #(CLK * 5);
        
        $display("[TEST] 2x2 GEMM via LCP");
        $display("  A = [[1,1],[2,2]]");
        $display("  B = [[1,2],[2,3]]");
        $display("  Expected C = [[3,5],[6,10]]");
        $display("");
        
        // Start LCP
        @(negedge clk);
        lcp_start_pc = 0;
        lcp_start = 1;
        @(posedge clk);
        @(negedge clk);
        lcp_start = 0;
        
        // Wait for completion
        i = 0;
        while (!lcp_done && !lcp_error && i < 500) begin
            @(posedge clk);
            i = i + 1;
        end
        
        if (lcp_done && !lcp_error) begin
            $display("");
            $display("LCP completed successfully");
            
            // Check results
            $display("");
            $display("Verifying results:");
            for (i = 0; i < 2; i = i + 1) begin
                for (j = 0; j < 2; j = j + 1) begin
                    actual_val = $signed(result_sram[16 + i][j*ACC_WIDTH +: ACC_WIDTH]);
                    if (actual_val == expected[i][j]) begin
                        $display("  C[%0d][%0d] = %0d (expected %0d) OK", i, j, actual_val, expected[i][j]);
                    end else begin
                        $display("  C[%0d][%0d] = %0d (expected %0d) MISMATCH", i, j, actual_val, expected[i][j]);
                        errors = errors + 1;
                    end
                end
            end
        end else begin
            $display("LCP failed: done=%b error=%b timeout=%b", lcp_done, lcp_error, i >= 500);
            errors = errors + 1;
        end
        
        $display("");
        $display("════════════════════════════════════════");
        if (errors == 0) begin
            $display(">>> TEST PASSED! <<<");
        end else begin
            $display(">>> TEST FAILED: %0d errors <<<", errors);
        end
        $display("");
        
        $finish;
    end
    
    initial begin $dumpfile("tpc_integration.vcd"); $dumpvars(0, tb_tpc_integration); end
    initial begin #(CLK * 10000); $display("TIMEOUT!"); $finish; end
    
endmodule

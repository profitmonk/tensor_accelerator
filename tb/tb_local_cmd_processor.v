//==============================================================================
// Local Command Processor (LCP) Testbench - Fixed Timing
//==============================================================================
`timescale 1ns / 1ps

module tb_local_cmd_processor;
    parameter CLK = 10;
    
    reg clk = 0;
    reg rst_n = 0;
    reg start = 0;
    reg [19:0] start_pc = 0;
    wire busy, done, error;
    
    wire [19:0] imem_addr;
    reg [127:0] imem_data = 0;
    wire imem_re;
    reg imem_valid = 0;
    
    wire [127:0] mxu_cmd, vpu_cmd, dma_cmd;
    wire mxu_valid, vpu_valid, dma_valid;
    reg mxu_ready = 1, vpu_ready = 1, dma_ready = 1;
    reg mxu_done = 0, vpu_done = 0, dma_done = 0;
    
    reg global_sync_in = 0;
    wire sync_request;
    reg sync_grant = 0;
    
    local_cmd_processor dut (.*);
    
    always #(CLK/2) clk = ~clk;
    
    // Opcodes
    localparam OP_NOP = 8'h00, OP_TENSOR = 8'h01, OP_DMA = 8'h03;
    localparam OP_SYNC = 8'h04, OP_LOOP = 8'h05, OP_ENDLOOP = 8'h06;
    localparam OP_BARRIER = 8'h07, OP_HALT = 8'hFF;
    localparam SYNC_MXU = 8'h01, SYNC_DMA = 8'h03;
    
    // Instruction memory
    reg [127:0] imem [0:31];
    
    always @(posedge clk) begin
        imem_valid <= imem_re;
        if (imem_re) imem_data <= imem[imem_addr[4:0]];
    end
    
    // Execution unit simulators with proper done generation
    reg [7:0] mxu_timer = 0, dma_timer = 0;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mxu_done <= 0; dma_done <= 0;
            mxu_timer <= 0; dma_timer <= 0;
        end else begin
            mxu_done <= 0; dma_done <= 0;
            
            if (mxu_valid && mxu_ready) mxu_timer <= 5;
            else if (mxu_timer > 1) mxu_timer <= mxu_timer - 1;
            else if (mxu_timer == 1) begin mxu_done <= 1; mxu_timer <= 0; end
            
            if (dma_valid && dma_ready) dma_timer <= 5;
            else if (dma_timer > 1) dma_timer <= dma_timer - 1;
            else if (dma_timer == 1) begin dma_done <= 1; dma_timer <= 0; end
        end
    end
    
    integer errors = 0, mxu_cnt = 0, dma_cnt = 0, i;
    
    always @(posedge clk) begin
        if (mxu_valid && mxu_ready) begin mxu_cnt = mxu_cnt + 1; $display("  MXU cmd #%0d", mxu_cnt); end
        if (dma_valid && dma_ready) begin dma_cnt = dma_cnt + 1; $display("  DMA cmd #%0d", dma_cnt); end
    end
    
    task do_start;
        begin
            @(negedge clk);
            start = 1;
            @(posedge clk);
            @(negedge clk);
            start = 0;
        end
    endtask
    
    task wait_done;
        input integer timeout;
        integer cnt;
        begin
            cnt = 0;
            while (!done && !error && cnt < timeout) begin
                @(posedge clk);
                cnt = cnt + 1;
            end
        end
    endtask
    
    initial begin
        $display("");
        $display("╔════════════════════════════════════════════════════════════╗");
        $display("║     Local Command Processor (LCP) Testbench                ║");
        $display("╚════════════════════════════════════════════════════════════╝");
        
        // Init
        for (i = 0; i < 32; i = i + 1) imem[i] = {OP_NOP, 120'd0};
        
        #(CLK * 5); rst_n = 1; #(CLK * 2);
        
        // TEST 1: NOP + HALT
        $display("\n[TEST 1] NOP + HALT");
        imem[0] = {OP_NOP, 120'd0};
        imem[1] = {OP_HALT, 120'd0};
        start_pc = 0;
        do_start();
        wait_done(100);
        if (done && !error) $display("  PASS"); 
        else begin $display("  FAIL"); errors = errors + 1; end
        #(CLK * 3);
        
        // TEST 2: TENSOR command
        $display("\n[TEST 2] TENSOR Command");
        mxu_cnt = 0;
        imem[0] = {OP_TENSOR, 8'h00, 48'd0, 16'd16, 16'h300, 16'h200, 16'h100};
        imem[1] = {OP_SYNC, SYNC_MXU, 112'd0};
        imem[2] = {OP_HALT, 120'd0};
        start_pc = 0;
        do_start();
        wait_done(200);
        if (done && !error && mxu_cnt == 1) $display("  PASS");
        else begin $display("  FAIL (mxu=%0d)", mxu_cnt); errors = errors + 1; end
        #(CLK * 3);
        
        // TEST 3: Loop
        $display("\n[TEST 3] Loop (3 iterations)");
        mxu_cnt = 0;
        imem[0] = {OP_LOOP, 8'd0, 64'd0, 16'd3, 32'd0};
        imem[1] = {OP_TENSOR, 120'd0};
        imem[2] = {OP_ENDLOOP, 120'd0};
        imem[3] = {OP_HALT, 120'd0};
        start_pc = 0;
        do_start();
        wait_done(500);
        if (done && !error && mxu_cnt == 3) $display("  PASS");
        else begin $display("  FAIL (mxu=%0d)", mxu_cnt); errors = errors + 1; end
        #(CLK * 3);
        
        // TEST 4: Barrier
        $display("\n[TEST 4] Barrier");
        imem[0] = {OP_BARRIER, 120'd0};
        imem[1] = {OP_HALT, 120'd0};
        start_pc = 0;
        do_start();
        
        // Wait for sync request
        i = 0;
        while (!sync_request && i < 50) begin @(posedge clk); i = i + 1; end
        
        if (sync_request) begin
            $display("  Sync request received");
            #(CLK * 5);
            @(negedge clk); sync_grant = 1; @(posedge clk); @(negedge clk); sync_grant = 0;
            wait_done(100);
            if (done && !error) $display("  PASS");
            else begin $display("  FAIL"); errors = errors + 1; end
        end else begin $display("  FAIL: no sync_request"); errors = errors + 1; end
        
        // Summary
        $display("\n════════════════════════════════════════");
        $display("Errors: %0d / 4 tests", errors);
        if (errors == 0) $display(">>> ALL TESTS PASSED! <<<");
        else $display(">>> SOME TESTS FAILED <<<");
        $display("");
        $finish;
    end
    
    initial begin $dumpfile("lcp.vcd"); $dumpvars(0, tb_local_cmd_processor); end
    initial begin #(CLK * 5000); $display("TIMEOUT!"); $finish; end
endmodule

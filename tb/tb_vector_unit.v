//==============================================================================
// Vector Processing Unit (VPU) Testbench - Robust Version
//==============================================================================
`timescale 1ns / 1ps

module tb_vector_unit;
    parameter CLK = 10;
    parameter LANES = 8;
    parameter DATA_WIDTH = 16;

    reg clk = 0;
    reg rst_n = 0;
    always #(CLK/2) clk = ~clk;

    reg [127:0] cmd = 0;
    reg cmd_valid = 0;
    wire cmd_ready, cmd_done;
    
    wire [19:0] sram_addr;
    wire [LANES*DATA_WIDTH-1:0] sram_wdata;
    reg [LANES*DATA_WIDTH-1:0] sram_rdata = 0;
    wire sram_we, sram_re;
    reg sram_ready = 1;

    vector_unit #(.LANES(LANES), .DATA_WIDTH(DATA_WIDTH)) dut (
        .clk(clk), .rst_n(rst_n),
        .cmd(cmd), .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_done(cmd_done),
        .sram_addr(sram_addr), .sram_wdata(sram_wdata), .sram_rdata(sram_rdata),
        .sram_we(sram_we), .sram_re(sram_re), .sram_ready(sram_ready)
    );

    // Subops
    localparam VOP_ADD = 8'h01, VOP_RELU = 8'h10, VOP_SUM = 8'h20, VOP_ZERO = 8'h34;

    integer errors = 0, i, timeout;
    reg signed [DATA_WIDTH-1:0] val;
    reg cmd_accepted;

    task issue_cmd;
        input [127:0] c;
        begin
            cmd = c;
            cmd_valid = 1;
            cmd_accepted = 0;
            timeout = 0;
            while (!cmd_done && timeout < 50) begin
                @(posedge clk);
                #1;
                if (cmd_ready && cmd_valid && !cmd_accepted) cmd_accepted = 1;
                if (cmd_accepted) cmd_valid = 0;
                timeout = timeout + 1;
            end
            @(posedge clk);  // Extra cycle for result to settle
        end
    endtask

    task print_vreg;
        input [4:0] r;
        integer j;
        begin
            $write("    V%0d = [", r);
            for (j = 0; j < LANES; j = j + 1) begin
                val = $signed(dut.vrf[r][j*DATA_WIDTH +: DATA_WIDTH]);
                $write("%0d", val);
                if (j < LANES-1) $write(", ");
            end
            $display("]");
        end
    endtask

    // Initialize all VRF to zero
    task init_vrf;
        integer r;
        begin
            for (r = 0; r < 32; r = r + 1) begin
                dut.vrf[r] = 0;
            end
        end
    endtask

    initial begin
        $display("");
        $display("╔════════════════════════════════════════════════════════════╗");
        $display("║         Vector Processing Unit Testbench                   ║");
        $display("╚════════════════════════════════════════════════════════════╝");

        // Initialize VRF before reset
        init_vrf();
        
        #(CLK*5); rst_n = 1; #(CLK*3);
        
        // Re-initialize after reset to be safe
        init_vrf();
        #(CLK*2);

        //======================================================================
        // TEST 1: Vector ADD
        //======================================================================
        $display("");
        $display("[TEST 1] Vector ADD: V1 = V0 + V1");
        
        dut.vrf[0] = {16'd8, 16'd7, 16'd6, 16'd5, 16'd4, 16'd3, 16'd2, 16'd1};
        dut.vrf[1] = {16'd80, 16'd70, 16'd60, 16'd50, 16'd40, 16'd30, 16'd20, 16'd10};
        print_vreg(0);
        print_vreg(1);
        
        // VOP_ADD = 0x01 -> vd = 1
        cmd = 0;
        cmd[127:120] = 8'h02;
        cmd[119:112] = VOP_ADD;
        cmd[111:107] = 5'd0;   // vs1
        cmd[106:102] = 5'd1;   // vs2
        issue_cmd(cmd);
        
        print_vreg(1);
        val = $signed(dut.vrf[1][0 +: DATA_WIDTH]);
        if (val == 11) $display("  PASS: V1[0] = %0d (1+10)", val);
        else begin $display("  FAIL: V1[0] = %0d, expected 11", val); errors = errors + 1; end

        #(CLK * 5);

        //======================================================================
        // TEST 2: ReLU
        //======================================================================
        $display("");
        $display("[TEST 2] ReLU: V16 = relu(V5)");
        
        // Initialize fresh
        init_vrf();
        dut.vrf[5] = {16'sh0007, 16'sh0005, 16'sh0003, 16'sh0001, 16'sh0000, 16'shFFFF, 16'shFFFD, 16'shFFFB};
        // That's [7, 5, 3, 1, 0, -1, -3, -5] in signed
        print_vreg(5);
        
        // VOP_RELU = 0x10 -> vd = 16
        cmd = 0;
        cmd[127:120] = 8'h02;
        cmd[119:112] = VOP_RELU;
        cmd[111:107] = 5'd5;
        issue_cmd(cmd);
        
        print_vreg(16);
        val = $signed(dut.vrf[16][0 +: DATA_WIDTH]);
        if (val == 0) $display("  PASS: relu(-5) = 0");
        else begin $display("  FAIL: relu(-5) = %0d, expected 0", val); errors = errors + 1; end
        
        val = $signed(dut.vrf[16][7*DATA_WIDTH +: DATA_WIDTH]);
        if (val == 7) $display("  PASS: relu(7) = 7");
        else begin $display("  FAIL: relu(7) = %0d, expected 7", val); errors = errors + 1; end

        #(CLK * 5);

        //======================================================================
        // TEST 3: Reduction SUM
        //======================================================================
        $display("");
        $display("[TEST 3] Reduction SUM: V0 = sum(V3)");
        
        init_vrf();
        dut.vrf[3] = {16'd8, 16'd7, 16'd6, 16'd5, 16'd4, 16'd3, 16'd2, 16'd1};
        print_vreg(3);
        
        // VOP_SUM = 0x20 -> vd = 0
        cmd = 0;
        cmd[127:120] = 8'h02;
        cmd[119:112] = VOP_SUM;
        cmd[111:107] = 5'd3;
        issue_cmd(cmd);
        
        print_vreg(0);
        val = $signed(dut.vrf[0][0 +: DATA_WIDTH]);
        if (val == 36) $display("  PASS: sum([1..8]) = %0d", val);
        else begin $display("  FAIL: sum = %0d, expected 36", val); errors = errors + 1; end

        #(CLK * 5);

        //======================================================================
        // TEST 4: ZERO
        //======================================================================
        $display("");
        $display("[TEST 4] Vector ZERO: V20 = 0");
        
        dut.vrf[20] = {LANES{16'hFFFF}};
        $display("    Before: V20 = %h", dut.vrf[20]);
        
        // VOP_ZERO = 0x34 -> vd = 20
        cmd = 0;
        cmd[127:120] = 8'h02;
        cmd[119:112] = VOP_ZERO;
        cmd[111:107] = 5'd0;
        issue_cmd(cmd);
        
        $display("    After:  V20 = %h", dut.vrf[20]);
        if (dut.vrf[20] == 0) $display("  PASS: V20 zeroed");
        else begin $display("  FAIL: V20 not zero"); errors = errors + 1; end

        //======================================================================
        // Summary
        //======================================================================
        #(CLK * 10);
        $display("");
        $display("════════════════════════════════════════");
        $display("Tests: 4, Errors: %0d", errors);
        if (errors == 0) $display(">>> ALL TESTS PASSED! <<<");
        else $display(">>> SOME TESTS FAILED <<<");
        $display("");
        $finish;
    end

    initial begin $dumpfile("vpu.vcd"); $dumpvars(0, tb_vector_unit); end
    initial begin #(CLK * 5000); $display("TIMEOUT!"); $finish; end
endmodule

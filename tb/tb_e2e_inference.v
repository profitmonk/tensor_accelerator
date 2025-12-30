`timescale 1ns / 1ps
//==============================================================================
// E2E Inference Layer Test: Y = ReLU(X × W + bias)
//
// Tests complete single-row inference:
//   GEMM → VPU ADD bias → VPU RELU → output
//
// Configuration:
//   - Input X row: [1,1,1,1] (4 elements)
//   - Weights W: 4×4 identity matrix
//   - Bias: [-2, -1, 0, 1]
//   - Expected: Z = [1,1,1,1], Z+b = [-1,0,1,2], ReLU = [0,0,1,2]
//==============================================================================

module tb_e2e_inference;

    parameter CLK = 10;
    parameter SRAM_WIDTH = 256;

    reg clk = 0;
    reg rst_n = 0;
    always #(CLK/2) clk = ~clk;

    reg tpc_start = 0;
    reg [19:0] tpc_start_pc = 0;
    wire tpc_busy, tpc_done, tpc_error;
    reg global_sync_in = 0;
    wire sync_request;
    reg sync_grant = 0;
    reg [SRAM_WIDTH-1:0] noc_rx_data = 0;
    reg [19:0] noc_rx_addr = 0;
    reg noc_rx_valid = 0;
    wire noc_rx_ready;
    reg noc_rx_is_instr = 0;
    wire [SRAM_WIDTH-1:0] noc_tx_data;
    wire [19:0] noc_tx_addr;
    wire noc_tx_valid;
    reg noc_tx_ready = 1;
    wire [39:0] axi_awaddr, axi_araddr;
    wire [7:0] axi_awlen, axi_arlen;
    wire axi_awvalid, axi_arvalid, axi_wvalid, axi_wlast, axi_rready, axi_bready;
    wire [255:0] axi_wdata;
    reg axi_awready = 1, axi_arready = 1, axi_wready = 1;
    reg axi_bvalid = 0, axi_rvalid = 0, axi_rlast = 0;
    reg [1:0] axi_bresp = 0;
    reg [255:0] axi_rdata = 0;

    tensor_processing_cluster #(
        .ARRAY_SIZE(4), .SRAM_WIDTH(SRAM_WIDTH),
        .SRAM_BANKS(4), .SRAM_DEPTH(256), .VPU_LANES(16)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .tpc_start(tpc_start), .tpc_start_pc(tpc_start_pc),
        .tpc_busy(tpc_busy), .tpc_done(tpc_done), .tpc_error(tpc_error),
        .global_sync_in(global_sync_in), .sync_request(sync_request), .sync_grant(sync_grant),
        .noc_rx_data(noc_rx_data), .noc_rx_addr(noc_rx_addr),
        .noc_rx_valid(noc_rx_valid), .noc_rx_ready(noc_rx_ready), .noc_rx_is_instr(noc_rx_is_instr),
        .noc_tx_data(noc_tx_data), .noc_tx_addr(noc_tx_addr),
        .noc_tx_valid(noc_tx_valid), .noc_tx_ready(noc_tx_ready),
        .axi_awaddr(axi_awaddr), .axi_awlen(axi_awlen), .axi_awvalid(axi_awvalid), .axi_awready(axi_awready),
        .axi_wdata(axi_wdata), .axi_wlast(axi_wlast), .axi_wvalid(axi_wvalid), .axi_wready(axi_wready),
        .axi_bresp(axi_bresp), .axi_bvalid(axi_bvalid), .axi_bready(axi_bready),
        .axi_araddr(axi_araddr), .axi_arlen(axi_arlen), .axi_arvalid(axi_arvalid), .axi_arready(axi_arready),
        .axi_rdata(axi_rdata), .axi_rlast(axi_rlast), .axi_rvalid(axi_rvalid), .axi_rready(axi_rready)
    );

    localparam OP_TENSOR = 8'h01;
    localparam OP_VECTOR = 8'h02;
    localparam OP_SYNC   = 8'h04;
    localparam OP_HALT   = 8'hFF;
    localparam VOP_ADD   = 8'h01;
    localparam VOP_RELU  = 8'h10;
    localparam VOP_LOAD  = 8'h30;
    localparam VOP_STORE = 8'h31;
    localparam SYNC_MXU  = 8'h01;
    localparam SYNC_VPU  = 8'h02;

    integer i, errors;
    reg [255:0] ones_row;
    reg signed [31:0] actual, expected;

    initial begin
        $display("");
        $display("╔══════════════════════════════════════════════════════════════╗");
        $display("║        E2E Inference Layer Test: Y = ReLU(X×W + bias)        ║");
        $display("║        Single Row: 4 inputs → 4 outputs                      ║");
        $display("╚══════════════════════════════════════════════════════════════╝");
        $display("");
        
        errors = 0;
        
        //======================================================================
        // Initialize SRAM
        //======================================================================
        $display("[SETUP] Initializing SRAM...");
        
        // X (input at 0x0000): 4 rows of [1,1,1,1]
        ones_row = {224'd0, 8'd1, 8'd1, 8'd1, 8'd1};
        dut.sram_inst.bank_gen[0].bank_inst.mem[0] = ones_row;
        dut.sram_inst.bank_gen[1].bank_inst.mem[0] = ones_row;
        dut.sram_inst.bank_gen[2].bank_inst.mem[0] = ones_row;
        dut.sram_inst.bank_gen[3].bank_inst.mem[0] = ones_row;
        $display("  X: 4×4 matrix of 1s");
        
        // W (weights at 0x0010): Identity matrix
        dut.sram_inst.bank_gen[0].bank_inst.mem[4] = {224'd0, 8'd0, 8'd0, 8'd0, 8'd1};
        dut.sram_inst.bank_gen[1].bank_inst.mem[4] = {224'd0, 8'd0, 8'd0, 8'd1, 8'd0};
        dut.sram_inst.bank_gen[2].bank_inst.mem[4] = {224'd0, 8'd0, 8'd1, 8'd0, 8'd0};
        dut.sram_inst.bank_gen[3].bank_inst.mem[4] = {224'd0, 8'd1, 8'd0, 8'd0, 8'd0};
        $display("  W: 4×4 identity matrix");
        
        // Bias at 0x0030 (bank0, word12): [-2, -1, 0, 1] packed in ONE word
        // Low bits first: bias[0]=-2 at [31:0], bias[1]=-1 at [63:32], etc.
        dut.sram_inst.bank_gen[0].bank_inst.mem[12] = {128'd0, 32'sd1, 32'sd0, -32'sd1, -32'sd2};
        $display("  Bias: [-2, -1, 0, 1]");
        
        $display("  Expected: Z=[1,1,1,1] → Z+b=[-1,0,1,2] → ReLU=[0,0,1,2]");
        
        //======================================================================
        // Load Program
        //======================================================================
        $display("");
        $display("[SETUP] Loading program...");
        
        i = 0;
        
        // GEMM: Z = X × W (dst=0x20, src1=0x00, src2=0x10)
        dut.instr_mem[i] = {OP_TENSOR, 8'h01, 16'h0020, 16'h0010, 16'h0000, 16'd4, 16'd4, 16'd4, 16'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_MXU, 112'd0}; i = i + 1;
        
        // VPU LOAD v0 <- Z (row 0 at 0x20)
        dut.instr_mem[i] = {OP_VECTOR, VOP_LOAD, 5'd0, 5'd0, 5'd0, 1'b0, 20'h00020, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        
        // VPU LOAD v1 <- bias (at 0x30)
        dut.instr_mem[i] = {OP_VECTOR, VOP_LOAD, 5'd1, 5'd0, 5'd0, 1'b0, 20'h00030, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        
        // VPU ADD v0 = v0 + v1 (Z + bias)
        dut.instr_mem[i] = {OP_VECTOR, VOP_ADD, 5'd0, 5'd0, 5'd1, 1'b0, 20'h00000, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        
        // VPU RELU v0 = relu(v0)
        dut.instr_mem[i] = {OP_VECTOR, VOP_RELU, 5'd0, 5'd0, 5'd0, 1'b0, 20'h00000, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        
        // VPU STORE v0 -> Y (at 0x40)
        dut.instr_mem[i] = {OP_VECTOR, VOP_STORE, 5'd0, 5'd0, 5'd0, 1'b0, 20'h00040, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        
        dut.instr_mem[i] = {OP_HALT, 120'd0};
        $display("  Program: %0d instructions", i+1);
        
        //======================================================================
        // Run Test
        //======================================================================
        $display("");
        $display("[EXEC] Running inference...");
        
        rst_n = 0; #(CLK*5);
        rst_n = 1; #(CLK*5);
        
        @(negedge clk);
        tpc_start = 1;
        @(posedge clk); @(posedge clk);
        @(negedge clk);
        tpc_start = 0;
        
        for (i = 0; i < 200; i = i + 1) begin
            @(posedge clk);
            if (tpc_done) begin
                $display("  Completed in %0d cycles", i);
                i = 1000;
            end
        end
        
        if (i < 1000) begin
            $display("  TIMEOUT!");
            errors = errors + 1;
        end
        
        #(CLK*5);
        
        //======================================================================
        // Verify Results
        //======================================================================
        $display("");
        $display("[VERIFY] Checking output...");
        
        // Y output at 0x40 (bank0, word16)
        $display("  Output Y (row 0): %h", dut.sram_inst.bank_gen[0].bank_inst.mem[16][127:0]);
        
        // Check each element (32-bit signed)
        // Expected: [0, 0, 1, 2]
        actual = $signed(dut.sram_inst.bank_gen[0].bank_inst.mem[16][31:0]);
        expected = 0;
        if (actual == expected) $display("    PASS: Y[0] = %0d", actual);
        else begin $display("    FAIL: Y[0] = %0d, expected %0d", actual, expected); errors = errors + 1; end
        
        actual = $signed(dut.sram_inst.bank_gen[0].bank_inst.mem[16][63:32]);
        expected = 0;
        if (actual == expected) $display("    PASS: Y[1] = %0d", actual);
        else begin $display("    FAIL: Y[1] = %0d, expected %0d", actual, expected); errors = errors + 1; end
        
        actual = $signed(dut.sram_inst.bank_gen[0].bank_inst.mem[16][95:64]);
        expected = 1;
        if (actual == expected) $display("    PASS: Y[2] = %0d", actual);
        else begin $display("    FAIL: Y[2] = %0d, expected %0d", actual, expected); errors = errors + 1; end
        
        actual = $signed(dut.sram_inst.bank_gen[0].bank_inst.mem[16][127:96]);
        expected = 2;
        if (actual == expected) $display("    PASS: Y[3] = %0d", actual);
        else begin $display("    FAIL: Y[3] = %0d, expected %0d", actual, expected); errors = errors + 1; end
        
        //======================================================================
        // Summary
        //======================================================================
        $display("");
        $display("╔══════════════════════════════════════════════════════════════╗");
        $display("║                    TEST SUMMARY                              ║");
        $display("╠══════════════════════════════════════════════════════════════╣");
        if (errors == 0) begin
            $display("║   PASSED: E2E inference layer (GEMM→ADD→ReLU)              ║");
            $display("╚══════════════════════════════════════════════════════════════╝");
            $display(">>> E2E INFERENCE TEST PASSED! <<<");
        end else begin
            $display("║   FAILED: %0d errors                                         ║", errors);
            $display("╚══════════════════════════════════════════════════════════════╝");
            $display(">>> E2E INFERENCE TEST FAILED <<<");
        end
        
        #(CLK * 10);
        $finish;
    end

    initial begin
        #(CLK * 10000);
        $display("GLOBAL TIMEOUT!");
        $finish;
    end

endmodule

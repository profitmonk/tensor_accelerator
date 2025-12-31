`timescale 1ns / 1ps
module tb_e2e_inference_debug;

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

    integer i;
    reg [255:0] ones_row;

    initial begin
        $display("E2E Inference Debug Test");

        // Initialize SRAM directly
        // X (input at 0x0000): 4 rows of [1,1,1,1]
        ones_row = {224'd0, 8'd1, 8'd1, 8'd1, 8'd1};
        dut.sram_inst.bank_gen[0].bank_inst.mem[0] = ones_row;
        dut.sram_inst.bank_gen[1].bank_inst.mem[0] = ones_row;
        dut.sram_inst.bank_gen[2].bank_inst.mem[0] = ones_row;
        dut.sram_inst.bank_gen[3].bank_inst.mem[0] = ones_row;
        
        // W (weights at 0x0010): Identity matrix
        dut.sram_inst.bank_gen[0].bank_inst.mem[4] = {224'd0, 8'd0, 8'd0, 8'd0, 8'd1};  // [1,0,0,0]
        dut.sram_inst.bank_gen[1].bank_inst.mem[4] = {224'd0, 8'd0, 8'd0, 8'd1, 8'd0};  // [0,1,0,0]
        dut.sram_inst.bank_gen[2].bank_inst.mem[4] = {224'd0, 8'd0, 8'd1, 8'd0, 8'd0};  // [0,0,1,0]
        dut.sram_inst.bank_gen[3].bank_inst.mem[4] = {224'd0, 8'd1, 8'd0, 8'd0, 8'd0};  // [0,0,0,1]
        
        // Bias (at 0x0030): [-2, -1, 0, 1] as 32-bit signed
        // Store bias directly as 32-bit values
        dut.sram_inst.bank_gen[0].bank_inst.mem[12] = {224'd0, -32'sd2};
        dut.sram_inst.bank_gen[1].bank_inst.mem[12] = {224'd0, -32'sd1};
        dut.sram_inst.bank_gen[2].bank_inst.mem[12] = {224'd0, 32'sd0};
        dut.sram_inst.bank_gen[3].bank_inst.mem[12] = {224'd0, 32'sd1};

        // Program:
        // 1. GEMM: Z = X × W (dst=0x20, src1=0x00, src2=0x10)
        // 2. VPU LOAD v0 from Z (0x20)
        // 3. VPU LOAD v1 from bias (0x30)
        // 4. VPU ADD v0 = v0 + v1
        // 5. VPU RELU v0 = relu(v0)
        // 6. VPU STORE v0 to Y (0x40)
        
        i = 0;
        dut.instr_mem[i] = {OP_TENSOR, 8'h01, 16'h0020, 16'h0010, 16'h0000, 16'd4, 16'd4, 16'd4, 16'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_MXU, 112'd0}; i = i + 1;
        
        // VPU LOAD v0 <- Z[0x20]
        dut.instr_mem[i] = {OP_VECTOR, VOP_LOAD, 5'd0, 5'd0, 5'd0, 1'b0, 20'h00020, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        
        // VPU LOAD v1 <- bias[0x30]
        dut.instr_mem[i] = {OP_VECTOR, VOP_LOAD, 5'd1, 5'd0, 5'd0, 1'b0, 20'h00030, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        
        // VPU ADD v0 = v0 + v1
        dut.instr_mem[i] = {OP_VECTOR, VOP_ADD, 5'd0, 5'd0, 5'd1, 1'b0, 20'h00000, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        
        // VPU RELU v0 = relu(v0)
        dut.instr_mem[i] = {OP_VECTOR, VOP_RELU, 5'd0, 5'd0, 5'd0, 1'b0, 20'h00000, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        
        // VPU STORE v0 -> Y[0x40]
        dut.instr_mem[i] = {OP_VECTOR, VOP_STORE, 5'd0, 5'd0, 5'd0, 1'b0, 20'h00040, 12'd0, 16'd4, 16'd0, 32'd0}; i = i + 1;
        dut.instr_mem[i] = {OP_SYNC, SYNC_VPU, 112'd0}; i = i + 1;
        
        dut.instr_mem[i] = {OP_HALT, 120'd0};

        rst_n = 0; #(CLK*5); rst_n = 1; #(CLK*5);

        @(negedge clk);
        tpc_start = 1;
        @(posedge clk); @(posedge clk);
        @(negedge clk);
        tpc_start = 0;

        // Wait for completion
        for (i = 0; i < 200; i = i + 1) begin
            @(posedge clk);
            if (tpc_done) begin
                $display("Done at cycle %0d", i);
                i = 1000;
            end
        end

        #(CLK*5);
        
        // Check intermediate results
        $display("");
        $display("=== SRAM Contents (Row 0 only - VPU reads one address) ===");
        $display("Z row 0 (at 0x20, bank0 word8): %h", dut.sram_inst.bank_gen[0].bank_inst.mem[8][31:0]);
        
        $display("Bias (at 0x30, bank0 word12): %d", $signed(dut.sram_inst.bank_gen[0].bank_inst.mem[12][31:0]));
        
        $display("Y row 0 (at 0x40, bank0 word16): %d", $signed(dut.sram_inst.bank_gen[0].bank_inst.mem[16][31:0]));
        
        $display("");
        $display("=== VPU Registers ===");
        $display("v0[0]: %d (GEMM output row 0, col 0)", $signed(dut.vpu_inst.vrf[0][15:0]));
        $display("v1[0]: %d (bias[0])", $signed(dut.vpu_inst.vrf[1][15:0]));
        
        // Check expected values
        // Z[0,0] = sum(X[0,k] * W[k,0]) for k=0..3 = 1*1 + 1*0 + 1*0 + 1*0 = 1
        // Z[0,0] + bias[0] = 1 + (-2) = -1
        // ReLU(-1) = 0
        $display("");
        $display("Expected: Z[0,0]=1, Z+bias=-1, ReLU=0");
        
        if ($signed(dut.sram_inst.bank_gen[0].bank_inst.mem[16][31:0]) == 0) begin
            $display(">>> SINGLE-ROW E2E INFERENCE CORRECT! <<<");
        end else begin
            $display(">>> TEST FAILED <<<");
        end
        
        $finish;
    end

    initial begin #(CLK*10000); $display("TIMEOUT"); $finish; end
endmodule

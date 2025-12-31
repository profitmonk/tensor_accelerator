`timescale 1ns / 1ps
//==============================================================================
// VPU + SRAM Integration Test
// Tests VPU LOAD/ADD/STORE through the TPC's SRAM subsystem
//==============================================================================

module tb_vpu_sram;

    parameter CLK = 10;
    
    reg clk = 0;
    reg rst_n = 0;
    always #(CLK/2) clk = ~clk;

    // TPC interface
    reg tpc_enable = 0;
    reg tpc_start = 0;
    wire tpc_done;
    wire tpc_busy;
    
    // Stub AXI (not used in this test)
    wire [39:0] axi_awaddr, axi_araddr;
    wire [7:0] axi_awlen, axi_arlen;
    wire [2:0] axi_awsize, axi_arsize;
    wire [1:0] axi_awburst, axi_arburst;
    wire axi_awvalid, axi_arvalid;
    reg axi_awready = 1, axi_arready = 1;
    wire [255:0] axi_wdata;
    wire [31:0] axi_wstrb;
    wire axi_wlast, axi_wvalid;
    reg axi_wready = 1;
    reg [1:0] axi_bresp = 0;
    reg axi_bvalid = 0;
    wire axi_bready;
    reg [255:0] axi_rdata = 0;
    reg axi_rlast = 0, axi_rvalid = 0;
    wire axi_rready;
    
    // NoC stubs
    wire [255:0] noc_tx_data;
    wire [15:0] noc_tx_dest;
    wire noc_tx_valid;
    reg noc_tx_ready = 1;
    reg [255:0] noc_rx_data = 0;
    reg [15:0] noc_rx_src = 0;
    reg noc_rx_valid = 0;
    wire noc_rx_ready;

    tensor_processing_cluster #(
        .ARRAY_SIZE(4),
        .DATA_WIDTH(8),
        .SRAM_BANKS(4),
        .SRAM_DEPTH(64),
        .VPU_LANES(8)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .tpc_enable(tpc_enable),
        .tpc_start(tpc_start),
        .tpc_done(tpc_done),
        .tpc_busy(tpc_busy),
        .axi_awaddr(axi_awaddr), .axi_awlen(axi_awlen), .axi_awsize(axi_awsize),
        .axi_awburst(axi_awburst), .axi_awvalid(axi_awvalid), .axi_awready(axi_awready),
        .axi_wdata(axi_wdata), .axi_wstrb(axi_wstrb), .axi_wlast(axi_wlast),
        .axi_wvalid(axi_wvalid), .axi_wready(axi_wready),
        .axi_bresp(axi_bresp), .axi_bvalid(axi_bvalid), .axi_bready(axi_bready),
        .axi_araddr(axi_araddr), .axi_arlen(axi_arlen), .axi_arsize(axi_arsize),
        .axi_arburst(axi_arburst), .axi_arvalid(axi_arvalid), .axi_arready(axi_arready),
        .axi_rdata(axi_rdata), .axi_rlast(axi_rlast), .axi_rvalid(axi_rvalid),
        .axi_rready(axi_rready),
        .noc_tx_data(noc_tx_data), .noc_tx_dest(noc_tx_dest),
        .noc_tx_valid(noc_tx_valid), .noc_tx_ready(noc_tx_ready),
        .noc_rx_data(noc_rx_data), .noc_rx_src(noc_rx_src),
        .noc_rx_valid(noc_rx_valid), .noc_rx_ready(noc_rx_ready)
    );

    // Opcodes
    localparam OP_VECTOR = 8'h02;
    localparam OP_SYNC   = 8'h04;
    localparam OP_HALT   = 8'hFF;
    
    localparam VOP_ADD   = 8'h01;
    localparam VOP_LOAD  = 8'h30;
    localparam VOP_STORE = 8'h31;
    localparam SYNC_VPU  = 8'h02;

    // Build VPU command
    // Format: [127:120]=op, [119:112]=subop, [111:107]=vd, [106:102]=vs1, 
    //         [101:97]=vs2, [95:76]=mem_addr, [63:48]=count
    function [127:0] vpu_cmd;
        input [7:0] subop;
        input [4:0] vd, vs1, vs2;
        input [19:0] mem_addr;
        input [15:0] count;
        begin
            vpu_cmd = {OP_VECTOR, subop, vd, vs1, vs2, 1'b0, mem_addr, 12'd0, count, 16'd0, 32'd0};
        end
    endfunction

    integer i, errors;
    reg [255:0] result;

    initial begin
        $display("");
        $display("╔════════════════════════════════════════════════════════════╗");
        $display("║         VPU + SRAM Integration Test                        ║");
        $display("╚════════════════════════════════════════════════════════════╝");
        
        errors = 0;
        
        // Reset
        rst_n = 0; #(CLK*5);
        rst_n = 1; #(CLK*5);

        //======================================================================
        // Initialize SRAM with test data
        // Using addresses 0-3 for data A, 4-7 for data B, 8-11 for output
        //======================================================================
        $display("");
        $display("[SETUP] Loading test data into SRAM...");
        
        // Data A at addresses 0-3: values 1,2,3,4,5,6,7,8 (8-bit elements)
        // With 4 banks and XOR mapping, addr 0->bank 0, addr 1->bank 1, etc.
        dut.sram_inst.bank_gen[0].bank_inst.mem[0] = {248'd0, 8'd1};  // addr 0
        dut.sram_inst.bank_gen[1].bank_inst.mem[0] = {248'd0, 8'd2};  // addr 1
        dut.sram_inst.bank_gen[2].bank_inst.mem[0] = {248'd0, 8'd3};  // addr 2
        dut.sram_inst.bank_gen[3].bank_inst.mem[0] = {248'd0, 8'd4};  // addr 3
        
        // Data B at addresses 4-7: values 10,20,30,40
        dut.sram_inst.bank_gen[0].bank_inst.mem[1] = {248'd0, 8'd10}; // addr 4 (bank=4^1=5%4=1? Let me check)
        dut.sram_inst.bank_gen[1].bank_inst.mem[1] = {248'd0, 8'd20}; // addr 5
        dut.sram_inst.bank_gen[2].bank_inst.mem[1] = {248'd0, 8'd30}; // addr 6
        dut.sram_inst.bank_gen[3].bank_inst.mem[1] = {248'd0, 8'd40}; // addr 7
        
        $display("  Data A @ 0x0: [1, 2, 3, 4]");
        $display("  Data B @ 0x4: [10, 20, 30, 40]");
        $display("  Expected C = A + B: [11, 22, 33, 44]");

        //======================================================================
        // Load program: VPU LOAD v0 from addr 0, LOAD v1 from addr 4, 
        //               ADD v0 = v0 + v1, STORE v0 to addr 8
        //======================================================================
        $display("");
        $display("[PROGRAM] VPU: v0 <- SRAM[0], v1 <- SRAM[4], v0 += v1, SRAM[8] <- v0");
        
        dut.instr_mem[0] = vpu_cmd(VOP_LOAD, 5'd0, 5'd0, 5'd0, 20'd0, 16'd4);   // v0 <- mem[0:3]
        dut.instr_mem[1] = {OP_SYNC, SYNC_VPU, 112'd0};
        dut.instr_mem[2] = vpu_cmd(VOP_LOAD, 5'd1, 5'd0, 5'd0, 20'd4, 16'd4);   // v1 <- mem[4:7]
        dut.instr_mem[3] = {OP_SYNC, SYNC_VPU, 112'd0};
        dut.instr_mem[4] = vpu_cmd(VOP_ADD, 5'd0, 5'd0, 5'd1, 20'd0, 16'd4);    // v0 = v0 + v1
        dut.instr_mem[5] = {OP_SYNC, SYNC_VPU, 112'd0};
        dut.instr_mem[6] = vpu_cmd(VOP_STORE, 5'd0, 5'd0, 5'd0, 20'd8, 16'd4);  // mem[8:11] <- v0
        dut.instr_mem[7] = {OP_SYNC, SYNC_VPU, 112'd0};
        dut.instr_mem[8] = {OP_HALT, 120'd0};

        //======================================================================
        // Execute
        //======================================================================
        $display("");
        $display("[EXEC] Starting TPC...");
        
        tpc_enable = 1;
        @(negedge clk);
        tpc_start = 1;
        @(posedge clk); @(posedge clk);
        @(negedge clk);
        tpc_start = 0;

        // Wait for completion with debug
        for (i = 0; i < 100; i = i + 1) begin
            @(posedge clk);
            if (i < 20 || i % 10 == 0) begin
                $display("  [%0d] lcp_state=%0d vpu_state=%0d done=%b",
                    i, dut.lcp_inst.state, dut.vpu_inst.state, tpc_done);
            end
            if (tpc_done) begin
                $display("  Completed at cycle %0d", i);
                i = 1000;
            end
        end

        //======================================================================
        // Verify results
        //======================================================================
        $display("");
        $display("[VERIFY] Checking SRAM output...");
        
        // Read results from addresses 8-11
        result = dut.sram_inst.bank_gen[0].bank_inst.mem[2];  // addr 8
        $display("  SRAM[8] = %0d (expected 11)", result[7:0]);
        if (result[7:0] != 8'd11) errors = errors + 1;
        
        result = dut.sram_inst.bank_gen[1].bank_inst.mem[2];  // addr 9
        $display("  SRAM[9] = %0d (expected 22)", result[7:0]);
        if (result[7:0] != 8'd22) errors = errors + 1;
        
        result = dut.sram_inst.bank_gen[2].bank_inst.mem[2];  // addr 10
        $display("  SRAM[10] = %0d (expected 33)", result[7:0]);
        if (result[7:0] != 8'd33) errors = errors + 1;
        
        result = dut.sram_inst.bank_gen[3].bank_inst.mem[2];  // addr 11
        $display("  SRAM[11] = %0d (expected 44)", result[7:0]);
        if (result[7:0] != 8'd44) errors = errors + 1;

        //======================================================================
        // Summary
        //======================================================================
        $display("");
        if (errors == 0) begin
            $display(">>> VPU SRAM TEST PASSED! <<<");
        end else begin
            $display(">>> VPU SRAM TEST FAILED: %0d errors <<<", errors);
        end
        
        #(CLK*10);
        $finish;
    end

    initial begin
        #(CLK * 5000);
        $display("TIMEOUT!");
        $finish;
    end

endmodule

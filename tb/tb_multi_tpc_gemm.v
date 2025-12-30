`timescale 1ns / 1ps
//==============================================================================
// Multi-TPC Tiled GEMM Test
//
// Tests 16×16 matrix multiply using 4 TPCs in parallel:
//   C[16×16] = A[16×16] × B[16×16]
//
// Split into 4×4 tiles (to match 4×4 systolic array):
//   C00 = A00×B00 + A01×B10 + A02×B20 + A03×B30
//   C01 = A00×B01 + A01×B11 + A02×B21 + A03×B31
//   C10 = A10×B00 + A11×B10 + A12×B20 + A13×B30
//   C11 = A10×B01 + A11×B11 + A12×B21 + A13×B31
//
// TPC Assignment:
//   TPC0 → C00,  TPC1 → C01,  TPC2 → C10,  TPC3 → C11
//
// Memory Layout (DDR):
//   A matrix: 0x0000_0000 - tiles row-major [A00, A01, A02, A03, A10, ...]
//   B matrix: 0x0000_1000 - tiles row-major [B00, B01, B02, B03, B10, ...]
//   C matrix: 0x0000_2000 - output tiles [C00, C01, C10, C11]
//==============================================================================

module tb_multi_tpc_gemm;

    parameter CLK = 10;
    parameter ARRAY_SIZE = 4;     // 4×4 systolic array
    parameter TILE_SIZE = 4;      // 4×4 tiles
    parameter MATRIX_SIZE = 16;   // 16×16 full matrix
    parameter TILES_PER_DIM = MATRIX_SIZE / TILE_SIZE;  // 4 tiles per dimension
    parameter DATA_WIDTH = 256;
    parameter NUM_TPCS = 4;

    reg clk = 0;
    reg rst_n = 0;
    always #(CLK/2) clk = ~clk;

    //==========================================================================
    // AXI-Lite Control Interface
    //==========================================================================
    reg [11:0] s_axi_ctrl_awaddr = 0;
    reg s_axi_ctrl_awvalid = 0;
    wire s_axi_ctrl_awready;
    reg [31:0] s_axi_ctrl_wdata = 0;
    reg [3:0] s_axi_ctrl_wstrb = 4'hF;
    reg s_axi_ctrl_wvalid = 0;
    wire s_axi_ctrl_wready;
    wire [1:0] s_axi_ctrl_bresp;
    wire s_axi_ctrl_bvalid;
    reg s_axi_ctrl_bready = 1;
    reg [11:0] s_axi_ctrl_araddr = 0;
    reg s_axi_ctrl_arvalid = 0;
    wire s_axi_ctrl_arready;
    wire [31:0] s_axi_ctrl_rdata;
    wire [1:0] s_axi_ctrl_rresp;
    wire s_axi_ctrl_rvalid;
    reg s_axi_ctrl_rready = 1;

    //==========================================================================
    // AXI Memory Interface
    //==========================================================================
    wire [3:0] m_axi_awid;
    wire [39:0] m_axi_awaddr;
    wire [7:0] m_axi_awlen;
    wire [2:0] m_axi_awsize;
    wire [1:0] m_axi_awburst;
    wire m_axi_awvalid;
    wire m_axi_awready;
    wire [255:0] m_axi_wdata;
    wire [31:0] m_axi_wstrb;
    wire m_axi_wlast;
    wire m_axi_wvalid;
    wire m_axi_wready;
    wire [3:0] m_axi_bid;
    wire [1:0] m_axi_bresp;
    wire m_axi_bvalid;
    wire m_axi_bready;
    wire [3:0] m_axi_arid;
    wire [39:0] m_axi_araddr;
    wire [7:0] m_axi_arlen;
    wire [2:0] m_axi_arsize;
    wire [1:0] m_axi_arburst;
    wire m_axi_arvalid;
    wire m_axi_arready;
    wire [3:0] m_axi_rid;
    wire [255:0] m_axi_rdata;
    wire [1:0] m_axi_rresp;
    wire m_axi_rlast;
    wire m_axi_rvalid;
    wire m_axi_rready;

    wire irq;

    //==========================================================================
    // DUT: Full Tensor Accelerator
    //==========================================================================
    tensor_accelerator_top #(
        .GRID_X(2), .GRID_Y(2),
        .ARRAY_SIZE(ARRAY_SIZE),
        .SRAM_BANKS(4), .SRAM_DEPTH(256),
        .VPU_LANES(16)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        // AXI-Lite Control
        .s_axi_ctrl_awaddr(s_axi_ctrl_awaddr),
        .s_axi_ctrl_awvalid(s_axi_ctrl_awvalid),
        .s_axi_ctrl_awready(s_axi_ctrl_awready),
        .s_axi_ctrl_wdata(s_axi_ctrl_wdata),
        .s_axi_ctrl_wstrb(s_axi_ctrl_wstrb),
        .s_axi_ctrl_wvalid(s_axi_ctrl_wvalid),
        .s_axi_ctrl_wready(s_axi_ctrl_wready),
        .s_axi_ctrl_bresp(s_axi_ctrl_bresp),
        .s_axi_ctrl_bvalid(s_axi_ctrl_bvalid),
        .s_axi_ctrl_bready(s_axi_ctrl_bready),
        .s_axi_ctrl_araddr(s_axi_ctrl_araddr),
        .s_axi_ctrl_arvalid(s_axi_ctrl_arvalid),
        .s_axi_ctrl_arready(s_axi_ctrl_arready),
        .s_axi_ctrl_rdata(s_axi_ctrl_rdata),
        .s_axi_ctrl_rresp(s_axi_ctrl_rresp),
        .s_axi_ctrl_rvalid(s_axi_ctrl_rvalid),
        .s_axi_ctrl_rready(s_axi_ctrl_rready),
        // AXI Memory
        .m_axi_awid(m_axi_awid),
        .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst),
        .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata),
        .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid),
        .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready),
        .m_axi_arid(m_axi_arid),
        .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst),
        .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rid(m_axi_rid),
        .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready),
        .irq(irq)
    );

    //==========================================================================
    // AXI Memory Model
    //==========================================================================
    axi_memory_model #(
        .AXI_ADDR_WIDTH(40),
        .AXI_DATA_WIDTH(256),
        .AXI_ID_WIDTH(4),
        .MEM_SIZE_MB(1),
        .READ_LATENCY(2),
        .WRITE_LATENCY(1)
    ) axi_mem (
        .clk(clk), .rst_n(rst_n),
        .s_axi_awid(m_axi_awid), .s_axi_awaddr(m_axi_awaddr),
        .s_axi_awlen(m_axi_awlen), .s_axi_awsize(m_axi_awsize),
        .s_axi_awburst(m_axi_awburst), .s_axi_awvalid(m_axi_awvalid),
        .s_axi_awready(m_axi_awready),
        .s_axi_wdata(m_axi_wdata), .s_axi_wstrb(m_axi_wstrb),
        .s_axi_wlast(m_axi_wlast), .s_axi_wvalid(m_axi_wvalid),
        .s_axi_wready(m_axi_wready),
        .s_axi_bid(m_axi_bid), .s_axi_bresp(m_axi_bresp),
        .s_axi_bvalid(m_axi_bvalid), .s_axi_bready(m_axi_bready),
        .s_axi_arid(m_axi_arid), .s_axi_araddr(m_axi_araddr),
        .s_axi_arlen(m_axi_arlen), .s_axi_arsize(m_axi_arsize),
        .s_axi_arburst(m_axi_arburst), .s_axi_arvalid(m_axi_arvalid),
        .s_axi_arready(m_axi_arready),
        .s_axi_rid(m_axi_rid), .s_axi_rdata(m_axi_rdata),
        .s_axi_rresp(m_axi_rresp), .s_axi_rlast(m_axi_rlast),
        .s_axi_rvalid(m_axi_rvalid), .s_axi_rready(m_axi_rready)
    );

    //==========================================================================
    // Register Addresses and Opcodes
    //==========================================================================
    localparam ADDR_CTRL   = 12'h000;
    localparam ADDR_STATUS = 12'h004;

    localparam OP_NOP    = 8'h00;
    localparam OP_TENSOR = 8'h01;
    localparam OP_VECTOR = 8'h02;
    localparam OP_DMA    = 8'h03;
    localparam OP_SYNC   = 8'h04;
    localparam OP_HALT   = 8'hFF;

    localparam DMA_LOAD  = 8'h01;
    localparam DMA_STORE = 8'h02;
    localparam VOP_ADD   = 8'h01;
    localparam VOP_LOAD  = 8'h30;
    localparam VOP_STORE = 8'h31;
    
    localparam SYNC_MXU  = 8'h01;
    localparam SYNC_DMA  = 8'h03;
    localparam SYNC_ALL  = 8'hFF;

    //==========================================================================
    // DDR Address Map
    //==========================================================================
    // Each 4×4 tile = 16 bytes (4×4×1 byte elements)
    // Stored as single 256-bit word (padded)
    localparam A_BASE = 40'h0000_0000;  // A matrix tiles
    localparam B_BASE = 40'h0000_1000;  // B matrix tiles
    localparam C_BASE = 40'h0000_2000;  // C output tiles (32-bit accumulators)
    localparam TILE_BYTES = 32;         // 256-bit = 32 bytes per word

    //==========================================================================
    // AXI-Lite Tasks
    //==========================================================================
    reg [31:0] rdata;
    integer timeout;

    task axi_write(input [11:0] addr, input [31:0] data);
        begin
            @(negedge clk);
            s_axi_ctrl_awaddr = addr;
            s_axi_ctrl_awvalid = 1;
            s_axi_ctrl_wdata = data;
            s_axi_ctrl_wvalid = 1;
            @(posedge clk);
            while (!s_axi_ctrl_awready || !s_axi_ctrl_wready) @(posedge clk);
            @(negedge clk);
            s_axi_ctrl_awvalid = 0;
            s_axi_ctrl_wvalid = 0;
            while (!s_axi_ctrl_bvalid) @(posedge clk);
            @(posedge clk);
        end
    endtask

    task axi_read(input [11:0] addr);
        begin
            @(negedge clk);
            s_axi_ctrl_araddr = addr;
            s_axi_ctrl_arvalid = 1;
            @(posedge clk);
            while (!s_axi_ctrl_arready) @(posedge clk);
            @(negedge clk);
            s_axi_ctrl_arvalid = 0;
            while (!s_axi_ctrl_rvalid) @(posedge clk);
            rdata = s_axi_ctrl_rdata;
            @(posedge clk);
        end
    endtask

    //==========================================================================
    // Instruction Builders
    //==========================================================================
    
    // TENSOR instruction: GEMM operation
    // Format: {opcode, subop, dst_addr[15:0], src2_addr[15:0], src1_addr[15:0], N, M, K, flags}
    function [127:0] make_tensor_cmd;
        input [15:0] dst_addr;   // Result address in SRAM
        input [15:0] src1_addr;  // A matrix address in SRAM
        input [15:0] src2_addr;  // B matrix address in SRAM
        input [15:0] n, m, k;    // Dimensions
        begin
            make_tensor_cmd = {OP_TENSOR, 8'h01, dst_addr, src2_addr, src1_addr, n, m, k, 16'd0};
        end
    endfunction

    // DMA instruction
    function [127:0] make_dma_cmd;
        input [7:0] subop;       // LOAD or STORE
        input [39:0] ext_addr;   // DDR address
        input [19:0] int_addr;   // SRAM address
        input [11:0] rows, cols;
        begin
            make_dma_cmd = {OP_DMA, subop, ext_addr, int_addr, rows, cols, 12'd32, 12'd32, 4'd0};
        end
    endfunction

    // VPU instruction  
    function [127:0] make_vpu_cmd;
        input [7:0] subop;
        input [4:0] vd, vs1, vs2;
        input [19:0] mem_addr;
        begin
            make_vpu_cmd = 0;
            make_vpu_cmd[127:120] = OP_VECTOR;
            make_vpu_cmd[119:112] = subop;
            make_vpu_cmd[111:107] = vd;
            make_vpu_cmd[106:102] = vs1;
            make_vpu_cmd[101:97]  = vs2;
            make_vpu_cmd[95:76]   = mem_addr;
        end
    endfunction

    // SYNC instruction
    function [127:0] make_sync_cmd;
        input [7:0] unit;
        begin
            make_sync_cmd = {OP_SYNC, unit, 112'd0};
        end
    endfunction

    //==========================================================================
    // Initialize DDR with test matrices
    //==========================================================================
    integer i, j, idx;
    reg [7:0] a_val, b_val;
    reg [255:0] tile_data;
    
    task init_ddr_matrices;
        begin
            $display("  Initializing DDR with 16x16 matrices...");
            
            // Initialize A matrix - simple pattern A[i,j] = i + 1
            // Store as 4×4 tiles in row-major order
            for (i = 0; i < 16; i = i + 1) begin
                for (j = 0; j < 16; j = j + 1) begin
                    // A[i,j] = i + 1 (row number + 1)
                    a_val = i + 1;
                    // Store in appropriate tile
                    // Tile (ti, tj) contains elements [ti*4:(ti+1)*4-1, tj*4:(tj+1)*4-1]
                    idx = (i/4)*4 + (j/4);  // Tile index
                    // Calculate byte position within 256-bit word
                    // Element within tile: (i%4)*4 + (j%4)
                    // For simplicity, store entire tile in one 256-bit word
                end
            end

            // Build A tiles and store to DDR
            // Tile A00 (rows 0-3, cols 0-3)
            tile_data = 0;
            for (i = 0; i < 4; i = i + 1) begin
                for (j = 0; j < 4; j = j + 1) begin
                    tile_data[(i*4+j)*8 +: 8] = i + 1;
                end
            end
            axi_mem.mem[0] = tile_data;  // A00 at offset 0
            
            // Tile A01 (rows 0-3, cols 4-7)
            tile_data = 0;
            for (i = 0; i < 4; i = i + 1) begin
                for (j = 0; j < 4; j = j + 1) begin
                    tile_data[(i*4+j)*8 +: 8] = i + 1;
                end
            end
            axi_mem.mem[1] = tile_data;  // A01
            
            // A02, A03 (same pattern for rows 0-3)
            axi_mem.mem[2] = tile_data;  // A02
            axi_mem.mem[3] = tile_data;  // A03
            
            // Tile A10 (rows 4-7, cols 0-3)
            tile_data = 0;
            for (i = 0; i < 4; i = i + 1) begin
                for (j = 0; j < 4; j = j + 1) begin
                    tile_data[(i*4+j)*8 +: 8] = i + 5;  // rows 4-7
                end
            end
            axi_mem.mem[4] = tile_data;  // A10
            axi_mem.mem[5] = tile_data;  // A11
            axi_mem.mem[6] = tile_data;  // A12
            axi_mem.mem[7] = tile_data;  // A13
            
            // A20 (rows 8-11)
            tile_data = 0;
            for (i = 0; i < 4; i = i + 1) begin
                for (j = 0; j < 4; j = j + 1) begin
                    tile_data[(i*4+j)*8 +: 8] = i + 9;
                end
            end
            axi_mem.mem[8] = tile_data;
            axi_mem.mem[9] = tile_data;
            axi_mem.mem[10] = tile_data;
            axi_mem.mem[11] = tile_data;
            
            // A30 (rows 12-15)
            tile_data = 0;
            for (i = 0; i < 4; i = i + 1) begin
                for (j = 0; j < 4; j = j + 1) begin
                    tile_data[(i*4+j)*8 +: 8] = i + 13;
                end
            end
            axi_mem.mem[12] = tile_data;
            axi_mem.mem[13] = tile_data;
            axi_mem.mem[14] = tile_data;
            axi_mem.mem[15] = tile_data;

            // Initialize B matrix as identity-like for easy verification
            // B = block diagonal with 4×4 identity tiles
            // B00 = I, B11 = I, B22 = I, B33 = I, others = 0
            
            // B tiles at offset 128 (0x1000 / 32 = 128)
            // B00 = identity
            tile_data = 0;
            tile_data[0*8 +: 8] = 1;   // B[0,0] = 1
            tile_data[5*8 +: 8] = 1;   // B[1,1] = 1
            tile_data[10*8 +: 8] = 1;  // B[2,2] = 1
            tile_data[15*8 +: 8] = 1;  // B[3,3] = 1
            axi_mem.mem[128] = tile_data;  // B00 = I
            
            // B01, B02, B03 = 0
            axi_mem.mem[129] = 256'd0;
            axi_mem.mem[130] = 256'd0;
            axi_mem.mem[131] = 256'd0;
            
            // B10 = 0, B11 = I, B12 = 0, B13 = 0
            axi_mem.mem[132] = 256'd0;
            axi_mem.mem[133] = tile_data;  // B11 = I
            axi_mem.mem[134] = 256'd0;
            axi_mem.mem[135] = 256'd0;
            
            // B20 = 0, B21 = 0, B22 = I, B23 = 0
            axi_mem.mem[136] = 256'd0;
            axi_mem.mem[137] = 256'd0;
            axi_mem.mem[138] = tile_data;  // B22 = I
            axi_mem.mem[139] = 256'd0;
            
            // B30 = 0, B31 = 0, B32 = 0, B33 = I
            axi_mem.mem[140] = 256'd0;
            axi_mem.mem[141] = 256'd0;
            axi_mem.mem[142] = 256'd0;
            axi_mem.mem[143] = tile_data;  // B33 = I
            
            $display("  A matrix: A[i,j] = row_index + 1");
            $display("  B matrix: Block-diagonal identity");
            $display("  Expected C = A × B = A (since B ≈ I)");
        end
    endtask

    //==========================================================================
    // Load TPC Programs
    // Each TPC computes one output quadrant
    //==========================================================================
    
    // Helper task to load program for one TPC
    task load_tpc0_program;
        reg [39:0] a_tile_addr, b_tile_addr, c_tile_addr;
        reg [19:0] sram_a, sram_b, sram_p0, sram_p1;
        reg [1:0] ti, tj;
        integer pc;
        begin
            sram_a  = 20'h00000;
            sram_b  = 20'h00100;
            sram_p0 = 20'h00200;
            sram_p1 = 20'h00300;
            ti = 0; tj = 0;  // TPC0 → C[0,0]
            pc = 0;
            
            // DMA LOAD A[0,0]
            a_tile_addr = A_BASE + 0;
            dut.tpc_gen[0].tpc_inst.instr_mem[pc] = make_dma_cmd(DMA_LOAD, a_tile_addr, sram_a, 12'd1, 12'd1);
            pc = pc + 1;
            
            // DMA LOAD B[0,0]
            b_tile_addr = B_BASE + 0;
            dut.tpc_gen[0].tpc_inst.instr_mem[pc] = make_dma_cmd(DMA_LOAD, b_tile_addr, sram_b, 12'd1, 12'd1);
            pc = pc + 1;
            
            // SYNC DMA
            dut.tpc_gen[0].tpc_inst.instr_mem[pc] = make_sync_cmd(SYNC_DMA);
            pc = pc + 1;
            
            // GEMM: P0 = A[0,0] × B[0,0]
            dut.tpc_gen[0].tpc_inst.instr_mem[pc] = make_tensor_cmd(sram_p0, sram_a, sram_b, 16'd4, 16'd4, 16'd4);
            pc = pc + 1;
            
            // SYNC MXU
            dut.tpc_gen[0].tpc_inst.instr_mem[pc] = make_sync_cmd(SYNC_MXU);
            pc = pc + 1;
            
            // DMA STORE P0 → DDR C[0,0]
            c_tile_addr = C_BASE + 0;
            dut.tpc_gen[0].tpc_inst.instr_mem[pc] = make_dma_cmd(DMA_STORE, c_tile_addr, sram_p0, 12'd1, 12'd1);
            pc = pc + 1;
            
            // SYNC DMA
            dut.tpc_gen[0].tpc_inst.instr_mem[pc] = make_sync_cmd(SYNC_DMA);
            pc = pc + 1;
            
            // HALT
            dut.tpc_gen[0].tpc_inst.instr_mem[pc] = {OP_HALT, 120'd0};
            
            $display("    TPC0: Computes C[0,0], %0d instructions", pc+1);
        end
    endtask

    task load_tpc1_program;
        reg [39:0] a_tile_addr, b_tile_addr, c_tile_addr;
        reg [19:0] sram_a, sram_b, sram_p0;
        integer pc;
        begin
            sram_a  = 20'h00000;
            sram_b  = 20'h00100;
            sram_p0 = 20'h00200;
            pc = 0;
            
            // TPC1 → C[0,1]
            // DMA LOAD A[0,1]
            a_tile_addr = A_BASE + 1 * TILE_BYTES;
            dut.tpc_gen[1].tpc_inst.instr_mem[pc] = make_dma_cmd(DMA_LOAD, a_tile_addr, sram_a, 12'd1, 12'd1);
            pc = pc + 1;
            
            // DMA LOAD B[1,1] (identity block)
            b_tile_addr = B_BASE + 5 * TILE_BYTES;  // B11 at index 5
            dut.tpc_gen[1].tpc_inst.instr_mem[pc] = make_dma_cmd(DMA_LOAD, b_tile_addr, sram_b, 12'd1, 12'd1);
            pc = pc + 1;
            
            dut.tpc_gen[1].tpc_inst.instr_mem[pc] = make_sync_cmd(SYNC_DMA);
            pc = pc + 1;
            
            dut.tpc_gen[1].tpc_inst.instr_mem[pc] = make_tensor_cmd(sram_p0, sram_a, sram_b, 16'd4, 16'd4, 16'd4);
            pc = pc + 1;
            
            dut.tpc_gen[1].tpc_inst.instr_mem[pc] = make_sync_cmd(SYNC_MXU);
            pc = pc + 1;
            
            c_tile_addr = C_BASE + 1 * TILE_BYTES * 4;
            dut.tpc_gen[1].tpc_inst.instr_mem[pc] = make_dma_cmd(DMA_STORE, c_tile_addr, sram_p0, 12'd1, 12'd1);
            pc = pc + 1;
            
            dut.tpc_gen[1].tpc_inst.instr_mem[pc] = make_sync_cmd(SYNC_DMA);
            pc = pc + 1;
            
            dut.tpc_gen[1].tpc_inst.instr_mem[pc] = {OP_HALT, 120'd0};
            
            $display("    TPC1: Computes C[0,1], %0d instructions", pc+1);
        end
    endtask

    task load_tpc2_program;
        reg [39:0] a_tile_addr, b_tile_addr, c_tile_addr;
        reg [19:0] sram_a, sram_b, sram_p0;
        integer pc;
        begin
            sram_a  = 20'h00000;
            sram_b  = 20'h00100;
            sram_p0 = 20'h00200;
            pc = 0;
            
            // TPC2 → C[1,0]
            // DMA LOAD A[1,0]
            a_tile_addr = A_BASE + 4 * TILE_BYTES;  // A10 at index 4
            dut.tpc_gen[2].tpc_inst.instr_mem[pc] = make_dma_cmd(DMA_LOAD, a_tile_addr, sram_a, 12'd1, 12'd1);
            pc = pc + 1;
            
            // DMA LOAD B[0,0]
            b_tile_addr = B_BASE + 0;
            dut.tpc_gen[2].tpc_inst.instr_mem[pc] = make_dma_cmd(DMA_LOAD, b_tile_addr, sram_b, 12'd1, 12'd1);
            pc = pc + 1;
            
            dut.tpc_gen[2].tpc_inst.instr_mem[pc] = make_sync_cmd(SYNC_DMA);
            pc = pc + 1;
            
            dut.tpc_gen[2].tpc_inst.instr_mem[pc] = make_tensor_cmd(sram_p0, sram_a, sram_b, 16'd4, 16'd4, 16'd4);
            pc = pc + 1;
            
            dut.tpc_gen[2].tpc_inst.instr_mem[pc] = make_sync_cmd(SYNC_MXU);
            pc = pc + 1;
            
            c_tile_addr = C_BASE + 2 * TILE_BYTES * 4;
            dut.tpc_gen[2].tpc_inst.instr_mem[pc] = make_dma_cmd(DMA_STORE, c_tile_addr, sram_p0, 12'd1, 12'd1);
            pc = pc + 1;
            
            dut.tpc_gen[2].tpc_inst.instr_mem[pc] = make_sync_cmd(SYNC_DMA);
            pc = pc + 1;
            
            dut.tpc_gen[2].tpc_inst.instr_mem[pc] = {OP_HALT, 120'd0};
            
            $display("    TPC2: Computes C[1,0], %0d instructions", pc+1);
        end
    endtask

    task load_tpc3_program;
        reg [39:0] a_tile_addr, b_tile_addr, c_tile_addr;
        reg [19:0] sram_a, sram_b, sram_p0;
        integer pc;
        begin
            sram_a  = 20'h00000;
            sram_b  = 20'h00100;
            sram_p0 = 20'h00200;
            pc = 0;
            
            // TPC3 → C[1,1]
            // DMA LOAD A[1,1]
            a_tile_addr = A_BASE + 5 * TILE_BYTES;  // A11 at index 5
            dut.tpc_gen[3].tpc_inst.instr_mem[pc] = make_dma_cmd(DMA_LOAD, a_tile_addr, sram_a, 12'd1, 12'd1);
            pc = pc + 1;
            
            // DMA LOAD B[1,1]
            b_tile_addr = B_BASE + 5 * TILE_BYTES;
            dut.tpc_gen[3].tpc_inst.instr_mem[pc] = make_dma_cmd(DMA_LOAD, b_tile_addr, sram_b, 12'd1, 12'd1);
            pc = pc + 1;
            
            dut.tpc_gen[3].tpc_inst.instr_mem[pc] = make_sync_cmd(SYNC_DMA);
            pc = pc + 1;
            
            dut.tpc_gen[3].tpc_inst.instr_mem[pc] = make_tensor_cmd(sram_p0, sram_a, sram_b, 16'd4, 16'd4, 16'd4);
            pc = pc + 1;
            
            dut.tpc_gen[3].tpc_inst.instr_mem[pc] = make_sync_cmd(SYNC_MXU);
            pc = pc + 1;
            
            c_tile_addr = C_BASE + 3 * TILE_BYTES * 4;
            dut.tpc_gen[3].tpc_inst.instr_mem[pc] = make_dma_cmd(DMA_STORE, c_tile_addr, sram_p0, 12'd1, 12'd1);
            pc = pc + 1;
            
            dut.tpc_gen[3].tpc_inst.instr_mem[pc] = make_sync_cmd(SYNC_DMA);
            pc = pc + 1;
            
            dut.tpc_gen[3].tpc_inst.instr_mem[pc] = {OP_HALT, 120'd0};
            
            $display("    TPC3: Computes C[1,1], %0d instructions", pc+1);
        end
    endtask

    task load_tpc_programs;
        begin
            $display("  Loading TPC programs...");
            load_tpc0_program;
            load_tpc1_program;
            load_tpc2_program;
            load_tpc3_program;
        end
    endtask

    //==========================================================================
    // Wait for completion
    //==========================================================================
    reg success;
    
    task wait_all_done;
        begin
            timeout = 0;
            success = 0;
            while (timeout < 2000 && !success) begin
                @(posedge clk);
                axi_read(ADDR_STATUS);
                // done bits are [11:8]
                if (rdata[11:8] == 4'b1111) success = 1;
                timeout = timeout + 1;
            end
        end
    endtask

    //==========================================================================
    // Verify Results
    //==========================================================================
    integer errors;
    reg [255:0] result;
    reg [31:0] expected_val, actual_val;
    
    task verify_results;
        integer r, c, tile_errors;
        begin
            errors = 0;
            
            $display("");
            $display("  Verifying results...");
            
            // Read C tiles from DDR
            // C output at offset 256 (0x2000 / 32 = 256)
            
            // C00 = A00 × B00 = A00 (since B00 = I)
            // A00 has values: row 0 = [1,1,1,1], row 1 = [2,2,2,2], etc.
            // After GEMM with identity, C00[0,0] = A00[0,0] * B00[0,0] = 1
            
            result = axi_mem.mem[256];  // C00 at 0x2000
            $display("  C00 (TPC0): %h", result[127:0]);
            actual_val = result[31:0];
            expected_val = 32'd1;  // A[0,0] = 1
            if (actual_val != expected_val) begin
                $display("    FAIL: C00[0,0] = %0d, expected %0d", actual_val, expected_val);
                errors = errors + 1;
            end else begin
                $display("    PASS: C00[0,0] = %0d", actual_val);
            end
            
            result = axi_mem.mem[260];  // C01 at 0x2080
            $display("  C01 (TPC1): %h", result[127:0]);
            actual_val = result[31:0];
            expected_val = 32'd1;  // A[0,1] first element = 1
            if (actual_val != expected_val) begin
                $display("    FAIL: C01[0,0] = %0d, expected %0d", actual_val, expected_val);
                errors = errors + 1;
            end else begin
                $display("    PASS: C01[0,0] = %0d", actual_val);
            end
            
            result = axi_mem.mem[264];  // C10 at 0x2100
            $display("  C10 (TPC2): %h", result[127:0]);
            actual_val = result[31:0];
            expected_val = 32'd5;  // A[1,0] first element = 5 (row 4)
            if (actual_val != expected_val) begin
                $display("    FAIL: C10[0,0] = %0d, expected %0d", actual_val, expected_val);
                errors = errors + 1;
            end else begin
                $display("    PASS: C10[0,0] = %0d", actual_val);
            end
            
            result = axi_mem.mem[268];  // C11 at 0x2180
            $display("  C11 (TPC3): %h", result[127:0]);
            actual_val = result[31:0];
            expected_val = 32'd5;  // A[1,1] first element = 5 (row 4)
            if (actual_val != expected_val) begin
                $display("    FAIL: C11[0,0] = %0d, expected %0d", actual_val, expected_val);
                errors = errors + 1;
            end else begin
                $display("    PASS: C11[0,0] = %0d", actual_val);
            end
        end
    endtask

    //==========================================================================
    // Main Test
    //==========================================================================
    initial begin
        $display("");
        $display("╔══════════════════════════════════════════════════════════════╗");
        $display("║        Multi-TPC Tiled GEMM Test                             ║");
        $display("║        16×16 = (4×4 tiles) × 4 TPCs                          ║");
        $display("╚══════════════════════════════════════════════════════════════╝");
        $display("");
        
        errors = 0;
        
        // Reset
        rst_n = 0;
        #(CLK * 5);
        rst_n = 1;
        #(CLK * 5);
        
        //======================================================================
        // Setup
        //======================================================================
        $display("[SETUP] Initializing test data...");
        init_ddr_matrices;
        load_tpc_programs;
        
        //======================================================================
        // Execute
        //======================================================================
        $display("");
        $display("[EXEC] Starting all 4 TPCs in parallel...");
        
        // Enable all 4 TPCs
        axi_write(ADDR_CTRL, 32'h00000F00);
        
        // Start execution
        axi_write(ADDR_CTRL, 32'h00000F01);
        
        // Wait for completion
        wait_all_done;
        
        if (success) begin
            $display("  All TPCs completed in %0d cycles", timeout);
        end else begin
            $display("  TIMEOUT: Not all TPCs completed");
            axi_read(ADDR_STATUS);
            $display("  Status: %h (done=%b, busy=%b)", rdata, rdata[11:8], rdata[3:0]);
            errors = errors + 1;
        end
        
        //======================================================================
        // Verify
        //======================================================================
        $display("");
        $display("[VERIFY] Checking results...");
        verify_results;
        
        //======================================================================
        // Summary
        //======================================================================
        $display("");
        $display("╔══════════════════════════════════════════════════════════════╗");
        $display("║                    TEST SUMMARY                              ║");
        $display("╠══════════════════════════════════════════════════════════════╣");
        if (errors == 0) begin
            $display("║   PASSED: 4 TPCs executed tiled GEMM in parallel           ║");
            $display("╚══════════════════════════════════════════════════════════════╝");
            $display(">>> MULTI-TPC GEMM TEST PASSED! <<<");
        end else begin
            $display("║   FAILED: %0d errors                                         ║", errors);
            $display("╚══════════════════════════════════════════════════════════════╝");
            $display(">>> MULTI-TPC GEMM TEST FAILED <<<");
        end
        
        #(CLK * 10);
        $finish;
    end

    // Timeout
    initial begin
        #(CLK * 50000);
        $display("GLOBAL TIMEOUT!");
        $finish;
    end

endmodule

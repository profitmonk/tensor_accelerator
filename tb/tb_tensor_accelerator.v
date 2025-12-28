//==============================================================================
// Tensor Accelerator Testbench
//
// Comprehensive verification environment:
// 1. AXI-Lite host interface driver
// 2. AXI4 memory model
// 3. Instruction loading
// 4. Test stimulus generation
// 5. Result checking
//==============================================================================

`timescale 1ns / 1ps

module tb_tensor_accelerator;

    //==========================================================================
    // Parameters
    //==========================================================================
    
    parameter CLK_PERIOD = 10;  // 100 MHz
    
    // Match top-level parameters
    parameter GRID_X      = 2;
    parameter GRID_Y      = 2;
    parameter NUM_TPCS    = GRID_X * GRID_Y;
    parameter ARRAY_SIZE  = 16;
    parameter DATA_WIDTH  = 8;
    parameter ACC_WIDTH   = 32;
    parameter SRAM_WIDTH  = 256;
    parameter SRAM_ADDR_W = 20;
    parameter AXI_ADDR_W  = 40;
    parameter AXI_DATA_W  = 256;
    parameter CTRL_ADDR_W = 12;
    parameter CTRL_DATA_W = 32;

    //==========================================================================
    // Clock and Reset
    //==========================================================================
    
    reg clk;
    reg rst_n;
    
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    initial begin
        rst_n = 0;
        #(CLK_PERIOD * 10);
        rst_n = 1;
    end

    //==========================================================================
    // DUT Signals
    //==========================================================================
    
    // Control interface
    reg  [CTRL_ADDR_W-1:0]  s_axi_ctrl_awaddr;
    reg                      s_axi_ctrl_awvalid;
    wire                     s_axi_ctrl_awready;
    reg  [CTRL_DATA_W-1:0]  s_axi_ctrl_wdata;
    reg  [3:0]              s_axi_ctrl_wstrb;
    reg                      s_axi_ctrl_wvalid;
    wire                     s_axi_ctrl_wready;
    wire [1:0]              s_axi_ctrl_bresp;
    wire                     s_axi_ctrl_bvalid;
    reg                      s_axi_ctrl_bready;
    reg  [CTRL_ADDR_W-1:0]  s_axi_ctrl_araddr;
    reg                      s_axi_ctrl_arvalid;
    wire                     s_axi_ctrl_arready;
    wire [CTRL_DATA_W-1:0]  s_axi_ctrl_rdata;
    wire [1:0]              s_axi_ctrl_rresp;
    wire                     s_axi_ctrl_rvalid;
    reg                      s_axi_ctrl_rready;
    
    // Memory interface
    wire [3:0]              m_axi_awid;
    wire [AXI_ADDR_W-1:0]   m_axi_awaddr;
    wire [7:0]              m_axi_awlen;
    wire [2:0]              m_axi_awsize;
    wire [1:0]              m_axi_awburst;
    wire                     m_axi_awvalid;
    reg                      m_axi_awready;
    wire [AXI_DATA_W-1:0]   m_axi_wdata;
    wire [31:0]             m_axi_wstrb;
    wire                     m_axi_wlast;
    wire                     m_axi_wvalid;
    reg                      m_axi_wready;
    reg  [3:0]              m_axi_bid;
    reg  [1:0]              m_axi_bresp;
    reg                      m_axi_bvalid;
    wire                     m_axi_bready;
    wire [3:0]              m_axi_arid;
    wire [AXI_ADDR_W-1:0]   m_axi_araddr;
    wire [7:0]              m_axi_arlen;
    wire [2:0]              m_axi_arsize;
    wire [1:0]              m_axi_arburst;
    wire                     m_axi_arvalid;
    reg                      m_axi_arready;
    reg  [3:0]              m_axi_rid;
    reg  [AXI_DATA_W-1:0]   m_axi_rdata;
    reg  [1:0]              m_axi_rresp;
    reg                      m_axi_rlast;
    reg                      m_axi_rvalid;
    wire                     m_axi_rready;
    
    // Interrupt
    wire irq;

    //==========================================================================
    // DUT Instantiation
    //==========================================================================
    
    tensor_accelerator_top #(
        .GRID_X      (GRID_X),
        .GRID_Y      (GRID_Y),
        .ARRAY_SIZE  (ARRAY_SIZE),
        .DATA_WIDTH  (DATA_WIDTH),
        .ACC_WIDTH   (ACC_WIDTH),
        .SRAM_WIDTH  (SRAM_WIDTH),
        .SRAM_ADDR_W (SRAM_ADDR_W),
        .AXI_ADDR_W  (AXI_ADDR_W),
        .AXI_DATA_W  (AXI_DATA_W),
        .CTRL_ADDR_W (CTRL_ADDR_W),
        .CTRL_DATA_W (CTRL_DATA_W)
    ) dut (
        .clk                (clk),
        .rst_n              (rst_n),
        
        // Control
        .s_axi_ctrl_awaddr  (s_axi_ctrl_awaddr),
        .s_axi_ctrl_awvalid (s_axi_ctrl_awvalid),
        .s_axi_ctrl_awready (s_axi_ctrl_awready),
        .s_axi_ctrl_wdata   (s_axi_ctrl_wdata),
        .s_axi_ctrl_wstrb   (s_axi_ctrl_wstrb),
        .s_axi_ctrl_wvalid  (s_axi_ctrl_wvalid),
        .s_axi_ctrl_wready  (s_axi_ctrl_wready),
        .s_axi_ctrl_bresp   (s_axi_ctrl_bresp),
        .s_axi_ctrl_bvalid  (s_axi_ctrl_bvalid),
        .s_axi_ctrl_bready  (s_axi_ctrl_bready),
        .s_axi_ctrl_araddr  (s_axi_ctrl_araddr),
        .s_axi_ctrl_arvalid (s_axi_ctrl_arvalid),
        .s_axi_ctrl_arready (s_axi_ctrl_arready),
        .s_axi_ctrl_rdata   (s_axi_ctrl_rdata),
        .s_axi_ctrl_rresp   (s_axi_ctrl_rresp),
        .s_axi_ctrl_rvalid  (s_axi_ctrl_rvalid),
        .s_axi_ctrl_rready  (s_axi_ctrl_rready),
        
        // Memory
        .m_axi_awid         (m_axi_awid),
        .m_axi_awaddr       (m_axi_awaddr),
        .m_axi_awlen        (m_axi_awlen),
        .m_axi_awsize       (m_axi_awsize),
        .m_axi_awburst      (m_axi_awburst),
        .m_axi_awvalid      (m_axi_awvalid),
        .m_axi_awready      (m_axi_awready),
        .m_axi_wdata        (m_axi_wdata),
        .m_axi_wstrb        (m_axi_wstrb),
        .m_axi_wlast        (m_axi_wlast),
        .m_axi_wvalid       (m_axi_wvalid),
        .m_axi_wready       (m_axi_wready),
        .m_axi_bid          (m_axi_bid),
        .m_axi_bresp        (m_axi_bresp),
        .m_axi_bvalid       (m_axi_bvalid),
        .m_axi_bready       (m_axi_bready),
        .m_axi_arid         (m_axi_arid),
        .m_axi_araddr       (m_axi_araddr),
        .m_axi_arlen        (m_axi_arlen),
        .m_axi_arsize       (m_axi_arsize),
        .m_axi_arburst      (m_axi_arburst),
        .m_axi_arvalid      (m_axi_arvalid),
        .m_axi_arready      (m_axi_arready),
        .m_axi_rid          (m_axi_rid),
        .m_axi_rdata        (m_axi_rdata),
        .m_axi_rresp        (m_axi_rresp),
        .m_axi_rlast        (m_axi_rlast),
        .m_axi_rvalid       (m_axi_rvalid),
        .m_axi_rready       (m_axi_rready),
        
        .irq                (irq)
    );

    //==========================================================================
    // Memory Model
    //==========================================================================
    
    // Simplified memory model - 1MB addressable
    reg [7:0] memory [0:1048575];
    
    // AXI read state
    reg [7:0] read_burst_cnt;
    reg [AXI_ADDR_W-1:0] read_addr;
    reg [7:0] read_len;
    
    // AXI write state
    reg [7:0] write_burst_cnt;
    reg [AXI_ADDR_W-1:0] write_addr;
    
    // Memory initialization
    initial begin
        integer i;
        for (i = 0; i < 1048576; i = i + 1) begin
            memory[i] = 8'h00;
        end
    end
    
    // AXI read handling
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_arready <= 1'b1;
            m_axi_rvalid <= 1'b0;
            m_axi_rlast <= 1'b0;
            read_burst_cnt <= 0;
        end else begin
            // Accept read address
            if (m_axi_arvalid && m_axi_arready) begin
                read_addr <= m_axi_araddr;
                read_len <= m_axi_arlen;
                read_burst_cnt <= 0;
                m_axi_arready <= 1'b0;
            end
            
            // Generate read data
            if (!m_axi_arready && !m_axi_rvalid) begin
                m_axi_rvalid <= 1'b1;
                m_axi_rid <= m_axi_arid;
                m_axi_rresp <= 2'b00;
                
                // Read 32 bytes (256 bits) from memory
                for (integer j = 0; j < 32; j = j + 1) begin
                    m_axi_rdata[j*8 +: 8] <= memory[(read_addr[19:0] + read_burst_cnt*32 + j) % 1048576];
                end
                
                m_axi_rlast <= (read_burst_cnt >= read_len);
            end
            
            // Handle read data acceptance
            if (m_axi_rvalid && m_axi_rready) begin
                if (m_axi_rlast) begin
                    m_axi_rvalid <= 1'b0;
                    m_axi_rlast <= 1'b0;
                    m_axi_arready <= 1'b1;
                end else begin
                    read_burst_cnt <= read_burst_cnt + 1;
                    for (integer j = 0; j < 32; j = j + 1) begin
                        m_axi_rdata[j*8 +: 8] <= memory[(read_addr[19:0] + (read_burst_cnt+1)*32 + j) % 1048576];
                    end
                    m_axi_rlast <= ((read_burst_cnt + 1) >= read_len);
                end
            end
        end
    end
    
    // AXI write handling
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_awready <= 1'b1;
            m_axi_wready <= 1'b0;
            m_axi_bvalid <= 1'b0;
            write_burst_cnt <= 0;
        end else begin
            // Accept write address
            if (m_axi_awvalid && m_axi_awready) begin
                write_addr <= m_axi_awaddr;
                m_axi_awready <= 1'b0;
                m_axi_wready <= 1'b1;
            end
            
            // Accept write data
            if (m_axi_wvalid && m_axi_wready) begin
                // Write 32 bytes to memory
                for (integer j = 0; j < 32; j = j + 1) begin
                    if (m_axi_wstrb[j]) begin
                        memory[(write_addr[19:0] + write_burst_cnt*32 + j) % 1048576] <= m_axi_wdata[j*8 +: 8];
                    end
                end
                
                if (m_axi_wlast) begin
                    m_axi_wready <= 1'b0;
                    m_axi_bvalid <= 1'b1;
                    m_axi_bid <= m_axi_awid;
                    m_axi_bresp <= 2'b00;
                    write_burst_cnt <= 0;
                end else begin
                    write_burst_cnt <= write_burst_cnt + 1;
                end
            end
            
            // Handle write response acceptance
            if (m_axi_bvalid && m_axi_bready) begin
                m_axi_bvalid <= 1'b0;
                m_axi_awready <= 1'b1;
            end
        end
    end

    //==========================================================================
    // AXI-Lite Tasks
    //==========================================================================
    
    task axi_lite_write;
        input [CTRL_ADDR_W-1:0] addr;
        input [CTRL_DATA_W-1:0] data;
        begin
            @(posedge clk);
            s_axi_ctrl_awaddr <= addr;
            s_axi_ctrl_awvalid <= 1'b1;
            s_axi_ctrl_wdata <= data;
            s_axi_ctrl_wstrb <= 4'hF;
            s_axi_ctrl_wvalid <= 1'b1;
            s_axi_ctrl_bready <= 1'b1;
            
            // Wait for address and data accepted
            wait(s_axi_ctrl_awready && s_axi_ctrl_wready);
            @(posedge clk);
            s_axi_ctrl_awvalid <= 1'b0;
            s_axi_ctrl_wvalid <= 1'b0;
            
            // Wait for response
            wait(s_axi_ctrl_bvalid);
            @(posedge clk);
            s_axi_ctrl_bready <= 1'b0;
            
            $display("[%0t] AXI-Lite Write: addr=0x%03h data=0x%08h", $time, addr, data);
        end
    endtask
    
    task axi_lite_read;
        input  [CTRL_ADDR_W-1:0] addr;
        output [CTRL_DATA_W-1:0] data;
        begin
            @(posedge clk);
            s_axi_ctrl_araddr <= addr;
            s_axi_ctrl_arvalid <= 1'b1;
            s_axi_ctrl_rready <= 1'b1;
            
            // Wait for address accepted
            wait(s_axi_ctrl_arready);
            @(posedge clk);
            s_axi_ctrl_arvalid <= 1'b0;
            
            // Wait for data
            wait(s_axi_ctrl_rvalid);
            data = s_axi_ctrl_rdata;
            @(posedge clk);
            s_axi_ctrl_rready <= 1'b0;
            
            $display("[%0t] AXI-Lite Read: addr=0x%03h data=0x%08h", $time, addr, data);
        end
    endtask

    //==========================================================================
    // Test Utilities
    //==========================================================================
    
    // Load test data into memory
    task load_test_data;
        input [AXI_ADDR_W-1:0] base_addr;
        input [31:0] size;
        input [7:0] pattern;
        integer i;
        begin
            $display("[%0t] Loading test data: addr=0x%h size=%0d pattern=0x%02h", 
                     $time, base_addr, size, pattern);
            for (i = 0; i < size; i = i + 1) begin
                memory[(base_addr[19:0] + i) % 1048576] = pattern + i[7:0];
            end
        end
    endtask
    
    // Load random matrix
    task load_random_matrix;
        input [AXI_ADDR_W-1:0] base_addr;
        input [15:0] rows;
        input [15:0] cols;
        integer i, j;
        begin
            $display("[%0t] Loading random matrix: addr=0x%h rows=%0d cols=%0d", 
                     $time, base_addr, rows, cols);
            for (i = 0; i < rows; i = i + 1) begin
                for (j = 0; j < cols; j = j + 1) begin
                    memory[(base_addr[19:0] + i*cols + j) % 1048576] = $random & 8'h7F;  // INT8, positive
                end
            end
        end
    endtask
    
    // Load instructions from hex file
    task load_instructions;
        input [AXI_ADDR_W-1:0] base_addr;
        input [1023:0] filename;
        integer fd, i, cnt;
        reg [127:0] instr;
        begin
            $display("[%0t] Loading instructions from %s to addr=0x%h", $time, filename, base_addr);
            
            // For simulation, we'll load predefined instructions
            // In real flow, use $readmemh
            
            // Simple GEMM test program
            // NOP
            memory[(base_addr[19:0] + 0) % 1048576] = 8'h00;
            for (i = 1; i < 16; i = i + 1) begin
                memory[(base_addr[19:0] + i) % 1048576] = 8'h00;
            end
            
            // TENSOR.GEMM dst=0x6000, src0=0x0000, src1=0x2000, M=16, N=16, K=16
            memory[(base_addr[19:0] + 16) % 1048576] = 8'h01;   // Opcode: TENSOR
            memory[(base_addr[19:0] + 17) % 1048576] = 8'h01;   // Subop: GEMM
            memory[(base_addr[19:0] + 18) % 1048576] = 8'h60;   // dst high
            memory[(base_addr[19:0] + 19) % 1048576] = 8'h00;   // dst low
            memory[(base_addr[19:0] + 20) % 1048576] = 8'h00;   // src0 high
            memory[(base_addr[19:0] + 21) % 1048576] = 8'h00;   // src0 low
            memory[(base_addr[19:0] + 22) % 1048576] = 8'h20;   // src1 high
            memory[(base_addr[19:0] + 23) % 1048576] = 8'h00;   // src1 low
            memory[(base_addr[19:0] + 24) % 1048576] = 8'h00;   // M high
            memory[(base_addr[19:0] + 25) % 1048576] = 8'h10;   // M low (16)
            memory[(base_addr[19:0] + 26) % 1048576] = 8'h00;   // N high
            memory[(base_addr[19:0] + 27) % 1048576] = 8'h10;   // N low (16)
            memory[(base_addr[19:0] + 28) % 1048576] = 8'h00;   // K high
            memory[(base_addr[19:0] + 29) % 1048576] = 8'h10;   // K low (16)
            memory[(base_addr[19:0] + 30) % 1048576] = 8'h00;   // flags high
            memory[(base_addr[19:0] + 31) % 1048576] = 8'h00;   // flags low
            
            // SYNC.WAIT_MXU
            memory[(base_addr[19:0] + 32) % 1048576] = 8'h04;   // Opcode: SYNC
            memory[(base_addr[19:0] + 33) % 1048576] = 8'h01;   // Subop: WAIT_MXU
            for (i = 34; i < 48; i = i + 1) begin
                memory[(base_addr[19:0] + i) % 1048576] = 8'h00;
            end
            
            // HALT
            memory[(base_addr[19:0] + 48) % 1048576] = 8'hFF;   // Opcode: HALT
            for (i = 49; i < 64; i = i + 1) begin
                memory[(base_addr[19:0] + i) % 1048576] = 8'h00;
            end
            
            $display("[%0t] Loaded 4 instructions", $time);
        end
    endtask
    
    // Check results
    task check_gemm_result;
        input [AXI_ADDR_W-1:0] result_addr;
        input [AXI_ADDR_W-1:0] a_addr;
        input [AXI_ADDR_W-1:0] b_addr;
        input [15:0] M;
        input [15:0] N;
        input [15:0] K;
        
        integer i, j, k;
        integer expected, actual;
        integer errors;
        begin
            errors = 0;
            
            for (i = 0; i < M && i < 4; i = i + 1) begin  // Check first few rows
                for (j = 0; j < N && j < 4; j = j + 1) begin  // Check first few cols
                    // Compute expected: C[i,j] = sum(A[i,k] * B[k,j])
                    expected = 0;
                    for (k = 0; k < K; k = k + 1) begin
                        expected = expected + 
                            $signed(memory[(a_addr[19:0] + i*K + k) % 1048576]) *
                            $signed(memory[(b_addr[19:0] + k*N + j) % 1048576]);
                    end
                    
                    // Read actual (4 bytes for INT32 accumulator)
                    actual = {memory[(result_addr[19:0] + (i*N + j)*4 + 3) % 1048576],
                              memory[(result_addr[19:0] + (i*N + j)*4 + 2) % 1048576],
                              memory[(result_addr[19:0] + (i*N + j)*4 + 1) % 1048576],
                              memory[(result_addr[19:0] + (i*N + j)*4 + 0) % 1048576]};
                    
                    if (expected !== actual) begin
                        $display("ERROR: C[%0d,%0d] expected=%0d actual=%0d", i, j, expected, actual);
                        errors = errors + 1;
                    end
                end
            end
            
            if (errors == 0) begin
                $display("[%0t] GEMM result check PASSED", $time);
            end else begin
                $display("[%0t] GEMM result check FAILED with %0d errors", $time, errors);
            end
        end
    endtask

    //==========================================================================
    // Test Cases
    //==========================================================================
    
    // Register addresses
    localparam ADDR_CTRL       = 12'h000;
    localparam ADDR_STATUS     = 12'h004;
    localparam ADDR_IRQ_EN     = 12'h008;
    localparam ADDR_IRQ_STATUS = 12'h00C;
    localparam ADDR_TPC0_PC    = 12'h100;
    
    reg [31:0] status;
    
    initial begin
        // Initialize signals
        s_axi_ctrl_awaddr  = 0;
        s_axi_ctrl_awvalid = 0;
        s_axi_ctrl_wdata   = 0;
        s_axi_ctrl_wstrb   = 0;
        s_axi_ctrl_wvalid  = 0;
        s_axi_ctrl_bready  = 0;
        s_axi_ctrl_araddr  = 0;
        s_axi_ctrl_arvalid = 0;
        s_axi_ctrl_rready  = 0;
        
        // Wait for reset
        wait(rst_n);
        #(CLK_PERIOD * 10);
        
        $display("========================================");
        $display("Tensor Accelerator Testbench");
        $display("========================================");
        
        //----------------------------------------------------------------------
        // Test 1: Simple GEMM (16x16)
        //----------------------------------------------------------------------
        $display("\n[TEST 1] Simple 16x16 GEMM");
        
        // Load test matrices
        load_random_matrix(40'h80000000, 16, 16);  // Matrix A: 16x16
        load_random_matrix(40'h80002000, 16, 16);  // Matrix B: 16x16
        
        // Load instructions (will be loaded to TPC instruction memory)
        load_instructions(40'h80010000, "test_gemm.hex");
        
        // Configure TPC 0
        axi_lite_write(ADDR_TPC0_PC, 32'h00000000);  // Start PC = 0
        
        // Enable TPC 0 and start
        axi_lite_write(ADDR_CTRL, 32'h00000101);  // Enable TPC0, start
        
        // Wait for completion
        status = 0;
        while ((status & 32'h00010000) == 0) begin  // Wait for all_done
            #(CLK_PERIOD * 100);
            axi_lite_read(ADDR_STATUS, status);
        end
        
        $display("[TEST 1] Completed");
        
        //----------------------------------------------------------------------
        // Test 2: Multiple TPCs in parallel
        //----------------------------------------------------------------------
        $display("\n[TEST 2] Multiple TPCs in parallel");
        
        // Configure all 4 TPCs with same program
        axi_lite_write(12'h100, 32'h00000000);  // TPC0 PC
        axi_lite_write(12'h110, 32'h00000000);  // TPC1 PC
        axi_lite_write(12'h120, 32'h00000000);  // TPC2 PC
        axi_lite_write(12'h130, 32'h00000000);  // TPC3 PC
        
        // Enable all TPCs and start
        axi_lite_write(ADDR_CTRL, 32'h00000F01);  // Enable TPC0-3, start
        
        // Wait for completion
        status = 0;
        while ((status & 32'h00010000) == 0) begin
            #(CLK_PERIOD * 100);
            axi_lite_read(ADDR_STATUS, status);
        end
        
        $display("[TEST 2] Completed");
        
        //----------------------------------------------------------------------
        // Test Complete
        //----------------------------------------------------------------------
        
        #(CLK_PERIOD * 100);
        $display("\n========================================");
        $display("All tests completed");
        $display("========================================");
        $finish;
    end

    //==========================================================================
    // Waveform Dump
    //==========================================================================
    
    initial begin
        $dumpfile("tb_tensor_accelerator.vcd");
        $dumpvars(0, tb_tensor_accelerator);
    end

    //==========================================================================
    // Timeout
    //==========================================================================
    
    initial begin
        #(CLK_PERIOD * 1000000);  // 10ms timeout
        $display("ERROR: Simulation timeout!");
        $finish;
    end

endmodule

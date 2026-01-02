//==============================================================================
// AXI4 Verification Testbench 
//
// Top-level module for cocotb AXI protocol verification.
// 128-bit data width to support realistic transfer sizes.
// Supports error injection for SLVERR/DECERR coverage.
//==============================================================================

`timescale 1ns / 1ps

module axi_memory_simple #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 128,   // 128-bit for realistic DMA testing
    parameter ID_WIDTH   = 4,
    parameter MEM_DEPTH  = 16384  // 256KB for 128-bit
)(
    input  wire                      clk,
    input  wire                      rst_n,
    
    // Error injection control
    input  wire                      inject_error,
    input  wire [ADDR_WIDTH-1:0]     error_addr_start,
    input  wire [ADDR_WIDTH-1:0]     error_addr_end,
    input  wire [1:0]                error_type,
    
    // Write Address Channel
    input  wire [ID_WIDTH-1:0]       m_axi_awid,
    input  wire [ADDR_WIDTH-1:0]     m_axi_awaddr,
    input  wire [7:0]                m_axi_awlen,
    input  wire [2:0]                m_axi_awsize,
    input  wire [1:0]                m_axi_awburst,
    input  wire                      m_axi_awvalid,
    output reg                       m_axi_awready,
    
    // Write Data Channel
    input  wire [DATA_WIDTH-1:0]     m_axi_wdata,
    input  wire [DATA_WIDTH/8-1:0]   m_axi_wstrb,
    input  wire                      m_axi_wlast,
    input  wire                      m_axi_wvalid,
    output reg                       m_axi_wready,
    
    // Write Response Channel
    output reg  [ID_WIDTH-1:0]       m_axi_bid,
    output reg  [1:0]                m_axi_bresp,
    output reg                       m_axi_bvalid,
    input  wire                      m_axi_bready,
    
    // Read Address Channel
    input  wire [ID_WIDTH-1:0]       m_axi_arid,
    input  wire [ADDR_WIDTH-1:0]     m_axi_araddr,
    input  wire [7:0]                m_axi_arlen,
    input  wire [2:0]                m_axi_arsize,
    input  wire [1:0]                m_axi_arburst,
    input  wire                      m_axi_arvalid,
    output reg                       m_axi_arready,
    
    // Read Data Channel
    output reg  [ID_WIDTH-1:0]       m_axi_rid,
    output reg  [DATA_WIDTH-1:0]     m_axi_rdata,
    output reg  [1:0]                m_axi_rresp,
    output reg                       m_axi_rlast,
    output reg                       m_axi_rvalid,
    input  wire                      m_axi_rready
);

    //--------------------------------------------------------------------------
    // Memory Array - byte addressable
    //--------------------------------------------------------------------------
    reg [7:0] mem [0:MEM_DEPTH*DATA_WIDTH/8-1];  // Byte-addressable memory
    
    // Address LSB for word alignment
    localparam ADDR_LSB = $clog2(DATA_WIDTH/8);  // 4 for 128-bit
    
    //--------------------------------------------------------------------------
    // Write Logic with narrow transfer support
    //--------------------------------------------------------------------------
    reg [1:0] wr_state;
    localparam WR_IDLE = 2'd0, WR_DATA = 2'd1, WR_RESP = 2'd2;
    
    reg [ID_WIDTH-1:0]   wr_id;
    reg [ADDR_WIDTH-1:0] wr_addr;
    reg [ADDR_WIDTH-1:0] wr_addr_latch;
    reg [7:0]            wr_len;
    reg [7:0]            wr_cnt;
    reg [1:0]            wr_burst;
    reg [2:0]            wr_size;
    reg                  wr_error;
    reg [1:0]            wr_error_type;
    
    // Calculate address increment based on size
    function [ADDR_WIDTH-1:0] get_size_bytes;
        input [2:0] size;
        begin
            case (size)
                3'b000: get_size_bytes = 1;    // 1 byte
                3'b001: get_size_bytes = 2;    // 2 bytes
                3'b010: get_size_bytes = 4;    // 4 bytes
                3'b011: get_size_bytes = 8;    // 8 bytes
                3'b100: get_size_bytes = 16;   // 16 bytes
                3'b101: get_size_bytes = 32;   // 32 bytes
                3'b110: get_size_bytes = 64;   // 64 bytes
                3'b111: get_size_bytes = 128;  // 128 bytes
                default: get_size_bytes = 4;
            endcase
        end
    endfunction
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_state <= WR_IDLE;
            m_axi_awready <= 1'b1;
            m_axi_wready <= 1'b0;
            m_axi_bvalid <= 1'b0;
            m_axi_bid <= 0;
            m_axi_bresp <= 2'b00;
            wr_id <= 0;
            wr_addr <= 0;
            wr_addr_latch <= 0;
            wr_len <= 0;
            wr_cnt <= 0;
            wr_burst <= 0;
            wr_size <= 0;
            wr_error <= 0;
            wr_error_type <= 0;
        end else begin
            case (wr_state)
                WR_IDLE: begin
                    m_axi_awready <= 1'b1;
                    m_axi_wready <= 1'b0;
                    m_axi_bvalid <= 1'b0;
                    
                    if (m_axi_awvalid && m_axi_awready) begin
                        wr_id <= m_axi_awid;
                        wr_addr <= m_axi_awaddr;
                        wr_addr_latch <= m_axi_awaddr;
                        wr_len <= m_axi_awlen;
                        wr_burst <= m_axi_awburst;
                        wr_size <= m_axi_awsize;
                        wr_cnt <= 0;
                        wr_error <= inject_error && 
                                   (m_axi_awaddr >= error_addr_start) && 
                                   (m_axi_awaddr <= error_addr_end);
                        wr_error_type <= error_type;
                        m_axi_awready <= 1'b0;
                        m_axi_wready <= 1'b1;
                        wr_state <= WR_DATA;
                    end
                end
                
                WR_DATA: begin
                    if (m_axi_wvalid && m_axi_wready) begin
                        // Write bytes based on strobe
                        begin : wr_byte_loop
                            integer i;
                            for (i = 0; i < DATA_WIDTH/8; i = i + 1) begin
                                if (m_axi_wstrb[i] && (wr_addr + i) < MEM_DEPTH*DATA_WIDTH/8)
                                    mem[wr_addr + i] <= m_axi_wdata[i*8 +: 8];
                            end
                        end
                        
                        // Update address based on burst type and size
                        if (wr_burst == 2'b01) // INCR
                            wr_addr <= wr_addr + get_size_bytes(wr_size);
                            
                        wr_cnt <= wr_cnt + 1;
                        
                        if (m_axi_wlast) begin
                            m_axi_wready <= 1'b0;
                            m_axi_bvalid <= 1'b1;
                            m_axi_bid <= wr_id;
                            m_axi_bresp <= wr_error ? wr_error_type : 2'b00;
                            wr_state <= WR_RESP;
                        end
                    end
                end
                
                WR_RESP: begin
                    if (m_axi_bvalid && m_axi_bready) begin
                        m_axi_bvalid <= 1'b0;
                        wr_state <= WR_IDLE;
                    end
                end
                
                default: wr_state <= WR_IDLE;
            endcase
        end
    end

    //--------------------------------------------------------------------------
    // Read Logic with narrow transfer support
    //--------------------------------------------------------------------------
    reg [1:0] rd_state;
    localparam RD_IDLE = 2'd0, RD_DATA = 2'd1;
    
    reg [ID_WIDTH-1:0]   rd_id;
    reg [ADDR_WIDTH-1:0] rd_addr;
    reg [7:0]            rd_len;
    reg [7:0]            rd_cnt;
    reg [1:0]            rd_burst;
    reg [2:0]            rd_size;
    reg                  rd_error;
    reg [1:0]            rd_error_type;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_state <= RD_IDLE;
            m_axi_arready <= 1'b1;
            m_axi_rvalid <= 1'b0;
            m_axi_rid <= 0;
            m_axi_rdata <= 0;
            m_axi_rresp <= 2'b00;
            m_axi_rlast <= 1'b0;
            rd_id <= 0;
            rd_addr <= 0;
            rd_len <= 0;
            rd_cnt <= 0;
            rd_burst <= 0;
            rd_size <= 0;
            rd_error <= 0;
            rd_error_type <= 0;
        end else begin
            case (rd_state)
                RD_IDLE: begin
                    m_axi_arready <= 1'b1;
                    m_axi_rvalid <= 1'b0;
                    m_axi_rlast <= 1'b0;
                    
                    if (m_axi_arvalid && m_axi_arready) begin
                        rd_id <= m_axi_arid;
                        rd_addr <= m_axi_araddr;
                        rd_len <= m_axi_arlen;
                        rd_burst <= m_axi_arburst;
                        rd_size <= m_axi_arsize;
                        rd_cnt <= 0;
                        rd_error <= inject_error && 
                                   (m_axi_araddr >= error_addr_start) && 
                                   (m_axi_araddr <= error_addr_end);
                        rd_error_type <= error_type;
                        m_axi_arready <= 1'b0;
                        rd_state <= RD_DATA;
                    end
                end
                
                RD_DATA: begin
                    if (!m_axi_rvalid || m_axi_rready) begin
                        // Read bytes into data bus
                        begin : rd_byte_loop
                            integer i;
                            for (i = 0; i < DATA_WIDTH/8; i = i + 1) begin
                                if ((rd_addr + i) < MEM_DEPTH*DATA_WIDTH/8)
                                    m_axi_rdata[i*8 +: 8] <= mem[rd_addr + i];
                                else
                                    m_axi_rdata[i*8 +: 8] <= 8'hDE;
                            end
                        end
                            
                        m_axi_rid <= rd_id;
                        m_axi_rresp <= rd_error ? rd_error_type : 2'b00;
                        m_axi_rvalid <= 1'b1;
                        m_axi_rlast <= (rd_cnt == rd_len);
                        
                        // Update address
                        if (rd_burst == 2'b01)
                            rd_addr <= rd_addr + get_size_bytes(rd_size);
                            
                        rd_cnt <= rd_cnt + 1;
                        
                        if (rd_cnt == rd_len) begin
                            rd_state <= RD_IDLE;
                        end
                    end
                end
                
                default: rd_state <= RD_IDLE;
            endcase
        end
    end

    //--------------------------------------------------------------------------
    // Initialize memory
    //--------------------------------------------------------------------------
    integer init_i;
    initial begin
        for (init_i = 0; init_i < MEM_DEPTH*DATA_WIDTH/8; init_i = init_i + 1)
            mem[init_i] = 8'h0;
    end

endmodule

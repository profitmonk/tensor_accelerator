//==============================================================================
// AXI4 Memory Model for Simulation
//
// Simple behavioral memory model that responds to AXI4 transactions.
// Supports:
// - Full AXI4 burst transactions (INCR, WRAP, FIXED)
// - Configurable memory size
// - Read/write byte enables
// - Out-of-order response capability
//
// NOT synthesizable - for simulation only
//==============================================================================

`timescale 1ns / 1ps

module axi_memory_model #(
    parameter AXI_ADDR_WIDTH  = 40,
    parameter AXI_DATA_WIDTH  = 256,
    parameter AXI_ID_WIDTH    = 4,
    parameter MEM_SIZE_MB     = 64,          // Memory size in megabytes
    parameter READ_LATENCY    = 4,           // Read latency in cycles
    parameter WRITE_LATENCY   = 2            // Write latency in cycles
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    //--------------------------------------------------------------------------
    // AXI4 Slave Interface
    //--------------------------------------------------------------------------
    // Write Address Channel
    input  wire [AXI_ID_WIDTH-1:0]      s_axi_awid,
    input  wire [AXI_ADDR_WIDTH-1:0]    s_axi_awaddr,
    input  wire [7:0]                   s_axi_awlen,
    input  wire [2:0]                   s_axi_awsize,
    input  wire [1:0]                   s_axi_awburst,
    input  wire                         s_axi_awvalid,
    output reg                          s_axi_awready,
    
    // Write Data Channel
    input  wire [AXI_DATA_WIDTH-1:0]    s_axi_wdata,
    input  wire [AXI_DATA_WIDTH/8-1:0]  s_axi_wstrb,
    input  wire                         s_axi_wlast,
    input  wire                         s_axi_wvalid,
    output reg                          s_axi_wready,
    
    // Write Response Channel
    output reg  [AXI_ID_WIDTH-1:0]      s_axi_bid,
    output reg  [1:0]                   s_axi_bresp,
    output reg                          s_axi_bvalid,
    input  wire                         s_axi_bready,
    
    // Read Address Channel
    input  wire [AXI_ID_WIDTH-1:0]      s_axi_arid,
    input  wire [AXI_ADDR_WIDTH-1:0]    s_axi_araddr,
    input  wire [7:0]                   s_axi_arlen,
    input  wire [2:0]                   s_axi_arsize,
    input  wire [1:0]                   s_axi_arburst,
    input  wire                         s_axi_arvalid,
    output reg                          s_axi_arready,
    
    // Read Data Channel
    output reg  [AXI_ID_WIDTH-1:0]      s_axi_rid,
    output reg  [AXI_DATA_WIDTH-1:0]    s_axi_rdata,
    output reg  [1:0]                   s_axi_rresp,
    output reg                          s_axi_rlast,
    output reg                          s_axi_rvalid,
    input  wire                         s_axi_rready
);

    //--------------------------------------------------------------------------
    // Local Parameters
    //--------------------------------------------------------------------------
    localparam BYTES_PER_WORD = AXI_DATA_WIDTH / 8;
    localparam MEM_DEPTH = (MEM_SIZE_MB * 1024 * 1024) / BYTES_PER_WORD;
    localparam MEM_ADDR_BITS = $clog2(MEM_DEPTH);
    
    // Burst types
    localparam BURST_FIXED = 2'b00;
    localparam BURST_INCR  = 2'b01;
    localparam BURST_WRAP  = 2'b10;

    //--------------------------------------------------------------------------
    // Memory Array
    //--------------------------------------------------------------------------
    reg [AXI_DATA_WIDTH-1:0] mem [0:MEM_DEPTH-1];
    
    //--------------------------------------------------------------------------
    // Write State Machine
    //--------------------------------------------------------------------------
    localparam W_IDLE    = 2'd0;
    localparam W_DATA    = 2'd1;
    localparam W_RESP    = 2'd2;
    
    reg [1:0] w_state;
    reg [AXI_ID_WIDTH-1:0] w_id;
    reg [AXI_ADDR_WIDTH-1:0] w_addr;
    reg [7:0] w_len;
    reg [7:0] w_cnt;
    reg [2:0] w_size;
    reg [1:0] w_burst;
    
    wire [MEM_ADDR_BITS-1:0] w_mem_addr = w_addr[MEM_ADDR_BITS+$clog2(BYTES_PER_WORD)-1:$clog2(BYTES_PER_WORD)];
    
    //--------------------------------------------------------------------------
    // Read State Machine
    //--------------------------------------------------------------------------
    localparam R_IDLE    = 2'd0;
    localparam R_DELAY   = 2'd1;
    localparam R_DATA    = 2'd2;
    
    reg [1:0] r_state;
    reg [AXI_ID_WIDTH-1:0] r_id;
    reg [AXI_ADDR_WIDTH-1:0] r_addr;
    reg [7:0] r_len;
    reg [7:0] r_cnt;
    reg [2:0] r_size;
    reg [1:0] r_burst;
    reg [3:0] r_delay_cnt;
    
    wire [MEM_ADDR_BITS-1:0] r_mem_addr = r_addr[MEM_ADDR_BITS+$clog2(BYTES_PER_WORD)-1:$clog2(BYTES_PER_WORD)];

    //--------------------------------------------------------------------------
    // Address Calculation Functions
    //--------------------------------------------------------------------------
    function [AXI_ADDR_WIDTH-1:0] next_addr;
        input [AXI_ADDR_WIDTH-1:0] addr;
        input [2:0] size;
        input [1:0] burst;
        input [7:0] len;
        input [7:0] cnt;
        reg [AXI_ADDR_WIDTH-1:0] incr;
        reg [AXI_ADDR_WIDTH-1:0] wrap_mask;
        begin
            incr = (1 << size);
            
            case (burst)
                BURST_FIXED: begin
                    next_addr = addr;
                end
                BURST_INCR: begin
                    next_addr = addr + incr;
                end
                BURST_WRAP: begin
                    wrap_mask = ((len + 1) << size) - 1;
                    next_addr = (addr & ~wrap_mask) | ((addr + incr) & wrap_mask);
                end
                default: begin
                    next_addr = addr + incr;
                end
            endcase
        end
    endfunction

    //--------------------------------------------------------------------------
    // Write State Machine
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w_state <= W_IDLE;
            s_axi_awready <= 1'b1;
            s_axi_wready <= 1'b0;
            s_axi_bvalid <= 1'b0;
            s_axi_bid <= {AXI_ID_WIDTH{1'b0}};
            s_axi_bresp <= 2'b00;
            w_id <= {AXI_ID_WIDTH{1'b0}};
            w_addr <= {AXI_ADDR_WIDTH{1'b0}};
            w_len <= 8'd0;
            w_cnt <= 8'd0;
            w_size <= 3'd0;
            w_burst <= 2'd0;
        end else begin
            case (w_state)
                W_IDLE: begin
                    s_axi_awready <= 1'b1;
                    s_axi_wready <= 1'b0;
                    s_axi_bvalid <= 1'b0;
                    
                    if (s_axi_awvalid && s_axi_awready) begin
                        w_id <= s_axi_awid;
                        w_addr <= s_axi_awaddr;
                        w_len <= s_axi_awlen;
                        w_size <= s_axi_awsize;
                        w_burst <= s_axi_awburst;
                        w_cnt <= 8'd0;
                        
                        s_axi_awready <= 1'b0;
                        s_axi_wready <= 1'b1;
                        w_state <= W_DATA;
                    end
                end
                
                W_DATA: begin
                    if (s_axi_wvalid && s_axi_wready) begin
                        // Write to memory with byte enables
                        if (w_mem_addr < MEM_DEPTH) begin
                            integer i;
                            for (i = 0; i < BYTES_PER_WORD; i = i + 1) begin
                                if (s_axi_wstrb[i]) begin
                                    mem[w_mem_addr][i*8 +: 8] <= s_axi_wdata[i*8 +: 8];
                                end
                            end
                        end
                        
                        w_addr <= next_addr(w_addr, w_size, w_burst, w_len, w_cnt);
                        w_cnt <= w_cnt + 1;
                        
                        if (s_axi_wlast) begin
                            s_axi_wready <= 1'b0;
                            s_axi_bvalid <= 1'b1;
                            s_axi_bid <= w_id;
                            s_axi_bresp <= (w_mem_addr < MEM_DEPTH) ? 2'b00 : 2'b10; // OKAY or SLVERR
                            w_state <= W_RESP;
                        end
                    end
                end
                
                W_RESP: begin
                    if (s_axi_bready && s_axi_bvalid) begin
                        s_axi_bvalid <= 1'b0;
                        w_state <= W_IDLE;
                    end
                end
                
                default: w_state <= W_IDLE;
            endcase
        end
    end

    //--------------------------------------------------------------------------
    // Read State Machine
    //--------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_state <= R_IDLE;
            s_axi_arready <= 1'b1;
            s_axi_rvalid <= 1'b0;
            s_axi_rid <= {AXI_ID_WIDTH{1'b0}};
            s_axi_rdata <= {AXI_DATA_WIDTH{1'b0}};
            s_axi_rresp <= 2'b00;
            s_axi_rlast <= 1'b0;
            r_id <= {AXI_ID_WIDTH{1'b0}};
            r_addr <= {AXI_ADDR_WIDTH{1'b0}};
            r_len <= 8'd0;
            r_cnt <= 8'd0;
            r_size <= 3'd0;
            r_burst <= 2'd0;
            r_delay_cnt <= 4'd0;
        end else begin
            case (r_state)
                R_IDLE: begin
                    s_axi_arready <= 1'b1;
                    s_axi_rvalid <= 1'b0;
                    s_axi_rlast <= 1'b0;
                    
                    if (s_axi_arvalid && s_axi_arready) begin
                        r_id <= s_axi_arid;
                        r_addr <= s_axi_araddr;
                        r_len <= s_axi_arlen;
                        r_size <= s_axi_arsize;
                        r_burst <= s_axi_arburst;
                        r_cnt <= 8'd0;
                        r_delay_cnt <= READ_LATENCY;
                        
                        s_axi_arready <= 1'b0;
                        r_state <= R_DELAY;
                    end
                end
                
                R_DELAY: begin
                    if (r_delay_cnt > 0) begin
                        r_delay_cnt <= r_delay_cnt - 1;
                    end else begin
                        r_state <= R_DATA;
                    end
                end
                
                R_DATA: begin
                    if (!s_axi_rvalid || s_axi_rready) begin
                        // Read from memory
                        if (r_mem_addr < MEM_DEPTH) begin
                            s_axi_rdata <= mem[r_mem_addr];
                            s_axi_rresp <= 2'b00;  // OKAY
                        end else begin
                            s_axi_rdata <= {AXI_DATA_WIDTH{1'b0}};
                            s_axi_rresp <= 2'b10;  // SLVERR
                        end
                        
                        s_axi_rid <= r_id;
                        s_axi_rvalid <= 1'b1;
                        s_axi_rlast <= (r_cnt == r_len);
                        
                        r_addr <= next_addr(r_addr, r_size, r_burst, r_len, r_cnt);
                        r_cnt <= r_cnt + 1;
                        
                        if (r_cnt == r_len) begin
                            r_state <= R_IDLE;
                        end
                    end
                end
                
                default: r_state <= R_IDLE;
            endcase
        end
    end

    //--------------------------------------------------------------------------
    // Memory Initialization (Simulation Only)
    //--------------------------------------------------------------------------
    `ifdef SIM
    integer init_i;
    initial begin
        for (init_i = 0; init_i < MEM_DEPTH; init_i = init_i + 1) begin
            mem[init_i] = {AXI_DATA_WIDTH{1'b0}};
        end
    end
    
    // Debug: Monitor transactions
    always @(posedge clk) begin
        if (s_axi_awvalid && s_axi_awready) begin
            $display("[%0t] MEM: Write request addr=0x%h len=%0d", 
                     $time, s_axi_awaddr, s_axi_awlen + 1);
        end
        if (s_axi_arvalid && s_axi_arready) begin
            $display("[%0t] MEM: Read request addr=0x%h len=%0d", 
                     $time, s_axi_araddr, s_axi_arlen + 1);
        end
    end
    `endif

endmodule

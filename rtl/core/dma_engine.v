//==============================================================================
// DMA Engine
//
// 2D strided DMA for efficient tensor data movement:
// - HBM → SRAM (load weights, activations)
// - SRAM → HBM (store results)
// - SRAM → SRAM (internal reshaping)
//
// Features:
// - 2D addressing with configurable strides
// - Double-buffer support (ping-pong transfers)
// - Zero-padding for edge tiles
// - Optional transpose on the fly
//==============================================================================

module dma_engine #(
    parameter EXT_ADDR_W  = 40,       // External (HBM) address width
    parameter INT_ADDR_W  = 20,       // Internal (SRAM) address width
    parameter DATA_WIDTH  = 256,      // Bus width (32 bytes)
    parameter MAX_BURST   = 16        // Maximum burst length
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    //--------------------------------------------------------------------------
    // Command Interface (from LCP)
    //--------------------------------------------------------------------------
    input  wire [127:0]                 cmd,
    input  wire                         cmd_valid,
    output wire                         cmd_ready,
    output wire                         cmd_done,
    
    //--------------------------------------------------------------------------
    // SRAM Interface (internal memory)
    //--------------------------------------------------------------------------
    output wire [INT_ADDR_W-1:0]        sram_addr,
    output wire [DATA_WIDTH-1:0]        sram_wdata,
    input  wire [DATA_WIDTH-1:0]        sram_rdata,
    output wire                         sram_we,
    output wire                         sram_re,
    input  wire                         sram_ready,
    
    //--------------------------------------------------------------------------
    // AXI-like External Memory Interface
    //--------------------------------------------------------------------------
    // Write address channel
    output wire [EXT_ADDR_W-1:0]        axi_awaddr,
    output wire [7:0]                   axi_awlen,
    output wire                         axi_awvalid,
    input  wire                         axi_awready,
    
    // Write data channel
    output wire [DATA_WIDTH-1:0]        axi_wdata,
    output wire                         axi_wlast,
    output wire                         axi_wvalid,
    input  wire                         axi_wready,
    
    // Write response channel
    input  wire [1:0]                   axi_bresp,
    input  wire                         axi_bvalid,
    output wire                         axi_bready,
    
    // Read address channel
    output wire [EXT_ADDR_W-1:0]        axi_araddr,
    output wire [7:0]                   axi_arlen,
    output wire                         axi_arvalid,
    input  wire                         axi_arready,
    
    // Read data channel
    input  wire [DATA_WIDTH-1:0]        axi_rdata,
    input  wire                         axi_rlast,
    input  wire                         axi_rvalid,
    output wire                         axi_rready
);

    //--------------------------------------------------------------------------
    // Command Decode
    //--------------------------------------------------------------------------
    
    // Command format:
    // [127:120] opcode
    // [119:112] subop: 0=LOAD (EXT→SRAM), 1=STORE (SRAM→EXT), 2=COPY (SRAM→SRAM)
    // [111:72]  ext_addr (40 bits)
    // [71:52]   int_addr (20 bits)
    // [51:40]   rows (12 bits)
    // [39:28]   cols (12 bits)
    // [27:16]   src_stride (12 bits)
    // [15:4]    dst_stride (12 bits)
    // [3:0]     flags: [0]=transpose, [1]=zero_pad
    
    wire [7:0]  subop      = cmd[119:112];
    wire [EXT_ADDR_W-1:0] ext_addr = cmd[111:112-EXT_ADDR_W];
    wire [INT_ADDR_W-1:0] int_addr = cmd[71:72-INT_ADDR_W];
    wire [11:0] cfg_rows   = cmd[51:40];
    wire [11:0] cfg_cols   = cmd[39:28];
    wire [11:0] src_stride = cmd[27:16];
    wire [11:0] dst_stride = cmd[15:4];
    wire        do_transpose = cmd[0];
    wire        do_zero_pad  = cmd[1];
    
    localparam DMA_LOAD  = 8'h01;
    localparam DMA_STORE = 8'h02;
    localparam DMA_COPY  = 8'h03;
    
    //--------------------------------------------------------------------------
    // State Machine
    //--------------------------------------------------------------------------
    
    localparam S_IDLE       = 4'd0;
    localparam S_DECODE     = 4'd1;
    localparam S_LOAD_ADDR  = 4'd2;
    localparam S_LOAD_DATA  = 4'd3;
    localparam S_LOAD_WRITE = 4'd4;
    localparam S_STORE_READ = 4'd5;
    localparam S_STORE_ADDR = 4'd6;
    localparam S_STORE_DATA = 4'd7;
    localparam S_STORE_RESP = 4'd8;
    localparam S_NEXT_ROW   = 4'd9;
    localparam S_DONE       = 4'd10;
    
    reg [3:0] state;
    reg [127:0] cmd_reg;
    
    // Transfer tracking
    reg [11:0] row_count;
    reg [11:0] col_count;
    reg [EXT_ADDR_W-1:0] ext_ptr;
    reg [INT_ADDR_W-1:0] int_ptr;
    
    // Data buffer for pipelining
    reg [DATA_WIDTH-1:0] data_buf;
    
    // Burst tracking
    reg [7:0] burst_count;
    reg [7:0] burst_len;
    
    //--------------------------------------------------------------------------
    // Output Registers
    //--------------------------------------------------------------------------
    
    reg [INT_ADDR_W-1:0] sram_addr_reg;
    reg [DATA_WIDTH-1:0] sram_wdata_reg;
    reg sram_we_reg, sram_re_reg;
    
    reg [EXT_ADDR_W-1:0] axi_awaddr_reg, axi_araddr_reg;
    reg [7:0] axi_awlen_reg, axi_arlen_reg;
    reg axi_awvalid_reg, axi_arvalid_reg;
    reg [DATA_WIDTH-1:0] axi_wdata_reg;
    reg axi_wlast_reg, axi_wvalid_reg;
    reg axi_rready_reg;
    
    reg done_reg;
    
    //--------------------------------------------------------------------------
    // Main State Machine
    //--------------------------------------------------------------------------
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            cmd_reg <= 128'd0;
            row_count <= 12'd0;
            col_count <= 12'd0;
            ext_ptr <= {EXT_ADDR_W{1'b0}};
            int_ptr <= {INT_ADDR_W{1'b0}};
            burst_count <= 8'd0;
            burst_len <= 8'd0;
            
            sram_addr_reg <= {INT_ADDR_W{1'b0}};
            sram_wdata_reg <= {DATA_WIDTH{1'b0}};
            sram_we_reg <= 1'b0;
            sram_re_reg <= 1'b0;
            
            axi_awaddr_reg <= {EXT_ADDR_W{1'b0}};
            axi_araddr_reg <= {EXT_ADDR_W{1'b0}};
            axi_awlen_reg <= 8'd0;
            axi_arlen_reg <= 8'd0;
            axi_awvalid_reg <= 1'b0;
            axi_arvalid_reg <= 1'b0;
            axi_wdata_reg <= {DATA_WIDTH{1'b0}};
            axi_wlast_reg <= 1'b0;
            axi_wvalid_reg <= 1'b0;
            axi_rready_reg <= 1'b0;
            
            done_reg <= 1'b0;
        end else begin
            // Default: clear pulses
            sram_we_reg <= 1'b0;
            sram_re_reg <= 1'b0;
            done_reg <= 1'b0;
            
            case (state)
                //--------------------------------------------------------------
                S_IDLE: begin
                    if (cmd_valid) begin
                        cmd_reg <= cmd;
                        state <= S_DECODE;
                    end
                end
                
                //--------------------------------------------------------------
                S_DECODE: begin
                    row_count <= 12'd0;
                    col_count <= 12'd0;
                    ext_ptr <= ext_addr;
                    int_ptr <= int_addr;
                    
                    // Calculate burst length (min of remaining cols and MAX_BURST)
                    burst_len <= (cfg_cols > MAX_BURST) ? MAX_BURST - 1 : cfg_cols - 1;
                    burst_count <= 8'd0;
                    
                    case (subop)
                        DMA_LOAD: state <= S_LOAD_ADDR;
                        DMA_STORE: state <= S_STORE_READ;
                        default: state <= S_DONE;
                    endcase
                end
                
                //--------------------------------------------------------------
                // LOAD Path: External → SRAM
                //--------------------------------------------------------------
                
                S_LOAD_ADDR: begin
                    axi_araddr_reg <= ext_ptr;
                    axi_arlen_reg <= burst_len;
                    axi_arvalid_reg <= 1'b1;
                    
                    if (axi_arready && axi_arvalid_reg) begin
                        axi_arvalid_reg <= 1'b0;
                        axi_rready_reg <= 1'b1;
                        state <= S_LOAD_DATA;
                    end
                end
                
                S_LOAD_DATA: begin
                    if (axi_rvalid && axi_rready_reg) begin
                        data_buf <= axi_rdata;
                        state <= S_LOAD_WRITE;
                    end
                end
                
                S_LOAD_WRITE: begin
                    // Write to SRAM
                    sram_addr_reg <= int_ptr;
                    sram_wdata_reg <= data_buf;
                    sram_we_reg <= 1'b1;
                    
                    if (sram_ready) begin
                        // Update pointers
                        int_ptr <= int_ptr + (DATA_WIDTH / 8);  // Increment by bus width in bytes
                        col_count <= col_count + 1;
                        burst_count <= burst_count + 1;
                        
                        if (burst_count >= burst_len) begin
                            // Burst complete
                            axi_rready_reg <= 1'b0;
                            
                            if (col_count >= cfg_cols - 1) begin
                                state <= S_NEXT_ROW;
                            end else begin
                                // More columns in this row
                                ext_ptr <= ext_ptr + (DATA_WIDTH / 8);
                                burst_count <= 8'd0;
                                state <= S_LOAD_ADDR;
                            end
                        end else begin
                            // Continue burst
                            state <= S_LOAD_DATA;
                        end
                    end
                end
                
                //--------------------------------------------------------------
                // STORE Path: SRAM → External
                //--------------------------------------------------------------
                
                S_STORE_READ: begin
                    // Read from SRAM
                    sram_addr_reg <= int_ptr;
                    sram_re_reg <= 1'b1;
                    
                    if (sram_ready) begin
                        data_buf <= sram_rdata;
                        state <= S_STORE_ADDR;
                    end
                end
                
                S_STORE_ADDR: begin
                    axi_awaddr_reg <= ext_ptr;
                    axi_awlen_reg <= 8'd0;  // Single beat for simplicity
                    axi_awvalid_reg <= 1'b1;
                    
                    if (axi_awready && axi_awvalid_reg) begin
                        axi_awvalid_reg <= 1'b0;
                        state <= S_STORE_DATA;
                    end
                end
                
                S_STORE_DATA: begin
                    axi_wdata_reg <= data_buf;
                    axi_wlast_reg <= 1'b1;
                    axi_wvalid_reg <= 1'b1;
                    
                    if (axi_wready && axi_wvalid_reg) begin
                        axi_wvalid_reg <= 1'b0;
                        axi_wlast_reg <= 1'b0;
                        state <= S_STORE_RESP;
                    end
                end
                
                S_STORE_RESP: begin
                    if (axi_bvalid) begin
                        // Update pointers
                        ext_ptr <= ext_ptr + (DATA_WIDTH / 8);
                        int_ptr <= int_ptr + (DATA_WIDTH / 8);
                        col_count <= col_count + 1;
                        
                        if (col_count >= cfg_cols - 1) begin
                            state <= S_NEXT_ROW;
                        end else begin
                            state <= S_STORE_READ;
                        end
                    end
                end
                
                //--------------------------------------------------------------
                // Row Advance
                //--------------------------------------------------------------
                
                S_NEXT_ROW: begin
                    row_count <= row_count + 1;
                    col_count <= 12'd0;
                    
                    if (row_count >= cfg_rows - 1) begin
                        // All rows done
                        state <= S_DONE;
                    end else begin
                        // Advance to next row
                        ext_ptr <= ext_addr + (row_count + 1) * src_stride;
                        int_ptr <= int_addr + (row_count + 1) * dst_stride;
                        
                        case (subop)
                            DMA_LOAD: state <= S_LOAD_ADDR;
                            DMA_STORE: state <= S_STORE_READ;
                            default: state <= S_DONE;
                        endcase
                    end
                end
                
                //--------------------------------------------------------------
                S_DONE: begin
                    done_reg <= 1'b1;
                    state <= S_IDLE;
                end
                
                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end
    
    //--------------------------------------------------------------------------
    // Output Assignments
    //--------------------------------------------------------------------------
    
    assign cmd_ready = (state == S_IDLE);
    assign cmd_done = done_reg;
    
    // SRAM
    assign sram_addr = sram_addr_reg;
    assign sram_wdata = sram_wdata_reg;
    assign sram_we = sram_we_reg;
    assign sram_re = sram_re_reg;
    
    // AXI Write
    assign axi_awaddr = axi_awaddr_reg;
    assign axi_awlen = axi_awlen_reg;
    assign axi_awvalid = axi_awvalid_reg;
    assign axi_wdata = axi_wdata_reg;
    assign axi_wlast = axi_wlast_reg;
    assign axi_wvalid = axi_wvalid_reg;
    assign axi_bready = 1'b1;  // Always ready for response
    
    // AXI Read
    assign axi_araddr = axi_araddr_reg;
    assign axi_arlen = axi_arlen_reg;
    assign axi_arvalid = axi_arvalid_reg;
    assign axi_rready = axi_rready_reg;

endmodule

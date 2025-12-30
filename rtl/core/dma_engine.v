//==============================================================================
// DMA Engine (Fixed)
//
// 2D strided DMA for efficient tensor data movement:
// - HBM → SRAM (load weights, activations)
// - SRAM → HBM (store results)
//
// Features:
// - 2D addressing with configurable strides
// - Burst support for efficient AXI transfers
// - Proper handling of SRAM read latency
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
    // Note: SRAM has 1-cycle read latency
    //--------------------------------------------------------------------------
    output wire [INT_ADDR_W-1:0]        sram_addr,
    output wire [DATA_WIDTH-1:0]        sram_wdata,
    input  wire [DATA_WIDTH-1:0]        sram_rdata,
    output wire                         sram_we,
    output wire                         sram_re,
    input  wire                         sram_ready,
    
    //--------------------------------------------------------------------------
    // AXI4 External Memory Interface
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
    // [127:120] opcode (0x03 = DMA)
    // [119:112] subop: 0x01=LOAD (EXT→SRAM), 0x02=STORE (SRAM→EXT)
    // [111:72]  ext_addr (40 bits)
    // [71:52]   int_addr (20 bits)
    // [51:40]   rows (12 bits)
    // [39:28]   cols (12 bits) - number of DATA_WIDTH words per row
    // [27:16]   ext_stride (12 bits) - row stride for external memory (bytes)
    // [15:4]    int_stride (12 bits) - row stride for SRAM (bytes)
    // [3:0]     flags: [0]=transpose, [1]=zero_pad (future use)
    
    wire [7:0]  subop      = cmd[119:112];
    wire [EXT_ADDR_W-1:0] ext_addr = cmd[111:112-EXT_ADDR_W];
    wire [INT_ADDR_W-1:0] int_addr = cmd[71:72-INT_ADDR_W];
    wire [11:0] cfg_rows   = cmd[51:40];
    wire [11:0] cfg_cols   = cmd[39:28];
    wire [11:0] ext_stride = cmd[27:16];
    wire [11:0] int_stride = cmd[15:4];
    
    localparam DMA_LOAD  = 8'h01;
    localparam DMA_STORE = 8'h02;
    
    localparam BYTES_PER_WORD = DATA_WIDTH / 8;
    
    //--------------------------------------------------------------------------
    // State Machine
    //--------------------------------------------------------------------------
    
    localparam S_IDLE       = 4'd0;
    localparam S_DECODE     = 4'd1;
    // LOAD states
    localparam S_LOAD_ADDR  = 4'd2;
    localparam S_LOAD_DATA  = 4'd3;
    localparam S_LOAD_WRITE = 4'd4;
    // STORE states
    localparam S_STORE_REQ  = 4'd5;   // Assert SRAM read request
    localparam S_STORE_WAIT = 4'd6;   // Wait cycle 1 for SRAM latency
    localparam S_STORE_CAP  = 4'd13;  // Capture SRAM read data
    localparam S_STORE_ADDR = 4'd7;   // Send AXI write address
    localparam S_STORE_DATA = 4'd8;   // Send AXI write data
    localparam S_STORE_RESP = 4'd9;   // Wait for AXI response
    // Common states
    localparam S_NEXT_COL   = 4'd10;
    localparam S_NEXT_ROW   = 4'd11;
    localparam S_DONE       = 4'd12;
    
    reg [3:0] state;
    reg [7:0] op_type;  // Latched subop
    
    // Transfer tracking
    reg [11:0] row_count;
    reg [11:0] col_count;
    reg [11:0] rows_cfg, cols_cfg;
    reg [11:0] ext_stride_cfg, int_stride_cfg;
    reg [EXT_ADDR_W-1:0] ext_base, ext_ptr;
    reg [INT_ADDR_W-1:0] int_base, int_ptr;
    
    // Data buffer
    reg [DATA_WIDTH-1:0] data_buf;
    
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
            op_type <= 8'd0;
            row_count <= 12'd0;
            col_count <= 12'd0;
            rows_cfg <= 12'd0;
            cols_cfg <= 12'd0;
            ext_stride_cfg <= 12'd0;
            int_stride_cfg <= 12'd0;
            ext_base <= {EXT_ADDR_W{1'b0}};
            ext_ptr <= {EXT_ADDR_W{1'b0}};
            int_base <= {INT_ADDR_W{1'b0}};
            int_ptr <= {INT_ADDR_W{1'b0}};
            data_buf <= {DATA_WIDTH{1'b0}};
            
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
            // Default: clear single-cycle signals
            sram_we_reg <= 1'b0;
            sram_re_reg <= 1'b0;
            done_reg <= 1'b0;
            
            case (state)
                //--------------------------------------------------------------
                S_IDLE: begin
                    if (cmd_valid) begin
                        // Latch command parameters
                        op_type <= subop;
                        ext_base <= ext_addr;
                        int_base <= int_addr;
                        rows_cfg <= cfg_rows;
                        cols_cfg <= cfg_cols;
                        ext_stride_cfg <= ext_stride;
                        int_stride_cfg <= int_stride;
                        state <= S_DECODE;
                    end
                end
                
                //--------------------------------------------------------------
                S_DECODE: begin
                    // Initialize pointers
                    ext_ptr <= ext_base;
                    int_ptr <= int_base;
                    row_count <= 12'd0;
                    col_count <= 12'd0;
                    
                    case (op_type)
                        DMA_LOAD:  state <= S_LOAD_ADDR;
                        DMA_STORE: state <= S_STORE_REQ;
                        default:   state <= S_DONE;
                    endcase
                end
                
                //==============================================================
                // LOAD Path: External Memory → SRAM
                //==============================================================
                
                S_LOAD_ADDR: begin
                    // Issue AXI read request (single beat for simplicity)
                    axi_araddr_reg <= ext_ptr;
                    axi_arlen_reg <= 8'd0;  // Single beat
                    axi_arvalid_reg <= 1'b1;
                    
                    if (axi_arready && axi_arvalid_reg) begin
                        axi_arvalid_reg <= 1'b0;
                        axi_rready_reg <= 1'b1;
                        state <= S_LOAD_DATA;
                    end
                end
                
                S_LOAD_DATA: begin
                    // Wait for read data
                    if (axi_rvalid && axi_rready_reg) begin
                        data_buf <= axi_rdata;
                        axi_rready_reg <= 1'b0;
                        state <= S_LOAD_WRITE;
                    end
                end
                
                S_LOAD_WRITE: begin
                    // Write to SRAM
                    sram_addr_reg <= int_ptr;
                    sram_wdata_reg <= data_buf;
                    sram_we_reg <= 1'b1;
                    
                    if (sram_ready) begin
                        state <= S_NEXT_COL;
                    end
                end
                
                //==============================================================
                // STORE Path: SRAM → External Memory
                // Timing: Need 2 cycles after asserting read to capture data
                // Cycle N:   Assert address & re
                // Cycle N+1: SRAM registers read internally
                // Cycle N+2: sram_rdata valid, capture it
                //==============================================================
                
                S_STORE_REQ: begin
                    // Assert SRAM read request
                    sram_addr_reg <= int_ptr;
                    sram_re_reg <= 1'b1;
                    
                    if (sram_ready) begin
                        // SRAM accepted request, wait for latency
                        state <= S_STORE_WAIT;
                    end
                end
                
                S_STORE_WAIT: begin
                    // Wait cycle 1: SRAM is processing read
                    state <= S_STORE_CAP;
                end
                
                S_STORE_CAP: begin
                    // Cycle 2: sram_rdata is now valid, capture it
                    data_buf <= sram_rdata;
                    state <= S_STORE_ADDR;
                end
                
                S_STORE_ADDR: begin
                    // Issue AXI write address
                    axi_awaddr_reg <= ext_ptr;
                    axi_awlen_reg <= 8'd0;  // Single beat
                    axi_awvalid_reg <= 1'b1;
                    
                    if (axi_awready && axi_awvalid_reg) begin
                        axi_awvalid_reg <= 1'b0;
                        state <= S_STORE_DATA;
                    end
                end
                
                S_STORE_DATA: begin
                    // Send write data
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
                    // Wait for write response
                    if (axi_bvalid) begin
                        state <= S_NEXT_COL;
                    end
                end
                
                //==============================================================
                // Column/Row Advancement (shared by LOAD and STORE)
                //==============================================================
                
                S_NEXT_COL: begin
                    col_count <= col_count + 1;
                    ext_ptr <= ext_ptr + BYTES_PER_WORD;
                    int_ptr <= int_ptr + BYTES_PER_WORD;
                    
                    if (col_count >= cols_cfg - 1) begin
                        // End of row
                        state <= S_NEXT_ROW;
                    end else begin
                        // More columns
                        case (op_type)
                            DMA_LOAD:  state <= S_LOAD_ADDR;
                            DMA_STORE: state <= S_STORE_REQ;
                            default:   state <= S_DONE;
                        endcase
                    end
                end
                
                S_NEXT_ROW: begin
                    row_count <= row_count + 1;
                    col_count <= 12'd0;
                    
                    if (row_count >= rows_cfg - 1) begin
                        // All done
                        state <= S_DONE;
                    end else begin
                        // Advance to next row using strides
                        ext_ptr <= ext_base + (row_count + 1) * ext_stride_cfg;
                        int_ptr <= int_base + (row_count + 1) * int_stride_cfg;
                        
                        case (op_type)
                            DMA_LOAD:  state <= S_LOAD_ADDR;
                            DMA_STORE: state <= S_STORE_REQ;
                            default:   state <= S_DONE;
                        endcase
                    end
                end
                
                //--------------------------------------------------------------
                S_DONE: begin
                    done_reg <= 1'b1;
                    state <= S_IDLE;
                end
                
                default: state <= S_IDLE;
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

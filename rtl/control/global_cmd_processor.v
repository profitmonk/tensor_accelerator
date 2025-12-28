//==============================================================================
// Global Command Processor (GCP)
//
// Simplified GCP for FPGA proof-of-concept:
// - AXI-Lite slave for host control
// - Work descriptor registers per TPC
// - Global synchronization controller
// - Status monitoring
//
// For production: Replace with RISC-V core + firmware
//==============================================================================

module global_cmd_processor #(
    parameter NUM_TPCS     = 4,           // Number of TPCs to control
    parameter SRAM_ADDR_W  = 20,          // SRAM address width
    parameter AXI_ADDR_W   = 12,          // AXI address width
    parameter AXI_DATA_W   = 32           // AXI data width
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    //--------------------------------------------------------------------------
    // AXI-Lite Slave Interface (Host Control)
    //--------------------------------------------------------------------------
    // Write address
    input  wire [AXI_ADDR_W-1:0]        s_axi_awaddr,
    input  wire                         s_axi_awvalid,
    output wire                         s_axi_awready,
    
    // Write data
    input  wire [AXI_DATA_W-1:0]        s_axi_wdata,
    input  wire [AXI_DATA_W/8-1:0]      s_axi_wstrb,
    input  wire                         s_axi_wvalid,
    output wire                         s_axi_wready,
    
    // Write response
    output wire [1:0]                   s_axi_bresp,
    output wire                         s_axi_bvalid,
    input  wire                         s_axi_bready,
    
    // Read address
    input  wire [AXI_ADDR_W-1:0]        s_axi_araddr,
    input  wire                         s_axi_arvalid,
    output wire                         s_axi_arready,
    
    // Read data
    output wire [AXI_DATA_W-1:0]        s_axi_rdata,
    output wire [1:0]                   s_axi_rresp,
    output wire                         s_axi_rvalid,
    input  wire                         s_axi_rready,
    
    //--------------------------------------------------------------------------
    // TPC Control Interface
    //--------------------------------------------------------------------------
    output wire [NUM_TPCS-1:0]          tpc_start,
    output wire [SRAM_ADDR_W-1:0]       tpc_start_pc [0:NUM_TPCS-1],
    input  wire [NUM_TPCS-1:0]          tpc_busy,
    input  wire [NUM_TPCS-1:0]          tpc_done,
    input  wire [NUM_TPCS-1:0]          tpc_error,
    
    //--------------------------------------------------------------------------
    // Global Synchronization
    //--------------------------------------------------------------------------
    output wire                         global_sync_out,
    input  wire [NUM_TPCS-1:0]          sync_request,
    output wire [NUM_TPCS-1:0]          sync_grant,
    
    //--------------------------------------------------------------------------
    // Interrupt Output
    //--------------------------------------------------------------------------
    output wire                         irq
);

    //--------------------------------------------------------------------------
    // Register Map (32-bit aligned)
    //--------------------------------------------------------------------------
    // 0x000: Control Register
    //        [0]     = Global Start (write 1 to start all enabled TPCs)
    //        [1]     = Global Reset (write 1 to reset)
    //        [15:8]  = TPC Enable Mask
    // 0x004: Status Register (read-only)
    //        [3:0]   = TPC Busy (for 4 TPCs)
    //        [7:4]   = TPC Done
    //        [11:8]  = TPC Error
    //        [16]    = All Done
    // 0x008: Interrupt Enable Register
    //        [0]     = Enable interrupt on all TPCs done
    // 0x00C: Interrupt Status Register (write 1 to clear)
    //        [0]     = All TPCs done interrupt
    //
    // 0x100 + n*0x10: TPC n Start PC (20 bits)
    // 0x104 + n*0x10: TPC n Status (read-only)
    // 0x108 + n*0x10: Reserved
    // 0x10C + n*0x10: Reserved
    //--------------------------------------------------------------------------
    
    localparam ADDR_CTRL       = 12'h000;
    localparam ADDR_STATUS     = 12'h004;
    localparam ADDR_IRQ_EN     = 12'h008;
    localparam ADDR_IRQ_STATUS = 12'h00C;
    localparam ADDR_TPC_BASE   = 12'h100;
    localparam ADDR_TPC_STRIDE = 12'h010;
    
    //--------------------------------------------------------------------------
    // Registers
    //--------------------------------------------------------------------------
    
    reg [7:0]  tpc_enable;
    reg        global_start_pulse;
    reg        irq_enable;
    reg        irq_status;
    reg [SRAM_ADDR_W-1:0] tpc_pc [0:NUM_TPCS-1];
    
    // AXI state machine
    localparam AXI_IDLE       = 3'd0;
    localparam AXI_WRITE_RESP = 3'd1;
    localparam AXI_READ_DATA  = 3'd2;
    
    reg [2:0] axi_state;
    reg [AXI_ADDR_W-1:0] axi_addr_reg;
    reg [AXI_DATA_W-1:0] axi_rdata_reg;
    
    //--------------------------------------------------------------------------
    // Status Generation
    //--------------------------------------------------------------------------
    
    wire all_enabled_done;
    reg [NUM_TPCS-1:0] tpc_done_latch;
    
    // Latch done signals
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tpc_done_latch <= {NUM_TPCS{1'b0}};
        end else begin
            if (global_start_pulse) begin
                tpc_done_latch <= {NUM_TPCS{1'b0}};
            end else begin
                tpc_done_latch <= tpc_done_latch | tpc_done;
            end
        end
    end
    
    // Check if all enabled TPCs are done
    assign all_enabled_done = &((tpc_done_latch | ~tpc_enable[NUM_TPCS-1:0]) | 
                                ~{NUM_TPCS{1'b1}});
    
    //--------------------------------------------------------------------------
    // AXI-Lite State Machine
    //--------------------------------------------------------------------------
    
    reg s_axi_awready_reg, s_axi_wready_reg;
    reg s_axi_bvalid_reg;
    reg s_axi_arready_reg;
    reg s_axi_rvalid_reg;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            axi_state <= AXI_IDLE;
            axi_addr_reg <= {AXI_ADDR_W{1'b0}};
            axi_rdata_reg <= {AXI_DATA_W{1'b0}};
            s_axi_awready_reg <= 1'b0;
            s_axi_wready_reg <= 1'b0;
            s_axi_bvalid_reg <= 1'b0;
            s_axi_arready_reg <= 1'b0;
            s_axi_rvalid_reg <= 1'b0;
            
            tpc_enable <= 8'hFF;
            global_start_pulse <= 1'b0;
            irq_enable <= 1'b0;
            irq_status <= 1'b0;
            
            for (integer i = 0; i < NUM_TPCS; i = i + 1) begin
                tpc_pc[i] <= {SRAM_ADDR_W{1'b0}};
            end
        end else begin
            global_start_pulse <= 1'b0;
            s_axi_awready_reg <= 1'b0;
            s_axi_wready_reg <= 1'b0;
            s_axi_arready_reg <= 1'b0;
            
            // Update interrupt status
            if (all_enabled_done && irq_enable) begin
                irq_status <= 1'b1;
            end
            
            case (axi_state)
                AXI_IDLE: begin
                    // Prioritize writes over reads
                    if (s_axi_awvalid && s_axi_wvalid) begin
                        s_axi_awready_reg <= 1'b1;
                        s_axi_wready_reg <= 1'b1;
                        axi_addr_reg <= s_axi_awaddr;
                        
                        // Process write
                        case (s_axi_awaddr)
                            ADDR_CTRL: begin
                                if (s_axi_wdata[0]) global_start_pulse <= 1'b1;
                                tpc_enable <= s_axi_wdata[15:8];
                            end
                            
                            ADDR_IRQ_EN: begin
                                irq_enable <= s_axi_wdata[0];
                            end
                            
                            ADDR_IRQ_STATUS: begin
                                // Write 1 to clear
                                if (s_axi_wdata[0]) irq_status <= 1'b0;
                            end
                            
                            default: begin
                                // TPC registers
                                if (s_axi_awaddr >= ADDR_TPC_BASE) begin
                                    // Calculate TPC index
                                    // tpc_idx = (addr - TPC_BASE) / TPC_STRIDE
                                    // offset = (addr - TPC_BASE) % TPC_STRIDE
                                    for (integer i = 0; i < NUM_TPCS; i = i + 1) begin
                                        if (s_axi_awaddr == ADDR_TPC_BASE + i*ADDR_TPC_STRIDE) begin
                                            tpc_pc[i] <= s_axi_wdata[SRAM_ADDR_W-1:0];
                                        end
                                    end
                                end
                            end
                        endcase
                        
                        axi_state <= AXI_WRITE_RESP;
                    end else if (s_axi_arvalid) begin
                        s_axi_arready_reg <= 1'b1;
                        axi_addr_reg <= s_axi_araddr;
                        
                        // Process read
                        case (s_axi_araddr)
                            ADDR_CTRL: begin
                                axi_rdata_reg <= {16'b0, tpc_enable, 8'b0};
                            end
                            
                            ADDR_STATUS: begin
                                axi_rdata_reg <= {15'b0, all_enabled_done, 
                                                  {(8-NUM_TPCS){1'b0}}, tpc_error,
                                                  {(8-NUM_TPCS){1'b0}}, tpc_done_latch,
                                                  {(8-NUM_TPCS){1'b0}}, tpc_busy};
                            end
                            
                            ADDR_IRQ_EN: begin
                                axi_rdata_reg <= {31'b0, irq_enable};
                            end
                            
                            ADDR_IRQ_STATUS: begin
                                axi_rdata_reg <= {31'b0, irq_status};
                            end
                            
                            default: begin
                                axi_rdata_reg <= 32'h0;
                                // TPC registers
                                for (integer i = 0; i < NUM_TPCS; i = i + 1) begin
                                    if (s_axi_araddr == ADDR_TPC_BASE + i*ADDR_TPC_STRIDE) begin
                                        axi_rdata_reg <= {{(32-SRAM_ADDR_W){1'b0}}, tpc_pc[i]};
                                    end
                                    if (s_axi_araddr == ADDR_TPC_BASE + i*ADDR_TPC_STRIDE + 4) begin
                                        axi_rdata_reg <= {29'b0, tpc_error[i], tpc_done_latch[i], tpc_busy[i]};
                                    end
                                end
                            end
                        endcase
                        
                        axi_state <= AXI_READ_DATA;
                    end
                end
                
                AXI_WRITE_RESP: begin
                    s_axi_bvalid_reg <= 1'b1;
                    if (s_axi_bready && s_axi_bvalid_reg) begin
                        s_axi_bvalid_reg <= 1'b0;
                        axi_state <= AXI_IDLE;
                    end
                end
                
                AXI_READ_DATA: begin
                    s_axi_rvalid_reg <= 1'b1;
                    if (s_axi_rready && s_axi_rvalid_reg) begin
                        s_axi_rvalid_reg <= 1'b0;
                        axi_state <= AXI_IDLE;
                    end
                end
                
                default: begin
                    axi_state <= AXI_IDLE;
                end
            endcase
        end
    end
    
    //--------------------------------------------------------------------------
    // Global Synchronization Logic
    //--------------------------------------------------------------------------
    
    // Simple barrier: wait for all TPCs to request, then grant all
    wire all_sync_requested = &(sync_request | ~tpc_enable[NUM_TPCS-1:0] | 
                                 ~{NUM_TPCS{1'b1}});
    reg barrier_active;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            barrier_active <= 1'b0;
        end else begin
            if (all_sync_requested && !barrier_active) begin
                barrier_active <= 1'b1;
            end else if (barrier_active) begin
                barrier_active <= 1'b0;  // Grant for one cycle
            end
        end
    end
    
    //--------------------------------------------------------------------------
    // Output Assignments
    //--------------------------------------------------------------------------
    
    // AXI-Lite
    assign s_axi_awready = s_axi_awready_reg;
    assign s_axi_wready  = s_axi_wready_reg;
    assign s_axi_bresp   = 2'b00;  // OKAY
    assign s_axi_bvalid  = s_axi_bvalid_reg;
    assign s_axi_arready = s_axi_arready_reg;
    assign s_axi_rdata   = axi_rdata_reg;
    assign s_axi_rresp   = 2'b00;  // OKAY
    assign s_axi_rvalid  = s_axi_rvalid_reg;
    
    // TPC control
    genvar t;
    generate
        for (t = 0; t < NUM_TPCS; t = t + 1) begin : tpc_ctrl_gen
            assign tpc_start[t] = global_start_pulse && tpc_enable[t];
            assign tpc_start_pc[t] = tpc_pc[t];
            assign sync_grant[t] = barrier_active && sync_request[t];
        end
    endgenerate
    
    assign global_sync_out = barrier_active;
    assign irq = irq_status;

endmodule

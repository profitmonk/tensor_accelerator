//==============================================================================
// Tensor Processing Cluster (TPC)
//
// Complete processing cluster containing:
// - Local Command Processor (LCP)
// - Matrix Multiply Unit (MXU) - Systolic Array
// - Vector Processing Unit (VPU)
// - DMA Engine
// - Banked SRAM Subsystem
// - NoC Interface
//
// This is the replicated unit for scaling the accelerator.
//==============================================================================

module tensor_processing_cluster #(
    // Array parameters
    parameter ARRAY_SIZE   = 16,          // Systolic array dimension
    parameter DATA_WIDTH   = 8,           // Operand width (INT8)
    parameter ACC_WIDTH    = 32,          // Accumulator width
    
    // Vector unit parameters  
    parameter VPU_LANES    = 64,          // SIMD width
    parameter VPU_DATA_W   = 16,          // VPU operand width (BF16)
    
    // Memory parameters
    parameter SRAM_BANKS   = 16,          // Number of SRAM banks
    parameter SRAM_DEPTH   = 4096,        // Words per bank
    parameter SRAM_WIDTH   = 256,         // Bits per word
    parameter SRAM_ADDR_W  = 20,          // SRAM address width
    
    // External interface
    parameter EXT_ADDR_W   = 40,          // External memory address width
    parameter EXT_DATA_W   = 256,         // External memory data width
    
    // TPC identification
    parameter TPC_ID       = 0            // Unique TPC identifier
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    //--------------------------------------------------------------------------
    // Control Interface (from GCP via NoC)
    //--------------------------------------------------------------------------
    input  wire                         tpc_start,
    input  wire [SRAM_ADDR_W-1:0]       tpc_start_pc,
    output wire                         tpc_busy,
    output wire                         tpc_done,
    output wire                         tpc_error,
    
    //--------------------------------------------------------------------------
    // Global Synchronization
    //--------------------------------------------------------------------------
    input  wire                         global_sync_in,
    output wire                         sync_request,
    input  wire                         sync_grant,
    
    //--------------------------------------------------------------------------
    // NoC Interface (for instruction/data loading)
    //--------------------------------------------------------------------------
    // Receive port
    input  wire [SRAM_WIDTH-1:0]        noc_rx_data,
    input  wire [SRAM_ADDR_W-1:0]       noc_rx_addr,
    input  wire                         noc_rx_valid,
    output wire                         noc_rx_ready,
    input  wire                         noc_rx_is_instr,   // 1=instruction, 0=data
    
    // Transmit port
    output wire [SRAM_WIDTH-1:0]        noc_tx_data,
    output wire [SRAM_ADDR_W-1:0]       noc_tx_addr,
    output wire                         noc_tx_valid,
    input  wire                         noc_tx_ready,
    
    //--------------------------------------------------------------------------
    // External Memory Interface (AXI-like)
    //--------------------------------------------------------------------------
    // Write address
    output wire [EXT_ADDR_W-1:0]        axi_awaddr,
    output wire [7:0]                   axi_awlen,
    output wire                         axi_awvalid,
    input  wire                         axi_awready,
    
    // Write data
    output wire [EXT_DATA_W-1:0]        axi_wdata,
    output wire                         axi_wlast,
    output wire                         axi_wvalid,
    input  wire                         axi_wready,
    
    // Write response
    input  wire [1:0]                   axi_bresp,
    input  wire                         axi_bvalid,
    output wire                         axi_bready,
    
    // Read address
    output wire [EXT_ADDR_W-1:0]        axi_araddr,
    output wire [7:0]                   axi_arlen,
    output wire                         axi_arvalid,
    input  wire                         axi_arready,
    
    // Read data
    input  wire [EXT_DATA_W-1:0]        axi_rdata,
    input  wire                         axi_rlast,
    input  wire                         axi_rvalid,
    output wire                         axi_rready
);

    //==========================================================================
    // Internal Signals
    //==========================================================================
    
    // LCP <-> Instruction Memory
    wire [SRAM_ADDR_W-1:0]   lcp_imem_addr;
    wire [127:0]             lcp_imem_data;
    wire                     lcp_imem_re;
    wire                     lcp_imem_valid;
    
    // LCP <-> MXU
    wire [127:0]             lcp_mxu_cmd;
    wire                     lcp_mxu_valid;
    wire                     mxu_lcp_ready;
    wire                     mxu_lcp_done;
    
    // LCP <-> VPU
    wire [127:0]             lcp_vpu_cmd;
    wire                     lcp_vpu_valid;
    wire                     vpu_lcp_ready;
    wire                     vpu_lcp_done;
    
    // LCP <-> DMA
    wire [127:0]             lcp_dma_cmd;
    wire                     lcp_dma_valid;
    wire                     dma_lcp_ready;
    wire                     dma_lcp_done;
    
    // MXU <-> SRAM
    wire [SRAM_ADDR_W-1:0]   mxu_w_addr, mxu_a_addr, mxu_o_addr;
    wire [SRAM_WIDTH-1:0]    mxu_w_rdata, mxu_a_rdata, mxu_o_wdata;
    wire                     mxu_w_re, mxu_a_re, mxu_o_we;
    wire                     mxu_w_ready, mxu_a_ready, mxu_o_ready;
    
    // VPU <-> SRAM
    wire [SRAM_ADDR_W-1:0]   vpu_sram_addr;
    wire [SRAM_WIDTH-1:0]    vpu_sram_wdata, vpu_sram_rdata;
    wire                     vpu_sram_we, vpu_sram_re;
    wire                     vpu_sram_ready;
    
    // DMA <-> SRAM
    wire [SRAM_ADDR_W-1:0]   dma_sram_addr;
    wire [SRAM_WIDTH-1:0]    dma_sram_wdata, dma_sram_rdata;
    wire                     dma_sram_we, dma_sram_re;
    wire                     dma_sram_ready;
    
    //==========================================================================
    // Instruction Memory (dedicated SRAM bank for instructions)
    //==========================================================================
    
    // Instruction memory - 4K x 128-bit instructions
    (* ram_style = "block" *)
    reg [127:0] instr_mem [0:4095];
    reg [127:0] instr_rdata_reg;
    reg         instr_valid_reg;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            instr_rdata_reg <= 128'd0;
            instr_valid_reg <= 1'b0;
        end else begin
            instr_valid_reg <= lcp_imem_re;
            if (lcp_imem_re) begin
                instr_rdata_reg <= instr_mem[lcp_imem_addr[11:0]];
            end
        end
    end
    
    assign lcp_imem_data = instr_rdata_reg;
    assign lcp_imem_valid = instr_valid_reg;
    
    // Instruction loading from NoC
    always @(posedge clk) begin
        if (noc_rx_valid && noc_rx_ready && noc_rx_is_instr) begin
            instr_mem[noc_rx_addr[11:0]] <= noc_rx_data[127:0];
        end
    end
    
    //==========================================================================
    // Local Command Processor (LCP)
    //==========================================================================
    
    local_cmd_processor #(
        .INSTR_WIDTH  (128),
        .INSTR_DEPTH  (4096),
        .MAX_LOOP_NEST(4),
        .SRAM_ADDR_W  (SRAM_ADDR_W)
    ) lcp_inst (
        .clk            (clk),
        .rst_n          (rst_n),
        
        // Control
        .start          (tpc_start),
        .start_pc       (tpc_start_pc),
        .busy           (tpc_busy),
        .done           (tpc_done),
        .error          (tpc_error),
        
        // Instruction memory
        .imem_addr      (lcp_imem_addr),
        .imem_data      (lcp_imem_data),
        .imem_re        (lcp_imem_re),
        .imem_valid     (lcp_imem_valid),
        
        // MXU interface
        .mxu_cmd        (lcp_mxu_cmd),
        .mxu_valid      (lcp_mxu_valid),
        .mxu_ready      (mxu_lcp_ready),
        .mxu_done       (mxu_lcp_done),
        
        // VPU interface
        .vpu_cmd        (lcp_vpu_cmd),
        .vpu_valid      (lcp_vpu_valid),
        .vpu_ready      (vpu_lcp_ready),
        .vpu_done       (vpu_lcp_done),
        
        // DMA interface
        .dma_cmd        (lcp_dma_cmd),
        .dma_valid      (lcp_dma_valid),
        .dma_ready      (dma_lcp_ready),
        .dma_done       (dma_lcp_done),
        
        // Synchronization
        .global_sync_in (global_sync_in),
        .sync_request   (sync_request),
        .sync_grant     (sync_grant)
    );
    
    //==========================================================================
    // MXU Controller + Systolic Array
    //==========================================================================
    
    // MXU command decode
    wire [7:0]  mxu_subop    = lcp_mxu_cmd[119:112];
    wire [15:0] mxu_dst_addr = lcp_mxu_cmd[111:96];
    wire [15:0] mxu_src0_addr = lcp_mxu_cmd[95:80];
    wire [15:0] mxu_src1_addr = lcp_mxu_cmd[79:64];
    wire [15:0] mxu_cfg_m    = lcp_mxu_cmd[63:48];
    wire [15:0] mxu_cfg_n    = lcp_mxu_cmd[47:32];
    wire [15:0] mxu_cfg_k    = lcp_mxu_cmd[31:16];
    
    // MXU state machine
    localparam MXU_IDLE      = 3'd0;
    localparam MXU_LOAD_W    = 3'd1;
    localparam MXU_COMPUTE   = 3'd2;
    localparam MXU_DRAIN     = 3'd3;
    localparam MXU_DONE      = 3'd4;
    
    reg [2:0] mxu_state;
    reg [15:0] mxu_cycle_cnt;
    reg [4:0] mxu_col_cnt;
    reg mxu_start_array;
    reg mxu_ready_reg;
    reg mxu_done_reg;
    
    // Pipelined weight control signals (1 cycle delay to match SRAM latency)
    reg weight_load_en_d;
    reg [$clog2(ARRAY_SIZE)-1:0] weight_load_col_d;
    reg act_valid_d, act_valid_d2;  // Double-delayed for state machine alignment
    reg mxu_start_array_d;
    reg [ARRAY_SIZE*DATA_WIDTH-1:0] act_data_d;  // Delayed activation data
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_load_en_d <= 1'b0;
            weight_load_col_d <= 0;
            act_valid_d <= 1'b0;
            act_valid_d2 <= 1'b0;
            mxu_start_array_d <= 1'b0;
            act_data_d <= 0;
        end else begin
            weight_load_en_d <= (mxu_state == MXU_LOAD_W);
            weight_load_col_d <= mxu_col_cnt[$clog2(ARRAY_SIZE)-1:0];
            act_valid_d <= (mxu_state == MXU_COMPUTE);
            act_valid_d2 <= act_valid_d;  // Second delay stage
            mxu_start_array_d <= mxu_start_array;
            act_data_d <= mxu_a_rdata[ARRAY_SIZE*DATA_WIDTH-1:0];  // Delay data to match valid
        end
    end
    
    // Systolic array instance
    wire systolic_busy, systolic_done;
    wire [ARRAY_SIZE*DATA_WIDTH-1:0] systolic_act_data;
    wire [ARRAY_SIZE*ACC_WIDTH-1:0]  systolic_result;
    wire systolic_result_valid;
    
    systolic_array #(
        .ARRAY_SIZE (ARRAY_SIZE),
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH)
    ) mxu_array (
        .clk              (clk),
        .rst_n            (rst_n),
        .start            (mxu_start_array_d),
        .clear_acc        (1'b1),
        .busy             (systolic_busy),
        .done             (systolic_done),
        .cfg_k_tiles      (mxu_cfg_k),
        .weight_load_en   (weight_load_en_d),
        .weight_load_col  (weight_load_col_d),
        .weight_load_data (mxu_w_rdata[ARRAY_SIZE*DATA_WIDTH-1:0]),
        .act_valid        (act_valid_d2),
        .act_data         (act_data_d),
        .act_ready        (),
        .result_valid     (systolic_result_valid),
        .result_data      (systolic_result),
        .result_ready     (1'b1)
    );
    
    // MXU control state machine
    reg [15:0] mxu_out_cnt;  // Separate counter for output rows
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mxu_state <= MXU_IDLE;
            mxu_cycle_cnt <= 16'd0;
            mxu_out_cnt <= 16'd0;
            mxu_col_cnt <= 5'd0;
            mxu_start_array <= 1'b0;
            mxu_ready_reg <= 1'b1;
            mxu_done_reg <= 1'b0;
        end else begin
            mxu_start_array <= 1'b0;
            mxu_done_reg <= 1'b0;
            
            // Output counter - increment when writing results
            // Output counter - increment when writing results
            // Results can be valid during COMPUTE (late pipeline) or DRAIN
            if (systolic_result_valid && mxu_o_ready && 
                (mxu_state == MXU_COMPUTE || mxu_state == MXU_DRAIN)) begin
                mxu_out_cnt <= mxu_out_cnt + 1;
            end
            
            case (mxu_state)
                MXU_IDLE: begin
                    mxu_ready_reg <= 1'b1;
                    mxu_out_cnt <= 16'd0;
                    if (lcp_mxu_valid) begin
                        mxu_ready_reg <= 1'b0;
                        mxu_col_cnt <= 5'd0;
                        mxu_state <= MXU_LOAD_W;
                    end
                end
                
                MXU_LOAD_W: begin
                    // Load weights column by column
                    if (mxu_w_ready) begin
                        mxu_col_cnt <= mxu_col_cnt + 1;
                        if (mxu_col_cnt >= ARRAY_SIZE - 1) begin
                            mxu_start_array <= 1'b1;
                            mxu_cycle_cnt <= 16'd0;
                            mxu_state <= MXU_COMPUTE;
                        end
                    end
                end
                
                MXU_COMPUTE: begin
                    if (mxu_a_ready) begin
                        mxu_cycle_cnt <= mxu_cycle_cnt + 1;
                    end
                    if (systolic_done) begin
                        mxu_state <= MXU_DRAIN;
                    end
                end
                
                MXU_DRAIN: begin
                    // Wait for all outputs to be written
                    if (mxu_out_cnt >= ARRAY_SIZE) begin
                        mxu_state <= MXU_DONE;
                    end
                end
                
                MXU_DONE: begin
                    mxu_done_reg <= 1'b1;
                    mxu_state <= MXU_IDLE;
                end
            endcase
        end
    end
    
    // MXU SRAM interface
    assign mxu_w_addr = mxu_src1_addr + mxu_col_cnt;
    assign mxu_w_re = (mxu_state == MXU_LOAD_W);
    assign mxu_a_addr = mxu_src0_addr + mxu_cycle_cnt;
    assign mxu_a_re = (mxu_state == MXU_COMPUTE);
    assign mxu_o_addr = mxu_dst_addr + mxu_out_cnt;
    assign mxu_o_wdata = systolic_result[SRAM_WIDTH-1:0];
    assign mxu_o_we = systolic_result_valid && 
                      (mxu_state == MXU_COMPUTE || mxu_state == MXU_DRAIN);
    assign mxu_lcp_ready = mxu_ready_reg;
    assign mxu_lcp_done = mxu_done_reg;
    
    //==========================================================================
    // Vector Processing Unit (VPU)
    //==========================================================================
    
    vector_unit #(
        .LANES      (VPU_LANES),
        .DATA_WIDTH (VPU_DATA_W),
        .VREG_COUNT (32),
        .SRAM_ADDR_W(SRAM_ADDR_W)
    ) vpu_inst (
        .clk        (clk),
        .rst_n      (rst_n),
        .cmd        (lcp_vpu_cmd),
        .cmd_valid  (lcp_vpu_valid),
        .cmd_ready  (vpu_lcp_ready),
        .cmd_done   (vpu_lcp_done),
        .sram_addr  (vpu_sram_addr),
        .sram_wdata (vpu_sram_wdata),
        .sram_rdata (vpu_sram_rdata),
        .sram_we    (vpu_sram_we),
        .sram_re    (vpu_sram_re),
        .sram_ready (vpu_sram_ready)
    );
    
    //==========================================================================
    // DMA Engine
    //==========================================================================
    
    dma_engine #(
        .EXT_ADDR_W (EXT_ADDR_W),
        .INT_ADDR_W (SRAM_ADDR_W),
        .DATA_WIDTH (EXT_DATA_W),
        .MAX_BURST  (16)
    ) dma_inst (
        .clk        (clk),
        .rst_n      (rst_n),
        
        // Command
        .cmd        (lcp_dma_cmd),
        .cmd_valid  (lcp_dma_valid),
        .cmd_ready  (dma_lcp_ready),
        .cmd_done   (dma_lcp_done),
        
        // SRAM interface
        .sram_addr  (dma_sram_addr),
        .sram_wdata (dma_sram_wdata),
        .sram_rdata (dma_sram_rdata),
        .sram_we    (dma_sram_we),
        .sram_re    (dma_sram_re),
        .sram_ready (dma_sram_ready),
        
        // AXI interface
        .axi_awaddr (axi_awaddr),
        .axi_awlen  (axi_awlen),
        .axi_awvalid(axi_awvalid),
        .axi_awready(axi_awready),
        .axi_wdata  (axi_wdata),
        .axi_wlast  (axi_wlast),
        .axi_wvalid (axi_wvalid),
        .axi_wready (axi_wready),
        .axi_bresp  (axi_bresp),
        .axi_bvalid (axi_bvalid),
        .axi_bready (axi_bready),
        .axi_araddr (axi_araddr),
        .axi_arlen  (axi_arlen),
        .axi_arvalid(axi_arvalid),
        .axi_arready(axi_arready),
        .axi_rdata  (axi_rdata),
        .axi_rlast  (axi_rlast),
        .axi_rvalid (axi_rvalid),
        .axi_rready (axi_rready)
    );
    
    //==========================================================================
    // SRAM Subsystem
    //==========================================================================
    
    sram_subsystem #(
        .NUM_BANKS  (SRAM_BANKS),
        .BANK_DEPTH (SRAM_DEPTH),
        .DATA_WIDTH (SRAM_WIDTH),
        .ADDR_WIDTH (SRAM_ADDR_W)
    ) sram_inst (
        .clk         (clk),
        .rst_n       (rst_n),
        
        // MXU ports
        .mxu_w_addr  (mxu_w_addr),
        .mxu_w_re    (mxu_w_re),
        .mxu_w_rdata (mxu_w_rdata),
        .mxu_w_ready (mxu_w_ready),
        
        .mxu_a_addr  (mxu_a_addr),
        .mxu_a_re    (mxu_a_re),
        .mxu_a_rdata (mxu_a_rdata),
        .mxu_a_ready (mxu_a_ready),
        
        .mxu_o_addr  (mxu_o_addr),
        .mxu_o_wdata (mxu_o_wdata),
        .mxu_o_we    (mxu_o_we),
        .mxu_o_ready (mxu_o_ready),
        
        // VPU port
        .vpu_addr    (vpu_sram_addr),
        .vpu_wdata   (vpu_sram_wdata),
        .vpu_we      (vpu_sram_we),
        .vpu_re      (vpu_sram_re),
        .vpu_rdata   (vpu_sram_rdata),
        .vpu_ready   (vpu_sram_ready),
        
        // DMA port
        .dma_addr    (dma_sram_addr),
        .dma_wdata   (dma_sram_wdata),
        .dma_we      (dma_sram_we),
        .dma_re      (dma_sram_re),
        .dma_rdata   (dma_sram_rdata),
        .dma_ready   (dma_sram_ready)
    );
    
    //==========================================================================
    // NoC Interface Logic
    //==========================================================================
    
    // Accept NoC writes when not busy with internal operations
    assign noc_rx_ready = !tpc_busy;
    
    // NoC data loading to SRAM (when not instruction)
    wire noc_data_write = noc_rx_valid && noc_rx_ready && !noc_rx_is_instr;
    
    // TODO: Implement proper NoC TX for sending results
    assign noc_tx_data = {SRAM_WIDTH{1'b0}};
    assign noc_tx_addr = {SRAM_ADDR_W{1'b0}};
    assign noc_tx_valid = 1'b0;

endmodule

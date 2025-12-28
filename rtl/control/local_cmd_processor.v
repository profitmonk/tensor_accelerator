//==============================================================================
// Local Command Processor (LCP)
//
// Custom microsequencer for tensor instruction dispatch:
// - Fetches instructions from local SRAM
// - Decodes and dispatches to MXU, VPU, DMA
// - Hardware loop support (nested loops)
// - Scoreboard for dependency tracking
// - Synchronization with other TPCs and GCP
//==============================================================================

module local_cmd_processor #(
    parameter INSTR_WIDTH  = 128,
    parameter INSTR_DEPTH  = 4096,
    parameter MAX_LOOP_NEST = 4,
    parameter SRAM_ADDR_W  = 20
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    //--------------------------------------------------------------------------
    // Control Interface (from GCP/NoC)
    //--------------------------------------------------------------------------
    input  wire                         start,          // Start execution
    input  wire [SRAM_ADDR_W-1:0]       start_pc,       // Starting instruction address
    output wire                         busy,           // Currently executing
    output wire                         done,           // Execution complete
    output wire                         error,          // Error occurred
    
    //--------------------------------------------------------------------------
    // Instruction Memory Interface
    //--------------------------------------------------------------------------
    output wire [SRAM_ADDR_W-1:0]       imem_addr,
    input  wire [INSTR_WIDTH-1:0]       imem_data,
    output wire                         imem_re,
    input  wire                         imem_valid,
    
    //--------------------------------------------------------------------------
    // MXU Command Interface
    //--------------------------------------------------------------------------
    output wire [INSTR_WIDTH-1:0]       mxu_cmd,
    output wire                         mxu_valid,
    input  wire                         mxu_ready,
    input  wire                         mxu_done,
    
    //--------------------------------------------------------------------------
    // VPU Command Interface
    //--------------------------------------------------------------------------
    output wire [INSTR_WIDTH-1:0]       vpu_cmd,
    output wire                         vpu_valid,
    input  wire                         vpu_ready,
    input  wire                         vpu_done,
    
    //--------------------------------------------------------------------------
    // DMA Command Interface
    //--------------------------------------------------------------------------
    output wire [INSTR_WIDTH-1:0]       dma_cmd,
    output wire                         dma_valid,
    input  wire                         dma_ready,
    input  wire                         dma_done,
    
    //--------------------------------------------------------------------------
    // Synchronization Interface
    //--------------------------------------------------------------------------
    input  wire                         global_sync_in,  // Sync from GCP
    output wire                         sync_request,    // Request global sync
    input  wire                         sync_grant       // Sync granted
);

    //--------------------------------------------------------------------------
    // Instruction Decode
    //--------------------------------------------------------------------------
    
    wire [7:0]  opcode = imem_data[127:120];
    wire [7:0]  subop  = imem_data[119:112];
    wire [15:0] loop_count = imem_data[47:32];
    wire [7:0]  sync_mask = imem_data[111:104];
    
    // Opcodes
    localparam OP_NOP       = 8'h00;
    localparam OP_TENSOR    = 8'h01;
    localparam OP_VECTOR    = 8'h02;
    localparam OP_DMA       = 8'h03;
    localparam OP_SYNC      = 8'h04;
    localparam OP_LOOP      = 8'h05;
    localparam OP_ENDLOOP   = 8'h06;
    localparam OP_BARRIER   = 8'h07;
    localparam OP_HALT      = 8'hFF;
    
    // Sync subops
    localparam SYNC_MXU     = 8'h01;
    localparam SYNC_VPU     = 8'h02;
    localparam SYNC_DMA     = 8'h03;
    localparam SYNC_ALL     = 8'hFF;
    
    //--------------------------------------------------------------------------
    // State Machine
    //--------------------------------------------------------------------------
    
    localparam S_IDLE       = 4'd0;
    localparam S_FETCH      = 4'd1;
    localparam S_FETCH_WAIT = 4'd2;
    localparam S_DECODE     = 4'd3;
    localparam S_CHECK_DEP  = 4'd4;
    localparam S_ISSUE      = 4'd5;
    localparam S_WAIT_SYNC  = 4'd6;
    localparam S_BARRIER    = 4'd7;
    localparam S_HALTED     = 4'd8;
    localparam S_ERROR      = 4'd9;
    
    reg [3:0] state;
    reg [SRAM_ADDR_W-1:0] pc;
    reg [INSTR_WIDTH-1:0] instr_reg;
    
    //--------------------------------------------------------------------------
    // Hardware Loop Stack
    //--------------------------------------------------------------------------
    
    reg [SRAM_ADDR_W-1:0] loop_start_addr [0:MAX_LOOP_NEST-1];
    reg [15:0]            loop_counter    [0:MAX_LOOP_NEST-1];
    reg [$clog2(MAX_LOOP_NEST)-1:0] loop_sp;
    
    //--------------------------------------------------------------------------
    // Scoreboard: Track Pending Operations
    //--------------------------------------------------------------------------
    
    reg [7:0] pending_mxu;
    reg [7:0] pending_vpu;
    reg [7:0] pending_dma;
    
    wire all_done = (pending_mxu == 0) && (pending_vpu == 0) && (pending_dma == 0);
    wire mxu_clear = (pending_mxu == 0);
    wire vpu_clear = (pending_vpu == 0);
    wire dma_clear = (pending_dma == 0);
    
    //--------------------------------------------------------------------------
    // Output Registers
    //--------------------------------------------------------------------------
    
    reg [SRAM_ADDR_W-1:0]  imem_addr_reg;
    reg                    imem_re_reg;
    
    reg [INSTR_WIDTH-1:0]  mxu_cmd_reg;
    reg                    mxu_valid_reg;
    
    reg [INSTR_WIDTH-1:0]  vpu_cmd_reg;
    reg                    vpu_valid_reg;
    
    reg [INSTR_WIDTH-1:0]  dma_cmd_reg;
    reg                    dma_valid_reg;
    
    reg                    done_reg;
    reg                    error_reg;
    reg                    sync_request_reg;
    
    //--------------------------------------------------------------------------
    // Completion Tracking
    //--------------------------------------------------------------------------
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pending_mxu <= 8'd0;
            pending_vpu <= 8'd0;
            pending_dma <= 8'd0;
        end else begin
            // Decrement on completion
            if (mxu_done && pending_mxu > 0) pending_mxu <= pending_mxu - 1;
            if (vpu_done && pending_vpu > 0) pending_vpu <= pending_vpu - 1;
            if (dma_done && pending_dma > 0) pending_dma <= pending_dma - 1;
            
            // Increment on issue
            if (mxu_valid_reg && mxu_ready) pending_mxu <= pending_mxu + 1;
            if (vpu_valid_reg && vpu_ready) pending_vpu <= pending_vpu + 1;
            if (dma_valid_reg && dma_ready) pending_dma <= pending_dma + 1;
        end
    end
    
    //--------------------------------------------------------------------------
    // Main State Machine
    //--------------------------------------------------------------------------
    
    reg [7:0] decoded_opcode;
    reg [7:0] decoded_subop;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            pc <= {SRAM_ADDR_W{1'b0}};
            instr_reg <= {INSTR_WIDTH{1'b0}};
            loop_sp <= 0;
            
            imem_addr_reg <= {SRAM_ADDR_W{1'b0}};
            imem_re_reg <= 1'b0;
            
            mxu_cmd_reg <= {INSTR_WIDTH{1'b0}};
            mxu_valid_reg <= 1'b0;
            
            vpu_cmd_reg <= {INSTR_WIDTH{1'b0}};
            vpu_valid_reg <= 1'b0;
            
            dma_cmd_reg <= {INSTR_WIDTH{1'b0}};
            dma_valid_reg <= 1'b0;
            
            done_reg <= 1'b0;
            error_reg <= 1'b0;
            sync_request_reg <= 1'b0;
            
            decoded_opcode <= 8'd0;
            decoded_subop <= 8'd0;
            
            // Initialize loop stack
            for (integer i = 0; i < MAX_LOOP_NEST; i = i + 1) begin
                loop_start_addr[i] <= {SRAM_ADDR_W{1'b0}};
                loop_counter[i] <= 16'd0;
            end
        end else begin
            // Clear handshake signals
            if (mxu_valid_reg && mxu_ready) mxu_valid_reg <= 1'b0;
            if (vpu_valid_reg && vpu_ready) vpu_valid_reg <= 1'b0;
            if (dma_valid_reg && dma_ready) dma_valid_reg <= 1'b0;
            
            done_reg <= 1'b0;
            imem_re_reg <= 1'b0;
            
            case (state)
                //--------------------------------------------------------------
                S_IDLE: begin
                    if (start) begin
                        pc <= start_pc;
                        loop_sp <= 0;
                        error_reg <= 1'b0;
                        state <= S_FETCH;
                    end
                end
                
                //--------------------------------------------------------------
                S_FETCH: begin
                    imem_addr_reg <= pc;
                    imem_re_reg <= 1'b1;
                    state <= S_FETCH_WAIT;
                end
                
                //--------------------------------------------------------------
                S_FETCH_WAIT: begin
                    if (imem_valid) begin
                        instr_reg <= imem_data;
                        decoded_opcode <= imem_data[127:120];
                        decoded_subop <= imem_data[119:112];
                        state <= S_DECODE;
                    end
                end
                
                //--------------------------------------------------------------
                S_DECODE: begin
                    case (decoded_opcode)
                        OP_NOP: begin
                            pc <= pc + 1;
                            state <= S_FETCH;
                        end
                        
                        OP_TENSOR, OP_VECTOR, OP_DMA: begin
                            state <= S_CHECK_DEP;
                        end
                        
                        OP_SYNC: begin
                            state <= S_WAIT_SYNC;
                        end
                        
                        OP_LOOP: begin
                            // Push loop context
                            if (loop_sp < MAX_LOOP_NEST) begin
                                loop_start_addr[loop_sp] <= pc + 1;
                                loop_counter[loop_sp] <= instr_reg[47:32];
                                loop_sp <= loop_sp + 1;
                                pc <= pc + 1;
                                state <= S_FETCH;
                            end else begin
                                // Loop stack overflow
                                error_reg <= 1'b1;
                                state <= S_ERROR;
                            end
                        end
                        
                        OP_ENDLOOP: begin
                            if (loop_sp > 0) begin
                                if (loop_counter[loop_sp-1] > 1) begin
                                    // Continue loop
                                    loop_counter[loop_sp-1] <= loop_counter[loop_sp-1] - 1;
                                    pc <= loop_start_addr[loop_sp-1];
                                end else begin
                                    // Exit loop
                                    loop_sp <= loop_sp - 1;
                                    pc <= pc + 1;
                                end
                                state <= S_FETCH;
                            end else begin
                                // Loop stack underflow
                                error_reg <= 1'b1;
                                state <= S_ERROR;
                            end
                        end
                        
                        OP_BARRIER: begin
                            sync_request_reg <= 1'b1;
                            state <= S_BARRIER;
                        end
                        
                        OP_HALT: begin
                            done_reg <= 1'b1;
                            state <= S_HALTED;
                        end
                        
                        default: begin
                            // Unknown opcode
                            error_reg <= 1'b1;
                            state <= S_ERROR;
                        end
                    endcase
                end
                
                //--------------------------------------------------------------
                S_CHECK_DEP: begin
                    // For now: simple approach - wait for all pending ops
                    // Advanced: check specific address/register dependencies
                    if (all_done) begin
                        state <= S_ISSUE;
                    end
                    // Stay in CHECK_DEP if dependencies not clear
                end
                
                //--------------------------------------------------------------
                S_ISSUE: begin
                    case (decoded_opcode)
                        OP_TENSOR: begin
                            mxu_cmd_reg <= instr_reg;
                            mxu_valid_reg <= 1'b1;
                            if (mxu_ready) begin
                                pc <= pc + 1;
                                state <= S_FETCH;
                            end
                        end
                        
                        OP_VECTOR: begin
                            vpu_cmd_reg <= instr_reg;
                            vpu_valid_reg <= 1'b1;
                            if (vpu_ready) begin
                                pc <= pc + 1;
                                state <= S_FETCH;
                            end
                        end
                        
                        OP_DMA: begin
                            dma_cmd_reg <= instr_reg;
                            dma_valid_reg <= 1'b1;
                            if (dma_ready) begin
                                pc <= pc + 1;
                                state <= S_FETCH;
                            end
                        end
                        
                        default: begin
                            pc <= pc + 1;
                            state <= S_FETCH;
                        end
                    endcase
                end
                
                //--------------------------------------------------------------
                S_WAIT_SYNC: begin
                    case (decoded_subop)
                        SYNC_MXU: begin
                            if (mxu_clear) begin
                                pc <= pc + 1;
                                state <= S_FETCH;
                            end
                        end
                        
                        SYNC_VPU: begin
                            if (vpu_clear) begin
                                pc <= pc + 1;
                                state <= S_FETCH;
                            end
                        end
                        
                        SYNC_DMA: begin
                            if (dma_clear) begin
                                pc <= pc + 1;
                                state <= S_FETCH;
                            end
                        end
                        
                        SYNC_ALL: begin
                            if (all_done) begin
                                pc <= pc + 1;
                                state <= S_FETCH;
                            end
                        end
                        
                        default: begin
                            pc <= pc + 1;
                            state <= S_FETCH;
                        end
                    endcase
                end
                
                //--------------------------------------------------------------
                S_BARRIER: begin
                    // Wait for global sync grant
                    if (sync_grant) begin
                        sync_request_reg <= 1'b0;
                        pc <= pc + 1;
                        state <= S_FETCH;
                    end
                end
                
                //--------------------------------------------------------------
                S_HALTED: begin
                    // Wait for restart
                    if (start) begin
                        pc <= start_pc;
                        loop_sp <= 0;
                        done_reg <= 1'b0;
                        state <= S_FETCH;
                    end
                end
                
                //--------------------------------------------------------------
                S_ERROR: begin
                    // Stay in error state until reset
                    if (start) begin
                        error_reg <= 1'b0;
                        pc <= start_pc;
                        loop_sp <= 0;
                        state <= S_FETCH;
                    end
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
    
    assign imem_addr = imem_addr_reg;
    assign imem_re = imem_re_reg;
    
    assign mxu_cmd = mxu_cmd_reg;
    assign mxu_valid = mxu_valid_reg;
    
    assign vpu_cmd = vpu_cmd_reg;
    assign vpu_valid = vpu_valid_reg;
    
    assign dma_cmd = dma_cmd_reg;
    assign dma_valid = dma_valid_reg;
    
    assign busy = (state != S_IDLE) && (state != S_HALTED) && (state != S_ERROR);
    assign done = done_reg;
    assign error = error_reg;
    assign sync_request = sync_request_reg;

endmodule

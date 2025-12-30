//==============================================================================
// Vector Processing Unit (VPU)
//
// SIMD vector unit for element-wise operations:
// - Arithmetic: ADD, SUB, MUL, MADD (fused multiply-add)
// - Activation: RELU, GELU (approximation), SILU
// - Reduction: SUM, MAX, MIN (tree-based)
// - Data movement: LOAD, STORE, BROADCAST
// - Conversion: Quantize/Dequantize
//
// Architecture:
// - LANES parallel ALUs
// - 2-read, 1-write vector register file
// - Pipelined execution (3-4 stages)
//==============================================================================

module vector_unit #(
    parameter LANES       = 64,       // SIMD width
    parameter DATA_WIDTH  = 16,       // BF16 or FP16
    parameter VREG_COUNT  = 32,       // Number of vector registers
    parameter SRAM_ADDR_W = 20        // SRAM address width
)(
    input  wire                             clk,
    input  wire                             rst_n,
    
    //--------------------------------------------------------------------------
    // Command Interface (from LCP)
    //--------------------------------------------------------------------------
    input  wire [127:0]                     cmd,
    input  wire                             cmd_valid,
    output wire                             cmd_ready,
    output wire                             cmd_done,
    
    //--------------------------------------------------------------------------
    // SRAM Interface
    //--------------------------------------------------------------------------
    output wire [SRAM_ADDR_W-1:0]           sram_addr,
    output wire [LANES*DATA_WIDTH-1:0]      sram_wdata,
    input  wire [LANES*DATA_WIDTH-1:0]      sram_rdata,
    output wire                             sram_we,
    output wire                             sram_re,
    input  wire                             sram_ready
);

    //--------------------------------------------------------------------------
    // Command Decode - use registered command for proper timing
    //--------------------------------------------------------------------------
    
    // Registered command fields (valid after S_IDLE)
    reg [7:0]  subop_reg;
    reg [4:0]  vd_reg;
    reg [4:0]  vs1_reg;
    reg [4:0]  vs2_reg;
    reg [15:0] imm_reg;
    reg [SRAM_ADDR_W-1:0] mem_addr_reg;
    reg [15:0] count_reg;
    
    // Immediate decode from input (used only in S_IDLE)
    // Field layout (128-bit command):
    //   [127:120] opcode (8 bits)
    //   [119:112] subop (8 bits)
    //   [111:107] vd - destination register (5 bits)
    //   [106:102] vs1 - source 1 register (5 bits)
    //   [101:97]  vs2 - source 2 register (5 bits)
    //   [95:76]   mem_addr (20 bits)
    //   [63:48]   count (16 bits)
    //   [47:32]   imm (16 bits)
    wire [7:0]  opcode  = cmd[127:120];
    wire [7:0]  subop   = cmd[119:112];
    wire [4:0]  vd      = cmd[111:107];     // Destination vreg (fixed: was 116:112)
    wire [4:0]  vs1     = cmd[106:102];     // Source 1 vreg (fixed: was 111:107)
    wire [4:0]  vs2     = cmd[101:97];      // Source 2 vreg (fixed: was 106:102)
    wire [15:0] imm     = cmd[47:32];       // Immediate value
    wire [SRAM_ADDR_W-1:0] mem_addr = cmd[95:96-SRAM_ADDR_W];
    wire [15:0] count   = cmd[63:48];       // Element count
    
    // Suboperation codes
    localparam VOP_ADD       = 8'h01;
    localparam VOP_SUB       = 8'h02;
    localparam VOP_MUL       = 8'h03;
    localparam VOP_MADD      = 8'h04;
    localparam VOP_RELU      = 8'h10;
    localparam VOP_GELU      = 8'h11;
    localparam VOP_SILU      = 8'h12;
    localparam VOP_SIGMOID   = 8'h13;
    localparam VOP_TANH      = 8'h14;
    localparam VOP_SUM       = 8'h20;
    localparam VOP_MAX       = 8'h21;
    localparam VOP_MIN       = 8'h22;
    localparam VOP_LOAD      = 8'h30;
    localparam VOP_STORE     = 8'h31;
    localparam VOP_BCAST     = 8'h32;
    localparam VOP_MOV       = 8'h33;
    localparam VOP_ZERO      = 8'h34;
    
    //--------------------------------------------------------------------------
    // Vector Register File
    //--------------------------------------------------------------------------
    
    reg [LANES*DATA_WIDTH-1:0] vrf [0:VREG_COUNT-1];
    
    // Read ports - use registered indices for proper timing
    // During S_IDLE, these may be invalid but that's OK since we don't use them yet
    wire [LANES*DATA_WIDTH-1:0] vs1_data = vrf[vs1_reg];
    wire [LANES*DATA_WIDTH-1:0] vs2_data = vrf[vs2_reg];
    
    //--------------------------------------------------------------------------
    // State Machine
    //--------------------------------------------------------------------------
    
    localparam S_IDLE     = 3'd0;
    localparam S_DECODE   = 3'd1;
    localparam S_EXECUTE  = 3'd2;
    localparam S_MEM_WAIT = 3'd3;
    localparam S_REDUCE   = 3'd4;
    localparam S_WRITEBACK = 3'd5;
    localparam S_DONE     = 3'd6;
    
    reg [2:0] state;
    reg [127:0] cmd_reg;
    reg [15:0] elem_count;
    reg [SRAM_ADDR_W-1:0] addr_reg;
    
    //--------------------------------------------------------------------------
    // ALU - Per-Lane Processing
    //--------------------------------------------------------------------------
    
    reg [LANES*DATA_WIDTH-1:0] alu_result;
    
    // Extract individual lane data
    wire [DATA_WIDTH-1:0] lane_a [0:LANES-1];
    wire [DATA_WIDTH-1:0] lane_b [0:LANES-1];
    reg  [DATA_WIDTH-1:0] lane_result [0:LANES-1];
    
    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : lane_extract
            assign lane_a[i] = vs1_data[i*DATA_WIDTH +: DATA_WIDTH];
            assign lane_b[i] = vs2_data[i*DATA_WIDTH +: DATA_WIDTH];
            
            always @(*) begin
                lane_result[i] = {DATA_WIDTH{1'b0}};
                
                case (subop_reg)
                    VOP_ADD: begin
                        // Simple integer add (for BF16, would need FP adder)
                        lane_result[i] = lane_a[i] + lane_b[i];
                    end
                    
                    VOP_SUB: begin
                        lane_result[i] = lane_a[i] - lane_b[i];
                    end
                    
                    VOP_MUL: begin
                        // Truncated multiply
                        lane_result[i] = lane_a[i] * lane_b[i];
                    end
                    
                    VOP_RELU: begin
                        // ReLU: max(0, x) - check sign bit
                        lane_result[i] = lane_a[i][DATA_WIDTH-1] ? {DATA_WIDTH{1'b0}} : lane_a[i];
                    end
                    
                    VOP_GELU: begin
                        // GELU approximation: x * sigmoid(1.702 * x)
                        // For FPGA POC, use ReLU as placeholder
                        lane_result[i] = lane_a[i][DATA_WIDTH-1] ? {DATA_WIDTH{1'b0}} : lane_a[i];
                    end
                    
                    VOP_ZERO: begin
                        lane_result[i] = {DATA_WIDTH{1'b0}};
                    end
                    
                    VOP_MOV: begin
                        lane_result[i] = lane_a[i];
                    end
                    
                    VOP_BCAST: begin
                        // Broadcast immediate or first element
                        lane_result[i] = imm_reg;
                    end
                    
                    default: begin
                        lane_result[i] = lane_a[i];
                    end
                endcase
            end
            
            // Pack results back
            always @(*) begin
                alu_result[i*DATA_WIDTH +: DATA_WIDTH] = lane_result[i];
            end
        end
    endgenerate
    
    //--------------------------------------------------------------------------
    // Reduction Tree (for SUM, MAX, MIN)
    //--------------------------------------------------------------------------
    
    // Log2(LANES) stages of reduction
    localparam REDUCE_STAGES = $clog2(LANES);
    
    reg [DATA_WIDTH-1:0] reduce_tree [0:REDUCE_STAGES][0:LANES-1];
    reg [DATA_WIDTH-1:0] reduce_result;
    reg [$clog2(REDUCE_STAGES+1)-1:0] reduce_stage;
    
    // Initialize first stage with input data
    integer stage, lane;
    always @(*) begin
        for (lane = 0; lane < LANES; lane = lane + 1) begin
            reduce_tree[0][lane] = lane_a[lane];
        end
        
        // Build reduction tree
        for (stage = 1; stage <= REDUCE_STAGES; stage = stage + 1) begin
            for (lane = 0; lane < (LANES >> stage); lane = lane + 1) begin
                case (subop_reg)
                    VOP_SUM: begin
                        reduce_tree[stage][lane] = reduce_tree[stage-1][lane*2] + 
                                                   reduce_tree[stage-1][lane*2+1];
                    end
                    VOP_MAX: begin
                        reduce_tree[stage][lane] = ($signed(reduce_tree[stage-1][lane*2]) > 
                                                    $signed(reduce_tree[stage-1][lane*2+1])) ?
                                                   reduce_tree[stage-1][lane*2] :
                                                   reduce_tree[stage-1][lane*2+1];
                    end
                    VOP_MIN: begin
                        reduce_tree[stage][lane] = ($signed(reduce_tree[stage-1][lane*2]) < 
                                                    $signed(reduce_tree[stage-1][lane*2+1])) ?
                                                   reduce_tree[stage-1][lane*2] :
                                                   reduce_tree[stage-1][lane*2+1];
                    end
                    default: begin
                        reduce_tree[stage][lane] = reduce_tree[stage-1][lane*2];
                    end
                endcase
            end
        end
        
        reduce_result = reduce_tree[REDUCE_STAGES][0];
    end
    
    //--------------------------------------------------------------------------
    // Main State Machine
    //--------------------------------------------------------------------------
    
    reg sram_we_reg, sram_re_reg;
    reg [SRAM_ADDR_W-1:0] sram_addr_reg;
    reg [LANES*DATA_WIDTH-1:0] sram_wdata_reg;
    reg done_reg;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            cmd_reg <= 128'd0;
            elem_count <= 16'd0;
            addr_reg <= {SRAM_ADDR_W{1'b0}};
            sram_we_reg <= 1'b0;
            sram_re_reg <= 1'b0;
            done_reg <= 1'b0;
            // Reset registered command fields
            subop_reg <= 8'd0;
            vd_reg <= 5'd0;
            vs1_reg <= 5'd0;
            vs2_reg <= 5'd0;
            imm_reg <= 16'd0;
            mem_addr_reg <= {SRAM_ADDR_W{1'b0}};
            count_reg <= 16'd0;
        end else begin
            sram_we_reg <= 1'b0;
            sram_re_reg <= 1'b0;
            done_reg <= 1'b0;
            
            case (state)
                S_IDLE: begin
                    if (cmd_valid) begin
                        cmd_reg <= cmd;
                        // Register all command fields for use in later states
                        subop_reg <= subop;
                        vd_reg <= vd;
                        vs1_reg <= vs1;
                        vs2_reg <= vs2;
                        imm_reg <= imm;
                        mem_addr_reg <= mem_addr;
                        count_reg <= count;
                        state <= S_DECODE;
                    end
                end
                
                S_DECODE: begin
                    elem_count <= count_reg;
                    addr_reg <= mem_addr_reg;
                    
                    case (subop_reg)
                        VOP_LOAD: begin
                            sram_re_reg <= 1'b1;
                            sram_addr_reg <= mem_addr_reg;
                            state <= S_MEM_WAIT;
                        end
                        
                        VOP_STORE: begin
                            sram_we_reg <= 1'b1;
                            sram_addr_reg <= mem_addr_reg;
                            sram_wdata_reg <= vs1_data;
                            state <= S_MEM_WAIT;
                        end
                        
                        VOP_SUM, VOP_MAX, VOP_MIN: begin
                            state <= S_REDUCE;
                        end
                        
                        default: begin
                            state <= S_EXECUTE;
                        end
                    endcase
                end
                
                S_EXECUTE: begin
                    // Write result to register file
                    vrf[vd_reg] <= alu_result;
                    state <= S_DONE;
                end
                
                S_MEM_WAIT: begin
                    if (sram_ready) begin
                        if (subop_reg == VOP_LOAD) begin
                            // Data arrives 1 cycle after ready, go to writeback
                            state <= S_WRITEBACK;
                        end else begin
                            // Stores complete immediately
                            state <= S_DONE;
                        end
                    end
                end
                
                S_WRITEBACK: begin
                    // Capture delayed read data into VRF
                    vrf[vd_reg] <= sram_rdata;
                    state <= S_DONE;
                end
                
                S_REDUCE: begin
                    // Reduction result goes to first element of destination
                    vrf[vd_reg] <= {{(LANES-1)*DATA_WIDTH{1'b0}}, reduce_result};
                    state <= S_DONE;
                end
                
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
    assign sram_addr = sram_addr_reg;
    assign sram_wdata = sram_wdata_reg;
    assign sram_we = sram_we_reg;
    assign sram_re = sram_re_reg;

endmodule

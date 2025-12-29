//==============================================================================
// Systolic Array - Rewritten to Match Python Functional Model
//
// Weight-stationary dataflow:
// - Weights loaded into PEs (stationary)
// - Activations flow horizontally with input skewing (row i delayed by i cycles)
// - Partial sums flow vertically (top to bottom)
// - Outputs de-skewed (column j delayed by 2*(ARRAY_SIZE-1-j) cycles)
//
// This implementation exactly matches model/systolic_array_model.py
//==============================================================================

module systolic_array #(
    parameter ARRAY_SIZE = 4,
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  wire                                 clk,
    input  wire                                 rst_n,
    
    // Control
    input  wire                                 start,
    input  wire                                 clear_acc,
    output wire                                 busy,
    output wire                                 done,
    
    // Configuration
    input  wire [15:0]                          cfg_k_tiles,
    
    // Weight loading interface
    input  wire                                 weight_load_en,
    input  wire [$clog2(ARRAY_SIZE)-1:0]        weight_load_col,
    input  wire [ARRAY_SIZE*DATA_WIDTH-1:0]     weight_load_data,
    
    // Activation input interface
    input  wire                                 act_valid,
    input  wire [ARRAY_SIZE*DATA_WIDTH-1:0]     act_data,
    output wire                                 act_ready,
    
    // Result output interface
    output wire                                 result_valid,
    output wire [ARRAY_SIZE*ACC_WIDTH-1:0]      result_data,
    input  wire                                 result_ready
);

    //==========================================================================
    // State Machine
    //==========================================================================
    localparam S_IDLE    = 3'd0;
    localparam S_LOAD    = 3'd1;
    localparam S_COMPUTE = 3'd2;
    localparam S_DRAIN   = 3'd3;
    localparam S_DONE    = 3'd4;
    
    reg [2:0] state, state_next;
    reg [15:0] cycle_count, cycle_count_next;
    
    // State register
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            cycle_count <= 16'd0;
        end else begin
            state <= state_next;
            cycle_count <= cycle_count_next;
        end
    end
    
    // Next state logic
    always @(*) begin
        state_next = state;
        cycle_count_next = cycle_count;
        
        case (state)
            S_IDLE: begin
                if (start) begin
                    state_next = weight_load_en ? S_LOAD : S_COMPUTE;
                    cycle_count_next = 16'd0;
                end
            end
            
            S_LOAD: begin
                if (!weight_load_en) begin
                    state_next = S_COMPUTE;
                    cycle_count_next = 16'd0;
                end
            end
            
            S_COMPUTE: begin
                cycle_count_next = cycle_count + 1;
                if (cycle_count >= cfg_k_tiles - 1) begin
                    state_next = S_DRAIN;
                    cycle_count_next = 16'd0;
                end
            end
            
            S_DRAIN: begin
                cycle_count_next = cycle_count + 1;
                // Drain cycles: 3*ARRAY_SIZE - 2
                if (cycle_count >= 3*ARRAY_SIZE - 3) begin
                    state_next = S_DONE;
                    cycle_count_next = 16'd0;
                end
            end
            
            S_DONE: begin
                state_next = S_IDLE;
            end
        endcase
    end
    
    // Control signals
    wire pe_enable = (state == S_COMPUTE) || (state == S_DRAIN) ||
                     (state == S_IDLE && start && !weight_load_en);
    wire skew_enable = pe_enable;
    
    assign busy = (state != S_IDLE) && (state != S_DONE);
    assign done = (state == S_DONE);
    assign act_ready = (state == S_COMPUTE);
    
    // Result valid after propagation delay
    // Propagation delay = 3 * ARRAY_SIZE - 3 cycles
    wire [15:0] propagation_delay = 3 * ARRAY_SIZE - 3;
    assign result_valid = ((state == S_COMPUTE) && (cycle_count >= propagation_delay)) ||
                          (state == S_DRAIN);

    //==========================================================================
    // Input Skewing Registers
    // Row i has i delay stages + 1 output register
    //==========================================================================
    
    wire [DATA_WIDTH-1:0] skew_input [0:ARRAY_SIZE-1];
    wire [DATA_WIDTH-1:0] skew_output [0:ARRAY_SIZE-1];
    
    genvar row;
    generate
        for (row = 0; row < ARRAY_SIZE; row = row + 1) begin : gen_skew
            // Input to skew register
            assign skew_input[row] = act_valid ? act_data[row*DATA_WIDTH +: DATA_WIDTH] : {DATA_WIDTH{1'b0}};
            
            if (row == 0) begin : row0_skew
                // Row 0: Just output register, no delay stages
                reg [DATA_WIDTH-1:0] out_reg;
                
                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n)
                        out_reg <= {DATA_WIDTH{1'b0}};
                    else if (skew_enable)
                        out_reg <= skew_input[row];
                end
                
                assign skew_output[row] = out_reg;
                
            end else begin : rowN_skew
                // Row i: i delay stages + 1 output register
                reg [DATA_WIDTH-1:0] delay_stages [0:row-1];
                reg [DATA_WIDTH-1:0] out_reg;
                integer i;
                
                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        for (i = 0; i < row; i = i + 1)
                            delay_stages[i] <= {DATA_WIDTH{1'b0}};
                        out_reg <= {DATA_WIDTH{1'b0}};
                    end else if (skew_enable) begin
                        // Stage 0 gets new input
                        delay_stages[0] <= skew_input[row];
                        // Shift through delay stages
                        for (i = 1; i < row; i = i + 1)
                            delay_stages[i] <= delay_stages[i-1];
                        // Output register gets from last delay stage
                        out_reg <= delay_stages[row-1];
                    end
                end
                
                assign skew_output[row] = out_reg;
            end
        end
    endgenerate

    //==========================================================================
    // PE Array and Internal Wiring
    //==========================================================================
    
    // Horizontal activation wiring: act_h[row][col] feeds PE[row][col]
    wire [DATA_WIDTH-1:0] act_h [0:ARRAY_SIZE-1][0:ARRAY_SIZE];
    
    // Vertical psum wiring: psum_v[row][col] feeds PE[row][col] from top
    wire [ACC_WIDTH-1:0] psum_v [0:ARRAY_SIZE][0:ARRAY_SIZE-1];
    
    // Wire skew outputs to column 0
    generate
        for (row = 0; row < ARRAY_SIZE; row = row + 1) begin : wire_col0
            assign act_h[row][0] = skew_output[row];
        end
    endgenerate
    
    // Top row psum input is zero
    genvar col;
    generate
        for (col = 0; col < ARRAY_SIZE; col = col + 1) begin : wire_psum_top
            assign psum_v[0][col] = {ACC_WIDTH{1'b0}};
        end
    endgenerate
    
    // Instantiate PE array
    generate
        for (row = 0; row < ARRAY_SIZE; row = row + 1) begin : pe_row
            for (col = 0; col < ARRAY_SIZE; col = col + 1) begin : pe_col
                
                wire load_weight = weight_load_en && (weight_load_col == col);
                wire [DATA_WIDTH-1:0] weight_in = weight_load_data[row*DATA_WIDTH +: DATA_WIDTH];
                wire do_clear = clear_acc && (state == S_COMPUTE || (state == S_IDLE && start)) && (cycle_count == 0);
                
                mac_pe #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH(ACC_WIDTH)
                ) pe_inst (
                    .clk         (clk),
                    .rst_n       (rst_n),
                    .enable      (pe_enable),
                    .load_weight (load_weight),
                    .clear_acc   (do_clear),
                    .weight_in   (weight_in),
                    .act_in      (act_h[row][col]),
                    .act_out     (act_h[row][col+1]),
                    .psum_in     (psum_v[row][col]),
                    .psum_out    (psum_v[row+1][col])
                );
            end
        end
    endgenerate

    //==========================================================================
    // Output De-skewing Registers
    // Column j has 2*(ARRAY_SIZE-1-j) delay stages
    //==========================================================================
    
    wire [ACC_WIDTH-1:0] psum_bottom [0:ARRAY_SIZE-1];
    wire [ACC_WIDTH-1:0] deskew_output [0:ARRAY_SIZE-1];
    
    generate
        for (col = 0; col < ARRAY_SIZE; col = col + 1) begin : gen_deskew
            assign psum_bottom[col] = psum_v[ARRAY_SIZE][col];
            
            localparam NUM_STAGES = 2 * (ARRAY_SIZE - 1 - col);
            
            if (NUM_STAGES == 0) begin : col_no_delay
                // Rightmost column: direct passthrough
                assign deskew_output[col] = psum_bottom[col];
                
            end else begin : col_with_delay
                reg [ACC_WIDTH-1:0] delay_stages [0:NUM_STAGES-1];
                integer i;
                
                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        for (i = 0; i < NUM_STAGES; i = i + 1)
                            delay_stages[i] <= {ACC_WIDTH{1'b0}};
                    end else if (pe_enable) begin
                        delay_stages[0] <= psum_bottom[col];
                        for (i = 1; i < NUM_STAGES; i = i + 1)
                            delay_stages[i] <= delay_stages[i-1];
                    end
                end
                
                assign deskew_output[col] = delay_stages[NUM_STAGES-1];
            end
        end
    endgenerate
    
    // Output wiring
    generate
        for (col = 0; col < ARRAY_SIZE; col = col + 1) begin : wire_output
            assign result_data[col*ACC_WIDTH +: ACC_WIDTH] = deskew_output[col];
        end
    endgenerate

endmodule

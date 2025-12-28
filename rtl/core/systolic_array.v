//==============================================================================
// Systolic Array - Matrix Multiply Unit (MXU)
//
// Implements a weight-stationary NxN systolic array for matrix multiplication.
// Computes C = A × B where:
//   - A is M×K (activations, streamed row-wise)
//   - B is K×N (weights, loaded into PE array)
//   - C is M×N (output, drained from bottom)
//
// Dataflow:
// 1. LOAD phase: Weights broadcast column-wise into PEs
// 2. COMPUTE phase: Activations stream left-to-right (skewed)
//                   Partial sums accumulate top-to-bottom
// 3. DRAIN phase: Results emerge from bottom row
//
// Latency: N cycles to fill + K cycles to compute + N cycles to drain
//==============================================================================

module systolic_array #(
    parameter ARRAY_SIZE  = 16,       // NxN array dimension
    parameter DATA_WIDTH  = 8,        // INT8 operands
    parameter ACC_WIDTH   = 32        // Accumulator width
)(
    input  wire                                 clk,
    input  wire                                 rst_n,
    
    //--------------------------------------------------------------------------
    // Control Interface
    //--------------------------------------------------------------------------
    input  wire                                 start,          // Start operation
    input  wire                                 clear_acc,      // Clear accumulators
    output wire                                 busy,           // Array is processing
    output wire                                 done,           // Operation complete
    
    // Dimensions (for tracking completion)
    input  wire [15:0]                          cfg_k_tiles,    // Number of K iterations
    
    //--------------------------------------------------------------------------
    // Weight Loading Interface (column broadcast)
    //--------------------------------------------------------------------------
    input  wire                                 weight_load_en,
    input  wire [$clog2(ARRAY_SIZE)-1:0]        weight_load_col,    // Which column to load
    input  wire [ARRAY_SIZE*DATA_WIDTH-1:0]     weight_load_data,   // All rows for this column
    
    //--------------------------------------------------------------------------
    // Activation Input (left edge, one per row)
    //--------------------------------------------------------------------------
    input  wire                                 act_valid,
    input  wire [ARRAY_SIZE*DATA_WIDTH-1:0]     act_data,       // One element per row
    output wire                                 act_ready,
    
    //--------------------------------------------------------------------------
    // Result Output (bottom edge, one per column)
    //--------------------------------------------------------------------------
    output wire                                 result_valid,
    output wire [ARRAY_SIZE*ACC_WIDTH-1:0]      result_data,    // One element per column
    input  wire                                 result_ready
);

    //--------------------------------------------------------------------------
    // Internal Signals
    //--------------------------------------------------------------------------
    
    // State machine
    localparam S_IDLE    = 3'd0;
    localparam S_LOAD    = 3'd1;
    localparam S_COMPUTE = 3'd2;
    localparam S_DRAIN   = 3'd3;
    localparam S_DONE    = 3'd4;
    
    reg [2:0] state, state_next;
    reg [15:0] cycle_count;
    reg [15:0] k_count;
    
    // PE enable signal
    wire pe_enable = (state == S_COMPUTE) && act_valid;
    
    // Inter-PE wiring
    // Horizontal activation wires: act_h[row][col]
    wire [DATA_WIDTH-1:0] act_h [0:ARRAY_SIZE-1][0:ARRAY_SIZE];
    
    // Vertical partial sum wires: psum_v[row][col]
    wire [ACC_WIDTH-1:0] psum_v [0:ARRAY_SIZE][0:ARRAY_SIZE-1];
    
    // Weight loading signals per PE
    wire [DATA_WIDTH-1:0] weight_to_pe [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    wire load_weight_pe [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    
    //--------------------------------------------------------------------------
    // Input Skewing Registers
    // Row i gets its input delayed by i cycles
    //--------------------------------------------------------------------------
    
    reg [DATA_WIDTH-1:0] skew_regs [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    
    genvar row, col, d;
    generate
        for (row = 0; row < ARRAY_SIZE; row = row + 1) begin : skew_gen
            // Row 0: no delay
            // Row 1: 1 cycle delay
            // Row N-1: N-1 cycles delay
            
            if (row == 0) begin : no_delay
                assign act_h[row][0] = act_data[row*DATA_WIDTH +: DATA_WIDTH];
            end else begin : with_delay
                // Shift register chain
                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        for (integer i = 0; i < row; i = i + 1) begin
                            skew_regs[row][i] <= {DATA_WIDTH{1'b0}};
                        end
                    end else if (pe_enable) begin
                        skew_regs[row][0] <= act_data[row*DATA_WIDTH +: DATA_WIDTH];
                        for (integer i = 1; i < row; i = i + 1) begin
                            skew_regs[row][i] <= skew_regs[row][i-1];
                        end
                    end
                end
                
                assign act_h[row][0] = skew_regs[row][row-1];
            end
        end
    endgenerate
    
    //--------------------------------------------------------------------------
    // PE Array Instantiation
    //--------------------------------------------------------------------------
    
    generate
        for (row = 0; row < ARRAY_SIZE; row = row + 1) begin : pe_row
            for (col = 0; col < ARRAY_SIZE; col = col + 1) begin : pe_col
                
                // Weight loading: column-wise broadcast
                assign weight_to_pe[row][col] = weight_load_data[row*DATA_WIDTH +: DATA_WIDTH];
                assign load_weight_pe[row][col] = weight_load_en && (weight_load_col == col);
                
                // Top row gets zero psum input
                wire [ACC_WIDTH-1:0] psum_input;
                if (row == 0) begin : top_row
                    assign psum_input = {ACC_WIDTH{1'b0}};
                end else begin : other_row
                    assign psum_input = psum_v[row][col];
                end
                
                mac_pe #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH(ACC_WIDTH)
                ) pe_inst (
                    .clk         (clk),
                    .rst_n       (rst_n),
                    .enable      (pe_enable),
                    .load_weight (load_weight_pe[row][col]),
                    .clear_acc   (clear_acc && (state == S_COMPUTE) && (cycle_count == 0)),
                    .weight_in   (weight_to_pe[row][col]),
                    .act_in      (act_h[row][col]),
                    .act_out     (act_h[row][col+1]),
                    .psum_in     (psum_input),
                    .psum_out    (psum_v[row+1][col])
                );
            end
        end
    endgenerate
    
    //--------------------------------------------------------------------------
    // Output Collection (Bottom Row)
    //--------------------------------------------------------------------------
    
    generate
        for (col = 0; col < ARRAY_SIZE; col = col + 1) begin : out_col
            assign result_data[col*ACC_WIDTH +: ACC_WIDTH] = psum_v[ARRAY_SIZE][col];
        end
    endgenerate
    
    //--------------------------------------------------------------------------
    // Output De-skewing Registers
    // Column i result is valid i cycles after column 0
    //--------------------------------------------------------------------------
    
    reg [ARRAY_SIZE-1:0] result_valid_pipe;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result_valid_pipe <= {ARRAY_SIZE{1'b0}};
        end else begin
            if (state == S_DRAIN) begin
                result_valid_pipe <= {result_valid_pipe[ARRAY_SIZE-2:0], 1'b1};
            end else begin
                result_valid_pipe <= {ARRAY_SIZE{1'b0}};
            end
        end
    end
    
    // Result is valid when all columns have valid data
    assign result_valid = (state == S_DRAIN) && result_valid_pipe[ARRAY_SIZE-1];
    
    //--------------------------------------------------------------------------
    // State Machine
    //--------------------------------------------------------------------------
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            cycle_count <= 16'd0;
            k_count <= 16'd0;
        end else begin
            state <= state_next;
            
            case (state)
                S_IDLE: begin
                    cycle_count <= 16'd0;
                    k_count <= 16'd0;
                end
                
                S_LOAD: begin
                    if (weight_load_en) begin
                        cycle_count <= cycle_count + 1;
                    end
                end
                
                S_COMPUTE: begin
                    if (act_valid) begin
                        cycle_count <= cycle_count + 1;
                    end
                end
                
                S_DRAIN: begin
                    if (result_ready) begin
                        cycle_count <= cycle_count + 1;
                    end
                end
                
                default: ;
            endcase
        end
    end
    
    always @(*) begin
        state_next = state;
        
        case (state)
            S_IDLE: begin
                if (start) begin
                    state_next = S_LOAD;
                end
            end
            
            S_LOAD: begin
                // Wait for all columns to be loaded
                if (weight_load_en && (weight_load_col == ARRAY_SIZE-1)) begin
                    state_next = S_COMPUTE;
                end
            end
            
            S_COMPUTE: begin
                // Compute for cfg_k_tiles cycles (plus array fill time)
                if (cycle_count >= cfg_k_tiles + ARRAY_SIZE - 1) begin
                    state_next = S_DRAIN;
                end
            end
            
            S_DRAIN: begin
                // Drain results for ARRAY_SIZE cycles
                if (cycle_count >= ARRAY_SIZE) begin
                    state_next = S_DONE;
                end
            end
            
            S_DONE: begin
                state_next = S_IDLE;
            end
        endcase
    end
    
    //--------------------------------------------------------------------------
    // Status Outputs
    //--------------------------------------------------------------------------
    
    assign busy = (state != S_IDLE);
    assign done = (state == S_DONE);
    assign act_ready = (state == S_COMPUTE);

endmodule

//==============================================================================
// MAC Processing Element for Systolic Array
// 
// Weight-stationary dataflow:
// - Weight is loaded once and held in register
// - Activations stream through horizontally
// - Partial sums accumulate vertically
//
// Parameters:
// - DATA_WIDTH: Input operand width (typically 8 for INT8)
// - ACC_WIDTH: Accumulator width (typically 32 to prevent overflow)
//==============================================================================

module mac_pe #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Control
    input  wire                     enable,         // Pipeline enable
    input  wire                     load_weight,    // Load new weight
    input  wire                     clear_acc,      // Clear accumulator
    
    // Weight input (loaded once per tile)
    input  wire [DATA_WIDTH-1:0]    weight_in,
    
    // Activation flow (horizontal: left to right)
    input  wire [DATA_WIDTH-1:0]    act_in,
    output reg  [DATA_WIDTH-1:0]    act_out,
    
    // Partial sum flow (vertical: top to bottom)
    input  wire [ACC_WIDTH-1:0]     psum_in,
    output reg  [ACC_WIDTH-1:0]     psum_out
);

    //--------------------------------------------------------------------------
    // Internal Registers
    //--------------------------------------------------------------------------
    
    // Stationary weight register
    reg [DATA_WIDTH-1:0] weight_reg;
    
    // Pipeline registers for timing
    reg [DATA_WIDTH-1:0] act_reg;
    reg [ACC_WIDTH-1:0]  psum_reg;
    
    //--------------------------------------------------------------------------
    // Multiplication (signed)
    //--------------------------------------------------------------------------
    
    wire signed [DATA_WIDTH-1:0]   a_signed = $signed(act_reg);
    wire signed [DATA_WIDTH-1:0]   w_signed = $signed(weight_reg);
    wire signed [2*DATA_WIDTH-1:0] product  = a_signed * w_signed;
    
    // Sign-extend product to accumulator width
    wire signed [ACC_WIDTH-1:0] product_ext = {{(ACC_WIDTH-2*DATA_WIDTH){product[2*DATA_WIDTH-1]}}, product};
    
    //--------------------------------------------------------------------------
    // Main Sequential Logic
    //--------------------------------------------------------------------------
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg <= {DATA_WIDTH{1'b0}};
            act_reg    <= {DATA_WIDTH{1'b0}};
            act_out    <= {DATA_WIDTH{1'b0}};
            psum_out   <= {ACC_WIDTH{1'b0}};
        end else begin
            // Weight loading (independent of enable)
            if (load_weight) begin
                weight_reg <= weight_in;
            end
            
            // Main datapath (when enabled)
            if (enable) begin
                // Register activation for timing
                act_reg  <= act_in;
                
                // Pass activation to right neighbor (1 cycle delay)
                act_out <= act_reg;
                
                // Compute: psum_out = psum_in + (act * weight)
                // Use psum_in directly (not registered) so accumulation
                // happens in the same cycle the partial sum arrives
                if (clear_acc) begin
                    psum_out <= product_ext;
                end else begin
                    psum_out <= psum_in + product_ext;
                end
            end
        end
    end

endmodule

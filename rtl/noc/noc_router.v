//==============================================================================
// NoC Router
//
// 5-port router for 2D mesh network:
// - North, South, East, West, Local ports
// - X-Y dimension-order routing (deadlock-free)
// - Simple round-robin arbitration
// - Credit-based flow control
//==============================================================================

module noc_router #(
    parameter DATA_WIDTH  = 256,          // Flit data width
    parameter ADDR_WIDTH  = 20,           // Address field width
    parameter COORD_BITS  = 4,            // Bits for X/Y coordinates
    parameter FIFO_DEPTH  = 4,            // Input buffer depth
    parameter ROUTER_X    = 0,            // This router's X coordinate
    parameter ROUTER_Y    = 0             // This router's Y coordinate
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    //--------------------------------------------------------------------------
    // North Port
    //--------------------------------------------------------------------------
    input  wire [DATA_WIDTH-1:0]        north_in_data,
    input  wire [COORD_BITS-1:0]        north_in_dest_x,
    input  wire [COORD_BITS-1:0]        north_in_dest_y,
    input  wire                         north_in_valid,
    output wire                         north_in_ready,
    
    output wire [DATA_WIDTH-1:0]        north_out_data,
    output wire [COORD_BITS-1:0]        north_out_dest_x,
    output wire [COORD_BITS-1:0]        north_out_dest_y,
    output wire                         north_out_valid,
    input  wire                         north_out_ready,
    
    //--------------------------------------------------------------------------
    // South Port
    //--------------------------------------------------------------------------
    input  wire [DATA_WIDTH-1:0]        south_in_data,
    input  wire [COORD_BITS-1:0]        south_in_dest_x,
    input  wire [COORD_BITS-1:0]        south_in_dest_y,
    input  wire                         south_in_valid,
    output wire                         south_in_ready,
    
    output wire [DATA_WIDTH-1:0]        south_out_data,
    output wire [COORD_BITS-1:0]        south_out_dest_x,
    output wire [COORD_BITS-1:0]        south_out_dest_y,
    output wire                         south_out_valid,
    input  wire                         south_out_ready,
    
    //--------------------------------------------------------------------------
    // East Port
    //--------------------------------------------------------------------------
    input  wire [DATA_WIDTH-1:0]        east_in_data,
    input  wire [COORD_BITS-1:0]        east_in_dest_x,
    input  wire [COORD_BITS-1:0]        east_in_dest_y,
    input  wire                         east_in_valid,
    output wire                         east_in_ready,
    
    output wire [DATA_WIDTH-1:0]        east_out_data,
    output wire [COORD_BITS-1:0]        east_out_dest_x,
    output wire [COORD_BITS-1:0]        east_out_dest_y,
    output wire                         east_out_valid,
    input  wire                         east_out_ready,
    
    //--------------------------------------------------------------------------
    // West Port
    //--------------------------------------------------------------------------
    input  wire [DATA_WIDTH-1:0]        west_in_data,
    input  wire [COORD_BITS-1:0]        west_in_dest_x,
    input  wire [COORD_BITS-1:0]        west_in_dest_y,
    input  wire                         west_in_valid,
    output wire                         west_in_ready,
    
    output wire [DATA_WIDTH-1:0]        west_out_data,
    output wire [COORD_BITS-1:0]        west_out_dest_x,
    output wire [COORD_BITS-1:0]        west_out_dest_y,
    output wire                         west_out_valid,
    input  wire                         west_out_ready,
    
    //--------------------------------------------------------------------------
    // Local Port (to/from TPC)
    //--------------------------------------------------------------------------
    input  wire [DATA_WIDTH-1:0]        local_in_data,
    input  wire [COORD_BITS-1:0]        local_in_dest_x,
    input  wire [COORD_BITS-1:0]        local_in_dest_y,
    input  wire                         local_in_valid,
    output wire                         local_in_ready,
    
    output wire [DATA_WIDTH-1:0]        local_out_data,
    output wire [COORD_BITS-1:0]        local_out_dest_x,
    output wire [COORD_BITS-1:0]        local_out_dest_y,
    output wire                         local_out_valid,
    input  wire                         local_out_ready
);

    //--------------------------------------------------------------------------
    // Port indices
    //--------------------------------------------------------------------------
    localparam PORT_NORTH = 3'd0;
    localparam PORT_SOUTH = 3'd1;
    localparam PORT_EAST  = 3'd2;
    localparam PORT_WEST  = 3'd3;
    localparam PORT_LOCAL = 3'd4;
    localparam NUM_PORTS  = 5;
    
    //--------------------------------------------------------------------------
    // Flit structure: {dest_y, dest_x, data}
    //--------------------------------------------------------------------------
    localparam FLIT_WIDTH = DATA_WIDTH + 2*COORD_BITS;
    
    //--------------------------------------------------------------------------
    // Input FIFOs
    //--------------------------------------------------------------------------
    
    wire [FLIT_WIDTH-1:0] fifo_in [0:NUM_PORTS-1];
    wire [FLIT_WIDTH-1:0] fifo_out [0:NUM_PORTS-1];
    wire [NUM_PORTS-1:0]  fifo_wr_en;
    wire [NUM_PORTS-1:0]  fifo_rd_en;
    wire [NUM_PORTS-1:0]  fifo_empty;
    wire [NUM_PORTS-1:0]  fifo_full;
    
    // Pack inputs into flits
    assign fifo_in[PORT_NORTH] = {north_in_dest_y, north_in_dest_x, north_in_data};
    assign fifo_in[PORT_SOUTH] = {south_in_dest_y, south_in_dest_x, south_in_data};
    assign fifo_in[PORT_EAST]  = {east_in_dest_y, east_in_dest_x, east_in_data};
    assign fifo_in[PORT_WEST]  = {west_in_dest_y, west_in_dest_x, west_in_data};
    assign fifo_in[PORT_LOCAL] = {local_in_dest_y, local_in_dest_x, local_in_data};
    
    assign fifo_wr_en[PORT_NORTH] = north_in_valid && !fifo_full[PORT_NORTH];
    assign fifo_wr_en[PORT_SOUTH] = south_in_valid && !fifo_full[PORT_SOUTH];
    assign fifo_wr_en[PORT_EAST]  = east_in_valid && !fifo_full[PORT_EAST];
    assign fifo_wr_en[PORT_WEST]  = west_in_valid && !fifo_full[PORT_WEST];
    assign fifo_wr_en[PORT_LOCAL] = local_in_valid && !fifo_full[PORT_LOCAL];
    
    assign north_in_ready = !fifo_full[PORT_NORTH];
    assign south_in_ready = !fifo_full[PORT_SOUTH];
    assign east_in_ready  = !fifo_full[PORT_EAST];
    assign west_in_ready  = !fifo_full[PORT_WEST];
    assign local_in_ready = !fifo_full[PORT_LOCAL];
    
    // Instantiate FIFOs
    genvar p;
    generate
        for (p = 0; p < NUM_PORTS; p = p + 1) begin : fifo_gen
            sync_fifo #(
                .WIDTH(FLIT_WIDTH),
                .DEPTH(FIFO_DEPTH)
            ) input_fifo (
                .clk    (clk),
                .rst_n  (rst_n),
                .wr_en  (fifo_wr_en[p]),
                .wr_data(fifo_in[p]),
                .rd_en  (fifo_rd_en[p]),
                .rd_data(fifo_out[p]),
                .empty  (fifo_empty[p]),
                .full   (fifo_full[p])
            );
        end
    endgenerate
    
    //--------------------------------------------------------------------------
    // Route Computation (X-Y Dimension Order Routing)
    //--------------------------------------------------------------------------
    
    // Extract destination from each FIFO head
    wire [COORD_BITS-1:0] dest_x [0:NUM_PORTS-1];
    wire [COORD_BITS-1:0] dest_y [0:NUM_PORTS-1];
    wire [2:0] out_port [0:NUM_PORTS-1];
    
    generate
        for (p = 0; p < NUM_PORTS; p = p + 1) begin : route_gen
            assign dest_x[p] = fifo_out[p][DATA_WIDTH +: COORD_BITS];
            assign dest_y[p] = fifo_out[p][DATA_WIDTH + COORD_BITS +: COORD_BITS];
            
            // X-Y routing: first route in X, then in Y
            assign out_port[p] = 
                (dest_x[p] < ROUTER_X) ? PORT_WEST :
                (dest_x[p] > ROUTER_X) ? PORT_EAST :
                (dest_y[p] < ROUTER_Y) ? PORT_SOUTH :
                (dest_y[p] > ROUTER_Y) ? PORT_NORTH :
                PORT_LOCAL;  // Destination reached
        end
    endgenerate
    
    //--------------------------------------------------------------------------
    // Arbitration and Crossbar
    //--------------------------------------------------------------------------
    
    // Request matrix: req[in_port][out_port]
    wire [NUM_PORTS-1:0] req [0:NUM_PORTS-1];
    reg  [NUM_PORTS-1:0] grant [0:NUM_PORTS-1];
    
    // Output port ready signals
    wire [NUM_PORTS-1:0] out_ready;
    assign out_ready[PORT_NORTH] = north_out_ready;
    assign out_ready[PORT_SOUTH] = south_out_ready;
    assign out_ready[PORT_EAST]  = east_out_ready;
    assign out_ready[PORT_WEST]  = west_out_ready;
    assign out_ready[PORT_LOCAL] = local_out_ready;
    
    // Generate request signals
    generate
        for (p = 0; p < NUM_PORTS; p = p + 1) begin : req_gen
            assign req[p][PORT_NORTH] = !fifo_empty[p] && (out_port[p] == PORT_NORTH);
            assign req[p][PORT_SOUTH] = !fifo_empty[p] && (out_port[p] == PORT_SOUTH);
            assign req[p][PORT_EAST]  = !fifo_empty[p] && (out_port[p] == PORT_EAST);
            assign req[p][PORT_WEST]  = !fifo_empty[p] && (out_port[p] == PORT_WEST);
            assign req[p][PORT_LOCAL] = !fifo_empty[p] && (out_port[p] == PORT_LOCAL);
        end
    endgenerate
    
    // Round-robin priority per output port
    reg [2:0] priority [0:NUM_PORTS-1];
    
    // Arbitration logic (simplified round-robin)
    integer out_p, in_p;
    always @(*) begin
        // Initialize grants to zero
        for (out_p = 0; out_p < NUM_PORTS; out_p = out_p + 1) begin
            grant[out_p] = {NUM_PORTS{1'b0}};
        end
        
        // For each output port, grant to highest priority requester
        for (out_p = 0; out_p < NUM_PORTS; out_p = out_p + 1) begin
            if (out_ready[out_p]) begin
                for (in_p = 0; in_p < NUM_PORTS; in_p = in_p + 1) begin
                    // Check each input in priority order
                    if (req[(priority[out_p] + in_p) % NUM_PORTS][out_p] && 
                        grant[out_p] == {NUM_PORTS{1'b0}}) begin
                        grant[out_p][(priority[out_p] + in_p) % NUM_PORTS] = 1'b1;
                    end
                end
            end
        end
    end
    
    // Update priority after grant
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (integer i = 0; i < NUM_PORTS; i = i + 1) begin
                priority[i] <= 3'd0;
            end
        end else begin
            for (out_p = 0; out_p < NUM_PORTS; out_p = out_p + 1) begin
                if (grant[out_p] != {NUM_PORTS{1'b0}}) begin
                    // Advance priority to next input
                    for (in_p = 0; in_p < NUM_PORTS; in_p = in_p + 1) begin
                        if (grant[out_p][in_p]) begin
                            priority[out_p] <= (in_p + 1) % NUM_PORTS;
                        end
                    end
                end
            end
        end
    end
    
    // FIFO read enable based on grants
    generate
        for (p = 0; p < NUM_PORTS; p = p + 1) begin : rd_en_gen
            assign fifo_rd_en[p] = grant[PORT_NORTH][p] || 
                                   grant[PORT_SOUTH][p] ||
                                   grant[PORT_EAST][p] ||
                                   grant[PORT_WEST][p] ||
                                   grant[PORT_LOCAL][p];
        end
    endgenerate
    
    //--------------------------------------------------------------------------
    // Output Multiplexers
    //--------------------------------------------------------------------------
    
    // Select which input feeds each output
    reg [FLIT_WIDTH-1:0] out_flit [0:NUM_PORTS-1];
    reg [NUM_PORTS-1:0]  out_valid;
    
    always @(*) begin
        for (out_p = 0; out_p < NUM_PORTS; out_p = out_p + 1) begin
            out_flit[out_p] = {FLIT_WIDTH{1'b0}};
            out_valid[out_p] = 1'b0;
            
            for (in_p = 0; in_p < NUM_PORTS; in_p = in_p + 1) begin
                if (grant[out_p][in_p]) begin
                    out_flit[out_p] = fifo_out[in_p];
                    out_valid[out_p] = 1'b1;
                end
            end
        end
    end
    
    //--------------------------------------------------------------------------
    // Output Assignments
    //--------------------------------------------------------------------------
    
    // North output
    assign north_out_data   = out_flit[PORT_NORTH][DATA_WIDTH-1:0];
    assign north_out_dest_x = out_flit[PORT_NORTH][DATA_WIDTH +: COORD_BITS];
    assign north_out_dest_y = out_flit[PORT_NORTH][DATA_WIDTH + COORD_BITS +: COORD_BITS];
    assign north_out_valid  = out_valid[PORT_NORTH];
    
    // South output
    assign south_out_data   = out_flit[PORT_SOUTH][DATA_WIDTH-1:0];
    assign south_out_dest_x = out_flit[PORT_SOUTH][DATA_WIDTH +: COORD_BITS];
    assign south_out_dest_y = out_flit[PORT_SOUTH][DATA_WIDTH + COORD_BITS +: COORD_BITS];
    assign south_out_valid  = out_valid[PORT_SOUTH];
    
    // East output
    assign east_out_data   = out_flit[PORT_EAST][DATA_WIDTH-1:0];
    assign east_out_dest_x = out_flit[PORT_EAST][DATA_WIDTH +: COORD_BITS];
    assign east_out_dest_y = out_flit[PORT_EAST][DATA_WIDTH + COORD_BITS +: COORD_BITS];
    assign east_out_valid  = out_valid[PORT_EAST];
    
    // West output
    assign west_out_data   = out_flit[PORT_WEST][DATA_WIDTH-1:0];
    assign west_out_dest_x = out_flit[PORT_WEST][DATA_WIDTH +: COORD_BITS];
    assign west_out_dest_y = out_flit[PORT_WEST][DATA_WIDTH + COORD_BITS +: COORD_BITS];
    assign west_out_valid  = out_valid[PORT_WEST];
    
    // Local output
    assign local_out_data   = out_flit[PORT_LOCAL][DATA_WIDTH-1:0];
    assign local_out_dest_x = out_flit[PORT_LOCAL][DATA_WIDTH +: COORD_BITS];
    assign local_out_dest_y = out_flit[PORT_LOCAL][DATA_WIDTH + COORD_BITS +: COORD_BITS];
    assign local_out_valid  = out_valid[PORT_LOCAL];

endmodule

//==============================================================================
// Synchronous FIFO
//==============================================================================

module sync_fifo #(
    parameter WIDTH = 256,
    parameter DEPTH = 4
)(
    input  wire             clk,
    input  wire             rst_n,
    input  wire             wr_en,
    input  wire [WIDTH-1:0] wr_data,
    input  wire             rd_en,
    output wire [WIDTH-1:0] rd_data,
    output wire             empty,
    output wire             full
);

    localparam ADDR_BITS = $clog2(DEPTH);
    
    reg [WIDTH-1:0] mem [0:DEPTH-1];
    reg [ADDR_BITS:0] wr_ptr, rd_ptr;
    
    wire [ADDR_BITS-1:0] wr_addr = wr_ptr[ADDR_BITS-1:0];
    wire [ADDR_BITS-1:0] rd_addr = rd_ptr[ADDR_BITS-1:0];
    
    assign empty = (wr_ptr == rd_ptr);
    assign full  = (wr_ptr[ADDR_BITS] != rd_ptr[ADDR_BITS]) && 
                   (wr_ptr[ADDR_BITS-1:0] == rd_ptr[ADDR_BITS-1:0]);
    
    assign rd_data = mem[rd_addr];
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= 0;
            rd_ptr <= 0;
        end else begin
            if (wr_en && !full) begin
                mem[wr_addr] <= wr_data;
                wr_ptr <= wr_ptr + 1;
            end
            if (rd_en && !empty) begin
                rd_ptr <= rd_ptr + 1;
            end
        end
    end

endmodule

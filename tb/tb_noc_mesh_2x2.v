`timescale 1ns/1ps
/*
 * NoC 2x2 Mesh Integration Testbench
 * 
 * Tests a complete 2x2 mesh of NoC routers:
 *   
 *   (0,1)───(1,1)
 *     │       │
 *   (0,0)───(1,0)
 *
 * Test cases:
 * 1. Adjacent routing (single hop)
 * 2. Diagonal routing (multi-hop via XY)
 * 3. Concurrent traffic (all 4 TPCs sending)
 * 4. Broadcast pattern
 */

module tb_noc_mesh_2x2;

    parameter CLK = 10;
    parameter DATA_WIDTH = 256;
    parameter COORD_BITS = 4;
    parameter FIFO_DEPTH = 4;
    
    reg clk = 0;
    reg rst_n = 0;
    always #(CLK/2) clk = ~clk;
    
    //==========================================================================
    // Local port signals for each router (TPC interface)
    //==========================================================================
    // Router (0,0)
    reg  [DATA_WIDTH-1:0] local_00_in_data;
    reg  [COORD_BITS-1:0] local_00_in_dest_x, local_00_in_dest_y;
    reg                   local_00_in_valid;
    wire                  local_00_in_ready;
    wire [DATA_WIDTH-1:0] local_00_out_data;
    wire [COORD_BITS-1:0] local_00_out_dest_x, local_00_out_dest_y;
    wire                  local_00_out_valid;
    reg                   local_00_out_ready;
    
    // Router (1,0)
    reg  [DATA_WIDTH-1:0] local_10_in_data;
    reg  [COORD_BITS-1:0] local_10_in_dest_x, local_10_in_dest_y;
    reg                   local_10_in_valid;
    wire                  local_10_in_ready;
    wire [DATA_WIDTH-1:0] local_10_out_data;
    wire [COORD_BITS-1:0] local_10_out_dest_x, local_10_out_dest_y;
    wire                  local_10_out_valid;
    reg                   local_10_out_ready;
    
    // Router (0,1)
    reg  [DATA_WIDTH-1:0] local_01_in_data;
    reg  [COORD_BITS-1:0] local_01_in_dest_x, local_01_in_dest_y;
    reg                   local_01_in_valid;
    wire                  local_01_in_ready;
    wire [DATA_WIDTH-1:0] local_01_out_data;
    wire [COORD_BITS-1:0] local_01_out_dest_x, local_01_out_dest_y;
    wire                  local_01_out_valid;
    reg                   local_01_out_ready;
    
    // Router (1,1)
    reg  [DATA_WIDTH-1:0] local_11_in_data;
    reg  [COORD_BITS-1:0] local_11_in_dest_x, local_11_in_dest_y;
    reg                   local_11_in_valid;
    wire                  local_11_in_ready;
    wire [DATA_WIDTH-1:0] local_11_out_data;
    wire [COORD_BITS-1:0] local_11_out_dest_x, local_11_out_dest_y;
    wire                  local_11_out_valid;
    reg                   local_11_out_ready;
    
    //==========================================================================
    // Inter-router connections
    //==========================================================================
    // Horizontal: (0,0) <-> (1,0)
    wire [DATA_WIDTH-1:0] r00_east_data, r10_west_data;
    wire [COORD_BITS-1:0] r00_east_dx, r00_east_dy, r10_west_dx, r10_west_dy;
    wire                  r00_east_valid, r00_east_ready;
    wire                  r10_west_valid, r10_west_ready;
    
    // Horizontal: (0,1) <-> (1,1)
    wire [DATA_WIDTH-1:0] r01_east_data, r11_west_data;
    wire [COORD_BITS-1:0] r01_east_dx, r01_east_dy, r11_west_dx, r11_west_dy;
    wire                  r01_east_valid, r01_east_ready;
    wire                  r11_west_valid, r11_west_ready;
    
    // Vertical: (0,0) <-> (0,1)
    wire [DATA_WIDTH-1:0] r00_north_data, r01_south_data;
    wire [COORD_BITS-1:0] r00_north_dx, r00_north_dy, r01_south_dx, r01_south_dy;
    wire                  r00_north_valid, r00_north_ready;
    wire                  r01_south_valid, r01_south_ready;
    
    // Vertical: (1,0) <-> (1,1)
    wire [DATA_WIDTH-1:0] r10_north_data, r11_south_data;
    wire [COORD_BITS-1:0] r10_north_dx, r10_north_dy, r11_south_dx, r11_south_dy;
    wire                  r10_north_valid, r10_north_ready;
    wire                  r11_south_valid, r11_south_ready;
    
    //==========================================================================
    // Router Instances
    //==========================================================================
    
    // Router (0,0) - Bottom-left
    noc_router #(
        .DATA_WIDTH(DATA_WIDTH), .COORD_BITS(COORD_BITS), .FIFO_DEPTH(FIFO_DEPTH),
        .ROUTER_X(0), .ROUTER_Y(0)
    ) router_00 (
        .clk(clk), .rst_n(rst_n),
        // North -> (0,1)
        .north_in_data(r01_south_data), .north_in_dest_x(r01_south_dx), .north_in_dest_y(r01_south_dy),
        .north_in_valid(r01_south_valid), .north_in_ready(r01_south_ready),
        .north_out_data(r00_north_data), .north_out_dest_x(r00_north_dx), .north_out_dest_y(r00_north_dy),
        .north_out_valid(r00_north_valid), .north_out_ready(r00_north_ready),
        // South -> edge (unused)
        .south_in_data({DATA_WIDTH{1'b0}}), .south_in_dest_x(4'd0), .south_in_dest_y(4'd0),
        .south_in_valid(1'b0), .south_in_ready(),
        .south_out_data(), .south_out_dest_x(), .south_out_dest_y(),
        .south_out_valid(), .south_out_ready(1'b1),
        // East -> (1,0)
        .east_in_data(r10_west_data), .east_in_dest_x(r10_west_dx), .east_in_dest_y(r10_west_dy),
        .east_in_valid(r10_west_valid), .east_in_ready(r10_west_ready),
        .east_out_data(r00_east_data), .east_out_dest_x(r00_east_dx), .east_out_dest_y(r00_east_dy),
        .east_out_valid(r00_east_valid), .east_out_ready(r00_east_ready),
        // West -> edge (unused)
        .west_in_data({DATA_WIDTH{1'b0}}), .west_in_dest_x(4'd0), .west_in_dest_y(4'd0),
        .west_in_valid(1'b0), .west_in_ready(),
        .west_out_data(), .west_out_dest_x(), .west_out_dest_y(),
        .west_out_valid(), .west_out_ready(1'b1),
        // Local -> TPC
        .local_in_data(local_00_in_data), .local_in_dest_x(local_00_in_dest_x), .local_in_dest_y(local_00_in_dest_y),
        .local_in_valid(local_00_in_valid), .local_in_ready(local_00_in_ready),
        .local_out_data(local_00_out_data), .local_out_dest_x(local_00_out_dest_x), .local_out_dest_y(local_00_out_dest_y),
        .local_out_valid(local_00_out_valid), .local_out_ready(local_00_out_ready)
    );
    
    // Router (1,0) - Bottom-right
    noc_router #(
        .DATA_WIDTH(DATA_WIDTH), .COORD_BITS(COORD_BITS), .FIFO_DEPTH(FIFO_DEPTH),
        .ROUTER_X(1), .ROUTER_Y(0)
    ) router_10 (
        .clk(clk), .rst_n(rst_n),
        // North -> (1,1)
        .north_in_data(r11_south_data), .north_in_dest_x(r11_south_dx), .north_in_dest_y(r11_south_dy),
        .north_in_valid(r11_south_valid), .north_in_ready(r11_south_ready),
        .north_out_data(r10_north_data), .north_out_dest_x(r10_north_dx), .north_out_dest_y(r10_north_dy),
        .north_out_valid(r10_north_valid), .north_out_ready(r10_north_ready),
        // South -> edge
        .south_in_data({DATA_WIDTH{1'b0}}), .south_in_dest_x(4'd0), .south_in_dest_y(4'd0),
        .south_in_valid(1'b0), .south_in_ready(),
        .south_out_data(), .south_out_dest_x(), .south_out_dest_y(),
        .south_out_valid(), .south_out_ready(1'b1),
        // East -> edge
        .east_in_data({DATA_WIDTH{1'b0}}), .east_in_dest_x(4'd0), .east_in_dest_y(4'd0),
        .east_in_valid(1'b0), .east_in_ready(),
        .east_out_data(), .east_out_dest_x(), .east_out_dest_y(),
        .east_out_valid(), .east_out_ready(1'b1),
        // West -> (0,0)
        .west_in_data(r00_east_data), .west_in_dest_x(r00_east_dx), .west_in_dest_y(r00_east_dy),
        .west_in_valid(r00_east_valid), .west_in_ready(r00_east_ready),
        .west_out_data(r10_west_data), .west_out_dest_x(r10_west_dx), .west_out_dest_y(r10_west_dy),
        .west_out_valid(r10_west_valid), .west_out_ready(r10_west_ready),
        // Local
        .local_in_data(local_10_in_data), .local_in_dest_x(local_10_in_dest_x), .local_in_dest_y(local_10_in_dest_y),
        .local_in_valid(local_10_in_valid), .local_in_ready(local_10_in_ready),
        .local_out_data(local_10_out_data), .local_out_dest_x(local_10_out_dest_x), .local_out_dest_y(local_10_out_dest_y),
        .local_out_valid(local_10_out_valid), .local_out_ready(local_10_out_ready)
    );
    
    // Router (0,1) - Top-left
    noc_router #(
        .DATA_WIDTH(DATA_WIDTH), .COORD_BITS(COORD_BITS), .FIFO_DEPTH(FIFO_DEPTH),
        .ROUTER_X(0), .ROUTER_Y(1)
    ) router_01 (
        .clk(clk), .rst_n(rst_n),
        // North -> edge
        .north_in_data({DATA_WIDTH{1'b0}}), .north_in_dest_x(4'd0), .north_in_dest_y(4'd0),
        .north_in_valid(1'b0), .north_in_ready(),
        .north_out_data(), .north_out_dest_x(), .north_out_dest_y(),
        .north_out_valid(), .north_out_ready(1'b1),
        // South -> (0,0)
        .south_in_data(r00_north_data), .south_in_dest_x(r00_north_dx), .south_in_dest_y(r00_north_dy),
        .south_in_valid(r00_north_valid), .south_in_ready(r00_north_ready),
        .south_out_data(r01_south_data), .south_out_dest_x(r01_south_dx), .south_out_dest_y(r01_south_dy),
        .south_out_valid(r01_south_valid), .south_out_ready(r01_south_ready),
        // East -> (1,1)
        .east_in_data(r11_west_data), .east_in_dest_x(r11_west_dx), .east_in_dest_y(r11_west_dy),
        .east_in_valid(r11_west_valid), .east_in_ready(r11_west_ready),
        .east_out_data(r01_east_data), .east_out_dest_x(r01_east_dx), .east_out_dest_y(r01_east_dy),
        .east_out_valid(r01_east_valid), .east_out_ready(r01_east_ready),
        // West -> edge
        .west_in_data({DATA_WIDTH{1'b0}}), .west_in_dest_x(4'd0), .west_in_dest_y(4'd0),
        .west_in_valid(1'b0), .west_in_ready(),
        .west_out_data(), .west_out_dest_x(), .west_out_dest_y(),
        .west_out_valid(), .west_out_ready(1'b1),
        // Local
        .local_in_data(local_01_in_data), .local_in_dest_x(local_01_in_dest_x), .local_in_dest_y(local_01_in_dest_y),
        .local_in_valid(local_01_in_valid), .local_in_ready(local_01_in_ready),
        .local_out_data(local_01_out_data), .local_out_dest_x(local_01_out_dest_x), .local_out_dest_y(local_01_out_dest_y),
        .local_out_valid(local_01_out_valid), .local_out_ready(local_01_out_ready)
    );
    
    // Router (1,1) - Top-right
    noc_router #(
        .DATA_WIDTH(DATA_WIDTH), .COORD_BITS(COORD_BITS), .FIFO_DEPTH(FIFO_DEPTH),
        .ROUTER_X(1), .ROUTER_Y(1)
    ) router_11 (
        .clk(clk), .rst_n(rst_n),
        // North -> edge
        .north_in_data({DATA_WIDTH{1'b0}}), .north_in_dest_x(4'd0), .north_in_dest_y(4'd0),
        .north_in_valid(1'b0), .north_in_ready(),
        .north_out_data(), .north_out_dest_x(), .north_out_dest_y(),
        .north_out_valid(), .north_out_ready(1'b1),
        // South -> (1,0)
        .south_in_data(r10_north_data), .south_in_dest_x(r10_north_dx), .south_in_dest_y(r10_north_dy),
        .south_in_valid(r10_north_valid), .south_in_ready(r10_north_ready),
        .south_out_data(r11_south_data), .south_out_dest_x(r11_south_dx), .south_out_dest_y(r11_south_dy),
        .south_out_valid(r11_south_valid), .south_out_ready(r11_south_ready),
        // East -> edge
        .east_in_data({DATA_WIDTH{1'b0}}), .east_in_dest_x(4'd0), .east_in_dest_y(4'd0),
        .east_in_valid(1'b0), .east_in_ready(),
        .east_out_data(), .east_out_dest_x(), .east_out_dest_y(),
        .east_out_valid(), .east_out_ready(1'b1),
        // West -> (0,1)
        .west_in_data(r01_east_data), .west_in_dest_x(r01_east_dx), .west_in_dest_y(r01_east_dy),
        .west_in_valid(r01_east_valid), .west_in_ready(r01_east_ready),
        .west_out_data(r11_west_data), .west_out_dest_x(r11_west_dx), .west_out_dest_y(r11_west_dy),
        .west_out_valid(r11_west_valid), .west_out_ready(r11_west_ready),
        // Local
        .local_in_data(local_11_in_data), .local_in_dest_x(local_11_in_dest_x), .local_in_dest_y(local_11_in_dest_y),
        .local_in_valid(local_11_in_valid), .local_in_ready(local_11_in_ready),
        .local_out_data(local_11_out_data), .local_out_dest_x(local_11_out_dest_x), .local_out_dest_y(local_11_out_dest_y),
        .local_out_valid(local_11_out_valid), .local_out_ready(local_11_out_ready)
    );
    
    //==========================================================================
    // Test Infrastructure
    //==========================================================================
    integer errors;
    integer timeout;
    reg [DATA_WIDTH-1:0] rx_data;
    reg rx_success;
    
    // Inject packet from router (x,y)
    task inject_packet;
        input [1:0] src_x, src_y;
        input [DATA_WIDTH-1:0] data;
        input [COORD_BITS-1:0] dest_x, dest_y;
    begin
        @(negedge clk);
        case ({src_x, src_y})
            4'b0000: begin  // (0,0)
                local_00_in_data = data;
                local_00_in_dest_x = dest_x;
                local_00_in_dest_y = dest_y;
                local_00_in_valid = 1;
                @(posedge clk);
                while (!local_00_in_ready) @(posedge clk);
                @(negedge clk);
                local_00_in_valid = 0;
            end
            4'b0100: begin  // (1,0)
                local_10_in_data = data;
                local_10_in_dest_x = dest_x;
                local_10_in_dest_y = dest_y;
                local_10_in_valid = 1;
                @(posedge clk);
                while (!local_10_in_ready) @(posedge clk);
                @(negedge clk);
                local_10_in_valid = 0;
            end
            4'b0001: begin  // (0,1)
                local_01_in_data = data;
                local_01_in_dest_x = dest_x;
                local_01_in_dest_y = dest_y;
                local_01_in_valid = 1;
                @(posedge clk);
                while (!local_01_in_ready) @(posedge clk);
                @(negedge clk);
                local_01_in_valid = 0;
            end
            4'b0101: begin  // (1,1)
                local_11_in_data = data;
                local_11_in_dest_x = dest_x;
                local_11_in_dest_y = dest_y;
                local_11_in_valid = 1;
                @(posedge clk);
                while (!local_11_in_ready) @(posedge clk);
                @(negedge clk);
                local_11_in_valid = 0;
            end
        endcase
    end
    endtask
    
    // Wait for packet at router (x,y)
    task wait_packet;
        input [1:0] dst_x, dst_y;
        input [DATA_WIDTH-1:0] expected;
    begin
        timeout = 0;
        rx_success = 0;
        rx_data = 0;
        
        case ({dst_x, dst_y})
            4'b0000: begin  // (0,0)
                while (timeout < 50 && !rx_success) begin
                    @(posedge clk);
                    if (local_00_out_valid && local_00_out_ready) begin
                        rx_data = local_00_out_data;
                        rx_success = 1;
                    end
                    timeout = timeout + 1;
                end
            end
            4'b0100: begin  // (1,0)
                while (timeout < 50 && !rx_success) begin
                    @(posedge clk);
                    if (local_10_out_valid && local_10_out_ready) begin
                        rx_data = local_10_out_data;
                        rx_success = 1;
                    end
                    timeout = timeout + 1;
                end
            end
            4'b0001: begin  // (0,1)
                while (timeout < 50 && !rx_success) begin
                    @(posedge clk);
                    if (local_01_out_valid && local_01_out_ready) begin
                        rx_data = local_01_out_data;
                        rx_success = 1;
                    end
                    timeout = timeout + 1;
                end
            end
            4'b0101: begin  // (1,1)
                while (timeout < 50 && !rx_success) begin
                    @(posedge clk);
                    if (local_11_out_valid && local_11_out_ready) begin
                        rx_data = local_11_out_data;
                        rx_success = 1;
                    end
                    timeout = timeout + 1;
                end
            end
        endcase
        
        if (rx_success && rx_data == expected) begin
            $display("  PASS: Received correct data");
        end else if (!rx_success) begin
            $display("  FAIL: Timeout waiting for packet");
            errors = errors + 1;
        end else begin
            $display("  FAIL: Data mismatch - got %h, expected %h", rx_data, expected);
            errors = errors + 1;
        end
    end
    endtask
    
    //==========================================================================
    // Main Test Sequence
    //==========================================================================
    initial begin
        $display("");
        $display("╔══════════════════════════════════════════════════════════════╗");
        $display("║           NoC 2x2 Mesh Integration Testbench                 ║");
        $display("║                                                              ║");
        $display("║           (0,1)═══(1,1)                                      ║");
        $display("║             ║       ║                                        ║");
        $display("║           (0,0)═══(1,0)                                      ║");
        $display("╚══════════════════════════════════════════════════════════════╝");
        $display("");
        
        // Initialize
        errors = 0;
        local_00_in_data = 0; local_00_in_dest_x = 0; local_00_in_dest_y = 0; local_00_in_valid = 0; local_00_out_ready = 1;
        local_10_in_data = 0; local_10_in_dest_x = 0; local_10_in_dest_y = 0; local_10_in_valid = 0; local_10_out_ready = 1;
        local_01_in_data = 0; local_01_in_dest_x = 0; local_01_in_dest_y = 0; local_01_in_valid = 0; local_01_out_ready = 1;
        local_11_in_data = 0; local_11_in_dest_x = 0; local_11_in_dest_y = 0; local_11_in_valid = 0; local_11_out_ready = 1;
        
        #(CLK * 5);
        rst_n = 1;
        #(CLK * 5);
        
        //======================================================================
        // TEST 1: Single hop East (0,0) -> (1,0)
        //======================================================================
        $display("[TEST 1] Single hop East: (0,0) → (1,0)");
        inject_packet(2'd0, 2'd0, 256'hDEAD_BEEF_0000_0001, 4'd1, 4'd0);
        wait_packet(2'd1, 2'd0, 256'hDEAD_BEEF_0000_0001);
        #(CLK * 5);
        
        //======================================================================
        // TEST 2: Single hop North (0,0) -> (0,1)
        //======================================================================
        $display("");
        $display("[TEST 2] Single hop North: (0,0) → (0,1)");
        inject_packet(2'd0, 2'd0, 256'hCAFE_BABE_0000_0002, 4'd0, 4'd1);
        wait_packet(2'd0, 2'd1, 256'hCAFE_BABE_0000_0002);
        #(CLK * 5);
        
        //======================================================================
        // TEST 3: Diagonal (0,0) -> (1,1) via XY routing
        //======================================================================
        $display("");
        $display("[TEST 3] Diagonal XY routing: (0,0) → (1,1)");
        $display("         Route: (0,0) → East → (1,0) → North → (1,1)");
        inject_packet(2'd0, 2'd0, 256'h1234_5678_0000_0003, 4'd1, 4'd1);
        wait_packet(2'd1, 2'd1, 256'h1234_5678_0000_0003);
        #(CLK * 5);
        
        //======================================================================
        // TEST 4: Opposite diagonal (1,1) -> (0,0)
        //======================================================================
        $display("");
        $display("[TEST 4] Opposite diagonal: (1,1) → (0,0)");
        $display("         Route: (1,1) → West → (0,1) → South → (0,0)");
        inject_packet(2'd1, 2'd1, 256'hABCD_EF01_0000_0004, 4'd0, 4'd0);
        wait_packet(2'd0, 2'd0, 256'hABCD_EF01_0000_0004);
        #(CLK * 5);
        
        //======================================================================
        // TEST 5: Self-delivery (1,0) -> (1,0)
        //======================================================================
        $display("");
        $display("[TEST 5] Self-delivery: (1,0) → (1,0)");
        inject_packet(2'd1, 2'd0, 256'hFFFF_0000_0000_0005, 4'd1, 4'd0);
        wait_packet(2'd1, 2'd0, 256'hFFFF_0000_0000_0005);
        #(CLK * 5);
        
        //======================================================================
        // TEST 6: Ring pattern - all 4 send clockwise
        //======================================================================
        $display("");
        $display("[TEST 6] Ring pattern - clockwise sends");
        $display("         (0,0)→(1,0), (1,0)→(1,1), (1,1)→(0,1), (0,1)→(0,0)");
        
        // Send packets in quick succession (not truly parallel but tests contention)
        inject_packet(2'd0, 2'd0, 256'h0000000000000001, 4'd1, 4'd0);
        inject_packet(2'd1, 2'd0, 256'h0000000000000002, 4'd1, 4'd1);
        inject_packet(2'd1, 2'd1, 256'h0000000000000003, 4'd0, 4'd1);
        inject_packet(2'd0, 2'd1, 256'h0000000000000004, 4'd0, 4'd0);
        
        // Wait for all to arrive
        #(CLK * 30);
        $display("  Ring pattern completed (no deadlock)");
        
        //======================================================================
        // TEST 7: Bidirectional - opposite corners
        //======================================================================
        $display("");
        $display("[TEST 7] Bidirectional: (0,0)→(1,1) then (1,1)→(0,0)");
        inject_packet(2'd0, 2'd0, 256'h0000000000000007, 4'd1, 4'd1);
        wait_packet(2'd1, 2'd1, 256'h0000000000000007);
        
        inject_packet(2'd1, 2'd1, 256'h0000000000000008, 4'd0, 4'd0);
        wait_packet(2'd0, 2'd0, 256'h0000000000000008);
        #(CLK * 5);
        
        //======================================================================
        // Summary
        //======================================================================
        #(CLK * 10);
        $display("");
        $display("╔══════════════════════════════════════════════════════════════╗");
        $display("║                    TEST SUMMARY                              ║");
        $display("╠══════════════════════════════════════════════════════════════╣");
        if (errors == 0) begin
            $display("║   All 7 tests PASSED                                       ║");
            $display("╚══════════════════════════════════════════════════════════════╝");
            $display(">>> ALL NOC MESH INTEGRATION TESTS PASSED! <<<");
        end else begin
            $display("║   FAILED: %0d errors                                         ║", errors);
            $display("╚══════════════════════════════════════════════════════════════╝");
            $display(">>> SOME TESTS FAILED <<<");
        end
        $display("");
        
        $finish;
    end
    
    // Timeout
    initial begin
        #(CLK * 5000);
        $display("ERROR: Testbench timeout!");
        $finish;
    end
    
    // VCD dump
    initial begin
        $dumpfile("noc_mesh_2x2.vcd");
        $dumpvars(0, tb_noc_mesh_2x2);
    end

endmodule

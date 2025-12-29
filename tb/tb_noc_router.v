//==============================================================================
// NoC Router Unit Testbench
//==============================================================================
`timescale 1ns / 1ps

module tb_noc_router;

    parameter CLK = 10;
    parameter DATA_WIDTH = 64;
    parameter COORD_BITS = 4;
    parameter FIFO_DEPTH = 4;

    reg clk = 0;
    reg rst_n = 0;
    always #(CLK/2) clk = ~clk;

    localparam ROUTER_X = 1;
    localparam ROUTER_Y = 1;

    // North port
    reg [DATA_WIDTH-1:0] north_in_data = 0;
    reg [COORD_BITS-1:0] north_in_dest_x = 0, north_in_dest_y = 0;
    reg north_in_valid = 0;
    wire north_in_ready;
    wire [DATA_WIDTH-1:0] north_out_data;
    wire [COORD_BITS-1:0] north_out_dest_x, north_out_dest_y;
    wire north_out_valid;
    reg north_out_ready = 1;

    // South port
    reg [DATA_WIDTH-1:0] south_in_data = 0;
    reg [COORD_BITS-1:0] south_in_dest_x = 0, south_in_dest_y = 0;
    reg south_in_valid = 0;
    wire south_in_ready;
    wire [DATA_WIDTH-1:0] south_out_data;
    wire [COORD_BITS-1:0] south_out_dest_x, south_out_dest_y;
    wire south_out_valid;
    reg south_out_ready = 1;

    // East port
    reg [DATA_WIDTH-1:0] east_in_data = 0;
    reg [COORD_BITS-1:0] east_in_dest_x = 0, east_in_dest_y = 0;
    reg east_in_valid = 0;
    wire east_in_ready;
    wire [DATA_WIDTH-1:0] east_out_data;
    wire [COORD_BITS-1:0] east_out_dest_x, east_out_dest_y;
    wire east_out_valid;
    reg east_out_ready = 1;

    // West port
    reg [DATA_WIDTH-1:0] west_in_data = 0;
    reg [COORD_BITS-1:0] west_in_dest_x = 0, west_in_dest_y = 0;
    reg west_in_valid = 0;
    wire west_in_ready;
    wire [DATA_WIDTH-1:0] west_out_data;
    wire [COORD_BITS-1:0] west_out_dest_x, west_out_dest_y;
    wire west_out_valid;
    reg west_out_ready = 1;

    // Local port
    reg [DATA_WIDTH-1:0] local_in_data = 0;
    reg [COORD_BITS-1:0] local_in_dest_x = 0, local_in_dest_y = 0;
    reg local_in_valid = 0;
    wire local_in_ready;
    wire [DATA_WIDTH-1:0] local_out_data;
    wire [COORD_BITS-1:0] local_out_dest_x, local_out_dest_y;
    wire local_out_valid;
    reg local_out_ready = 1;

    // DUT
    noc_router #(
        .DATA_WIDTH(DATA_WIDTH),
        .COORD_BITS(COORD_BITS),
        .FIFO_DEPTH(FIFO_DEPTH),
        .ROUTER_X(ROUTER_X),
        .ROUTER_Y(ROUTER_Y)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .north_in_data(north_in_data), .north_in_dest_x(north_in_dest_x), .north_in_dest_y(north_in_dest_y),
        .north_in_valid(north_in_valid), .north_in_ready(north_in_ready),
        .north_out_data(north_out_data), .north_out_dest_x(north_out_dest_x), .north_out_dest_y(north_out_dest_y),
        .north_out_valid(north_out_valid), .north_out_ready(north_out_ready),
        .south_in_data(south_in_data), .south_in_dest_x(south_in_dest_x), .south_in_dest_y(south_in_dest_y),
        .south_in_valid(south_in_valid), .south_in_ready(south_in_ready),
        .south_out_data(south_out_data), .south_out_dest_x(south_out_dest_x), .south_out_dest_y(south_out_dest_y),
        .south_out_valid(south_out_valid), .south_out_ready(south_out_ready),
        .east_in_data(east_in_data), .east_in_dest_x(east_in_dest_x), .east_in_dest_y(east_in_dest_y),
        .east_in_valid(east_in_valid), .east_in_ready(east_in_ready),
        .east_out_data(east_out_data), .east_out_dest_x(east_out_dest_x), .east_out_dest_y(east_out_dest_y),
        .east_out_valid(east_out_valid), .east_out_ready(east_out_ready),
        .west_in_data(west_in_data), .west_in_dest_x(west_in_dest_x), .west_in_dest_y(west_in_dest_y),
        .west_in_valid(west_in_valid), .west_in_ready(west_in_ready),
        .west_out_data(west_out_data), .west_out_dest_x(west_out_dest_x), .west_out_dest_y(west_out_dest_y),
        .west_out_valid(west_out_valid), .west_out_ready(west_out_ready),
        .local_in_data(local_in_data), .local_in_dest_x(local_in_dest_x), .local_in_dest_y(local_in_dest_y),
        .local_in_valid(local_in_valid), .local_in_ready(local_in_ready),
        .local_out_data(local_out_data), .local_out_dest_x(local_out_dest_x), .local_out_dest_y(local_out_dest_y),
        .local_out_valid(local_out_valid), .local_out_ready(local_out_ready)
    );

    integer errors = 0;
    integer timeout;
    reg [DATA_WIDTH-1:0] rx_data;
    reg rx_success;

    // Inject packet tasks - use old-style task declaration
    task inject_local;
        input [DATA_WIDTH-1:0] data;
        input [COORD_BITS-1:0] dest_x;
        input [COORD_BITS-1:0] dest_y;
        begin
            @(negedge clk);
            local_in_data = data;
            local_in_dest_x = dest_x;
            local_in_dest_y = dest_y;
            local_in_valid = 1;
            @(posedge clk);
            while (!local_in_ready) @(posedge clk);
            @(negedge clk);
            local_in_valid = 0;
        end
    endtask

    task inject_west;
        input [DATA_WIDTH-1:0] data;
        input [COORD_BITS-1:0] dest_x;
        input [COORD_BITS-1:0] dest_y;
        begin
            @(negedge clk);
            west_in_data = data;
            west_in_dest_x = dest_x;
            west_in_dest_y = dest_y;
            west_in_valid = 1;
            @(posedge clk);
            while (!west_in_ready) @(posedge clk);
            @(negedge clk);
            west_in_valid = 0;
        end
    endtask

    task inject_south;
        input [DATA_WIDTH-1:0] data;
        input [COORD_BITS-1:0] dest_x;
        input [COORD_BITS-1:0] dest_y;
        begin
            @(negedge clk);
            south_in_data = data;
            south_in_dest_x = dest_x;
            south_in_dest_y = dest_y;
            south_in_valid = 1;
            @(posedge clk);
            while (!south_in_ready) @(posedge clk);
            @(negedge clk);
            south_in_valid = 0;
        end
    endtask

    // Wait tasks - no parameters, use globals
    task wait_east_out;
        begin
            timeout = 0; rx_success = 0;
            while (timeout < 20 && !rx_success) begin
                @(posedge clk);
                if (east_out_valid && east_out_ready) begin
                    rx_data = east_out_data;
                    rx_success = 1;
                end
                timeout = timeout + 1;
            end
        end
    endtask

    task wait_north_out;
        begin
            timeout = 0; rx_success = 0;
            while (timeout < 20 && !rx_success) begin
                @(posedge clk);
                if (north_out_valid && north_out_ready) begin
                    rx_data = north_out_data;
                    rx_success = 1;
                end
                timeout = timeout + 1;
            end
        end
    endtask

    task wait_local_out;
        begin
            timeout = 0; rx_success = 0;
            while (timeout < 20 && !rx_success) begin
                @(posedge clk);
                if (local_out_valid && local_out_ready) begin
                    rx_data = local_out_data;
                    rx_success = 1;
                end
                timeout = timeout + 1;
            end
        end
    endtask

    task wait_west_out;
        begin
            timeout = 0; rx_success = 0;
            while (timeout < 20 && !rx_success) begin
                @(posedge clk);
                if (west_out_valid && west_out_ready) begin
                    rx_data = west_out_data;
                    rx_success = 1;
                end
                timeout = timeout + 1;
            end
        end
    endtask

    task wait_south_out;
        begin
            timeout = 0; rx_success = 0;
            while (timeout < 20 && !rx_success) begin
                @(posedge clk);
                if (south_out_valid && south_out_ready) begin
                    rx_data = south_out_data;
                    rx_success = 1;
                end
                timeout = timeout + 1;
            end
        end
    endtask

    // Test sequence
    initial begin
        $display("");
        $display("╔════════════════════════════════════════════════════════════╗");
        $display("║           NoC Router Unit Testbench                        ║");
        $display("║           Router Position: (%0d,%0d)                          ║", ROUTER_X, ROUTER_Y);
        $display("╚════════════════════════════════════════════════════════════╝");

        #(CLK * 5); rst_n = 1; #(CLK * 5);

        // TEST 1: Local -> East
        $display("");
        $display("[TEST 1] Local -> East (dest_x=2 > router_x=1)");
        inject_local(64'hDEADBEEF_CAFEBABE, 4'd2, 4'd1);
        wait_east_out;
        if (rx_success && rx_data == 64'hDEADBEEF_CAFEBABE) $display("  PASS");
        else begin $display("  FAIL: success=%b", rx_success); errors = errors + 1; end
        #(CLK * 5);

        // TEST 2: Local -> West
        $display("");
        $display("[TEST 2] Local -> West (dest_x=0 < router_x=1)");
        inject_local(64'h12345678_9ABCDEF0, 4'd0, 4'd1);
        wait_west_out;
        if (rx_success && rx_data == 64'h12345678_9ABCDEF0) $display("  PASS");
        else begin $display("  FAIL: success=%b", rx_success); errors = errors + 1; end
        #(CLK * 5);

        // TEST 3: Local -> North
        $display("");
        $display("[TEST 3] Local -> North (dest_y=2 > router_y=1)");
        inject_local(64'hAAAABBBB_CCCCDDDD, 4'd1, 4'd2);
        wait_north_out;
        if (rx_success && rx_data == 64'hAAAABBBB_CCCCDDDD) $display("  PASS");
        else begin $display("  FAIL: success=%b", rx_success); errors = errors + 1; end
        #(CLK * 5);

        // TEST 4: Local -> South
        $display("");
        $display("[TEST 4] Local -> South (dest_y=0 < router_y=1)");
        inject_local(64'h11112222_33334444, 4'd1, 4'd0);
        wait_south_out;
        if (rx_success && rx_data == 64'h11112222_33334444) $display("  PASS");
        else begin $display("  FAIL: success=%b", rx_success); errors = errors + 1; end
        #(CLK * 5);

        // TEST 5: Local -> Local (self delivery)
        $display("");
        $display("[TEST 5] Local -> Local (dest = this router)");
        inject_local(64'hFEDCBA98_76543210, 4'd1, 4'd1);
        wait_local_out;
        if (rx_success && rx_data == 64'hFEDCBA98_76543210) $display("  PASS");
        else begin $display("  FAIL: success=%b", rx_success); errors = errors + 1; end
        #(CLK * 5);

        // TEST 6: Transit West -> East
        $display("");
        $display("[TEST 6] Transit: West -> East");
        inject_west(64'h55556666_77778888, 4'd3, 4'd1);
        wait_east_out;
        if (rx_success && rx_data == 64'h55556666_77778888) $display("  PASS");
        else begin $display("  FAIL: success=%b", rx_success); errors = errors + 1; end
        #(CLK * 5);

        // TEST 7: X-Y Routing (South -> East for X first)
        $display("");
        $display("[TEST 7] X-Y Routing: South input -> East output");
        inject_south(64'hABCDABCD_12341234, 4'd2, 4'd2);
        wait_east_out;
        if (rx_success && rx_data == 64'hABCDABCD_12341234) $display("  PASS");
        else begin $display("  FAIL: success=%b", rx_success); errors = errors + 1; end
        #(CLK * 5);

        // TEST 8: Backpressure
        $display("");
        $display("[TEST 8] Backpressure Handling");
        east_out_ready = 0;
        inject_local(64'hBA0C9E55_00000000, 4'd2, 4'd1);
        #(CLK * 5);
        east_out_ready = 1;
        wait_east_out;
        if (rx_success && rx_data == 64'hBA0C9E55_00000000) $display("  PASS");
        else begin $display("  FAIL: success=%b", rx_success); errors = errors + 1; end
        #(CLK * 5);

        // Summary
        $display("");
        $display("════════════════════════════════════════");
        $display("Tests: 8, Errors: %0d", errors);
        if (errors == 0) $display(">>> ALL TESTS PASSED! <<<");
        else $display(">>> SOME TESTS FAILED <<<");
        $display("");
        $finish;
    end

    initial begin $dumpfile("noc_router.vcd"); $dumpvars(0, tb_noc_router); end
    initial begin #(CLK * 5000); $display("TIMEOUT!"); $finish; end

endmodule

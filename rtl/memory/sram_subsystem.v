//==============================================================================
// Banked SRAM Subsystem
//
// Multi-bank SRAM with arbitration for high-bandwidth parallel access:
// - Multiple independent banks for parallel access
// - Multi-port arbitration (MXU > VPU > DMA priority)
// - XOR-based bank mapping to reduce stride conflicts
// - Configurable bank count and depth
//==============================================================================

module sram_subsystem #(
    parameter NUM_BANKS   = 16,           // Number of SRAM banks
    parameter BANK_DEPTH  = 4096,         // Words per bank
    parameter DATA_WIDTH  = 256,          // Bits per word (32 bytes)
    parameter ADDR_WIDTH  = 20            // Total address width
)(
    input  wire                           clk,
    input  wire                           rst_n,
    
    //--------------------------------------------------------------------------
    // Port A: MXU Weight Read (highest priority)
    //--------------------------------------------------------------------------
    input  wire [ADDR_WIDTH-1:0]          mxu_w_addr,
    input  wire                           mxu_w_re,
    output wire [DATA_WIDTH-1:0]          mxu_w_rdata,
    output wire                           mxu_w_ready,
    
    //--------------------------------------------------------------------------
    // Port B: MXU Activation Read
    //--------------------------------------------------------------------------
    input  wire [ADDR_WIDTH-1:0]          mxu_a_addr,
    input  wire                           mxu_a_re,
    output wire [DATA_WIDTH-1:0]          mxu_a_rdata,
    output wire                           mxu_a_ready,
    
    //--------------------------------------------------------------------------
    // Port C: MXU Result Write
    //--------------------------------------------------------------------------
    input  wire [ADDR_WIDTH-1:0]          mxu_o_addr,
    input  wire [DATA_WIDTH-1:0]          mxu_o_wdata,
    input  wire                           mxu_o_we,
    output wire                           mxu_o_ready,
    
    //--------------------------------------------------------------------------
    // Port D: Vector Unit Read/Write
    //--------------------------------------------------------------------------
    input  wire [ADDR_WIDTH-1:0]          vpu_addr,
    input  wire [DATA_WIDTH-1:0]          vpu_wdata,
    input  wire                           vpu_we,
    input  wire                           vpu_re,
    output wire [DATA_WIDTH-1:0]          vpu_rdata,
    output wire                           vpu_ready,
    
    //--------------------------------------------------------------------------
    // Port E: DMA Read/Write (lowest priority)
    //--------------------------------------------------------------------------
    input  wire [ADDR_WIDTH-1:0]          dma_addr,
    input  wire [DATA_WIDTH-1:0]          dma_wdata,
    input  wire                           dma_we,
    input  wire                           dma_re,
    output wire [DATA_WIDTH-1:0]          dma_rdata,
    output wire                           dma_ready
);

    //--------------------------------------------------------------------------
    // Address Decomposition
    //--------------------------------------------------------------------------
    
    localparam BANK_BITS = $clog2(NUM_BANKS);
    localparam WORD_BITS = $clog2(BANK_DEPTH);
    
    // Bank selection using XOR of high and low bits (reduces stride conflicts)
    function [BANK_BITS-1:0] get_bank;
        input [ADDR_WIDTH-1:0] addr;
        begin
            // XOR upper and lower bits for better distribution
            get_bank = addr[BANK_BITS-1:0] ^ addr[BANK_BITS+WORD_BITS-1:WORD_BITS];
        end
    endfunction
    
    function [WORD_BITS-1:0] get_word;
        input [ADDR_WIDTH-1:0] addr;
        begin
            get_word = addr[BANK_BITS+WORD_BITS-1:BANK_BITS];
        end
    endfunction
    
    //--------------------------------------------------------------------------
    // Bank Selection for Each Port
    //--------------------------------------------------------------------------
    
    wire [BANK_BITS-1:0] bank_mxu_w = get_bank(mxu_w_addr);
    wire [BANK_BITS-1:0] bank_mxu_a = get_bank(mxu_a_addr);
    wire [BANK_BITS-1:0] bank_mxu_o = get_bank(mxu_o_addr);
    wire [BANK_BITS-1:0] bank_vpu   = get_bank(vpu_addr);
    wire [BANK_BITS-1:0] bank_dma   = get_bank(dma_addr);
    
    wire [WORD_BITS-1:0] word_mxu_w = get_word(mxu_w_addr);
    wire [WORD_BITS-1:0] word_mxu_a = get_word(mxu_a_addr);
    wire [WORD_BITS-1:0] word_mxu_o = get_word(mxu_o_addr);
    wire [WORD_BITS-1:0] word_vpu   = get_word(vpu_addr);
    wire [WORD_BITS-1:0] word_dma   = get_word(dma_addr);
    
    //--------------------------------------------------------------------------
    // Per-Bank Arbitration
    //--------------------------------------------------------------------------
    
    // Request signals per bank
    reg [NUM_BANKS-1:0] req_mxu_w, req_mxu_a, req_mxu_o, req_vpu, req_dma;
    
    // Grant signals per bank
    reg [NUM_BANKS-1:0] grant_mxu_w, grant_mxu_a, grant_mxu_o, grant_vpu, grant_dma;
    
    // Bank interface signals
    reg  [WORD_BITS-1:0]  bank_addr  [0:NUM_BANKS-1];
    reg  [DATA_WIDTH-1:0] bank_wdata [0:NUM_BANKS-1];
    reg  [NUM_BANKS-1:0]  bank_we;
    reg  [NUM_BANKS-1:0]  bank_re;
    wire [DATA_WIDTH-1:0] bank_rdata [0:NUM_BANKS-1];
    
    // Generate request signals
    integer b;
    always @(*) begin
        for (b = 0; b < NUM_BANKS; b = b + 1) begin
            req_mxu_w[b] = mxu_w_re && (bank_mxu_w == b);
            req_mxu_a[b] = mxu_a_re && (bank_mxu_a == b);
            req_mxu_o[b] = mxu_o_we && (bank_mxu_o == b);
            req_vpu[b]   = (vpu_we || vpu_re) && (bank_vpu == b);
            req_dma[b]   = (dma_we || dma_re) && (bank_dma == b);
        end
    end
    
    // Priority arbitration per bank
    // Priority: MXU_W > MXU_A > MXU_O > VPU > DMA
    always @(*) begin
        for (b = 0; b < NUM_BANKS; b = b + 1) begin
            grant_mxu_w[b] = req_mxu_w[b];
            grant_mxu_a[b] = req_mxu_a[b] && !req_mxu_w[b];
            grant_mxu_o[b] = req_mxu_o[b] && !req_mxu_w[b] && !req_mxu_a[b];
            grant_vpu[b]   = req_vpu[b] && !req_mxu_w[b] && !req_mxu_a[b] && !req_mxu_o[b];
            grant_dma[b]   = req_dma[b] && !req_mxu_w[b] && !req_mxu_a[b] && !req_mxu_o[b] && !req_vpu[b];
            
            // Mux bank inputs based on grants
            if (grant_mxu_w[b]) begin
                bank_addr[b]  = word_mxu_w;
                bank_wdata[b] = {DATA_WIDTH{1'b0}};
                bank_we[b]    = 1'b0;
                bank_re[b]    = 1'b1;
            end else if (grant_mxu_a[b]) begin
                bank_addr[b]  = word_mxu_a;
                bank_wdata[b] = {DATA_WIDTH{1'b0}};
                bank_we[b]    = 1'b0;
                bank_re[b]    = 1'b1;
            end else if (grant_mxu_o[b]) begin
                bank_addr[b]  = word_mxu_o;
                bank_wdata[b] = mxu_o_wdata;
                bank_we[b]    = 1'b1;
                bank_re[b]    = 1'b0;
            end else if (grant_vpu[b]) begin
                bank_addr[b]  = word_vpu;
                bank_wdata[b] = vpu_wdata;
                bank_we[b]    = vpu_we;
                bank_re[b]    = vpu_re;
            end else if (grant_dma[b]) begin
                bank_addr[b]  = word_dma;
                bank_wdata[b] = dma_wdata;
                bank_we[b]    = dma_we;
                bank_re[b]    = dma_re;
            end else begin
                bank_addr[b]  = {WORD_BITS{1'b0}};
                bank_wdata[b] = {DATA_WIDTH{1'b0}};
                bank_we[b]    = 1'b0;
                bank_re[b]    = 1'b0;
            end
        end
    end
    
    //--------------------------------------------------------------------------
    // SRAM Bank Instances
    //--------------------------------------------------------------------------
    
    genvar i;
    generate
        for (i = 0; i < NUM_BANKS; i = i + 1) begin : bank_gen
            sram_bank #(
                .DEPTH(BANK_DEPTH),
                .WIDTH(DATA_WIDTH)
            ) bank_inst (
                .clk   (clk),
                .addr  (bank_addr[i]),
                .wdata (bank_wdata[i]),
                .we    (bank_we[i]),
                .re    (bank_re[i]),
                .rdata (bank_rdata[i])
            );
        end
    endgenerate
    
    //--------------------------------------------------------------------------
    // Output Muxing
    //--------------------------------------------------------------------------
    
    // Read data output registers (1 cycle latency)
    reg [DATA_WIDTH-1:0] mxu_w_rdata_reg;
    reg [DATA_WIDTH-1:0] mxu_a_rdata_reg;
    reg [DATA_WIDTH-1:0] vpu_rdata_reg;
    reg [DATA_WIDTH-1:0] dma_rdata_reg;
    
    reg [BANK_BITS-1:0] bank_mxu_w_d, bank_mxu_a_d, bank_vpu_d, bank_dma_d;
    
    always @(posedge clk) begin
        // Register bank selections for read data muxing
        bank_mxu_w_d <= bank_mxu_w;
        bank_mxu_a_d <= bank_mxu_a;
        bank_vpu_d   <= bank_vpu;
        bank_dma_d   <= bank_dma;
    end
    
    // Mux read data based on registered bank selection
    always @(*) begin
        mxu_w_rdata_reg = bank_rdata[bank_mxu_w_d];
        mxu_a_rdata_reg = bank_rdata[bank_mxu_a_d];
        vpu_rdata_reg   = bank_rdata[bank_vpu_d];
        dma_rdata_reg   = bank_rdata[bank_dma_d];
    end
    
    //--------------------------------------------------------------------------
    // Ready Signals
    //--------------------------------------------------------------------------
    
    // Ready when granted access to requested bank
    assign mxu_w_ready = grant_mxu_w[bank_mxu_w];
    assign mxu_a_ready = grant_mxu_a[bank_mxu_a];
    assign mxu_o_ready = grant_mxu_o[bank_mxu_o];
    assign vpu_ready   = grant_vpu[bank_vpu];
    assign dma_ready   = grant_dma[bank_dma];
    
    //--------------------------------------------------------------------------
    // Output Assignments
    //--------------------------------------------------------------------------
    
    assign mxu_w_rdata = mxu_w_rdata_reg;
    assign mxu_a_rdata = mxu_a_rdata_reg;
    assign vpu_rdata   = vpu_rdata_reg;
    assign dma_rdata   = dma_rdata_reg;

endmodule

//==============================================================================
// Single SRAM Bank
//==============================================================================

module sram_bank #(
    parameter DEPTH = 4096,
    parameter WIDTH = 256
)(
    input  wire                      clk,
    input  wire [$clog2(DEPTH)-1:0]  addr,
    input  wire [WIDTH-1:0]          wdata,
    input  wire                      we,
    input  wire                      re,
    output reg  [WIDTH-1:0]          rdata
);

    // Memory array
    (* ram_style = "block" *)  // Hint for FPGA synthesis
    reg [WIDTH-1:0] mem [0:DEPTH-1];
    
    // Synchronous read/write
    always @(posedge clk) begin
        if (we) begin
            mem[addr] <= wdata;
        end
        if (re) begin
            rdata <= mem[addr];
        end
    end
    
    // Note: For FPGA synthesis, BRAM is automatically initialized to zero
    // For simulation, add +define+SIM to enable initialization
    `ifdef SIM
    integer i;
    initial begin
        for (i = 0; i < DEPTH; i = i + 1) begin
            mem[i] = {WIDTH{1'b0}};
        end
    end
    `endif

endmodule

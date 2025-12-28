# Tensor Accelerator Verilog Tutorial

## A Step-by-Step Guide to Understanding the Complete Design

---

# Part 1: The Big Picture

## What Are We Building?

A **tensor accelerator** is specialized hardware that does matrix multiplication very fast. Neural networks are essentially thousands of matrix multiplications, so this hardware can run AI models 10-100× faster than a CPU.

```
Neural Network Layer:
  Output = Activation(Input × Weights + Bias)
           └────────────────┘
           This is matrix multiplication!
```

## The Architecture at 10,000 Feet

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TENSOR ACCELERATOR                               │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐│
│  │    TPC 0    │  │    TPC 1    │  │    TPC 2    │  │    TPC 3    ││
│  │  (16×16     │  │  (16×16     │  │  (16×16     │  │  (16×16     ││
│  │   systolic  │  │   systolic  │  │   systolic  │  │   systolic  ││
│  │   array)    │  │   array)    │  │   array)    │  │   array)    ││
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘│
│         │               │               │               │          │
│         └───────────────┴───────────────┴───────────────┘          │
│                              │                                      │
│                    ┌─────────▼─────────┐                           │
│                    │   Network on Chip  │                           │
│                    │       (NoC)        │                           │
│                    └─────────┬─────────┘                           │
│                              │                                      │
│                    ┌─────────▼─────────┐                           │
│                    │  Global Controller │                           │
│                    │       (GCP)        │                           │
│                    └─────────┬─────────┘                           │
│                              │                                      │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   External Memory   │
                    │   (DDR4/LPDDR5)     │
                    └─────────────────────┘
```

**TPC** = Tensor Processing Cluster (the compute unit)
**NoC** = Network on Chip (moves data around)
**GCP** = Global Command Processor (the boss)

---

# Part 2: The MAC PE (Multiply-Accumulate Processing Element)

## The Simplest Building Block

Everything starts with the **MAC PE**. It does one simple thing:

```
output = input_activation × stored_weight + partial_sum_from_above
```

### The Code Explained

```verilog
// File: rtl/core/mac_pe.v

module mac_pe #(
    parameter DATA_WIDTH = 8,    // 8-bit integers (INT8)
    parameter ACC_WIDTH  = 32    // 32-bit accumulator (prevents overflow)
)(
    input  wire                  clk,         // Clock signal
    input  wire                  rst_n,       // Reset (active low)
    
    // Control signals
    input  wire                  enable,      // When high, PE computes
    input  wire                  load_weight, // When high, store new weight
    input  wire                  clear_acc,   // When high, start fresh
    
    // Data signals
    input  wire [DATA_WIDTH-1:0] weight_in,   // Weight to store
    input  wire [DATA_WIDTH-1:0] act_in,      // Activation from left
    output reg  [DATA_WIDTH-1:0] act_out,     // Activation to right
    input  wire [ACC_WIDTH-1:0]  psum_in,     // Partial sum from above
    output reg  [ACC_WIDTH-1:0]  psum_out     // Partial sum to below
);
```

**What are these signals?**

| Signal | Direction | Purpose |
|--------|-----------|---------|
| `clk` | Input | The heartbeat - everything happens on clock edges |
| `rst_n` | Input | Reset everything to zero (the `_n` means active-LOW) |
| `enable` | Input | "Go!" signal - when 1, the PE computes |
| `load_weight` | Input | "Store this weight" - happens once per tile |
| `weight_in` | Input | The weight value to store |
| `act_in` | Input | Activation coming from the PE on the left |
| `act_out` | Output | Same activation, passed to PE on the right |
| `psum_in` | Input | Partial sum from the PE above |
| `psum_out` | Output | New partial sum, sent to PE below |

### The Weight Register

```verilog
    // This register holds the weight - it stays constant during computation
    reg [DATA_WIDTH-1:0] weight_reg;
```

Think of `weight_reg` like a sticky note. You write a number on it once, and it stays there while you do many multiplications.

### The Multiplication

```verilog
    // Signed multiplication
    wire signed [DATA_WIDTH-1:0]   a_signed = $signed(act_reg);
    wire signed [DATA_WIDTH-1:0]   w_signed = $signed(weight_reg);
    wire signed [2*DATA_WIDTH-1:0] product  = a_signed * w_signed;
```

**Why `$signed`?** 
- Neural network values can be negative (-128 to +127 for INT8)
- `$signed()` tells Verilog to treat the bits as a signed number
- 8-bit × 8-bit = 16-bit result (that's why `2*DATA_WIDTH`)

### Sign Extension

```verilog
    // Expand 16-bit product to 32-bit accumulator
    wire signed [ACC_WIDTH-1:0] product_ext = 
        {{(ACC_WIDTH-2*DATA_WIDTH){product[2*DATA_WIDTH-1]}}, product};
```

**What is this doing?**

```
product (16-bit):     1111_1111_1111_0110  (-10 in decimal)
                      ↑
                      This is the sign bit

product_ext (32-bit): 1111_1111_1111_1111_1111_1111_1111_0110
                      ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
                      Replicate sign bit to fill 32 bits
```

This is called **sign extension** - it preserves the negative value when expanding to more bits.

### The Main Logic

```verilog
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset everything to zero
            weight_reg <= 0;
            act_out    <= 0;
            psum_out   <= 0;
        end else begin
            // Load weight when told to
            if (load_weight) begin
                weight_reg <= weight_in;
            end
            
            // Main computation when enabled
            if (enable) begin
                // Pass activation to the right
                act_out <= act_in;
                
                // Compute: psum_out = psum_in + (act × weight)
                if (clear_acc)
                    psum_out <= product_ext;           // Start fresh
                else
                    psum_out <= psum_in + product_ext; // Accumulate
            end
        end
    end
```

**The key insight:** 
- `always @(posedge clk)` means "do this on every rising clock edge"
- The PE does ONE multiply-add per clock cycle
- Activations flow horizontally (left → right)
- Partial sums flow vertically (top → bottom)

---

# Part 3: The Systolic Array

## 256 PEs Working in Harmony

A **systolic array** is a grid of PEs that work together like a beating heart (systolic = pumping).

```
                    Weights loaded once (stay in place)
                              ↓ ↓ ↓ ↓
                    ┌────┬────┬────┬────┐
  Activations  →    │ PE │ PE │ PE │ PE │ → (activations flow right)
  flow in           ├────┼────┼────┼────┤
  from left    →    │ PE │ PE │ PE │ PE │
                    ├────┼────┼────┼────┤
               →    │ PE │ PE │ PE │ PE │
                    ├────┼────┼────┼────┤
               →    │ PE │ PE │ PE │ PE │
                    └────┴────┴────┴────┘
                      ↓    ↓    ↓    ↓
                    Results come out bottom
```

### Why "Systolic"?

Data pulses through the array like blood through a heart:
1. **Weights** are loaded once and stay put
2. **Activations** flow left-to-right, one column per cycle
3. **Partial sums** flow top-to-bottom
4. **Results** emerge from the bottom after filling the array

### The Code Structure

```verilog
// File: rtl/core/systolic_array.v

module systolic_array #(
    parameter ARRAY_SIZE = 16,    // 16×16 grid = 256 PEs
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  wire        clk,
    input  wire        rst_n,
    
    // Control
    input  wire        start,
    output wire        busy,
    output wire        done,
    
    // Weight loading (one column at a time)
    input  wire        weight_load_en,
    input  wire [3:0]  weight_load_col,      // Which column (0-15)
    input  wire [127:0] weight_load_data,    // 16 weights × 8 bits
    
    // Activation input (one per row)
    input  wire        act_valid,
    input  wire [127:0] act_data,            // 16 activations × 8 bits
    
    // Result output (one per column)
    output wire        result_valid,
    output wire [511:0] result_data          // 16 results × 32 bits
);
```

### Wiring Up the PEs

```verilog
    // Wires connecting PEs horizontally (activations)
    wire [DATA_WIDTH-1:0] act_h [0:ARRAY_SIZE-1][0:ARRAY_SIZE];
    
    // Wires connecting PEs vertically (partial sums)  
    wire [ACC_WIDTH-1:0] psum_v [0:ARRAY_SIZE][0:ARRAY_SIZE-1];
```

**Visualization:**

```
         col 0    col 1    col 2    col 3
        ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
act_h   │      │ │      │ │      │ │      │
[0][0]→ │ PE   │→│ PE   │→│ PE   │→│ PE   │→ act_h[0][4]
        │ 0,0  │ │ 0,1  │ │ 0,2  │ │ 0,3  │
        └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘
           ↓        ↓        ↓        ↓
        psum_v   psum_v   psum_v   psum_v
        [1][0]   [1][1]   [1][2]   [1][3]
           ↓        ↓        ↓        ↓
        ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
act_h   │      │ │      │ │      │ │      │
[1][0]→ │ PE   │→│ PE   │→│ PE   │→│ PE   │→
        │ 1,0  │ │ 1,1  │ │ 1,2  │ │ 1,3  │
        └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘
           ↓        ↓        ↓        ↓
        (continues for all 16 rows)
```

### The Generate Loop

```verilog
    // Create 16×16 = 256 PEs
    generate
        for (row = 0; row < ARRAY_SIZE; row = row + 1) begin : pe_row
            for (col = 0; col < ARRAY_SIZE; col = col + 1) begin : pe_col
                
                mac_pe #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH(ACC_WIDTH)
                ) pe_inst (
                    .clk        (clk),
                    .rst_n      (rst_n),
                    .enable     (pe_enable),
                    
                    // Weight loading for this specific PE
                    .load_weight(load_weight_pe[row][col]),
                    .weight_in  (weight_to_pe[row][col]),
                    
                    // Horizontal activation flow
                    .act_in     (act_h[row][col]),      // From left
                    .act_out    (act_h[row][col+1]),    // To right
                    
                    // Vertical partial sum flow
                    .psum_in    (psum_input),           // From above
                    .psum_out   (psum_v[row+1][col])    // To below
                );
            end
        end
    endgenerate
```

**The `generate` block:**
- It's like a for-loop that creates hardware
- Runs at *compile time*, not runtime
- Creates 256 separate PE instances
- Each PE is connected to its neighbors

### The State Machine

```verilog
    // States
    localparam S_IDLE    = 3'd0;  // Waiting for work
    localparam S_LOAD    = 3'd1;  // Loading weights
    localparam S_COMPUTE = 3'd2;  // Matrix multiplication
    localparam S_DRAIN   = 3'd3;  // Outputting results
    localparam S_DONE    = 3'd4;  // Finished
    
    reg [2:0] state;
```

**State Diagram:**

```
    ┌──────────────────────────────────────────────────────┐
    │                                                      │
    ▼                                                      │
┌──────┐  start   ┌──────┐  weights   ┌─────────┐        │
│ IDLE │────────▶│ LOAD │──loaded───▶│ COMPUTE │        │
└──────┘          └──────┘            └────┬────┘        │
    ▲                                      │              │
    │         ┌──────┐    results    ┌─────▼────┐        │
    └─────────│ DONE │◀───drained────│  DRAIN   │        │
              └──────┘               └──────────┘        │
                  │                                      │
                  └──────────────────────────────────────┘
```

---

# Part 4: How Matrix Multiplication Actually Happens

## The Math We're Computing

```
C = A × B

Where:
  A is [M × K] - activations (e.g., 16×16)
  B is [K × N] - weights (e.g., 16×16)
  C is [M × N] - output (e.g., 16×16)

C[i][j] = Σ(k=0 to K-1) A[i][k] × B[k][j]
```

## Step-by-Step Example (4×4 for simplicity)

### Step 1: Load Weights

Weights are loaded column-by-column and stay in place:

```
Cycle 1: Load column 0          Cycle 2: Load column 1
┌────┬────┬────┬────┐          ┌────┬────┬────┬────┐
│ W00│    │    │    │          │ W00│ W01│    │    │
├────┼────┼────┼────┤          ├────┼────┼────┼────┤
│ W10│    │    │    │          │ W10│ W11│    │    │
├────┼────┼────┼────┤          ├────┼────┼────┼────┤
│ W20│    │    │    │          │ W20│ W21│    │    │
├────┼────┼────┼────┤          ├────┼────┼────┼────┤
│ W30│    │    │    │          │ W30│ W31│    │    │
└────┴────┴────┴────┘          └────┴────┴────┴────┘
```

### Step 2: Stream Activations (with skewing)

Activations enter with a diagonal skew pattern:

```
Cycle 0:                        Cycle 1:
┌────┬────┬────┬────┐          ┌────┬────┬────┬────┐
│ A00│    │    │    │          │ A01│ A00│    │    │
├────┼────┼────┼────┤          ├────┼────┼────┼────┤
│    │    │    │    │          │ A10│    │    │    │
├────┼────┼────┼────┤          ├────┼────┼────┼────┤
│    │    │    │    │          │    │    │    │    │
├────┼────┼────┼────┤          ├────┼────┼────┼────┤
│    │    │    │    │          │    │    │    │    │
└────┴────┴────┴────┘          └────┴────┴────┴────┘

Cycle 2:                        Cycle 3:
┌────┬────┬────┬────┐          ┌────┬────┬────┬────┐
│ A02│ A01│ A00│    │          │ A03│ A02│ A01│ A00│
├────┼────┼────┼────┤          ├────┼────┼────┼────┤
│ A11│ A10│    │    │          │ A12│ A11│ A10│    │
├────┼────┼────┼────┤          ├────┼────┼────┼────┤
│ A20│    │    │    │          │ A21│ A20│    │    │
├────┼────┼────┼────┤          ├────┼────┼────┼────┤
│    │    │    │    │          │ A30│    │    │    │
└────┴────┴────┴────┘          └────┴────┴────┴────┘
```

**Why skewing?** It ensures that by the time an activation reaches a PE, the partial sum from above has also arrived. It's like a perfectly choreographed dance!

### Step 3: Accumulation

Each PE computes: `psum_out = psum_in + (act × weight)`

After all K cycles, the bottom row contains the final results.

---

# Part 5: The Vector Processing Unit (VPU)

## What It Does

The VPU handles operations that aren't matrix multiplication:
- **Activation functions**: ReLU, GELU, Sigmoid
- **Normalization**: LayerNorm, BatchNorm
- **Element-wise**: Add, Multiply, Scale
- **Reductions**: Sum, Max (for Softmax)

### SIMD Architecture

```
64 Lanes processing in parallel:

Input[0]  Input[1]  Input[2]  ...  Input[63]
    │         │         │              │
    ▼         ▼         ▼              ▼
┌───────┐ ┌───────┐ ┌───────┐     ┌───────┐
│ Lane 0│ │ Lane 1│ │ Lane 2│ ... │Lane 63│
│  ALU  │ │  ALU  │ │  ALU  │     │  ALU  │
└───────┘ └───────┘ └───────┘     └───────┘
    │         │         │              │
    ▼         ▼         ▼              ▼
Output[0] Output[1] Output[2] ... Output[63]
```

**SIMD** = Single Instruction, Multiple Data
- One instruction controls all 64 lanes
- Each lane processes one element
- 64× parallelism!

### The Code

```verilog
// File: rtl/core/vector_unit.v

module vector_unit #(
    parameter LANES      = 64,    // 64 parallel lanes
    parameter DATA_WIDTH = 16,    // 16-bit (for more precision)
    parameter ACC_WIDTH  = 32
)(
    input  wire                        clk,
    input  wire                        rst_n,
    
    // Operation select
    input  wire [4:0]                  opcode,    // What operation?
    input  wire                        op_valid,  // Start operation
    output wire                        op_ready,  // Ready for next
    
    // Input operands (64 elements × 16 bits = 1024 bits)
    input  wire [LANES*DATA_WIDTH-1:0] src0_data,
    input  wire [LANES*DATA_WIDTH-1:0] src1_data,
    
    // Output result
    output wire [LANES*DATA_WIDTH-1:0] dst_data,
    output wire                        dst_valid
);
```

### Supported Operations

```verilog
    // Operation codes
    localparam OP_ADD     = 5'd0;   // dst = src0 + src1
    localparam OP_MUL     = 5'd1;   // dst = src0 × src1
    localparam OP_RELU    = 5'd2;   // dst = max(0, src0)
    localparam OP_GELU    = 5'd3;   // dst = GELU(src0)
    localparam OP_MAX     = 5'd4;   // dst = max(src0, src1)
    localparam OP_SUM     = 5'd5;   // Reduce: sum all 64 elements
    localparam OP_SCALE   = 5'd6;   // dst = src0 × scalar
    localparam OP_SOFTMAX = 5'd7;   // Multi-step softmax
```

### ReLU Example

```verilog
    // ReLU is simple: if negative, output 0; otherwise pass through
    generate
        for (i = 0; i < LANES; i = i + 1) begin : relu_lane
            wire signed [DATA_WIDTH-1:0] val = src0_data[i*DATA_WIDTH +: DATA_WIDTH];
            
            // Check sign bit (MSB)
            assign relu_result[i*DATA_WIDTH +: DATA_WIDTH] = 
                val[DATA_WIDTH-1] ? {DATA_WIDTH{1'b0}} : val;
            //  ↑ if negative        ↑ output zero      ↑ else pass through
        end
    endgenerate
```

---

# Part 6: The DMA Engine

## Moving Data Efficiently

The **DMA (Direct Memory Access) Engine** moves data between:
- External memory (DDR) ↔ On-chip SRAM
- SRAM ↔ Systolic array input buffers
- Systolic array output ↔ SRAM

### Why DMA?

Without DMA:
```
CPU: "Read address 0"
Memory: Returns data
CPU: "Write to SRAM address 0"
CPU: "Read address 1"
... (repeat 1 million times - very slow!)
```

With DMA:
```
CPU: "Transfer 1MB from address X to SRAM starting at Y"
DMA: (Does it autonomously while CPU/accelerator does other work)
DMA: "Done!"
```

### 2D DMA for Tensors

Neural network data is multi-dimensional. 2D DMA handles this:

```
Memory layout (row-major):
┌─────────────────────────────────────────────┐
│ Row 0 ──────────────────────────────────────│
│ Row 1 ──────────────────────────────────────│
│ Row 2 ──────────────────────────────────────│
│ ...                                         │
└─────────────────────────────────────────────┘

2D Transfer Parameters:
  - Base address: Starting point
  - Width: Bytes per row to transfer
  - Height: Number of rows
  - Stride: Bytes between row starts (may be > width for padding)
```

### The Code

```verilog
// File: rtl/core/dma_engine.v

module dma_engine #(
    parameter DATA_WIDTH = 256,    // 256-bit wide transfers
    parameter ADDR_WIDTH = 40      // 40-bit addresses (1TB)
)(
    input  wire        clk,
    input  wire        rst_n,
    
    // Command interface
    input  wire        cmd_valid,
    input  wire [2:0]  cmd_opcode,     // LOAD_2D, STORE_2D, COPY
    input  wire [39:0] cmd_ext_addr,   // External memory address
    input  wire [19:0] cmd_int_addr,   // Internal SRAM address
    input  wire [15:0] cmd_width,      // Transfer width
    input  wire [15:0] cmd_height,     // Transfer height
    input  wire [15:0] cmd_stride,     // Stride between rows
    output wire        cmd_ready,
    
    // Status
    output wire        busy,
    output wire        done,
    
    // AXI Master interface (to external memory)
    output wire [39:0] m_axi_araddr,
    output wire [7:0]  m_axi_arlen,    // Burst length
    output wire        m_axi_arvalid,
    input  wire        m_axi_arready,
    // ... (full AXI interface)
    
    // SRAM interface
    output wire [19:0] sram_addr,
    output wire        sram_we,
    output wire [255:0] sram_wdata,
    input  wire [255:0] sram_rdata
);
```

### DMA State Machine

```verilog
    localparam DMA_IDLE     = 3'd0;
    localparam DMA_LOAD_REQ = 3'd1;  // Request read from DDR
    localparam DMA_LOAD_DAT = 3'd2;  // Receive data, write to SRAM
    localparam DMA_STORE_REQ= 3'd3;  // Read from SRAM
    localparam DMA_STORE_DAT= 3'd4;  // Write to DDR
    localparam DMA_DONE     = 3'd5;

    // Track progress through 2D transfer
    reg [15:0] row_count;    // Current row
    reg [15:0] col_count;    // Current column within row
```

---

# Part 7: The SRAM Subsystem

## On-Chip Memory

SRAM is fast memory inside the chip. It's used for:
- Caching weights (so we don't constantly access slow DDR)
- Double-buffering activations (load next while computing current)
- Storing intermediate results

### Banked Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SRAM Subsystem                           │
│                                                             │
│  ┌────────┐ ┌────────┐ ┌────────┐     ┌────────┐          │
│  │ Bank 0 │ │ Bank 1 │ │ Bank 2 │ ... │Bank 15 │          │
│  │  4KB   │ │  4KB   │ │  4KB   │     │  4KB   │          │
│  └────┬───┘ └────┬───┘ └────┬───┘     └────┬───┘          │
│       │         │         │              │                 │
│       └─────────┴─────────┴──────────────┘                 │
│                      │                                      │
│               ┌──────▼──────┐                              │
│               │  Crossbar   │                              │
│               │  Arbiter    │                              │
│               └──────┬──────┘                              │
│                      │                                      │
│    ┌─────────────────┼─────────────────┐                   │
│    │                 │                 │                   │
│ ┌──▼──┐          ┌───▼───┐         ┌───▼───┐              │
│ │ DMA │          │Systolic│         │  VPU  │              │
│ │Port │          │ Array  │         │ Port  │              │
│ └─────┘          └───────┘         └───────┘              │
└─────────────────────────────────────────────────────────────┘
```

**Why banks?**
- Multiple units can access different banks simultaneously
- Bank 0 → DMA loading weights
- Bank 1 → Systolic array reading activations
- No conflict = full bandwidth!

### The Code

```verilog
// File: rtl/memory/sram_subsystem.v

module sram_subsystem #(
    parameter NUM_BANKS   = 16,      // 16 banks
    parameter BANK_DEPTH  = 4096,    // 4K words per bank
    parameter DATA_WIDTH  = 256      // 256-bit words (32 bytes)
)(
    input  wire        clk,
    input  wire        rst_n,
    
    // Port A (e.g., DMA)
    input  wire [19:0] porta_addr,
    input  wire        porta_we,
    input  wire [255:0] porta_wdata,
    output wire [255:0] porta_rdata,
    
    // Port B (e.g., Systolic Array)
    input  wire [19:0] portb_addr,
    input  wire        portb_we,
    input  wire [255:0] portb_wdata,
    output wire [255:0] portb_rdata
);

    // Bank select from address bits
    wire [3:0] porta_bank = porta_addr[3:0];  // Lower 4 bits select bank
    wire [11:0] porta_word = porta_addr[15:4]; // Upper bits select word in bank
```

### Single-Port SRAM Block

```verilog
// Simple SRAM block (will infer BRAM in FPGA)
module sram_block #(
    parameter DEPTH = 4096,
    parameter WIDTH = 256
)(
    input  wire                     clk,
    input  wire [$clog2(DEPTH)-1:0] addr,
    input  wire                     we,
    input  wire [WIDTH-1:0]         wdata,
    output reg  [WIDTH-1:0]         rdata
);
    // Memory array
    reg [WIDTH-1:0] mem [0:DEPTH-1];
    
    // Synchronous read/write
    always @(posedge clk) begin
        if (we) begin
            mem[addr] <= wdata;
        end
        rdata <= mem[addr];
    end
endmodule
```

**Key insight:** This simple pattern (`reg [WIDTH-1:0] mem [...]` with synchronous read/write) is recognized by FPGA tools and mapped to efficient Block RAM (BRAM).

---

# Part 8: The Local Command Processor (LCP)

## The Per-TPC Controller

Each TPC has its own LCP that:
- Fetches instructions from SRAM
- Decodes them
- Orchestrates the systolic array, VPU, and DMA
- Handles hardware loops

### Instruction Format

```
128-bit instruction:
┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│ Opcode │ Subop  │  Dst   │  Src0  │  Src1  │  Dim_M │  Dim_N │ Flags  │
│ [7:0]  │ [7:0]  │ [15:0] │ [15:0] │ [15:0] │ [15:0] │ [15:0] │ [31:0] │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
```

### Supported Instructions

```verilog
// Opcodes
localparam OP_TENSOR = 8'h01;  // Matrix operations
localparam OP_VECTOR = 8'h02;  // VPU operations
localparam OP_DMA    = 8'h03;  // Data movement
localparam OP_SYNC   = 8'h04;  // Synchronization
localparam OP_LOOP   = 8'h05;  // Hardware loop start
localparam OP_ENDLOOP= 8'h06;  // Hardware loop end
localparam OP_BARRIER= 8'h07;  // Wait for all TPCs
localparam OP_HALT   = 8'hFF;  // Stop execution
```

### The Fetch-Decode-Execute Pipeline

```verilog
// File: rtl/control/local_cmd_processor.v

module local_cmd_processor (
    input  wire        clk,
    input  wire        rst_n,
    
    // Control
    input  wire        start,
    input  wire [19:0] pc_start,    // Starting instruction address
    output wire        done,
    output wire        halted,
    
    // Instruction memory interface
    output wire [19:0] imem_addr,
    input  wire [127:0] imem_data,
    
    // Control outputs to datapath
    output wire        mxu_start,   // Start systolic array
    output wire        vpu_start,   // Start VPU
    output wire        dma_start,   // Start DMA
    // ... configuration signals
);

    // Program counter
    reg [19:0] pc;
    
    // Pipeline registers
    reg [127:0] instr_reg;
    
    // State machine
    localparam S_IDLE   = 3'd0;
    localparam S_FETCH  = 3'd1;
    localparam S_DECODE = 3'd2;
    localparam S_EXEC   = 3'd3;
    localparam S_WAIT   = 3'd4;  // Wait for long operations
    localparam S_HALT   = 3'd5;
```

### Hardware Loops

```verilog
    // Loop stack (nested loops)
    reg [19:0] loop_start [0:3];  // Start address
    reg [15:0] loop_count [0:3];  // Iteration count
    reg [1:0]  loop_depth;        // Current nesting level
    
    // When we hit ENDLOOP:
    always @(posedge clk) begin
        if (is_endloop) begin
            if (loop_count[loop_depth] > 1) begin
                // More iterations: jump back
                pc <= loop_start[loop_depth];
                loop_count[loop_depth] <= loop_count[loop_depth] - 1;
            end else begin
                // Done: pop the loop stack
                loop_depth <= loop_depth - 1;
                pc <= pc + 1;
            end
        end
    end
```

---

# Part 9: The Network on Chip (NoC)

## Connecting Everything

The NoC is like a highway system inside the chip:

```
        ┌─────────┐           ┌─────────┐
        │  TPC 0  │◄────────►│  TPC 1  │
        └────┬────┘           └────┬────┘
             │                     │
             ▼                     ▼
        ┌─────────┐           ┌─────────┐
        │ Router  │◄────────►│ Router  │
        │  (0,0)  │           │  (1,0)  │
        └────┬────┘           └────┬────┘
             │                     │
             ▼                     ▼
        ┌─────────┐           ┌─────────┐
        │ Router  │◄────────►│ Router  │
        │  (0,1)  │           │  (1,1)  │
        └────┬────┘           └────┬────┘
             │                     │
             ▼                     ▼
        ┌─────────┐           ┌─────────┐
        │  TPC 2  │◄────────►│  TPC 3  │
        └─────────┘           └─────────┘
```

### Router Architecture

```verilog
// File: rtl/noc/noc_router.v

module noc_router #(
    parameter FLIT_WIDTH = 256,
    parameter X_COORD    = 0,
    parameter Y_COORD    = 0
)(
    input  wire        clk,
    input  wire        rst_n,
    
    // 5 ports: Local, North, South, East, West
    // Input port (from neighbor)
    input  wire [FLIT_WIDTH-1:0] north_in_data,
    input  wire                  north_in_valid,
    output wire                  north_in_ready,
    
    // Output port (to neighbor)
    output wire [FLIT_WIDTH-1:0] north_out_data,
    output wire                  north_out_valid,
    input  wire                  north_out_ready,
    
    // ... same for South, East, West, Local
);
```

### XY Routing

Simple, deadlock-free routing:
1. First go in X direction until X matches
2. Then go in Y direction until Y matches

```verilog
    // Extract destination from flit header
    wire [3:0] dest_x = flit_data[7:4];
    wire [3:0] dest_y = flit_data[3:0];
    
    // Routing decision
    always @(*) begin
        if (dest_x < X_COORD)
            route_dir = WEST;
        else if (dest_x > X_COORD)
            route_dir = EAST;
        else if (dest_y < Y_COORD)
            route_dir = NORTH;
        else if (dest_y > Y_COORD)
            route_dir = SOUTH;
        else
            route_dir = LOCAL;  // We've arrived!
    end
```

---

# Part 10: Putting It All Together

## The Top-Level Module

```verilog
// File: rtl/top/tensor_accelerator_top.v

module tensor_accelerator_top #(
    parameter NUM_TPCS     = 4,       // 2×2 grid of TPCs
    parameter ARRAY_SIZE   = 16,      // 16×16 systolic arrays
    parameter DATA_WIDTH   = 8,
    parameter AXI_DATA_W   = 256
)(
    input  wire        clk,
    input  wire        rst_n,
    
    // AXI-Lite control interface (from host CPU)
    input  wire [31:0] s_axi_ctrl_awaddr,
    input  wire        s_axi_ctrl_awvalid,
    output wire        s_axi_ctrl_awready,
    input  wire [31:0] s_axi_ctrl_wdata,
    input  wire        s_axi_ctrl_wvalid,
    output wire        s_axi_ctrl_wready,
    // ... (full AXI-Lite interface)
    
    // AXI4 memory interface (to DDR)
    output wire [39:0] m_axi_awaddr,
    output wire [7:0]  m_axi_awlen,
    output wire        m_axi_awvalid,
    input  wire        m_axi_awready,
    // ... (full AXI4 interface)
    
    // Interrupts
    output wire        irq_done,
    output wire        irq_error
);
```

### Instantiating TPCs

```verilog
    generate
        for (i = 0; i < NUM_TPCS; i = i + 1) begin : tpc_gen
            tensor_processing_cluster #(
                .ARRAY_SIZE(ARRAY_SIZE),
                .DATA_WIDTH(DATA_WIDTH),
                .TPC_ID(i)
            ) u_tpc (
                .clk        (clk),
                .rst_n      (rst_n),
                
                // Local control
                .start      (tpc_start[i]),
                .done       (tpc_done[i]),
                
                // NoC interface
                .noc_in_data  (noc_to_tpc[i]),
                .noc_out_data (tpc_to_noc[i]),
                
                // Memory interface
                .mem_req    (tpc_mem_req[i]),
                .mem_addr   (tpc_mem_addr[i]),
                .mem_wdata  (tpc_mem_wdata[i]),
                .mem_rdata  (mem_rdata),
                .mem_grant  (tpc_mem_grant[i])
            );
        end
    endgenerate
```

### The Control Register Map

```verilog
    // Register addresses (accessible via AXI-Lite)
    localparam REG_CTRL     = 12'h000;  // Control: start, stop, reset
    localparam REG_STATUS   = 12'h004;  // Status: busy, done, error
    localparam REG_IRQ_EN   = 12'h008;  // Interrupt enable
    localparam REG_IRQ_STAT = 12'h00C;  // Interrupt status
    
    // Per-TPC registers (at offset 0x100 + TPC_ID × 0x10)
    localparam REG_TPC_CTRL = 12'h100;  // TPC control
    localparam REG_TPC_PC   = 12'h104;  // Program counter start
    localparam REG_TPC_STAT = 12'h108;  // TPC status
```

---

# Part 11: Simulation and Testing

## The Testbench Structure

```verilog
// File: tb/tb_systolic_array.v

module tb_systolic_array;
    
    // Create signals to connect to DUT (Design Under Test)
    reg clk, rst_n, start;
    wire busy, done;
    // ...
    
    // Instantiate the design
    systolic_array #(
        .ARRAY_SIZE(16)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .busy(busy),
        .done(done)
        // ...
    );
    
    // Generate clock (10ns period = 100MHz)
    initial begin
        clk = 0;
        forever #5 clk = ~clk;  // Toggle every 5ns
    end
    
    // Test sequence
    initial begin
        // Initialize
        rst_n = 0;
        start = 0;
        #100;           // Wait 100ns
        
        rst_n = 1;      // Release reset
        #50;
        
        // Load weights...
        // Start computation...
        // Check results...
        
        $finish;
    end
    
    // Dump waveforms for viewing
    initial begin
        $dumpfile("waveform.vcd");
        $dumpvars(0, tb_systolic_array);
    end
endmodule
```

---

# Part 12: Summary - Data Flow Example

## Running a Neural Network Layer

Let's trace what happens when we run: `Output = ReLU(Input × Weights)`

### Step 1: Host Writes Command
```
Host CPU writes to control registers via AXI-Lite:
  REG_TPC_PC   = 0x1000    (instruction address)
  REG_TPC_CTRL = 0x1       (start TPC 0)
```

### Step 2: LCP Fetches Instructions
```
LCP reads instruction at 0x1000:
  DMA_LOAD_2D: Load weights from DDR to SRAM
```

### Step 3: DMA Loads Weights
```
DMA Engine:
  1. Issues AXI4 read burst to DDR
  2. Receives weight data
  3. Writes to SRAM banks 0-3
  4. Signals completion
```

### Step 4: LCP Continues
```
Next instruction:
  DMA_LOAD_2D: Load activations from DDR to SRAM
```

### Step 5: Start Matrix Multiply
```
Instruction: TENSOR GEMM
  1. LCP loads weights from SRAM into systolic array
  2. LCP streams activations from SRAM
  3. Systolic array computes for N cycles
  4. Results accumulate in bottom row
  5. Results written back to SRAM
```

### Step 6: Apply ReLU
```
Instruction: VECTOR RELU
  1. VPU reads from SRAM
  2. Applies ReLU in parallel (64 elements/cycle)
  3. Writes results back to SRAM
```

### Step 7: Store Output
```
Instruction: DMA_STORE_2D
  1. DMA reads results from SRAM
  2. Issues AXI4 write burst to DDR
  3. Signals completion
```

### Step 8: Done!
```
Instruction: HALT
  1. LCP enters halted state
  2. TPC signals done
  3. GCP generates interrupt
  4. Host CPU notified
```

---

# Quick Reference

## File Organization

```
rtl/
├── core/
│   ├── mac_pe.v           # Basic multiply-accumulate unit
│   ├── systolic_array.v   # 16×16 array of MACs
│   ├── vector_unit.v      # SIMD processor for activations
│   └── dma_engine.v       # Data movement engine
├── memory/
│   └── sram_subsystem.v   # On-chip memory banks
├── control/
│   ├── local_cmd_processor.v   # Per-TPC controller
│   └── global_cmd_processor.v  # System controller
├── noc/
│   └── noc_router.v       # Network router
└── top/
    ├── tensor_processing_cluster.v  # One TPC
    └── tensor_accelerator_top.v     # Full system
```

## Key Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `ARRAY_SIZE` | 16 | 16×16 systolic array |
| `DATA_WIDTH` | 8 | INT8 operations |
| `ACC_WIDTH` | 32 | 32-bit accumulators |
| `VPU_LANES` | 64 | 64-wide SIMD |
| `NUM_BANKS` | 16 | 16 SRAM banks |
| `NUM_TPCS` | 4 | 4 TPCs total |

## Performance

```
Per TPC @ 200 MHz:
  - 16 × 16 × 2 × 200M = 102.4 GOPS (INT8)

4 TPCs total:
  - 4 × 102.4 = 409.6 GOPS (INT8)
  - = 0.41 TOPS
```

---

# Congratulations! 🎉

You now understand:
1. How a MAC PE performs multiply-accumulate
2. How systolic arrays orchestrate matrix multiplication
3. How the VPU handles activation functions
4. How DMA moves data efficiently
5. How SRAM provides fast local storage
6. How the LCP executes programs
7. How it all connects together

The beauty of this design is its **modularity** - each piece is simple, but they combine to create a powerful AI accelerator!

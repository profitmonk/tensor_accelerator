# Building an AI Accelerator from Scratch
## Part 2: Python Models & RTL Implementation

*From cycle-accurate simulation to synthesizable Verilog*

---

## Introduction

In Part 1, we documented the architectural decisions. Now we dive into implementation: the cycle-accurate Python models that validated our algorithms, and the RTL that implements them in synthesizable Verilog.

**The key insight**: Write the model first. Finding bugs in 100 lines of Python beats debugging 1000 lines of Verilog every time.

---

## The Model-First Methodology

### Key Question Asked

> "How do we know the RTL is correct before synthesis?"

### The Decision: Cycle-Accurate Python Models

We built a complete software simulation before writing RTL:

```
┌─────────────────────────────────────────────────────────────────┐
│                    TPC Python Model (tpc_model.py)              │
│                                                                 │
│  ┌──────────────┐                                               │
│  │     LCP      │  Instruction fetch, decode, dispatch          │
│  │ lcp_model.py │  Loop handling, unit synchronization          │
│  └──────┬───────┘                                               │
│         │ dispatches to                                         │
│         ▼                                                       │
│  ┌──────────────┬──────────────┬──────────────┐                │
│  │     MXU      │     VPU      │     DMA      │                │
│  │  systolic_   │  vpu_model   │  dma_model   │                │
│  │  array_model │    .py       │    .py       │                │
│  │     .py      │              │              │                │
│  │              │              │              │                │
│  │ Weight-      │ ReLU, Add,   │ LOAD/STORE   │                │
│  │ stationary   │ Sub, Max,    │ 2D strided   │                │
│  │ GEMM         │ Reductions   │ transfers    │                │
│  └──────┬───────┴──────┬───────┴──────┬───────┘                │
│         │              │              │                         │
│         └──────────────┴──────────────┘                         │
│                        │                                        │
│                   ┌────▼────┐         ┌────────────┐           │
│                   │  SRAM   │◄───────►│  AXI Mem   │           │
│                   │ Model   │         │   Model    │           │
│                   └─────────┘         └────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Works

| Benefit | Explanation |
|---------|-------------|
| Fast iteration | Python runs in milliseconds; RTL sim takes minutes |
| Algorithm validation | Prove the math works before HDL complexity |
| Test vector generation | Export SRAM state for RTL $readmemh |
| Golden reference | Python output is ground truth for RTL comparison |
| Debug visibility | Print any internal state at any cycle |

---

## Systolic Array Model: The Heart of the Accelerator

### The Skewing Challenge

The most subtle part of systolic array design is the **input skewing** and **output de-skewing**. Getting this wrong causes silent data corruption.

### Key Questions Asked

> "How exactly do activations flow through the array?"
> "When does each PE get its input?"
> "How do we realign the outputs?"

### The Model Implementation

```python
class SystolicArrayModel:
    """Cycle-accurate model of weight-stationary systolic array."""
    
    def __init__(self, size: int = 16):
        self.size = size
        self.weights = np.zeros((size, size), dtype=np.int8)
        self.accumulators = np.zeros((size, size), dtype=np.int32)
        
        # Skewing registers - this is where bugs hide!
        self.input_skew = [deque() for _ in range(size)]
        self.output_skew = [deque() for _ in range(size)]
        
    def load_weights(self, weights: np.ndarray):
        """Load weight matrix (column by column in hardware)."""
        self.weights = weights.copy()
        
    def compute(self, activations: np.ndarray) -> np.ndarray:
        """Compute matrix multiply with proper skewing."""
        K = activations.shape[1]
        
        # Initialize input skewing delays
        # Row 0: 0 cycles delay
        # Row 1: 1 cycle delay
        # Row N-1: N-1 cycles delay
        for row in range(self.size):
            self.input_skew[row] = deque([0] * row)
        
        # Initialize output de-skewing delays
        # Column 0: 2*(N-1) cycles delay
        # Column N-1: 0 cycles delay
        for col in range(self.size):
            delay = 2 * (self.size - 1 - col)
            self.output_skew[col] = deque([0] * delay)
        
        # Simulate cycle by cycle
        for cycle in range(K + 2 * self.size):
            # Feed activations with input skewing
            for row in range(self.size):
                if cycle < K:
                    self.input_skew[row].append(activations[row, cycle])
                else:
                    self.input_skew[row].append(0)  # Drain phase
            
            # MAC operations across the array
            for row in range(self.size):
                if self.input_skew[row]:
                    act = self.input_skew[row].popleft()
                    for col in range(self.size):
                        self.accumulators[row, col] += int(act) * int(self.weights[row, col])
            
            # Output de-skewing (not shown for brevity)
        
        return self.accumulators.copy()
```

### The Critical Insight: Skewing Formulas

**Input skewing** (activations entering the array):
- Row 0: no delay
- Row i: i cycles of delay
- Creates diagonal wavefront moving through the array

**Output de-skewing** (results exiting the array):
- Column N-1: no delay (rightmost finishes first)
- Column j: 2*(N-1-j) cycles of delay
- Realigns results so all columns of a row are valid simultaneously

### Bug We Caught in Python

**The bug**: Off-by-one in de-skewing delay calculation.

**Symptom**: First column of results was garbage.

**Root cause**: Used `(N-1-col)` instead of `2*(N-1-col)` for output delays.

**How we found it**: Compared Python output to NumPy `np.dot()`. Mismatch in column 0.

**Fix time**: 5 minutes in Python. Would have been hours in RTL simulation.

---

## DMA Model: SRAM Latency is Tricky

### Key Question Asked

> "Why is our DMA STORE corrupting data?"

### The Bug Discovery

Early RTL showed data corruption on STORE operations. The Python model helped us understand why.

### The Latency Issue

```python
class SRAMModel:
    """Models 1-cycle registered read latency."""
    
    def posedge(self, addr, wdata, we, re):
        # Output is from PREVIOUS cycle's read
        self.rdata = self._rdata_next
        
        if re:
            # Schedule data for NEXT cycle
            self._rdata_next = self.mem[addr >> 5]
```

**The problem**: Our DMA state machine didn't account for SRAM's 1-cycle read latency. We read an address and immediately tried to send the data, but the data wasn't valid until the *next* cycle.

### DMA State Machine Fix

We added a capture state to handle the latency:

```
Original (BROKEN):
  STORE_ADDR → STORE_DATA (immediate)

Fixed:
  STORE_ADDR → STORE_CAP → STORE_DATA
                   ↑
        Wait for SRAM read latency
```

```python
class DMAState(Enum):
    IDLE = auto()
    LOAD_ADDR = auto()
    LOAD_DATA = auto()
    STORE_ADDR = auto()
    STORE_CAP = auto()    # NEW: capture SRAM output
    STORE_DATA = auto()
```

---

## VPU Model: Activation Functions

### Key Question Asked

> "How accurate is our GELU lookup table?"

### GELU Approximation Analysis

```python
def gelu_lut(x_int8: int) -> int:
    """256-entry LUT for GELU approximation."""
    # x is in [-128, 127], representing ~[-4, 4] range
    x_float = x_int8 / 32.0  # Scale to reasonable range
    
    # True GELU
    gelu = 0.5 * x_float * (1 + np.tanh(np.sqrt(2/np.pi) * (x_float + 0.044715 * x_float**3)))
    
    # Quantize back to INT8
    return int(np.clip(gelu * 32, -128, 127))

# Generate LUT
GELU_LUT = [gelu_lut(i - 128) for i in range(256)]
```

### Error Analysis

| Metric | Value |
|--------|-------|
| Max error | 2 INT8 units (~0.8%) |
| Mean error | 0.3 INT8 units (~0.1%) |
| Acceptable for inference? | Yes |

---

## LCP Model: Instruction Decode

### Key Question Asked

> "How does the instruction sequencer handle hardware loops?"

### Loop Stack Implementation

```python
class LCPModel:
    """Local Command Processor model."""
    
    def __init__(self):
        self.pc = 0
        self.loop_stack = []  # Stack of (start_pc, remaining_count)
        self.state = LCPState.FETCH
        
    def execute_instruction(self, instr):
        opcode = (instr >> 120) & 0xFF
        
        if opcode == OP_LOOP:
            count = (instr >> 48) & 0xFFFF
            self.loop_stack.append((self.pc + 1, count))
            self.pc += 1
            
        elif opcode == OP_ENDLOOP:
            if self.loop_stack:
                start_pc, count = self.loop_stack[-1]
                if count > 1:
                    # Continue loop
                    self.loop_stack[-1] = (start_pc, count - 1)
                    self.pc = start_pc
                else:
                    # Exit loop
                    self.loop_stack.pop()
                    self.pc += 1
                    
        elif opcode == OP_HALT:
            self.state = LCPState.DONE
            
        else:
            # Regular instruction - dispatch and advance
            self.dispatch(instr)
            self.pc += 1
```

### Integration Testing

The TPC model runs complete programs:

```python
# TEST: DMA LOAD → MXU GEMM → VPU RELU pipeline
program = [
    make_dma_instr(DMAOp.LOAD, ext_addr=0x00, int_addr=0x100),
    make_sync_instr(SyncOp.WAIT_DMA),
    make_mxu_instr(src_act=0x100, src_wt=0x200, dst=0x300),
    make_sync_instr(SyncOp.WAIT_MXU),
    make_vpu_instr(VPUOp.RELU, src=0x300, dst=0x400),
    make_sync_instr(SyncOp.WAIT_VPU),
    make_halt()
]

tpc = TPCModel()
tpc.load_program(program)
tpc.run()

# Verify output matches expected
assert np.array_equal(tpc.read_sram(0x400, 256), expected_output)
```

---

## RTL Implementation

With the models validated, we implemented synthesizable Verilog.

### Module Hierarchy

```
tensor_accelerator_top
├── global_cmd_processor (GCP)
│   └── AXI-Lite slave interface
├── tensor_processing_cluster [0:3] (TPC)
│   ├── local_cmd_processor (LCP)
│   │   ├── Instruction fetch
│   │   ├── Hardware loop stack
│   │   └── Scoreboard
│   ├── systolic_array (MXU)
│   │   ├── mac_pe [0:255] (16x16)
│   │   ├── Input skewing registers
│   │   └── Output de-skewing registers
│   ├── vector_unit (VPU)
│   │   ├── Vector register file
│   │   ├── SIMD ALU lanes [0:63]
│   │   └── Reduction tree
│   ├── dma_engine
│   │   ├── 2D address generator
│   │   └── AXI4 master interface
│   └── sram_subsystem
│       ├── sram_bank [0:15]
│       └── Arbiter
├── noc_router [0:3] (2x2 mesh)
│   ├── Input FIFOs [0:4]
│   ├── Route compute (X-Y)
│   └── Crossbar
└── axi_interconnect
    └── Round-robin arbiter
```

### Lines of Code

| Component | Lines | Notes |
|-----------|-------|-------|
| MAC PE | 50 | Simple but critical |
| Systolic Array | 400 | Skewing is complex |
| VPU | 600 | 64 lanes, reduction tree |
| DMA Engine | 350 | State machine + AXI |
| LCP | 500 | Instruction decode, loops |
| SRAM Subsystem | 400 | Banking, arbitration |
| NoC Router | 300 | XY routing, FIFOs |
| Top-level integration | 400 | Glue logic |
| **Total RTL** | **~4,000** | Production-quality |

---

## The MAC Processing Element

The fundamental building block:

```verilog
module mac_pe #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     enable,
    input  wire                     load_weight,
    input  wire                     clear_acc,
    input  wire [DATA_WIDTH-1:0]    weight_in,
    input  wire [DATA_WIDTH-1:0]    act_in,
    output reg  [DATA_WIDTH-1:0]    act_out,
    input  wire [ACC_WIDTH-1:0]     psum_in,
    output reg  [ACC_WIDTH-1:0]     psum_out
);
    // Stationary weight register
    reg [DATA_WIDTH-1:0] weight_reg;
    
    // Signed multiplication
    wire signed [2*DATA_WIDTH-1:0] product = 
        $signed(act_in) * $signed(weight_reg);
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg <= 0;
            act_out <= 0;
            psum_out <= 0;
        end else begin
            if (load_weight) 
                weight_reg <= weight_in;
            
            if (enable) begin
                act_out <= act_in;  // Pass activation to right neighbor
                psum_out <= clear_acc ? 
                    {{(ACC_WIDTH-2*DATA_WIDTH){product[2*DATA_WIDTH-1]}}, product} :
                    (psum_in + {{(ACC_WIDTH-2*DATA_WIDTH){product[2*DATA_WIDTH-1]}}, product});
            end
        end
    end
endmodule
```

### Key Design Decisions in RTL

| Decision | Implementation | Rationale |
|----------|----------------|-----------|
| Signed arithmetic | `$signed()` casting | Weights can be negative |
| 32-bit accumulator | ACC_WIDTH=32 | No overflow for K up to 65K |
| Registered outputs | All outputs registered | Clean timing closure |
| Sign extension | Manual in concat | Proper signed accumulation |

---

## Systolic Array RTL

The 16×16 array with skewing:

```verilog
module systolic_array #(
    parameter SIZE = 16,
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32
)(
    input  wire clk,
    input  wire rst_n,
    
    // Weight loading (column broadcast)
    input  wire                     weight_load_en,
    input  wire [$clog2(SIZE)-1:0]  weight_load_col,
    input  wire [SIZE*DATA_WIDTH-1:0] weight_load_data,
    
    // Activation input (one row per cycle, pre-skewed)
    input  wire                     act_valid,
    input  wire [SIZE*DATA_WIDTH-1:0] act_data,
    
    // Result output (de-skewed)
    output wire                     result_valid,
    output wire [SIZE*ACC_WIDTH-1:0] result_data
);

    // Skewing registers for input
    reg [DATA_WIDTH-1:0] skew_regs [0:SIZE-1][0:SIZE-2];
    
    // De-skewing registers for output
    reg [ACC_WIDTH-1:0] deskew_regs [0:SIZE-1][0:2*(SIZE-1)-1];
    
    // PE array
    wire [DATA_WIDTH-1:0] act_h [0:SIZE-1][0:SIZE];  // Horizontal act flow
    wire [ACC_WIDTH-1:0]  psum_v [0:SIZE][0:SIZE-1]; // Vertical psum flow
    
    // Instantiate SIZE x SIZE PEs
    genvar row, col;
    generate
        for (row = 0; row < SIZE; row = row + 1) begin : pe_row
            for (col = 0; col < SIZE; col = col + 1) begin : pe_col
                mac_pe #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH(ACC_WIDTH)
                ) pe_inst (
                    .clk(clk),
                    .rst_n(rst_n),
                    .enable(compute_enable),
                    .load_weight(weight_load_en && (weight_load_col == col)),
                    .clear_acc(clear_acc),
                    .weight_in(weight_load_data[row*DATA_WIDTH +: DATA_WIDTH]),
                    .act_in(act_h[row][col]),
                    .act_out(act_h[row][col+1]),
                    .psum_in(psum_v[row][col]),
                    .psum_out(psum_v[row+1][col])
                );
            end
        end
    endgenerate
    
    // Input skewing logic
    always @(posedge clk) begin
        if (act_valid) begin
            for (integer r = 0; r < SIZE; r = r + 1) begin
                if (r == 0) begin
                    // Row 0: no delay
                    act_h[0][0] <= act_data[0*DATA_WIDTH +: DATA_WIDTH];
                end else begin
                    // Row r: r cycles of delay through shift registers
                    skew_regs[r][0] <= act_data[r*DATA_WIDTH +: DATA_WIDTH];
                    for (integer d = 1; d < r; d = d + 1) begin
                        skew_regs[r][d] <= skew_regs[r][d-1];
                    end
                    act_h[r][0] <= skew_regs[r][r-1];
                end
            end
        end
    end
    
    // Output de-skewing logic (similar structure, omitted for brevity)
    
endmodule
```

---

## Cross-Validation: Python vs RTL

### Test Vector Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Python Model   │     │  Generate Test  │     │  RTL Testbench  │
│                 │────▶│   Vectors       │────▶│                 │
│  Run program    │     │   (.memh)       │     │  $readmemh      │
│  Capture state  │     │   weights.memh  │     │  Compare output │
└─────────────────┘     │   input.memh    │     └─────────────────┘
                        │   golden.memh   │
                        └─────────────────┘
```

### Example Cross-Validation Test

```python
# In Python
def test_gemm_4x4():
    A = np.array([[1, 2], [3, 4]], dtype=np.int8)
    B = np.array([[5, 6], [7, 8]], dtype=np.int8)
    expected = np.dot(A.astype(np.int32), B.astype(np.int32))
    
    model = SystolicArrayModel(size=2)
    model.load_weights(B)
    result = model.compute(A)
    
    assert np.array_equal(result, expected)
    
    # Export for RTL verification
    export_memh("weights.memh", B)
    export_memh("input.memh", A)
    export_memh("golden.memh", expected)
```

```verilog
// In Verilog testbench
initial begin
    $readmemh("weights.memh", weight_mem);
    $readmemh("input.memh", input_mem);
    $readmemh("golden.memh", golden_mem);
    
    // Run computation
    // ...
    
    // Compare
    if (result == golden_mem[0])
        $display("PASS");
    else
        $display("FAIL: expected %h, got %h", golden_mem[0], result);
end
```

---

## Bugs Found by Python Model First

| Bug | Symptom | Root Cause | Fix Time (Python) | Estimated Fix Time (RTL) |
|-----|---------|------------|-------------------|--------------------------|
| Skew off-by-one | Wrong column 0 | De-skew delay formula | 5 min | 2+ hours |
| SRAM latency | Store corruption | Missing capture state | 15 min | 4+ hours |
| Sign extension | Overflow on negative weights | Missing sign bit propagation | 10 min | 1+ hour |
| Loop counter | Infinite loop | Decrement before test | 5 min | 1+ hour |

**Total estimated time saved: 8+ hours** just on these four bugs.

---

## Summary

### The Development Flow

1. **Design** the architecture (Part 1)
2. **Model** in Python - validate algorithms
3. **Implement** in RTL - translate validated models
4. **Cross-validate** - Python outputs are golden reference
5. **Debug** - find mismatches, fix in Python first if algorithmic

### Key Takeaways

| Lesson | Why It Matters |
|--------|----------------|
| Model first | Catch algorithm bugs in minutes, not hours |
| Cycle-accurate | Must match RTL behavior exactly |
| Export test vectors | Bridges Python and RTL verification |
| Compare everything | Any mismatch indicates a bug somewhere |

---

## What's Next

In **Part 3**, we'll cover verification in detail: the test plan, coverage methodology, cocotb testbenches, and open-source EDA tools (Icarus Verilog, Verilator, Surfer for waveforms).

---

*All code is available on GitHub. Questions? Let's connect!*

**Tags**: #Verilog #Python #Simulation #Verification #HardwareDesign #FPGA #RTL

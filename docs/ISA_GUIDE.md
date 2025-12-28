# Tensor Accelerator Instruction Set Architecture (ISA)

## Overview

The tensor accelerator uses a custom 128-bit instruction format executed by the Local Command Processor (LCP). This document explains:

1. How assembly (.asm) files work
2. The instruction encoding format
3. How the assembler compiles to binary
4. How the LCP executes instructions

---

## The Compilation Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Source Code   │     │    Assembler    │     │  Binary/Hex     │
│   (.asm file)   │────▶│   (Python)      │────▶│  Instructions   │
│                 │     │                 │     │                 │
│ TENSOR.GEMM ... │     │ Parse + Encode  │     │ 01010800...     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │  Instruction    │
                                               │    Memory       │
                                               │  (SRAM/BRAM)    │
                                               └────────┬────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │  LCP Fetches    │
                                               │  & Executes     │
                                               └─────────────────┘
```

---

## Instruction Format (128 bits)

Each instruction is exactly 128 bits (16 bytes):

```
Bit Position:  127    120 119    112 111      96 95       80 79       64
               ┌────────┬────────┬───────────┬───────────┬───────────┐
               │ Opcode │ Subop  │    Dst    │   Src0    │   Src1    │
               │ (8b)   │ (8b)   │   (16b)   │   (16b)   │   (16b)   │
               └────────┴────────┴───────────┴───────────┴───────────┘

Bit Position:  63       48 47       32 31       16 15        0
               ┌───────────┬───────────┬───────────┬───────────┐
               │   Dim_M   │   Dim_N   │   Dim_K   │   Flags   │
               │   (16b)   │   (16b)   │   (16b)   │   (16b)   │
               └───────────┴───────────┴───────────┴───────────┘
```

### Field Descriptions

| Field | Bits | Description |
|-------|------|-------------|
| Opcode | 8 | Operation type (TENSOR, VECTOR, DMA, etc.) |
| Subop | 8 | Sub-operation (GEMM, RELU, LOAD_2D, etc.) |
| Dst | 16 | Destination address or register |
| Src0 | 16 | First source address or register |
| Src1 | 16 | Second source address or register |
| Dim_M | 16 | First dimension (rows, loop count, etc.) |
| Dim_N | 16 | Second dimension (columns) |
| Dim_K | 16 | Third dimension (reduction) |
| Flags | 16 | Operation-specific flags |

---

## Opcodes

| Opcode | Value | Description |
|--------|-------|-------------|
| NOP | 0x00 | No operation |
| TENSOR | 0x01 | Matrix operations (systolic array) |
| VECTOR | 0x02 | Vector/SIMD operations (VPU) |
| DMA | 0x03 | Data movement operations |
| SYNC | 0x04 | Synchronization primitives |
| LOOP | 0x05 | Hardware loop start |
| ENDLOOP | 0x06 | Hardware loop end |
| BARRIER | 0x07 | Multi-TPC barrier |
| HALT | 0xFF | Stop execution |

---

## Example: Step-by-Step Encoding

### Assembly Line
```asm
TENSOR.GEMM  OUT_BUF, ACT_BUF, WT_BUF, 16, 16, 16, 1
```

### What It Means
- **TENSOR.GEMM**: Matrix multiply on systolic array
- **OUT_BUF (0x0800)**: Destination address in SRAM
- **ACT_BUF (0x0000)**: Activation source
- **WT_BUF (0x0400)**: Weight source
- **16, 16, 16**: Matrix dimensions M×N×K
- **1**: Flags (accumulate mode)

### Encoding Process

```python
# Step 1: Look up opcode
opcode = 0x01  # TENSOR

# Step 2: Look up subop
subop = 0x01   # GEMM

# Step 3: Parse addresses
dst  = 0x0800  # OUT_BUF
src0 = 0x0000  # ACT_BUF  
src1 = 0x0400  # WT_BUF

# Step 4: Parse dimensions
dim_m = 16
dim_n = 16
dim_k = 16

# Step 5: Parse flags
flags = 1

# Step 6: Pack into 128 bits (big-endian)
instruction = struct.pack('>BBHHHHHHH',
    opcode,  # 0x01
    subop,   # 0x01
    dst,     # 0x0800
    src0,    # 0x0000
    src1,    # 0x0400
    dim_m,   # 0x0010
    dim_n,   # 0x0010
    dim_k,   # 0x0010
    flags    # 0x0001
)
```

### Binary Result
```
Hex: 01 01 08 00 00 00 04 00 00 10 00 10 00 10 00 01
     ── ── ───── ───── ───── ───── ───── ───── ─────
     op sub dst   src0  src1  dim_m dim_n dim_k flags
```

---

## TENSOR Instructions (Systolic Array)

### TENSOR.GEMM
Matrix multiply: `C = A × B`

```asm
TENSOR.GEMM  dst, src_act, src_wt, M, N, K, flags
```

| Field | Value |
|-------|-------|
| Opcode | 0x01 |
| Subop | 0x01 (GEMM) |
| Dst | Output SRAM address |
| Src0 | Activation SRAM address |
| Src1 | Weight SRAM address |
| Dim_M | Output rows |
| Dim_N | Output columns |
| Dim_K | Reduction dimension |

### TENSOR.GEMM_ACC
Matrix multiply with accumulation: `C += A × B`

```asm
TENSOR.GEMM_ACC  dst, src_act, src_wt, M, N, K, flags
```
Same encoding as GEMM, but Subop = 0x02

---

## VECTOR Instructions (VPU)

### Element-wise Operations

```asm
VEC.RELU   vd, vs        # vd = max(0, vs)
VEC.GELU   vd, vs        # vd = GELU(vs)
VEC.ADD    vd, vs1, vs2  # vd = vs1 + vs2
VEC.MUL    vd, vs1, vs2  # vd = vs1 × vs2
```

| Field | Value |
|-------|-------|
| Opcode | 0x02 |
| Subop | Operation code |
| Dst | Destination register/address |
| Src0 | First source |
| Src1 | Second source (if needed) |

### Subop Codes

| Subop | Value | Operation |
|-------|-------|-----------|
| ADD | 0x01 | vd = vs1 + vs2 |
| SUB | 0x02 | vd = vs1 - vs2 |
| MUL | 0x03 | vd = vs1 × vs2 |
| RELU | 0x10 | vd = max(0, vs) |
| GELU | 0x11 | vd = GELU(vs) |
| SIGMOID | 0x13 | vd = σ(vs) |
| SUM | 0x20 | vd = Σ(vs) |
| LOAD | 0x30 | vd = mem[addr] |
| STORE | 0x31 | mem[addr] = vs |

---

## DMA Instructions

### DMA.LOAD_2D
Load 2D tile from external memory to SRAM

```asm
DMA.LOAD_2D  dst, src, rows, cols, src_stride, dst_stride
```

| Field | Value |
|-------|-------|
| Opcode | 0x03 |
| Subop | 0x01 (LOAD_2D) |
| Dst | SRAM destination address |
| Src0 | External memory address (lower 16 bits) |
| Dim_M | Number of rows |
| Dim_N | Number of columns |
| Dim_K | Source stride |
| Src1 | Destination stride |

### DMA.STORE_2D
Store 2D tile from SRAM to external memory

```asm
DMA.STORE_2D  dst, src, rows, cols, dst_stride, src_stride
```

---

## SYNC Instructions

Wait for hardware units to complete:

```asm
SYNC.WAIT_MXU   # Wait for systolic array
SYNC.WAIT_VPU   # Wait for vector unit
SYNC.WAIT_DMA   # Wait for DMA
SYNC.WAIT_ALL   # Wait for everything
```

| Subop | Value |
|-------|-------|
| WAIT_MXU | 0x01 |
| WAIT_VPU | 0x02 |
| WAIT_DMA | 0x03 |
| WAIT_ALL | 0xFF |

---

## Control Flow

### Hardware Loops

```asm
LOOP        64          # Start loop, 64 iterations
    # ... loop body ...
ENDLOOP                 # End of loop
```

The LCP maintains a loop stack that tracks:
- Loop start address (PC to jump back to)
- Remaining iteration count

### Nested Loops

```asm
LOOP        N_TILES     # Outer loop
    LOOP    M_TILES     # Middle loop
        LOOP K_TILES    # Inner loop
            # ... compute ...
        ENDLOOP
    ENDLOOP
ENDLOOP
```

Maximum nesting depth: 4 levels

---

## Memory Map

### SRAM Addresses (16-bit)

| Symbol | Address | Size | Purpose |
|--------|---------|------|---------|
| SRAM_ACT_A | 0x0000 | 4KB | Activation buffer A |
| SRAM_ACT_B | 0x1000 | 4KB | Activation buffer B |
| SRAM_WT_A | 0x2000 | 8KB | Weight buffer A |
| SRAM_WT_B | 0x4000 | 8KB | Weight buffer B |
| SRAM_OUT | 0x6000 | 4KB | Output buffer |
| SRAM_SCRATCH | 0x7000 | 4KB | Scratch space |
| SRAM_VEC | 0x8000 | 4KB | Vector registers |

### External Memory (40-bit)

```asm
.equ    HBM_WEIGHTS,    0x80000000
.equ    HBM_ACTIVATIONS,0x81000000
.equ    HBM_OUTPUT,     0x82000000
```

---

## Using the Assembler

### Command Line

```bash
# Compile to hex (for Verilog $readmemh)
python3 sw/assembler/assembler.py input.asm -o output.hex

# Compile to binary
python3 sw/assembler/assembler.py input.asm -o output.bin -f bin

# Compile to Xilinx COE format
python3 sw/assembler/assembler.py input.asm -o output.coe -f coe
```

### Output Formats

**Hex format (.hex)** - For Verilog simulation:
```
01010800000004000010001000100001  // 0000
02100000080000000000000000000000  // 0001
ff000000000000000000000000000000  // 0002
```

**Binary format (.bin)** - Raw bytes, 16 bytes per instruction

**COE format (.coe)** - For Xilinx Block RAM initialization:
```
memory_initialization_radix=16;
memory_initialization_vector=
01010800000004000010001000100001,
02100000080000000000000000000000,
ff000000000000000000000000000000;
```

---

## Loading Instructions into Hardware

### In Simulation (Verilog)

```verilog
module instruction_memory (
    input  wire        clk,
    input  wire [19:0] addr,
    output reg [127:0] data
);
    // 4K instruction slots
    reg [127:0] mem [0:4095];
    
    // Load from hex file
    initial begin
        $readmemh("program.hex", mem);
    end
    
    always @(posedge clk) begin
        data <= mem[addr];
    end
endmodule
```

### On FPGA (Vivado)

1. **Block RAM Initialization**:
   - Use COE file in Block RAM IP configuration
   - Or use `$readmemh` in synthesis

2. **Runtime Loading**:
   - Host CPU writes instructions via AXI-Lite
   - DMA transfer from DDR to instruction SRAM

```c
// Host driver code
void load_program(uint32_t* instructions, int count) {
    for (int i = 0; i < count; i++) {
        // Each instruction is 128 bits = 4 words
        writel(instructions[i*4 + 0], INSTR_MEM_BASE + i*16 + 0);
        writel(instructions[i*4 + 1], INSTR_MEM_BASE + i*16 + 4);
        writel(instructions[i*4 + 2], INSTR_MEM_BASE + i*16 + 8);
        writel(instructions[i*4 + 3], INSTR_MEM_BASE + i*16 + 12);
    }
}
```

---

## LCP Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    LCP State Machine                            │
│                                                                 │
│   ┌──────┐    ┌───────┐    ┌────────┐    ┌──────┐    ┌──────┐  │
│   │ IDLE │───▶│ FETCH │───▶│ DECODE │───▶│ EXEC │───▶│ WAIT │  │
│   └──────┘    └───────┘    └────────┘    └──────┘    └──┬───┘  │
│       ▲                                        │         │      │
│       │                                        │         │      │
│       └────────────────────────────────────────┴─────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Execution Steps

1. **FETCH**: Read 128-bit instruction from `mem[PC]`
2. **DECODE**: Extract opcode, operands, dimensions
3. **EXECUTE**: 
   - Configure hardware unit (MXU/VPU/DMA)
   - Assert start signal
   - Increment PC (or branch for loops)
4. **WAIT**: For long operations, wait for done signal
5. **Repeat** until HALT

### Loop Handling

```verilog
// When LOOP instruction is decoded:
if (opcode == OP_LOOP) begin
    loop_stack[loop_depth].start_pc <= pc + 1;
    loop_stack[loop_depth].count <= dim_n;  // iteration count
    loop_depth <= loop_depth + 1;
end

// When ENDLOOP instruction is decoded:
if (opcode == OP_ENDLOOP) begin
    if (loop_stack[loop_depth-1].count > 1) begin
        pc <= loop_stack[loop_depth-1].start_pc;  // jump back
        loop_stack[loop_depth-1].count <= loop_stack[loop_depth-1].count - 1;
    end else begin
        loop_depth <= loop_depth - 1;  // exit loop
        pc <= pc + 1;
    end
end
```

---

## Complete Example

### Assembly Program

```asm
# Simple 16×16 matrix multiply with ReLU
# C = ReLU(A × B)

.equ    ACT_BUF,    0x0000
.equ    WT_BUF,     0x0400
.equ    OUT_BUF,    0x0800

main:
    # Load activation tile (16×16)
    DMA.LOAD_2D     ACT_BUF, 0x80000000, 16, 16, 256, 16
    SYNC.WAIT_DMA
    
    # Load weight tile (16×16)
    DMA.LOAD_2D     WT_BUF, 0x80010000, 16, 16, 256, 16
    SYNC.WAIT_DMA
    
    # Matrix multiply
    TENSOR.GEMM     OUT_BUF, ACT_BUF, WT_BUF, 16, 16, 16, 0
    SYNC.WAIT_MXU
    
    # Apply ReLU
    VEC.LOAD        v0, OUT_BUF, 256
    VEC.RELU        v0, v0
    VEC.STORE       v0, OUT_BUF, 256
    
    # Store result
    DMA.STORE_2D    0x80020000, OUT_BUF, 16, 16, 256, 16
    SYNC.WAIT_DMA
    
    HALT
```

### Compiled Output

```
# Instruction  0: DMA.LOAD_2D ACT_BUF, 0x80000000, 16, 16, 256, 16
03010000000000000010001001000010

# Instruction  1: SYNC.WAIT_DMA
04030000000000000000000000000000

# Instruction  2: DMA.LOAD_2D WT_BUF, 0x80010000, 16, 16, 256, 16
03010400000100000010001001000010

# Instruction  3: SYNC.WAIT_DMA
04030000000000000000000000000000

# Instruction  4: TENSOR.GEMM OUT_BUF, ACT_BUF, WT_BUF, 16, 16, 16, 0
01010800000004000010001000100000

# Instruction  5: SYNC.WAIT_MXU
04010000000000000000000000000000

# Instruction  6: VEC.LOAD v0, OUT_BUF, 256
02300000080000000100000000000000

# Instruction  7: VEC.RELU v0, v0
02100000000000000000000000000000

# Instruction  8: VEC.STORE v0, OUT_BUF, 256
02310000080000000100000000000000

# Instruction  9: DMA.STORE_2D 0x80020000, OUT_BUF, 16, 16, 256, 16
03020002080000000010001001000010

# Instruction 10: SYNC.WAIT_DMA
04030000000000000000000000000000

# Instruction 11: HALT
ff000000000000000000000000000000
```

---

## Summary

| Stage | Tool | Input | Output |
|-------|------|-------|--------|
| Write | Text editor | Algorithm | `.asm` file |
| Assemble | `assembler.py` | `.asm` | `.hex` / `.bin` / `.coe` |
| Load | Verilog / Driver | `.hex` | Instruction memory |
| Execute | LCP hardware | Instructions | Tensor operations |

The assembler is the bridge between human-readable programs and the 128-bit binary encoding that the LCP hardware understands.

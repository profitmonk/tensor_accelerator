# Building an AI Accelerator from Scratch
## Part 4: Software Flow — From ONNX Model to Binary Execution

*The complete journey from a trained neural network to instructions running on custom silicon*

---

## Introduction

This is the final part of our series on building an AI accelerator from scratch. We've covered the architecture (Part 1), RTL implementation (Part 2), and verification (Part 3). Now we tackle the software: **how do you get a trained neural network to run on this hardware?**

The gap between "trained model" and "running on accelerator" is where many hardware projects die. You can have beautiful RTL, but without a working toolchain, it's just expensive paperwork.

**Our Approach**: Build a complete compilation stack from ONNX → assembly → binary, with Python golden models generating test vectors at every stage.

---

## The Full Picture: ONNX to Execution

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT SOURCES                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ONNX Model          PyTorch Model         Hand-built IR Graph    │
│   (.onnx)             (via export)          (Python API)           │
│       │                    │                      │                 │
│       └────────────────────┼──────────────────────┘                 │
│                            │                                        │
│                            ▼                                        │
│                  ┌─────────────────┐                               │
│                  │   IR Graph      │                               │
│                  │   (in memory)   │                               │
│                  └────────┬────────┘                               │
│                           │                                        │
└───────────────────────────┼─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      COMPILER PIPELINE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐              │
│   │  Quantizer  │──▶│   Tiler     │──▶│  Scheduler  │              │
│   │ FP32→INT8   │   │ Break into  │   │ Order ops,  │              │
│   │             │   │ HW tiles    │   │ assign TPCs │              │
│   └─────────────┘   └─────────────┘   └──────┬──────┘              │
│                                              │                      │
│                                              ▼                      │
│                                       ┌─────────────┐              │
│                                       │  CodeGen    │              │
│                                       │ Emit ASM    │              │
│                                       └──────┬──────┘              │
│                                              │                      │
└──────────────────────────────────────────────┼──────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         ASSEMBLER                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   program.asm ────▶ Assembler ────▶ program.hex                    │
│   (human-readable)               (128-bit binary)                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      RTL SIMULATION                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Test Package:                                                     │
│   ├── program.hex   ──▶  Instruction Memory                        │
│   ├── weights.memh  ──▶  Weight SRAM                               │
│   ├── input.memh    ──▶  Input SRAM                                │
│   └── golden.memh   ──▶  Expected Output (for comparison)          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: The Frontend — Parsing ONNX Models

### What is ONNX?

ONNX (Open Neural Network Exchange) is the lingua franca of trained models. PyTorch, TensorFlow, and most frameworks can export to ONNX. Our compiler accepts ONNX as input.

### The ONNX Parser

**Location**: `sw/compiler/frontend/onnx_parser.py`

The parser walks the ONNX graph and converts each operator to our internal representation (IR):

```python
def load_onnx(path: str, verbose: bool = False) -> IRGraph:
    """Load ONNX model and convert to IR."""
    model = onnx.load(path)
    graph = IRGraph()
    
    for node in model.graph.node:
        op_type = node.op_type
        
        if op_type == "Gemm":
            ir_op = parse_gemm(node)
        elif op_type == "Conv":
            ir_op = parse_conv(node)  # Converts to im2col + GEMM
        elif op_type == "Relu":
            ir_op = parse_relu(node)
        elif op_type == "MatMul":
            ir_op = parse_matmul(node)
        # ... more ops
        
        graph.add_operation(ir_op)
    
    return graph
```

### Supported Operations

| ONNX Op | IR OpType | How We Handle It |
|---------|-----------|------------------|
| Gemm | GEMM | Direct mapping to systolic array |
| MatMul | MATMUL | Direct mapping to systolic array |
| Conv | CONV2D | im2col transform → GEMM |
| Relu | RELU | VPU element-wise op |
| Add/Sub/Mul | ADD/SUB/MUL | VPU element-wise ops |
| Softmax | SOFTMAX | 3-pass: find_max → exp → normalize |
| MaxPool | MAXPOOL | VPU with compare operations |

### The im2col Transform: Making Conv into GEMM

Convolutions dominate CNN compute, but our systolic array only does GEMM. The solution: **im2col** (image to column) transformation.

```
Input: 5×5 image, 3×3 kernel

Original convolution:
┌─────────────┐     ┌───────┐
│ a b c d e   │     │ 1 2 3 │
│ f g h i j   │  *  │ 4 5 6 │  = 3×3 output
│ k l m n o   │     │ 7 8 9 │
│ p q r s t   │     └───────┘
│ u v w x y   │
└─────────────┘

After im2col:
Input matrix (9×9)          Weight column (9×1)     Output (9×1)
┌─────────────────────┐     ┌───┐                   ┌───┐
│ a b c f g h k l m   │     │ 1 │                   │   │
│ b c d g h i l m n   │     │ 2 │                   │   │
│ c d e h i j m n o   │     │ 3 │                   │   │
│ f g h k l m p q r   │  ×  │ 4 │                 = │   │
│ g h i l m n q r s   │     │ 5 │                   │ 9 │
│ h i j m n o r s t   │     │ 6 │                   │   │
│ k l m p q r u v w   │     │ 7 │                   │   │
│ l m n q r s v w x   │     │ 8 │                   │   │
│ m n o r s t w x y   │     │ 9 │                   │   │
└─────────────────────┘     └───┘                   └───┘

Now it's just a GEMM: (9×9) × (9×1) = (9×1)
```

This transformation happens at compile time in the tiler, turning every Conv into a GEMM that maps directly to our systolic array.

---

## Stage 2: Quantization — FP32 to INT8

### Why Quantize?

Trained models use FP32 (or FP16). Our accelerator uses INT8 for efficiency:
- 4× smaller weights
- 4× more compute per watt
- ~95% of FP32 accuracy for inference

### The Quantization Formula

```
Q(x) = round(x / scale) + zero_point

Where:
  scale = (max(x) - min(x)) / 255  (for uint8)
  scale = (max(|x|)) / 127         (for symmetric int8)
```

We use **symmetric quantization** (zero_point = 0) for simplicity:

```python
class Quantizer:
    def quantize_tensor(self, tensor: np.ndarray) -> Tuple[np.ndarray, float]:
        """Quantize FP32 tensor to INT8."""
        max_val = np.max(np.abs(tensor))
        scale = max_val / 127.0
        
        q_tensor = np.round(tensor / scale).astype(np.int8)
        
        return q_tensor, scale
```

### Quantization Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| Per-tensor | One scale for entire tensor | Simple, less accurate |
| Per-channel | One scale per output channel | Better for weights |
| Dynamic | Compute scale at runtime | Best accuracy, more complex |

We implement per-channel quantization for weights:

```python
def quantize_per_channel(weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize weights with per-output-channel scales."""
    out_channels = weights.shape[0]
    scales = np.zeros(out_channels)
    q_weights = np.zeros_like(weights, dtype=np.int8)
    
    for oc in range(out_channels):
        channel_weights = weights[oc]
        scales[oc] = np.max(np.abs(channel_weights)) / 127.0
        q_weights[oc] = np.round(channel_weights / scales[oc]).astype(np.int8)
    
    return q_weights, scales
```

---

## Stage 3: Tiling — Breaking Big into Hardware-Sized Pieces

### The Problem

Our systolic array is 8×8. What if the GEMM is 1024×1024×1024? We need to tile.

### Tiling Strategy

For GEMM (C = A × B), we tile in three dimensions:

```
Original: A[M,K] × B[K,N] = C[M,N]
Tiled:    A[m,k] × B[k,n] = C[m,n]  (accumulated over k tiles)

Tile size selection:
  - m: Limited by activation SRAM (256KB → 256K elements → m ≤ 512)
  - n: Limited by output SRAM (256KB → n ≤ 512)
  - k: Trade-off between accumulation passes and memory usage
```

### Tiling Code

```python
class Tiler:
    def __init__(self, hw_config):
        self.systolic_size = hw_config['systolic_size']  # 8
        self.sram_size = hw_config['sram_per_tpc']        # 2MB
        
    def tile_gemm(self, M: int, N: int, K: int) -> List[GemmTile]:
        """Generate tile schedule for GEMM."""
        tiles = []
        
        # Round up to systolic array size
        tile_m = min(self.systolic_size * 8, M)   # 64
        tile_n = min(self.systolic_size * 8, N)   # 64
        tile_k = min(256, K)  # Accumulate in chunks
        
        for m_start in range(0, M, tile_m):
            for n_start in range(0, N, tile_n):
                for k_start in range(0, K, tile_k):
                    tile = GemmTile(
                        m_start=m_start, m_size=min(tile_m, M - m_start),
                        n_start=n_start, n_size=min(tile_n, N - n_start),
                        k_start=k_start, k_size=min(tile_k, K - k_start),
                        accumulate=(k_start > 0)  # Accumulate after first k tile
                    )
                    tiles.append(tile)
        
        return tiles
```

### Example: Tiling a Large GEMM

```
GEMM(256×256×256):
  Tile size: 64×64×64
  Number of tiles: 4×4×4 = 64 tiles

Execution order (K-first for accumulation):
  Tile[0,0,0] → Tile[0,0,1] → ... → Tile[0,0,3]  (first output tile complete)
  Tile[0,1,0] → Tile[0,1,1] → ... → Tile[0,1,3]  (second output tile complete)
  ...
```

---

## Stage 4: Scheduling — Who Does What, When

### The Scheduler's Job

Given a tiled graph, decide:
1. Which TPC executes each tile
2. What order tiles execute
3. Where in SRAM to place data
4. When to transfer data via DMA

### Single-TPC Scheduling

For single-TPC execution, scheduling is straightforward:

```python
def schedule_single_tpc(tiles: List[Tile]) -> Schedule:
    schedule = Schedule()
    sram_allocator = SRAMAllocator(size=2*1024*1024)  # 2MB
    
    for tile in tiles:
        # Allocate SRAM regions
        weight_addr = sram_allocator.alloc(tile.weight_size)
        act_addr = sram_allocator.alloc(tile.activation_size)
        out_addr = sram_allocator.alloc(tile.output_size)
        
        # Add DMA operations
        schedule.add(DMALoad(dst=weight_addr, src=tile.weight_ddr_addr))
        schedule.add(DMALoad(dst=act_addr, src=tile.activation_ddr_addr))
        schedule.add(Sync(WAIT_DMA))
        
        # Add compute
        schedule.add(GEMM(
            dst=out_addr, src_a=act_addr, src_b=weight_addr,
            M=tile.m_size, N=tile.n_size, K=tile.k_size,
            accumulate=tile.accumulate
        ))
        schedule.add(Sync(WAIT_MXU))
        
        # Store if this is last K tile
        if tile.is_last_k:
            schedule.add(DMAStore(dst=tile.output_ddr_addr, src=out_addr))
    
    return schedule
```

### Multi-TPC Scheduling (Advanced)

For 4 TPCs, we can parallelize across M or N tiles:

```
GEMM tiled into 4×4 = 16 output tiles
Assign to 4 TPCs:

TPC0: tiles (0,0), (0,1), (0,2), (0,3)
TPC1: tiles (1,0), (1,1), (1,2), (1,3)
TPC2: tiles (2,0), (2,1), (2,2), (2,3)
TPC3: tiles (3,0), (3,1), (3,2), (3,3)

Each TPC processes its tiles independently
Final result gathered in DDR
```

---

## Stage 5: Code Generation — Emitting Assembly

### The Assembly Language

Our custom ISA has 128-bit instructions with this format:

```
[127:120] opcode     (8 bits)   - Instruction type
[119:112] subop      (8 bits)   - Operation variant
[111:96]  dst        (16 bits)  - Destination address
[95:80]   src0       (16 bits)  - Source 0 address
[79:64]   src1       (16 bits)  - Source 1 address
[63:48]   dim_m      (16 bits)  - M dimension
[47:32]   dim_n      (16 bits)  - N dimension
[31:16]   dim_k      (16 bits)  - K dimension
[15:0]    flags      (16 bits)  - Operation flags
```

### Assembly Syntax

```asm
# Matrix multiplication test
.equ WEIGHT_ADDR, 0x00100000      # DDR address for weights
.equ SRAM_ACT_A, 0x0000           # SRAM address for activations
.equ SRAM_WT_A, 0x2000            # SRAM address for weights
.equ SRAM_OUT, 0x6000             # SRAM address for output

# Load weights from DDR to SRAM
DMA.LOAD_1D SRAM_WT_A, WEIGHT_ADDR, 256
SYNC.WAIT_DMA

# Load activations
DMA.LOAD_1D SRAM_ACT_A, 0x00200000, 256
SYNC.WAIT_DMA

# Execute GEMM: OUT = ACT × WT
TENSOR.GEMM SRAM_OUT, SRAM_ACT_A, SRAM_WT_A, 16, 16, 16
SYNC.WAIT_MXU

# Apply ReLU activation
VECTOR.RELU SRAM_OUT, SRAM_OUT, 256
SYNC.WAIT_VPU

# Store result to DDR
DMA.STORE_1D 0x00300000, SRAM_OUT, 256
SYNC.WAIT_DMA

HALT
```

### Code Generator Implementation

```python
class CodeGen:
    def __init__(self):
        self.instructions = []
        self.symbols = {}
        
    def emit(self, instr: str):
        self.instructions.append(instr)
        
    def generate(self, schedule: Schedule) -> str:
        # Header with symbol definitions
        for name, value in self.symbols.items():
            self.emit(f".equ {name}, 0x{value:08X}")
        self.emit("")
        
        # Generate instructions for each operation
        for op in schedule.operations:
            if isinstance(op, DMALoad):
                self.emit(f"DMA.LOAD_1D 0x{op.dst:04X}, 0x{op.src:08X}, {op.size}")
            elif isinstance(op, DMAStore):
                self.emit(f"DMA.STORE_1D 0x{op.dst:08X}, 0x{op.src:04X}, {op.size}")
            elif isinstance(op, GEMM):
                variant = "GEMM_ACC" if op.accumulate else "GEMM"
                self.emit(f"TENSOR.{variant} 0x{op.dst:04X}, 0x{op.src_a:04X}, 0x{op.src_b:04X}, {op.M}, {op.N}, {op.K}")
            elif isinstance(op, RELU):
                self.emit(f"VECTOR.RELU 0x{op.dst:04X}, 0x{op.src:04X}, {op.count}")
            elif isinstance(op, Sync):
                self.emit(f"SYNC.{op.wait_type}")
        
        self.emit("HALT")
        
        return "\n".join(self.instructions)
```

---

## Stage 6: Assembly — From Text to Binary

### The Assembler

The assembler converts human-readable assembly to 128-bit binary instructions.

```python
class Assembler:
    OPCODES = {
        'TENSOR': 0x01,
        'VECTOR': 0x02,
        'DMA':    0x03,
        'SYNC':   0x04,
        'HALT':   0xFF,
    }
    
    TENSOR_SUBOPS = {
        'GEMM':      0x01,
        'GEMM_ACC':  0x02,
        'GEMM_RELU': 0x03,
    }
    
    def assemble_instruction(self, line: str) -> bytes:
        """Convert one assembly line to 16 bytes."""
        parts = line.replace(',', ' ').split()
        mnemonic = parts[0]
        
        # Parse "CATEGORY.OPERATION"
        category, operation = mnemonic.split('.')
        
        opcode = self.OPCODES[category]
        subop = self.get_subop(category, operation)
        
        # Parse operands based on instruction type
        if category == 'TENSOR':
            dst, src0, src1 = self.parse_addr(parts[1:4])
            M, N, K = int(parts[4]), int(parts[5]), int(parts[6])
            return self.encode(opcode, subop, dst, src0, src1, M, N, K, 0)
            
        elif category == 'DMA':
            # ... similar parsing
            
        elif category == 'SYNC':
            wait_type = {'WAIT_MXU': 1, 'WAIT_VPU': 2, 'WAIT_DMA': 3, 'WAIT_ALL': 0xFF}
            return self.encode(opcode, wait_type[operation], 0, 0, 0, 0, 0, 0, 0)
            
    def encode(self, opcode, subop, dst, src0, src1, m, n, k, flags) -> bytes:
        """Pack fields into 128-bit instruction."""
        instr = (
            (opcode << 120) |
            (subop << 112) |
            (dst << 96) |
            (src0 << 80) |
            (src1 << 64) |
            (m << 48) |
            (n << 32) |
            (k << 16) |
            flags
        )
        return instr.to_bytes(16, byteorder='big')
```

### Output Format

The assembler outputs `.hex` files for Verilog `$readmemh`:

```
01016000000020000010001000100000
04010000000000000000000000000000
ff000000000000000000000000000000
```

Each line is 32 hex characters = 128 bits = one instruction.

---

## Stage 7: Test Package Generation

### What the RTL Needs

For simulation, the RTL testbench needs:

1. **program.hex**: Instructions for the LCP
2. **weights.memh**: Weight data for SRAM
3. **input.memh**: Input activation data
4. **golden.memh**: Expected output (computed by Python)

### The Test Generator

```python
def generate_test_package(test_name: str, M: int, N: int, K: int, seed: int = 42):
    """Generate complete test package for RTL simulation."""
    np.random.seed(seed)
    
    # Generate random INT8 inputs
    A = np.random.randint(-64, 64, (M, K), dtype=np.int8)
    B = np.random.randint(-64, 64, (K, N), dtype=np.int8)
    
    # Compute golden output (INT32 accumulation)
    C = A.astype(np.int32) @ B.astype(np.int32)
    
    # Generate assembly
    compiler = Compiler()
    asm_code = compiler.compile_gemm(M, N, K)
    
    # Assemble to binary
    assembler = Assembler()
    program_hex = assembler.assemble(asm_code)
    
    # Write files
    write_hex(f"{test_name}/program.hex", program_hex)
    write_memh(f"{test_name}/weights.memh", B.flatten())
    write_memh(f"{test_name}/input.memh", A.flatten())
    write_memh(f"{test_name}/golden.memh", C.flatten())
    
    # Write metadata
    with open(f"{test_name}/config.json", 'w') as f:
        json.dump({
            "name": test_name,
            "dimensions": {"M": M, "N": N, "K": K},
            "seed": seed,
        }, f, indent=2)
```

### Memory File Format

For 256-bit wide SRAM (our configuration):

```python
def write_memh(filename: str, data: np.ndarray, width_bytes: int = 32):
    """Write data in $readmemh format."""
    with open(filename, 'w') as f:
        for i in range(0, len(data), width_bytes):
            # Pack 32 bytes into one line
            chunk = data[i:i+width_bytes]
            hex_str = ''.join(f'{b & 0xFF:02x}' for b in chunk)
            f.write(hex_str + '\n')
```

---

## Putting It All Together: End-to-End Example

### Example: Compiling a Simple MLP

Input: 2-layer MLP (64 → 32 → 10)

**Step 1: Create ONNX model (or export from PyTorch)**

```python
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Export to ONNX
model = SimpleMLP()
dummy_input = torch.randn(1, 64)
torch.onnx.export(model, dummy_input, "mlp.onnx")
```

**Step 2: Compile to assembly**

```bash
$ python compile.py mlp.onnx -o mlp.asm -v

Loading ONNX model: mlp.onnx
Parsing operations:
  [0] Gemm: (1,64) x (64,32) -> (1,32)
  [1] Relu: (1,32) -> (1,32)
  [2] Gemm: (1,32) x (32,10) -> (1,10)

Quantizing weights:
  fc1.weight: FP32 -> INT8 (scale=0.0312)
  fc2.weight: FP32 -> INT8 (scale=0.0287)

Generating assembly: mlp.asm
  12 instructions generated
```

**Step 3: Generated assembly**

```asm
# MLP: 64 -> 32 -> 10
# Generated by tensor-accelerator compiler

.equ SRAM_INPUT, 0x0000
.equ SRAM_WT1, 0x1000
.equ SRAM_WT2, 0x2000
.equ SRAM_HIDDEN, 0x3000
.equ SRAM_OUTPUT, 0x4000

# Load weights
DMA.LOAD_1D SRAM_WT1, 0x00100000, 2048    # 64*32 bytes
DMA.LOAD_1D SRAM_WT2, 0x00100800, 320     # 32*10 bytes
SYNC.WAIT_DMA

# Load input
DMA.LOAD_1D SRAM_INPUT, 0x00200000, 64
SYNC.WAIT_DMA

# Layer 1: FC (64 -> 32)
TENSOR.GEMM SRAM_HIDDEN, SRAM_INPUT, SRAM_WT1, 1, 32, 64
SYNC.WAIT_MXU

# ReLU
VECTOR.RELU SRAM_HIDDEN, SRAM_HIDDEN, 32
SYNC.WAIT_VPU

# Layer 2: FC (32 -> 10)
TENSOR.GEMM SRAM_OUTPUT, SRAM_HIDDEN, SRAM_WT2, 1, 10, 32
SYNC.WAIT_MXU

# Store output
DMA.STORE_1D 0x00300000, SRAM_OUTPUT, 10
SYNC.WAIT_DMA

HALT
```

**Step 4: Assemble and generate test package**

```bash
$ python assembler.py mlp.asm -o mlp.hex
Assembled 12 instructions to mlp.hex

$ python generate_test_package.py --asm mlp.asm --seed 42 -o tests/mlp
Generated test package:
  tests/mlp/program.hex   (12 instructions)
  tests/mlp/weights.memh  (2368 bytes)
  tests/mlp/input.memh    (64 bytes)
  tests/mlp/golden.memh   (40 bytes, INT32)
  tests/mlp/config.json
```

**Step 5: Run RTL simulation**

```bash
$ ./run_tests.sh tests/mlp

============================================================
Running test: mlp
============================================================
Loading instruction memory: tests/mlp/program.hex
Loading weight SRAM: tests/mlp/weights.memh
Loading input SRAM: tests/mlp/input.memh

Simulation started...
  Cycle 100: DMA load complete
  Cycle 150: Layer 1 GEMM complete
  Cycle 160: ReLU complete
  Cycle 200: Layer 2 GEMM complete
  Cycle 250: DMA store complete
  Cycle 251: HALT

Comparing output to golden reference...
✅ TEST PASSED: Output matches golden (10 elements, 0 errors)

Total cycles: 251
============================================================
```

---

## Lessons Learned

### 1. Build the Toolchain First

The temptation is to build cool RTL first. Resist it. Without a toolchain:
- You can't generate realistic test cases
- You can't verify against golden models
- You can't demonstrate the system works

Build compiler → assembler → test generator before complex RTL.

### 2. Use Python for Everything Else

Verilog for RTL. Python for everything else:
- Compiler? Python.
- Assembler? Python.
- Test generation? Python.
- Golden models? Python (NumPy).

One language for the entire tool stack = faster iteration.

### 3. Generate Tests from the Compiler

Don't write RTL tests by hand. Have the compiler generate them:
- Compiles model
- Generates assembly
- Produces weight/input/golden files
- All consistent, all correct

### 4. Keep Assembly Human-Readable

Even with a compiler, you'll debug in assembly. Make it readable:
- Meaningful labels (`.equ SRAM_WEIGHTS, 0x1000`)
- Comments explaining what each section does
- Consistent formatting

---

## Summary

The software stack is as important as the hardware. Our complete flow:

1. **Frontend**: Parse ONNX models to IR
2. **Quantizer**: Convert FP32 → INT8 with proper scaling
3. **Tiler**: Break large ops into hardware-sized pieces
4. **Scheduler**: Assign tiles to TPCs, allocate SRAM
5. **CodeGen**: Emit human-readable assembly
6. **Assembler**: Convert to 128-bit binary
7. **Test Generator**: Create RTL test packages with golden references

Total software: ~3,000 lines of Python. Enables compilation of real models to working accelerator code.

---

## Conclusion: What We Built

Across this four-part series, we've covered:

**Part 1: Architecture**
- Systolic array vs. sea of MACs decision
- 4-TPC design with NoC mesh
- Custom 128-bit ISA with hardware loops

**Part 2: Implementation**
- Python models for every module
- ~4,000 lines of RTL
- Cross-validation methodology

**Part 3: Verification**
- 53 tests, 100% pass rate
- Bottom-up verification strategy
- Open-source EDA tooling (Icarus, cocotb, Verilator)

**Part 4: Software**
- ONNX → assembly → binary flow
- ~3,000 lines of compiler/assembler
- Automated test generation

**The Result**: A working AI accelerator prototype—from RTL to running inference—built entirely with open-source tools and documented from first principles.

---

*This project demonstrates what's possible when you invest in both hardware and software together. The best accelerator in the world is useless without a toolchain to program it. Build them together.*

*All code is available for reference. Questions welcome—especially on the tricky parts like systolic skewing, quantization accuracy, and multi-TPC scheduling.*

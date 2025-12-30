# Complex Use Case Test Plan

## Overview

Six complex test cases with paired Python golden models and RTL verification:

| # | Use Case | Python Model | RTL Test | Key Verification |
|---|----------|--------------|----------|------------------|
| 1 | 2-Layer MLP | `model_mlp_2layer.py` | `tb_mlp_2layer.v` | Layer chaining, DDR handoff |
| 2 | Large Tiled GEMM | `model_tiled_gemm.py` | `tb_tiled_gemm_64x64.v` | Output-stationary batching |
| 3 | Batch Processing | `model_batch_inference.py` | `tb_batch_inference.v` | Weight reuse across batches |
| 4 | Attention Score | `model_attention.py` | `tb_attention.v` | Q×K^T, softmax approx, ×V |
| 5 | Multi-Channel Conv2D | `model_conv2d_multi.py` | `tb_conv2d_multichannel.v` | Channel accumulation |
| 6 | Residual Add | `model_residual.py` | `tb_residual_block.v` | Skip connections |

---

## Test 1: 2-Layer MLP

### Description
```
Layer 1: H = ReLU(X × W1 + b1)   [Input → Hidden]
Layer 2: Y = ReLU(H × W2 + b2)   [Hidden → Output]
```

### Parameters
- Input X: 8×8 matrix (batch=8, features=8)
- W1: 8×16 (8 inputs → 16 hidden)
- b1: 16 elements
- W2: 16×4 (16 hidden → 4 outputs)
- b2: 4 elements
- Output Y: 8×4 (batch=8, outputs=4)

### TPC Assignment
- **Phase 1**: TPC0-3 compute H tiles in parallel
- **Phase 2**: TPC0-3 compute Y tiles in parallel
- Synchronization: DDR writeback between phases

### Memory Layout (DDR)
```
0x0000_0000: X input      (8×8 = 64 bytes)
0x0000_1000: W1 weights   (8×16 = 128 bytes)
0x0000_2000: b1 bias      (16×4 = 64 bytes)
0x0000_3000: H hidden     (8×16 = 128 bytes, intermediate)
0x0000_4000: W2 weights   (16×4 = 64 bytes)
0x0000_5000: b2 bias      (4×4 = 16 bytes)
0x0000_6000: Y output     (8×4 = 32 bytes)
```

### Python Golden Model
```python
def mlp_2layer(X, W1, b1, W2, b2):
    H = np.maximum(0, X @ W1 + b1)  # ReLU
    Y = np.maximum(0, H @ W2 + b2)  # ReLU
    return H, Y
```

### RTL Test Flow
1. Load X, W1, b1, W2, b2 to DDR
2. Start Phase 1: Each TPC computes H tiles
3. Barrier: Wait for all TPCs, H in DDR
4. Start Phase 2: Each TPC computes Y tiles
5. Verify Y against Python golden

### Files
- `python/model_mlp_2layer.py`
- `tb/tb_mlp_2layer.v`

---

## Test 2: Large Tiled GEMM (64×64)

### Description
```
C[64×64] = A[64×64] × B[64×64]

Tiled into 8×8 tiles = 64 output tiles total
Each TPC handles 16 output tiles (4×4 region)
K-dimension: 8 tiles to accumulate
```

### Parameters
- A: 64×64 INT8 (4096 bytes)
- B: 64×64 INT8 (4096 bytes)
- C: 64×64 INT32 (16384 bytes)
- Tile size: 8×8
- Tiles per dimension: 8
- K accumulations: 8 partial products per output tile

### TPC Assignment (Output-Stationary)
```
TPC0: C[0:32, 0:32]   → 16 output tiles
TPC1: C[0:32, 32:64]  → 16 output tiles
TPC2: C[32:64, 0:32]  → 16 output tiles
TPC3: C[32:64, 32:64] → 16 output tiles
```

### Memory Layout (DDR)
```
0x0000_0000: A matrix    (64×64 = 4KB)
0x0000_2000: B matrix    (64×64 = 4KB)
0x0000_4000: C matrix    (64×64×4 = 16KB)
```

### SRAM Budget per TPC (2MB available, using ~200KB)
- A tile buffer: 8×8 = 64 bytes × 8 tiles = 512 bytes
- B tile buffer: 8×8 = 64 bytes × 8 tiles = 512 bytes
- Partial product: 8×8×4 = 256 bytes
- Output accumulators: 16 tiles × 256 = 4KB
- Total: ~6KB per TPC

### Python Golden Model
```python
def tiled_gemm_64x64(A, B, tile_size=8):
    C = np.zeros((64, 64), dtype=np.int32)
    for i in range(0, 64, tile_size):
        for j in range(0, 64, tile_size):
            for k in range(0, 64, tile_size):
                C[i:i+tile_size, j:j+tile_size] += \
                    A[i:i+tile_size, k:k+tile_size] @ \
                    B[k:k+tile_size, j:j+tile_size]
    return C
```

### RTL Test Flow
1. Initialize DDR with random A, B matrices
2. Each TPC loops: Load A[i,k], B[k,j] → GEMM → VPU accumulate
3. After K-loop complete, DMA store C tiles to DDR
4. Verify C against numpy A @ B

### Files
- `python/model_tiled_gemm.py`
- `tb/tb_tiled_gemm_64x64.v`

---

## Test 3: Batch Processing

### Description
```
Process N input samples through same weights:
Y[n] = ReLU(X[n] × W + b)  for n = 0..N-1

Weights W and bias b are reused across all batches.
```

### Parameters
- Batch size N: 16 samples
- Input features: 8
- Output features: 8
- X: 16×8 (N samples × features)
- W: 8×8 (shared weights)
- b: 8 (shared bias)
- Y: 16×8 output

### TPC Assignment
```
TPC0: Samples 0-3   (4 samples)
TPC1: Samples 4-7   (4 samples)
TPC2: Samples 8-11  (4 samples)
TPC3: Samples 12-15 (4 samples)
```

### Key Optimization: Weight Broadcast
- W loaded once per TPC, reused for 4 samples
- Demonstrates weight-stationary benefit

### Python Golden Model
```python
def batch_inference(X_batch, W, b):
    # X_batch: [N, in_features]
    # W: [in_features, out_features]
    # b: [out_features]
    Y = np.maximum(0, X_batch @ W + b)
    return Y
```

### RTL Test Flow
1. Load W, b to all TPC SRAMs (broadcast)
2. Stream X samples to assigned TPCs
3. Each TPC processes its samples sequentially
4. Collect Y outputs to DDR
5. Verify against Python golden

### Files
- `python/model_batch_inference.py`
- `tb/tb_batch_inference.v`

---

## Test 4: Attention Score (Simplified)

### Description
```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V

Simplified for hardware:
1. S = Q × K^T           (score matrix)
2. S' = approx_softmax(S) (row-wise normalization)
3. O = S' × V            (output)
```

### Parameters
- Sequence length: 8 tokens
- Head dimension d: 8
- Q, K, V: 8×8 each
- S (scores): 8×8
- O (output): 8×8

### Softmax Approximation (VPU)
```
For each row:
1. Find max value (VPU MAX reduction)
2. Subtract max (numerical stability)
3. Approximate exp() with piecewise linear
4. Sum and normalize
```

Note: Full softmax is complex. We'll implement a simplified version:
- VPU MAX to find row maximum
- VPU SUB to center values
- Use ReLU + linear approx for exp
- VPU reduction for sum

### Python Golden Model
```python
def attention_simplified(Q, K, V, use_approx=True):
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)
    
    if use_approx:
        # Simplified: ReLU-based attention (no true softmax)
        scores = np.maximum(0, scores)
        row_sums = scores.sum(axis=1, keepdims=True) + 1e-6
        attn = scores / row_sums
    else:
        attn = softmax(scores, axis=-1)
    
    output = attn @ V
    return output, scores, attn
```

### TPC Assignment
```
Phase 1: S = Q × K^T     (all TPCs parallel on tiles)
Phase 2: S' = normalize  (VPU ops per row)
Phase 3: O = S' × V      (all TPCs parallel on tiles)
```

### Files
- `python/model_attention.py`
- `tb/tb_attention.v`

---

## Test 5: Multi-Channel Conv2D

### Description
```
Output[Co] = Σ(Ci) Conv2D(Input[Ci], Kernel[Co, Ci])

Multiple input channels → Multiple output channels
Using im2col + GEMM approach
```

### Parameters
- Input: 8×8 spatial, 4 input channels (Ci=4)
- Kernel: 3×3, 4 input channels, 8 output channels (Co=8)
- Output: 6×6 spatial, 8 output channels
- Stride: 1, Padding: 0

### im2col Transformation
```
Input [Ci, H, W] → Patches [Ci×K×K, out_H×out_W]
                 = [4×3×3, 6×6] = [36, 36]

Kernel [Co, Ci, K, K] → Weights [Co, Ci×K×K]
                      = [8, 36]

Output = Weights × Patches = [8, 36]
       → reshape to [8, 6, 6]
```

### Key: Channel Accumulation
Each output channel accumulates across all input channels:
```
for co in range(Co):
    for ci in range(Ci):
        output[co] += conv2d(input[ci], kernel[co, ci])
```

### Python Golden Model
```python
def conv2d_multichannel(input, kernel, bias=None):
    # input: [Ci, H, W]
    # kernel: [Co, Ci, Kh, Kw]
    Ci, H, W = input.shape
    Co, _, Kh, Kw = kernel.shape
    
    # im2col
    patches = im2col(input, Kh, Kw)  # [Ci*Kh*Kw, out_H*out_W]
    weights = kernel.reshape(Co, -1)  # [Co, Ci*Kh*Kw]
    
    # GEMM
    output = weights @ patches  # [Co, out_H*out_W]
    
    if bias is not None:
        output += bias.reshape(-1, 1)
    
    return output.reshape(Co, H-Kh+1, W-Kw+1)
```

### TPC Assignment
- Partition output channels across TPCs
- TPC0: Co 0-1, TPC1: Co 2-3, TPC2: Co 4-5, TPC3: Co 6-7

### Files
- `python/model_conv2d_multi.py`
- `tb/tb_conv2d_multichannel.v`

---

## Test 6: Residual Block

### Description
```
Residual connection (skip connection):
Y = F(X) + X

Where F(X) = ReLU(X × W + b)

If dimensions match: Y = ReLU(X × W + b) + X
If dimensions differ: Y = ReLU(X × W + b) + X × W_skip
```

### Parameters (Matching Dimensions)
- Input X: 8×8
- W: 8×8 (preserves dimensions)
- b: 8
- Output Y: 8×8

### Data Flow
```
X ──────────────────────────┐
│                           │
├─→ GEMM(X, W) ─→ +b ─→ ReLU ─→ ADD ─→ Y
                              ↑
                              │
X ────────────────────────────┘ (skip)
```

### Key Operations
1. Store X in SRAM (for skip connection)
2. Compute F(X) = ReLU(X × W + b)
3. VPU ADD: Y = F(X) + X
4. Output Y

### Python Golden Model
```python
def residual_block(X, W, b):
    # Main path
    F_X = np.maximum(0, X @ W + b)  # ReLU(X @ W + b)
    
    # Skip connection (identity for matching dims)
    Y = F_X + X
    
    return Y, F_X
```

### TPC Assignment
- Single TPC can handle 8×8 residual block
- For larger blocks, tile across TPCs

### Files
- `python/model_residual.py`
- `tb/tb_residual_block.v`

---

## Implementation Order

| Phase | Test | Complexity | Dependencies |
|-------|------|------------|--------------|
| 1 | Residual Add | Low | VPU ADD (done) |
| 2 | Batch Processing | Low | Weight reuse pattern |
| 3 | 2-Layer MLP | Medium | DDR handoff, phases |
| 4 | Large Tiled GEMM | Medium | K-accumulation (done) |
| 5 | Multi-Channel Conv2D | Medium | Channel accumulation |
| 6 | Attention Score | High | Softmax approximation |

---

## Directory Structure

```
tensor_accelerator/
├── python/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_mlp_2layer.py
│   │   ├── model_tiled_gemm.py
│   │   ├── model_batch_inference.py
│   │   ├── model_attention.py
│   │   ├── model_conv2d_multi.py
│   │   └── model_residual.py
│   ├── golden/
│   │   └── generate_golden.py      # Generates test vectors
│   └── run_golden_tests.py         # Runs all Python models
├── tb/
│   ├── tb_mlp_2layer.v
│   ├── tb_tiled_gemm_64x64.v
│   ├── tb_batch_inference.v
│   ├── tb_attention.v
│   ├── tb_conv2d_multichannel.v
│   └── tb_residual_block.v
└── golden_vectors/
    ├── mlp_2layer_golden.hex
    ├── tiled_gemm_golden.hex
    ├── batch_inference_golden.hex
    ├── attention_golden.hex
    ├── conv2d_multi_golden.hex
    └── residual_golden.hex
```

---

## Golden Vector Format

Each test generates a `.hex` file with:
```
# Header
# TEST: <test_name>
# PARAMS: <param1>=<val1>, <param2>=<val2>, ...
# INPUTS: <count>
# OUTPUTS: <count>

# Input data (hex, 32-bit per line)
@INPUT_START
00000001
00000002
...
@INPUT_END

# Expected output (hex, 32-bit per line)
@OUTPUT_START
00000003
00000004
...
@OUTPUT_END
```

RTL testbench reads these files using `$readmemh()`.

---

## Verification Flow

```
┌─────────────────┐     ┌─────────────────┐
│  Python Model   │────→│  Golden Vectors │
│  (numpy-based)  │     │  (.hex files)   │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
┌─────────────────┐     ┌─────────────────┐
│  RTL Testbench  │←────│  $readmemh()    │
│  (Verilog)      │     │  Load vectors   │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│  Compare Output │
│  vs Golden      │
└────────┬────────┘
         │
         ▼
    PASS / FAIL
```

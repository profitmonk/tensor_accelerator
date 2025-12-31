# Complex Use Case Implementation Plan
# 6 Tests: Python Golden Model + RTL Verification

## Status Overview

| # | Use Case | Python Model | RTL Test | Status |
|---|----------|--------------|----------|--------|
| 1 | Residual Add | ✅ Done | ✅ Done | **COMPLETE** |
| 2 | Batch Processing | ✅ Done | 🔄 Debug | In Progress |
| 3 | 2-Layer MLP | ⬚ TODO | ⬚ TODO | Pending |
| 4 | Large Tiled GEMM | ⬚ TODO | ⬚ TODO | Pending |
| 5 | Multi-Channel Conv2D | ⬚ TODO | ⬚ TODO | Pending |
| 6 | Attention Score | ⬚ TODO | ⬚ TODO | Pending |

---

## Implementation Order (by complexity)

### Phase 1: Foundation Tests (Simple VPU + GEMM patterns)
1. **Residual Add** ✅ - Skip connection: Y = ReLU(X×W+b) + X
2. **Batch Processing** 🔄 - Weight reuse: Y[n] = ReLU(X[n]×W+b)

### Phase 2: Multi-Stage Compute
3. **2-Layer MLP** - Layer chaining with DDR handoff
4. **Large Tiled GEMM** - K-accumulation across tiles

### Phase 3: Advanced Patterns
5. **Multi-Channel Conv2D** - Channel accumulation via im2col
6. **Attention Score** - Q×K^T, normalization, ×V

---

## Test 1: Residual Add ✅ COMPLETE

**Files:**
- `python/models/model_residual.py` ✅
- `tb/tb_residual_block.v` ✅

**Verification:** PASSED

---

## Test 2: Batch Processing

**Concept:** Process N samples with shared weights
```
Y[0] = ReLU(X[0] × W + b)
Y[1] = ReLU(X[1] × W + b)
...
Y[N-1] = ReLU(X[N-1] × W + b)
```

**Key Insight:** Single GEMM processes all samples at once!
- X_batch[N×F] × W[F×F] = Z[N×F]
- Each row of Z is one sample's output

**Files:**
- `python/models/model_batch_inference.py` ✅
- `tb/tb_batch_inference.v` 🔄

**Debug Note:** GEMM output correct, need to verify VPU bias add per-row

---

## Test 3: 2-Layer MLP

**Concept:** Two sequential linear layers with activation
```
H = ReLU(X × W1 + b1)    # Layer 1: Input → Hidden
Y = ReLU(H × W2 + b2)    # Layer 2: Hidden → Output
```

**Parameters:**
- X: 4×4 (batch=4, in_features=4)
- W1: 4×8, b1: 8 (hidden_dim=8)
- W2: 8×4, b2: 4 (out_features=4)
- Y: 4×4

**Data Flow:**
1. Load X, W1, b1 to SRAM
2. GEMM: Z1 = X × W1
3. VPU: H = ReLU(Z1 + b1)
4. Store H to SRAM (or keep in place)
5. Load W2, b2
6. GEMM: Z2 = H × W2
7. VPU: Y = ReLU(Z2 + b2)

**Files:**
- `python/models/model_mlp_2layer.py`
- `tb/tb_mlp_2layer.v`

---

## Test 4: Large Tiled GEMM (16×16)

**Concept:** Matrix too large for single GEMM, tiled with K-accumulation
```
C[16×16] = A[16×16] × B[16×16]

Tile size: 4×4
Tiles per dim: 4
K-tiles: 4 partial products per output tile
```

**Algorithm:**
```
for i in [0,1,2,3]:      # Output tile rows
  for j in [0,1,2,3]:    # Output tile cols
    C_tile[i,j] = 0
    for k in [0,1,2,3]:  # K-dimension tiles
      C_tile[i,j] += A_tile[i,k] × B_tile[k,j]
```

**Key Operations:**
- GEMM for each partial product
- VPU ADD for K-accumulation (already verified!)

**Files:**
- `python/models/model_tiled_gemm.py`
- `tb/tb_tiled_gemm_16x16.v`

---

## Test 5: Multi-Channel Conv2D

**Concept:** Convolution with multiple input/output channels via im2col
```
Input:  [Ci, H, W] = [2, 6, 6]
Kernel: [Co, Ci, Kh, Kw] = [2, 2, 3, 3]
Output: [Co, Ho, Wo] = [2, 4, 4]
```

**im2col Transform:**
```
Patches: [Ci×Kh×Kw, Ho×Wo] = [18, 16]
Weights: [Co, Ci×Kh×Kw] = [2, 18]
Output:  Weights × Patches = [2, 16] → reshape [2, 4, 4]
```

**Simplified Test (fits in 4×4 systolic):**
- Input: [2, 4, 4] - 2 channels, 4×4 spatial
- Kernel: 2×2 with 2 input, 2 output channels
- Output: [2, 3, 3]

**Files:**
- `python/models/model_conv2d_multi.py`
- `tb/tb_conv2d_multichannel.v`

---

## Test 6: Attention Score (Simplified)

**Concept:** Core attention mechanism
```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

**Simplified (no true softmax):**
```
S = Q × K^T           # Score matrix
S' = ReLU(S)          # Approximate "attention" (simplified)
O = S' × V            # Output
```

**Parameters:**
- Sequence length: 4
- Head dimension: 4
- Q, K, V: 4×4 each

**True softmax requires:**
- Row-wise max (VPU MAX reduction)
- Subtract max (VPU SUB)
- Exp approximation (piecewise linear or lookup)
- Sum reduction and divide

For POC, we'll use ReLU-attention as approximation.

**Files:**
- `python/models/model_attention.py`
- `tb/tb_attention.v`

---

## Golden Vector Flow

```
┌─────────────────────┐
│  Python Model       │
│  (numpy reference)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  generate_golden()  │
│  - Input matrices   │
│  - Expected outputs │
│  - Intermediate vals│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  golden_vectors/    │
│  - test_X.hex       │
│  - test_W.hex       │
│  - test_Y_golden.hex│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  RTL Testbench      │
│  - $readmemh()      │
│  - Compare outputs  │
└─────────────────────┘
```

---

## File Structure

```
tensor_accelerator/
├── python/
│   └── models/
│       ├── model_residual.py        ✅
│       ├── model_batch_inference.py ✅
│       ├── model_mlp_2layer.py      TODO
│       ├── model_tiled_gemm.py      TODO
│       ├── model_conv2d_multi.py    TODO
│       └── model_attention.py       TODO
├── golden_vectors/
│   ├── residual_*.hex               ✅
│   ├── batch_*.hex                  ✅
│   └── ...
├── tb/
│   ├── tb_residual_block.v          ✅
│   ├── tb_batch_inference.v         🔄
│   ├── tb_mlp_2layer.v              TODO
│   ├── tb_tiled_gemm_16x16.v        TODO
│   ├── tb_conv2d_multichannel.v     TODO
│   └── tb_attention.v               TODO
└── run_tests.sh                     (updated)
```

---

## Next Steps

1. Fix batch processing RTL test
2. Implement 2-Layer MLP (Python + RTL)
3. Implement Large Tiled GEMM (Python + RTL)
4. Implement Multi-Channel Conv2D (Python + RTL)
5. Implement Attention Score (Python + RTL)
6. Full regression test suite

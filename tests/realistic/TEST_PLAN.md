# Realistic Model Test Plan

## Completed Tests

### LeNet-5 Layer Tests (28×28 MNIST)

| Layer | Operation | Dimensions | Tiles | Cycles | Status |
|-------|-----------|------------|-------|--------|--------|
| L1 | Conv1 | (576,25)×(25,6)→(576,6) | 288 | 6,408 | ✅ PASS |
| L3 | Pool1 | (6,24,24)→(6,12,12) | - | 864 | ✅ PASS |
| L7 | FC1 | (1,256)×(256,120)→(1,120) | 480 | 11,520 | ✅ PASS |

### ResNet-18 Block Test (56×56×16)

| Stage | Operation | Dimensions | Tiles | Cycles | Status |
|-------|-----------|------------|-------|--------|--------|
| Conv1 | 3×3 conv | (3136,144)×(144,16) | 14,112 | 338,688 | ✅ PASS |
| Conv2 | 3×3 conv | (3136,144)×(144,16) | 14,112 | 338,688 | ✅ PASS |
| **Total** | | | **28,224** | **677,376** | ✅ PASS |

## Test Flow

```
Python golden.py
    │
    ├── Generates INT8/INT32 hex files
    ├── Generates FP32 npy files (debugging)
    │
    ▼
Verilog testbench
    │
    ├── $readmemh loads hex files
    ├── Behavioral tiled GEMM computation
    ├── Compares against expected output
    │
    ▼
PASS/FAIL
```

## File Structure

```
tests/realistic/
├── lenet/
│   ├── golden.py              # Full LeNet model, generates all vectors
│   ├── test_vectors/          # 70+ hex/npy files
│   ├── tb_lenet_layer1_conv.v # Conv1 test (28×28 input)
│   ├── tb_lenet_layer3_pool.v # Pool1 test (2×2 avgpool)
│   └── tb_lenet_layer7_fc.v   # FC1 test (256→120)
│
├── resnet_block/
│   ├── golden.py              # ResNet block, generates vectors
│   ├── test_vectors/          # Conv1+Conv2 hex/npy files
│   └── tb_resnet_block.v      # Full block test (56×56×16)
│
└── TEST_PLAN.md
```

## Remaining Work

### Phase C: Requantization
- [ ] Test INT32→INT8 requantization with proper scale factors
- [ ] Add bias addition after GEMM
- [ ] Chain layers with proper quantization

### Phase D: Transformer Ops
- [ ] LayerNorm
- [ ] Softmax (exp, sum, divide)
- [ ] GELU activation

### Phase E: Stress Testing
- [ ] Back-to-back operations
- [ ] Maximum SRAM utilization
- [ ] NoC congestion

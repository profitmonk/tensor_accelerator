# Tensor Accelerator Memory Map

## 1. Host Control Interface (AXI-Lite)

Base: Configured by FPGA address decoder

| Offset | Name | R/W | Description |
|--------|------|-----|-------------|
| 0x000 | CTRL | R/W | `[0]` start, `[15:8]` tpc_enable |
| 0x004 | STATUS | R | `[3:0]` busy, `[11:8]` done, `[19:16]` error, `[16]` all_done |
| 0x008 | IRQ_EN | R/W | `[0]` completion IRQ enable |
| 0x00C | IRQ_STATUS | R/W1C | `[0]` completion IRQ (write 1 to clear) |
| 0x100 | TPC0_PC | R/W | TPC0 start program counter |
| 0x110 | TPC1_PC | R/W | TPC1 start program counter |
| 0x120 | TPC2_PC | R/W | TPC2 start program counter |
| 0x130 | TPC3_PC | R/W | TPC3 start program counter |

## 2. TPC Internal Address Space (20-bit, per-TPC)

Each TPC has isolated 20-bit address space:

| Address Range | Size | Description |
|---------------|------|-------------|
| 0x00000-0x00FFF | 4K × 128b | Instruction Memory (IMEM) |
| 0x10000-0x1FFFF | 64K × 256b | Data SRAM (16 banks × 4K words) |

### SRAM Bank Addressing
```
Address[19:0] = {bank[3:0], word[11:0], byte[4:0]}
- bank: Selects 1 of 16 banks
- word: Selects 1 of 4096 words per bank
- byte: Byte offset within 256-bit (32-byte) word
```

### SRAM Layout Convention
| Bank Range | Typical Use |
|------------|-------------|
| 0-3 | Weight matrices |
| 4-7 | Activation matrices |
| 8-11 | Output/accumulator |
| 12-15 | Vector scratch |

## 3. External Memory (AXI4, 40-bit address)

| Address Range | Description |
|---------------|-------------|
| 0x0_0000_0000 - 0x0_FFFF_FFFF | DDR/HBM Bank 0 (4GB) |
| 0x1_0000_0000 - 0x1_FFFF_FFFF | DDR/HBM Bank 1 (4GB) |
| ... | Additional banks |

DMA transfers between external memory and TPC SRAM.

## 4. Instruction Encoding (128-bit)

```
[127:120] opcode
[119:112] subop
[111:72]  operand fields (vary by opcode)
[71:52]   src0_addr (SRAM address)
[51:32]   src1_addr (SRAM address)  
[31:12]   dst_addr (SRAM address)
[11:0]    immediate/config
```

### Opcode Map
| Opcode | Name | Description |
|--------|------|-------------|
| 0x00 | NOP | No operation |
| 0x01 | TENSOR | MXU operation |
| 0x02 | VECTOR | VPU operation |
| 0x03 | DMA | Memory transfer |
| 0x04 | SYNC | Wait for signal |
| 0x05 | LOOP | Begin loop |
| 0x06 | ENDLOOP | End loop |
| 0x07 | BARRIER | Global sync |
| 0xFF | HALT | Stop execution |

## 5. DMA Command Format (128-bit)

```
[127:120] opcode = 0x03
[119:112] subop: 0x01=LOAD, 0x02=STORE
[111:72]  ext_addr (40-bit external address)
[71:52]   int_addr (20-bit SRAM address)
[51:40]   num_rows
[39:28]   num_cols
[27:16]   ext_stride (bytes)
[15:4]    int_stride (bytes)
[3:0]     reserved
```

## 6. MXU Command Format (128-bit)

```
[127:120] opcode = 0x01
[119:112] subop: 0x01=GEMM
[111:92]  weight_addr (SRAM)
[91:72]   activation_addr (SRAM)
[71:52]   output_addr (SRAM)
[51:36]   M dimension
[35:20]   K dimension
[19:4]    N dimension
[3:0]     config (accumulate mode, etc)
```

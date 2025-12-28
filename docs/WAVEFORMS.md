# Waveform Capture Guide

This guide explains how to view waveforms and capture screenshots for documentation.

## Using Surfer (Recommended for macOS)

### Installation

```bash
# macOS
brew install surfer

# Or download from: https://surfer-project.org/
```

### Viewing Waveforms

```bash
# MAC PE waveforms
surfer sim/waves/mac_pe.vcd

# Systolic Array waveforms
surfer sim/waves/systolic_array.vcd
```

### Surfer Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `+` / `-` | Zoom in/out |
| `f` | Fit all to window |
| `←` `→` | Pan left/right |
| `Home` | Go to start |
| `End` | Go to end |
| `s` | Add signal |
| `g` | Go to time |

### Capturing Screenshots

1. **Open the waveform:**
   ```bash
   surfer sim/waves/mac_pe.vcd
   ```

2. **Add signals:** Click the `+` button or press `s`, then select signals:
   - `tb_mac_pe.clk`
   - `tb_mac_pe.rst_n`
   - `tb_mac_pe.enable`
   - `tb_mac_pe.weight_in`
   - `tb_mac_pe.act_in`
   - `tb_mac_pe.psum_out`

3. **Format signals:**
   - Right-click signal → Set radix → Decimal (for data)
   - Right-click signal → Set radix → Hex (for addresses)

4. **Zoom to interesting area:**
   - Press `f` to fit all
   - Use mouse wheel to zoom to test cycles

5. **Take screenshot:**
   - macOS: `Cmd + Shift + 4` then select area
   - Save to `docs/images/` folder

### Recommended Screenshots

#### 1. MAC PE Test (mac_pe_waveform.png)
- Show cycles 50-150ns
- Signals: clk, rst_n, load_weight, weight_in, enable, act_in, psum_out
- Highlight the multiplication: weight=3, act=4, result=12

#### 2. Systolic Array Loading (systolic_load.png)  
- Show cycles 0-500ns
- Signals: state, weight_load_en, weight_load_col, weight_load_data
- Show weights being loaded column by column

#### 3. Systolic Array Compute (systolic_compute.png)
- Show cycles 500-1500ns
- Signals: state, act_valid, act_data, result_valid, result_data
- Show activations streaming and results emerging

---

## Using GTKWave (Alternative)

### Installation

```bash
# macOS
brew install --cask gtkwave

# If blocked by Gatekeeper:
xattr -d com.apple.quarantine /Applications/gtkwave.app
```

### Viewing with Preset Signals

```bash
# Use our preset signal configurations
gtkwave sim/waves/mac_pe.vcd sim/waves/mac_pe.gtkw
gtkwave sim/waves/systolic_array.vcd sim/waves/systolic_array.gtkw
```

### GTKWave Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl +` / `Ctrl -` | Zoom in/out |
| `Ctrl 0` | Fit to window |
| `Ctrl F` | Find signal |
| Left-click | Place primary cursor |
| Middle-click | Place secondary cursor |

---

## Adding Waveforms to README

After capturing screenshots, add them to the README:

```markdown
## Simulation Results

### MAC PE Verification

![MAC PE Waveform](docs/images/mac_pe_waveform.png)

The waveform shows:
- Weight loading at t=50ns (weight=3)
- First computation at t=100ns (3 × 4 = 12)
- Accumulation at t=150ns (12 + 3×5 = 27)

### Systolic Array Operation

![Systolic Array Waveform](docs/images/systolic_compute.png)

The systolic array processing a 2×2 matrix multiplication.
```

---

## Generating VCD Files

If you need to regenerate the VCD files:

```bash
# Run the debug menu
./debug.sh

# Select option 1 or 2 to run tests
# VCD files are saved to sim/waves/
```

Or manually:

```bash
# Compile
iverilog -o sim/mac_pe_tb rtl/core/mac_pe.v tb/tb_mac_pe.v

# Run (generates mac_pe.vcd)
cd sim && vvp mac_pe_tb && mv mac_pe.vcd waves/
```

---

## Tips for Good Waveform Screenshots

1. **Use consistent zoom level** - same scale for related screenshots
2. **Show complete transactions** - don't cut off in the middle of an operation
3. **Use decimal radix for data** - easier to verify values
4. **Add markers/cursors** - highlight key events
5. **Crop to relevant signals** - don't show everything
6. **Use dark theme** - matches README dark mode on GitHub

---

## Example Waveform Analysis

### What to Look For in MAC PE

| Time | Signal | Expected Value | What It Means |
|------|--------|----------------|---------------|
| 50ns | load_weight | 1 | Loading weight |
| 50ns | weight_in | 3 | Weight value |
| 60ns | weight_reg | 3 | Weight stored ✓ |
| 100ns | enable | 1 | Start computing |
| 100ns | act_in | 4 | Activation input |
| 110ns | psum_out | 12 | Result: 3×4=12 ✓ |

### What to Look For in Systolic Array

| Phase | State | Duration | What Happens |
|-------|-------|----------|--------------|
| IDLE | 0 | Until start | Waiting |
| LOAD | 1 | 16 cycles | Weights → PEs |
| COMPUTE | 2 | K cycles | Data flowing |
| DRAIN | 3 | 16 cycles | Results out |
| DONE | 4 | 1 cycle | Complete |

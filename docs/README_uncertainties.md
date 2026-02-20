# uncertainties.py – Quantities with Measurement Uncertainty

## Purpose

Define and propagate measurements with uncertainty and units. Create quantities from experimental data, auto-detect symbols, extract numeric values, and propagate errors through calculations symbolically.

---

## Core Concept: The Quantity Dictionary

Every quantity is a Python dictionary with these stable keys:

- **`symbol`** (str): Variable name (e.g., `"V"`, `"mass"`)
- **`unit`** (str): Display unit 
  - If `normalize=True` (default): SI SYMBOL (e.g., `"V"`, `"A"`, `"Hz"`, `"m"`)
  - If `normalize=False`: original user-specified unit (e.g., `"mV"`, `"mA"`, `"mm"`)
- **`measure`** (tuple): Displayed measurement `(value, sigma)` 
  - If `normalize=True`: values in SI base units
  - If `normalize=False`: values in original units
- **`measure_si`** (tuple): **Internal only** — always in SI base units for calculations
- **`expr`** (SymPy expression or str): Formula definition (e.g., `"V/I"` for resistance)
- **`result`** (tuple): Computed result `(value, sigma)` in SI base units, or `None` (cached, never used as input)

**Critical Guarantee:** `value ± sigma` in `measure` and `measure_si` ALWAYS have **exactly the same units** within their respective tuples.

### Example

```python
# User creates quantity with "mV"
V = mh.quantity(5000.0, 100.0, "mV", symbol="V")
# (normalize=True by default)

# Internal storage:
{
    'symbol': 'V',
    'unit': 'V',                 # SI SYMBOL, not "volt"!
    'measure': (5.0, 0.1),       # Displayed as SI: 5.0 ± 0.1 V
    'measure_si': (5.0, 0.1),    # Calculations use SI
    'expr': None,
    'result': None
}

# With normalize=False:
V = mh.quantity(5000.0, 100.0, "mV", symbol="V", normalize=False)
{
    'symbol': 'V',
    'unit': 'mV',                # Original unit (display unit)
    'measure': (5000, 100),      # Displayed as original: 5000 ± 100 mV
    'measure_si': (5.0, 0.1),    # Calculations still use SI: 5.0V±0.1V
    'expr': None,
    'result': None
}

# When displayed:
tex = mh.latex_quantity(V_normalized)
# Output: $V = 5.0 \pm 0.1 \, \mathrm{V}$  ← SI SYMBOL
tex = mh.latex_quantity(V_original)
# Output: $V = 5000 \pm 100 \, \mathrm{mV}$  ← Original unit
```

**Key Insight:** 
- **Calculations always use SI base units** (from `measure_si`)
- **Display shows SI SYMBOLS when normalize=True** (V, A, Hz, not volt, ampere, hertz)
- **Display shows original units when normalize=False** (mV, mA, mm)
- No conversion overhead during propagation
- Dimensional analysis always works correctly

---

## Creating Quantities: `quantity()`

### Syntaxes

```python
# Basic patterns (positional args)
quantity(value, unit, symbol=None)                    # No uncertainty (sigma=0)
quantity(value, sigma, unit, symbol=None)             # With uncertainty
quantity(expr_str, unit, symbol=None)                 # Expression only (for computed quantities)
quantity(value, sigma, unit, expr_str, symbol=None)   # Value + expression
```

### Parameters

- **`value`**: Numeric value (scalar, list, array)
- **`sigma`**: Uncertainty (same shape as `value`); if not provided, treated as 0
- **`unit`**: Physical unit string (e.g., `"V"`, `"kg"`, `"m/s²"`) - supports SI prefixes like `"mV"`, `"GHz"`, `"mm^3"`
- **`expr_str`**: Optional formula (string or SymPy expression) for computed quantities
- **`symbol`**: Optional variable name (keyword-only); if `None`, can be auto-detected via `register()`

---

## Automatic Unit Conversion

The module automatically handles unit conversion using [pint](https://pint.readthedocs.io/) as backend:

### Supported Features

✅ **SI Prefixes**: `m`, `k`, `M`, `G`, `T`, etc.
- `"mV"` → millivolt
- `"GHz"` → gigahertz  
- `"µm"` or `"um"` → micrometer

✅ **Compound Units**: 
- `"m/s"` → meter per second
- `"kg*m/s^2"` → kilogram·meter/second²
- `"mm^3"` → cubic millimeter

✅ **Automatic Conversion**: Values are converted to SI base units internally
- You specify: `5000 mV`
- Stored internally as: `5.0 V` (SI symbol)
- Displayed as: `5.0 ± 0.1 V` (if normalize=True)
- Displayed as: `5000 ± 100 mV` (if normalize=False)

### Examples

```python
import marhare as mh

# Voltage in millivolts (normalize=True by default)
V = mh.quantity(5000.0, 100.0, "mV", symbol="V")
# Internally: 5.0 ± 0.1 V (SI symbol)
# Display: 5.0 ± 0.1 V (SI)

# Same voltage, keep original units
V_orig = mh.quantity(5000.0, 100.0, "mV", symbol="V", normalize=False)
# Internally: 5.0 ± 0.1 V (for calculations)
# Display: 5000 ± 100 mV (original)

# Frequency in gigahertz
f = mh.quantity(2.4, 0.05, "GHz", symbol="f")
# Internally: 2.4e9 ± 5e7 Hz (SI symbol)
# Display: 2.4e9 ± 5e7 Hz (if normalize=True)
# Or: 2.4 ± 0.05 GHz (if normalize=False)

# Volume in cubic millimeters
vol = mh.quantity(1000.0, 10.0, "mm^3", symbol="V")
# Internally: 1e-6 ± 1e-8 m³ (SI symbol)
# Display: 1e-6 ± 1e-8 m³ (if normalize=True)
# Or: 1000 ± 10 mm³ (if normalize=False)

# Mixed units in calculations work automatically!
R = mh.quantity("V/I", "ohm", symbol="R")  # Will handle mV, mA, etc.
```

### What Happens Behind the Scenes

When you write `quantity(5000, 100, "mV", symbol="V")` with default `normalize=True`:

**Step 1: Parse**
```
"mV" → recognized as millivolt
```

**Step 2: Normalize to SI** (both value and uncertainty with same factor)
```
5000 mV × (1 V / 1000 mV) = 5.0 V
100 mV × (1 V / 1000 mV) = 0.1 V
```
✓ Same conversion factor applied to both!

**Step 3: Convert to SI SYMBOL** (volt → V, hertz → Hz, etc.)
```python
{
    'unit': 'V',              # What to display (SI SYMBOL)
    'measure': (5.0, 0.1),    # What to display (SI values)
    'measure_si': (5.0, 0.1), # For calculations (SI values)
}
```

**Step 4: Use in Calculations**
- All uncertainty propagation uses `measure_si` (5.0 ± 0.1 volt)
- Prevents unit mismatch errors
- Dimensional analysis works correctly

**Step 5: Display with SI Symbol**
```python
# What appears in LaTeX:
tex = mh.latex_quantity(V)
# Output: $V = 5.0 \pm 0.1 \, \mathrm{V}$  ← SI SYMBOL

# What appears in tables/plots:
# Table column: "V (V)"  ← SI SYMBOL!
# Plot axis: "Voltage [V]"  ← SI SYMBOL!
```

**With `normalize=False`:**
```python
V = mh.quantity(5000, 100, "mV", normalize=False)

{
    'unit': 'mV',             # What to display (original)
    'measure': (5000, 100),   # What to display (original values)
    'measure_si': (5.0, 0.1), # For calculations (always SI)
}

# Display output:
# LaTeX:  $V = 5000 \pm 100 \, \mathrm{mV}$  ← Original unit
# Table:  "V (mV)"
# Plots:  axis label "[mV]"
```

**The guarantee:** Calculations are always dimensionally correct (using SI base), but you control what your users see as output.

### Graceful Degradation

If `pint` is not installed:
- Unit conversion is disabled
- Units are treated as plain strings (no validation)
- Everything else works normally
- Warning is shown once at import

Install pint with: `pip install pint`

### Controlling Unit Normalization

By default, units are normalized to SI base. You can disable this with `normalize=False`:

```python
import marhare as mh

# Default behavior: normalizes to meters
x1 = mh.quantity(10.0, 0.5, "cm", symbol="x1")
# Internal: 0.1 m, Display: 10 cm

# Keep original units: stays in centimeters
x2 = mh.quantity(10.0, 0.5, "cm", symbol="x2", normalize=False)
# Internal: 10 cm, Display: 10 cm

# All calculations in "cm" now
length_total = mh.quantity("x2 + x2", "cm", symbol="L")
magnitudes = mh.register(x2, length_total)
L_result = mh.propagate_quantity("L", magnitudes)
# Result: 20 ± ... cm (not normalized to meters)
```

**When to use `normalize=False`:**
- Working in a specific unit system (CGS, laboratory units)
- Avoiding floating-point precision issues with very large/small conversions
- Educational purposes (showing calculations in non-SI units)
- Maintaining consistency with external data sources

**When to keep default (`normalize=True`):**
- Mixing different prefixes (mV + V + kV)
- General scientific calculations
- When dimensional validation is important

---

### What Units Appear in Output (After Normalization)?

**Short Answer:** The unit you originally specified (`"mV"`, `"GHz"`, etc.) — NOT the SI base unit.

Here's a concrete example to clarify:

```python
import marhare as mh
import pandas as pd

# Create a voltage measurement with mV
V = mh.quantity(5000.0, 100.0, "mV", symbol="V")

print("\n=== What's stored internally? ===")
print(V)
# {'symbol': 'V', 'unit': 'volt', 'unit_display': 'mV', 'measure': (5.0, 0.1), ...}
#                                    ↑
#                                  YOUR unit!

print("\n=== LaTeX Output ===")
tex = mh.latex_quantity(V)
print(tex)
# $V = 5000 \pm 100 \, \mathrm{mV}$
# ↑ Shows 5000 ± 100 mV (not 5.0 ± 0.1 V)

print("\n=== Table Output ===")
table = mh.quantities_table([V], significance_digits=2)
print(table)
# Output includes column: "V (mV)"  ← Your original unit!
#                         5000 ± 100 mV

print("\n=== Plot Axis Label ===")
# When you plot with: mh.plot_result(V, "V", "V")
# Axis label will be: "V [mV]"  ← Your unit!
# NOT "V [volt]" or "V [V]"
```

**The rule:**
- `unit_display` is set from your input (`"mV"`)
- `unit` is the SI base (`"volt"`)
- **All display functions (LaTeX, tables, plots) use `unit_display`**
- **All calculations use `unit` internally**

**When `normalize=False`:**
```python
x = mh.quantity(150, 2, "cm", normalize=False, symbol="x")
print(x)
# {'unit': 'cm', 'unit_display': 'cm', 'measure': (150, 2), ...}
#               ↑
#           The SAME because no normalization happened

tex = mh.latex_quantity(x)
# $x = 150 \pm 2 \, \mathrm{cm}$  ← Still shows cm
```

**Summary:**
| Operation | Uses | Example |
|-----------|------|---------|
| LaTeX output | `unit_display` | `5000 ± 100 mV` |
| Table column header | `unit_display` | `V (mV)` |
| Plot axis label | `unit_display` | `[mV]` |
| Internal calculations | `unit` | `5.0 V` |
| Dictionary key "unit" | SI base | `"volt"` |
| Dictionary key "unit_display" | Original | `"mV"` |

---

### Common Patterns


#### 1. **Measured Scalar**

```python
import marhare as mh

# Single measurement: voltage = 5.0 V ± 0.1 V
V = mh.quantity(5.0, 0.1, "V", symbol="V")

print(V)
# {'symbol': 'V', 'unit': 'V', 'expr': None, 'measure': (5.0, 0.1), 'result': None}
```

#### 2. **Measured Array (Multiple Measurements)**

```python
import numpy as np

# Series of measurements with uncertainties
times = mh.quantity(
    np.array([1.0, 2.0, 3.0, 4.0]),
    np.array([0.05, 0.05, 0.1, 0.1]),
    "s",
    symbol="t"
)
```

#### 2b. **Importing from Excel (pandas)**

```python
import pandas as pd
import marhare as mh

df = pd.read_excel(
    r"C:\...\electron.xlsx",
    sheet_name="Hoja1"
)

# Example columns: V, sV, I, sI
V = mh.quantity(df["V"].to_numpy(), df["sV"].to_numpy(), "V", symbol="V")
I = mh.quantity(df["I"].to_numpy(), df["sI"].to_numpy(), "A", symbol="I")

# Use quantities normally from here
```

#### 3. **Measured Quantity Without Explicit Symbol**

The `register()` function can infer the symbol from variable names:

```python
measurement = mh.quantity(9.81, 0.05, "m/s²")  # No symbol yet
registry = mh.register(measurement)  # Infers symbol='measurement'
```

#### 4. **Computed Quantity (Formula)**

```python
# Ohm's law: R = V/I
# Define as a formula; will compute when propagating
R = mh.quantity("V/I", "ohm", symbol="R")

print(R)
# {'symbol': 'R', 'unit': 'ohm', 'expr': V/I, 'measure': (None, None), 'result': None}
```

#### 5. **Measured + Formula (Advanced)**

```python
# Quantity with both direct measure AND a formula (formula takes precedence in propagation)
Q = mh.quantity(5.0, 0.1, "J", "F*d", symbol="work")
```

---

## Auto-Detecting Symbols: `register()`

### Syntax

```python
magnitudes = register(*quantities)
```

### Purpose

Inspect the Python stack to find variable names and assign them as symbols automatically.

### Example

```python
import marhare as mh

# Create quantities without explicit symbols
voltage = mh.quantity(5.0, 0.1, "V")
current = mh.quantity(0.2, 0.01, "A")
resistance = mh.quantity("voltage/current", "ohm")

# Call register to infer symbols from variable names
magnitudes = mh.register(voltage, current, resistance)

# Now symbols are set:
print(voltage['symbol'])        # 'voltage'
print(current['symbol'])        # 'current'
print(resistance['symbol'])     # 'resistance'
```

### How It Works

`register()` uses Python's `inspect` module to read variable names from the calling frame:

```python
# This works:
my_var = mh.quantity(10, 1, "m")
reg = mh.register(my_var)       # ✓ Finds name "my_var"

# This doesn't work:
reg = mh.register(mh.quantity(10, 1, "m"))  # ✗ Can't find variable name
```

---

## Extracting Values: `value_quantity()`

### Syntax

```python
(value, sigma) = value_quantity(quantity_dict)
```

### Purpose

Get numeric `(value, sigma)` from a quantity, preferring computed results over measurements.

### Selection Rule

1. If `result` exists → use `result`
2. Otherwise → use `measure`
3. If neither exist → return `(None, None)`

### Example

```python
import marhare as mh

q = mh.quantity(5.0, 0.1, "V", symbol="V")

# Extract before propagation (uses measure)
v, s = mh.value_quantity(q)
print(v, s)  # 5.0 0.1

# After propagation, result is set, so value_quantity prefers it
# (See propagate_quantity example below)
```

---

## Symbolic Error Propagation: `propagate_quantity()`

### Syntax

```python
result_quantity = propagate_quantity(target, magnitudes, simplify=True)
```

### Parameters

- **`target`** (str): Symbol of quantity to compute (e.g., `"resistance"`)
- **`magnitudes`** (dict): Registry from `register()` containing all quantities
- **`simplify`** (bool): Attempt symbolic simplification (default `True`)

### Purpose

Given fundamental measurements and formulas, compute derived quantities with propagated uncertainty.

### Example: Ohm's Law

```python
import marhare as mh

# Step 1: Define measurements
V = mh.quantity(10.0, 0.5, "V", symbol="V")           # Voltage measured
I = mh.quantity(2.0, 0.1, "A", symbol="I")            # Current measured

# Step 2: Define computed quantity (formula)
R = mh.quantity("V/I", "ohm", symbol="R")             # Resistance = V/I

# Step 3: Register all
magnitudes = mh.register(V, I, R)

# Step 4: Propagate error
R_result = mh.propagate_quantity("R", magnitudes)

# Step 5: Extract computed value
v, s = mh.value_quantity(R_result)
print(f"R = {v:.2f} ± {s:.2f} ohm")
# R = 5.00 ± 0.28 ohm
```

### How Uncertainty Propagates

For a function $f(V, I) = V/I$:

$$\sigma_R = \sqrt{\left(\frac{\partial f}{\partial V} \sigma_V\right)^2 + \left(\frac{\partial f}{\partial I} \sigma_I\right)^2}$$

The function computes partial derivatives symbolically and evaluates at the measured values.

---

## Full Workflow: From Measurement to Result

### Step-by-Step Example

```python
import marhare as mh
import numpy as np

# ============ STEP 1: CREATE MEASUREMENTS ============
# Measured mass (scale precision ±10 g)
m = mh.quantity(1.250, 0.010, "kg", symbol="m")

# Measured velocity (video analysis precision ±0.05 m/s)
v = mh.quantity(2.5, 0.05, "m/s", symbol="v")

# ============ STEP 2: DEFINE FORMULAS ============
# Kinetic energy: KE = 0.5 * m * v²
KE = mh.quantity("0.5*m*v**2", "J", symbol="KE")

# ============ STEP 3: REGISTER (Auto-detect symbols) ============
magnitudes = mh.register(m, v, KE)

# ============ STEP 4: PROPAGATE ============
KE_computed = mh.propagate_quantity("KE", magnitudes)

# ============ STEP 5: EXTRACT & DISPLAY ============
ke_val, ke_unc = mh.value_quantity(KE_computed)
print(f"Kinetic Energy: {ke_val:.3f} ± {ke_unc:.3f} J")
# Output: Kinetic Energy: 3.906 ± 0.098 J

# ============ STEP 6: FORMAT FOR LATEX ============
tex = mh.latex_quantity(KE_computed)
print(tex)
# Output: $3.91 \pm 0.10 \, \mathrm{J}$
```

---

## Integration with Graphics and LaTeX

### Plotting Quantities

Quantities auto-format axes labels:

```python
import marhare as mh
import numpy as np

# Measurements with uncertainty
distance = mh.quantity([1, 2, 3, 4], [0.1, 0.1, 0.2, 0.2], "m", symbol="s")
time = mh.quantity([0.5, 1.0, 1.5, 2.0], [0.05, 0.05, 0.05, 0.1], "s", symbol="t")

# Plot with auto-labeled axes
mh.plot(distance, time, title="Position vs Time")
# X-axis: "time [s]", Y-axis: "distance [m]"
```

### Formatting for Scientific Papers

```python
import marhare as mh

# Measurement with 2 sig figs of uncertainty
Q = mh.quantity(9.806, 0.015, "m/s²", symbol="g")

# Generate LaTeX (rounds appropriately)
tex = mh.latex_quantity(Q, cifras=2)
print(tex)
# Output: $g = 9.81 \pm 0.02 \, \mathrm{m/s^2}$

# Write to file for inclusion in paper
with open("results.tex", "w") as f:
    f.write(f"\\newcommand{{\\gravity}}{{{tex}}}\n")
```

---

## Automatic Compact Units with `to_compact()`

### The Problem

Large or tiny numbers are hard to read:
- `1e-9 m` is harder to understand than `1 nm`
- `2400000000 Hz` is harder than `2.4 GHz`
- `5000 mV` is harder than `5 V`

### The Solution: Compact Units

The `to_compact()` method automatically selects the best SI prefix to keep numbers between 1-999:

```python
import marhare as mh
from marhare.unit_converter import get_compact_units

# Example 1: Nanoseconds
val, sig, unit = get_compact_units(1e-9, 1e-12, "s")
# Output: (1.0, 0.001, "nanosecond")
# Display: 1.0 ± 0.001 ns (much more readable!)

# Example 2: Millivolts to Volts
val, sig, unit = get_compact_units(5000, 100, "mV")
# Output: (5.0, 0.1, "volt")
# Display: 5.0 ± 0.1 V

# Example 3: Gigahertz
val, sig, unit = get_compact_units(2.4e9, 1e8, "Hz")
# Output: (2.4, 0.1, "gigahertz")
# Display: 2.4 ± 0.1 GHz
```

### Using with `propagate_quantity()`

After computing a derived quantity, automatically compact the result units:

```python
# Define base quantities
V = mh.quantity(5000.0, 100.0, "mV", symbol="V")
R = mh.quantity(1000.0, 10.0, "ohm", symbol="R")

# Define derived quantity
I = {"symbol": "I", "expr": "V/R", "unit": "A"}

# Propagate WITHOUT compacting
result_normal = mh.propagate_quantity(I, [V, R])
print(result_normal["measure"])  # (5.0, 0.1) in amperes

# Propagate WITH automatic compacting
result_compact = mh.propagate_quantity(I, [V, R], compact=True)
# If result is 0.005 A, compacts to:
# measure: (5.0, 0.1) in milliamperes
# unit: "milliampere" (not "A")
print(result_compact["measure"])  # (5.0, 0.1)
print(result_compact["unit"])     # "milliampere"
```

### How It Works

1. **Convert to SI base units** internally
2. **Apply pint's `to_compact()`** to find best prefix
3. **Scale sigma with same factor as value** (ensures consistency!)
4. **Return human-readable values** with best-fit prefix

### Examples of Automatic Prefix Selection

| Input | Output | Why |
|-------|--------|-----|
| `1e-9 m` | `1.0 nm` | Nano keeps it at 1 (readable) |
| `5000 mV` | `5.0 V` | Volt is cleaner than k-mV |
| `2.4e9 Hz` | `2.4 GHz` | Giga avoids scientific notation |
| `0.0001 A` | `0.1 mA` | Milli keeps it as 0.1 (readable) |
| `0.5 V` | `500 mV` | Milli helps show precision |

### Key Guarantee: Consistency

**The sigma is ALWAYS scaled by the exact same factor as the value.**

```python
# Input: 5000 mV ± 100 mV
val, sig, unit = get_compact_units(5000, 100, "mV")
# Output: 5.0 V ± 0.1 V
# Check: (5.0 / 5000) = 0.001 = (0.1 / 100) ✓ Same factor!
```

This ensures `value ± sigma` measurements are always dimensionally consistent.

---

## Advanced: Multiple Variables and Complex Expressions


### Example: Gravitational Potential Energy

```python
import marhare as mh

# Measured values
m = mh.quantity(2.5, 0.05, "kg", symbol="m")
h = mh.quantity(10.0, 0.2, "m", symbol="h")
g = mh.quantity(9.81, 0.01, "m/s²", symbol="g")  # Constant, but with precision

# Define energy formula
PE = mh.quantity("m * g * h", "J", symbol="PE")

# Propagate
magnitudes = mh.register(m, h, g, PE)
PE_result = mh.propagate_quantity("PE", magnitudes)

v, s = mh.value_quantity(PE_result)
print(f"PE = {v:.1f} ± {s:.1f} J")
```

---

## Typical Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `Symbol not in registry` | Formula uses undefined variable | Add all variables to `register()` |
| `Negative sigma` | Uncertainty < 0 | Check input data; uncertainty must be ≥ 0 |
| `Circular dependency` | Formula refers to itself | Define separate quantities for measurements vs. formulas |
| `Missing symbol` | `register()` called inside `quantity()` | Call `register()` after all quantities are created |

---

## Reference: Key Functions

| Function | Purpose |
|----------|---------|
| `quantity(value, unit, symbol=None)` | Create scalar/vector with sigma=0 |
| `quantity(value, sigma, unit, symbol=None)` | Create with uncertainty |
| `quantity(expr_str, unit, symbol=None)` | Create computed quantity with formula |
| `register(*quantities)` | Auto-detect symbols from variable names |
| `value_quantity(q)` | Extract `(value, sigma)` from quantity |
| `propagate_quantity(target, magnitudes, simplify, compact=False)` | Compute derived quantity; `compact=True` auto-selects SI prefix |
| `get_compact_units(value, sigma, unit)` | Convert any measurement to readable SI prefix |
| `propagate(expr, values, sigmas, simplify)` | Low-level error propagation |
| `uncertainty_propagation(f, vars_, values, sigmas, cov)` | Advanced: includes covariance |

---

## Next Steps

- See [README_graphics.md](README_graphics.md) to plot quantities with auto-labeled axes
- See [README_latex_tools.md](README_latex_tools.md) to format for scientific papers
- See [README_monte_carlo.md](README_monte_carlo.md) for Monte Carlo uncertainty estimation

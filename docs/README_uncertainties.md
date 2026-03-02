# uncertainties.py – Quantities with Measurement Uncertainty

**Version:** 1.0 ✅ Production-ready  
**Architecture:** Immutable, Unit-separated, Validated  
**Status:** Backward-compatible with all v0.x code  

> **📘 Architecture Documentation:**  
> - **[v1.0 Release Notes](UNCERTAINTIES_V1_RELEASE.md)** — What's new in v1.0  
> - **[v1.0 Architecture Contract](UNCERTAINTIES_V1_CONTRACT.md)** — Formal guarantees & test verification  
> - **💡 v1.0 LaTeX Updates:** [CHANGELOG_V1_LATEXTOOLS.md](CHANGELOG_V1_LATEXTOOLS.md) — Groups as table columns, group inheritance, cleaner formatting

---

## Table of Contents

1. [Purpose](#purpose)
2. [Core Concept: The Quantity Dictionary](#core-concept-the-quantity-dictionary)
3. [Creating Quantities](#creating-quantities)
4. [Automatic Unit Conversion](#automatic-unit-conversion)
5. [Common Patterns](#common-patterns)
6. [Auto-Detecting Symbols: `register()`](#auto-detecting-symbols-register)
7. [Extracting Values: `value_quantity()`](#extracting-values-value_quantity)
8. [Symbolic Error Propagation: `propagate_quantity()`](#symbolic-error-propagation-propagate_quantity)
9. [Full Workflow: From Measurement to Result](#full-workflow-from-measurement-to-result)
10. [Integration with Graphics and LaTeX](#integration-with-graphics-and-latex)
11. [Advanced Topics](#advanced-topics)
12. [Typical Errors and Solutions](#typical-errors-and-solutions)
13. [Reference: Key Functions](#reference-key-functions)
14. [Next Steps](#next-steps)

---

## Purpose

Define and propagate measurements with uncertainty and units. Create quantities from experimental data, register symbols, extract numeric values, and propagate errors through calculations symbolically.

---

## Core Concept: The Quantity Object

Every quantity created by `quantity()` returns a `Quantity` object that behaves like a Python dictionary with stable keys:

- **`symbol`** (str): Variable name (e.g., `"V"`, `"mass"`)
- **`unit`** (str): Display unit 
  - If `normalize=True` (default): SI SYMBOL (e.g., `"V"`, `"A"`, `"Hz"`, `"m"`)
  - If `normalize=False`: original user-specified unit (e.g., `"mV"`, `"mA"`, `"mm"`)
- **`measure`** (tuple): Displayed measurement `(value, sigma)` 
  - If `normalize=True`: values in SI base units
  - If `normalize=False`: values in original units
- **`measure_si`** (tuple): **Internal only** — always in SI base units for calculations
- **`expr`** (SymPy expression or str): Formula definition (e.g., `"V/I"` for resistance)
- **`result`** (tuple): Computed result `(value, sigma)` in display units, or `None`
- **`expr_latex`** (str or None): LaTeX string of the formula
- **`sigma_latex`** (str or None): LaTeX string of the uncertainty propagation formula

**Critical Guarantee:** `value ± sigma` in `measure` and `measure_si` ALWAYS have **exactly the same units** within their respective tuples.

### Example

```python
# User creates quantity with "mV"
V = mh.quantity(5000.0, 100.0, "mV", symbol="V")
# (normalize=True by default)

# Internal storage:
{
    'symbol': 'V',
    'unit': 'V',                 # SI SYMBOL
    'measure': (5.0, 0.1),       # SI: 5.0 ± 0.1 V
    'measure_si': (5.0, 0.1),    # Calculations use SI
    'expr': None,
    'result': None
}

# With normalize=False:
V = mh.quantity(5000.0, 100.0, "mV", symbol="V", normalize=False)
{
    'symbol': 'V',
    'unit': 'mV',                # Original unit
    'measure': (5000, 100),      # Original: 5000 ± 100 mV
    'measure_si': (5.0, 0.1),    # Calculations use SI
    'expr': None,
    'result': None
}
```

**Key Insight:** 
- **Calculations always use SI base units** (from `measure_si`)
- **Display shows SI SYMBOLS when normalize=True** (V, A, Hz)
- **Display shows original units when normalize=False** (mV, mA, mm)
- No conversion overhead during propagation
- Dimensional analysis always works correctly

### Implementation Details: Quantity as an Encapsulated Object

While `Quantity` behaves like a dictionary (supporting `q['key']`, `q.get()`, `'key' in q`), it is **not** a dictionary subclass. Instead:

- **Encapsulation**: Uses `__slots__` to prevent arbitrary attribute creation
- **Dict-like interface**: Implements `__getitem__()`, `get()`, `keys()`, `values()`, `items()`, `__contains__()` for seamless dict-like access
- **Immutability intent**: Designed to prevent accidental mutation after creation
- **Backward compatibility**: All code using `q['unit']`, `q.get('symbol')`, etc. works unchanged
- **Legacy export**: Call `q.as_dict()` to get a plain dictionary copy if needed

**No user code changes required** — existing scripts continue to work unchanged.

---

## Creating Quantities

### Main Function: `quantity()`

#### Syntaxes

```python
# Basic patterns (positional args)
quantity(value, unit, symbol=None, normalize=True, nan_policy="keep")
quantity(value, sigma, unit, symbol=None, normalize=True, nan_policy="keep")
quantity(expr_str, unit, symbol=None, normalize=True, nan_policy="keep")
quantity(value, sigma, unit, expr_str, symbol=None, normalize=True, nan_policy="keep")

# With experimental groups (keyword-only)
quantity(groups={...}, unit="...", symbol=None, normalize=True, nan_policy="keep")
```

#### Basic Parameters

- **`value`**: Numeric value (scalar, list, array)
- **`sigma`**: Uncertainty (same shape as `value`); if not provided, treated as 0
- **`unit`**: Physical unit string (e.g., `"V"`, `"kg"`, `"m/s²"`) - supports SI prefixes
- **`expr_str`**: Optional formula (string or SymPy expression) for computed quantities
- **`symbol`**: Optional variable name (keyword-only); auto-detected via `register()` if `None`
- **`groups`**: Keyword-only dict of experimental groups: `{"group_name": (value, sigma), ...}` or `{"group_name": {"value": ..., "sigma": ...}, ...}` (alternative to individual `value`/`sigma`)
- **`normalize`**: If `True` (default), uses SI-normalized units; if `False`, keeps original display units
- **`nan_policy`**: How to handle NaN/inf in `value`
    - `"keep"` (default): keeps all entries
    - `"drop"`: removes entries where `value` is NaN/inf
    - `"raise"`: raises `ValueError` if NaN/inf found

#### Important Notes

**`sigma` must always be numeric**, not a string:
```python
# ✓ Correct
q = mh.quantity([1, 2, 3], 0.5, "m")

# ✗ Wrong
q = mh.quantity([1, 2, 3], "0.5", "m")    # TypeError
```

**Argument order is strict:**
```python
quantity(value, unit)                      # 2 args: sigma defaults to 0
quantity(value, sigma, unit)               # 3 args: standard measurement
quantity(expr, unit)                       # 2 args, expr is string
quantity(value, sigma, unit, expr)         # 4 args: measurement + formula
```

### Grouped Experimental Data

Create a quantity representing the **same physical magnitude** with multiple experimental groups:

```python
# Clean tuple syntax: (value, sigma)
wl = mh.quantity(
    groups={
        "red": ([600, 605, 602], [2, 2, 1.5]),
        "blue": ([450, 452, 448], [1.5, 1, 2]),
        "LHC-b": ([700, 705], [3, 3])
    },
    unit="nm",
    symbol=r"\lambda"
)

# Dict syntax also works
wl = mh.quantity(
    groups={
        "red": {"value": [600, 605], "sigma": [2, 2]},
        "blue": {"value": [450, 455], "sigma": [1, 1]}
    },
    unit="nm",
    symbol=r"\lambda"
)

# Global view (all data concatenated)
print(wl.value)   # [600, 605, 602, 450, 452, 448, 700, 705]
print(wl.sigma)   # [2, 2, 1.5, 1.5, 1, 2, 3, 3]

# Group-specific view
print(wl["red"].value)   # [600, 605, 602]
```

**Important rules:**

- Groups are **subsets** of one magnitude, not separate magnitudes
- Each group can be a tuple `(value, sigma)` or dict `{"value": ..., "sigma": ...}`
- `nan_policy` applies independently to each group

### Derived Quantities and Dimensionless Unit

For derived quantities without dimension, use unit `"1"` (or `"dimensionless"`):

```python
n = mh.quantity("delta_m * alpha / wl", "1", symbol="n")
```

---

## Automatic Unit Conversion

The module automatically handles unit conversion using [pint](https://pint.readthedocs.io/) as backend.

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
- Displayed as: `5.0 ± 0.1 V` (if normalize=True)
- Displayed as: `5000 ± 100 mV` (if normalize=False)

### Examples

```python
import marhare as mh

# Voltage in millivolts (normalize=True by default)
V = mh.quantity(5000.0, 100.0, "mV", symbol="V")
# Display: 5.0 ± 0.1 V (SI)

# Same voltage, keep original units
V_orig = mh.quantity(5000.0, 100.0, "mV", symbol="V", normalize=False)
# Display: 5000 ± 100 mV (original)

# Frequency in gigahertz
f = mh.quantity(2.4, 0.05, "GHz", symbol="f")
# Display: 2.4 ± 0.05 GHz (if normalize=False)

# Mixed units in calculations work automatically
R = mh.quantity("V/I", "ohm", symbol="R")
```

### What Happens Behind the Scenes

When you write `quantity(5000, 100, "mV", symbol="V")` with `normalize=True`:

**Step 1-2: Parse and Normalize to SI** (apply same factor to value and sigma)
```
5000 mV × (1 V / 1000 mV) = 5.0 V
100 mV × (1 V / 1000 mV) = 0.1 V
```

**Step 3: Convert to SI SYMBOL**
```python
{
    'unit': 'V',              # SI SYMBOL
    'measure': (5.0, 0.1),    # SI values
    'measure_si': (5.0, 0.1), # For calculations (SI)
}
```

**Step 4: Use in Calculations**
- All uncertainty propagation uses `measure_si` (5.0 ± 0.1 V)
- Prevents unit mismatch errors

**Step 5: Display with SI Symbol**
```python
tex = mh.latex_quantity(V)
# Output: $V = 5.0 \pm 0.1 \, \mathrm{V}$  ← SI SYMBOL
```

### Controlling Unit Normalization

By default, units are normalized to SI base. Disable with `normalize=False`:

```python
# Default: normalizes to meters
x1 = mh.quantity(10.0, 0.5, "cm", symbol="x1")
# Display: 0.1 m

# Keep original units
x2 = mh.quantity(10.0, 0.5, "cm", symbol="x2", normalize=False)
# Display: 10 cm
# (Internal calculations still use SI)
```

**When to use `normalize=False`:**
- Working in a specific unit system (CGS, lab units)
- Avoiding floating-point precision issues
- Educational purposes
- Maintaining consistency with external data sources

**When to keep default (`normalize=True`):**
- Mixing different prefixes (mV + V + kV)
- General scientific calculations
- When dimensional validation is important

### Graceful Degradation

If `pint` is not installed:
- Unit conversion is disabled
- Units are treated as plain strings
- Everything else works normally
- Warning shown once at import

Install pint with: `pip install pint`

---

## Common Patterns

### Pattern 1: Measured Scalar

```python
import marhare as mh

# Single measurement: voltage = 5.0 V ± 0.1 V
V = mh.quantity(5.0, 0.1, "V", symbol="V")
```

### Pattern 2: Measured Array

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

### Pattern 3: Importing from Excel

```python
import pandas as pd
import marhare as mh

df = pd.read_excel(r"C:\...\data.xlsx", sheet_name="Sheet1")

# Example columns: V, sV, I, sI
V = mh.quantity(df["V"].to_numpy(), df["sV"].to_numpy(), "V", symbol="V")
I = mh.quantity(df["I"].to_numpy(), df["sI"].to_numpy(), "A", symbol="I")
```

### Pattern 4: Computed Quantity (Formula)

```python
# Ohm's law: R = V/I
R = mh.quantity("V/I", "ohm", symbol="R")
```

### Pattern 5: Measured Quantity Without Explicit Symbol

```python
measurement = mh.quantity(9.81, 0.05, "m/s²")
registry = mh.register(measurement)  # Infers symbol='measurement'
```

### Pattern 6: Grouped Experimental Data

```python
wl = mh.quantity(
    groups={
        "red": ([600, 605, 602], [2, 2, 1.5]),
        "blue": ([450, 452, 448], [1.5, 1, 2])
    },
    unit="nm",
    symbol=r"\lambda"
)
```

---

## Auto-Detecting Symbols: `register()`

### Syntax

```python
magnitudes = register(*quantities)
```

### Purpose

Build a symbol registry for propagation.

- If a quantity already has `symbol`, `register()` keeps it
- If `symbol` is missing, `register()` auto-detects the variable name

### Example

```python
import marhare as mh

# Explicit symbols are preserved
V = mh.quantity(5.0, 0.1, "V", symbol="V")
I = mh.quantity(0.2, 0.01, "A", symbol="I")
R = mh.quantity("V/I", "ohm", symbol="R")
magnitudes = mh.register(V, I, R)

# Fallback auto-detection
voltage = mh.quantity(5.0, 0.1, "V")
reg2 = mh.register(voltage)
print(voltage['symbol'])  # 'voltage'
```

### How It Works

`register()` uses Python's `inspect` module to read variable names:

```python
# This works:
my_var = mh.quantity(10, 1, "m")
reg = mh.register(my_var)       # ✓ Finds name

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

# Extract measurement
v, s = mh.value_quantity(q)
print(v, s)  # 5.0 0.1
```

---

## Symbolic Error Propagation: `propagate_quantity()`

### Syntax

```python
result_quantity = propagate_quantity(target, magnitudes, simplify=True, group=None, compact=False)
```

### Parameters

- **`target`** (str or dict): Quantity to compute (symbol string like `"R"` or quantity object)
- **`magnitudes`** (dict): Registry from `register()` containing all quantities
- **`simplify`** (bool): Attempt symbolic simplification (default `True`)
- **`group`** (str or None): For grouped quantities:
  - `None` (default): Use global concatenated data
  - `"red"`: Use specific group only
  - Auto-inheritance: If all dependencies share identical groups
- **`compact`** (bool): Automatically select best SI prefix (default `False`)

### Purpose

Given fundamental measurements and formulas, compute derived quantities with propagated uncertainty.

### Example: Ohm's Law

```python
import marhare as mh

# Step 1: Define measurements
V = mh.quantity(10.0, 0.5, "V", symbol="V")
I = mh.quantity(2.0, 0.1, "A", symbol="I")

# Step 2: Define formula
R = mh.quantity("V/I", "ohm", symbol="R")

# Step 3: Register all
magnitudes = mh.register(V, I, R)

# Step 4: Propagate
R_result = mh.propagate_quantity(R, magnitudes)

# Step 5: Extract value
v, s = mh.value_quantity(R_result)
print(f"R = {v:.2f} ± {s:.2f} ohm")
# R = 5.00 ± 0.28 ohm
```

### How Uncertainty Propagates

For a function $f(V, I) = V/I$:

$$\sigma_R = \sqrt{\left(\frac{\partial f}{\partial V} \sigma_V\right)^2 + \left(\frac{\partial f}{\partial I} \sigma_I\right)^2}$$

The function computes partial derivatives symbolically and evaluates at measured values.

### Accessing Symbolic Expressions (LaTeX)

After calling `propagate_quantity()`, the returned quantity dict contains LaTeX strings:

- **`expr_latex`**: LaTeX representation of the formula (e.g., `\frac{V}{I}`)
- **`sigma_latex`**: LaTeX representation of the uncertainty formula

**Example:**

```python
import marhare as mh

V = mh.quantity(10.0, 0.5, "V", symbol="V")
I = mh.quantity(2.0, 0.1, "A", symbol="I")
R = mh.quantity("V/I", "ohm", symbol="R")

magnitudes = mh.register(V, I, R)
R_result = mh.propagate_quantity(R, magnitudes)

print("Formula:", R_result["expr_latex"])
# Output: \frac{V}{I}

print("Uncertainty:", R_result["sigma_latex"])
# Output: \sqrt{\frac{\sigma_{I}^{2} V^{2}}{I^{4}} + \frac{\sigma_{V}^{2}}{I^{2}}}

# Use in LaTeX documents
print(f"$$R = {R_result['expr_latex']}$$")
print(f"$$\\sigma_R = {R_result['sigma_latex']}$$")
```

**Notes:**
- Base quantities have `expr_latex = None` and `sigma_latex = None`
- The `simplify=True` parameter simplifies expressions before converting to LaTeX
- These show the *analytical* propagation formulas, not numeric values

### Group-Aware Propagation Modes

`propagate_quantity()` supports three modes for grouped quantities:

1. **Global mode (default, `group=None`)**
   - Uses concatenated values from all groups
   - Single result combining all experimental subsets

2. **Specific group mode (`group="red"`)**
   - Computes result for only that group
   - Other groups are ignored

3. **Automatic inheritance (when dependencies have identical groups)**
   - Each group processes independently
   - Result inherits group structure
   - ✅ **NEW:** Works even if SOME inputs don't have groups! (treated as global)

**Example with Mixed Groups:**

```python
import marhare as mh
import numpy as np

# Measurement WITH groups (e.g., different colored filters)
wavelength = mh.quantity(
    groups={
        "red":   (np.array([650.1, 650.2, 650.3]), 0.5),
        "green": (np.array([550.2, 550.1, 550.3]), 0.5),
        "blue":  (np.array([470.0, 470.1, 469.9]), 0.5),
    },
    unit="nm",
    symbol="λ"
)

# Constant WITHOUT groups (same for all colors)
speed_light = mh.quantity(3e8, 0, "m/s", symbol="c")

# Frequency formula
frequency = mh.quantity("c / λ", "Hz", symbol="ν")

# Register all
magnitudes = mh.register(wavelength, speed_light, frequency)

# Propagate: frequency INHERITS groups from wavelength even though speed_light doesn't have groups!
freq_result = mh.propagate_quantity(frequency, magnitudes)

# Result has groups: red, green, blue (one frequency per group)
print(freq_result.groups)  # ['red', 'green', 'blue']

# Generate table with groups as columns
print(mh.latex_quantity(freq_result))
```

**Key Rules:**
- ✅ If ANY input has groups, result inherits that group structure
- ✅ Inputs without groups are treated as **global** (same value for all groups)
- ✅ All grouped inputs must have **identical** group names
- ✅ Group structure is preserved through unit propagation

**Original example (all inputs grouped):**

```python
import marhare as mh

# Wavelengths from two experiments
wl = mh.quantity(
    groups={
        "red": {"value": [600, 605], "sigma": [2, 2]},
        "blue": {"value": [450, 455], "sigma": [1, 1]}
    },
    unit="nm",
    symbol=r"\lambda"
)

# Speed of light
c = mh.quantity(3e8, 0, "m/s", symbol="c")

# Frequency = c / \lambda (or use "c / wl" where wl is the quantity variable)
f = mh.quantity("c / wl", "Hz", symbol="f")
magnitudes = mh.register(wl, c, f)

# Mode 1: Global (uses all frequencies)
f_global = mh.propagate_quantity(f, magnitudes)

# Mode 2: Specific group (red only)
f_red = mh.propagate_quantity(f, magnitudes, group="red")
```

---

## Full Workflow: From Measurement to Result

### Step-by-Step Example

```python
import marhare as mh

# ============ STEP 1: CREATE MEASUREMENTS ============
m = mh.quantity(1.250, 0.010, "kg", symbol="m")
v = mh.quantity(2.5, 0.05, "m/s", symbol="v")

# ============ STEP 2: DEFINE FORMULAS ============
# Kinetic energy: KE = 0.5 * m * v²
KE = mh.quantity("0.5*m*v**2", "J", symbol="KE")

# ============ STEP 3: REGISTER ============
magnitudes = mh.register(m, v, KE)

# ============ STEP 4: PROPAGATE ============
KE_computed = mh.propagate_quantity(KE, magnitudes)

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

distance = mh.quantity([1, 2, 3, 4], [0.1, 0.1, 0.2, 0.2], "m", symbol="s")
time = mh.quantity([0.5, 1.0, 1.5, 2.0], [0.05, 0.05, 0.05, 0.1], "s", symbol="t")

# Plot with auto-labeled axes
mh.plot(distance, time, title="Position vs Time")
# X-axis: "time [s]", Y-axis: "distance [m]"
```

### Plotting Experimental Groups with Auto-Colored Series

When quantities have **experimental groups**, `plot()` automatically detects them and visualizes each group as a colored series:

```python
import marhare as mh

# Measurements of refractive index (two material samples)
wavelength = mh.quantity(
    groups={
        "glass": ([450, 500, 550], [2, 2, 2]),
        "plastic": ([451, 502, 551], [3, 3, 3])
    },
    unit="nm",
    symbol=r"\lambda"
)

refractive_index = mh.quantity(
    groups={
        "glass": ([1.52, 1.51, 1.50], [0.01, 0.01, 0.01]),
        "plastic": ([1.50, 1.49, 1.48], [0.02, 0.02, 0.02])
    },
    unit="",
    symbol="n"
)

# Single plot command—automatically creates colored series (glass, plastic)
mh.plot(wavelength, refractive_index, title="Refractive Index vs Wavelength")
# Output: Two colored scatter series on the same subplot, labeled in legend

# Custom colors per group
mh.plot(wavelength, refractive_index, 
        colors={"glass": "#1f4e79", "plastic": "#6c2c2c"},
        title="Refractive Index vs Wavelength")

# Single color for all groups
mh.plot(wavelength, refractive_index, colors="#FF5500",
        title="Refractive Index vs Wavelength")
```

**Key features:**
- Groups are auto-detected from both `wavelength` and `refractive_index`
- All groups are **drawn on the same subplot**
- Legend shows group names ("glass", "plastic")
- Axes auto-label from quantity symbols and units
- Color parameter (`colors`) supports:
  - Default (automatic theme cycle) if `colors=None`
  - Single color if `colors="color_name"`
  - Per-group mapping if `colors={"group1": "color1", ...}`

---

### Formatting for Scientific Papers

```python
import marhare as mh

Q = mh.quantity(9.806, 0.015, "m/s²", symbol="g")

# Generate LaTeX (rounds appropriately)
tex = mh.latex_quantity(Q, cifras=2)
print(tex)
# Output: $g = 9.81 \pm 0.02 \, \mathrm{m/s^2}$

# Write to file for paper
with open("results.tex", "w") as f:
    f.write(f"\\newcommand{{\\gravity}}{{{tex}}}\n")
```

---

## Advanced Topics

### Automatic Compact Units with `get_compact_units()`

#### The Problem

Large or tiny numbers are hard to read:
- `1e-9 m` is harder than `1 nm`
- `2400000000 Hz` is harder than `2.4 GHz`
- `5000 mV` is harder than `5 V`

#### The Solution

The `get_compact_units()` function automatically selects the best SI prefix to keep numbers between 1-999:

```python
import marhare as mh
from marhare.unit_converter import get_compact_units

# Example 1: Nanoseconds
val, sig, unit = get_compact_units(1e-9, 1e-12, "s")
# Output: (1.0, 0.001, "nanosecond")

# Example 2: Millivolts to Volts
val, sig, unit = get_compact_units(5000, 100, "mV")
# Output: (5.0, 0.1, "volt")

# Example 3: Gigahertz
val, sig, unit = get_compact_units(2.4e9, 1e8, "Hz")
# Output: (2.4, 0.1, "gigahertz")
```

#### Using with `propagate_quantity()`

```python
# Define base quantities
V = mh.quantity(5000.0, 100.0, "mV", symbol="V")
R = mh.quantity(1000.0, 10.0, "ohm", symbol="R")

# Define derived quantity
I = mh.quantity("V/R", "A", symbol="I")

# Propagate WITH automatic compacting
result_compact = mh.propagate_quantity(I, [V, R], compact=True)
# If result is 0.005 A, compacts to:
print(result_compact["measure"])  # (5.0, 0.1)
print(result_compact["unit"])     # "milliampere"
```

#### How It Works

1. **Convert to SI base units** internally
2. **Apply pint's `to_compact()`** to find best prefix
3. **Scale sigma with same factor as value** (consistency guaranteed!)
4. **Return human-readable values** with best-fit prefix

#### Examples of Automatic Prefix Selection

| Input | Output | Why |
|-------|--------|-----|
| `1e-9 m` | `1.0 nm` | Nano keeps readable |
| `5000 mV` | `5.0 volt` | Cleaner than k-mV |
| `2.4e9 Hz` | `2.4 GHz` | Avoids scientific notation |
| `0.0001 A` | `0.1 mA` | Keeps as 0.1 |
| `0.5 V` | `500 mV` | Shows precision |

#### Key Guarantee

**The sigma is ALWAYS scaled by the exact same factor as the value.**

```python
# Input: 5000 mV ± 100 mV
val, sig, unit = get_compact_units(5000, 100, "mV")
# Output: 5.0 volt ± 0.1 volt
# Check: (5.0 / 5000) = 0.001 = (0.1 / 100) ✓
```

### Multiple Variables and Complex Expressions

#### Example: Gravitational Potential Energy

```python
import marhare as mh

# Measured values
m = mh.quantity(2.5, 0.05, "kg", symbol="m")
h = mh.quantity(10.0, 0.2, "m", symbol="h")
g = mh.quantity(9.81, 0.01, "m/s²", symbol="g")

# Define energy formula
PE = mh.quantity("m * g * h", "J", symbol="PE")

# Propagate
magnitudes = mh.register(m, h, g, PE)
PE_result = mh.propagate_quantity(PE, magnitudes)

v, s = mh.value_quantity(PE_result)
print(f"PE = {v:.1f} ± {s:.1f} J")
```

---

## Typical Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `Symbol not in registry` | Formula uses undefined variable | Add all variables to `register()` |
| `Negative sigma` | Uncertainty < 0 | Check input data; uncertainty must be ≥ 0 |
| `TypeError: ... is not numeric (string)` | Passed `sigma="1"` instead of `sigma=1` | Always use numeric sigma |
| `value contains NaN or infinite values` | `nan_policy="raise"` with NaN/inf | Use `"keep"` / `"drop"` instead |
| `Group tuple/list must have exactly 2 elements` | Wrong format in `groups` | Use `(value, sigma)` or `{"value": ..., "sigma": ...}` |
| `Circular dependency` | Formula refers to itself | Define separate quantities |
| `Missing symbol` | `register()` called wrong | Call after all quantities created |

---

## Reference: Key Functions

| Function | Purpose |
|----------|---------|
| `quantity(value, unit, **kwargs)` | Create scalar/vector with sigma=0 |
| `quantity(value, sigma, unit, **kwargs)` | Create with uncertainty |
| `quantity(expr_str, unit, **kwargs)` | Create computed quantity |
| `quantity(groups={...}, unit="...", **kwargs)` | Create grouped experimental data |
| `register(*quantities)` | Build symbol registry |
| `value_quantity(q)` | Extract `(value, sigma)` tuple |
| `propagate_quantity(target, magnitudes, simplify=True, group=None, compact=False)` | Compute derived quantity with error propagation |
| `get_compact_units(value, sigma, unit)` | Auto-select best SI prefix |
| `latex_quantity(q, **kwargs)` | Format for LaTeX |

---

## Next Steps

- See [README_graphics.md](README_graphics.md) for plotting with auto-labeled axes
- See [README_latex_tools.md](README_latex_tools.md) for scientific paper formatting
- See [README_monte_carlo.md](README_monte_carlo.md) for Monte Carlo uncertainty estimation

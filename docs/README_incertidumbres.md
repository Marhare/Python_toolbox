# uncertainties.py – Quantities with Measurement Uncertainty

## Purpose

Define and propagate measurements with uncertainty and units. Create quantities from experimental data, auto-detect symbols, extract numeric values, and propagate errors through calculations symbolically.

---

## Core Concept: The Quantity Dictionary

Every quantity is a Python dictionary with these stable keys:

- **`symbol`** (str): Variable name (e.g., `"V"`, `"mass"`)
- **`unit`** (str): Measurement unit (e.g., `"V"`, `"kg"`, `"m/s²"`)
- **`expr`** (SymPy expression or str): Formula definition (e.g., `"V/I"` for resistance)
- **`measure`** (tuple): Experimental measurement `(value, sigma)` or `None`
- **`result`** (tuple): Computed result `(value, sigma)` or `None` (cached, never used as input)

### Example

```python
{
    'symbol': 'R',
    'unit': 'ohm',
    'expr': 'V/I',
    'measure': None,          # Computed quantity
    'result': (5.0, 0.1)      # Value ± uncertainty
}
```

---

## Creating Quantities: `quantity()`

### Syntax

```python
quantity(value, sigma=None, unit="", expr=None, symbol=None)
```

### Parameters

- **`value`**: Numeric value (scalar, list, array, or SymPy expression as string)
- **`sigma`**: Uncertainty (same shape as `value`); default `None` (treated as 0)
- **`unit`**: Physical unit string; default `""`
- **`expr`**: Optional formula (string or SymPy); ignored if `measure` is provided
- **`symbol`**: Optional variable name; if `None`, tries auto-detection via `register()`

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
R = mh.quantity(expr="V/I", unit="ohm", symbol="R")

print(R)
# {'symbol': 'R', 'unit': 'ohm', 'expr': V/I, 'measure': None, 'result': None}
```

#### 5. **Measured + Formula (Rare)**

```python
# Quantity with both direct measure AND a formula (formula takes precedence in propagation)
Q = mh.quantity(5.0, 0.1, "J", expr="F*d", symbol="work")
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
resistance = mh.quantity(expr="voltage/current", unit="ohm")

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
R = mh.quantity(expr="V/I", unit="ohm", symbol="R")   # Resistance = V/I

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
mass = mh.quantity(1.250, 0.010, "kg", symbol="m")  

# Measured velocity (video analysis precision ±0.05 m/s)
velocity = mh.quantity(2.5, 0.05, "m/s", symbol="v")

# ============ STEP 2: DEFINE FORMULAS ============
# Kinetic energy: KE = 0.5 * m * v²
kinetic_energy = mh.quantity(expr="0.5*m*v**2", unit="J", symbol="KE")

# ============ STEP 3: REGISTER (Auto-detect symbols) ============
magnitudes = mh.register(mass, velocity, kinetic_energy)

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

## Advanced: Multiple Variables and Complex Expressions

### Example: Gravitational Potential Energy

```python
import marhare as mh

# Measured values
mass = mh.quantity(2.5, 0.05, "kg", symbol="m")
height = mh.quantity(10.0, 0.2, "m", symbol="h")
g = mh.quantity(9.81, 0.01, "m/s²", symbol="g")  # Constant, but with precision

# Define energy formula
PE = mh.quantity(expr="m * g * h", unit="J", symbol="PE")

# Propagate
magnitudes = mh.register(mass, height, g, PE)
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
| `quantity(value, sigma, unit, expr, symbol)` | Create a quantity |
| `register(*quantities)` | Auto-detect symbols from variable names |
| `value_quantity(q)` | Extract `(value, sigma)` from quantity |
| `propagate_quantity(target, magnitudes, simplify)` | Compute derived quantity with error |
| `propagate(expr, values, sigmas, simplify)` | Low-level error propagation |
| `uncertainty_propagation(f, vars_, values, sigmas, cov)` | Advanced: includes covariance |

---

## Next Steps

- See [README_graficos.md](README_graficos.md) to plot quantities with auto-labeled axes
- See [README_latex_tools.md](README_latex_tools.md) to format for scientific papers
- See [README_montecarlo.md](README_montecarlo.md) for Monte Carlo uncertainty estimation

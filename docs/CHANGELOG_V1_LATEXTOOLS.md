# v1.0 LaTeX Tools Updates

## Overview

This document summarizes all changes to `latex_tools.py` relative to v0.x.

---

## 1. Experimental Groups as Table Columns

### Feature

When a quantity has **experimental groups** (e.g., measurements for different colors, samples, or trials), `latex_quantity()` automatically renders a table with groups as columns.

### Example

```python
import marhare as mh
import numpy as np

wavelength = mh.quantity(
    groups={
        "red":    (np.array([650.1, 650.2, 650.3]), 0.5),
        "green":  (np.array([550.2, 550.1, 550.3]), 0.5),
        "blue":   (np.array([470.0, 470.1, 469.9]), 0.5),
    },
    unit="nm",
    symbol="λ"
)

tex = mh.latex_quantity(wavelength)
print(tex)
```

**Output:**
```
\begin{table}[htbp]
\centering
\begin{tabular}{cccc}
\hline
λ & blue & green & red \\
\hline
$470.0 \pm 0.5\,\mathrm{nm}$ & $550.2 \pm 0.5\,\mathrm{nm}$ & $650.1 \pm 0.5\,\mathrm{nm}$ \\
$470.1 \pm 0.5\,\mathrm{nm}$ & $550.1 \pm 0.5\,\mathrm{nm}$ & $650.2 \pm 0.5\,\mathrm{nm}$ \\
$469.9 \pm 0.5\,\mathrm{nm}$ & $550.3 \pm 0.5\,\mathrm{nm}$ & $650.3 \pm 0.5\,\mathrm{nm}$ \\
\hline
\end{tabular}
\end{table}
```

### Key Features
- ✅ Groups automatically sorted alphabetically as columns
- ✅ Symbol displayed as row header
- ✅ Units included in each cell
- ✅ Array values expanded row-by-row across groups

---

## 2. Group Inheritance in Propagation

### Feature

When propagating a derived quantity, groups are automatically inherited from input quantities **even if not all inputs have groups**.

### Example: Mixed Grouped/Ungrouped Inputs

```python
import marhare as mh
import numpy as np

# Quantity WITH groups
delta_m = mh.quantity(
    groups={
        "rojo":     (np.array([1.0, 1.1, 1.2]), 0.05),
        "amarillo": (np.array([1.5, 1.6, 1.7]), 0.05),
        "verde":    (np.array([1.2, 1.3, 1.4]), 0.05),
        "azul":     (np.array([1.3, 1.4, 1.5]), 0.05),
    },
    unit="degree",
    symbol="Δm"
)

# Quantity WITHOUT groups (scalar)
alpha = mh.quantity(0.8, 0.1, "degree", symbol="α")

# Derived formula
n = mh.quantity("sin((delta_m + alpha)/2) / sin(alpha/2)", unit="1", symbol="n")

# Register and propagate
registry = mh.register(delta_m, alpha, n)  
n_result = mh.propagate_quantity(n, registry)

# n_result INHERITS groups from delta_m!
print(n_result.groups)  # ['rojo', 'amarillo', 'verde', 'azul']

# Automatic table with groups as columns
print(mh.latex_quantity(n_result))
```

### Rules
- ✅ If **ANY** input has groups, result inherits that group structure
- ✅ Inputs without groups are treated as **global** (broadcasted to all groups)
- ✅ All grouped inputs must have **identical** group names
- ✅ Group structure preserved through unit propagation

---

## 3. Cleaner Table Formatting

### 3.1 No Parentheses in Magnitude Tables

**Before (v0.x):**
```latex
m & $(1.25 \pm 0.01)\,\mathrm{kg}$ \\    ← parentheses
```

**After (v1.0+):**
```latex
m & $1.25 \pm 0.01\,\mathrm{kg}$ \\      ← no parentheses
```

### 3.2 Dimensionless Quantities No Longer Show "(1)"

**Before (v0.x):**
```python
n = mh.quantity(1.5, 0.1, unit="1", symbol="n")
print(mh.latex_quantity(n))
# Output: $n = 1.5 \pm 0.1 \, (1)$  ← awkward
```

**After (v1.0+):**
```python
n = mh.quantity(1.5, 0.1, unit="1", symbol="n")
print(mh.latex_quantity(n))
# Output: $n = 1.5 \pm 0.1$  ← cleaner
```

### Summary
- ✅ Reduced visual clutter in tables
- ✅ Cleaner output for thesis/journal submissions
- ✅ Dimensionless quantities handled gracefully

---

## 4. Auto-Generate Symbols Without Explicit Assignment

### Feature

When you propagate a quantity without an explicit symbol, a default symbol `"_result"` is assigned automatically.

```python
import marhare as mh

# Define inputs
V = mh.quantity(10.0, 0.5, "V", symbol="V")
I = mh.quantity(2.0, 0.1, "A", symbol="I")

# Quantity WITHOUT explicit symbol still works!
R = mh.quantity("V/I", "ohm")  # No symbol= parameter

registry = mh.register(V, I, R)
R_result = mh.propagate_quantity(R, registry)

print(R_result.symbol)  # "_result"
print(mh.latex_quantity(R_result))  # Still generates table correctly
```

**Benefit:** More flexible workflow; you don't need explicit symbols for intermediate calculations.

---

## Migration Guide from v0.x

| What Changed | Old Code | New Code (v1.0+) |
|--------------|----------|-----------------|
| Groups as columns | Had to manually create grouped tables | `latex_quantity(q_with_groups)` auto-generates |
| Group inheritance | Didn't work; all inputs needed groups | Works automatically if ANY input has groups |
| Parentheses in tables | Always shown: `(1.5 ± 0.1)` | No longer shown: `1.5 ± 0.1` |
| Dimensionless units | Showed "(1)" | Omitted for cleanliness |
| Target without symbol | Error/exception | Auto-assign "_result" |

---

## Examples Across Use Cases

### Case 1: Single Grouped Measurement

```python
import marhare as mh
import numpy as np

# Refractive index measurements for different materials
n_index = mh.quantity(
    groups={
        "glass":  (np.array([1.50, 1.51, 1.49]), 0.02),
        "plastic": (np.array([1.48, 1.49, 1.47]), 0.02),
    },
    unit="1",  # dimensionless
    symbol="n"
)

# Automatic table with no awkward (1)
print(mh.latex_quantity(n_index))
```

### Case 2: Derived Quantity with Groups

```python
import marhare as mh
import numpy as np

# Wavelengths for different colors
wl = mh.quantity(
    groups={
        "red":   (np.array([650, 655, 652]), 1),
        "blue":  (np.array([450, 452, 451]), 1),
    },
    unit="nm",
    symbol="λ"
)

# Speed of light (no groups)
c = mh.quantity(3e8, 0, "m/s", symbol="c")

# Frequency = c / λ
freq = mh.quantity("c / wl", "Hz", symbol="f")

registry = mh.register(wl, c, freq)
freq_result = mh.propagate_quantity(freq, registry)

# freq_result inherits red, blue groups!
print(mh.latex_quantity(freq_result))
```

### Case 3: Multiple Quantities without Explicit Symbols

```python
import marhare as mh

# Calculate power loss in resistors
V = mh.quantity(10.0, 0.5, "V", symbol="V")
R = mh.quantity(100.0, 5.0, "Ω", symbol="R")

# Power without explicit symbol
P = mh.quantity("V**2 / R", "W")  # No symbol=

registry = mh.register(V, R, P)
P_result = mh.propagate_quantity(P, registry)

# "_result" used automatically, no error
print(mh.latex_quantity(P_result))
```

---

## Compatibility

✅ **Fully backward compatible** with v0.x code  
✅ Existing tables still work  
✅ New features opt-in  
✅ No breaking changes to public API


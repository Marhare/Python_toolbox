# latex_tools.py – Scientific Notation and Uncertainty Formatting

## Purpose

Format measurements with uncertainty and units for publication-quality LaTeX documents. Handles scalars, vectors, and groups with proper significant figures, metrological rounding, and scientific notation.

---

## Core Functions

### 1. **`latex_quantity(magnitude, cifras=2, siunitx=False)`** — Main Interface

**Purpose:** Format a quantity dict with symbol, unit, and uncertainty for LaTeX.

**Syntax:**
```python
latex_string = latex_quantity(quantity_dict, cifras=2, siunitx=False)
```

**Parameters:**
- `magnitude`: Quantity dict from `marhare.quantity()`
- `cifras` (int): Significant figures for rounding uncertainty (default 2)
- `siunitx` (bool): Use `siunitx` package format (default False)

**Examples:**

```python
import marhare as mh

# Example 1: Simple scalar measurement
voltage = mh.quantity(5.234, 0.048, "V", symbol="U")
tex = mh.latex_quantity(voltage, cifras=2)
print(tex)
# Output: $U = 5.23 \pm 0.05 \, \mathrm{V}$

# Example 2: With siunitx package
tex = mh.latex_quantity(voltage, cifras=2, siunitx=True)
print(tex)
# Output: \SI{5.23 \pm 0.05}{\volt}

# Example 3: Gravitational acceleration (commonly cited)
g = mh.quantity(9.807, 0.011, "m/s²", symbol="g")
tex = mh.latex_quantity(g, cifras=2)
print(tex)
# Output: $g = 9.81 \pm 0.01 \, \mathrm{m/s^2}$
```

**Tables from multiple magnitudes (auto column names):**

`latex_quantity()` accepts a list of quantity dicts. Column names are derived
automatically from each magnitude symbol and unit.

```python
import marhare as mh
import numpy as np

# Two vector magnitudes with symbols and units
V = mh.quantity(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.1, 0.1]), "V", symbol="V")
I = mh.quantity(np.array([0.2, 0.4, 0.6]), np.array([0.01, 0.01, 0.01]), "A", symbol="I")

tex = mh.latex_quantity([V, I], cifras=2)
print(tex)
# Output:
# \begin{tabular}{cc}
# \hline
# V (V) & I (A) \\
# \hline
# (1.00 \pm 0.10) & (0.20 \pm 0.01) \\
# (2.00 \pm 0.10) & (0.40 \pm 0.01) \\
# (3.00 \pm 0.10) & (0.60 \pm 0.01) \\
# \hline
# \end{tabular}
```

**Scalar magnitudes table (auto headers):**

```python
import marhare as mh

m = mh.quantity(1.250, 0.010, "kg", symbol="m")
v = mh.quantity(2.50, 0.05, "m/s", symbol="v")
E = mh.quantity(3.90, 0.10, "J", symbol="E_k")

tex = mh.latex_quantity([m, v, E], cifras=2)
print(tex)
# Output:
# \begin{tabular}{cc}
# \hline
# Magnitud & Valor \\
# \hline
# m & (1.25 \pm 0.01)\,\mathrm{kg} \\
# v & (2.50 \pm 0.05)\,\mathrm{m/s} \\
# E_k & (3.90 \pm 0.10)\,\mathrm{J} \\
# \hline
# \end{tabular}
```

---

### 2. **`valor_pm(value, sigma=None, unidad=None, cifras=2, siunitx=False, ...)`** — Scalar & Vector Formatting

**Purpose:** Format a single value with uncertainty, or generate tables for arrays.

**Syntax:**
```python
latex_string = valor_pm(value, sigma, unidad, cifras=2, siunitx=False, ...)
```

**Scalar Example:**

```python
import marhare as mh

# Simple scalar: resistivity
rho = 27.5  # µΩ·m
delta_rho = 1.2
tex = mh.valor_pm(rho, delta_rho, unidad="µΩ·m", cifras=2)
print(tex)
# Output: $(27.5 \pm 1.2) \, \mu\Omega \cdot m$
```

**Vector/Table Example:**

```python
import numpy as np

# Multiple measurements (e.g., repeated experiments)
temperatures = np.array([20.1, 20.3, 20.0, 20.2])
temp_unc = np.array([0.2, 0.2, 0.2, 0.2])

tex = mh.valor_pm(temperatures, temp_unc, unidad="°C", cifras=1)
print(tex)
# Generates:
# \begin{tabular}{lr}
# Value & Uncertainty \\
# 20.1 & 0.2 \\
# 20.3 & 0.2 \\
# 20.0 & 0.2 \\
# 20.2 & 0.2 \\
# \end{tabular}
```

---

## Practical Workflows

### Workflow 1: Single Scalar Measurement

**Scenario:** You measured gravitational acceleration in a physics lab.

```python
import marhare as mh

# 1. Create quantity with measurement
g_exp = mh.quantity(9.754, 0.084, "m/s²", symbol="g_{\\text{exp}}")

# 2. Format for LaTeX
tex = mh.latex_quantity(g_exp, cifras=2)
# Output: $g_{\text{exp}} = 9.75 \pm 0.08 \, \mathrm{m/s^2}$
```

---

### Workflow 2: Derived Quantity with Propagated Error

**Scenario:** You measured V and I, now compute R = V/I.

```python
import marhare as mh

# 1. Define measurements
U = mh.quantity(12.5, 0.3, "V", symbol="U")
I = mh.quantity(2.5, 0.1, "A", symbol="I")

# 2. Define formula
R = mh.quantity("U/I", "Ω", symbol="R")

# 3. Propagate error
magnitudes = mh.register(U, I, R)
R_computed = mh.propagate_quantity("R", magnitudes)

# 4. Format for publication
tex = mh.latex_quantity(R_computed, cifras=2)
# Output: $R = 5.0 \pm 0.2 \, \Omega$
```

---

### Workflow 3: Vector of Measurements (Lab Series)

**Scenario:** You have multiple measurements from repeated experiments.

```python
import marhare as mh
import numpy as np

# Series of time measurements (stopwatch precision)
times = mh.quantity(
    np.array([2.15, 2.18, 2.12, 2.20, 2.16]),
    np.array([0.05, 0.05, 0.05, 0.05, 0.05]),
    "s",
    symbol="t"
)

# Calculate mean and std dev
# Extract numeric values from quantity
values, _ = mh.value_quantity(times)

# Calculate statistics using statistics.py
t_mean = mh.mean(values)
t_se = mh.standard_error(values)

# Create summary quantity
t_summary = mh.quantity(t_mean, t_se, "s", symbol="\\bar{t}")
tex = mh.latex_quantity(t_summary, cifras=2)

# Output:
# \begin{tabular}{ll}
# \hline
# Measurement & Time (s) \\
# \hline
# 1 & $2.15 \pm 0.05$ \\
# 2 & $2.18 \pm 0.05$ \\
# 3 & $2.12 \pm 0.05$ \\
# 4 & $2.20 \pm 0.05$ \\
# 5 & $2.16 \pm 0.05$ \\
# \hline
# Mean & {tex} \\
# \hline
# \end{tabular}
```

---

### Workflow 4: Group of Related Quantities

**Scenario:** You measure several properties of a material and want them formatted consistently.

```python
import marhare as mh

# Define properties of a metal sample
properties = {
    'density': mh.quantity(8.96, 0.08, "g/cm³", symbol="\\rho"),
    'melting_point': mh.quantity(1085, 5, "°C", symbol="T_m"),
    'conductivity': mh.quantity(386, 12, "W/(m·K)", symbol="\\sigma"),
    'resistivity': mh.quantity(1.68e-8, 0.05e-8, "Ω·m", symbol="\\rho_e")
}

# Generate LaTeX for each
tex_lines = []
for name, qty in properties.items():
    tex = mh.latex_quantity(qty, cifras=2)
    tex_lines.append(f"\\item {name.replace('_', ' ').title()}: {tex}")

# Output:
# \begin{itemize}
# \item Density: ...
# \item Melting Point: ...
# \item Conductivity: ...
# \item Resistivity: ...
# \end{itemize}
```

---

## Significant Figures and Uncertainty

### Rule: Match Precision to Uncertainty

The uncertainty determines how many decimal places to keep. The `cifras` parameter controls how many significant figures the uncertainty is reported with:

| Uncertainty | cifras=1 | cifras=2 | Typical Use |
|-------------|----------|----------|-----|
| ±0.08 | 0.1 | 0.08 | Higher precision |
| ±0.8 | 1 | 0.8 | Medium precision |
| ±8 | 10 | 8 | Lower precision |

---

## Integration with Graphics

Plot measurements and format for LaTeX:

```python
import marhare as mh
import numpy as np

# Create measurements
x = mh.quantity(np.array([1, 2, 3, 4]), [0.1, 0.1, 0.1, 0.1], "m", symbol="x")
y = mh.quantity(np.array([2, 4.2, 5.9, 8.1]), [0.2, 0.2, 0.3, 0.3], "s", symbol="t")

# Plot with auto-labels
mh.plot(x, y, title="Measurement Data")

# Format results for paper
x_mean = mh.quantity(x['measure'][0].mean(), 0.05, "m", symbol="\\bar{x}")
y_mean = mh.quantity(y['measure'][0].mean(), 0.15, "s", symbol="\\bar{t}")

tex_x = mh.latex_quantity(x_mean)
tex_y = mh.latex_quantity(y_mean)

# Output:
# Mean distance: {tex_x}
# Mean time: {tex_y}
```

---

## Advanced: Custom siunitx Output

For documents using the `siunitx` package:

```python
import marhare as mh

# Enable siunitx formatting
g = mh.quantity(9.81, 0.02, "m/s^2", symbol="g")
tex = mh.latex_quantity(g, cifras=2, siunitx=True)
print(tex)
# Output: \SI{9.81 \pm 0.02}{\meter\per\second\squared}
```

---

## Common Patterns

| Task | Code |
|------|------|
| Format scalar | `latex_quantity(q)` |
| Scalar with 1 sig fig of σ | `latex_quantity(q, cifras=1)` |
| Format array as table | `valor_pm(arr, arr_unc)` |
| With siunitx | `latex_quantity(q, siunitx=True)` |
| Vector magnitudes table | `latex_quantity([q1, q2, ...])` |

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| Uncertainty has too many decimals | `cifras` too high | Reduce `cifras` parameter |
| Can't compile units | Missing escapes | Use raw strings: `r"unit"` |
| siunitx format looks wrong | Package not loaded | Add `\usepackage{siunitx}` |
| Super large/small numbers | No scientific notation | Use `cifras=1` for aggressive rounding |

---

## Reference

| Function | Purpose |
|----------|---------|
| `latex_quantity(q, cifras, siunitx)` | Format quantity dict |
| `valor_pm(v, σ, unit, cifras, siunitx)` | Format scalar/array |

---

## Next Steps

- See [README_uncertainties.md](README_uncertainties.md) to create and propagate quantities
- See [README_graphics.md](README_graphics.md) to visualize measurements
- See [README_statistics.md](README_statistics.md) for statistical analysis tools

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

### 3. **`redondeo_incertidumbre(valor, sigma, cifras=2)`** — Metrological Rounding

**Purpose:** Round value and uncertainty according to ISO standards (1–2 significant figures for uncertainty).

**Syntax:**
```python
rounded_value, rounded_sigma = redondeo_incertidumbre(value, sigma, cifras=2)
```

**Parameters:**
- `valor`: Numeric value
- `sigma`: Uncertainty (σ)
- `cifras` (int): Significant figures of uncertainty (1 or 2)

**Examples:**

```python
from marhare import latex_tools

# Example 1: Standard rounding (2 sig figs of σ)
v, s = latex_tools.redondeo_incertidumbre(12.3456, 0.0789, cifras=2)
print(f"{v} ± {s}")
# Output: 12.35 ± 0.08

# Example 2: Single sig fig of uncertainty (more conservative)
v, s = latex_tools.redondeo_incertidumbre(100.5, 15.3, cifras=1)
print(f"{v} ± {s}")
# Output: 101 ± 20

# Example 3: Large uncertainty
v, s = latex_tools.redondeo_incertidumbre(5.234, 0.89, cifras=2)
print(f"{v} ± {s}")
# Output: 5.2 ± 0.9
```

---

### 4. **`exportar(filename, contenido, modo="w")`** — Save to File

**Purpose:** Write LaTeX strings to `.tex` files for document inclusion.

**Syntax:**
```python
exportar(filename, contenido, modo="w")
```

**Example:**

```python
import marhare as mh

# Create several measurements
measurements_tex = """\\section{Results}

\\subsection{Electrical Properties}

$R = 4.7 \\pm 0.2 \\, \\Omega$

$P = 21.5 \\pm 0.8 \\, \\text{W}$

\\subsection{Thermal Data}

$T = (25.3 \\pm 0.5) \\, ^\\circ\\text{C}$
"""

# Export to file
mh.exportar("results.tex", measurements_tex)
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

# 3. Write to document
with open("measurements.tex", "w") as f:
    f.write(f"\\newcommand{{\\gexp}}{{{tex}}}\n")

# In LaTeX source:
# \documentclass{article}
# \input{measurements.tex}
# \begin{document}
# Experimental result: \gexp
# \end{document}
```

---

### Workflow 2: Derived Quantity with Propagated Error

**Scenario:** You measured V and I, now compute R = V/I.

```python
import marhare as mh

# 1. Define measurements
V = mh.quantity(12.5, 0.3, "V", symbol="U")
I = mh.quantity(2.5, 0.1, "A", symbol="I")

# 2. Define formula
R = mh.quantity(expr="U/I", unit="Ω", symbol="R")

# 3. Propagate error
magnitudes = mh.register(V, I, R)
R_computed = mh.propagate_quantity("R", magnitudes)

# 4. Format for publication
tex = mh.latex_quantity(R_computed, cifras=2)
# Output: $R = 5.0 \pm 0.2 \, \Omega$

# Export
mh.exportar("ohms_law_result.tex", tex)
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
t_mean = times['measure'][0].mean()
t_std = times['measure'][0].std() / np.sqrt(len(times['measure'][0]))

# Create summary quantity
t_summary = mh.quantity(t_mean, t_std, "s", symbol="\\bar{t}")
tex = mh.latex_quantity(t_summary, cifras=2)

# Use in table
with open("table_results.tex", "w") as f:
    f.write(f"""
\\begin{{table}}
\\centering
\\begin{{tabular}}{{ll}}
\\hline
Measurement & Time (s) \\\\
\\hline
1 & 2.15 ± 0.05 \\\\
2 & 2.18 ± 0.05 \\\\
3 & 2.12 ± 0.05 \\\\
4 & 2.20 ± 0.05 \\\\
5 & 2.16 ± 0.05 \\\\
\\hline
Mean & {tex} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
""")
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

# Write to document
with open("material_properties.tex", "w") as f:
    f.write("\\begin{itemize}\n")
    f.write("\n".join(tex_lines))
    f.write("\n\\end{itemize}\n")
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

**Example:**

```python
from marhare import latex_tools

# Value with moderate uncertainty
v, s = latex_tools.redondeo_incertidumbre(123.456, 1.234, cifras=2)
print(f"{v} ± {s}")
# Output: 123.5 ± 1.2

# Same value, more conservative rounding
v, s = latex_tools.redondeo_incertidumbre(123.456, 1.234, cifras=1)
print(f"{v} ± {s}")
# Output: 123 ± 1
```

---

## Integration with Graphics

Plot measurements and export to LaTeX:

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

# Write summary to file
with open("summary.tex", "w") as f:
    f.write(f"Mean distance: {tex_x}\n")
    f.write(f"Mean time: {tex_y}\n")
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

# In preamble:
# \usepackage{siunitx}
# \sisetup{uncertainty-mode=separate}
```

---

## Common Patterns

| Task | Code |
|------|------|
| Format scalar | `latex_quantity(q)` |
| Scalar with 1 sig fig of σ | `latex_quantity(q, cifras=1)` |
| Format array as table | `valor_pm(arr, arr_unc)` |
| Round before formatting | `redondeo_incertidumbre(v, s)` then format |
| With siunitx | `latex_quantity(q, siunitx=True)` |
| Save to file | `exportar("file.tex", tex)` |
| Include in document | `\documentclass{...} \input{file.tex}` |

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| Uncertainty has too many decimals | `cifras` too high | Reduce `cifras` parameter |
| Can't include in LaTeX | Missing special chars | Use raw strings: `r"unit"` |
| siunitx format looks wrong | Package not loaded | Add `\usepackage{siunitx}` |
| Super large/small numbers | No scientific notation | Use `cifras=1` for aggressive rounding |

---

## Reference

| Function | Purpose |
|----------|---------|
| `latex_quantity(q, cifras, siunitx)` | Format quantity dict |
| `valor_pm(v, σ, unit, cifras, siunitx)` | Format scalar/array |
| `redondeo_incertidumbre(v, σ, cifras)` | Metrological rounding |
| `exportar(file, content, modo)` | Save to `.tex` file |

---

## Next Steps

- See [README_incertidumbres.md](README_incertidumbres.md) to create and propagate quantities
- See [README_graficos.md](README_graficos.md) to visualize measurements
- See [README_estadistica.md](README_estadistica.md) for statistical analysis tools

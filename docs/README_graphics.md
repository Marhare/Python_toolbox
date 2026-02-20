# graficos.py – Scientific Visualization

## Purpose

High-level scientific visualization based on the universal `plot()` function. Express **what you want to visualize** and the engine decides **how to render it**. Supports measurements with uncertainty, symbolic functions, 2D/3D plots, heatmaps, surfaces, and complex multi-panel layouts.

---

## The Main Interface: `plot()`

```python
plot(*objetos, mode=None, layout=None, dims="2D", show=True, figsize=None, 
     xlabel=None, ylabel=None, zlabel=None, title=None, **kwargs)
```

**Parameters:**
- `*objetos`: Data objects, quantities, Functions, arrays, or semantic types
- `mode`: Visualization mode: `"scatter"` (default), `"line"`, `"heatmap"`, `"surface"`
- `dims`: `"2D"` (default) or `"3D"` for 3D visualization
- `show`: Display the plot (default `True`)
- `figsize`, `xlabel`, `ylabel`, `title`: Standard plot parameters
- `**kwargs`: Style customization (color, marker, linestyle, etc.)

**Returns:** `(fig, ax)` tuple

---

## Quick Start: Plotting with Quantities

Use `marhare.quantity()` to create measurements with uncertainty and units:

```python
import marhare as mh
import numpy as np

# Create arrays of measurements with uncertainty
length = mh.quantity([5.20, 5.23, 5.25], [0.05, 0.05, 0.05], "m", symbol="L")
time = mh.quantity([2.08, 2.10, 2.12], [0.1, 0.1, 0.1], "s", symbol="t")

# Plot directly – auto-labels with symbol and unit
mh.plot(length, time, title="Measurement")
# X-axis: "L [m]", Y-axis: "t [s]"
```

You can also plot single scalar measurements (automatically wrapped as single-point series):

```python
# Single measurements
length = mh.quantity(5.23, 0.05, "m", symbol="L")
time = mh.quantity(2.1, 0.1, "s", symbol="t")
mh.plot(length, time, title="Measurement")  # Creates a 1-point scatter plot
```

---

## Core Visualization Modes

### 1. **Default (Scatter) Mode**

Plot points from arrays or quantities:

```python
x = np.array([1, 2, 3, 4])
y = np.array([2, 4, 5.5, 8])

# Simple scatter
mh.plot(x, y, title="Data Points")

# With error bars (two equivalent syntaxes)
sy = np.array([0.3, 0.2, 0.4, 0.3])
mh.plot(x, y, yerr=sy, title="Data with Uncertainty")
mh.plot(x, y, sy=sy, title="Data with Uncertainty")  # alias for yerr
```

### 2. **Line Mode: Smooth Curves**

Use `mode="line"` for continuous curves, or pass a `Function`:

```python
# Fitted curve
x_fit = np.linspace(0, 5, 100)
y_fit = 2 * x_fit + 1

mh.plot(x, y, mode="scatter")  # Data
mh.plot(x_fit, y_fit, mode="line", label="Linear fit")  # Curve
```

### 3. **Function Mode: Symbolic Evaluation**

Pass a symbolic `Function` – it auto-evaluates on a 400-point dense grid:

```python
from marhare import Function

x = np.linspace(0, 2*np.pi, 50)
f = Function("sin(x)")
g = Function("cos(x)")

# Functions auto-evaluate over the x range
mh.plot(x, f, label="sin(x)")
mh.plot(x, g, label="cos(x)", mode="line")
```

### 4. **Heatmap Mode: 2D Data**

Visualize 2D matrices with color mapping:

```python
# Create a 2D matrix (e.g., image or temperature field)
Z = np.random.randn(10, 10)

# Method 1: mode parameter (automatically includes colorbar)
mh.plot(Z, mode="heatmap", title="2D Heatmap", figsize=(8, 6))

# Method 2: Semantic object
from marhare.graphics import Heatmap
hm = Heatmap(Z)
mh.plot(hm, title="2D Heatmap")
```

**Parameters:** `cmap` (colormap, default 'viridis'). Colorbar is added automatically.

### 5. **Surface Mode: 3D Mesh**

Render 3D surfaces from 2D data:

```python
# 2D array → 3D surface
Z = np.sin(np.linspace(0, 3, 30)[:, None]) * np.cos(np.linspace(0, 3, 30))

# Method 1: mode parameter + dims="3D"
mh.plot(Z, mode="surface", dims="3D", title="3D Surface")

# Method 2: Semantic object
from marhare.graphics import Surface
surf = Surface(Z)
mh.plot(surf, dims="3D", title="3D Surface")
```

---

## Advanced: Working with Quantities

### Auto-Labeled Axes

Quantities automatically format labels as `"symbol [unit]"`:

```python
import marhare as mh

# Create a measurement series
x_vals = [10, 20, 30, 40]
x_unc = [0.5, 0.5, 1.0, 1.0]
x_qty = mh.quantity(x_vals, x_unc, "cm", symbol="x")

y_vals = [5.2, 10.1, 15.3, 20.0]
y_unc = [0.2, 0.2, 0.3, 0.3]
y_qty = mh.quantity(y_vals, y_unc, "s", symbol="t")

# X-axis shows "x [cm]", Y-axis shows "t [s]"
mh.plot(x_qty, y_qty, title="Time vs Distance")
```

### Functions with Quantities

Combine symbolic expressions with measured data:

```python
import marhare as mh
import numpy as np
from marhare.uncertainties import value_quantity
from marhare.graphics import SeriesWithError, Fit, Panel, Scene

# Measured voltage and current
V = mh.quantity([1.0, 2.0, 3.0], [0.1, 0.1, 0.1], "V", symbol="V")
I = mh.quantity([0.2, 0.4, 0.6], [0.01, 0.01, 0.01], "A", symbol="I")

# Plot measured data directly – quantities auto-extract values and errors
mh.plot(I, V, title="Voltage vs Current")  # Auto-labels "I [A]" and "V [V]"

# Plot with a fitted line overlay
I_val, I_err = value_quantity(I)
V_val, V_err = value_quantity(V)

x_fit = np.linspace(0.1, 0.7, 50)
y_fit = 5 * x_fit  # Theory: R = 5Ω

panel = Panel(
    SeriesWithError(I_val, V_val, sy=V_err, label="Measured"),
    Fit(x_fit, y_fit, label="R=5Ω")
)
mh.plot(panel, xlabel="I [A]", ylabel="V [V]", title="V-I Curve with Theory")
```

---

## Semantic Objects (Alternative Interface)

For explicit control, use semantic classes:

- **`Series(x, y, label=None, marker=None)`** – Scatter points
- **`SeriesWithError(x, y, sy=None, sx=None, label=None)`** – Points with error bars
- **`Fit(x, yfit, label=None)`** – Smooth fitted curve
- **`Band(x, y_low, y_high, label=None)`** – Shaded confidence band
- **`Histogram(data, bins=30, label=None)`** – Histogram
- **`Series3D(x, y, z, label=None)`** – 3D scatter points
- **`Heatmap(Z, colorbar=True, cmap=None)`** – 2D matrix visualization
- **`Surface(Z, cmap='viridis')`** – 3D mesh surface

**Example:**

```python
from marhare.graphics import Series, SeriesWithError, Fit, Panel, Scene

x = np.array([1, 2, 3, 4])
y = np.array([2.1, 4.0, 5.8, 8.1])
sy = np.array([0.2, 0.2, 0.3, 0.3])
yfit = 2*x

# Method 1: Plan + Scene for labels and title
scene = Scene(
    Panel(
        SeriesWithError(x, y, sy=sy, label="Measured"),
        Fit(x, yfit, label="Linear fit y=2x")
    ),
    ylabel="Y value", title="Data and Fit"
)
mh.plot(scene)

# Method 2: If no specific labels needed, Panel alone
panel = Panel(
    SeriesWithError(x, y, sy=sy, label="Measured"),
    Fit(x, yfit, label="Linear fit y=2x")
)
mh.plot(panel)
```

---

## Multi-Panel Layouts with `Panel` and `Scene`

### `Panel`: Group objects in one subplot

```python
from marhare.graphics import Panel

data_panel = Panel(
    SeriesWithError(x, y, sy=sy, label="Data"),
    Fit(x, yfit, label="Fit")
)
residual_panel = Panel(
    Series(x, y - yfit, label="Residuals")
)

mh.plot(data_panel, residual_panel, layout=(1, 2), title="Analysis")
```

### `Scene`: Build complex layouts

```python
from marhare.graphics import Scene

scene = Scene(
    Panel(data, fit, ylabel="Y", title="Data"),
    Panel(Series(x, y - yfit), ylabel="Residual", title="Residuals"),
    layout=(1, 2),
    figsize=(12, 5),
    title="Complete Analysis"
)

mh.plot(scene)
```

---

## Style Customization

All style options from `PLOT_DEFAULTS` can be overridden:

```python
mh.plot(x, y, 
    color='red', 
    marker='s',           # square marker
    linestyle='--',       # dashed line
    linewidth=2,
    markersize=8,
    grid=True,
    legend=True
)
```

---

## Practical Examples

### Example 1: Experimental Data with Fit

```python
import marhare as mh
import numpy as np

# Experimental measurements
x_exp = mh.quantity([1, 2, 3, 4], [0.1, 0.1, 0.1, 0.1], "m", symbol="x")
y_exp = mh.quantity([2.1, 4.0, 5.9, 8.2], [0.3, 0.3, 0.4, 0.4], "s", symbol="t")

# Fitted model
x_fit = np.linspace(0.5, 4.5, 100)
y_fit = 2.05 * x_fit - 0.05

mh.plot(
    x_exp, y_exp,
    x_fit, y_fit, mode="line", label="Linear fit",
    title="Kinematics: Position vs Time"
)
```

### Example 2: 2D Heat Distribution

```python
# Temperature field
T = 25 + 10*np.sin(np.linspace(0, np.pi, 50)[:, None]) * \
        np.cos(np.linspace(0, np.pi, 50))

mh.plot(T, mode="heatmap", title="Temperature Distribution [°C]", figsize=(8, 6))
```

### Example 3: Function Family

```python
x = np.linspace(0, 10, 200)

# Multiple symbolic functions
f1 = mh.Function("sin(x)", vars=["x"])
f2 = mh.Function("sin(x/2)", vars=["x"])
f3 = mh.Function("sin(2*x)", vars=["x"])

mh.plot(x, f1, label="sin(x)", mode="line")
mh.plot(x, f2, label="sin(x/2)", mode="line", linestyle='--')
mh.plot(x, f3, label="sin(2x)", mode="line", linestyle=':')
```

---

## Visualization Decision Tree

```
Do you have...
├─ Arrays (x, y)? → Use default scatter or mode="line"
├─ Uncertainties (σ)? → Quantities auto-label axes
├─ Symbolic expression? → Use Function class, auto-evaluates
├─ 2D matrix (Z)? → Use mode="heatmap" or Heatmap object
├─ 3D surface? → Use mode="surface" or Surface object
├─ Multiple plots? → Use Panel or Scene
└─ Error bars + fit + residuals? → Combine semantic objects
```

---

## Common Patterns

| Task | Code |
|------|------|
| Simple scatter | `plot(x, y)` |
| Scatter + error bars | `plot(x, y, yerr=sy)` or `plot(x, y, sy=sy)` |
| Smooth curve | `plot(x, y_fit, mode="line")` |
| Symbolic function | `plot(x, Function("sin(x)"))` |
| With quantities | `plot(qty_x, qty_y)` → auto-labels |
| 2D matrix | `plot(Z, mode="heatmap")` |
| 3D surface | `plot(Z, mode="surface", dims="3D")` |
| Multiple panels | `plot(Panel(...), Panel(...), layout=(1,2))` |

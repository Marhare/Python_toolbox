# graficos.py

## Purpose
High‑level scientific visualization based on semantic objects. The user declares **what** to draw and the engine decides **how** to draw it.

## Semantic objects
- `Serie(x, y, label=None, marker=None)`
- `SerieConError(x, y, sy=None, sx=None, label=None)`
- `Histograma(data, bins=30, label=None)`
- `Ajuste(x, yfit, label=None)`
- `Banda(x, y_low, y_high, label=None)`
- `Serie3D(x, y, z, label=None)`
- `Panel(*objetos)`
- `Scene(*paneles, layout=None, dims="2D", figsize=None, xlabel=None, ylabel=None, title=None)`

## Plotting engine
- `plot(*objetos, layout=None, dims="2D", show=True, figsize=None, xlabel=None, ylabel=None, title=None, **kwargs)`
  - Supports `Scene` as a single argument.
  - `Panel` groups objects in the same subplot.

## Style
- Global configuration in `PLOT_DEFAULTS` (palette, size, grid, typography).
- Style is adjusted with `**kwargs` in `plot()`.

## Saving
- Integration with `guardar()` (if present in the module) to export to PDF/PNG.

## Notes
- `Scene` is the recommended unit for animations.
- `dims="3D"` requires `Serie3D`.

## Examples
```python
import numpy as np
from graficos import graficos

x = np.linspace(0, 1, 20)
y = 1 + 2*x
yfit = y
serie = graficos.Serie(x, y, label="Datos")
ajuste = graficos.Ajuste(x, yfit, label="Ajuste")
graficos.plot(serie, ajuste, title="Datos + Ajuste")
```

## Mini examples (per function)

### Serie(x, y, label=None, marker=None)
**Case 1 (typical):** If you do this:
```python
import numpy as np
from graficos import graficos

x = np.array([1, 2, 3, 4])
y = np.array([2, 4, 6, 8])
s = graficos.Serie(x, y, label="y=2x")
graficos.plot(s)
```
You do this: Create a series of points and plot it.

You get this:
```
Plot with 4 connected points, label "y=2x" in the legend
```

**Case 2 (edge):** If you do this:
```python
s = graficos.Serie(np.array([0]), np.array([5]), marker="o")
graficos.plot(s)
```
You do this: A series with a single point and a specific marker.

You get this:
```
Plot with a circular point at (0, 5)
```

### SerieConError(x, y, sy=None, sx=None, label=None)
**Case 1 (typical):** If you do this:
```python
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.1, 5.9])
sy = np.array([0.2, 0.2, 0.3])
s = graficos.SerieConError(x, y, sy=sy, label="Datos ±σ")
graficos.plot(s)
```
You do this: Series with error bars in y.

You get this:
```
Plot with 3 points, each with a vertical uncertainty bar
```

**Case 2 (edge):** If you do this:
```python
x = np.array([1.0, 2.0])
y = np.array([3.0, 4.0])
sy = np.array([0.0, 0.0])
s = graficos.SerieConError(x, y, sy=sy)
graficos.plot(s)
```
You do this: SerieConError with zero errors.

You get this:
```
Plot without visible error bars (like a normal Serie)
```

### Histograma(data, bins=30, label=None)
**Case 1 (typical):** If you do this:
```python
import numpy as np
from graficos import graficos

data = np.random.default_rng(0).normal(0, 1, 1000)
hist = graficos.Histograma(data, bins=40, label="Normal(0,1)")
graficos.plot(hist)
```
You do this: Histogram of 1000 normal samples.

You get this:
```
Plot with 40 bins, bell shape centered at 0
```

**Case 2 (edge):** If you do this:
```python
hist = graficos.Histograma(np.array([5.0, 5.0, 5.0]), bins=10)
graficos.plot(hist)
```
You do this: Histogram with identical values.

You get this:
```
A single bin with height 3, the rest empty
```

### Ajuste(x, yfit, label=None)
**Case 1 (typical):** If you do this:
```python
x = np.linspace(0, 2, 50)
yfit = 1 + 0.5*x
ajuste = graficos.Ajuste(x, yfit, label="y=1+0.5x")
graficos.plot(ajuste)
```
You do this: Plot a fitted curve.

You get this:
```
Continuous line from (0,1) to (2,2), labeled
```

**Case 2 (edge):** If you do this:
```python
ayuste = graficos.Ajuste(np.array([1.0]), np.array([1.0]))
graficos.plot(ajuste)
```
You do this: Fit with a single point.

You get this:
```
A point without a visible line
```

### Banda(x, y_low, y_high, label=None)
**Case 1 (typical):** If you do this:
```python
x = np.linspace(0, 1, 20)
y_low = x
y_high = x + 0.2
banda = graficos.Banda(x, y_low, y_high, label="Intervalo")
graficos.plot(banda)
```
You do this: Shaded area between two curves.

You get this:
```
Gray or colored area between y=x and y=x+0.2
```

**Case 2 (edge):** If you do this:
```python
banda = graficos.Banda(np.array([0, 1]), np.array([1, 1]), np.array([1, 1]))
graficos.plot(banda)
```
You do this: Band with zero thickness (y_low = y_high).

You get this:
```
Horizontal line at y=1
```
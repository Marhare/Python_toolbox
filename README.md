# Python Toolbox 🧰

A collection of utilities and tools for Python, designed to simplify common tasks in scientific computing, data analysis, and laboratory work.

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Description

`python_toolbox` is my personal library that centralizes functions I frequently use across different projects. The main purpose of this library is to avoid code duplication and maintain a set of optimized and tested tools. Through this development process, I continue to deepen my knowledge of Python architecture and numerical calculation.

### ✨ New Feature: Automatic Unit Conversion

The library now includes **automatic unit conversion** for all quantity-based calculations:
- **SI Prefixes**: Use `"mV"`, `"GHz"`, `"mm^3"` naturally
- **Transparent**: No API changes - it just works!
- **Intelligent**: Converts to SI base internally, displays original units
- **Validated**: Dimensional analysis prevents unit errors

```python
import marhare as mh

# Create quantities with any SI prefix
V = mh.quantity(5000, 100, "mV", symbol="V")  # millivolts
I = mh.quantity(2000, 50, "mA", symbol="I")   # milliamps

# Calculations work automatically - no manual conversion!
R = mh.quantity("V/I", "ohm", symbol="R")
magnitudes = mh.register(V, I, R)
R_result = mh.propagate_quantity("R", magnitudes)

# LaTeX output preserves your original units
print(mh.latex_quantity(V))
# Output: $V = 5000 \pm 100 \, \mathrm{mV}$
```

See [docs/UNIT_CONVERSION_IMPLEMENTATION.md](docs/UNIT_CONVERSION_IMPLEMENTATION.md) for details.

## 🚀 Main Modules (Detailed)

**Module documentation:**
- [docs/README_fitting.md](docs/README_fitting.md)
- [docs/README_animations.md](docs/README_animations.md)
- [docs/README_statistics.md](docs/README_statistics.md)
- [docs/README_fft_tools.md](docs/README_fft_tools.md)
- [docs/README_functions.md](docs/README_functions.md)
- [docs/README_graphics.md](docs/README_graphics.md)
- [docs/README_uncertainties.md](docs/README_uncertainties.md)
- [docs/README_latex_tools.md](docs/README_latex_tools.md)
- [docs/README_monte_carlo.md](docs/README_monte_carlo.md)

### fitting.py
**Purpose:** weighted least‑squares (WLS) curve fitting with covariances for uncertainty propagation.

**Assumptions:**
- `sy` are known absolute uncertainties in `y`.
- Residuals are Gaussian and independent.
- `absolute_sigma=True` (no error rescaling).

**Main API:**
- `linear_fit(x, y, sy=None)`
- `polynomial_fit(x, y, degree, sy=None)`
- `fit(model, x, y, sy=None, p0=None, variable="x")`
- `parameter_confidence_interval(fit_result, level=0.95)`
- `prediction_uncertainty(fit_result, model, x0)`

**Typical errors:** incompatible lengths, non‑positive `sy`, invalid model.

**Quick example:**
```python
from marhare.fitting import _Fitting

res = _Fitting.linear_fit(x, y, sy=sy)
print(res["parameters"], res["chi2_red"], res["p"])
```

---

### animations.py
**Purpose:** declarative time engine to animate objects from `graphics.py`.

**Main API:**
- `animate(scene, evolve, duration, fps=30, speed=1.0, loop=False, show=True)`

**`evolve` rules:**
- `Series` → `y(t)`
- `Series3D` → `(x, y, z)`
- `Band` → `(y_low, y_high)`
- `Fit` → `yfit(t)`

**Notes:** in notebooks with inline backend you may see a static frame; saving to GIF/MP4 is recommended.

**Quick example:**
```python
import numpy as np
import marhare as mh
from marhare.graphics import Series, Scene
from marhare.animations import animate

serie = Series(x, y)
scene = Scene([serie], title="Demo")
anim = animate(scene, {serie: lambda t: y*np.cos(t)}, duration=2.0)
```

---

### statistics.py
**Purpose:** descriptive statistics, confidence intervals, and hypothesis tests.

**Descriptive:** `mean`, `variance`, `standard_deviation`, `standard_error`.

**Weighted:** `weighted_mean`, `weighted_variance`.

**Confidence intervals:** `confidence_interval` (normal/poisson/binomial).

**Tests:**
- `mean_test` (z/t, Poisson exact, Binomial exact)
- `ks_test` (normal or uniform)

**Quick example:**
```python
from marhare.statistics import statistics

res = statistics.mean_test(x, mu0=0.0, distribucion="normal")
print(res["estadistico"], res["p_valor"])
```

---

### uncertainties.py
**Purpose:** quantities with uncertainty and symbolic propagation.

**API:**
- `quantity(value, sigma, unit, expr=None, symbol=None)`
- `register(*quantities)`
- `propagate_quantity(target, magnitudes, simplify=True)`
- `value_quantity(q)`

**Notes:** integrates with `latex_quantity` for LaTeX.

**Quick example:**
```python
import marhare as mh

V = mh.quantity(10.0, 0.2, "V")
I = mh.quantity(2.0, 0.1, "A")
R = mh.quantity("V/I", "ohm")

mags = mh.register(V, I, R)
res = mh.propagate_quantity("R", mags)
print(res["value"], res["uncertainty"])
```

---

### fft_tools.py
**Purpose:** 1D FFT with `scipy.fft`.

**API:**
- `fft(signal, dt)`
- `espectro_potencia(signal, dt)`

**Quick example:**
```python
from fft_tools import fft_tools
spec = fft_tools.espectro_potencia(signal, dt)
```

---

### graphics.py
**Purpose:** scientific visualization with semantic objects and consistent styling.

**Objects:** `Series`, `SeriesWithError`, `Histogram`, `Fit`, `Band`, `Series3D`, `Panel`, `Scene`.

**Engine:** `plot(*objects, layout=None, dims="2D", show=True, ...)`

**Quick example:**
```python
import marhare as mh
serie = mh.graphics.Series(x, y, label="Data")
mh.plot(serie)
```

---

### monte_carlo.py
**Purpose:** Monte Carlo integration and propagation.

**API:**
- `integral_1d(f, a, b, n=10000)`
- `propagacion(fun, generadores, n=10000)`

**Quick example:**
```python
from marhare.monte_carlo import montecarlo
res = montecarlo.integral_1d(lambda t: t**2, 0, 1, n=5000)
```

---

### numerics.py
**Purpose:** numeric‑symbolic evaluator with auto‑detection.

**Main API:**
- `evaluar`

**Quick example:**
```python
from marhare.numerics import numericos
import sympy as sp

x = sp.Symbol("x")
expr = sp.sin(x) + x**2
val = numericos.evaluar(expr, {"x": 1.5})
```

---

### functions.py
**Purpose:** lightweight symbolic `Function` object with lazy compilation, operator overloading, and calculus helpers.

**Core objects:**
- `Function(expr_str, vars=None, backend="numpy")` — symbolic expression wrapper
- `D(f, *vars)` — derivative operator
- `I(f, var=None, interval=None)` — integration operator

**Interactions:**
- **Graphics:** `plot()` accepts `Function` objects and evaluates them on dense grids
- **LaTeX:** `Function.latex()` generates SymPy LaTeX strings for documents
- **Operators:** combine functions with `+`, `-`, `*`, `/`, `**`

**Quick example:**
```python
from marhare import Function, D, I

f = Function("x**2 + sin(x)")
df = D(f, "x")
F = I(f, "x")

# Evaluate and plot
print(df(1.5))
marhare.plot(x, f, label="f(x)")
```

---

### latex_tools.py
**Purpose:** scientific LaTeX (metrological rounding, values with uncertainty, tables, and export).

**Main API:**
- `redondeo_incertidumbre`
- `valor_pm`
- `latex_quantity`
- `exportar`

**Quick example:**
```python
import marhare as mh
tex = mh.valor_pm(9.81, 0.05, unidad="m/s^2", cifras=2)
```

## 🛠️ Requirements

This toolbox primarily relies on the scientific Python stack:
- `numpy`
- `scipy`
- `matplotlib`
- `sympy`

## ⚙️ Installation

### Option 1: Install Directly from GitHub (Recommended)

No need to clone - install directly using pip:

```bash
pip install git+https://github.com/Marhare/Python_toolbox.git
```

Or with a specific branch:

```bash
pip install git+https://github.com/Marhare/Python_toolbox.git@main
```

### Option 2: Install Locally (For Development)

Clone the repository and install in editable mode:

```bash
git clone https://github.com/Marhare/Python_toolbox
cd Python_toolbox
pip install -e .
```

### Usage

After installation, import and use the module:

```python
import marhare as mh

# Create quantities with units
V = mh.quantity(5000, 100, "mV", symbol="V")

# Use any module
# xq, yq are quantity objects
fit_result = mh.fit_quantity("linear", xq, yq)
plot_data = mh.plot(*objects)
```

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

### fitting.py
Weighted least‑squares fitting with covariance, chi2, prediction uncertainty, and confidence intervals. Also supports quantity‑first workflows via `fit_quantity`.

Docs: [docs/README_fitting.md](docs/README_fitting.md)

---

### animations.py
Declarative animation engine for `graphics.py` scenes, with support for evolving series, bands, and fit objects over time.

Docs: [docs/README_animations.md](docs/README_animations.md)

---

### statistics.py
Descriptive and weighted statistics plus confidence intervals and hypothesis tests (mean and KS).

Docs: [docs/README_statistics.md](docs/README_statistics.md)

---

### uncertainties.py
Quantity objects with uncertainty, unit normalization, and symbolic propagation. Integrates with LaTeX and graphics output.

Docs: [docs/README_uncertainties.md](docs/README_uncertainties.md)

---

### fft_tools.py
FFT helpers and power‑spectrum utilities built on `scipy.fft`.

Docs: [docs/README_fft_tools.md](docs/README_fft_tools.md)

---

### graphics.py
Scientific visualization with semantic objects, consistent styling, and scene/layout helpers for 2D and 3D plots.

Docs: [docs/README_graphics.md](docs/README_graphics.md)

---

### monte_carlo.py
Monte Carlo integration and uncertainty propagation with generator‑driven sampling.

Docs: [docs/README_monte_carlo.md](docs/README_monte_carlo.md)

---

### functions.py
Lightweight symbolic `Function` wrapper with lazy compilation, operator overloading, and calculus helpers. Includes total derivatives (`D`, `dt`) and partial derivatives (`dp`).

Docs: [docs/README_functions.md](docs/README_functions.md)

---

### latex_tools.py
Scientific LaTeX formatting for values with uncertainty, tables, and exports.

Docs: [docs/README_latex_tools.md](docs/README_latex_tools.md)

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

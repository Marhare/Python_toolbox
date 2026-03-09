# Python Toolbox (marhare)

**Scientific Python toolkit for experimental physics data analysis**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📦 Installation

```bash
pip install -e .
```

Or directly from GitHub:
```bash
pip install git+https://github.com/Marhare/Python_toolbox.git
```

---

## 🧪 Core Modules

### 1. **uncertainties** ✨ v1.0 — Production-ready
> **Immutable quantities with automatic uncertainty propagation**

- Symbolic error propagation through arbitrary formulas
- Automatic unit conversion with SI normalization
- LaTeX output for reports and publications
- Grouped experimental data support

**Quick Example:**
```python
import marhare as mh

V = mh.quantity(10.0, 0.5, "V", symbol="V")
I = mh.quantity(2.0, 0.1, "A", symbol="I")
R = mh.quantity("V/I", "ohm", symbol="R")

magnitudes = mh.register(V, I, R)
R_result = mh.propagate_quantity(R, magnitudes)
# R = 5.00 ± 0.35 ohm
```

📖 **Documentation:**
- **[User Guide](docs/README_uncertainties.md)** — Complete tutorial and API reference
- **[v1.0 Release Notes](docs/UNCERTAINTIES_V1_RELEASE.md)** — What's new in v1.0
- **[Architecture Contract](docs/UNCERTAINTIES_V1_CONTRACT.md)** — Formal guarantees and test verification
- **[Quick Start](docs/README_V1.md)** — v1.0 at a glance

---

### 2. **statistics**
> **Statistical analysis with built-in uncertainty propagation**

- Mean, median, standard deviation with uncertainties
- Weighted statistics
- Outlier detection
- Bootstrap resampling

📖 **[Documentation](docs/README_statistics.md)**

---

### 3. **monte_carlo**
> **Monte Carlo simulations for error propagation**

- Non-linear uncertainty propagation
- Distribution sampling
- Confidence intervals
- Correlation analysis

📖 **[Documentation](docs/README_monte_carlo.md)**

---

### 4. **fitting**
> **Curve fitting with uncertainty estimation**

- Linear and non-linear regression
- Custom model fitting
- Goodness-of-fit statistics
- Residual analysis

📖 **[Documentation](docs/README_fitting.md)**

---

### 5. **graphics**
> **Publication-quality plots with LaTeX integration**

- Automatic error bar plotting
- LaTeX-formatted labels
- Multi-panel figures
- Export to PDF/PNG/SVG

📖 **[Documentation](docs/README_graphics.md)**

---

### 6. **latex_tools**
> **Generate LaTeX tables and formatted output**

- Automatic rounding following metrological rules
- Value ± uncertainty formatting
- Table generation from arrays
- siunitx support

📖 **[Documentation](docs/README_latex_tools.md)**

---

### 7. **fft_tools**
> **FFT analysis and signal processing**

- Fourier transforms with proper frequency scaling
- Power spectral density
- Window functions
- Filter design

📖 **[Documentation](docs/README_fft_tools.md)**

---

### 8. **animations**
> **Animated plots for presentations**

- Time-series animations
- Parameter sweeps
- Frame-by-frame control
- Export to GIF/MP4

📖 **[Documentation](docs/README_animations.md)**

---

### 9. **functions**
> **Common mathematical functions for physics**

- Special functions (error function, Bessel, etc.)
- Numeric integration and differentiation
- Root finding
- Interpolation

📖 **[Documentation](docs/README_functions.md)**

---

## 🚀 Quick Start

### Basic Workflow

```python
import marhare as mh
import numpy as np

# 1. Create quantities with uncertainties
V = mh.quantity(10.0, 0.5, "V", symbol="V")
I = mh.quantity(2.0, 0.1, "A", symbol="I")

# 2. Define derived formula
R = mh.quantity("V/I", "ohm", symbol="R")

# 3. Register all quantities
magnitudes = mh.register(V, I, R)

# 4. Propagate uncertainties
R_result = mh.propagate_quantity(R, magnitudes)

# 5. Extract results
v, s = mh.value_quantity(R_result)
print(f"R = {v:.2f} ± {s:.2f} ohm")
# R = 5.00 ± 0.35 ohm

# 6. Generate LaTeX output
latex_str = mh.valor_pm(R_result, cifras=2)
print(latex_str)
# $(5.0 \pm 0.4)\,\mathrm{ohm}$
```

---

## 📊 Testing

Run all tests:
```bash
# Comprehensive v1.0 architecture tests
python tests/test_v1_comprehensive.py

# README examples verification
python tests/test_readme_examples.py
```

Current test status:
- ✅ `test_v1_comprehensive.py`: 33/33 tests passing
- ✅ `test_readme_examples.py`: 34/34 examples passing

---

## 📚 Documentation Index

### Module Documentation
- **[uncertainties](docs/README_uncertainties.md)** — Uncertainty propagation (v1.0)
- **[statistics](docs/README_statistics.md)** — Statistical analysis
- **[monte_carlo](docs/README_monte_carlo.md)** — Monte Carlo simulations
- **[fitting](docs/README_fitting.md)** — Curve fitting
- **[graphics](docs/README_graphics.md)** — Plotting and visualization
- **[latex_tools](docs/README_latex_tools.md)** — LaTeX formatting
- **[fft_tools](docs/README_fft_tools.md)** — FFT and signal processing
- **[animations](docs/README_animations.md)** — Animated plots
- **[functions](docs/README_functions.md)** — Mathematical functions

### Architecture Documentation (uncertainties v1.0)
- **[v1.0 Quick Start](docs/README_V1.md)** — What's new at a glance
- **[v1.0 Release Notes](docs/UNCERTAINTIES_V1_RELEASE.md)** — Complete changelog (450+ lines)
- **[v1.0 Architecture Contract](docs/UNCERTAINTIES_V1_CONTRACT.md)** — Formal guarantees (600+ lines)
- **[v1.0 Implementation Summary](docs/UNCERTAINTIES_V1_SUMMARY.md)** — Technical details
- **[Unit Conversion System](docs/UNIT_CONVERSION_IMPLEMENTATION.md)** — How pint integration works

---

## 🏗️ Project Structure

```
Python_toolbox/
├── README.md                   # This file
├── LICENSE
├── pyproject.toml             # Package configuration
│
├── marhare/                   # Main package
│   ├── __init__.py
│   ├── uncertainties.py       # v1.0 - Immutable quantities
│   ├── statistics.py
│   ├── monte_carlo.py
│   ├── fitting.py
│   ├── graphics.py
│   ├── latex_tools.py
│   ├── fft_tools.py
│   ├── animations.py
│   ├── functions.py
│   └── unit_converter.py      # Unit system backend
│
├── docs/                      # All documentation here
│   ├── README_*.md            # Module user guides
│   ├── UNCERTAINTIES_V1_*.md  # v1.0 architecture docs
│   ├── UNIT_CONVERSION_*.md
│   └── img/                   # Documentation images
│
└── tests/                     # Test suite
    ├── test_v1_comprehensive.py    # 33 architecture tests
    ├── test_readme_examples.py     # 34 documentation tests
    └── test_v1_quick.py            # Quick smoke test
```

---

## 🎯 Design Philosophy

### uncertainties v1.0 Guarantees

1. **Immutability** — Quantities cannot be accidentally mutated after construction
2. **Unit Separation** — Formal 3-tier system (raw input / physics / display)
3. **Groups Blindado** — Experimental groups stored in consistent SI base units
4. **Comprehensive Validation** — Errors caught at system boundaries
5. **API Stability** — Zero breaking changes from v0.x

**Test Verification:** 67/67 tests passing (100%)
- Architecture: 33/33 ✅
- Documentation: 34/34 ✅

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/yourusername/Python_toolbox.git
cd Python_toolbox
pip install -e .[dev]
python tests/test_v1_comprehensive.py
```

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

- Built with [NumPy](https://numpy.org/), [SymPy](https://www.sympy.org/), and [Pint](https://pint.readthedocs.io/)
- Inspired by the [uncertainties](https://pythonhosted.org/uncertainties/) package
- LaTeX integration via [matplotlib](https://matplotlib.org/)

---

## 📞 Contact

**Questions or issues?** Open an issue on GitHub or contact the maintainers.

**Version:** 1.0 (March 2026)  
**Status:** Production-ready ✅

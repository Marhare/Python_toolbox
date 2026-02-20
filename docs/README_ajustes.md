# ajustes.py - Curve Fitting with Quantity Objects

## Purpose
Weighted least-squares (WLS) curve fitting designed to work naturally with `quantity` objects. The key idea: pass measured `x` and `y` as quantities (values + uncertainties), and use `fit_quantity()` to handle extraction, weighting, and metadata.

---

## Quantity-First Workflow

```python
import marhare as mh
import numpy as np

# Measured data with uncertainties
xq = mh.quantity(
    np.array([0.5, 1.0, 1.5, 2.0]),
    np.array([0.02, 0.02, 0.03, 0.03]),
    "s",
    symbol="t"
)

yq = mh.quantity(
    np.array([1.2, 2.1, 3.1, 4.0]),
    np.array([0.1, 0.1, 0.1, 0.1]),
    "m",
    symbol="x"
)

# Fit a linear model using quantities
fit = mh.fit_quantity("linear", xq, yq)

print(fit.raw["parametros"], fit.raw["chi2_red"], fit.raw["p"])
```

---

## Main Quantity API

### `fit_quantity(model, xq, yq, *, degree=None, p0=None, variable="x")`

**Purpose:** Fit `yq` vs `xq` directly from quantities. It extracts values and uncertainties, applies WLS, and returns a `FitResult` wrapper.

**Inputs:**
- `model`: `"linear"` | `"polynomial"` | callable | `sympy.Expr`
- `xq`, `yq`: quantity dicts
- `degree`: required for polynomial fits
- `p0`: initial guess for non-linear models
- `variable`: variable name for symbolic models

**Output:** `FitResult`
- `fit.raw`: dict with parameters, errors, covariance, chi2, ndof, chi2_red, p, yfit
- `fit.confidence_interval(level=0.95)`
- `fit.prediction(x0)`

---

## Examples

### 1) Linear Fit (Quantity)

```python
fit = mh.fit_quantity("linear", xq, yq)
print(fit.raw["parametros"])  # [a, b] for y = a + b x
```

### 2) Polynomial Fit (Quantity)

```python
fit = mh.fit_quantity("polynomial", xq, yq, degree=2)
print(fit.raw["parametros"])  # [a2, a1, a0]
```

### 3) Symbolic Model Fit

```python
import sympy as sp
x = sp.Symbol("x")
model = sp.exp(-x) * sp.Symbol("A") + sp.Symbol("B")

fit = mh.fit_quantity(model, xq, yq, p0=[1.0, 0.0], variable="x")
print(fit.raw["parametros"])  # [A, B]
```

### 4) Prediction Uncertainty

```python
pred = fit.prediction(1.25)
print(pred["y"], pred["sigma_modelo"])
```

### 5) Parameter Confidence Interval

```python
ci = fit.confidence_interval(level=0.95)
print(ci)
```

---

## When to Use the Low-Level API

If you already have numeric arrays (not quantities), you can still use these functions from the `_Fitting` class (access via `from marhare.fitting import _Fitting`):

- `_Fitting.linear_fit(x, y, sy=None)`
- `_Fitting.polynomial_fit(x, y, grado, sy=None)`
- `_Fitting.fit(modelo, x, y, sy=None, p0=None, variable="x")`

But for experimental data with uncertainties, prefer `fit_quantity()`.

---

## Typical Errors

- Incompatible lengths among `x`, `y`, `sy`
- Non-positive `sy` values
- Missing `degree` for polynomial fit
- Symbolic model without params list or variable name
- Insufficient data points for the model complexity

---

## Notes

- `fit_quantity()` uses `value_quantity()` internally; uncertainties are treated as absolute.
- Returned `FitResult` is a lightweight wrapper with convenience helpers.

---

## Next Steps

- See [README_incertidumbres.md](README_incertidumbres.md) to build `quantity` objects
- See [README_graficos.md](README_graficos.md) to plot data and fits
- See [README_estadistica.md](README_estadistica.md) to analyze residuals
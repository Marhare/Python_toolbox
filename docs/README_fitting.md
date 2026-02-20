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

print(fit.raw["parameters"], fit.raw["chi2_red"], fit.raw["p"])
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
- `fit.raw`: dict with `parameters`, `errors`, `covariance`, `chi2`, `ndof`, `chi2_red`, `p`, `yfit`
- `fit.confidence_interval(level=0.95)` -> dict with parameter confidence intervals
- `fit.prediction(x0)` -> dict with model prediction and uncertainty

---

## Examples

### 1) Linear Fit (Quantity)

```python
fit = mh.fit_quantity("linear", xq, yq)
print(fit.raw["parameters"])  # {"a": intercept, "b": slope} for y = a + b*x
```

### 2) Polynomial Fit (Quantity)

```python
fit = mh.fit_quantity("polynomial", xq, yq, degree=2)
print(fit.raw["parameters"])  # array [a2, a1, a0] for y = a2*x^2 + a1*x + a0
```

### 3) Symbolic Model Fit

```python
import sympy as sp
x = sp.Symbol("x")
model = sp.exp(-x) * sp.Symbol("A") + sp.Symbol("B")

fit = mh.fit_quantity(model, xq, yq, p0=[1.0, 0.0], variable="x")
print(fit.raw["parameters"])  # array [A, B]
print(fit.raw["expression"])  # Shows symbolic expression
print(fit.raw["symbolic_parameters"])  # List of symbols [A, B]
```

### 4) Prediction Uncertainty

```python
pred = fit.prediction(1.25)
print(pred["y"], pred["sigma_model"])  # Model value and its uncertainty
print(pred["x"])  # Input point
```

### 5) Parameter Confidence Interval

```python
ci = fit.confidence_interval(level=0.95)
print(ci)  # Automatic formatted output

# Or access data programmatically:
for param in ci["parameters"]:
    print(f"{param['name']}: [{param['lower_bound']:.3f}, {param['upper_bound']:.3f}]")
```

---

## When to Use the Low-Level API

If you already have numeric arrays (not quantities), you can still use these functions from the `_Fitting` class (access via `from marhare.fitting import _Fitting`):

- `_Fitting.linear_fit(x, y, sy=None)` -> returns dict with English keys
- `_Fitting.polynomial_fit(x, y, degree, sy=None)` -> returns dict with English keys
- `_Fitting.fit(model, x, y, sy=None, p0=None, variable="x`)` -> returns dict with English keys

**Important:** All returned dictionaries use English keys: `"parameters"`, `"errors"`, `"covariance"`, etc.

But for experimental data with uncertainties, prefer `fit_quantity()`.

---

## Return Dictionary Keys Reference

All fitting functions return dictionaries with **English keys**:

### Basic Fit Results (`fit.raw`)
- `"parameters"`: fitted parameter values (dict for linear, array for polynomial/generic)
- `"errors"`: standard errors of parameters (dict for linear, array for others)
- `"covariance"`: covariance matrix (2D array)
- `"yfit"`: fitted y values at the input x points
- `"chi2"`: chi-squared statistic
- `"ndof"`: degrees of freedom (n_points - n_parameters)
- `"chi2_red"`: reduced chi-squared (chi2/ndof)
- `"p"`: p-value from chi-squared test

### Additional Keys for Symbolic Fits
- `"expression"`: the original sympy expression
- `"symbolic_parameters"`: list of sympy symbols for parameters

### Confidence Interval Results (`fit.confidence_interval()`)
Returns `ConfidenceIntervalResult` object that:
- Prints automatically formatted when used with `print()`
- Can be accessed like a dict with key `"parameters"` containing a list of dicts:
  - `"name"`: parameter name
  - `"estimate"`: fitted value
  - `"error"`: standard error
  - `"lower_bound"`: lower bound of confidence interval
  - `"upper_bound"`: upper bound of confidence interval
  - `"level"`: confidence level used (e.g., 0.95)
  - `"distribution"`: statistical distribution used ("t" or "normal")

### Prediction Results (`fit.prediction(x0)`)
- `"x"`: evaluation point(s)
- `"y"`: model prediction at x
- `"sigma_model"`: statistical uncertainty of the prediction (parameter uncertainty only)

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

## Complete Example: Linear Fit with Analysis

```python
import marhare as mh
import numpy as np

# Create measured data with uncertainties
x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_data = np.array([2.1, 4.3, 5.9, 8.2, 10.1])
y_errors = np.array([0.3, 0.3, 0.4, 0.4, 0.5])

xq = mh.quantity(x_data, 0.1, "m", symbol="x")
yq = mh.quantity(y_data, y_errors, "cm", symbol="y")

# Perform linear fit
fit = mh.fit_quantity("linear", xq, yq)

# Access fit results
print("=== Fit Results ===")
print(f"Parameters: {fit.raw['parameters']}")  # {'a': ..., 'b': ...}
print(f"Errors: {fit.raw['errors']}")          # {'sa': ..., 'sb': ...}
print(f"Chi2/dof: {fit.raw['chi2_red']:.3f}")
print(f"p-value: {fit.raw['p']:.3f}")

# Get parameter confidence intervals
ci = fit.confidence_interval(level=0.95)
print(ci)  # Automatically formatted

# Predict at a new point
x_new = 3.5
pred = fit.prediction(x_new)
print(f"\n=== Prediction at x={x_new} ===")
print(f"y = {pred['y']:.3f} ± {pred['sigma_model']:.3f}")

# Plot data and fit together
# Method 1: Pass callable (auto-evaluates on dense 400-point grid)
y_fit_func = lambda x: fit.raw['parameters']['a'] + fit.raw['parameters']['b']*x
mh.plot(xq, yq, y_fit=y_fit_func, label="Data")

# Method 2: Pass evaluated array (same length as data)
# a = fit.raw['parameters']['a']
# b = fit.raw['parameters']['b']
# x_vals, _ = mh.value_quantity(xq)
# mh.plot(xq, yq, y_fit=a + b*x_vals, label="Data")

# Method 3: Use ax parameter for custom x grid
# fig, ax = mh.plot(xq, yq, label="Data", show=False)
# x_fit = np.linspace(0.5, 5.5, 100)
# mh.plot(x_fit, y_fit_func, mode="line", label="Fit", ax=ax, show=True)
```

---

## Next Steps

- See [README_uncertainties.md](README_uncertainties.md) to build `quantity` objects
- See [README_graphics.md](README_graphics.md) to plot data and fits
- See [README_statistics.md](README_statistics.md) to analyze residuals
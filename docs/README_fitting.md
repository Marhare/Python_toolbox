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
- `model`: `"linear"` | `"polynomial"` | callable
- `xq`, `yq`: quantity dicts
- `degree`: required for polynomial fits
- `p0`: initial guess for non-linear models
- `variable`: reserved compatibility argument (kept for API stability)

**Output:** `FitResult`
- `fit.raw`: dict with `parameters`, `errors`, `covariance`, `chi2`, `ndof`, `chi2_red`, `p`, `yfit`
- `fit.confidence_interval(level=0.95)` -> dict with parameter confidence intervals
- `fit.prediction(x0)` -> dict with model prediction and uncertainty
- `fit.parameter_quantity(name_or_index, unit="1")` -> one fitted parameter as `Quantity`

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

### 3) Callable Model Fit (Python `def`)

You can also define the model as a regular Python function with signature
`f(x, *params)`. For example:

```python
def f(x, a, b):
    return a*x + b

fit = mh.fit_quantity(f, xq, yq, p0=[1.0, 0.0])
print(fit.raw["parameters"])  # array [a, b]
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

### 6) Using fitted parameters as `Quantity`

This is useful when a fitted parameter is reused in uncertainty propagation.

How parameter selection works:

- By name (string): use the exact parameter key, for example `"a"` or `"b"` in linear fits.
- By index (int): use the parameter position (`0`, `1`, ...), useful for array-based models.
- If the name/index does not exist, an error is raised (`KeyError` or `IndexError`).

```python
# Linear model y = a + b*x
bq = fit.parameter_quantity("a", unit="um", symbol="a")
bq = fit.parameter_quantity("b", unit="um/cm", symbol="b")

# Same idea by index
p0 = fit.parameter_quantity(0, unit="um")
p1 = fit.parameter_quantity(1, unit="um/cm")

# Inspect available keys/shape first
print(fit.raw["parameters"])

# If d is a quantity, lambda can be propagated directly
bq = fit.parameter_quantity("b", unit="um/cm", symbol="b")
lambda_q = bq * d
print(lambda_q.value, lambda_q.sigma, lambda_q.unit)
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

- `fit_quantity()` accepts quantity inputs directly and handles values/sigmas internally; uncertainties are treated as absolute.
- By default, quantities are normalized to SI before fitting (`normalize=True` in `quantity(...)`). This means fit parameters are computed from internal normalized values.
- If you want lab units to be preserved in fit inputs (for example `um` vs `cm`), build those input quantities with `normalize=False` consistently.
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

# Plot data and fit together (current API)
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
mh.errorbar(xq, yq, ax=ax, label="Data")
mh.plot_fit(fit, ax=ax, label="Linear fit")
ax.legend()
plt.show()
```

## Matplotlib Best Practices (Ready-to-Reuse Patterns)

### A) Two subplots in the same figure (fit + residuals)

```python
import marhare as mh
import numpy as np
import matplotlib.pyplot as plt

xq = mh.quantity(np.array([1, 2, 3, 4, 5]), 0.1, "cm", symbol="D")
yq = mh.quantity(np.array([2.0, 4.1, 6.0, 8.2, 10.1]), np.array([0.3, 0.3, 0.4, 0.4, 0.5]), "um", symbol="i")

fit = mh.fit_quantity("linear", xq, yq)

fig, (ax_fit, ax_res) = plt.subplots(
    1,
    2,
    figsize=(11, 4),
    constrained_layout=True,
)

# Left: data + fitted model
mh.errorbar(xq, yq, ax=ax_fit, fmt="o", capsize=3, label="Data")
mh.plot_fit(fit, ax=ax_fit, color="tab:red", linewidth=2, label="Linear fit")
ax_fit.set_title("Fit")
ax_fit.set_xlabel("D (cm)")
ax_fit.set_ylabel("i (um)")
ax_fit.grid(alpha=0.25)
ax_fit.legend()

# Right: residuals with y=0 reference
res = fit.residuals
ax_res.axhline(0.0, color="black", linewidth=1)
ax_res.errorbar(xq.value, res, yerr=yq.sigma, fmt="o", capsize=3)
ax_res.set_title("Residuals")
ax_res.set_xlabel("D (cm)")
ax_res.set_ylabel("i - i_fit (um)")
ax_res.grid(alpha=0.25)

plt.show()
```

### B) Multiple datasets in one axis (same figure, same plot)

```python
import marhare as mh
import numpy as np
import matplotlib.pyplot as plt

groups = {
    "cam_1": {
        "x": np.array([1, 2, 3, 4]),
        "y": np.array([1.9, 3.8, 6.2, 8.0]),
        "sy": np.array([0.2, 0.2, 0.3, 0.3]),
    },
    "cam_2": {
        "x": np.array([1, 2, 3, 4]),
        "y": np.array([2.2, 4.4, 6.1, 8.4]),
        "sy": np.array([0.2, 0.2, 0.3, 0.3]),
    },
}

fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

for name, data in groups.items():
    xq = mh.quantity(data["x"], 0.1, "cm", symbol="D")
    yq = mh.quantity(data["y"], data["sy"], "um", symbol="i")
    fit = mh.fit_quantity("linear", xq, yq)

    mh.errorbar(xq, yq, ax=ax, fmt="o", capsize=3, label=f"{name} data")
    mh.plot_fit(fit, ax=ax, linewidth=2, label=f"{name} fit")

ax.set_title("Interference spacing vs distance")
ax.set_xlabel("D (cm)")
ax.set_ylabel("i (um)")
ax.grid(alpha=0.25)
ax.legend(ncol=2)

plt.show()
```

---

## Next Steps

- See [README_uncertainties.md](README_uncertainties.md) to build `quantity` objects
- See [README_statistics.md](README_statistics.md) to analyze residuals

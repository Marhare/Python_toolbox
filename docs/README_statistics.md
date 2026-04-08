# statistics.py - Statistics with Quantity Support

Navigation: [Documentation Index](INDEX.md) | [Main README](../README.md)

## Purpose
Descriptive statistics, confidence intervals, and hypothesis tests, designed to work naturally with `quantity` objects. The workflow is:

1. Build measurements with `quantity()`
2. Apply statistical functions directly on `q.value` / `q.sigma`
3. Or use `q.weighted()` when you want mean + weighted error in one step
4. Wrap results back into a `quantity` for reporting or LaTeX

---

## Quick Start (Quantity Workflow)

```python
import marhare as mh
import numpy as np

# 1) Create a quantity with repeated measurements
times = mh.quantity(
    np.array([2.15, 2.18, 2.12, 2.20, 2.16]),
    np.array([0.05, 0.05, 0.05, 0.05, 0.05]),
    "s",
    symbol="t"
)

# 2) Compute statistics directly from quantity arrays
t_mean = mh.mean(times.value)
t_se = mh.standard_error(times.value)
ci = mh.confidence_interval(times.value, nivel=0.95, distribucion="normal")

# 3) Wrap back into a quantity
t_summary = mh.quantity(t_mean, t_se, "s", symbol="\\bar{t}")

# Optional shortcut for weighted summary
t_weighted = times.weighted(symbol="t_w")

print(t_summary)
print(t_weighted)
print(ci)
```

---

## Descriptive Statistics

All functions accept `array_like` numeric data. With quantities, pass `q.value`.

- `mean(x)`
- `variance(x, ddof=1)`
- `standard_deviation(x, ddof=1)`
- `standard_error(x)`

**Example (from quantity):**

```python
mu = mh.mean(times.value)
s = mh.standard_deviation(times.value)
se = mh.standard_error(times.value)
```

---

## Weighted Statistics

Use measurement uncertainties as inverse-variance weights.

- `weighted_mean(x, sigma=None, w=None)`
- `weighted_standard_error(x, sigma=None, w=None)`
- `weighted_variance(x, sigma=None, w=None, ddof=1, tipo="frecuentista")`

**Example (weights from quantity sigmas):**

```python
mu_w = mh.weighted_mean(times.value, sigma=times.sigma)
se_w = mh.weighted_standard_error(times.value, sigma=times.sigma)
var_w = mh.weighted_variance(times.value, sigma=times.sigma)

# Shortcut quantity output
t_weighted = times.weighted(symbol="t_w")
```

**Note:** When using `weighted_mean` with `sigma` parameter, the correct uncertainty is given by `weighted_standard_error`, not `standard_error`. The weighted standard error uses the formula $\sigma_w = \sqrt{1/\sum w_i}$ where $w_i = 1/\sigma_i^2$.

---

## Confidence Intervals

- `confidence_interval(x, nivel=0.95, distribucion="normal", sigma=None)`

Supported distributions:
- `"normal"`
- `"poisson"`
- `"binomial"`

**Example:**

```python
ci = mh.confidence_interval(times.value, nivel=0.95, distribucion="normal")
print(ci)
```

---

## Hypothesis Tests

- `mean_test(x, mu0, alternativa="dos_colas", distribucion="normal", sigma=None)`
- `ks_test(x, distribucion="normal")`

**Example (mean test with quantity):**

```python
test = mh.mean_test(times.value, mu0=2.0, alternativa="dos_colas", distribucion="normal")
print(test)
```

---

## Full Example: Quantity -> Stats -> LaTeX

```python
import marhare as mh
import numpy as np

measurements = mh.quantity(
    np.array([9.78, 9.81, 9.79, 9.82, 9.80]),
    np.array([0.04, 0.04, 0.05, 0.04, 0.04]),
    "m/s^2",
    symbol="g"
)

g_summary = measurements.weighted(symbol="\\bar{g}")
tex = mh.latex_quantity(g_summary, cifras=2)

print(tex)
```

---

## Output Format

Statistical tests and confidence intervals return dictionaries with:

- `estadistico`
- `p_valor`
- `rechaza`
- `n`
- `grados_libertad`

---

## Typical Errors

- Empty samples or invalid sizes
- Non-finite values (NaN or inf)
- Parameters out of range (e.g., `sigma <= 0`, `nivel` outside (0, 1))
- Unsupported distribution names

---

## Reference

| Function | Purpose |
|----------|---------|
| `mean(x)` | Arithmetic mean |
| `variance(x, ddof)` | Sample variance |
| `standard_deviation(x, ddof)` | Sample standard deviation |
| `standard_error(x)` | Standard error of the mean |
| `weighted_mean(x, w, sigma)` | Weighted mean (use `sigma` for inverse-variance weights) || `weighted_standard_error(x, w, sigma)` | Standard error of weighted mean: $\sqrt{1/\sum w_i}$ || `weighted_variance(x, w, sigma, ddof, tipo)` | Weighted variance |
| `confidence_interval(x, nivel, distribucion, sigma)` | Confidence interval |
| `mean_test(x, mu0, alternativa, distribucion, sigma)` | Mean hypothesis test |
| `ks_test(x, distribucion)` | Kolmogorov-Smirnov test |

---

## Next Steps

- See [README_uncertainties.md](README_uncertainties.md) to build `quantity` objects
- See [README_latex_tools.md](README_latex_tools.md) to export statistical results to LaTeX
- Use Matplotlib (`matplotlib.pyplot`) to plot measurements and summaries

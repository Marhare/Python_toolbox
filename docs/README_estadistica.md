# statistics.py - Statistics with Quantity Support

## Purpose
Descriptive statistics, confidence intervals, and hypothesis tests, designed to work naturally with `quantity` objects. The workflow is:

1. Build measurements with `quantity()`
2. Extract numeric arrays with `value_quantity()`
3. Apply statistical functions on the numeric arrays
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

# 2) Extract numeric values
values, sigmas = mh.value_quantity(times)

# 3) Compute statistics
t_mean = mh.mean(values)
t_se = mh.standard_error(values)
ci = mh.confidence_interval(values, nivel=0.95, distribucion="normal")

# 4) Wrap back into a quantity
t_summary = mh.quantity(t_mean, t_se, "s", symbol="\\bar{t}")

print(t_summary)
print(ci)
```

---

## Descriptive Statistics

All functions accept `array_like` numeric data. When using a `quantity`, extract its values first.

- `mean(x)`
- `variance(x, ddof=1)`
- `standard_deviation(x, ddof=1)`
- `standard_error(x)`

**Example (from quantity):**

```python
values, _ = mh.value_quantity(times)
mu = mh.mean(values)
s = mh.standard_deviation(values)
se = mh.standard_error(values)
```

---

## Weighted Statistics

Use measurement uncertainties as inverse-variance weights with `sigma=`.

- `weighted_mean(x, w=None, sigma=None)`
- `weighted_variance(x, w=None, sigma=None, ddof=1, tipo="frecuentista")`

**Example (weights from quantity sigmas):**

```python
values, sigmas = mh.value_quantity(times)
mu_w = mh.weighted_mean(values, sigma=sigmas)
var_w = mh.weighted_variance(values, sigma=sigmas)
```

---

## Confidence Intervals

- `confidence_interval(x, nivel=0.95, distribucion="normal", sigma=None)`

Supported distributions:
- `"normal"`
- `"poisson"`
- `"binomial"`

**Example:**

```python
values, _ = mh.value_quantity(times)
ci = mh.confidence_interval(values, nivel=0.95, distribucion="normal")
print(ci)
```

---

## Hypothesis Tests

- `mean_test(x, mu0, alternativa="dos_colas", distribucion="normal", sigma=None)`
- `ks_test(x, distribucion="normal")`

**Example (mean test with quantity):**

```python
values, _ = mh.value_quantity(times)
test = mh.mean_test(values, mu0=2.0, alternativa="dos_colas", distribucion="normal")
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

values, sigmas = mh.value_quantity(measurements)

g_mean = mh.weighted_mean(values, sigma=sigmas)
g_se = mh.standard_error(values)

g_summary = mh.quantity(g_mean, g_se, "m/s^2", symbol="\\bar{g}")
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
| `weighted_mean(x, w, sigma)` | Weighted mean (use `sigma` for inverse-variance weights) |
| `weighted_variance(x, w, sigma, ddof, tipo)` | Weighted variance |
| `confidence_interval(x, nivel, distribucion, sigma)` | Confidence interval |
| `mean_test(x, mu0, alternativa, distribucion, sigma)` | Mean hypothesis test |
| `ks_test(x, distribucion)` | Kolmogorov-Smirnov test |

---

## Next Steps

- See [README_incertidumbres.md](README_incertidumbres.md) to build `quantity` objects
- See [README_latex_tools.md](README_latex_tools.md) to export statistical results to LaTeX
- See [README_graficos.md](README_graficos.md) to plot measurements and summaries
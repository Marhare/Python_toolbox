# ajustes.py

## Purpose
Weighted least‑squares (WLS) curve fitting with covariance calculation for uncertainty propagation. Designed for known experimental uncertainties in `y`.

## Assumptions
- `sy` are known absolute uncertainties in `y`.
- Residuals are independent and Gaussian.
- Uses `absolute_sigma=True` (no automatic rescaling).

## Main API
- `ajuste_lineal(x, y, sy=None)`
  - Analytic fit of `y = a + b·x`.
  - Returns parameters, errors, covariance, `yfit`, `chi2`, `ndof`, `chi2_red`, `p`.

- `ajuste_polinomico(x, y, grado, sy=None)`
  - WLS polynomial fit (coefficients in descending order).
  - Returns coefficients, errors, covariance, and chi‑square metrics.

- `ajuste(modelo, x, y, sy=None, p0=None, variable="x")`
  - Generic fit with `callable` or `sympy.Expr`.
  - If `sympy.Expr`, lambdifies and sorts symbolic parameters.

- `intervalo_confianza_parametros(resultado_ajuste, nivel=0.95)`
  - Parameter CI using Student‑t or normal depending on `ndof`.

- `incertidumbre_prediccion(resultado_ajuste, modelo, x0)`
  - Model confidence band (parameter uncertainty only).

## Typical errors
- Incompatible lengths among `x`, `y`, `sy`.
- Non‑positive `sy` values.
- Symbolic model without a valid variable.

## Recommended workflow
1. Fit (`ajuste_lineal`, `ajuste_polinomico`, or `ajuste`).
2. Inspect `chi2_red` and `p`.
3. Compute parameter CI.
4. Compute prediction uncertainty.

## Output
Results are always returned as dictionaries with stable keys for easy downstream use.

## Examples
```python
import numpy as np
from ajustes import ajustes

x = np.linspace(0, 10, 40)
y = 1.5 + 2.0*x
sy = 0.5*np.ones_like(x)

res = ajustes.ajuste_lineal(x, y, sy=sy)
print(res["parametros"], res["chi2_red"], res["p"])

pred = ajustes.incertidumbre_prediccion(res, lambda xv, a, b: a + b*xv, x0=7.5)
print(pred["y"], pred["sigma_modelo"])
```

## Mini examples (per function)
```python
from ajustes import ajustes

# ajuste_lineal
res_lin = ajustes.ajuste_lineal(x, y, sy=sy)

# ajuste_polinomico
res_poly = ajustes.ajuste_polinomico(x, y, grado=2, sy=sy)

# ajuste (modelo callable)
def modelo(xv, a, b):
  return a + b*xv
res_gen = ajustes.ajuste(modelo, x, y, sy=sy, p0=[1, 1])

# intervalo_confianza_parametros
ic = ajustes.intervalo_confianza_parametros(res_gen, nivel=0.95)

# incertidumbre_prediccion
pred = ajustes.incertidumbre_prediccion(res_gen, modelo, x0=5.0)
```
# incertidumbres.py

## Purpose
Create quantities with uncertainty and symbolic propagation with SymPy.

## Main API
- `u(x, sigmax=0.0)`
  - Returns `ufloat` if `x` is scalar.
  - Returns `unumpy.uarray` if `x` is an array.

- `propagacion_incertidumbre_sympy(f, vars_, valores, sigmas, cov=None, simplify=True)`
  - Computes gradient, propagated variance, and sigma.
  - Returns numeric values and LaTeX expressions.

## Dependencies
- `uncertainties`
- `sympy`

## Notes
- Integrates with `latex_tools` to generate LaTeX.
- Optional `cov` enables full covariances.

## Typical errors
- `sigmas` with negative values.
- `cov` with invalid dimensions.
- Missing variables in `valores` or `sigmas`.

## Examples
```python
from incertidumbres import incertidumbres

u = incertidumbres.u([1.0, 2.0, 3.0], [0.1, 0.1, 0.2])
```

## Mini examples (per function)

### u(x, sigmax=0.0)
**Case 1 (typical - scalar):** If you do this:
```python
from incertidumbres import incertidumbres

m = incertidumbres.u(9.81, 0.05)
print(m)
print(m.nominal_value, m.std_dev)
```
You do this: Create a quantity with nominal value 9.81 and uncertainty 0.05.

You get this:
```
9.81+/-0.05
9.81 0.05
```

**Case 2 (typical - array):** If you do this:
```python
x = incertidumbres.u([1.0, 2.0, 3.0], [0.1, 0.2, 0.15])
print(x)
resultado = x * 2
print(resultado)
```
You do this: Create an array of quantities and multiply it.

You get this:
```
array([1.0+/-0.1, 2.0+/-0.2, 3.0+/-0.15], dtype=object)
array([2.0+/-0.2, 4.0+/-0.4, 6.0+/-0.3], dtype=object)
```

**Case 3 (edge):** If you do this:
```python
z = incertidumbres.u(5.0, 0.0)
print(z)
y = z + 3
print(y)
```
You do this: Quantity with zero propagated uncertainty.

You get this:
```
5.0+/-0.0
8.0+/-0.0
```

### propagacion_incertidumbre_sympy(f, vars_, valores, sigmas, cov=None, simplify=True)
**Case 1 (typical):** If you do this:
```python
import sympy as sp
from incertidumbres import incertidumbres

x, y = sp.symbols('x y')
f = x**2 + y
valores = {'x': 2.0, 'y': 3.0}
sigmas = {'x': 0.1, 'y': 0.2}

resultado = incertidumbres.propagacion_incertidumbre_sympy(
    f, [x, y], valores, sigmas
)
print(resultado["sigma"])
print(resultado["valor"])
```
You do this: Propagate uncertainty for f=x²+y with x=2±0.1, y=3±0.2.

You get this:
```
sigma ≈ 0.408...
valor = 7.0
```

**Case 2 (edge - linear function):** If you do this:
```python
x, y = sp.symbols('x y')
f = 3*x + 2*y
valores = {'x': 1.0, 'y': 2.0}
sigmas = {'x': 0.1, 'y': 0.1}

resultado = incertidumbres.propagacion_incertidumbre_sympy(
    f, [x, y], valores, sigmas
)
print(resultado["sigma"])
```
You do this: Propagation in a linear function (σ_f = √(3²σ_x² + 2²σ_y²)).

You get this:
```
sigma = sqrt(9*0.01 + 4*0.01) ≈ 0.361...
```
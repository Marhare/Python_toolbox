# montecarlo.py

## Purpose
Monte Carlo utilities for integration and uncertainty propagation.

## API
- `integral_1d(f, a, b, n=10000)`
    - Estimates the integral by uniform sampling.
    - Returns `valor` and `error`.

- `propagacion(fun, generadores, n=10000)`
    - `generadores`: list of functions that generate samples.
    - Returns output `media` and `sigma`.

## Notes
- Error decreases as $1/\sqrt{n}$.
- Suitable for functions without a simple antiderivative or with high variability.

## Examples
```python
from montecarlo import montecarlo

res = montecarlo.integral_1d(lambda x: x**2, 0, 1, n=5000)
print(res["valor"], res["error"])
```

## Mini examples (per function)

### integral_1d(f, a, b, n=10000)
**Case 1 (typical):** If you do this:
```python
from montecarlo import montecarlo
import numpy as np

def f(x):
    return x**2

res = montecarlo.integral_1d(f, 0, 1, n=10000)
print(f"Valor: {res['valor']:.4f}")
print(f"Error: {res['error']:.4f}")
print(f"Theoretical value: 1/3 = {1/3:.4f}")
```
You do this: Estimate ∫₀¹ x² dx by uniform sampling (10k samples).

You get this:
```
Valor: 0.3346 (aprox)
Error: 0.0032
Theoretical value: 0.3333
```

**Case 2 (edge):** If you do this:
```python
def f(x):
    return 1.0  # constant function

res = montecarlo.integral_1d(f, 2, 5, n=5000)  # ∫₂⁵ 1 dx = 3
print(f"Valor: {res['valor']:.4f}")
print(f"Theoretical: 3.0")
```
You do this: Trivial integral of a constant function (width * height).

You get this:
```
Valor: 2.9825 (aprox, cercano a 3.0)
Theoretical: 3.0
```

### propagacion(fun, generadores, n=10000)
**Case 1 (typical):** If you do this:
```python
from montecarlo import montecarlo
import numpy as np

# x ~ N(2, 0.1), y ~ N(3, 0.2), calcular z = x*y
gen_x = lambda: np.random.normal(2.0, 0.1)
gen_y = lambda: np.random.normal(3.0, 0.2)

def fun(x, y):
    return x * y

res = montecarlo.propagacion(
    fun,
    [gen_x, gen_y],
    n=5000
)
print(f"Media: {res['media']:.4f}")
print(f"Sigma: {res['sigma']:.4f}")
print(f"Valor esperado: 6.0 ± √(0.01*9 + 0.04*4) ≈ 6.0 ± 0.514")
```
You do this: Propagate uncertainty from x and y through z=x*y.

You get this:
```
Media: 6.0023
Sigma: 0.5180
```

**Case 2 (edge):** If you do this:
```python
gen_const = lambda: 5.0  # siempre retorna 5

def fun(x):
    return x + 1

res = montecarlo.propagacion(fun, [gen_const], n=1000)
print(f"Media: {res['media']}")
print(f"Sigma: {res['sigma']}")
```
You do this: Propagation with deterministic input (no variability).

You get this:
```
Media: 6.0
Sigma: 0.0 (o muy cercana a 0)
```
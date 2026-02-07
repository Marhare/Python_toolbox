# uncertainties.py

## Purpose
Define quantities with uncertainty and propagate them symbolically and numerically.

## Data model
Each quantity is a dict with stable keys:

- `symbol`: str
- `unit`: str
- `expr`: SymPy expression or string, or None
- `measure`: (value, sigma) or None
- `result`: (value, sigma) or None

`expr` is the definition, `measure` is the experimental input, and `result` stores
computed values without altering the definition.

## Main API
- `quantity(value, sigma, unit, expr=None, symbol=None)`
  - Creates a quantity with measurement, expression, or both.
- `register(*quantities)`
  - Builds a registry by inferring symbols from caller variable names.
- `propagate_quantity(target, magnitudes, simplify=True)`
  - Propagates from fundamental measurements and stores results in `result`.
- `value_quantity(q)`
  - Returns numeric data using rule: `result` first, then `measure`.
- `propagate(expr, values, sigmas, simplify=True)`
- `uncertainty_propagation(f, vars_, values, sigmas, cov=None, simplify=True)`

## Dependencies
- `numpy`
- `sympy`

## Notes
- Integrates with `latex_tools.latex_quantity` for LaTeX output.
- `result` is cached output and never used as a physical input.

## Typical errors
- Missing symbols in the registry.
- Negative sigma values.
- Circular dependencies between expressions.

## Example
```python
import marhare as mh
import numpy as np

V = mh.quantity(np.array([1.0, 2.0]), np.array([0.1, 0.1]), "V")
I = mh.quantity(np.array([0.2, 0.3]), np.array([0.01, 0.01]), "A")
R = mh.quantity("V/I", "ohm")

magnitudes = mh.register(V, I, R)
res = mh.propagate_quantity("R", magnitudes)
print(res["value"], res["uncertainty"])
```
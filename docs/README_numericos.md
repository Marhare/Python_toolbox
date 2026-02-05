# numericos.py

## Purpose
Numeric‑symbolic calculator with auto‑detection: differentiation, integration, roots, and evaluation.

## API principal
- `derivar(expr, var, metodo="auto", h=1e-5)`
  - If `expr` is symbolic → differentiates with SymPy.
  - If `expr` is callable → returns a numeric derivative.

- `integrar_indefinida(expr, var)`
  - Symbolic antiderivative when possible.
  - If `expr` is callable, returns the accumulated integral from 0.

- `integrar_definida(expr, var, a, b)`
  - Symbolic when possible; otherwise uses numerical integration.

- `raiz_numerica(f, x0)`
  - Root using Newton (numeric).

- `evaluar(expr, valores)`
  - Evaluates SymPy expressions with a values dictionary.

- `rk4(f, intervalo, y0, dt)`
  - ODE integration with Runge‑Kutta 4.

## Notes
- Mixes `sympy.Expr`, `str`, or `callable` inputs.
- Keeps compatibility with legacy functions (`derivada`).

## Examples
```python
from numericos import numericos

# ODE with RK4
def f(t, y):
  return -0.8*y
rk = numericos.rk4(f, (0, 5), y0=1.0, dt=0.1)
print(rk["y"][-1])
```

## Mini examples (per function)
```python
from numericos import numericos

# derivar (symbolic)
dx = numericos.derivar("sin(x) + x**2", "x")

# derivar (numeric)
df = numericos.derivar(lambda t: t**2, "x", metodo="numerico")

# integrar_indefinida (symbolic)
F = numericos.integrar_indefinida("exp(-x)", "x")

# integrar_definida (numeric if no closed form)
I = numericos.integrar_definida(lambda t: t**2, "x", 0, 1)

# raiz_numerica
r = numericos.raiz_numerica(lambda t: t**2 - 2, 1.0)

# evaluar
val = numericos.evaluar("sin(x) + x**2", {"x": 1.2})
```
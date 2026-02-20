# functions.py – Symbolic Functions

## Purpose

Provide a lightweight symbolic `Function` object built on SymPy with lazy numeric compilation, operator overloading, and simple calculus helpers. The goal is to make symbolic expressions easy to combine, evaluate, and pass into other modules (graphics, latex, etc.).

---

## Core Class: `Function`

```python
from marhare import Function

f = Function("x**2 + 2*x + 1")
print(f(2))  # 9
```

### Constructor

```python
Function(expr, *args, vars=None, params=None, backend="numpy", symbol=None, indices=None)
```

- `expr`: string, SymPy expression, list, or tuple (for vectors/matrices).
- `vars`: optional list of variables (order matters). If omitted for single-variable expressions, auto-detected.
- `params`: optional list of parameters (distinct from variables).
- `backend`: backend passed to `sympy.lambdify` (default "numpy").
- `symbol`: optional symbol name for LaTeX rendering.
- `indices`: optional index notation for LaTeX (e.g., `["cov", "contra"]`).

### Evaluation

`Function` is callable and supports positional or named arguments:

```python
f = Function("x**2 + y", vars=["x", "y"])
print(f(2, 3))       # 7
print(f(x=2, y=3))   # 7

# Single-variable functions auto-detect vars
g = Function("sin(x)")
print(g(0))  # 0.0
```

---

## Operator Overloads

You can combine `Function` objects with scalars or other functions:

```python
f = Function("x", vars=["x"])
g = Function("sin(x)", vars=["x"])

h = 3*f + g - 2
k = f**2 / (1 + g)
print(h(1))  # Evaluates 3*1 + sin(1) - 2
```

All results are new `Function` objects with merged variables when necessary.

---

## Calculus Helpers

### Derivative: `D` or `dt`

Both `D` and `dt` compute the total derivative (with chain rule). Use `D` for brevity or `dt` for clarity.

```python
from marhare import Function, D, dt

f = Function("x**3", vars=["x"])
df = D(f, "x")  # Same as dt(f, "x")
print(df(2))  # 12
```

**Note:** For partial derivatives (without chain rule), use `dp` instead:

```python
from marhare import Function, dp

f = Function("x**2 + y**2", vars=["x", "y"])
df_dx = dp(f, "x")  # 2*x
print(df_dx(3, 4))  # 6
```

### Integral: `I`

```python
from marhare import Function, I

f = Function("3*x", vars=["x"])
F = I(f, "x")
print(F.expr)  # 3*x**2/2

# Definite integral
val = I(f, "x", interval=(0, 2))
print(val)  # 6
```

---

## Shifts and Substitutions

If you want $f(x-3)$, substitute in the SymPy expression:

```python
import sympy as sp
from marhare import Function, I

x = sp.Symbol("x")
f = Function("3*x", vars=["x"])
# Substitute x -> x-3
shifted = Function(f.expr.subs(x, x-3), vars=["x"])
F = I(shifted, "x")
print(F.expr)  # 3*(x-3)**2/2
```

---

## Interactions with Other Modules

### Graphics

`marhare.plot()` accepts `Function` objects directly and evaluates them on a dense grid. This makes it easy to overlay symbolic curves with data.

```python
import marhare as mh
import numpy as np, vars=["x"])

# Plot symbolic function over the x range
x = np.linspace(0, 2*np.pi, 50)
f = mh.Function("sin(x)")

mh.plot(x, f, label="sin(x)")
```

### LaTeX Tools

`Function.latex()` returns a SymPy LaTeX string. You can embed it in documents alongside values formatted with `latex_tools`.

```python
from marhare import Function
, vars=["x"]
f = Function("x**2 + 2*x + 1")
print(f.latex())  # x^{2} + 2 x + 1
```

---

## Tips

- If you want a specific variable order, pass `vars=["y", "x"]`.
- Use `backend="math"` for pure-Python evaluation when NumPy is not available.
- `Function` is a thin wrapper around SymPy, so any SymPy expression can be used as input.

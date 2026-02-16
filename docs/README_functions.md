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
Function(expr_str, *, vars=None, backend="numpy")
```

- `expr_str`: string or SymPy expression.
- `vars`: optional list of variables (order matters). If omitted, variables are sorted by name.
- `backend`: backend passed to `sympy.lambdify`.

### Evaluation

`Function` is callable and supports positional or named arguments:

```python
f = Function("x**2 + y")
print(f(2, 3))
print(f(x=2, y=3))
```

---

## Operator Overloads

You can combine `Function` objects with scalars or other functions:

```python
f = Function("x")
g = Function("sin(x)")

h = 3*f + g - 2
k = f**2 / (1 + g)
```

All results are new `Function` objects with merged variables.

---

## Calculus Helpers

### Derivative: `D`

```python
from marhare import Function, D

f = Function("x**3")
df = D(f, "x")
print(df(2))  # 12
```

### Integral: `I`

```python
from marhare import Function, I

f = Function("3*x")
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

f = Function("3*x")
shifted = Function(f.expr.subs({"x": "x-3"}))
F = I(shifted, "x")
```

---

## Interactions with Other Modules

### Graphics

`marhare.plot()` accepts `Function` objects directly and evaluates them on a dense grid. This makes it easy to overlay symbolic curves with data.

```python
import marhare as mh
import numpy as np

x = np.linspace(0, 2*np.pi, 50)
f = mh.Function("sin(x)")

mh.plot(x, f, label="sin(x)")
```

### LaTeX Tools

`Function.latex()` returns a SymPy LaTeX string. You can embed it in documents alongside values formatted with `latex_tools`.

```python
from marhare import Function

f = Function("x**2 + 2*x + 1")
print(f.latex())  # x^{2} + 2 x + 1
```

---

## Tips

- If you want a specific variable order, pass `vars=["y", "x"]`.
- Use `backend="math"` for pure-Python evaluation when NumPy is not available.
- `Function` is a thin wrapper around SymPy, so any SymPy expression can be used as input.

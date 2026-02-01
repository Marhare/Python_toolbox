# numericos.py

## Propósito
Calculadora numérico‑simbólica con auto‑detección: derivación, integración, raíces y evaluación.

## API principal
- `derivar(expr, var, metodo="auto", h=1e-5)`
  - Si `expr` es simbólica → deriva con SymPy.
  - Si `expr` es callable → devuelve derivada numérica.

- `integrar_indefinida(expr, var)`
  - Primitiva simbólica si es posible.
  - Si `expr` es callable, devuelve integral acumulada desde 0.

- `integrar_definida(expr, var, a, b)`
  - Simbólica si se puede; si no, usa integración numérica.

- `raiz_numerica(f, x0)`
  - Raíz usando Newton (numérico).

- `evaluar(expr, valores)`
  - Evalúa expresiones SymPy con un diccionario de valores.

- `rk4(f, intervalo, y0, dt)`
  - Integración de EDOs por Runge‑Kutta 4.

## Notas
- Mezcla entradas `sympy.Expr`, `str` o `callable`.
- Mantiene compatibilidad con funciones legacy (`derivada`).

## Ejemplos
```python
from numericos import numericos

# EDO con RK4
def f(t, y):
  return -0.8*y
rk = numericos.rk4(f, (0, 5), y0=1.0, dt=0.1)
print(rk["y"][-1])
```

## Mini ejemplos (por función)
```python
from numericos import numericos

# derivar (simbólico)
dx = numericos.derivar("sin(x) + x**2", "x")

# derivar (numérico)
df = numericos.derivar(lambda t: t**2, "x", metodo="numerico")

# integrar_indefinida (simbólica)
F = numericos.integrar_indefinida("exp(-x)", "x")

# integrar_definida (numérica si no hay primitiva cerrada)
I = numericos.integrar_definida(lambda t: t**2, "x", 0, 1)

# raiz_numerica
r = numericos.raiz_numerica(lambda t: t**2 - 2, 1.0)

# evaluar
val = numericos.evaluar("sin(x) + x**2", {"x": 1.2})
```
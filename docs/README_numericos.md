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
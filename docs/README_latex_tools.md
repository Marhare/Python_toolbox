# LaTeX Formatting Layer

Navigation: [Documentation Index](INDEX.md) | [Main README](../README.md)

This guide documents marhare.latex, the presentation layer.

## Recommended Imports

```python
from marhare.latex import latex_quantity, valor_pm, tabla_latex, exportar
from marhare.quantities import quantity
```

## Main Functions

- latex_quantity(obj, cifras=2, siunitx=False)
- valor_pm(valor, sigma=None, ...)
- tabla_latex(filas, ...)
- exportar(filename, contenido, modo="w")

## 1) latex_quantity for quantity-like objects

```python
from marhare.quantities import quantity
from marhare.latex import latex_quantity

V = quantity(5.234, 0.048, "V", symbol="U")
print(latex_quantity(V, cifras=2))
```

## 2) Formatting computed Quantity results

```python
from marhare.quantities import quantity
from marhare.latex import latex_quantity

U = quantity(12.5, 0.3, "V", symbol="U")
I = quantity(2.5, 0.1, "A", symbol="I")
R_res = U / I

print(latex_quantity(R_res, cifras=2))
```

## 3) Scalar and vector formatting with valor_pm

```python
from marhare.latex import valor_pm
import numpy as np

print(valor_pm(9.81, 0.05, unidad="m/s^2", cifras=2))

x = np.array([1.0, 2.0, 3.0])
sx = np.array([0.1, 0.1, 0.1])
print(valor_pm(x, sx, unidad="s", cifras=2, caption="Measurements"))
```

## 4) Build custom tables with tabla_latex

```python
from marhare.latex import tabla_latex

rows = [
    ["1.20 +/- 0.03", "5.0 +/- 0.2"],
    ["1.22 +/- 0.03", "5.1 +/- 0.2"],
]

tex = tabla_latex(
    rows,
    headers=["Voltage", "Current"],
    caption="Raw measurements",
    label="tab:raw",
)
print(tex)
```

## 5) Export to file

```python
from marhare.latex import exportar, latex_quantity
from marhare.quantities import quantity

q = quantity(3.14, 0.02, "1", symbol="pi")
tex = latex_quantity(q)
exportar("resultado.tex", tex)
```

## Notes

- marhare.latex depends on marhare.quantities.
- There is no reverse dependency from quantities to latex.
- Group-based formatting examples were removed because grouped quantities are not supported in the current architecture.

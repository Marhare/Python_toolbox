# latex_tools.py

## Purpose
Scientific LaTeX generation: metrological rounding, values with uncertainty, tables, and export to `.tex`.

## Main API
- `redondeo_incertidumbre(valor, sigma, cifras=2)`
  - Standard uncertainty rounding (1–2 significant digits).

- `valor_pm(valor, sigma=None, unidad=None, cifras=2, siunitx=False, ...)`
  - Scalar → `(v ± s)`.
  - Vector/matrix → configured LaTeX table.

- `expr_to_latex(expr, simplify=True)`
  - Converts SymPy expressions to LaTeX.

- `exportar(filename, contenido, modo="w")`
  - Writes LaTeX content to a file.

## Configuration
- `TABLA_CONFIG` controls table style (lines, size, environment).

## Notes
- Compatible with `siunitx` if `unidad` is provided and `siunitx=True`.
- Integrates with `incertidumbres.py`.

## Examples
```python
from latex_tools import latex_tools

tex = latex_tools.valor_pm(9.81, 0.05, unidad="m/s^2", cifras=2)
latex_tools.exportar("salidas/resultados.tex", tex)
```

## Mini examples (per function)

### redondeo_incertidumbre(valor, sigma, cifras=2)
**Case 1 (typical):** If you do this:
```python
from latex_tools import latex_tools

v, s = latex_tools.redondeo_incertidumbre(3.14159, 0.012345, cifras=2)
print(f"{v} ± {s}")
```
You do this: Round value and uncertainty to 2 significant digits.

You get this:
```
3.14 ± 0.01
```

**Case 2 (edge):** If you do this:
```python
v, s = latex_tools.redondeo_incertidumbre(100.5, 50.0, cifras=1)
print(f"{v} ± {s}")
```
You do this: Rounding with large uncertainty.

You get this:
```
100 ± 50
```

### valor_pm(valor, sigma=None, unidad=None, cifras=2, siunitx=False)
**Case 1 (typical - scalar):** If you do this:
```python
from latex_tools import latex_tools

tex = latex_tools.valor_pm(9.81, 0.05, unidad="m/s^2", cifras=2)
print(tex)
```
You do this: Generate LaTeX for a quantity with a unit.

You get this:
```
$9.81 \pm 0.05$ m/s$^2$
```

**Case 2 (typical - array):** If you do this:
```python
import numpy as np

valores = np.array([1.5, 2.3, 3.1])
sigmas = np.array([0.1, 0.15, 0.2])
tex = latex_tools.valor_pm(valores, sigmas, cifras=1)
print(tex)
```
You do this: Generate a LaTeX table of quantities.

You get this:
```
\begin{tabular}{lr}
Valor & Incertidumbre \\
1.5 & 0.1 \\
2.3 & 0.1 \\
3.1 & 0.2 \\
\end{tabular}
```

**Case 3 (edge):** If you do this:
```python
tex = latex_tools.valor_pm(42.0, siunitx=True, unidad="kg")
print(tex)
```
You do this: siunitx formatting for compilation with the siunitx package.

You get this:
```
\SI{42.0}{kg}
```

### expr_to_latex(expr, simplify=True)
**Case 1 (typical):** If you do this:
```python
import sympy as sp
from latex_tools import latex_tools

x = sp.Symbol('x')
expr = sp.diff(sp.sin(x) * sp.exp(x), x)
latex_str = latex_tools.expr_to_latex(expr, simplify=True)
print(latex_str)
```
You do this: Convert a symbolic derivative to LaTeX.

You get this:
```
e^{x} \sin(x) + e^{x} \cos(x)
```

**Case 2 (edge):** If you do this:
```python
expr = sp.integrate(sp.exp(-x**2), (x, 0, sp.oo))
latex_str = latex_tools.expr_to_latex(expr)
print(latex_str)
```
You do this: Definite integral with symbolic limits.

You get this:
```
\frac{\sqrt{\pi}}{2}
```

### exportar(filename, contenido, modo="w")
**Case 1 (typical):** If you do this:
```python
from latex_tools import latex_tools

contenido = """\\documentclass{article}
\\usepackage{amsmath}
\\begin{document}
$E = mc^2$
\\end{document}"""

latex_tools.exportar("salidas/ecuacion.tex", contenido)
print("Exportado exitosamente")
```
You do this: Save a full LaTeX document to a file.

You get this:
```
Archivo salidas/ecuacion.tex creado con contenido LaTeX
Mensaje: Exportado exitosamente
```

**Case 2 (edge):** If you do this:
```python
latex_tools.exportar("salidas/vacio.tex", "", modo="w")
print("Empty file created")
```
You do this: Save an empty LaTeX file.

You get this:
```
File salidas/vacio.tex created (size 0 bytes)
Message: Empty file created
```
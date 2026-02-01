# latex_tools.py

## Propósito
Generación de LaTeX científico: redondeo metrológico, valores con incertidumbre, tablas y exportación a `.tex`.

## API principal
- `redondeo_incertidumbre(valor, sigma, cifras=2)`
  - Redondeo estándar de incertidumbres (1–2 cifras significativas).

- `valor_pm(valor, sigma=None, unidad=None, cifras=2, siunitx=False, ...)`
  - Escalar → `(v ± s)`.
  - Vector/matriz → tabla LaTeX configurada.

- `expr_to_latex(expr, simplify=True)`
  - Convierte expresiones SymPy a LaTeX.

- `exportar(filename, contenido, modo="w")`
  - Escribe el contenido LaTeX a un archivo.

## Configuración
- `TABLA_CONFIG` controla estilo de tablas (lineas, tamaño, entorno).

## Notas
- Compatible con `siunitx` si se provee `unidad` y `siunitx=True`.
- Integra con `incertidumbres.py`.

## Ejemplos
```python
from latex_tools import latex_tools

tex = latex_tools.valor_pm(9.81, 0.05, unidad="m/s^2", cifras=2)
latex_tools.exportar("salidas/resultados.tex", tex)
```

## Mini ejemplos (por función)

### redondeo_incertidumbre(valor, sigma, cifras=2)
**Caso 1 (típico):** Si aplicas esto:
```python
from latex_tools import latex_tools

v, s = latex_tools.redondeo_incertidumbre(3.14159, 0.012345, cifras=2)
print(f"{v} ± {s}")
```
haces esto: Redondeas valor e incertidumbre a 2 cifras significativas.

Obtienes esto:
```
3.14 ± 0.01
```

**Caso 2 (borde):** Si aplicas esto:
```python
v, s = latex_tools.redondeo_incertidumbre(100.5, 50.0, cifras=1)
print(f"{v} ± {s}")
```
haces esto: Redondeo con incertidumbre grande.

Obtienes esto:
```
100 ± 50
```

### valor_pm(valor, sigma=None, unidad=None, cifras=2, siunitx=False)
**Caso 1 (típico - escalar):** Si aplicas esto:
```python
from latex_tools import latex_tools

tex = latex_tools.valor_pm(9.81, 0.05, unidad="m/s^2", cifras=2)
print(tex)
```
haces esto: Generas LaTeX para magnitud con unidad.

Obtienes esto:
```
$9.81 \pm 0.05$ m/s$^2$
```

**Caso 2 (típico - array):** Si aplicas esto:
```python
import numpy as np

valores = np.array([1.5, 2.3, 3.1])
sigmas = np.array([0.1, 0.15, 0.2])
tex = latex_tools.valor_pm(valores, sigmas, cifras=1)
print(tex)
```
haces esto: Generas tabla LaTeX de magnitudes.

Obtienes esto:
```
\begin{tabular}{lr}
Valor & Incertidumbre \\
1.5 & 0.1 \\
2.3 & 0.1 \\
3.1 & 0.2 \\
\end{tabular}
```

**Caso 3 (borde):** Si aplicas esto:
```python
tex = latex_tools.valor_pm(42.0, siunitx=True, unidad="kg")
print(tex)
```
haces esto: Formato siunitx para compilación con paquete siunitx.

Obtienes esto:
```
\SI{42.0}{kg}
```

### expr_to_latex(expr, simplify=True)
**Caso 1 (típico):** Si aplicas esto:
```python
import sympy as sp
from latex_tools import latex_tools

x = sp.Symbol('x')
expr = sp.diff(sp.sin(x) * sp.exp(x), x)
latex_str = latex_tools.expr_to_latex(expr, simplify=True)
print(latex_str)
```
haces esto: Conviertes derivada simbólica a LaTeX.

Obtienes esto:
```
e^{x} \sin(x) + e^{x} \cos(x)
```

**Caso 2 (borde):** Si aplicas esto:
```python
expr = sp.integrate(sp.exp(-x**2), (x, 0, sp.oo))
latex_str = latex_tools.expr_to_latex(expr)
print(latex_str)
```
haces esto: Integral indefinida con límites simbólicos.

Obtienes esto:
```
\frac{\sqrt{\pi}}{2}
```

### exportar(filename, contenido, modo="w")
**Caso 1 (típico):** Si aplicas esto:
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
haces esto: Guardas documento LaTeX completo a archivo.

Obtienes esto:
```
Archivo salidas/ecuacion.tex creado con contenido LaTeX
Mensaje: Exportado exitosamente
```

**Caso 2 (borde):** Si aplicas esto:
```python
latex_tools.exportar("salidas/vacio.tex", "", modo="w")
print("Archivo vacío creado")
```
haces esto: Guardas archivo LaTeX vacío.

Obtienes esto:
```
Archivo salidas/vacio.tex creado (tamaño 0 bytes)
Mensaje: Archivo vacío creado
```
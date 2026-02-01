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
# graficos.py

## Propósito
Visualización científica de alto nivel basada en objetos semánticos. El usuario declara **qué** dibujar y el motor decide **cómo** dibujarlo.

## Objetos semánticos
- `Serie(x, y, label=None, marker=None)`
- `SerieConError(x, y, sy=None, sx=None, label=None)`
- `Histograma(data, bins=30, label=None)`
- `Ajuste(x, yfit, label=None)`
- `Banda(x, y_low, y_high, label=None)`
- `Serie3D(x, y, z, label=None)`
- `Panel(*objetos)`
- `Scene(*paneles, layout=None, dims="2D", figsize=None, xlabel=None, ylabel=None, title=None)`

## Motor de plotting
- `plot(*objetos, layout=None, dims="2D", show=True, figsize=None, xlabel=None, ylabel=None, title=None, **kwargs)`
  - Soporta `Scene` como argumento único.
  - `Panel` permite agrupar objetos en el mismo subplot.

## Estilo
- Configuración global en `PLOT_DEFAULTS` (paleta, tamaño, grid, tipografía).
- Estilo se ajusta con `**kwargs` en `plot()`.

## Guardado
- Integración con `guardar()` (si aplica en el módulo) para exportar a PDF/PNG.

## Notas
- `Scene` es la unidad recomendada para animaciones.
- `dims="3D"` requiere `Serie3D`.
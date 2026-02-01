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

## Ejemplos
```python
import numpy as np
from graficos import graficos

x = np.linspace(0, 1, 20)
y = 1 + 2*x
yfit = y
serie = graficos.Serie(x, y, label="Datos")
ajuste = graficos.Ajuste(x, yfit, label="Ajuste")
graficos.plot(serie, ajuste, title="Datos + Ajuste")
```

## Mini ejemplos (por función)

### Serie(x, y, label=None, marker=None)
**Caso 1 (típico):** Si aplicas esto:
```python
import numpy as np
from graficos import graficos

x = np.array([1, 2, 3, 4])
y = np.array([2, 4, 6, 8])
s = graficos.Serie(x, y, label="y=2x")
graficos.plot(s)
```
haces esto: Creas una serie de puntos y la graficas.

Obtienes esto:
```
Gráfico con 4 puntos conectados, etiqueta "y=2x" en leyenda
```

**Caso 2 (borde):** Si aplicas esto:
```python
s = graficos.Serie(np.array([0]), np.array([5]), marker="o")
graficos.plot(s)
```
haces esto: Serie con un único punto y marcador específico.

Obtienes esto:
```
Gráfico con un punto circular en (0, 5)
```

### SerieConError(x, y, sy=None, sx=None, label=None)
**Caso 1 (típico):** Si aplicas esto:
```python
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.1, 5.9])
sy = np.array([0.2, 0.2, 0.3])
s = graficos.SerieConError(x, y, sy=sy, label="Datos ±σ")
graficos.plot(s)
```
haces esto: Series con barras de error en y.

Obtienes esto:
```
Gráfico con 3 puntos, cada uno con barra vertical de incertidumbre
```

**Caso 2 (borde):** Si aplicas esto:
```python
x = np.array([1.0, 2.0])
y = np.array([3.0, 4.0])
sy = np.array([0.0, 0.0])
s = graficos.SerieConError(x, y, sy=sy)
graficos.plot(s)
```
haces esto: SerieConError con errores nulos.

Obtienes esto:
```
Gráfico sin barras de error visibles (como Serie normal)
```

### Histograma(data, bins=30, label=None)
**Caso 1 (típico):** Si aplicas esto:
```python
import numpy as np
from graficos import graficos

data = np.random.default_rng(0).normal(0, 1, 1000)
hist = graficos.Histograma(data, bins=40, label="Normal(0,1)")
graficos.plot(hist)
```
haces esto: Histograma de 1000 muestras normales.

Obtienes esto:
```
Gráfico con 40 bins, forma de campana centrada en 0
```

**Caso 2 (borde):** Si aplicas esto:
```python
hist = graficos.Histograma(np.array([5.0, 5.0, 5.0]), bins=10)
graficos.plot(hist)
```
haces esto: Histograma con valores idénticos.

Obtienes esto:
```
Un único bin con altura 3, resto vacío
```

### Ajuste(x, yfit, label=None)
**Caso 1 (típico):** Si aplicas esto:
```python
x = np.linspace(0, 2, 50)
yfit = 1 + 0.5*x
ajuste = graficos.Ajuste(x, yfit, label="y=1+0.5x")
graficos.plot(ajuste)
```
haces esto: Graficas una curva de ajuste.

Obtienes esto:
```
Línea continua desde (0,1) a (2,2), etiquetada
```

**Caso 2 (borde):** Si aplicas esto:
```python
ayuste = graficos.Ajuste(np.array([1.0]), np.array([1.0]))
graficos.plot(ajuste)
```
haces esto: Ajuste con un solo punto.

Obtienes esto:
```
Un punto sin línea visible
```

### Banda(x, y_low, y_high, label=None)
**Caso 1 (típico):** Si aplicas esto:
```python
x = np.linspace(0, 1, 20)
y_low = x
y_high = x + 0.2
banda = graficos.Banda(x, y_low, y_high, label="Intervalo")
graficos.plot(banda)
```
haces esto: Área sombreada entre dos curvas.

Obtienes esto:
```
Área gris o coloreada entre y=x e y=x+0.2
```

**Caso 2 (borde):** Si aplicas esto:
```python
banda = graficos.Banda(np.array([0, 1]), np.array([1, 1]), np.array([1, 1]))
graficos.plot(banda)
```
haces esto: Banda con cero espesor (y_low = y_high).

Obtienes esto:
```
Línea horizontal en y=1
```
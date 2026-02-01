# estadistica.py

## Propósito
Estadística descriptiva, intervalos de confianza y tests de hipótesis con una API simple en español.

## Estadística descriptiva
- `media(x)`
- `varianza(x, ddof=1)`
- `desviacion_tipica(x, ddof=1)`
- `error_estandar(x)`

## Estadística ponderada
- `media_ponderada(x, w=None, sigma=None)`
- `varianza_ponderada(x, w=None, sigma=None, ddof=1, tipo="frecuentista")`

## Intervalos de confianza
- `intervalo_confianza(x, nivel=0.95, distribucion="normal", sigma=None)`
  - Normal, Poisson o Binomial según `distribucion`.

## Tests de hipótesis
- `test_media(x, mu0, alternativa="dos_colas", distribucion="normal", sigma=None)`
  - Normal: z‑test si `sigma` es conocida, t‑test si no.
  - Poisson: test exacto para la tasa.
  - Binomial: test exacto para probabilidad.

- `test_ks(x, distribucion="normal")`
  - Bondad de ajuste KS para normal o uniforme.

## Salida
Los métodos devuelven diccionarios con estadísticos, p‑valores y metadatos (`n`, `df`, etc.).

## Errores típicos
- Muestras vacías o con tamaños inválidos.
- Parámetros fuera de rango (p. ej. `sigma <= 0`, `nivel` fuera de (0,1)).
- Distribuciones no soportadas.

## Ejemplos
```python
import numpy as np
from estadistica import estadistica

x = np.random.default_rng(0).normal(0, 1, 30)
media = estadistica.media(x)
ic = estadistica.intervalo_confianza(x, nivel=0.95, distribucion="normal")
test = estadistica.test_media(x, mu0=0.0, distribucion="normal")
```

## Mini ejemplos (por función)

### media(x)
**Caso 1 (típico):** Si aplicas esto:
```python
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
media = estadistica.media(x)
```
haces esto: Calculas el promedio de 5 valores.

Obtienes esto:
```
3.0
```

**Caso 2 (borde):** Si aplicas esto:
```python
x = np.array([0.0])
media = estadistica.media(x)
```
haces esto: Calculas la media de un único valor.

Obtienes esto:
```
0.0
```

### varianza(x, ddof=1)
**Caso 1 (típico):** Si aplicas esto:
```python
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
varianza = estadistica.varianza(x, ddof=1)
```
haces esto: Calculas la varianza muestral (n-1).

Obtienes esto:
```
2.5
```

**Caso 2 (borde):** Si aplicas esto:
```python
x = np.array([5.0, 5.0, 5.0])
varianza = estadistica.varianza(x)
```
haces esto: Calculas varianza cuando todos los valores son idénticos.

Obtienes esto:
```
0.0
```

### desviacion_tipica(x, ddof=1)
**Caso 1 (típico):** Si aplicas esto:
```python
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
std = estadistica.desviacion_tipica(x)
```
haces esto: Calculas la desviación estándar.

Obtienes esto:
```
1.5811388...
```

**Caso 2 (borde):** Si aplicas esto:
```python
x = np.array([0.0, 0.0])
std = estadistica.desviacion_tipica(x, ddof=0)
```
haces esto: Cálculo poblacional de desviación.

Obtienes esto:
```
0.0
```

### error_estandar(x)
**Caso 1 (típico):** Si aplicas esto:
```python
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
se = estadistica.error_estandar(x)
```
haces esto: Calculas error estándar (σ/√n).

Obtienes esto:
```
0.7071067...
```

**Caso 2 (borde):** Si aplicas esto:
```python
x = np.array([10.0])
se = estadistica.error_estandar(x)
```
haces esto: Error estándar con una sola muestra (σ/√1).

Obtienes esto:
```
nan
```

### media_ponderada(x, w=None, sigma=None)
**Caso 1 (típico):** Si aplicas esto:
```python
x = np.array([1.0, 2.0, 3.0])
w = np.array([1.0, 2.0, 1.0])
mean_w = estadistica.media_ponderada(x, w=w)
```
haces esto: Calculas media ponderada con pesos [1, 2, 1].

Obtienes esto:
```
2.0
```

**Caso 2 (borde):** Si aplicas esto:
```python
x = np.array([5.0, 10.0])
sigma = np.array([1.0, 0.1])
mean_w = estadistica.media_ponderada(x, sigma=sigma)
```
haces esto: Pesos inversos a σ² para minimizar error.

Obtienes esto:
```
9.917355...
```

### intervalo_confianza(x, nivel=0.95, distribucion="normal", sigma=None)
**Caso 1 (típico):** Si aplicas esto:
```python
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
ic = estadistica.intervalo_confianza(x, nivel=0.95)
```
haces esto: Calculas IC al 95% con distribución normal.

Obtienes esto:
```
{'media': 3.0, 'lower': 0.627..., 'upper': 5.372..., 'error': 2.372..., 'n': 5, 'sigma': 1.581...}
```

**Caso 2 (borde):** Si aplicas esto:
```python
x = np.array([2.0, 2.0, 2.0, 2.0])
ic = estadistica.intervalo_confianza(x, nivel=0.99)
```
haces esto: IC con varianza cero y confianza alta.

Obtienes esto:
```
{'media': 2.0, 'lower': 2.0, 'upper': 2.0, 'error': 0.0, ...}
```

### test_media(x, mu0, alternativa="dos_colas", distribucion="normal", sigma=None)
**Caso 1 (típico):** Si aplicas esto:
```python
x = np.array([1.1, 1.9, 2.1, 1.8, 2.0])
test = estadistica.test_media(x, mu0=2.0, alternativa="dos_colas")
```
haces esto: Test t-bilaterial contra μ₀=2.0.

Obtienes esto:
```
{'media': 1.78, 'sigma_x': 0.450..., 't': -1.094..., 'p_valor': 0.346..., 'rechaza': False}
```

**Caso 2 (borde):** Si aplicas esto:
```python
x = np.array([5.0, 5.0, 5.0])
test = estadistica.test_media(x, mu0=5.0)
```
haces esto: Test contra valor idéntico a los datos.

Obtienes esto:
```
{'media': 5.0, 'sigma_x': 0.0, 't': 'inf' or 'nan', 'p_valor': 1.0 or nan, 'rechaza': False}
```

### test_ks(x, distribucion="normal")
**Caso 1 (típico):** Si aplicas esto:
```python
rng = np.random.default_rng(42)
x = rng.normal(0, 1, 50)
test = estadistica.test_ks(x, distribucion="normal")
```
haces esto: Test Kolmogorov-Smirnov sobre muestra normal.

Obtienes esto:
```
{'estadistico': 0.089..., 'p_valor': 0.689..., 'rechaza': False}
```

**Caso 2 (borde):** Si aplicas esto:
```python
x = np.linspace(0, 100, 100)
test = estadistica.test_ks(x, distribucion="uniforme")
```
haces esto: Test KS para distribución uniforme.

Obtienes esto:
```
{'estadistico': ~0.0, 'p_valor': > 0.9, 'rechaza': False}
```
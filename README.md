# Python Toolbox 🧰

A collection of utilities and tools for Python, designed to simplify common tasks in scientific computing, data analysis, and laboratory work.

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Description

`python_toolbox` is my personal library that centralizes functions I frequently use across different projects. The main purpose of this library is to avoid code duplication and maintain a set of optimized and tested tools. Through this development process, I continue to deepen my knowledge of Python architecture and numerical calculation.

## 🚀 Main Modules (Detailed)

**Documentación por módulo:**
- [docs/README_ajustes.md](docs/README_ajustes.md)
- [docs/README_animaciones.md](docs/README_animaciones.md)
- [docs/README_estadistica.md](docs/README_estadistica.md)
- [docs/README_fft_tools.md](docs/README_fft_tools.md)
- [docs/README_graficos.md](docs/README_graficos.md)
- [docs/README_incertidumbres.md](docs/README_incertidumbres.md)
- [docs/README_latex_tools.md](docs/README_latex_tools.md)
- [docs/README_montecarlo.md](docs/README_montecarlo.md)
- [docs/README_numericos.md](docs/README_numericos.md)

### ajustes.py
**Propósito:** ajuste de curvas por mínimos cuadrados ponderados (WLS) con covarianzas para propagación de incertidumbres.

**Supuestos:**
- `sy` son incertidumbres absolutas conocidas en `y`.
- Residuos gaussianos e independientes.
- `absolute_sigma=True` (sin reescalar errores).

**API principal:**
- `ajuste_lineal(x, y, sy=None)`
- `ajuste_polinomico(x, y, grado, sy=None)`
- `ajuste(modelo, x, y, sy=None, p0=None, variable="x")`
- `intervalo_confianza_parametros(resultado_ajuste, nivel=0.95)`
- `incertidumbre_prediccion(resultado_ajuste, modelo, x0)`

**Errores típicos:** longitudes incompatibles, `sy` no positivo, modelo inválido.

**Ejemplo rápido:**
```python
from ajustes import ajustes

res = ajustes.ajuste_lineal(x, y, sy=sy)
print(res["parametros"], res["chi2_red"], res["p"])
```

---

### animaciones.py
**Propósito:** motor temporal declarativo para animar objetos de `graficos.py`.

**API principal:**
- `animate(scene, evolve, duration, fps=30, speed=1.0, loop=False, show=True)`

**Reglas de `evolve`:**
- `Serie` → `y(t)`
- `Serie3D` → `(x, y, z)`
- `Banda` → `(y_low, y_high)`
- `Ajuste` → `yfit(t)`

**Notas:** en notebooks con backend inline puede verse un frame estático; se recomienda guardar a GIF/MP4.

**Ejemplo rápido:**
```python
from graficos import graficos
from animaciones import animaciones

serie = graficos.Serie(x, y)
scene = graficos.Scene(serie, title="Demo")
anim = animaciones.animate(scene, {serie: lambda t: y*np.cos(t)}, duration=2.0)
```

---

### estadistica.py
**Propósito:** estadística descriptiva, IC y tests de hipótesis.

**Descriptiva:** `media`, `varianza`, `desviacion_tipica`, `error_estandar`.

**Ponderada:** `media_ponderada`, `varianza_ponderada`.

**Intervalos de confianza:** `intervalo_confianza` (normal/poisson/binomial).

**Tests:**
- `test_media` (z/t, Poisson exacto, Binomial exacto)
- `test_ks` (normal o uniforme)

**Ejemplo rápido:**
```python
from estadistica import estadistica

res = estadistica.test_media(x, mu0=0.0, distribucion="normal")
print(res["estadistico"], res["p_valor"])
```

---

### incertidumbres.py
**Propósito:** magnitudes con incertidumbre y propagación simbólica.

**API:**
- `u(x, sigmax=0.0)`
- `propagacion_incertidumbre_sympy(f, vars_, valores, sigmas, cov=None, simplify=True)`

**Notas:** integra con `latex_tools` para LaTeX.

**Ejemplo rápido:**
```python
from incertidumbres import incertidumbres
u = incertidumbres.u(10.0, 0.2)
```

---

### fft_tools.py
**Propósito:** FFT unidimensional con `scipy.fft`.

**API:**
- `fft(signal, dt)`
- `espectro_potencia(signal, dt)`

**Ejemplo rápido:**
```python
from fft_tools import fft_tools
spec = fft_tools.espectro_potencia(signal, dt)
```

---

### graficos.py
**Propósito:** visualización científica con objetos semánticos y estilo coherente.

**Objetos:** `Serie`, `SerieConError`, `Histograma`, `Ajuste`, `Banda`, `Serie3D`, `Panel`, `Scene`.

**Motor:** `plot(*objetos, layout=None, dims="2D", show=True, ...)`

**Ejemplo rápido:**
```python
from graficos import graficos
serie = graficos.Serie(x, y, label="Datos")
graficos.plot(serie)
```

---

### montecarlo.py
**Propósito:** integración y propagación por Monte Carlo.

**API:**
- `integral_1d(f, a, b, n=10000)`
- `propagacion(fun, generadores, n=10000)`

**Ejemplo rápido:**
```python
from montecarlo import montecarlo
res = montecarlo.integral_1d(lambda t: t**2, 0, 1, n=5000)
```

---

### numericos.py
**Propósito:** calculadora numérico‑simbólica con auto‑detección.

**API principal:**
- `derivar`, `integrar_indefinida`, `integrar_definida`
- `raiz_numerica`, `evaluar`, `rk4`

**Ejemplo rápido (EDO con RK4):**
```python
from numericos import numericos
def f(t, y):
	return -0.8*y
rk = numericos.rk4(f, (0, 5), y0=1.0, dt=0.1)
```

---

### latex_tools.py
**Propósito:** LaTeX científico (redondeo metrológico, valores con incertidumbre, tablas y exportación).

**API principal:**
- `redondeo_incertidumbre`
- `valor_pm`
- `expr_to_latex`
- `exportar`

**Ejemplo rápido:**
```python
from latex_tools import latex_tools
tex = latex_tools.valor_pm(9.81, 0.05, unidad="m/s^2", cifras=2)
```

## 🛠️ Requirements

This toolbox primarily relies on the scientific Python stack:
- `numpy`
- `scipy`
- `matplotlib`
- `sympy`

## ⚙️ Installation

To use these tools in your local environment, clone the repository:

```bash
git clone https://github.com/tu-usuario/python_toolbox.git
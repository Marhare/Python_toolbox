# fft_tools.py

## Propósito
Herramientas simples para FFT unidimensional con `scipy.fft`.

## API
- `fft(signal, dt)`
  - Devuelve `freqs` y `fft` (transformada compleja) usando `rfft`.

- `espectro_potencia(signal, dt)`
  - Devuelve `freqs` y `psd` (|FFT|^2) para señal real.

## Notas
- `dt` es el paso temporal real.
- Usa `rfft`/`rfftfreq` (solo frecuencias no negativas).

## Salida
Diccionarios con `freqs` y `fft` o `psd`.

## Ejemplo
```python
from fft_tools import fft_tools

spec = fft_tools.espectro_potencia(signal, dt)
freqs, psd = spec["freqs"], spec["psd"]
```

## Mini ejemplos (por función)

### fft(signal, dt)
**Caso 1 (típico):** Si aplicas esto:
```python
import numpy as np
from fft_tools import fft_tools

dt = 0.01
t = np.arange(0, 1, dt)
signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz
resultado = fft_tools.fft(signal, dt)
freqs = resultado["freqs"]
fft_vals = resultado["fft"]
```
haces esto: FFT de una sinusoide pura de 5 Hz.

Obtienes esto:
```
freqs[:5] = [0., 1., 2., 3., 4.]
fft_vals[5] ≈ 50 (pico en 5 Hz, proporcional a amplitud * n/2)
```

**Caso 2 (borde):** Si aplicas esto:
```python
dt = 0.1
signal = np.array([1.0, 1.0, 1.0, 1.0])  # constante
resultado = fft_tools.fft(signal, dt)
```
haces esto: FFT de una señal constante (DC).

Obtienes esto:
```
freqs = [0., 2.5, 5., 7.5]
fft_vals[0] ≈ 4.0 (toda la energía en f=0)
fft_vals[1:] ≈ 0 (sin componentes AC)
```

### espectro_potencia(signal, dt)
**Caso 1 (típico):** Si aplicas esto:
```python
import numpy as np
from fft_tools import fft_tools

dt = 0.01
t = np.arange(0, 1, dt)
signal = np.sin(2 * np.pi * 3 * t) + 0.5 * np.sin(2 * np.pi * 7 * t)
spec = fft_tools.espectro_potencia(signal, dt)
freqs = spec["freqs"]
psd = spec["psd"]
```
haces esto: Espectro de potencia de suma de 3 Hz y 7 Hz.

Obtienes esto:
```
freqs[3] ≈ 3.0 Hz con psd máximo
freqs[7] ≈ 7.0 Hz con psd ≈ 1/4 del máximo (amplitud 0.5²)
```

**Caso 2 (borde):** Si aplicas esto:
```python
dt = 0.05
signal = np.random.normal(0, 0.1, 100)  # ruido blanco
spec = fft_tools.espectro_potencia(signal, dt)
```
haces esto: Espectro de poder para ruido aleatorio.

Obtienes esto:
```
psd relativamente uniforme en todas las frecuencias
(sin picos dominantes, dispersión aleatoria)
```
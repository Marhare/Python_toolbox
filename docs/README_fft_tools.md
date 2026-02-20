# fft_tools.py

## Purpose
Simple tools for 1D FFT with `scipy.fft`.

## API
- `fft(signal, dt)`
  - Returns `freqs` and `fft` (complex transform) using `rfft`.

- `power_spectrum(signal, dt)`
  - Returns `freqs` and `psd` (|FFT|^2) for real signals.

## Notes
- `dt` is the real time step.
- Uses `rfft`/`rfftfreq` (non‑negative frequencies only).

## Output
Dictionaries with `freqs` and `fft` or `psd`.

## Example
```python
from marhare.fft_tools import fft_tools

spec = fft_tools.power_spectrum(signal, dt)
freqs, psd = spec["freqs"], spec["psd"]
```

## Mini examples (per function)

### fft(signal, dt)
**Case 1 (typical):** If you do this:
```python
import numpy as np
from marhare.fft_tools import fft_tools

dt = 0.01
t = np.arange(0, 1, dt)
signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz
resultado = fft_tools.fft(signal, dt)
freqs = resultado["freqs"]
fft_vals = resultado["fft"]
```
You do this: FFT of a pure 5 Hz sine wave.

You get this:
```
freqs[:5] = [0., 1., 2., 3., 4., 5., ...]
fft_vals[5] ≈ 50 (peak at 5 Hz, proportional to amplitude * n/2)
```

**Case 2 (edge):** If you do this:
```python
dt = 0.1
signal = np.array([1.0, 1.0, 1.0, 1.0])  # constant
resultado = fft_tools.fft(signal, dt)
```
You do this: FFT of a constant (DC) signal.

You get this:
```
freqs = [0., 2.5, 5.]
fft_vals[0] ≈ 4.0 (all the energy at f=0)
fft_vals[1:] ≈ 0 (no AC components)
```

### power_spectrum(signal, dt)
**Case 1 (typical):** If you do this:
```python
import numpy as np
from marhare.fft_tools import fft_tools

dt = 0.01
t = np.arange(0, 1, dt)
signal = np.sin(2 * np.pi * 3 * t) + 0.5 * np.sin(2 * np.pi * 7 * t)
spec = fft_tools.power_spectrum(signal, dt)
freqs = spec["freqs"]
psd = spec["psd"]
```
You do this: Power spectrum of a sum of 3 Hz and 7 Hz.

You get this:
```
freqs[3] ≈ 3.0 Hz with maximum psd
freqs[7] ≈ 7.0 Hz with psd ≈ 1/4 of the maximum (amplitude 0.5²)
```

**Case 2 (edge):** If you do this:
```python
dt = 0.05
signal = np.random.normal(0, 0.1, 100)  # white noise
spec = fft_tools.power_spectrum(signal, dt)
```
You do this: Power spectrum for random noise.

You get this:
```
psd relatively uniform in all frequencies
(no dominant peaks, random dispersion)
```
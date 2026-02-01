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
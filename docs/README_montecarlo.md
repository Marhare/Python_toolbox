# montecarlo.py

## Propósito
Utilidades de Monte Carlo para integración y propagación de incertidumbre.

## API
- `integral_1d(f, a, b, n=10000)`
  - Estima la integral por muestreo uniforme.
  - Devuelve `valor` y `error`.

- `propagacion(fun, generadores, n=10000)`
  - `generadores`: lista de funciones que generan muestras.
  - Devuelve `media` y `sigma` de la salida.

## Notas
- El error disminuye como $1/\sqrt{n}$.
- Adecuado para funciones sin primitiva simple o alta variabilidad.
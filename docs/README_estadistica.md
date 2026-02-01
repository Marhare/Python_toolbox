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
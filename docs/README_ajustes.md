# ajustes.py

## Propósito
Ajuste de curvas por mínimos cuadrados ponderados (WLS) con cálculo de covarianzas para propagación de incertidumbres. Está diseñado para trabajar con incertidumbres experimentales conocidas en `y`.

## Supuestos
- Los errores `sy` son incertidumbres absolutas conocidas en `y`.
- Residuos independientes y gaussianos.
- Se usa `absolute_sigma=True` (no reescala errores automáticamente).

## API principal
- `ajuste_lineal(x, y, sy=None)`
  - Ajuste analítico de `y = a + b·x`.
  - Devuelve parámetros, errores, covarianza, `yfit`, `chi2`, `ndof`, `chi2_red`, `p`.

- `ajuste_polinomico(x, y, grado, sy=None)`
  - Ajuste polinómico WLS (coeficientes en orden descendente).
  - Devuelve coeficientes, errores, covarianza y métricas chi‑cuadrado.

- `ajuste(modelo, x, y, sy=None, p0=None, variable="x")`
  - Ajuste genérico con `callable` o `sympy.Expr`.
  - Si `sympy.Expr`, lambdifica y ordena parámetros simbólicos.

- `intervalo_confianza_parametros(resultado_ajuste, nivel=0.95)`
  - IC de parámetros usando t‑Student o normal según `ndof`.

- `incertidumbre_prediccion(resultado_ajuste, modelo, x0)`
  - Banda de confianza del modelo (solo incertidumbre de parámetros).

## Errores típicos
- Longitudes incompatibles entre `x`, `y`, `sy`.
- `sy` con valores no positivos.
- Modelo simbólico sin variable válida.

## Flujo recomendado
1. Ajustar (`ajuste_lineal`, `ajuste_polinomico` o `ajuste`).
2. Revisar `chi2_red` y `p`.
3. Calcular IC de parámetros.
4. Calcular incertidumbre de predicción.

## Salida
Los resultados siempre se devuelven como diccionarios con claves estables para facilitar su uso posterior.
# incertidumbres.py

## Propósito
Creación de magnitudes con incertidumbre y propagación simbólica con SymPy.

## API principal
- `u(x, sigmax=0.0)`
  - Devuelve `ufloat` si `x` es escalar.
  - Devuelve `unumpy.uarray` si `x` es array.

- `propagacion_incertidumbre_sympy(f, vars_, valores, sigmas, cov=None, simplify=True)`
  - Calcula gradiente, varianza propagada y sigma.
  - Devuelve valores numéricos y LaTeX de las expresiones.

## Dependencias
- `uncertainties`
- `sympy`

## Notas
- Integra con `latex_tools` para generar LaTeX.
- `cov` opcional permite covarianzas completas.

## Errores típicos
- `sigmas` con valores negativos.
- `cov` con dimensión inválida.
- Variables faltantes en `valores` o `sigmas`.
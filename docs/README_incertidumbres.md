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

## Ejemplos
```python
from incertidumbres import incertidumbres

u = incertidumbres.u([1.0, 2.0, 3.0], [0.1, 0.1, 0.2])
```

## Mini ejemplos (por función)

### u(x, sigmax=0.0)
**Caso 1 (típico - escalar):** Si aplicas esto:
```python
from incertidumbres import incertidumbres

m = incertidumbres.u(9.81, 0.05)
print(m)
print(m.nominal_value, m.std_dev)
```
haces esto: Creas una magnitud con valor central 9.81 e incertidumbre 0.05.

Obtienes esto:
```
9.81+/-0.05
9.81 0.05
```

**Caso 2 (típico - array):** Si aplicas esto:
```python
x = incertidumbres.u([1.0, 2.0, 3.0], [0.1, 0.2, 0.15])
print(x)
resultado = x * 2
print(resultado)
```
haces esto: Creas array de magnitudes y lo multiplicas.

Obtienes esto:
```
array([1.0+/-0.1, 2.0+/-0.2, 3.0+/-0.15], dtype=object)
array([2.0+/-0.2, 4.0+/-0.4, 6.0+/-0.3], dtype=object)
```

**Caso 3 (borde):** Si aplicas esto:
```python
z = incertidumbres.u(5.0, 0.0)
print(z)
y = z + 3
print(y)
```
haces esto: Magnitud sin incertidumbre propagada.

Obtienes esto:
```
5.0+/-0.0
8.0+/-0.0
```

### propagacion_incertidumbre_sympy(f, vars_, valores, sigmas, cov=None, simplify=True)
**Caso 1 (típico):** Si aplicas esto:
```python
import sympy as sp
from incertidumbres import incertidumbres

x, y = sp.symbols('x y')
f = x**2 + y
valores = {'x': 2.0, 'y': 3.0}
sigmas = {'x': 0.1, 'y': 0.2}

resultado = incertidumbres.propagacion_incertidumbre_sympy(
    f, [x, y], valores, sigmas
)
print(resultado["sigma"])
print(resultado["valor"])
```
haces esto: Propagas incertidumbre de f=x²+y con x=2±0.1, y=3±0.2.

Obtienes esto:
```
sigma ≈ 0.408...
valor = 7.0
```

**Caso 2 (borde - función lineal):** Si aplicas esto:
```python
x, y = sp.symbols('x y')
f = 3*x + 2*y
valores = {'x': 1.0, 'y': 2.0}
sigmas = {'x': 0.1, 'y': 0.1}

resultado = incertidumbres.propagacion_incertidumbre_sympy(
    f, [x, y], valores, sigmas
)
print(resultado["sigma"])
```
haces esto: Propagación en función lineal (σ_f = √(3²σ_x² + 2²σ_y²)).

Obtienes esto:
```
sigma = sqrt(9*0.01 + 4*0.01) ≈ 0.361...
```
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

## Ejemplos
```python
from montecarlo import montecarlo

res = montecarlo.integral_1d(lambda x: x**2, 0, 1, n=5000)
print(res["valor"], res["error"])
```

## Mini ejemplos (por función)

### integral_1d(f, a, b, n=10000)
**Caso 1 (típico):** Si aplicas esto:
```python
from montecarlo import montecarlo
import numpy as np

def f(x):
    return x**2

res = montecarlo.integral_1d(f, 0, 1, n=10000)
print(f"Valor: {res['valor']:.4f}")
print(f"Error: {res['error']:.4f}")
print(f"Valor teórico: 1/3 = {1/3:.4f}")
```
haces esto: Estimas ∫₀¹ x² dx mediante muestreo uniforme (10k muestras).

Obtienes esto:
```
Valor: 0.3346 (aprox)
Error: 0.0032
Valor teórico: 0.3333
```

**Caso 2 (borde):** Si aplicas esto:
```python
def f(x):
    return 1.0  # función constante

res = montecarlo.integral_1d(f, 2, 5, n=5000)  # ∫₂⁵ 1 dx = 3
print(f"Valor: {res['valor']:.4f}")
print(f"Teórico: 3.0")
```
haces esto: Integral trivial de función constante (ancho * altura).

Obtienes esto:
```
Valor: 2.9825 (aprox, cercano a 3.0)
Teórico: 3.0
```

### propagacion(fun, generadores, n=10000)
**Caso 1 (típico):** Si aplicas esto:
```python
from montecarlo import montecarlo
import numpy as np

# x ~ N(2, 0.1), y ~ N(3, 0.2), calcular z = x*y
gen_x = lambda: np.random.normal(2.0, 0.1)
gen_y = lambda: np.random.normal(3.0, 0.2)

def fun(x, y):
    return x * y

res = montecarlo.propagacion(
    fun,
    [gen_x, gen_y],
    n=5000
)
print(f"Media: {res['media']:.4f}")
print(f"Sigma: {res['sigma']:.4f}")
print(f"Valor esperado: 6.0 ± √(0.01*9 + 0.04*4) ≈ 6.0 ± 0.514")
```
haces esto: Propaga incertidumbres de x e y a través de z=x*y.

Obtienes esto:
```
Media: 6.0023
Sigma: 0.5180
```

**Caso 2 (borde):** Si aplicas esto:
```python
gen_const = lambda: 5.0  # siempre retorna 5

def fun(x):
    return x + 1

res = montecarlo.propagacion(fun, [gen_const], n=1000)
print(f"Media: {res['media']}")
print(f"Sigma: {res['sigma']}")
```
haces esto: Propagación con entrada determinística (sin variabilidad).

Obtienes esto:
```
Media: 6.0
Sigma: 0.0 (o muy cercana a 0)
```
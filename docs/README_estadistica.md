# estadistica.py

## Purpose
Descriptive statistics, confidence intervals, and hypothesis tests with a simple API.

## Descriptive statistics
- `media(x)`
- `varianza(x, ddof=1)`
- `desviacion_tipica(x, ddof=1)`
- `error_estandar(x)`

## Weighted statistics
- `media_ponderada(x, w=None, sigma=None)`
- `varianza_ponderada(x, w=None, sigma=None, ddof=1, tipo="frecuentista")`

## Confidence intervals
- `intervalo_confianza(x, nivel=0.95, distribucion="normal", sigma=None)`
  - Normal, Poisson, or Binomial depending on `distribucion`.

## Hypothesis tests
- `test_media(x, mu0, alternativa="dos_colas", distribucion="normal", sigma=None)`
  - Normal: z‑test if `sigma` is known, t‑test otherwise.
  - Poisson: exact test for the rate.
  - Binomial: exact test for the probability.

- `test_ks(x, distribucion="normal")`
  - KS goodness‑of‑fit for normal or uniform.

## Output
Methods return dictionaries with statistics, p‑values, and metadata (`n`, `df`, etc.).

## Typical errors
- Empty samples or invalid sizes.
- Parameters out of range (e.g. `sigma <= 0`, `nivel` outside (0,1)).
- Unsupported distributions.

## Examples
```python
import numpy as np
from estadistica import estadistica

x = np.random.default_rng(0).normal(0, 1, 30)
media = estadistica.media(x)
ic = estadistica.intervalo_confianza(x, nivel=0.95, distribucion="normal")
test = estadistica.test_media(x, mu0=0.0, distribucion="normal")
```

## Mini examples (per function)

### media(x)
**Case 1 (typical):** If you do this:
```python
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
media = estadistica.media(x)
```
You do this: Calculates the average of 5 values.

You get this:
```
3.0
```

**Case 2 (edge):** If you do this:
```python
x = np.array([0.0])
media = estadistica.media(x)
```
You do this: Calculates the mean of a single value.

You get this:
```
0.0
```

### varianza(x, ddof=1)
**Case 1 (typical):** If you do this:
```python
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
varianza = estadistica.varianza(x, ddof=1)
```
You do this: Calculates the sample variance (n-1).

You get this:
```
2.5
```

**Case 2 (edge):** If you do this:
```python
x = np.array([5.0, 5.0, 5.0])
varianza = estadistica.varianza(x)
```
You do this: Calculates variance when all values are identical.

You get this:
```
0.0
```

### desviacion_tipica(x, ddof=1)
**Case 1 (typical):** If you do this:
```python
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
std = estadistica.desviacion_tipica(x)
```
You do this: Calculates the standard deviation.

You get this:
```
1.5811388...
```

**Case 2 (edge):** If you do this:
```python
x = np.array([0.0, 0.0])
std = estadistica.desviacion_tipica(x, ddof=0)
```
You do this: Population standard deviation.

You get this:
```
0.0
```

### error_estandar(x)
**Case 1 (typical):** If you do this:
```python
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
se = estadistica.error_estandar(x)
```
You do this: Calculates standard error (σ/√n).

You get this:
```
0.7071067...
```

**Case 2 (edge):** If you do this:
```python
x = np.array([10.0])
se = estadistica.error_estandar(x)
```
You do this: Standard error with a single sample (σ/√1).

You get this:
```
nan
```

### media_ponderada(x, w=None, sigma=None)
**Case 1 (typical):** If you do this:
```python
x = np.array([1.0, 2.0, 3.0])
w = np.array([1.0, 2.0, 1.0])
mean_w = estadistica.media_ponderada(x, w=w)
```
You do this: Calculates weighted mean with weights [1, 2, 1].

You get this:
```
2.0
```

**Case 2 (edge):** If you do this:
```python
x = np.array([5.0, 10.0])
sigma = np.array([1.0, 0.1])
mean_w = estadistica.media_ponderada(x, sigma=sigma)
```
You do this: Uses inverse‑variance weights to minimize error.

You get this:
```
9.917355...
```

### intervalo_confianza(x, nivel=0.95, distribucion="normal", sigma=None)
**Case 1 (typical):** If you do this:
```python
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
ic = estadistica.intervalo_confianza(x, nivel=0.95)
```
You do this: Computes a 95% CI with a normal distribution.

You get this:
```
{'media': 3.0, 'lower': 0.627..., 'upper': 5.372..., 'error': 2.372..., 'n': 5, 'sigma': 1.581...}
```

**Case 2 (edge):** If you do this:
```python
x = np.array([2.0, 2.0, 2.0, 2.0])
ic = estadistica.intervalo_confianza(x, nivel=0.99)
```
You do this: CI with zero variance and high confidence.

You get this:
```
{'media': 2.0, 'lower': 2.0, 'upper': 2.0, 'error': 0.0, ...}
```

### test_media(x, mu0, alternativa="dos_colas", distribucion="normal", sigma=None)
**Case 1 (typical):** If you do this:
```python
x = np.array([1.1, 1.9, 2.1, 1.8, 2.0])
test = estadistica.test_media(x, mu0=2.0, alternativa="dos_colas")
```
You do this: Two‑sided t‑test against μ₀=2.0.

You get this:
```
{'media': 1.78, 'sigma_x': 0.450..., 't': -1.094..., 'p_valor': 0.346..., 'rechaza': False}
```

**Case 2 (edge):** If you do this:
```python
x = np.array([5.0, 5.0, 5.0])
test = estadistica.test_media(x, mu0=5.0)
```
You do this: Test against a value equal to the data.

You get this:
```
{'media': 5.0, 'sigma_x': 0.0, 't': 'inf' or 'nan', 'p_valor': 1.0 or nan, 'rechaza': False}
```

### test_ks(x, distribucion="normal")
**Case 1 (typical):** If you do this:
```python
rng = np.random.default_rng(42)
x = rng.normal(0, 1, 50)
test = estadistica.test_ks(x, distribucion="normal")
```
You do this: Kolmogorov‑Smirnov test on a normal sample.

You get this:
```
{'estadistico': 0.089..., 'p_valor': 0.689..., 'rechaza': False}
```

**Case 2 (edge):** If you do this:
```python
x = np.linspace(0, 100, 100)
test = estadistica.test_ks(x, distribucion="uniforme")
```
You do this: KS test for a uniform distribution.

You get this:
```
{'estadistico': ~0.0, 'p_valor': > 0.9, 'rechaza': False}
```
# Migration Guide (V1 -> V2)

This guide summarizes how to migrate from legacy V1 code to the current V2 API,
with the smallest possible changes.

## Compatibility Guarantee

V1 imports are still supported through compatibility aliases:

- `marhare.uncertainties`
- `marhare.quantities2`
- `marhare.propagation`
- `marhare.latex_tools`

This means existing V1 scripts can continue working while you migrate gradually.

## Recommended V2 Defaults

Use direct Quantity operations and concise helpers:

1. Direct operations:
   - `resistance = voltage / current`
2. Built-in data cleaning:
   - `quantity(series, sigma, unit)` uses `nan_policy="drop"` by default
3. Weighted summary shortcut:
   - `summary = q.weighted()`

## API Mapping

### Imports

V1:

```python
from marhare.uncertainties import quantity, register, evaluate_quantity
```

V2:

```python
from marhare.quantities import quantity
from marhare import evaluate_quantity
```

### Formula Construction

V1 style (legacy, still allowed):

```python
R = quantity("V/I", "ohm", symbol="R")
registry = register(V, I, R)
R_result = evaluate_quantity(R, registry)
```

V2 style (recommended):

```python
resistance = voltage / current
```

### Weighted Aggregation

V1 verbose:

```python
mean_val = weighted_mean(q.value, sigma=q.sigma)
mean_sig = weighted_standard_error(q.value, sigma=q.sigma)
q_mean = quantity(mean_val, mean_sig, q.unit)
```

V2 concise:

```python
q_mean = q.weighted()
```

## Practical Migration Steps

1. Keep old imports first, verify unchanged behavior.
2. Replace formula strings with direct operations.
3. Replace manual weighted summary blocks with `q.weighted()`.
4. Switch imports to `marhare.quantities` / `marhare.latex`.
5. Remove remaining legacy imports when your tests pass.

## Notes

- For row-aligned lab tables, use `nan_policy="keep"` during coupled operations,
  then apply `drop` before final statistics.
- For strict data validation, set `numeric_errors="raise"`.
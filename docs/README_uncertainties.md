# Quantities and Uncertainty Propagation

Navigation: [Documentation Index](INDEX.md) | [Main README](../README.md)

This guide documents the computation layer exposed by marhare.quantities.

## Recommended Imports

```python
from marhare.quantities import (
    Quantity,
    quantity,
    weighted_quantity,
)
```

You can still use marhare.uncertainties for legacy code, but it is deprecated.

## Installation

From repository root:

```bash
python -m pip install -e .
```

From any local folder path:

```bash
python -m pip install "C:/path/to/python_toolbox_v2"
python -m pip install -e "C:/path/to/python_toolbox_v2"
```

## Preferred Method (Direct Quantity Math)

The recommended workflow is direct operations between Quantity objects.

```python
from marhare.quantities import quantity

voltage = quantity(10.0, 0.5, "V", symbol="U")
current = quantity(2.0, 0.1, "A", symbol="I")

resistance = voltage / current
print(f"R = {float(resistance.value):.3f} +/- {float(resistance.sigma):.3f} {resistance.unit}")
```

This is now the priority path for regular use.

## Legacy Method (Obsolete but Compatible)

`register()` + `propagate_quantity()` still exist for compatibility, but V2 documentation and examples prioritize direct computation (`resistance = voltage / current`).

## What quantity() creates

quantity() returns a Quantity object with dict-like access compatibility.

```python
q = quantity(5.0, 0.1, "V", symbol="V")

print(q.value)          # 5.0
print(q.sigma)          # 0.1
print(q.unit)           # display unit
print(q["measure"])     # (value, sigma)
print(q["symbol"])      # "V"
```

LaTeX fields from a Quantity:

```python
# Preferred public properties
print(q.expr_latex)
print(q.sigma_latex)

# Dict export helper
tex = q.latex()
print(tex["expr_latex"])
print(tex["sigma_latex"])

# Mapping compatibility API
print(q["expr_latex"])
print(q["sigma_latex"])
```

Direct operations also keep symbolic LaTeX traces:

```python
f = (s2 * s2_im) / (s2 - s2_im)
print(f.expr_latex)
print(f.sigma_latex)
```

Behavior note:

- For base quantities (for example `s2`), `sigma_latex` is the symbol of its own uncertainty (for example `\\sigma_{s2}`).
- For composed expressions (for example `f = (s2*s2_im)/(s2-s2_im)`), `sigma_latex` is the propagated uncertainty formula.

Uncertainty budget (dominant sources) can be obtained directly:

```python
budget = f.uncertainty_budget()
print(budget["contributions"])          # absolute contributions to variance
print(budget["relative_contributions"]) # percentages (0-100)
print(budget.get("sigma_consistency"))  # optional diagnostic vs f.sigma

dominant = f.dominant_uncertainty()
print(dominant["name"], dominant["percentage"])
```

## Creation Patterns

```python
# value, unit (sigma defaults to 0)
q1 = quantity(5.0, "V", symbol="V")

# value, sigma, unit
q2 = quantity(5.0, 0.1, "V", symbol="V")
```

## Automatic Numeric Coercion and NaN Cleaning

`quantity(...)` can ingest pandas Series directly and coerce non-numeric values
to `NaN` internally (coerce-like behavior).

- Default: `numeric_errors="coerce"`
- Cleaning strategy: `nan_policy` in `{"keep", "drop", "raise"}`

Practical recommendation for lab tables:

```python
q = quantity(df["measured_column"], 0.1, "um", nan_policy="drop")
```

This removes non-numeric/NaN/inf rows automatically in that quantity.
Use strict mode when needed:

```python
q = quantity(df["measured_column"], 0.1, "um", nan_policy="raise", numeric_errors="raise")
```

## Unit Normalization

By default normalize=True, so values are converted to SI internally.

```python
V = quantity(5000.0, 100.0, "mV", symbol="V")
print(V["measure"])  # approximately (5.0, 0.1)

V_raw = quantity(5000.0, 100.0, "mV", symbol="V", normalize=False)
print(V_raw["measure"])  # (5000.0, 100.0)
```

`nan_policy="drop"` is the default behavior, so this is equivalent:

```python
q = quantity(df["measured_column"], 0.1, "um")
```

## Unit Conversion as Quantity Methods

You can convert a quantity directly without rebuilding it manually:

```python
from marhare.quantities import quantity

lam = quantity(6.27e-7, 1.0e-8, "m", symbol="lambda")

# Preferred short form
lam_nm = lam.to("nm")

# Explicit alias (same behavior)
lam_nm_2 = lam.to_unit("nm")

print(lam_nm.value, lam_nm.sigma, lam_nm.unit)
```

Notes:

- Both value and sigma are converted with the same factor.
- Use `normalize=False` (default in `to`/`to_unit`) to preserve the requested display unit.
- Unit string `"1"` means dimensionless.

## Weighted Aggregation Shortcut

If you repeatedly do:

1. `weighted_mean(q.value, q.sigma)`
2. `weighted_standard_error(q.value, sigma=q.sigma)`
3. `quantity(mean, error, q.unit)`

you can replace it with:

```python
A2 = A.weighted()
d = ro.weighted()
```

or the helper function:

```python
from marhare.quantities import weighted_quantity

l = weighted_quantity(lambd)
```

## Full Workflow Example

```python
from marhare.quantities import quantity
from marhare.latex import latex_quantity

voltage = quantity(12.0, 0.3, "V", symbol="U")
current = quantity(2.4, 0.1, "A", symbol="I")

power_result = voltage * current

power_summary = power_result.weighted()
print(f"P values: {power_result.value}")
print(f"P sigmas: {power_result.sigma}")
print(latex_quantity(power_summary, cifras=2))
```

## Real Lab Experiment with Dataset

For realistic laboratory workflows with multiple measurements, use the Dataset class to organize measurements and perform batch operations:

```python
from marhare import Dataset, quantity, latex_quantity
import numpy as np

# Organize resistor characterization data across multiple trials
experiment = Dataset(
    {
        "trial": np.array([1, 2, 3, 4, 5]),
        "voltage": quantity(
            np.array([5.0, 10.0, 7.5, 12.0, 9.0]),
            np.array([0.1, 0.2, 0.15, 0.2, 0.1]), 
            "V", 
            symbol="U"
        ),
        "current": quantity(
            np.array([0.50, 1.00, 0.75, 1.20, 0.90]),
            np.array([0.05, 0.05, 0.05, 0.05, 0.05]), 
            "A", 
            symbol="I"
        ),
    },
    name="Resistor_Characterization"
)

# Direct computation on entire dataset columns
R_values = experiment["voltage"] / experiment["current"]

# Direct access and weighted summary
print(f"Resistor values: {R_values.value}")
print(f"Uncertainties: {R_values.sigma}")
R_summary = R_values.weighted()

# Format for reporting
print(latex_quantity(R_summary, cifras=2))
```

This approach keeps your measurements organized and makes batch uncertainty propagation straightforward.

## Filtering Tables Before Quantities (Pandas)

It is common to split measurements before creating quantities. You can filter by number, by text, or by grouped values:

```python
import pandas as pd
from marhare import quantity

df = pd.read_excel("young.ods", sheet_name="D")

# 1) Numeric filter
df_cam_10 = df[df["posición cámara"] == 10]

# 2) Text filter (word match)
df_control = df[df["etiqueta"].astype(str).str.contains("control", case=False, na=False)]

# 3) Group filter pattern
for cam_pos, g in df.groupby("posición cámara", dropna=True):
  q = quantity(g["interfranxas (px)"], 1, "1", nan_policy="keep")
  print(cam_pos, q.weighted(symbol="n"))
```

This pattern is useful when the same column has repeated category values (for example, only 3 camera positions repeated across rows).

## Unit Syntax and Compatibility

The quantities parser supports common scientific unit forms and aliases.

Supported writing styles (examples):

- Micro prefix:
  - Preferred: `uA`, `uV`, `um`
  - Also accepted: `µA`, `µV`, `µm`, `microampere`
- Resistance:
  - Preferred: `ohm`
  - Also accepted: `Ω`
- Angles:
  - Radians: `rad`, `radian`, `radians`
  - Degrees: `deg`, `degree`, `degrees`
- Acceleration:
  - Preferred: `m/s^2`
  - Also accepted: `m/s²`, `m*s^-2`
- Common derived units:
  - `N`, `Pa`, `J`, `W`, `Hz`, `GHz`, `mV`, `V`

Angle example:

```python
import numpy as np
from marhare.quantities import quantity

theta_deg = quantity(180.0, 1.0, "degree")
theta_rad = quantity(np.pi, 0.01, "radian")

print(theta_deg.value)  # ~pi (normalized)
print(np.sin(theta_deg).value)  # ~0.0, dimensionless
print(np.sin(theta_rad).value)  # ~0.0, dimensionless
```

Behavior guarantees in operations:

- Addition/subtraction checks dimensional compatibility.
  - Example: `quantity(..., "m") + quantity(..., "s")` raises `ValueError`.
- Scalar multiplication/division preserves units.
  - Example: `5 * quantity(..., "m")` remains in meters.
- Derived operation units are simplified when possible.
  - Example: `quantity(..., "V") / quantity(..., "A")` yields `ohm`.

Normalization note:

- With `normalize=True` (default), internal units are SI-normalized.
- `q.unit_raw` keeps what the user wrote.
- `q.unit_internal` stores normalized/derived operation units.

## Common Errors

- Shape mismatch:
  - value and sigma must be broadcast-compatible.
- Negative sigma:
  - sigma must be non-negative.
- Trying grouped quantities:
  - groups are not supported in the current quantities2-first architecture.

## Compatibility Notes

- Legacy module marhare.uncertainties re-exports the quantities API and emits DeprecationWarning.
- New code should target marhare.quantities and marhare.latex directly.

## Legacy Compatibility (Allowed)

The following legacy forms are still supported and documented as allowed:

- `register(...)` + `propagate_quantity(...)`
- `value_quantity(q)` tuple extraction
- import path `marhare.uncertainties`

Use these only when maintaining old notebooks or scripts.
For new code, prefer direct operations, default `nan_policy="drop"`, and `q.weighted()`.

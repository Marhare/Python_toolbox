# Python Toolbox (marhare)

Scientific Python toolkit for experimental physics data analysis.

## Installation

```bash
pip install -e .
```

Install from any local folder path:

```bash
python -m pip install "C:/path/to/python_toolbox_v2"
```

Editable install from any local folder path (recommended for development):

```bash
python -m pip install -e "C:/path/to/python_toolbox_v2"
```

## Current Architecture

The library is now organized by layers:

- Computation layer: marhare.quantities
- Presentation layer: marhare.latex
- Backward compatibility: marhare.uncertainties (deprecated facade)

No circular dependency between layers: latex depends on quantities, never the opposite.

V2 policy: direct computation with Quantity objects is the default style.

## Quick Start

### Recommended modular imports (direct method)

```python
from marhare.quantities import quantity
from marhare.latex import valor_pm

voltage = quantity(10.0, 0.5, "V", symbol="U")
current = quantity(2.0, 0.1, "A", symbol="I")
resistance = voltage / current

print(f"R = {float(resistance.value):.2f} +/- {float(resistance.sigma):.2f} {resistance.unit}")
print(valor_pm(resistance, cifras=2))
```

### Optional root import style

```python
import marhare as mh

voltage = mh.quantity(10.0, 0.5, "V", symbol="U")
current = mh.quantity(2.0, 0.1, "A", symbol="I")
resistance = voltage / current
print(resistance.value, resistance.sigma)
print(mh.latex_quantity(resistance, cifras=2))
```

### Dataset for lab experiments

```python
import marhare as mh
import numpy as np

# Organize multiple measurements in a Dataset
lab_data = mh.Dataset(
    {
        "measurement": np.array([1, 2, 3, 4]),
        "voltage": mh.quantity(np.array([10.0, 12.0, 9.5, 11.0]), np.array([0.5, 0.5, 0.5, 0.5]), "V", symbol="U"),
        "current": mh.quantity(np.array([2.0, 2.4, 1.9, 2.2]), np.array([0.1, 0.1, 0.1, 0.1]), "A", symbol="I"),
    },
    name="Resistor_Test"
)

# Compute directly on dataset columns
R = lab_data["voltage"] / lab_data["current"]
R_summary = R.weighted()
print(f"R values: {R.value}")
print(f"Uncertainties: {R.sigma}")
print(f"Weighted summary: {mh.latex_quantity(R_summary, cifras=2)}")
```

## Unit Writing Guide

Unit algebra has been hardened for common workflows:

- Addition/subtraction now validates dimensional compatibility (for example, `m + s` raises an error).
- Scalar operations preserve units (for example, `5 * quantity(..., "m")` stays in meters).
- Derived units are simplified to common names when possible (for example, `voltage / current` becomes `ohm`).

Recommended unit strings and accepted aliases:

- Micro prefix:
    - Preferred: `uA`, `uV`, `um`
    - Also accepted: `µA`, `µV`, `µm`, `microampere`
- Resistance:
    - Preferred: `ohm`
    - Also accepted: `Ω`
- Angles:
    - Radians: `rad`, `radian`, `radians`
    - Degrees: `deg`, `degree`, `degrees`
- Accelerations and powers:
    - Preferred: `m/s^2`
    - Also accepted: `m/s²`, `m*s^-2`
- Compound units:
    - `N`, `Pa`, `J`, `W`, `Hz`, `GHz`, `mV`, `V` are accepted.

Notes about normalization and display:

- With default `normalize=True`, values are converted to SI internally.
- Use `q.unit_raw` to inspect the original input string.
- Use `q.unit_internal` (or `q.unit`) to inspect the normalized/operation result unit.

LaTeX fields access on `Quantity`:

- Preferred properties: `q.expr_latex`, `q.sigma_latex`
- Dict-style LaTeX export: `q.latex()["expr_latex"]`, `q.latex()["sigma_latex"]`
- Mapping compatibility: `q["expr_latex"]`, `q["sigma_latex"]`
- Safe mapping read: `q.get("expr_latex")`, `q.get("sigma_latex")`

Data cleaning shortcut in `quantity(...)`:

- `quantity(...)` now performs numeric coercion internally (similar to `pd.to_numeric(..., errors="coerce")`).
- With `nan_policy="drop"`, non-numeric, `NaN`, and `inf` entries are dropped automatically.
- Use `numeric_errors="raise"` for strict mode.

```python
import marhare as mh

# Pandas Series can be passed directly.
q = mh.quantity(df["column"], 0.1, "um", nan_policy="drop")

# Strict conversion instead of coercion:
q_strict = mh.quantity(df["column"], 0.1, "um", nan_policy="raise", numeric_errors="raise")
```

Filtering before building quantities (number, word, and groups):

```python
import pandas as pd
import marhare as mh

df = pd.read_excel("young.ods", sheet_name="D")

# 1) Filter by numeric value
df_num = df[df["posición cámara"] == 10]

# 2) Filter by word/text (case-insensitive)
df_text = df[df["etiqueta"].astype(str).str.contains("control", case=False, na=False)]

# 3) Group by camera position (one subset per unique value)
for cam_pos, g in df.groupby("posición cámara", dropna=True):
    fringe = mh.quantity(g["interfranxas (px)"], 1, "1", nan_policy="keep")
    print(cam_pos, len(g), fringe.weighted(symbol="n"))
```

Weighted aggregation shortcut:

- `q.weighted()` returns a scalar `Quantity` with weighted mean and weighted standard error.
- `mh.weighted_quantity(q)` provides the same behavior as a helper function.

```python
import marhare as mh

A2 = A.weighted()
d = ro.weighted()
l = mh.weighted_quantity(lambd)
```

Quantity unit conversion shortcut:

- Use `q.to("nm")` (or explicit alias `q.to_unit("nm")`) to convert value and sigma together.
- `"1"` means dimensionless.

```python
import marhare as mh

lam = mh.quantity(6.27e-7, 1.0e-8, "m", symbol="lambda")
lam_nm = lam.to_unit("nm")
print(mh.latex_quantity(lam_nm))
```

```python
import marhare as mh

# Micro aliases
i1 = mh.quantity(120.0, 1.0, "uA")
i2 = mh.quantity(120.0, 1.0, "µA")

# Derived unit simplification
voltage = mh.quantity(10.0, 0.5, "V")
current = mh.quantity(2.0, 0.1, "A")
R = voltage / current
print(R.unit)  # ohm

# Dimensional validation
try:
        bad = mh.quantity(5.0, 0.1, "m") + mh.quantity(2.0, 0.1, "s")
except ValueError as e:
        print(e)
```

## Core Modules

- quantities: Quantity object, symbolic propagation, unit normalization
- latex: LaTeX formatting helpers for scalar/vector/quantity-like outputs
- dataset: aligned tabular scientific data
- fitting: curve fitting and result wrappers
- statistics: descriptive and inferential statistics
- functions: symbolic/numeric helper functions

## Documentation

- docs/INDEX.md: full documentation index
- docs/MIGRATION_GUIDE.md: V1 to V2 migration and compatibility map
- docs/README_uncertainties.md: quantities module guide
- docs/README_latex_tools.md: latex layer guide
- docs/README_statistics.md: statistics guide
- docs/README_fitting.md: fitting guide
- docs/UNIT_CONVERSION_IMPLEMENTATION.md: unit conversion details

## Backward Compatibility

The following legacy paths still import correctly, but are deprecated:

- marhare.uncertainties
- marhare.quantities2
- marhare.propagation
- marhare.units
- marhare.latex_tools

New code should import from marhare.quantities and marhare.latex.

Legacy API forms still allowed:

- `register(...)` and `propagate_quantity(...)` remain available for prior workflows.
- `value_quantity(q)` remains available for tuple extraction.
- Legacy import paths (`marhare.uncertainties`, `marhare.quantities2`, `marhare.propagation`, `marhare.latex_tools`) remain functional via compatibility aliases.

Prefer the current defaults in new code:

- direct quantity operations (`q1 + q2`, `q1 * q2`, `q1 / q2`)
- `quantity(..., nan_policy="drop")` for cleaning tabular numeric input
- `q.weighted()` (or `weighted_quantity(q)`) for weighted summary quantities

## Testing

```bash
python tests/test_readme_examples_v2.py
python tests/test_v2_direct_workflow.py
python tests/test_readme_examples.py
python tests/test_deep_uncertainties.py
```

## License

MIT License.

# Unit Conversion System - Implementation Notes

## Overview

The marhare package now includes automatic unit conversion using `pint` as backend. This feature is **completely transparent** to the user - no API changes required.

## How It Works

### 1. **User Input**
User specifies units naturally with SI prefixes:
```python
V = mh.quantity(5000, 100, "mV", symbol="V")  # millivolts
f = mh.quantity(2.4, 0.05, "GHz", symbol="f")  # gigahertz
```

### 2. **Internal Normalization**
When `quantity()` is called with `normalize=True` (default):
1. Parse unit string with pint (`"mV"` → millivolt)
2. Convert value to SI base units (`5000 mV` → `5.0 V`)
3. Convert sigma by same factor (`100 mV` → `0.1 V`)
4. Store SI-normalized values in `measure` tuple
5. Store SI base unit in `unit` field
6. Also store SI-normalized values in hidden `measure_si` field

When called with `normalize=False`:
1. Parse unit string with pint
2. DO NOT convert—keep original values
3. Store original values in `measure` tuple
4. Store original unit in `unit` field
5. Store SI-normalized values in hidden `measure_si` field (for internal use only)

### 3. **Calculations**
All propagation uses `measure_si` (SI-normalized values):
- Even when user created quantity with `normalize=False`, calculations use SI
- No conversion needed during calculations
- All values are already in SI base units
- Partial derivatives computed correctly for dimensional analysis

### 4. **Display**
LaTeX and graphics use `unit` and `measure` directly:
```python
# With normalize=True (default)
tex = mh.latex_quantity(V)
# Output: $V = 5.0 \pm 0.1 \, \mathrm{V}$
# Shows SI units

# With normalize=False  
tex = mh.latex_quantity(V)  
# Output: $V = 5000 \pm 100 \, \mathrm{mV}$
# Shows original units
```

## Key Design Decisions

### Backward Compatibility
- If `pint` not installed: falls back to string-only units (no conversion)
- Existing code works unchanged
- New `unit_display` field is optional (defaults to `unit` if missing)

### Quantity Dictionary Structure

**New Architecture (Simplified):**
```python
{
    'symbol': 'V',
    'unit': 'V',                  # Display unit (SI SYMBOL if normalize=True, original if False)
    'measure': (5.0, 0.1),        # Display values (SI if normalize=True, original if False)
    'measure_si': (5.0, 0.1),     # ALWAYS in SI base units (for internal calculations)
    'expr': None,
    'result': None,
    'dimension': None
}
```

**Key difference from old architecture:**
- No `unit_display` field—simplified to single `unit` field
- `unit` always contains what should be displayed:
  - If `normalize=True`: SI SYMBOLS (V, A, Hz, m, etc.)
  - If `normalize=False`: original user unit (mV, mA, mm, etc.)
- `measure` always contains displayed values (SI if `normalize=True`, original if `normalize=False`)
- `measure_si` is ALWAYS in SI base units, used internally for all calculations 

**Example: normalize=True (default)**
```python
V = mh.quantity(5000.0, 100.0, "mV", symbol="V", normalize=True)
# Result:
{
    'symbol': 'V',
    'unit': 'V',                   # SI SYMBOL, not "volt"!
    'measure': (5.0, 0.1),         # Display: normalized to volt
    'measure_si': (5.0, 0.1),      # Calculation: volt (same as display)
}
```

**Example: normalize=False**
```python
V = mh.quantity(5000.0, 100.0, "mV", symbol="V", normalize=False)
# Result:
{
    'symbol': 'V',
    'unit': 'mV',                  # Display: original unit
    'measure': (5000, 100),        # Display: original values
    'measure_si': (5.0, 0.1),      # Calculation: SI base (volt)
}
```

### Normalization Strategy
- **Default (normalize=True)**: All values → SI base units (meter, kilogram, second, ampere, kelvin, mole, candela)
- **Optional (normalize=False)**: Keep original units unchanged
- Derived units reduced: `N → kg·m/s²`, `J → kg·m²/s²`
- Compound units preserved: `W/(m·K)` → `watt / meter / kelvin`

### Controlling Normalization

Users can control what units are displayed:

```python
# Default: normalize to SI base (display in SI)
x = mh.quantity(10.0, 0.5, "cm", symbol="x")
# Internal: measure_si = (0.1, 0.005) meter
# Display: measure = (0.1, 0.005), unit = "meter"
# LaTeX shows: x = 0.1 ± 0.005 m

# Keep original units (display as entered)
x = mh.quantity(10.0, 0.5, "cm", symbol="x", normalize=False)
# Internal: measure_si = (0.1, 0.005) meter  
# Display: measure = (10, 0.5), unit = "cm"
# LaTeX shows: x = 10 ± 0.5 cm
```

**When to use `normalize=False`:**
- Want output in original units despite SI base for calculations
- Working in specific unit systems (CGS, laboratory units)
- User preference for display (e.g., always show cm, not m)

**When to use default (`normalize=True`):**
- Standard scientific output (SI units)
- Mixing different prefixes (mV + V + kV)
- General scientific calculations

## Benefits

1. **Mixed Units Work Automatically**
   ```python
   V = mh.quantity(5000, 100, "mV")   # millivolts
   I = mh.quantity(2000, 50, "mA")     # milliamps
   R = mh.quantity("V/I", "ohm")       # Just works!
   # Result: ~2.5 ohm (5V / 2A)
   ```

2. **Dimensional Validation**
   ```python
   # These would raise errors:
   m = mh.quantity(5, 0.1, "kg")
   t = mh.quantity(2, 0.05, "s")
   wrong = mh.quantity("m + t", "???")  # Can't add mass + time
   ```

3. **Consistent Results**
   - No more manual conversion errors
   - Units always match in formulas
   - Uncertainty propagation correct across scales

## Files Modified

### New Files
- `marhare/unit_converter.py` - Core conversion logic
- `test_unit_conversion.py` - Test suite

### Modified Files
- `pyproject.toml` - Added `pint>=0.20.0` dependency
- `marhare/uncertainties.py` - Import converter, normalize in `quantity()`
- `marhare/latex_tools.py` - Use `unit_display` for formatting
- `marhare/graphics.py` - Use `unit_display` for axis labels
- `docs/README_uncertainties.md` - Document new feature

## Testing

Run test suite:
```bash
cd Python_toolbox
python test_unit_conversion.py
```

Expected output shows:
- Unit normalization working
- LaTeX preserving display units
- Propagation with mixed units
- Dimensional compatibility checks

## Automatic Compact Unit Selection (`to_compact()`)

A key feature for human-readable output is automatic selection of the best SI prefix.

### Implementation

The `UnitConverter.to_compact()` method:
1. Converts value and sigma to SI base units
2. Applies pint's `to_compact()` to select best prefix
3. Scales sigma by EXACT same factor as value (consistency!)
4. Returns (compact_value, compact_sigma, compact_unit)

### Usage

Direct usage:
```python
from marhare.unit_converter import get_compact_units

val, sig, unit = get_compact_units(1e-9, 1e-12, "s")
# Returns: (1.0, 0.001, "nanosecond")

val, sig, unit = get_compact_units(5000, 100, "mV")
# Returns: (5.0, 0.1, "volt")
```

With `propagate_quantity()`:
```python
# Propagate with automatic compacting
result = mh.propagate_quantity(target, magnitudes, compact=True)
# If result is 0.005 A, now displays as 5 mA
```

### Example: Automatic Prefix Selection

```python
# Large frequency in Hz → converts to GHz
val, sig, unit = get_compact_units(2.4e9, 1e8, "Hz")
# (2.4, 0.1, "gigahertz")

# Tiny resistance in ohm → converts to milli-ohm
val, sig, unit = get_compact_units(0.005, 0.0001, "ohm")
# (5.0, 0.1, "milliohm")

# Voltage needs no conversion (5V already readable)
val, sig, unit = get_compact_units(5.0, 0.1, "V")
# (5.0, 0.1, "volt")
```

### Key Algorithm

1. **Input**: value, sigma, unit_str
2. **Normalize**: Convert both to SI base using same factor
3. **Compact**: Apply pint's `to_compact()` to base quantity
4. **Scale**: Multiply sigma by compact conversion factor
5. **Output**: (compact_value, compact_sigma, compact_unit)

### Consistency Guarantee

The method ensures dimensional correctness:
```python
# Input: value=5000, sigma=100, unit="mV"
# Step 1: Normalize to SI
#   value_si = 5.0 V, sigma_si = 0.1 V
# Step 2: Compact
#   compact_value = 5.0 V (already readable)
# Step 3: Scale sigma
#   factor = 5.0 / 5.0 = 1.0
#   compact_sigma = 0.1 V * 1.0 = 0.1 V
# Result: (5.0, 0.1, "volt")
# Verify: (5.0/5000) == (0.1/100) ✓ Same factor!
```

---

## Future Enhancements

Potential improvements:
1. **Smart unit inference**: Guess output units from formula
   ```python
   R = mh.quantity("V/I", ???)  # Auto-detect "ohm"
   ```

2. **Custom display units**: Convert result to preferred unit
   ```python
   R_result.to_unit("kohm")  # Display as kilohms
   ```

3. **Unit aliases**: Support more notation styles
   ```python
   "Ω" ↔ "ohm" ↔ "Ohm"
   "µm" ↔ "um" ↔ "micrometer"
   ```

4. **Temperature handling**: Special rules for °C, K, °F

## Performance Considerations

- Unit parsing is cached
- Conversion happens once at `quantity()` creation
- No overhead during propagation
- Minimal memory increase (one extra string per quantity)

## Error Handling

Graceful degradation:
- Unknown units → warning, no conversion
- Parse errors → warning, use string as-is
- Missing pint → warning at import, features disabled
- Dimensional errors → clear error messages

## Limitations

Current system does NOT:
- Handle non-linear conversions (°C ↔ K requires offset)
- Validate formula dimensions automatically
- Support unit arithmetic in expressions (`"km/h"` must be `"km / h"`)
- Convert historical/non-SI units (inches, pounds, etc.) without pint definitions

## Conclusion

This implementation provides professional-grade unit handling with zero API changes. Users can write natural scientific code with prefixes, and the system handles all conversions transparently.

"""
Comprehensive test suite for uncertainties v1.0 consolidated architecture.

Tests verify:
1. ✅ Immutability (cannot mutate after construction)
2. ✅ Unit separation (_unit_raw, _unit_internal, _unit_display)
3. ✅ Groups always stored in unit_internal
4. ✅ Validations enforced on construction and updates
5. ✅ API backward compatibility
6. ✅ propagate_quantity() returns new instances
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from marhare.uncertainties import quantity, propagate_quantity, register

print("=" * 70)
print("UNCERTAINTIES v1.0 — COMPREHENSIVE TEST SUITE")
print("=" * 70)

# Test counters
passed = 0
failed = 0

def test(name, condition, details=""):
    global passed, failed
    if condition:
        print(f"OK {name}")
        if details:
            print(f"  → {details}")
        passed += 1
    else:
        print(f"FAIL {name}")
        if details:
            print(f"  → {details}")
        failed += 1

# ============================================================================
# TEST 1: Immutability Enforcement
# ============================================================================
print("\n[1] IMMUTABILITY")
print("-" * 70)

V = quantity(5, 0.1, "V", symbol="V")

# Test: Cannot reassign _measure_value
try:
    V._measure_value = 999
    test("1.1 Block _measure_value mutation", False, "Should raise AttributeError")
except AttributeError as e:
    test("1.1 Block _measure_value mutation", "immutable" in str(e).lower(), str(e))

# Test: Cannot reassign _unit_internal
try:
    V._unit_internal = "A"
    test("1.2 Block _unit_internal mutation", False, "Should raise AttributeError")
except AttributeError:
    test("1.2 Block _unit_internal mutation", True, "Correctly blocked")

# Test: Cannot reassign _symbol
try:
    V._symbol = "X"
    test("1.3 Block _symbol mutation", False, "Should raise AttributeError")
except AttributeError:
    test("1.3 Block _symbol mutation", True, "Correctly blocked")

# ============================================================================
# TEST 2: Unit Separation
# ============================================================================
print("\n[2] UNIT SEPARATION")
print("-" * 70)

# Test: Raw unit preserved
V_mV = quantity(5000, 10, "mV", symbol="V", normalize=True)
test("2.1 unit_raw preserved", V_mV.unit_raw == "mV", f"unit_raw={V_mV.unit_raw}")

# Test: Internal unit is SI base (fundamental units, can be compound like kg*m^2/A/s^3 for V)
# Note: When normalize=True, pint converts to FUNDAMENTAL SI base units
is_fundamental_si = (V_mV.unit_internal == "V" or
                      "kilogram" in V_mV.unit_internal or 
                      V_mV.unit_internal == "volt")
test("2.2 unit_internal is SI base", is_fundamental_si, f"unit_internal={V_mV.unit_internal}")

# Test: Display unit is set (contains the "human" symbol)
test("2.3 unit_display set", V_mV._unit_display is not None, f"unit_display={V_mV._unit_display}")

# Test: .unit property returns internal when display is None
test("2.4 .unit returns internal", V_mV.unit == "V", f".unit={V_mV.unit}")

# Test: Measure values normalized
expected_value = 5.0  # 5000 mV → 5 V
test("2.5 Values normalized to internal", abs(V_mV['measure'][0] - expected_value) < 0.001, 
     f"measure={V_mV['measure'][0]} (expected {expected_value})")

# ============================================================================
# TEST 3: Compact Mode (unit_display)
# ============================================================================
print("\n[3] COMPACT MODE")
print("-" * 70)

V2 = quantity(5, 0.1, "V", symbol="V")
I2 = quantity(0.5, 0.01, "A", symbol="I")
R2 = quantity("V/I", "ohm", symbol="R")
registry2 = register(V2, I2, R2)
R2_result = propagate_quantity(R2, registry2, compact=True)

# Test: unit_internal unchanged by compact
test("3.1 unit_internal unchanged", R2_result.unit_internal == "ohm", 
     f"unit_internal={R2_result.unit_internal}")

# Test: unit_display set by compact
test("3.2 unit_display set", R2_result._unit_display == "ohm",
     f"unit_display={R2_result._unit_display}")

# Test: .unit returns display when set
test("3.3 .unit returns display", R2_result.unit == "ohm",
     f".unit={R2_result.unit}")

# ============================================================================
# TEST 4: Groups in unit_internal
# ============================================================================
print("\n[4] GROUPS IN UNIT_INTERNAL")
print("-" * 70)

# Test: Groups with normalization
V_groups = quantity(
    groups={
        "red": ([5000, 5100], [10, 10]),  # mV
    },
    unit="mV",
    symbol="V_exp",
    normalize=True
)

is_si_base = (V_groups.unit_internal == "V" or 
               "kilogram" in V_groups.unit_internal or 
               V_groups.unit_internal == "volt")
test("4.1 Groups unit_internal is SI", is_si_base,
     f"unit_internal={V_groups.unit_internal}")

# Test: Group values normalized to unit_internal
red_value = V_groups['_groups']['red']['value'][0]
expected = 5.0  # 5000 mV → 5 V
test("4.2 Group values in unit_internal", abs(red_value - expected) < 0.001,
     f"red[0]={red_value} (expected {expected})")

# ============================================================================
# TEST 5: Validations
# ============================================================================
print("\n[5] VALIDATIONS")
print("-" * 70)

# Test: Negative sigma rejected
try:
    bad_qty = quantity(5, -0.1, "V", symbol="bad")
    test("5.1 Reject negative sigma", False, "Should raise ValueError")
except ValueError as e:
    test("5.1 Reject negative sigma", "negative" in str(e).lower(), str(e))

# Test: Shape mismatch rejected
try:
    bad_qty2 = quantity(np.array([1, 2, 3]), np.array([0.1, 0.2]), "m", symbol="bad")
    test("5.2 Reject shape mismatch", False, "Should raise ValueError")
except ValueError as e:
    test("5.2 Reject shape mismatch", "shape" in str(e).lower(), str(e))

# Test: Scalar sigma for vector value (allowed, broadcasts)
try:
    good_qty = quantity(np.array([1, 2, 3]), 0.1, "m", symbol="good")
    test("5.3 Allow scalar sigma for vector", True, "Correctly broadcasts")
except:
    test("5.3 Allow scalar sigma for vector", False, "Should allow broadcasting")

# ============================================================================
# TEST 6: Immutable Updates (_with_result)
# ============================================================================
print("\n[6] IMMUTABLE UPDATES")
print("-" * 70)

V3 = quantity(5, 0.1, "V", symbol="V")
I3 = quantity(0.5, 0.01, "A", symbol="I")
R3 = quantity("V/I", "ohm", symbol="R")
registry3 = register(V3, I3, R3)

# Original R3 before propagation
original_R3_id = id(R3)

R3_result = propagate_quantity(R3, registry3)

# Test: Returns NEW instance
test("6.1 Returns new instance", id(R3_result) != original_R3_id,
     f"original={original_R3_id}, new={id(R3_result)}")

# Test: Original unchanged
test("6.2 Original unchanged", R3['result'] is None,
     f"original result={R3['result']}")

# Test: New instance has result
test("6.3 New instance has result", R3_result['result'] is not None,
     f"new result={R3_result['result']}")

# ============================================================================
# TEST 7: API Backward Compatibility
# ============================================================================
print("\n[7] API BACKWARD COMPATIBILITY")
print("-" * 70)

# Test: Old-style dict access works
V4 = quantity(5, 0.1, "V", symbol="V")
test("7.1 Dict access ['measure']", V4['measure'] == (5.0, 0.1), f"{V4['measure']}")
test("7.2 Dict access ['unit']", V4['unit'] == 'V', f"{V4['unit']}")
test("7.3 Dict access ['symbol']", V4['symbol'] == 'V', f"{V4['symbol']}")

# Test: Property access works
test("7.4 Property .value", V4.value == 5.0, f"{V4.value}")
test("7.5 Property .sigma", V4.sigma == 0.1, f"{V4.sigma}")
test("7.6 Property .unit", V4.unit == 'V', f"{V4.unit}")
test("7.7 Property .symbol", V4.symbol == 'V', f"{V4.symbol}")

# Test: as_dict() works
d = V4.as_dict()
test("7.8 as_dict() returns dict", isinstance(d, dict), f"type={type(d)}")
test("7.9 as_dict()['unit']", d['unit'] == 'V', f"{d['unit']}")

# ============================================================================
# TEST 8: Propagation Correctness
# ============================================================================
print("\n[8] PROPAGATION CORRECTNESS")
print("-" * 70)

V5 = quantity(10, 0.2, "V", symbol="V")
I5 = quantity(2, 0.05, "A", symbol="I")
R5 = quantity("V/I", "ohm", symbol="R")
registry5 = register(V5, I5, R5)
R5_result = propagate_quantity(R5, registry5)

expected_R = 10.0 / 2.0  # 5.0 ohm
actual_R = R5_result['result'][0]
test("8.1 Propagation value correct", abs(actual_R - expected_R) < 0.01,
     f"R={actual_R} (expected {expected_R})")

# Test: Uncertainty propagated
sigma_R = R5_result['result'][1]
test("8.2 Uncertainty propagated", sigma_R > 0, f"sigma_R={sigma_R}")

# ============================================================================
# TEST 9: Unit Conversion Integrity
# ============================================================================
print("\n[9] UNIT CONVERSION INTEGRITY")
print("-" * 70)

# Test: Conversion preserves physical identity
km_qty = quantity(5, 0.1, "km", symbol="d", normalize=True)
is_meter = km_qty.unit_internal in ["m", "meter"]
test("9.1 km normalized to meter", is_meter,
     f"unit_internal={km_qty.unit_internal}")

test("9.2 km value scaled", abs(km_qty['measure'][0] - 5000) < 1,
     f"value={km_qty['measure'][0]} (expected 5000 m)")

test("9.3 km sigma scaled", abs(km_qty['measure'][1] - 100) < 1,
     f"sigma={km_qty['measure'][1]} (expected 100 m)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print(f"PASSED: {passed}")
print(f"FAILED: {failed}")
print(f"TOTAL:  {passed + failed}")

if failed == 0:
    print("\n✅ ALL TESTS PASSED — v1.0 ARCHITECTURE VERIFIED ✅")
    exit(0)
else:
    print(f"\n❌ {failed} TEST(S) FAILED")
    exit(1)

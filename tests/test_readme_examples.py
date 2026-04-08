"""
Test suite to verify all code examples in README_uncertainties.md work correctly.

This ensures documentation stays in sync with implementation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import marhare as mh

print("=" * 70)
print("README EXAMPLES VERIFICATION")
print("=" * 70)

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
# Example 1: Basic quantity creation (from Core Concept section)
# ============================================================================
print("\n[1] CORE CONCEPT EXAMPLE")
print("-" * 70)

V = mh.quantity(5000.0, 100.0, "mV", symbol="V")
test("1.1 Create quantity with mV", V is not None)
test("1.2 Symbol is 'V'", V['symbol'] == 'V', f"symbol={V['symbol']}")
is_si_base = (V['unit'] == 'V') or ("kilogram" in str(V['unit'])) or ("volt" in str(V['unit']))
test("1.3 Unit normalized to SI-base representation", is_si_base, f"unit={V['unit']}")
test("1.4 Measure normalized to SI", abs(V['measure'][0] - 5.0) < 0.01, f"measure={V['measure']}")

# With normalize=False
V2 = mh.quantity(5000.0, 100.0, "mV", symbol="V", normalize=False)
test("1.5 normalize=False preserves unit", V2['unit'] == 'mV', f"unit={V2['unit']}")
test("1.6 normalize=False preserves value", abs(V2['measure'][0] - 5000) < 1, f"measure={V2['measure']}")

# ============================================================================
# Example 2: Ohm's Law (from Symbolic Error Propagation section)
# ============================================================================
print("\n[2] OHM'S LAW EXAMPLE")
print("-" * 70)

# Step 1: Define measurements
V = mh.quantity(10.0, 0.5, "V", symbol="V")
I = mh.quantity(2.0, 0.1, "A", symbol="I")

test("2.1 Create voltage", V['symbol'] == 'V')
test("2.2 Create current", I['symbol'] == 'I')

# Step 2: Define formula
R = mh.quantity("V/I", "ohm", symbol="R")
test("2.3 Create resistance formula", R['expr'] == 'V/I', f"expr={R['expr']}")

# Step 3: Register all
magnitudes = mh.register(V, I, R)
test("2.4 Register quantities", 'V' in magnitudes and 'I' in magnitudes and 'R' in magnitudes)

# Step 4: Propagate (pure dict)
R_result = mh.propagate_quantity(R, magnitudes)
test("2.5 Propagate returns dict", isinstance(R_result, dict))

# Step 5: Extract value
v, s = R_result["value"], R_result["sigma"]
expected_R = 10.0 / 2.0  # 5.0 ohm
test("2.6 R value correct", abs(v - expected_R) < 0.01, f"R={v} (expected {expected_R})")
test("2.7 R uncertainty exists", s > 0, f"sigma_R={s}")

# Verify output format
output_str = f"R = {v:.2f} ± {s:.2f} ohm"
test("2.8 Output format valid", "R = 5.00 ±" in output_str, f"output={output_str}")

# ============================================================================
# Example 3: LaTeX expressions (from Accessing Symbolic Expressions section)
# ============================================================================
print("\n[3] LATEX EXPRESSIONS EXAMPLE")
print("-" * 70)

V = mh.quantity(10.0, 0.5, "V", symbol="V")
I = mh.quantity(2.0, 0.1, "A", symbol="I")
R = mh.quantity("V/I", "ohm", symbol="R")

magnitudes = mh.register(V, I, R)
R_result = mh.propagate_quantity(R, magnitudes)

test("3.1 expr_latex exists", 'expr_latex' in R_result)
test("3.2 sigma_latex exists", 'sigma_latex' in R_result)

expr_latex = R_result["expr_latex"]
sigma_latex = R_result["sigma_latex"]

test("3.3 expr_latex is string or None", expr_latex is None or isinstance(expr_latex, str))
test("3.4 sigma_latex is string or None", sigma_latex is None or isinstance(sigma_latex, str))

if expr_latex:
    test("3.5 expr_latex contains V/I", 'V' in expr_latex and 'I' in expr_latex, f"expr_latex={expr_latex}")

if sigma_latex:
    test("3.6 sigma_latex contains sigma", 'sigma' in sigma_latex.lower(), f"sigma_latex={sigma_latex[:50]}...")

# ============================================================================
# Example 4: evaluate_quantity() (from extraction/evaluation flow)
# ============================================================================
print("\n[4] EVALUATE_QUANTITY EXAMPLE")
print("-" * 70)

q = mh.quantity(5.0, 0.1, "V", symbol="V")
q_eval = mh.evaluate_quantity(q, mh.register(q))
v, s = q_eval.value, q_eval.sigma

test("4.1 evaluate_quantity returns Quantity", q_eval is not None)
test("4.2 evaluated value correct", abs(v - 5.0) < 0.001, f"v={v}")
test("4.3 evaluated sigma correct", abs(s - 0.1) < 0.001, f"s={s}")

# ============================================================================
# Example 5: Pattern 1 - Measured Scalar (from Common Patterns section)
# ============================================================================
print("\n[5] COMMON PATTERNS - MEASURED SCALAR")
print("-" * 70)

V = mh.quantity(5.0, 0.1, "V", symbol="V")
test("5.1 Pattern 1 - scalar creation", V['symbol'] == 'V')
test("5.2 Pattern 1 - scalar value", V['measure'][0] == 5.0)
test("5.3 Pattern 1 - scalar sigma", V['measure'][1] == 0.1)

# ============================================================================
# Example 6: Pattern 2 - Measured Array (from Common Patterns section)
# ============================================================================
print("\n[6] COMMON PATTERNS - MEASURED ARRAY")
print("-" * 70)

times = mh.quantity(
    np.array([1.0, 2.0, 3.0, 4.0]),
    np.array([0.05, 0.05, 0.1, 0.1]),
    "s",
    symbol="t"
)
test("6.1 Pattern 2 - array creation", times['symbol'] == 't')
test("6.2 Pattern 2 - array shape", len(times['measure'][0]) == 4)
test("6.3 Pattern 2 - array value[0]", times['measure'][0][0] == 1.0)

# ============================================================================
# Example 7: No-groups policy verification
# ============================================================================
print("\n[7] NO-GROUPS POLICY")
print("-" * 70)

try:
    mh.quantity(groups={"red": ([1, 2], [0.1, 0.1])}, unit="m", symbol="x")
    test("7.1 groups argument rejected", False, "Expected TypeError")
except TypeError:
    test("7.1 groups argument rejected", True, "groups is intentionally unsupported")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("README EXAMPLES SUMMARY")
print("=" * 70)
print(f"PASSED: {passed}")
print(f"FAILED: {failed}")
print(f"TOTAL:  {passed + failed}")

if failed == 0:
    print("\n✅ ALL README EXAMPLES WORK CORRECTLY ✅")
    exit(0)
else:
    print(f"\n❌ {failed} EXAMPLE(S) FAILED")
    exit(1)

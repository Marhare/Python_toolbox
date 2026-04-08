"""V2 direct-computation workflow tests.

These tests prioritize Quantity-to-Quantity operations (V2 philosophy):
- direct arithmetic between Quantity objects
- extraction via value_quantity
- latex rendering for direct results
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import marhare as mh


print("=" * 70)
print("V2 DIRECT WORKFLOW TESTS")
print("=" * 70)

passed = 0
failed = 0


def test(name, condition, details=""):
    global passed, failed
    if condition:
        print(f"OK {name}")
        if details:
            print(f"  -> {details}")
        passed += 1
    else:
        print(f"FAIL {name}")
        if details:
            print(f"  -> {details}")
        failed += 1


# ---------------------------------------------------------------------------
# 1) Direct scalar computation: R = V / I
# ---------------------------------------------------------------------------
print("\n[1] DIRECT SCALAR COMPUTATION")
print("-" * 70)

V = mh.quantity(10.0, 0.5, "V", symbol="V")
I = mh.quantity(2.0, 0.1, "A", symbol="I")
R = V / I

r, sr = mh.value_quantity(R)

test("1.1 R is Quantity-like", hasattr(R, "value") and hasattr(R, "sigma"))
test("1.2 R nominal value", abs(float(r) - 5.0) < 1e-9, f"R={r}")
test("1.3 R sigma positive", float(sr) > 0, f"sigma={sr}")


# ---------------------------------------------------------------------------
# 2) Direct mixed operations: P = V * I, Z = (V / I) + (V / I)
# ---------------------------------------------------------------------------
print("\n[2] DIRECT MIXED OPERATIONS")
print("-" * 70)

P = V * I
p, sp = mh.value_quantity(P)

test("2.1 P nominal value", abs(float(p) - 20.0) < 1e-9, f"P={p}")
test("2.2 P sigma positive", float(sp) > 0, f"sigma={sp}")

Z = (V / I) + (V / I)
z, sz = mh.value_quantity(Z)
test("2.3 Z nominal value", abs(float(z) - 10.0) < 1e-9, f"Z={z}")
test("2.4 Z sigma positive", float(sz) > 0, f"sigma={sz}")


# ---------------------------------------------------------------------------
# 3) Vector computation
# ---------------------------------------------------------------------------
print("\n[3] DIRECT VECTOR COMPUTATION")
print("-" * 70)

Vv = mh.quantity(np.array([4.0, 6.0, 8.0]), np.array([0.2, 0.2, 0.2]), "V", symbol="Vv")
Iv = mh.quantity(np.array([2.0, 2.0, 2.0]), np.array([0.1, 0.1, 0.1]), "A", symbol="Iv")
Rv = Vv / Iv

rv, srv = mh.value_quantity(Rv)

test("3.1 vector shape preserved", np.asarray(rv).shape == (3,), f"shape={np.asarray(rv).shape}")
test("3.2 vector values", np.allclose(rv, np.array([2.0, 3.0, 4.0])), f"rv={rv}")
test("3.3 vector sigma nonnegative", np.all(np.asarray(srv) >= 0), f"srv={srv}")


# ---------------------------------------------------------------------------
# 4) normalize behavior
# ---------------------------------------------------------------------------
print("\n[4] NORMALIZE BEHAVIOR")
print("-" * 70)

V_si = mh.quantity(5000.0, 100.0, "mV", symbol="Vsi")
V_raw = mh.quantity(5000.0, 100.0, "mV", symbol="Vraw", normalize=False)

vsi, ssi = mh.value_quantity(V_si)
vraw, sraw = mh.value_quantity(V_raw)

test("4.1 normalize=True converts value", abs(float(vsi) - 5.0) < 1e-9, f"vsi={vsi}")
test("4.2 normalize=False preserves value", abs(float(vraw) - 5000.0) < 1e-9, f"vraw={vraw}")
test("4.3 normalize=False preserves sigma", abs(float(sraw) - 100.0) < 1e-9, f"sraw={sraw}")


# ---------------------------------------------------------------------------
# 5) LaTeX formatting for direct results
# ---------------------------------------------------------------------------
print("\n[5] LATEX ON DIRECT RESULT")
print("-" * 70)

latex_r = mh.latex_quantity(R, cifras=2)
latex_pm = mh.valor_pm(R, cifras=2)

test("5.1 latex_quantity returns str", isinstance(latex_r, str) and len(latex_r) > 0)
test("5.2 valor_pm returns str", isinstance(latex_pm, str) and len(latex_pm) > 0)


# ---------------------------------------------------------------------------
# 6) Dataset workflow: multi-measurement experiment
# ---------------------------------------------------------------------------
print("\n[6] DATASET WORKFLOW")
print("-" * 70)

# Create a dataset with experimental measurements
dataset = mh.Dataset(
    {
        "trial": np.array([1, 2, 3, 4]),
        "V": mh.quantity(np.array([10.0, 12.0, 9.5, 11.0]), np.array([0.5, 0.5, 0.5, 0.5]), "V", symbol="V"),
        "I": mh.quantity(np.array([2.0, 2.4, 1.9, 2.2]), np.array([0.1, 0.1, 0.1, 0.1]), "A", symbol="I"),
    },
    name="Resistor_Test"
)

# Direct computation on dataset columns
R_dataset = dataset["V"] / dataset["I"]
r_vals, sr_vals = mh.value_quantity(R_dataset)

test("6.1 dataset creates Quantity", isinstance(R_dataset, mh.Quantity), f"type={type(R_dataset)}")
test("6.2 dataset result is array", isinstance(r_vals, (np.ndarray, list)), f"type={type(r_vals)}")
test("6.3 dataset values in range", np.all((r_vals > 4.0) & (r_vals < 6.0)), f"vals={r_vals}")
test("6.4 dataset sigma all positive", np.all(sr_vals > 0), f"sigma={sr_vals}")


print("\n" + "=" * 70)
print("V2 DIRECT WORKFLOW SUMMARY (19 tests)")
print("=" * 70)
print(f"PASSED: {passed}")
print(f"FAILED: {failed}")
print(f"TOTAL:  {passed + failed}")

if failed == 0:
    print("\nV2 direct workflow tests passed")
    raise SystemExit(0)

print(f"\n{failed} test(s) failed")
raise SystemExit(1)

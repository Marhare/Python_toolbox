"""README V2 examples verification.

This suite validates examples from the V2 documentation style:
- direct Quantity arithmetic (priority)
- no register/propagate dependency in main examples
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import marhare as mh


print("=" * 70)
print("README V2 EXAMPLES VERIFICATION")
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


# 1) Quantity creation and normalize
print("\n[1] QUANTITY CREATION")
print("-" * 70)

V = mh.quantity(5000.0, 100.0, "mV", symbol="V")
Vn = mh.quantity(5000.0, 100.0, "mV", symbol="V", normalize=False)

test("1.1 symbol preserved", V["symbol"] == "V")
test("1.2 normalized value", abs(float(V["measure"][0]) - 5.0) < 1e-9)
test("1.3 non-normalized value", abs(float(Vn["measure"][0]) - 5000.0) < 1e-9)


# 2) Direct Ohm law
print("\n[2] DIRECT OHM LAW")
print("-" * 70)

V = mh.quantity(10.0, 0.5, "V", symbol="V")
I = mh.quantity(2.0, 0.1, "A", symbol="I")
R = V / I

r, sr = mh.value_quantity(R)
test("2.1 direct divide value", abs(float(r) - 5.0) < 1e-9, f"R={r}")
test("2.2 direct divide sigma", float(sr) > 0, f"sigma={sr}")


# 3) Direct power computation
print("\n[3] DIRECT POWER")
print("-" * 70)

P = V * I
p, sp = mh.value_quantity(P)
test("3.1 direct multiply value", abs(float(p) - 20.0) < 1e-9, f"P={p}")
test("3.2 direct multiply sigma", float(sp) > 0, f"sigma={sp}")


# 4) LaTeX rendering from direct results
print("\n[4] LATEX RENDERING")
print("-" * 70)

tex1 = mh.latex_quantity(R, cifras=2)
tex2 = mh.valor_pm(P, cifras=2)
test("4.1 latex_quantity string", isinstance(tex1, str) and len(tex1) > 0)
test("4.2 valor_pm string", isinstance(tex2, str) and len(tex2) > 0)


# 5) Array direct workflow
print("\n[5] ARRAY DIRECT WORKFLOW")
print("-" * 70)

x = mh.quantity(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.1, 0.1]), "s", symbol="t")
y = x + x
vy, sy = mh.value_quantity(y)

test("5.1 array value", np.allclose(vy, np.array([2.0, 4.0, 6.0])), f"vy={vy}")
test("5.2 array sigma", np.all(np.asarray(sy) >= 0), f"sy={sy}")


# 6) Dataset workflow: realistic lab experiment
print("\n[6] DATASET LAB WORKFLOW")
print("-" * 70)

# Real experiment: measure resistor with different power supplies
exp_data = mh.Dataset(
    {
        "sample": np.array(["R1", "R1", "R2", "R2"]),
        "voltage": mh.quantity(np.array([5.0, 10.0, 5.0, 10.0]), np.array([0.1, 0.1, 0.1, 0.1]), "V", symbol="U"),
        "current": mh.quantity(np.array([0.5, 1.0, 1.0, 2.0]), np.array([0.05, 0.05, 0.05, 0.05]), "A", symbol="I"),
    },
    name="Resistor_Characterization"
)

# Direct computation on measurements
R_exp = exp_data["voltage"] / exp_data["current"]
r_exp, sr_exp = mh.value_quantity(R_exp)

test("6.1 lab dataset creates result", isinstance(R_exp, mh.Quantity))
test("6.2 resistance values reasonable", np.all((r_exp > 4.0) & (r_exp < 12.0)), f"R={r_exp}")
test("6.3 uncertainties propagated", np.all(sr_exp > 0), f"sigma={sr_exp}")
test("6.4 dataset has metadata", exp_data.metadata is not None)


print("\n" + "=" * 70)
print("README V2 SUMMARY (15 tests)")
print("=" * 70)
print(f"PASSED: {passed}")
print(f"FAILED: {failed}")
print(f"TOTAL:  {passed + failed}")

if failed == 0:
    print("\nAll V2 README examples passed")
    raise SystemExit(0)

print(f"\n{failed} V2 example(s) failed")
raise SystemExit(1)

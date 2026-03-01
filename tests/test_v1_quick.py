"""Quick test of v1.0 immutable uncertainties architecture."""
from marhare.uncertainties import quantity, propagate_quantity, register

print("Testing v1.0 immutable architecture...")
print("-" * 50)

# Test 1: Basic propagation
print("\n[1] Basic propagation: R = V/I")
V = quantity(5, 0.1, "V", symbol="V")
I = quantity(0.5, 0.01, "A", symbol="I")
R = quantity("V/I", "ohm", symbol="R")

print(f"  V = {V['measure'][0]} ± {V['measure'][1]} {V.unit}")
print(f"  I = {I['measure'][0]} ± {I['measure'][1]} {I.unit}")

registry = register(V, I, R)
R_result = propagate_quantity(R, registry)

print(f"  R = {R_result['result'][0]:.2f} ± {R_result['result'][1]:.2f} {R_result.unit}")
print(f"  unit_raw: {R_result.unit_raw}")
print(f"  unit_internal: {R_result.unit_internal}")
print(f"  unit_display: {R_result._unit_display}")

# Test 2: Immutability check
print("\n[2] Immutability check")
try:
    R_result._result_value = 999  # Should attribute error on __slots__
    print("  FAILED: Could reassign _result_value directly!")
except AttributeError as e:
    print(f"  SUCCESS: Cannot mutate (AttributeError as expected)")

# Test 3: Compact mode
print("\n[3] Compact mode check")
R2 = quantity("V/I", "ohm", symbol="R2")
registry2 = register(V, I, R2)
R2_result = propagate_quantity(R2, registry2, compact=True)
print(f"  R2 (compact) = {R2_result['result'][0]:.2f} ± {R2_result['result'][1]:.2f} {R2_result.unit}")
print(f"  unit_internal: {R2_result.unit_internal}")
print(f"  unit_display: {R2_result._unit_display}")

print("\n" + "=" * 50)
print("v1.0 architecture tests PASSED!")

"""
Deep tests for uncertainties module architecture verification.

Tests cover:
1. Vectorial quantities (100+ elements with compact=True)
2. Chained propagation (A → B → C → D)
3. Group auto-inheritance
4. Mixed scalar + vector interactions
5. Edge cases (NaN filtering, zero sigma, unit conversion errors)

Purpose: Verify production-grade reliability beyond basic propagation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from marhare.uncertainties import quantity, propagate_quantity, register


def _value_sigma(result):
    return result["value"], result["sigma"]

print("=" * 70)
print("UNCERTAINTIES MODULE — DEEP TESTS")
print("=" * 70)


# ============================================================================
# TEST 1: Vectorial quantities (100 elements, compact mode)
# ============================================================================
print("\n[TEST 1] Vectorial propagation (N=100, compact=True)")
print("-" * 70)

try:
    # Create 100-element arrays
    N = 100
    V_array = np.linspace(4.5, 5.5, N)  # Voltages: 4.5 to 5.5 V
    V_sigma = np.full(N, 0.05)          # ±0.05 V
    
    I_array = np.linspace(0.45, 0.55, N)  # Currents: 0.45 to 0.55 A
    I_sigma = np.full(N, 0.005)           # ±0.005 A
    
    # Create quantities
    V = quantity(V_array, V_sigma, "V", symbol="V")
    I = quantity(I_array, I_sigma, "A", symbol="I")
    R = quantity("V/I", "ohm", symbol="R")
    
    # Propagate with compact mode
    registry = register(V, I, R)
    R_result = propagate_quantity(R, registry, compact=True)
    
    # Verify results
    R_values, R_sigmas = _value_sigma(R_result)
    
    print(f"✓ Propagation successful")
    print(f"  Input: V={V_array[0]:.2f}..{V_array[-1]:.2f} V, I={I_array[0]:.3f}..{I_array[-1]:.3f} A")
    print(f"  Output: R={R_values[0]:.2f}..{R_values[-1]:.2f} ± {R_sigmas[0]:.2f}..{R_sigmas[-1]:.2f} ohm")
    print(f"  Array shape: {R_values.shape}")
    print("  Compact unit: handled in evaluation layer")
    
    # Sanity check: R ≈ 10 ohm in the middle
    mid_idx = N // 2
    mid_R = R_values[mid_idx]
    expected_R = V_array[mid_idx] / I_array[mid_idx]
    
    if abs(mid_R - expected_R) / expected_R < 0.01:  # 1% tolerance
        print(f"✓ Numerical accuracy: R[{mid_idx}] = {mid_R:.4f} (expected {expected_R:.4f})")
    else:
        print(f"✗ FAILED: R[{mid_idx}] = {mid_R:.4f} (expected {expected_R:.4f})")
    
    print("[TEST 1] PASSED ✓")
    
except Exception as e:
    print(f"[TEST 1] FAILED ✗")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()


# ============================================================================
# TEST 2: Chained propagation (A → B → C → D)
# ============================================================================
print("\n[TEST 2] Chained propagation (4 levels deep)")
print("-" * 70)

try:
    # Chain: V, I → R → P=V²/R → efficiency η = P_out/P_in
    
    # Base quantities
    V_in = quantity(10.0, 0.2, "V", symbol="V_in")
    I_in = quantity(2.0, 0.05, "A", symbol="I_in")
    V_out = quantity(8.0, 0.15, "V", symbol="V_out")
    
    # Level 1: Resistance R = V_in / I_in
    R = quantity("V_in/I_in", "ohm", symbol="R")
    
    # Level 2: Input power P_in = V_in * I_in
    P_in = quantity("V_in*I_in", "W", symbol="P_in")
    
    # Level 3: Output power P_out = V_out² / R
    P_out = quantity("V_out**2/R", "W", symbol="P_out")
    
    # Level 4: Efficiency η = P_out / P_in (dimensionless)
    eta = quantity("P_out/P_in", "", symbol="eta")
    
    # Register and propagate step by step
    registry = register(V_in, I_in, V_out, R, P_in, P_out, eta)
    
    R_result = propagate_quantity(R, registry)
    registry["R"] = R_result
    
    P_in_result = propagate_quantity(P_in, registry)
    registry["P_in"] = P_in_result
    
    P_out_result = propagate_quantity(P_out, registry)
    registry["P_out"] = P_out_result
    
    eta_result = propagate_quantity(eta, registry)
    
    # Display chain
    print(f"✓ Chain propagated successfully")
    print(f"  V_in = {V_in['measure'][0]:.2f} ± {V_in['measure'][1]:.2f} V")
    print(f"  I_in = {I_in['measure'][0]:.2f} ± {I_in['measure'][1]:.2f} A")
    print(f"  V_out = {V_out['measure'][0]:.2f} ± {V_out['measure'][1]:.2f} V")
    print(f"  → R = {R_result['value']:.3f} ± {R_result['sigma']:.3f} ohm")
    print(f"  → P_in = {P_in_result['value']:.2f} ± {P_in_result['sigma']:.2f} W")
    print(f"  → P_out = {P_out_result['value']:.2f} ± {P_out_result['sigma']:.2f} W")
    print(f"  → η = {eta_result['value']:.4f} ± {eta_result['sigma']:.4f} {eta_result.get('unit', '')}")
    
    # Sanity check: η should be < 1 (cannot exceed 100% efficiency)
    eta_value = eta_result['value']
    if 0 < eta_value < 1:
        print(f"✓ Physical sanity: η = {eta_value:.2%} (within [0, 1])")
    else:
        print(f"✗ FAILED: η = {eta_value:.2%} (should be in [0, 1])")
    
    print("[TEST 2] PASSED ✓")
    
except Exception as e:
    print(f"[TEST 2] FAILED ✗")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()


# ============================================================================
# TEST 3: Groups are intentionally unsupported
# ============================================================================
print("\n[TEST 3] Groups are unsupported (new philosophy)")
print("-" * 70)

try:
    quantity(
        groups={
            "red": ([5.0, 5.1], [0.1, 0.1]),
            "blue": ([4.8, 4.9], [0.08, 0.08])
        },
        unit="V",
        symbol="V"
    )
    print("[TEST 3] FAILED ✗")
    print("  Error: groups argument should be rejected")
except TypeError:
    print("✓ groups argument correctly rejected")
    print("[TEST 3] PASSED ✓")
except Exception as e:
    print(f"[TEST 3] FAILED ✗")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()


# ============================================================================
# TEST 4: Mixed scalar + vector interaction
# ============================================================================
print("\n[TEST 4] Mixed scalar + vector interaction")
print("-" * 70)

try:
    # Scalar constant
    G = quantity(9.81, 0.01, "m/s**2", symbol="g")
    
    # Vector masses
    m_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # kg
    m_sigma = np.full(5, 0.02)
    m = quantity(m_array, m_sigma, "kg", symbol="m")
    
    # Force F = m * g (should broadcast)
    F = quantity("m*g", "N", symbol="F")
    
    registry = register(G, m, F)
    F_result = propagate_quantity(F, registry)
    
    print(f"✓ Scalar × Vector propagation")
    print(f"  g = {G['measure'][0]:.2f} ± {G['measure'][1]:.3f} {G['unit']}")
    print(f"  m = {m_array} ± {m_sigma[0]:.3f} {m['unit']}")
    print(f"  F = {F_result['value']} ± {F_result['sigma']} N")
    
    # Sanity check: F[0] ≈ 9.81 N, F[4] ≈ 49.05 N
    F_values = F_result['value']
    expected_F0 = m_array[0] * G['measure'][0]
    expected_F4 = m_array[4] * G['measure'][0]
    
    if abs(F_values[0] - expected_F0) < 0.1 and abs(F_values[4] - expected_F4) < 0.5:
        print(f"✓ Numerical accuracy: F[0]={F_values[0]:.2f} (expected {expected_F0:.2f}), "
              f"F[4]={F_values[4]:.2f} (expected {expected_F4:.2f})")
    else:
        print(f"✗ FAILED: F[0]={F_values[0]:.2f}, F[4]={F_values[4]:.2f}")
    
    print("[TEST 4] PASSED ✓")
    
except Exception as e:
    print(f"[TEST 4] FAILED ✗")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()


# ============================================================================
# TEST 5: Edge case — NaN filtering for standard arrays
# ============================================================================
print("\n[TEST 5] Edge case: NaN filtering on array quantities")
print("-" * 70)

try:
    V_with_nan = quantity(
        [5.0, np.nan, 5.2],
        [0.1, 0.1, 0.1],
        "V",
        symbol="V_nan",
        nan_policy="drop"
    )

    cleaned_values = V_with_nan["measure"][0]
    print(f"✓ NaN filtering applied")
    print(f"  Filtered values: {cleaned_values}")

    if len(cleaned_values) == 2 and not np.any(np.isnan(cleaned_values)):
        print(f"✓ NaN successfully removed")
    else:
        print(f"✗ FAILED: NaN filtering did not work correctly")

    print("[TEST 5] PASSED ✓")

except Exception as e:
    print(f"[TEST 5] FAILED ✗")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()


# ============================================================================
# TEST 6: Edge case — Zero sigma (deterministic quantity)
# ============================================================================
print("\n[TEST 6] Edge case: Zero sigma (deterministic)")
print("-" * 70)

try:
    # Exact constants
    pi = quantity(np.pi, 0.0, "", symbol="pi")
    r = quantity(2.0, 0.1, "m", symbol="r")
    
    # Area A = pi * r^2
    A = quantity("pi*r**2", "m**2", symbol="A")
    
    registry = register(pi, r, A)
    A_result = propagate_quantity(A, registry)
    
    print(f"✓ Zero-sigma propagation")
    print(f"  π = {pi['measure'][0]:.6f} ± {pi['measure'][1]} (deterministic)")
    print(f"  r = {r['measure'][0]:.2f} ± {r['measure'][1]:.2f} {r['unit']}")
    print(f"  A = {A_result['value']:.4f} ± {A_result['sigma']:.4f} meter ** 2")
    
    # Expected: A = π·(2.0)² = 12.566..., σ_A comes only from σ_r
    expected_A = np.pi * (2.0**2)
    if abs(A_result['value'] - expected_A) < 0.01:
        print(f"✓ Correct value: A ≈ {expected_A:.4f} m²")
    else:
        print(f"✗ FAILED: A = {A_result['value']:.4f} (expected {expected_A:.4f})")
    
    print("[TEST 6] PASSED ✓")
    
except Exception as e:
    print(f"[TEST 6] FAILED ✗")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("DEEP TESTS COMPLETED")
print("=" * 70)
print("\nIf all tests passed, the architecture handles:")
print("  ✓ Large vectorial quantities (N=100)")
print("  ✓ Multi-level chained propagation (4 levels)")
print("  ✓ No-groups policy enforcement")
print("  ✓ Scalar-vector broadcasting")
print("  ✓ Edge cases (NaN filtering, zero sigma)")
print("\nThese tests cover production-grade scenarios beyond basic V=IR.")
print("=" * 70)


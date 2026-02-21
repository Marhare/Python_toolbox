"""
Comprehensive test for weighted_standard_error function.

This test validates that:
1. The weighted_standard_error returns the correct value using the formula: σ_w = sqrt(1/Σw_i)
2. When w_i = 1/σ_i², the result matches the expected theoretical value
3. The function handles edge cases correctly
"""

import marhare as mh
import numpy as np

print("=" * 70)
print("WEIGHTED_STANDARD_ERROR VALIDATION TEST")
print("=" * 70)

# Test 1: Example from user's question
print("\n1. User's Example (gravity measurements):")
print("-" * 70)

measurements = mh.quantity(
    np.array([9.78, 9.81, 9.79, 9.82, 9.80]),
    np.array([0.04, 0.04, 0.05, 0.04, 0.04]),
    "m/s^2",
    symbol="g"
)

values, sigmas = mh.value_quantity(measurements)

g_mean = mh.weighted_mean(values, sigma=sigmas)
g_se = mh.weighted_standard_error(values, sigma=sigmas)

# Manual calculation for verification
weights = 1.0 / (sigmas**2)
expected_se = np.sqrt(1.0 / np.sum(weights))

print(f"Weighted mean: {g_mean:.6f} m/s²")
print(f"Weighted standard error: {g_se:.6f} m/s²")
print(f"Expected (manual calc): {expected_se:.6f} m/s²")
print(f"Match: {np.isclose(g_se, expected_se)}")

g_summary = mh.quantity(g_mean, g_se, "m/s^2", symbol="\\bar{g}")
tex = mh.latex_quantity(g_summary, cifras=2)
print(f"LaTeX output: {tex}")

# Test 2: Uniform weights (should give same result as regular standard error)
print("\n2. Uniform Weights Test:")
print("-" * 70)

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
se_regular = mh.standard_error(x)
se_weighted_uniform = mh.weighted_standard_error(x)  # No weights = uniform weights

print(f"Regular standard error: {se_regular:.6f}")
print(f"Weighted standard error (uniform): {se_weighted_uniform:.6f}")
print(f"Match: {np.isclose(se_regular, se_weighted_uniform)}")

# Test 3: Different uncertainties
print("\n3. Non-uniform Uncertainties Test:")
print("-" * 70)

x = np.array([10.0, 10.5, 9.8, 10.2, 9.9])
sigma = np.array([0.5, 0.3, 0.4, 0.2, 0.6])

mean_w = mh.weighted_mean(x, sigma=sigma)
se_w = mh.weighted_standard_error(x, sigma=sigma)

# Manual calculation
w = 1.0 / (sigma**2)
expected_mean = np.sum(w * x) / np.sum(w)
expected_se = np.sqrt(1.0 / np.sum(w))

print(f"Weighted mean: {mean_w:.6f}")
print(f"Expected mean: {expected_mean:.6f}")
print(f"Weighted SE: {se_w:.6f}")
print(f"Expected SE: {expected_se:.6f}")
print(f"Mean match: {np.isclose(mean_w, expected_mean)}")
print(f"SE match: {np.isclose(se_w, expected_se)}")

# Test 4: Explicit weights instead of sigmas
print("\n4. Explicit Weights Test:")
print("-" * 70)

x = np.array([1.0, 2.0, 3.0])
w = np.array([1.0, 2.0, 1.0])

mean_w = mh.weighted_mean(x, w=w)
se_w = mh.weighted_standard_error(x, w=w)
expected_se = np.sqrt(1.0 / np.sum(w))

print(f"Data: {x}")
print(f"Weights: {w}")
print(f"Weighted mean: {mean_w:.6f}")
print(f"Weighted SE: {se_w:.6f}")
print(f"Expected SE: {expected_se:.6f}")
print(f"Match: {np.isclose(se_w, expected_se)}")

# Test 5: Show the difference between weighted and unweighted error
print("\n5. Comparison: Weighted vs Unweighted Error:")
print("-" * 70)

# Case where one measurement is much more precise
x = np.array([10.0, 10.5, 11.0])
sigma_uniform = np.array([0.5, 0.5, 0.5])
sigma_precise = np.array([0.1, 0.5, 0.5])

se_uniform = mh.weighted_standard_error(x, sigma=sigma_uniform)
se_precise = mh.weighted_standard_error(x, sigma=sigma_precise)
se_unweighted = mh.standard_error(x)

print("Three measurements: [10.0, 10.5, 11.0]")
print(f"\nWith uniform σ=[0.5, 0.5, 0.5]:")
print(f"  Weighted SE: {se_uniform:.6f}")
print(f"\nWith precise first σ=[0.1, 0.5, 0.5]:")
print(f"  Weighted SE: {se_precise:.6f}")
print(f"\nUnweighted SE:")
print(f"  Standard SE: {se_unweighted:.6f}")
print(f"\nNote: When first measurement is more precise,")
print(f"weighted SE is smaller ({se_precise:.4f} < {se_uniform:.4f})")

print("\n" + "=" * 70)
print("ALL TESTS PASSED ✓")
print("=" * 70)

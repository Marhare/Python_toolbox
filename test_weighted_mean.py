import marhare as mh
import numpy as np

measurements = mh.quantity(
    np.array([9.78, 9.81, 9.79, 9.82, 9.80]),
    np.array([0.04, 0.04, 0.05, 0.04, 0.04]),
    "m/s^2",
    symbol="g"
)

values, sigmas = mh.value_quantity(measurements)

# Calculate weighted mean and its correct standard error
g_mean = mh.weighted_mean(values, sigma=sigmas)
g_se = mh.weighted_standard_error(values, sigma=sigmas)  # Now using the correct function!

g_summary = mh.quantity(g_mean, g_se, "m/s^2", symbol="\\bar{g}")
tex = mh.latex_quantity(g_summary, cifras=2)

print("Weighted mean:", g_mean)
print("Weighted standard error:", g_se)
print("LaTeX:", tex)

# For comparison, let's show what the old (incorrect) standard error was
old_se = mh.standard_error(values)
print("\nOld (incorrect) standard error:", old_se)
print("Difference:", abs(g_se - old_se))

# Verify the formula: sigma_w = sqrt(1/sum(w)) where w = 1/sigma^2
weights = 1.0 / (sigmas**2)
expected_error = np.sqrt(1.0 / np.sum(weights))
print("\nExpected error from formula sqrt(1/sum(w)):", expected_error)
print("Match:", np.isclose(g_se, expected_error))

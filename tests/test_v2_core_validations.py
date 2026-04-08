"""Core validation tests requested for V2 workflow.

Coverage:
1. Unit algebra correctness (+ special angle units: degree/radian)
2. Analytical uncertainty propagation correctness
3. Weighted mean / weighted standard error correctness
4. Documentation explicitly mentions angle units (degree/radian)
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import marhare as mh


def test_unit_algebra_division_and_dimensional_guard():
    voltage = mh.quantity(10.0, 0.5, "V", symbol="U")
    current = mh.quantity(2.0, 0.1, "A", symbol="I")
    resistance = voltage / current

    assert float(resistance.value) == 5.0
    assert float(resistance.sigma) > 0.0
    assert "ohm" in str(resistance.unit).lower() or "ampere" in str(resistance.unit).lower()

    length = mh.quantity(5.0, 0.1, "m")
    time = mh.quantity(2.0, 0.1, "s")
    try:
        _ = length + time
        assert False, "Adding incompatible dimensions should raise ValueError"
    except ValueError:
        pass


def test_angle_units_degree_and_radian_behavior():
    angle_deg = mh.quantity(180.0, 1.0, "degree", symbol="theta_deg")
    angle_rad = mh.quantity(math.pi, 0.01, "radian", symbol="theta_rad")

    # degree should normalize to SI angle base (radian) when conversion is enabled.
    assert abs(float(angle_deg.value) - math.pi) < 1e-12
    assert "radian" in str(angle_deg.unit).lower() or str(angle_deg.unit) == "1"

    s_deg = np.sin(angle_deg)
    s_rad = np.sin(angle_rad)

    # sin(180 degree) = sin(pi) ~= 0 and output is dimensionless
    assert abs(float(s_deg.value)) < 1e-10
    assert abs(float(s_rad.value)) < 1e-10
    assert str(s_deg.unit) in ("1", "dimensionless") or "dimensionless" in str(s_deg.unit)


def test_analytical_propagation_for_division_matches_closed_form():
    voltage = mh.quantity(10.0, 0.5, "V", symbol="U")
    current = mh.quantity(2.0, 0.1, "A", symbol="I")
    resistance = voltage / current

    expected_value = 10.0 / 2.0
    expected_sigma = math.sqrt((0.5 / 2.0) ** 2 + ((10.0 * 0.1) / (2.0 ** 2)) ** 2)

    assert abs(float(resistance.value) - expected_value) < 1e-12
    assert abs(float(resistance.sigma) - expected_sigma) < 1e-10


def test_weighted_mean_and_error_match_reference_formula():
    values = np.array([1.0, 2.0, 3.0], dtype=float)
    sigmas = np.array([0.1, 0.2, 0.1], dtype=float)

    q = mh.quantity(values, sigmas, "V", symbol="x")
    q_mean = q.weighted(symbol="xbar")

    w = 1.0 / (sigmas ** 2)
    expected_mean = float(np.sum(w * values) / np.sum(w))
    expected_se = float(np.sqrt(1.0 / np.sum(w)))

    assert abs(float(q_mean.value) - expected_mean) < 1e-12
    assert abs(float(q_mean.sigma) - expected_se) < 1e-12
    assert str(q_mean.unit) == str(q.unit)


def test_docs_explicitly_mention_degree_and_radian():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    docs_files = [
        os.path.join(root, "README.md"),
        os.path.join(root, "docs", "README_uncertainties.md"),
    ]

    content = "\n".join(open(path, "r", encoding="utf-8").read().lower() for path in docs_files)
    assert "degree" in content
    assert "radian" in content


def test_latex_quantity_infers_symbols_for_derived_quantities():
    s1p = mh.quantity(
        np.array([0.711, 0.711]),
        np.array([0.010, 0.010]),
        "meter",
        symbol="s1p",
        normalize=False,
    )
    s2p = mh.quantity(
        np.array([1.080, 2.000]),
        np.array([0.010, 0.010]),
        "meter",
        symbol="s2p",
        normalize=False,
    )
    div = mh.quantity(
        np.array([0.575, 0.550]),
        np.array([0.010, 0.010]),
        "meter",
        symbol="div",
        normalize=False,
    )

    s2 = s1p - div
    s2_im = s2p - div
    f = (s2 * s2_im) / (s2 - s2_im)

    tex = mh.latex_quantity([s2, s2_im, f])

    assert "s2 (" in tex
    assert "s2_im (" in tex
    assert "f (" in tex

"""Deep pytest checks for uncertainties module architecture."""

import numpy as np
import pytest

from marhare.uncertainties import propagate_quantity, quantity, register


def test_vectorial_propagation_with_compact_mode():
    n_points = 100
    voltage_values = np.linspace(4.5, 5.5, n_points)
    voltage_sigmas = np.full(n_points, 0.05)
    current_values = np.linspace(0.45, 0.55, n_points)
    current_sigmas = np.full(n_points, 0.005)

    voltage = quantity(voltage_values, voltage_sigmas, "V", symbol="V")
    current = quantity(current_values, current_sigmas, "A", symbol="I")
    resistance_expr = quantity("V/I", "ohm", symbol="R")

    result = propagate_quantity(
        resistance_expr,
        register(voltage, current, resistance_expr),
        compact=True,
    )
    r_values = result["value"]
    r_sigmas = result["sigma"]

    assert np.asarray(r_values).shape == (n_points,)
    assert np.asarray(r_sigmas).shape == (n_points,)

    mid = n_points // 2
    expected = voltage_values[mid] / current_values[mid]
    assert float(r_values[mid]) == pytest.approx(expected, rel=1e-2)


def test_chained_propagation_four_levels():
    v_in = quantity(10.0, 0.2, "V", symbol="V_in")
    i_in = quantity(2.0, 0.05, "A", symbol="I_in")
    v_out = quantity(8.0, 0.15, "V", symbol="V_out")

    resistance_expr = quantity("V_in/I_in", "ohm", symbol="R")
    p_in_expr = quantity("V_in*I_in", "W", symbol="P_in")
    p_out_expr = quantity("V_out**2/R", "W", symbol="P_out")
    eta_expr = quantity("P_out/P_in", "", symbol="eta")

    registry = register(v_in, i_in, v_out, resistance_expr, p_in_expr, p_out_expr, eta_expr)

    r_result = propagate_quantity(resistance_expr, registry)
    registry["R"] = r_result
    p_in_result = propagate_quantity(p_in_expr, registry)
    registry["P_in"] = p_in_result
    p_out_result = propagate_quantity(p_out_expr, registry)
    registry["P_out"] = p_out_result
    eta_result = propagate_quantity(eta_expr, registry)

    assert float(r_result["value"]) == pytest.approx(5.0, rel=1e-6)
    assert float(p_in_result["value"]) == pytest.approx(20.0, rel=1e-6)
    assert float(eta_result["value"]) == pytest.approx(0.64, rel=1e-6)
    assert 0.0 < float(eta_result["value"]) < 1.0


def test_groups_argument_is_unsupported():
    with pytest.raises(TypeError):
        quantity(
            groups={
                "red": ([5.0, 5.1], [0.1, 0.1]),
                "blue": ([4.8, 4.9], [0.08, 0.08]),
            },
            unit="V",
            symbol="V",
        )


def test_scalar_vector_broadcasting_in_symbolic_propagation():
    gravity = quantity(9.81, 0.01, "m/s**2", symbol="g")
    masses = quantity(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.full(5, 0.02), "kg", symbol="m")
    force_expr = quantity("m*g", "N", symbol="F")

    result = propagate_quantity(force_expr, register(gravity, masses, force_expr))
    force_values = np.asarray(result["value"])
    force_sigmas = np.asarray(result["sigma"])

    assert force_values.shape == (5,)
    assert np.all(force_sigmas > 0)
    assert force_values[0] == pytest.approx(9.81, abs=0.1)
    assert force_values[-1] == pytest.approx(49.05, abs=0.5)


def test_nan_filtering_policy_drop_removes_nan_entries():
    voltage = quantity(
        [5.0, np.nan, 5.2],
        [0.1, 0.1, 0.1],
        "V",
        symbol="V_nan",
        nan_policy="drop",
    )

    cleaned = np.asarray(voltage["measure"][0])
    assert len(cleaned) == 2
    assert not np.any(np.isnan(cleaned))


def test_zero_sigma_quantity_behaves_as_deterministic_constant():
    pi_value = quantity(np.pi, 0.0, "", symbol="pi")
    radius = quantity(2.0, 0.1, "m", symbol="r")
    area_expr = quantity("pi*r**2", "m**2", symbol="A")

    area_result = propagate_quantity(area_expr, register(pi_value, radius, area_expr))

    assert float(area_result["value"]) == pytest.approx(np.pi * 4.0, abs=1e-2)
    assert float(area_result["sigma"]) > 0


"""Pytest suite validating README uncertainty examples."""

import numpy as np
import pytest

import marhare as mh


def _is_voltage_like(unit_obj):
    unit_text = str(unit_obj).lower()
    return unit_text in {"v", "volt"} or "kilogram" in unit_text


def test_readme_core_quantity_creation_examples():
    voltage = mh.quantity(5000.0, 100.0, "mV", symbol="V")
    assert voltage is not None
    assert voltage["symbol"] == "V"
    assert _is_voltage_like(voltage["unit"])
    assert float(voltage["measure"][0]) == pytest.approx(5.0, abs=1e-2)

    voltage_raw = mh.quantity(5000.0, 100.0, "mV", symbol="V", normalize=False)
    assert str(voltage_raw["unit"]) == "mV"
    assert float(voltage_raw["measure"][0]) == pytest.approx(5000.0, abs=1)


def test_readme_symbolic_ohm_law_workflow_example():
    voltage = mh.quantity(10.0, 0.5, "V", symbol="V")
    current = mh.quantity(2.0, 0.1, "A", symbol="I")
    resistance_expr = mh.quantity("V/I", "ohm", symbol="R")

    registry = mh.register(voltage, current, resistance_expr)
    result = mh.propagate_quantity(resistance_expr, registry)

    assert isinstance(result, dict)
    assert float(result["value"]) == pytest.approx(5.0, abs=1e-2)
    assert float(result["sigma"]) > 0


def test_readme_latex_expression_example():
    voltage = mh.quantity(10.0, 0.5, "V", symbol="V")
    current = mh.quantity(2.0, 0.1, "A", symbol="I")
    resistance_expr = mh.quantity("V/I", "ohm", symbol="R")

    result = mh.propagate_quantity(resistance_expr, mh.register(voltage, current, resistance_expr))

    assert "expr_latex" in result
    assert "sigma_latex" in result
    assert result["expr_latex"] is None or isinstance(result["expr_latex"], str)
    assert result["sigma_latex"] is None or isinstance(result["sigma_latex"], str)


def test_readme_evaluate_quantity_example():
    quantity_obj = mh.quantity(5.0, 0.1, "V", symbol="V")
    evaluated = mh.evaluate_quantity(quantity_obj, mh.register(quantity_obj))

    assert evaluated is not None
    assert float(evaluated.value) == pytest.approx(5.0, abs=1e-3)
    assert float(evaluated.sigma) == pytest.approx(0.1, abs=1e-3)


def test_readme_common_pattern_measured_scalar():
    voltage = mh.quantity(5.0, 0.1, "V", symbol="V")

    assert voltage["symbol"] == "V"
    assert float(voltage["measure"][0]) == pytest.approx(5.0)
    assert float(voltage["measure"][1]) == pytest.approx(0.1)


def test_readme_common_pattern_measured_array():
    times = mh.quantity(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([0.05, 0.05, 0.1, 0.1]),
        "s",
        symbol="t",
    )

    assert times["symbol"] == "t"
    assert len(times["measure"][0]) == 4
    assert float(times["measure"][0][0]) == pytest.approx(1.0)


def test_readme_no_groups_policy_example():
    with pytest.raises(TypeError):
        mh.quantity(groups={"red": ([1, 2], [0.1, 0.1])}, unit="m", symbol="x")

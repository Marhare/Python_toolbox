"""Comprehensive pytest suite for uncertainties v1 consolidated architecture."""

import numpy as np
import pytest

from marhare.uncertainties import evaluate_quantity, propagate_quantity, quantity, register


def _is_voltage_like(unit_obj):
    unit_text = str(unit_obj).lower()
    return unit_text in {"v", "volt"} or "kilogram" in unit_text


def test_immutability_enforcement():
    voltage = quantity(5.0, 0.1, "V", symbol="V")

    with pytest.raises(AttributeError):
        voltage._measure_value = 999
    with pytest.raises(AttributeError):
        voltage._unit_internal = "A"
    with pytest.raises(AttributeError):
        voltage._symbol = "X"


def test_unit_separation_and_normalization():
    voltage_mv = quantity(5000, 10, "mV", symbol="V", normalize=True)

    assert voltage_mv.unit_raw == "mV"
    assert _is_voltage_like(voltage_mv.unit_internal)
    assert voltage_mv._unit_display is not None
    assert _is_voltage_like(voltage_mv.unit)
    assert float(voltage_mv["measure"][0]) == pytest.approx(5.0, abs=1e-3)


def test_compact_mode_preserves_internal_and_sets_display():
    voltage = quantity(5.0, 0.1, "V", symbol="V")
    current = quantity(0.5, 0.01, "A", symbol="I")
    resistance_expr = quantity("V/I", "ohm", symbol="R")

    result = evaluate_quantity(
        resistance_expr,
        register(voltage, current, resistance_expr),
        compact=True,
    )

    assert str(result.unit_internal)
    assert result._unit_display is not None
    assert str(result.unit)


def test_groups_argument_is_rejected():
    with pytest.raises(TypeError):
        quantity(
            groups={"red": ([5000, 5100], [10, 10])},
            unit="mV",
            symbol="V_exp",
            normalize=True,
        )


def test_validations_are_enforced():
    with pytest.raises(ValueError):
        quantity(5, -0.1, "V", symbol="bad")

    with pytest.raises(ValueError):
        quantity(np.array([1, 2, 3]), np.array([0.1, 0.2]), "m", symbol="bad")

    good_quantity = quantity(np.array([1, 2, 3]), 0.1, "m", symbol="good")
    assert good_quantity["measure"][0].shape == (3,)


def test_evaluate_quantity_returns_new_immutable_instance():
    voltage = quantity(5.0, 0.1, "V", symbol="V")
    current = quantity(0.5, 0.01, "A", symbol="I")
    resistance_expr = quantity("V/I", "ohm", symbol="R")

    result = evaluate_quantity(resistance_expr, register(voltage, current, resistance_expr))

    assert id(result) != id(resistance_expr)
    assert resistance_expr["result"] is None
    assert result["result"] is not None


def test_backward_compatible_api_accessors():
    voltage = quantity(5, 0.1, "V", symbol="V")

    value, sigma = voltage["measure"]
    assert float(value) == pytest.approx(5.0)
    assert float(sigma) == pytest.approx(0.1)
    assert _is_voltage_like(voltage["unit"])
    assert voltage["symbol"] == "V"

    assert float(voltage.value) == pytest.approx(5.0)
    assert float(voltage.sigma) == pytest.approx(0.1)
    assert _is_voltage_like(voltage.unit)
    assert voltage.symbol == "V"

    as_dict = voltage.as_dict()
    assert isinstance(as_dict, dict)
    assert _is_voltage_like(as_dict["unit"])


def test_propagation_correctness():
    voltage = quantity(10, 0.2, "V", symbol="V")
    current = quantity(2, 0.05, "A", symbol="I")
    resistance_expr = quantity("V/I", "ohm", symbol="R")

    result = propagate_quantity(resistance_expr, register(voltage, current, resistance_expr))
    assert float(result["value"]) == pytest.approx(5.0, abs=1e-2)
    assert float(result["sigma"]) > 0


def test_unit_conversion_integrity_for_distance():
    distance = quantity(5, 0.1, "km", symbol="d", normalize=True)

    assert str(distance.unit_internal) in {"m", "meter"}
    assert float(distance["measure"][0]) == pytest.approx(5000, abs=1)
    assert float(distance["measure"][1]) == pytest.approx(100, abs=1)

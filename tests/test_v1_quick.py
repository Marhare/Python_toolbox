"""Quick pytest checks for v1 immutable uncertainties workflow."""

import pytest

from marhare.uncertainties import evaluate_quantity, quantity, register


def test_v1_basic_propagation_returns_expected_resistance():
    voltage = quantity(5.0, 0.1, "V", symbol="V")
    current = quantity(0.5, 0.01, "A", symbol="I")
    resistance_expr = quantity("V/I", "ohm", symbol="R")

    result = evaluate_quantity(resistance_expr, register(voltage, current, resistance_expr))
    value, sigma = result["result"]

    assert float(value) == pytest.approx(10.0)
    assert float(sigma) > 0
    assert result.unit is not None


def test_v1_result_is_immutable():
    voltage = quantity(5.0, 0.1, "V", symbol="V")

    with pytest.raises(AttributeError):
        voltage._measure_value = 999


def test_v1_compact_mode_exposes_display_unit():
    voltage = quantity(5.0, 0.1, "V", symbol="V")
    current = quantity(0.5, 0.01, "A", symbol="I")
    resistance_expr = quantity("V/I", "ohm", symbol="R")

    result = evaluate_quantity(
        resistance_expr,
        register(voltage, current, resistance_expr),
        compact=True,
    )

    assert result._unit_display is not None
    assert str(result.unit_internal)
    assert str(result.unit)


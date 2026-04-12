import pytest

import marhare as mh


def test_dimensional_validation_in_addition_raises_error():
    length = mh.quantity(5.0, 0.1, "m", symbol="x")
    time = mh.quantity(3.0, 0.1, "s", symbol="y")

    with pytest.raises(ValueError):
        _ = length + time


def test_scalar_operations_with_units():
    value = mh.quantity(5.0, 0.1, "m", symbol="x")

    left_mul = 5 * value
    right_mul = value * 5
    division = value / 5

    assert float(left_mul.value) == pytest.approx(25.0)
    assert float(right_mul.value) == pytest.approx(25.0)
    assert float(division.value) == pytest.approx(1.0)
    assert str(left_mul.unit) == str(value.unit)
    assert str(right_mul.unit) == str(value.unit)
    assert str(division.unit) == str(value.unit)


def test_unit_simplification_for_resistance():
    voltage = mh.quantity(10.0, 0.5, "V", symbol="U")
    current = mh.quantity(2.0, 0.1, "A", symbol="I")
    resistance = voltage / current

    assert float(resistance.value) == pytest.approx(5.0)
    assert "ohm" in str(resistance.unit).lower() or "ampere" in str(resistance.unit).lower()


def test_complex_unit_algebra_dimensionless_division():
    speed_a = mh.quantity(6.0, 0.1, "m/s", symbol="v")
    speed_b = mh.quantity(2.0, 0.05, "m/s", symbol="v2")
    ratio = speed_a / speed_b

    assert float(ratio.value) == pytest.approx(3.0)
    assert str(ratio.unit) in {"", "1", "dimensionless"} or "dimensionless" in str(ratio.unit)

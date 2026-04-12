import pint
import pytest


ureg = pint.UnitRegistry()


def test_pint_basic_units_are_available():
    assert str((1.0 * ureg.meter).units) == "meter"
    assert "second" in str((1.0 * ureg.meter / ureg.second).units)


def test_volt_over_ampere_is_ohm_equivalent():
    va = 1.0 * ureg.volt / ureg.ampere
    ohm = 1.0 * ureg.ohm

    assert va.dimensionality == ohm.dimensionality
    assert va.to("ohm").magnitude == pytest.approx(1.0)


def test_known_reference_units_have_dimensionality():
    for name in ["ohm", "watt", "joule", "pascal", "newton", "hertz"]:
        quantity = 1.0 * getattr(ureg, name)
        assert quantity.dimensionality is not None

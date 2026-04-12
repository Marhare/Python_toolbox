"""Pytest checks for the V2 direct-computation workflow."""

import numpy as np
import pytest

import marhare as mh


def test_direct_scalar_computation_for_resistance():
    voltage = mh.quantity(10.0, 0.5, "V", symbol="V")
    current = mh.quantity(2.0, 0.1, "A", symbol="I")
    resistance = voltage / current

    value, sigma = mh.value_quantity(resistance)
    assert hasattr(resistance, "value")
    assert hasattr(resistance, "sigma")
    assert float(value) == pytest.approx(5.0)
    assert float(sigma) > 0


def test_direct_mixed_operations():
    voltage = mh.quantity(10.0, 0.5, "V", symbol="V")
    current = mh.quantity(2.0, 0.1, "A", symbol="I")

    power = voltage * current
    p, sp = mh.value_quantity(power)
    assert float(p) == pytest.approx(20.0)
    assert float(sp) > 0

    double_resistance = (voltage / current) + (voltage / current)
    z, sz = mh.value_quantity(double_resistance)
    assert float(z) == pytest.approx(10.0)
    assert float(sz) > 0


def test_direct_vector_computation():
    voltage = mh.quantity(np.array([4.0, 6.0, 8.0]), np.array([0.2, 0.2, 0.2]), "V", symbol="Vv")
    current = mh.quantity(np.array([2.0, 2.0, 2.0]), np.array([0.1, 0.1, 0.1]), "A", symbol="Iv")
    resistance = voltage / current

    r_values, r_sigmas = mh.value_quantity(resistance)
    assert np.asarray(r_values).shape == (3,)
    assert np.allclose(r_values, np.array([2.0, 3.0, 4.0]))
    assert np.all(np.asarray(r_sigmas) >= 0)


def test_normalize_behavior():
    voltage_si = mh.quantity(5000.0, 100.0, "mV", symbol="Vsi")
    voltage_raw = mh.quantity(5000.0, 100.0, "mV", symbol="Vraw", normalize=False)

    vsi, _ = mh.value_quantity(voltage_si)
    vraw, sraw = mh.value_quantity(voltage_raw)

    assert float(vsi) == pytest.approx(5.0)
    assert float(vraw) == pytest.approx(5000.0)
    assert float(sraw) == pytest.approx(100.0)


def test_latex_helpers_on_direct_results():
    voltage = mh.quantity(10.0, 0.5, "V", symbol="V")
    current = mh.quantity(2.0, 0.1, "A", symbol="I")
    resistance = voltage / current

    latex_quantity = mh.latex_quantity(resistance, cifras=2)
    latex_pm = mh.valor_pm(resistance, cifras=2)

    assert isinstance(latex_quantity, str) and len(latex_quantity) > 0
    assert isinstance(latex_pm, str) and len(latex_pm) > 0


def test_dataset_direct_workflow():
    dataset = mh.Dataset(
        {
            "trial": np.array([1, 2, 3, 4]),
            "V": mh.quantity(
                np.array([10.0, 12.0, 9.5, 11.0]),
                np.array([0.5, 0.5, 0.5, 0.5]),
                "V",
                symbol="V",
            ),
            "I": mh.quantity(
                np.array([2.0, 2.4, 1.9, 2.2]),
                np.array([0.1, 0.1, 0.1, 0.1]),
                "A",
                symbol="I",
            ),
        },
        name="Resistor_Test",
    )

    resistance = dataset["V"] / dataset["I"]
    r_vals, sr_vals = mh.value_quantity(resistance)

    assert isinstance(resistance, mh.Quantity)
    assert isinstance(r_vals, (np.ndarray, list))
    assert np.all((r_vals > 4.0) & (r_vals < 6.0))
    assert np.all(sr_vals > 0)

"""Pytest suite validating README V2 direct-workflow examples."""

import numpy as np
import pytest

import marhare as mh


def test_v2_readme_quantity_creation_and_normalization():
    voltage = mh.quantity(5000.0, 100.0, "mV", symbol="V")
    voltage_raw = mh.quantity(5000.0, 100.0, "mV", symbol="V", normalize=False)

    assert voltage["symbol"] == "V"
    assert float(voltage["measure"][0]) == pytest.approx(5.0)
    assert float(voltage_raw["measure"][0]) == pytest.approx(5000.0)


def test_v2_readme_direct_ohm_law_and_power():
    voltage = mh.quantity(10.0, 0.5, "V", symbol="V")
    current = mh.quantity(2.0, 0.1, "A", symbol="I")

    resistance = voltage / current
    r, sr = mh.value_quantity(resistance)
    assert float(r) == pytest.approx(5.0)
    assert float(sr) > 0

    power = voltage * current
    p, sp = mh.value_quantity(power)
    assert float(p) == pytest.approx(20.0)
    assert float(sp) > 0


def test_v2_readme_latex_rendering_from_direct_results():
    voltage = mh.quantity(10.0, 0.5, "V", symbol="V")
    current = mh.quantity(2.0, 0.1, "A", symbol="I")
    resistance = voltage / current
    power = voltage * current

    tex_quantity = mh.latex_quantity(resistance, cifras=2)
    tex_pm = mh.valor_pm(power, cifras=2)

    assert isinstance(tex_quantity, str) and len(tex_quantity) > 0
    assert isinstance(tex_pm, str) and len(tex_pm) > 0


def test_v2_readme_array_direct_workflow():
    time = mh.quantity(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.1, 0.1]), "s", symbol="t")
    doubled = time + time
    values, sigmas = mh.value_quantity(doubled)

    assert np.allclose(values, np.array([2.0, 4.0, 6.0]))
    assert np.all(np.asarray(sigmas) >= 0)


def test_v2_readme_dataset_lab_workflow():
    experiment = mh.Dataset(
        {
            "sample": np.array(["R1", "R1", "R2", "R2"]),
            "voltage": mh.quantity(
                np.array([5.0, 10.0, 5.0, 10.0]),
                np.array([0.1, 0.1, 0.1, 0.1]),
                "V",
                symbol="U",
            ),
            "current": mh.quantity(
                np.array([0.5, 1.0, 1.0, 2.0]),
                np.array([0.05, 0.05, 0.05, 0.05]),
                "A",
                symbol="I",
            ),
        },
        name="Resistor_Characterization",
    )

    resistance = experiment["voltage"] / experiment["current"]
    values, sigmas = mh.value_quantity(resistance)

    assert isinstance(resistance, mh.Quantity)
    assert np.all((values > 4.0) & (values < 12.0))
    assert np.all(sigmas > 0)
    assert experiment.metadata is not None

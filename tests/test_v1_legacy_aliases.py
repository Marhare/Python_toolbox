"""Compatibility tests for legacy v1 import paths without physical shim modules."""

import importlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_legacy_modules_are_importable():
    m_unc = importlib.import_module("marhare.uncertainties")
    m_q2 = importlib.import_module("marhare.quantities2")
    m_prop = importlib.import_module("marhare.propagation")
    m_ltx = importlib.import_module("marhare.latex_tools")

    assert hasattr(m_unc, "quantity")
    assert hasattr(m_unc, "register")
    assert hasattr(m_unc, "evaluate_quantity")

    assert hasattr(m_q2, "quantity")
    assert hasattr(m_prop, "propagate_quantity")
    assert hasattr(m_ltx, "latex_quantity")


def test_v1_style_workflow_still_runs():
    from marhare.uncertainties import quantity, register, evaluate_quantity

    voltage = quantity(5.0, 0.1, "V", symbol="V")
    current = quantity(0.5, 0.01, "A", symbol="I")
    resistance_expr = quantity("V/I", "ohm", symbol="R")

    registry = register(voltage, current, resistance_expr)
    resistance_result = evaluate_quantity(resistance_expr, registry)

    val, sig = resistance_result["result"]
    assert float(val) > 0
    assert float(sig) > 0
    assert resistance_result.unit is not None

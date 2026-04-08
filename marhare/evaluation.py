"""Evaluation bridge from symbolic propagation results to ``Quantity`` objects.

This module takes the numeric/symbolic output from
``marhare.quantities.propagation.propagate_quantity`` and returns a new
``Quantity`` preserving metadata (symbol, units, traceability fields).
"""

from __future__ import annotations

from typing import Any

from marhare.quantities import units
from marhare.quantities.propagation import propagate_quantity
from marhare.quantities.quantities import Quantity


def evaluate_quantity(quantity: Any, registry: dict, simplify: bool = True, compact: bool = False, group=None, **bindings) -> Quantity:
    """Evaluate a quantity-like expression and return a new ``Quantity``.

    Parameters
    ----------
    quantity : Quantity | dict-like
        Target quantity-like object (symbolic or numeric).
    registry : dict
        Mapping from symbols to quantity-like inputs.
    simplify : bool, default True
        Whether to simplify symbolic expressions during propagation.
    compact : bool, default False
        If True, tries compact display units for the result.
    group : Any, optional
        Reserved compatibility argument.

    Returns
    -------
    Quantity
        New object preserving source metadata plus evaluated result fields.
    """
    result = propagate_quantity(
        quantity,
        magnitudes=registry,
        simplify=simplify,
        compact=False,
        group=group,
        **bindings,
    )

    if isinstance(quantity, Quantity):
        measure = quantity.get("measure", None)
        base_value, base_sigma = measure if measure is not None else (None, None)
        symbol = quantity.symbol
        unit_internal = quantity.unit_internal
        unit_raw = quantity.unit_raw
        unit_display = quantity.unit_display
        expr = quantity.expr
        traceable = quantity.is_traceable
    else:
        measure = quantity.get("measure", None) if isinstance(quantity, dict) else None
        base_value, base_sigma = measure if measure is not None else (None, None)
        symbol = quantity.get("symbol", None) if isinstance(quantity, dict) else None
        unit_internal = (quantity.get("unit_internal", quantity.get("unit", "1")) if isinstance(quantity, dict) else "1")
        unit_raw = quantity.get("unit_raw", unit_internal) if isinstance(quantity, dict) else unit_internal
        unit_display = quantity.get("unit_display", unit_internal) if isinstance(quantity, dict) else unit_internal
        expr = quantity.get("expr", None) if isinstance(quantity, dict) else None
        traceable = True

    result_value = result["value"]
    result_sigma = result["sigma"]
    result_unit_display = unit_display

    if compact and units.is_unit_conversion_available() and unit_internal is not None:
        compact_value, compact_sigma, compact_unit = units.compact_units(result_value, result_sigma, unit_internal)
        result_value = compact_value
        result_sigma = compact_sigma
        result_unit_display = compact_unit if compact_unit else unit_internal

    return Quantity(
        value=base_value,
        sigma=base_sigma,
        unit=unit_internal,
        symbol=symbol,
        traceable=traceable,
        _expr=expr,
        _expr_latex=result.get("expr_latex"),
        _sigma_latex=result.get("sigma_latex"),
        _unit_raw=unit_raw,
        _unit_display=result_unit_display,
        _result_value=result_value,
        _result_sigma=result_sigma,
    )

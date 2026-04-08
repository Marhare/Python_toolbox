"""
Pure symbolic uncertainty propagation utilities.

This module is intentionally Quantity-agnostic:
- it accepts quantity-like objects in registries (duck typing),
- it never constructs Quantity instances,
- it never mutates input registries.

Main public entry points are ``propagate_quantity`` and
``uncertainty_propagation``.
"""

from __future__ import annotations

import inspect
from typing import Any, Optional

import numpy as np
import sympy as sp


def _extract_symbol(q: Any) -> Optional[str]:
    if isinstance(q, dict):
        sym = q.get("symbol", None)
    else:
        sym = getattr(q, "symbol", None)
    if isinstance(sym, str):
        sym = sym.strip()
    return sym or None


def _extract_expr(q: Any):
    if isinstance(q, dict):
        return q.get("expr", None)
    return getattr(q, "expr", None)


def _extract_measure(q: Any):
    if isinstance(q, dict):
        if q.get("result", None) is not None:
            return q["result"]
        if q.get("measure", None) is not None:
            return q["measure"]
        if "value" in q and "sigma" in q:
            return (q["value"], q["sigma"])
        return None

    if hasattr(q, "value") and hasattr(q, "sigma"):
        return (q.value, q.sigma)
    return None


def _checker(value: Any, sigma: Any) -> tuple[Any, Any]:
    value_arr = np.asarray(value, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)

    if value_arr.shape != () and sigma_arr.shape == ():
        sigma_arr = np.full(value_arr.shape, float(sigma_arr), dtype=float)

    if value_arr.shape != sigma_arr.shape:
        raise ValueError(f"incompatible shapes: value={value_arr.shape}, sigma={sigma_arr.shape}")

    if np.any(sigma_arr < 0):
        raise ValueError("sigma cannot be negative")

    if value_arr.shape == ():
        return float(value_arr), float(sigma_arr)
    return value_arr, sigma_arr


def uncertainty_propagation(
    f: sp.Expr,
    vars_: list[sp.Symbol],
    values: dict[sp.Symbol, object],
    sigmas: dict[sp.Symbol, float],
    cov: sp.Matrix | None = None,
    simplify: bool = True,
) -> dict:
    for v in vars_:
        if v not in values:
            raise ValueError(f"Missing value for {v}")
        if v not in sigmas:
            raise ValueError(f"Missing sigma for {v}")
        if np.any(np.asarray(sigmas[v]) < 0):
            raise ValueError(f"Negative sigma for {v}")

    grad = sp.Matrix([sp.diff(f, v) for v in vars_])

    sigma_symbols = {v: sp.Symbol(f"sigma_{v.name}") for v in vars_}
    if cov is None:
        Sigma = sp.diag(*[sigma_symbols[v] ** 2 for v in vars_])
    else:
        if cov.shape != (len(vars_), len(vars_)):
            raise ValueError("Incorrect dimensions for cov")
        Sigma = cov

    var_f = (grad.T * Sigma * grad)[0]
    sigma_f_expr = sp.sqrt(var_f)

    if simplify:
        var_f = sp.simplify(var_f)
        sigma_f_expr = sp.simplify(sigma_f_expr)

    f_num = sp.lambdify(vars_, f, "numpy")
    sigma_syms = [sigma_symbols[v] for v in vars_]
    s_num = sp.lambdify(vars_ + sigma_syms, sigma_f_expr, "numpy")

    args = [values[v] for v in vars_]
    s_args = [sigmas[v] for v in vars_]

    return {
        "value": f_num(*args),
        "sigma": s_num(*args, *s_args),
        "expr_latex": sp.latex(f),
        "sigma_latex": sp.latex(sigma_f_expr),
    }


def _propagate(expr, values: dict, sigmas: dict, simplify=True):
    symbols = list(expr.free_symbols)

    for s in symbols:
        if s not in values:
            raise ValueError(f"Missing value for {s}")
        if s not in sigmas:
            raise ValueError(f"Missing sigma for {s}")

    vectorial = any(np.ndim(v) > 0 for v in values.values())
    if vectorial:
        lengths = [len(v) for v in values.values() if np.ndim(v) > 0]
        n = max(lengths) if lengths else 1
    else:
        n = 1

    f_vals = []
    s_vals = []

    for i in range(n):
        vals_i = {}
        sigmas_i = {}
        for s in symbols:
            v = values[s]
            vals_i[s] = v[i] if vectorial and np.ndim(v) > 0 else v
            sg = sigmas[s]
            sigmas_i[s] = sg[i] if vectorial and np.ndim(sg) > 0 else sg

        res = uncertainty_propagation(expr, symbols, vals_i, sigmas_i, simplify=simplify)
        f_vals.append(res["value"])
        s_vals.append(res["sigma"])

    return {
        "value": np.array(f_vals) if vectorial else f_vals[0],
        "sigma": np.array(s_vals) if vectorial else s_vals[0],
        "expr_latex": res["expr_latex"],
        "sigma_latex": res["sigma_latex"],
    }


def _build_registry(magnitudes=None, **bindings):
    if magnitudes is None:
        registry = {}
    elif isinstance(magnitudes, dict):
        registry = dict(magnitudes)
    else:
        registry = {}
        for q in magnitudes:
            symbol = _extract_symbol(q)
            if symbol is None:
                raise ValueError("All magnitudes must define a non-empty symbol")
            if symbol in registry:
                raise ValueError(f"Duplicate magnitude symbol: {symbol}")
            registry[symbol] = q

    for symbol, quantity_obj in bindings.items():
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Binding names must be non-empty strings")
        registry[symbol.strip()] = quantity_obj

    return registry


def _resolve_target_name(target, registry):
    if isinstance(target, str):
        return target

    name = _extract_symbol(target)
    if name:
        registry[name] = target
        return name

    matches = [k for k, q in registry.items() if q is target]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError("Target quantity appears multiple times in registry")

    name = "_result"
    registry[name] = target
    return name


def _evaluate_expression(expr_sym, values, sigmas, simplify=True):
    return _propagate(expr_sym, values, sigmas, simplify=simplify)


def _resolve_dependencies(target_name, registry, simplify=True):
    symbols = {k: sp.Symbol(k) for k in registry}
    cache = {}
    resolving = set()

    def resolve_quantity(key: str):
        if key in cache:
            return cache[key]
        if key not in registry:
            raise ValueError(f"Missing quantity for {key}")

        q = registry[key]
        expr = _extract_expr(q)
        measure = _extract_measure(q)

        if expr is None:
            if measure is None:
                raise ValueError(f"{key} has no measure or expression")
            value, sigma = _checker(measure[0], measure[1])
            out = {
                "value": value,
                "sigma": sigma,
                "expr_latex": None,
                "sigma_latex": None,
            }
            cache[key] = out
            return out

        if key in resolving:
            raise ValueError(f"Circular dependency detected at {key}")

        resolving.add(key)
        expr_sym = sp.sympify(expr, locals=symbols)
        expr_symbols = list(expr_sym.free_symbols)

        values = {}
        sigmas = {}
        for sym in expr_symbols:
            dep_res = resolve_quantity(sym.name)
            values[sym] = dep_res["value"]
            sigmas[sym] = dep_res["sigma"]

        out = _evaluate_expression(expr_sym, values, sigmas, simplify=simplify)
        cache[key] = out
        resolving.remove(key)
        return out

    return resolve_quantity(target_name)


def _handle_groups(group=None):
    if group is not None:
        raise ValueError("group propagation is not supported in quantities2-first mode")
    return None


def propagate_quantity(target, magnitudes=None, simplify=True, compact=False, group=None, **bindings):
    """
    Pure symbolic propagation.

    Returns a pure dictionary with keys:
    {
        "value": ...,
        "sigma": ...,
        "expr_latex": ...,
        "sigma_latex": ...,
        "groups": optional
    }
    """
    _ = compact  # Reserved for compatibility; display conversion belongs to evaluation layer.
    groups = _handle_groups(group=group)

    registry = _build_registry(magnitudes, **bindings)
    name = _resolve_target_name(target, registry)

    if name not in registry:
        raise ValueError(f"Missing quantity for {name}")

    res = _resolve_dependencies(name, registry, simplify=simplify)

    out = {
        "value": res["value"],
        "sigma": res["sigma"],
        "expr_latex": res["expr_latex"],
        "sigma_latex": res["sigma_latex"],
    }
    if groups is not None:
        out["groups"] = groups
    return out


def register(*magnitudes):
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        raise RuntimeError("register() could not access caller frame")

    try:
        caller_locals = frame.f_back.f_locals
        registry = {}
        seen_ids = set()

        for q in magnitudes:
            symbol = _extract_symbol(q)

            if symbol is None:
                names = [name for name, val in caller_locals.items() if val is q]
                if len(names) == 0:
                    raise ValueError("register(): unnamed magnitude not found in caller namespace")
                if len(names) > 1:
                    raise ValueError(
                        "register(): unnamed magnitude has multiple names in caller namespace: "
                        + ", ".join(sorted(names))
                    )
                symbol = names[0]

            if symbol in registry:
                raise ValueError(f"register(): duplicate symbol '{symbol}'")

            obj_id = id(q)
            if obj_id in seen_ids:
                raise ValueError("register(): duplicate magnitude object provided")
            seen_ids.add(obj_id)

            if isinstance(q, dict):
                q["symbol"] = symbol

            registry[symbol] = q

        return registry
    finally:
        del frame

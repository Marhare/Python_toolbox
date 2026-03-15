"""
Uncertainty propagation logic for uncertainties package.

This module handles:
- Symbolic uncertainty propagation via SymPy
- High-level propagate_quantity() for derived quantities  
- Symbol registry management
- Dependency resolution with cycle detection
- Group-based propagation modes

Design principles:
- Can import from quantities.py (Quantity, _Uncertainties.checker)
- Can import from units.py for compact units
- No imports from graphics, latex_tools, matplotlib
- No modification of unit_internal by compact option
- Circular dependency detection
"""

from __future__ import annotations
import functools
import inspect
import numpy as np
import sympy as sp
from typing import Optional
from .quantities import Quantity, _Uncertainties
from . import units


def uncertainty_propagation(
    f: sp.Expr,
    vars_: list[sp.Symbol],
    values: dict[sp.Symbol, object],
    sigmas: dict[sp.Symbol, float],
    cov: sp.Matrix | None = None,
    simplify: bool = True
) -> dict:
    """
    General uncertainty propagation for a single evaluation point.
    
    Parameters
    ----------
    f : sympy.Expr
        Symbolic expression to propagate
    vars_ : list of sympy.Symbol
        List of variables in expression
    values : dict
        Mapping from symbols to numeric values
    sigmas : dict
        Mapping from symbols to uncertainties
    cov : sympy.Matrix or None
        Covariance matrix (optional, defaults to diagonal)
    simplify : bool
        Whether to simplify symbolic expressions
    
    Returns
    -------
    dict with keys:
        - "valor": numeric result
        - "sigma": numeric uncertainty
        - "expr_latex": LaTeX expression
        - "sigma_latex": LaTeX uncertainty expression
    """
    for v in vars_:
        if v not in values:
            raise ValueError(f"Missing value for {v}")
        if v not in sigmas:
            raise ValueError(f"Missing sigma for {v}")
        if np.any(np.asarray(sigmas[v]) < 0):
            raise ValueError(f"Negative sigma for {v}")

    # Gradient
    grad = sp.Matrix([sp.diff(f, v) for v in vars_])

    # Covariances (symbolic sigmas by default)
    sigma_symbols = {v: sp.Symbol(f"sigma_{v.name}") for v in vars_}
    if cov is None:
        Sigma = sp.diag(*[sigma_symbols[v]**2 for v in vars_])
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
        "valor": f_num(*args),
        "sigma": s_num(*args, *s_args),
        "expr_latex": sp.latex(f),
        "sigma_latex": sp.latex(sigma_f_expr),
    }


def _propagate(expr, values: dict, sigmas: dict, simplify=True):
    """
    ROBUST uncertainty propagation (no ordering errors).

    Handles both scalar and vectorial inputs automatically.

    Parameters
    ----------
    expr : sympy.Expr
        Symbolic expression
    values : dict
        Mapping {Symbol: array | scalar}
    sigmas : dict
        Mapping {Symbol: float or array}
    simplify : bool
        Whether to simplify symbolic expressions
    
    Returns
    -------
    dict with keys:
        - "valor": numeric result (scalar or array)
        - "sigma": numeric uncertainty (scalar or array)
        - "expr_latex": LaTeX expression
        - "sigma_latex": LaTeX uncertainty expression
    """
    symbols = list(expr.free_symbols)

    # Validations
    for s in symbols:
        if s not in values:
            raise ValueError(f"Missing value for {s}")
        if s not in sigmas:
            raise ValueError(f"Missing sigma for {s}")

    # Vectorized: if any input is array
    vectorial = any(np.ndim(v) > 0 for v in values.values())
    if vectorial:
        longitudes = [len(v) for v in values.values() if np.ndim(v) > 0]
        N = max(longitudes) if longitudes else 1
    else:
        N = 1

    f_vals = []
    s_vals = []

    # Main loop (i lives here)
    for i in range(N):
        vals_i = {}
        sigmas_i = {}
        for s in symbols:
            v = values[s]
            if vectorial and np.ndim(v) > 0:
                vals_i[s] = v[i]
            else:
                vals_i[s] = v
            sg = sigmas[s]
            if vectorial and np.ndim(sg) > 0:
                sigmas_i[s] = sg[i]
            else:
                sigmas_i[s] = sg

        res = uncertainty_propagation(
            expr,
            symbols,
            vals_i,
            sigmas_i,
            simplify=simplify
        )

        f_vals.append(res["valor"])
        s_vals.append(res["sigma"])

    return {
        "valor": np.array(f_vals) if vectorial else f_vals[0],
        "sigma": np.array(s_vals) if vectorial else s_vals[0],
        "expr_latex": res["expr_latex"],
        "sigma_latex": res["sigma_latex"],
    }


def propagate_quantity(target, magnitudes=None, simplify=True, compact=False, group=None, **bindings):
    """
    High-level symbolic uncertainty propagation for derived quantities.

    Parameters
    ----------
    target : dict | Quantity | str
        Target quantity to be updated. If a string is provided, it is interpreted
        as the target symbol key in the registry.
    magnitudes : dict | iterable, optional
        Source registry of quantities. Accepted forms:
        - ``{"symbol": quantity_like, ...}``
        - iterable of quantity-like objects with ``symbol``.
    simplify : bool, default True
        Whether to simplify symbolic expressions before evaluation.
    compact : bool, default False
        If True, computes compact display units for the propagated result.
        This modifies display representation only.
    group : str | None, default None
        Group propagation mode:
        - ``None``: global propagation using concatenated/group-global data.
        - ``"group_name"``: propagate only one specific group.
        - auto-inherit: if grouped inputs share the same group set, result can
          be propagated per-group and keep that structure.
    **bindings
        Direct symbol bindings (keyword name is symbol). These override entries
        from ``magnitudes`` when keys overlap.

    Returns
    -------
    Quantity
        New immutable quantity with propagated result values and uncertainty,
        plus optional ``expr_latex`` and ``sigma_latex``.

    Notes
    -----
    - ``unit_internal`` is preserved as physical identity.
    - ``compact=True`` affects only ``unit_display``.
    - Circular dependencies between symbolic quantities are detected and reported.
    """
    # 1) Normalize magnitudes and apply direct bindings (kwargs have priority)
    if magnitudes is None:
        registry = {}
    elif isinstance(magnitudes, dict):
        registry = dict(magnitudes)
    else:
        registry = {}
        for q in magnitudes:
            if not isinstance(q, (dict, Quantity)):
                raise TypeError("magnitudes must be a dict or an iterable of quantity dicts/Quantity objects")
            symbol = q.get("symbol", None)
            if symbol is None:
                raise ValueError("All magnitudes must define a non-empty 'symbol'")
            if symbol in registry:
                raise ValueError(f"Duplicate magnitude symbol: {symbol}")
            registry[symbol] = q

    for symbol, quantity_dict in bindings.items():
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Binding names must be non-empty strings")
        if not isinstance(quantity_dict, (dict, Quantity)):
            raise TypeError(
                f"Binding '{symbol}' must be a quantity dict or Quantity, got {type(quantity_dict).__name__}"
            )
        registry[symbol.strip()] = quantity_dict

    if isinstance(target, (dict, Quantity)):
        name = target.get("symbol", None)
        if isinstance(name, str):
            name = name.strip() or None

        # If target has an explicit symbol, prioritize the passed target object.
        if name:
            registry[name] = target

        # Preferred path: resolve by symbol if present in registry.
        if name in registry:
            pass
        else:
            # Fallback: resolve by object identity so users can call
            # propagate_quantity(R, registry) without relying on string symbols.
            matches = [k for k, q in registry.items() if q is target]
            if len(matches) == 1:
                name = matches[0]
            elif len(matches) > 1:
                raise ValueError(
                    "Target magnitude appears multiple times in registry"
                )
            elif name is None:
                # No symbol and not in registry: use default "_result" symbol
                name = "_result"
                registry[name] = target
    else:
        name = target

    if name not in registry:
        raise ValueError(f"Missing quantity for {name}")

    # ========== GROUP MODE DETECTION ==========
    
    # Collect group information from all quantities (excluding target)
    quantities_with_groups = []
    quantities_without_groups = []
    all_group_names = []
    all_group_orders = []
    
    for sym, q in registry.items():
        if sym == name:
            # Skip the target itself
            continue
        if isinstance(q, Quantity) and q.has_groups():
            quantities_with_groups.append(sym)
            group_order = list(q.groups)
            all_group_orders.append(group_order)
            all_group_names.append(set(group_order))
        elif isinstance(q, dict) and "_groups" in q and q["_groups"]:
            quantities_with_groups.append(sym)
            group_order = list(q["_groups"].keys())
            all_group_orders.append(group_order)
            all_group_names.append(set(group_order))
        else:
            quantities_without_groups.append(sym)
    
    # Determine group mode
    inherit_groups = False
    target_groups = None
    
    if group is not None:
        # Mode 2: Specific group requested
        # Validate that all quantities with groups have this group
        for i, sym in enumerate(quantities_with_groups):
            if group not in all_group_names[i]:
                raise ValueError(
                    f"Group '{group}' not found in quantity '{sym}'. "
                    f"Available groups: {sorted(all_group_names[i])}"
                )
        # We'll extract this specific group data
        
    elif quantities_with_groups:
        # Mode 3: Auto-inheritance if quantities with groups have identical groups.
        # This works even if some quantities don't have groups (they're treated as global).
        if len(all_group_names) > 0:
            first_groups = all_group_names[0]
            if all(g == first_groups for g in all_group_names):
                # All grouped quantities have the same groups - auto-inherit
                inherit_groups = True
                # Preserve user insertion order from the first grouped quantity
                target_groups = all_group_orders[0]
    
    # If group is specified or we're inheriting groups, we need to process per-group
    process_groups = group is not None or inherit_groups

    # 2) Symbol registry
    symbols = {k: sp.Symbol(k) for k in registry}

    cache = {}
    resolving = set()
    
    def get_measure_for_group(q, group_name):
        """
        Extract measure (value, sigma) for a specific group from quantity q.
        If q has no groups, returns its global measure.
        """
        # Check if q has groups
        if isinstance(q, Quantity) and q.has_groups():
            if group_name not in q.groups:
                # This quantity has groups but not this one - use global
                measure_to_use = q.get("measure_si", None) or q.get("measure", None)
                if measure_to_use is None:
                    # Build from global views
                    if q.value is not None and q.sigma is not None:
                        return (q.value, q.sigma)
                return measure_to_use
            else:
                # Extract specific group
                group_data = q["_groups"][group_name]
                return (group_data["value"], group_data["sigma"])
        elif isinstance(q, dict) and "_groups" in q and q["_groups"]:
            if group_name not in q["_groups"]:
                # Use global if available
                return (q.get("_value_global"), q.get("_sigma_global"))
            else:
                group_data = q["_groups"][group_name]
                return (group_data["value"], group_data["sigma"])
        else:
            # No groups - use normal measure
            return q.get("measure_si", None) or q.get("measure", None)

    def resolve_quantity(key: str, for_group=None) -> dict:
        # Cache key includes group to avoid conflicts
        cache_key = (key, for_group) if for_group else key
        
        if cache_key in cache:
            return cache[cache_key]

        if key not in registry:
            raise ValueError(f"Missing quantity for {key}")

        q = registry[key]
        expr = q.get("expr", None)
        
        # Get measure based on group mode
        if for_group is not None:
            measure_to_use = get_measure_for_group(q, for_group)
        else:
            # Global mode - use global views if available, else normal measure
            if isinstance(q, Quantity) and q.has_groups():
                measure_to_use = (q.value, q.sigma)
            elif isinstance(q, dict) and "_groups" in q and q["_groups"]:
                measure_to_use = (q.get("_value_global"), q.get("_sigma_global"))
            else:
                measure_to_use = q.get("measure_si", None) or q.get("measure", None)

        # Base quantities are identified by having no expression.
        if expr is None:
            if measure_to_use is None:
                raise ValueError(f"{key} has no measure or expression")
            val, sig = measure_to_use
            info = _Uncertainties.checker(val, sig)
            sig_out = info["sigma_vec"] if info["kind"] == "vector" else sig
            res = {
                "value": val,
                "sigma": sig_out,
                "expr_latex": None,
                "sigma_latex": None,
            }
            cache[cache_key] = res
            return res

        if cache_key in resolving:
            raise ValueError(f"Circular dependency detected at {key}")

        resolving.add(cache_key)
        expr = sp.sympify(expr, locals=symbols)
        expr_symbols = list(expr.free_symbols)

        values = {}
        sigmas = {}
        for sym in expr_symbols:
            dep = sym.name
            dep_res = resolve_quantity(dep, for_group=for_group)
            values[sym] = dep_res["value"]
            sigmas[sym] = dep_res["sigma"]

        res = _propagate(expr, values, sigmas, simplify=simplify)
        resolving.remove(cache_key)

        out = {
            "value": res["valor"],
            "sigma": res["sigma"],
            "expr_latex": res["expr_latex"],
            "sigma_latex": res["sigma_latex"],
        }
        cache[cache_key] = out

        # Don't try to cache result in registry if it's a Quantity object
        # The result will be set later in the main propagate_quantity function
        if for_group is None and isinstance(registry[key], dict):
            registry[key]["result"] = (out["value"], out["sigma"])

        return out

    # 3) Resolve target based on group mode
    
    if group is not None:
        # Mode 2: Specific group
        res = resolve_quantity(name, for_group=group)
        
    elif inherit_groups:
        # Mode 3: Auto-inherit groups - resolve each group separately
        results_by_group = {}
        for group_name in target_groups:
            group_res = resolve_quantity(name, for_group=group_name)
            results_by_group[group_name] = {
                "value": group_res["value"],
                "sigma": group_res["sigma"]
            }
        
        # Get LaTeX from any group (they're all the same expression)
        first_res = resolve_quantity(name, for_group=target_groups[0])
        res = {
            "value": None,  # Will be set by Quantity class from groups
            "sigma": None,
            "expr_latex": first_res["expr_latex"],
            "sigma_latex": first_res["sigma_latex"],
            "_groups": results_by_group
        }
        
    else:
        # Mode 1: Global
        res = resolve_quantity(name, for_group=None)

    # 4) Get target quantity from registry (may be dict or Quantity)
    target_qty = registry[name]
    
    # Convert to Quantity if it's a plain dict (for consistency)
    if not isinstance(target_qty, Quantity):
        target_qty = Quantity(target_qty)
        registry[name] = target_qty
    
    # 5) Build result based on group mode (IMMUTABLE: use _with_result)
    if inherit_groups and "_groups" in res:
        # Mode 3: Build result with inherited groups
        result_dict = {
            "symbol": target_qty.symbol,
            "expr": target_qty.expr,
            "unit_raw": target_qty.unit_raw if hasattr(target_qty, 'unit_raw') else target_qty.unit,
            "unit_internal": target_qty.unit_internal if hasattr(target_qty, 'unit_internal') else target_qty.unit,
            "unit_display": target_qty._unit_display if hasattr(target_qty, '_unit_display') else None,
            "dimension": None,
            "expr_latex": res["expr_latex"],
            "sigma_latex": res["sigma_latex"],
            "_groups": res["_groups"],
            "result": None,
        }
        target_qty = Quantity(result_dict)
        registry[name] = target_qty
        
    elif group is not None:
        # Mode 2: Single group result (immutable update)
        target_qty = target_qty._with_result(
            res["value"],
            res["sigma"],
            expr_latex=res.get("expr_latex"),
            sigma_latex=res.get("sigma_latex"),
        )
        registry[name] = target_qty
        
    else:
        # Mode 1: Global result (immutable update)
        target_qty = target_qty._with_result(
            res["value"],
            res["sigma"],
            expr_latex=res.get("expr_latex"),
            sigma_latex=res.get("sigma_latex"),
        )
        registry[name] = target_qty
    
    # 6) Apply compact units if requested (only for non-grouped results)
    # v1.0 GUARANTEE: compact affects ONLY unit_display, NOT unit_internal
    if compact and units.is_unit_conversion_available() and group is None and not inherit_groups:
        unit_str = target_qty.unit_internal if hasattr(target_qty, 'unit_internal') else target_qty.unit
        if unit_str is not None:
            value, sigma = res["value"], res["sigma"]
            if value is not None and sigma is not None:
                compact_value, compact_sigma, compact_unit = units.compact_units(
                    value, sigma, unit_str
                )
                
                # Immutable update: only change unit_display
                target_qty = target_qty._with_result(
                    value=compact_value,
                    sigma=compact_sigma,
                    expr_latex=target_qty._expr_latex,
                    sigma_latex=target_qty._sigma_latex,
                    unit_display=compact_unit if compact_unit else unit_str
                )
                registry[name] = target_qty
    
    return target_qty


def register(*magnitudes):
    """
    Build a magnitudes registry.

    Symbol resolution policy:
    - If a quantity defines a non-empty ``symbol``, it is used.
    - Otherwise, the caller-local variable name is used as fallback.

    Raises ValueError on duplicate symbols or unresolved unnamed quantities.
    
    Parameters
    ----------
    *magnitudes : dict or Quantity objects
        Quantity objects to register
    
    Returns
    -------
    dict
        Registry mapping symbols to quantity objects
        
    Examples
    --------
    >>> V = quantity(5, 0.1, "V", symbol="V")
    >>> I = quantity(0.5, 0.01, "A", symbol="I")
    >>> R = quantity("V/I", "ohm", symbol="R")
    >>> registry = register(V, I, R)
    >>> propagate_quantity(R, registry)
    """
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        raise RuntimeError("register() could not access caller frame")

    try:
        caller_locals = frame.f_back.f_locals
        registry = {}
        seen_ids = set()

        for q in magnitudes:
            if not isinstance(q, (dict, Quantity)):
                raise TypeError("register() expects magnitude dicts/Quantity objects from quantity()")

            existing_symbol = q.get("symbol", None)
            if isinstance(existing_symbol, str):
                existing_symbol = existing_symbol.strip()

            if existing_symbol:
                symbol = existing_symbol
            else:
                names = [name for name, val in caller_locals.items() if val is q]
                if len(names) == 0:
                    raise ValueError(
                        "register(): unnamed magnitude not found in caller namespace"
                    )
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

            # Set symbol on the object (if it's a dict, for backward compat)
            if isinstance(q, dict):
                q["symbol"] = symbol
            
            registry[symbol] = q

        return registry
    finally:
        del frame

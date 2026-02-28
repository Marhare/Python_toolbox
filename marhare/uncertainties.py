"""
Herramientas de incertidumbres con soporte opcional de grupos experimentales.

Concepto clave:
- Una magnitud física mantiene un único símbolo en `register`.
- Los grupos (por ejemplo "red", "blue") son subconjuntos experimentales
    de esa misma magnitud, no magnitudes nuevas.

API práctica:
- `q.value` y `q.sigma` devuelven la vista global concatenada.
- `q["red"].value` devuelve solo el subconjunto del grupo.
- `propagate_quantity(..., group="red")` propaga solo ese grupo.
"""

from __future__ import annotations
import functools
import inspect
import numpy as np

# Required dependency: uncertainties ALWAYS installed

import sympy as sp

# Unit conversion system (optional, graceful degradation if pint unavailable)
try:
    from . import unit_converter
    UNIT_CONVERSION_AVAILABLE = unit_converter.is_unit_conversion_enabled()
except ImportError:
    UNIT_CONVERSION_AVAILABLE = False
    unit_converter = None






class _Uncertainties:
    '''
    Docstring for _Uncertainties
    '''
    # --------- Construction ---------
    @staticmethod
    def checker(value, sigma):
        """
        Safety check for measurement inputs.

        - Ensures value and sigma are numeric (scalar or array-like)
        - Classifies scalar vs vector
        - Checks shape compatibility
        - Returns inherited type and shape
        """

        value = np.asarray(value, dtype=object)
        sigma = np.asarray(sigma, dtype=object)

        # --- numeric check ---
        if value.dtype.kind in ("U", "S") or sigma.dtype.kind in ("U", "S"):#What's "U" and "S"?
            raise TypeError("value or sigma is not numeric (string)")

        try:
            if not np.issubdtype(value.dtype, np.number):
                np.asarray(value, dtype=float)
            if not np.issubdtype(sigma.dtype, np.number):
                np.asarray(sigma, dtype=float)
        except Exception:
            raise TypeError("value or sigma is not numeric")

        # --- scalar vs vector ---
        value_is_vec = value.ndim >= 1 and value.shape != ()
        sigma_is_vec = sigma.ndim >= 1 and sigma.shape != ()

        # --- compatibility ---
        if not value_is_vec and sigma_is_vec:
            raise ValueError("sigma is vector but value is scalar")

        if not value_is_vec and not sigma_is_vec:
            return {"shape": None, "kind": "scalar", "sigma_vec": None}

        if value_is_vec and not sigma_is_vec:       #Transform scalar sigma to vector
            sigma_scalar = float(np.asarray(sigma, dtype=float))
            sigma_vec = np.full(value.shape, sigma_scalar, dtype=float)
            return {"shape": value.shape, "kind": "vector", "sigma_vec": sigma_vec}

        if value.shape != sigma.shape:
            raise ValueError(
                f"incompatible shapes: value={value.shape}, sigma={sigma.shape}"
            )

        sigma_vec = np.asarray(sigma, dtype=float)
        return {"shape": value.shape, "kind": "vector", "sigma_vec": sigma_vec}
    
    @staticmethod
    def quantity(*args, symbol=None, normalize=True, nan_policy="keep", groups=None, unit=None):
        """
        Unified quantity constructor.

        Accepted signatures:
        1) quantity(value, unit)
        2) quantity(value, sigma, unit)
        3) quantity(expr, unit)
        4) quantity(value, sigma, unit, expr)
        5) quantity(groups={...}, unit=..., symbol=...)

        Optional keywords:
        - symbol: str | None
        - normalize: bool (default True)
        - nan_policy: "keep" | "drop" | "raise"
        - groups: dict | None - Experimental groups structure:
                  {"red": {"value": array, "sigma": array}, "blue": {...}, ...}
                  When provided, the quantity represents multiple experimental
                  realizations of the same physical magnitude.
        - unit: str | None - Can be provided as keyword when using groups

        If nan_policy:
            - "keep": keeps NaN (default)
            - "drop": removes entries where value is NaN/inf (works with scalar or vector sigma).
                     When value is array and sigma is scalar, only invalid value
                     entries are removed and sigma remains constant for remaining values.
            - "raise": raises error if NaN/inf present
            
        Groups:
            When groups are provided, no positional args are needed. Simply use:
            quantity(groups={...}, unit="...", symbol="...")
        """

        if nan_policy not in ("keep", "drop", "raise"):
            raise ValueError("nan_policy must be 'keep', 'drop', or 'raise'")

        # ================= GROUPS MODE =================
        
        if groups is not None and len(args) == 0:
            # Pure keyword mode for groups
            if unit is None:
                raise TypeError("unit is required when using groups")
            
            # Process groups directly (skip argument parsing)
            if not isinstance(groups, dict):
                raise TypeError("groups must be a dict")
            
            if len(groups) == 0:
                raise ValueError("groups dict cannot be empty")
            
            # Validate and process each group
            processed_groups = {}
            for group_name, group_data in groups.items():
                if not isinstance(group_name, str):
                    raise TypeError(f"Group name must be string, got {type(group_name)}")
                
                if not isinstance(group_data, dict):
                    raise TypeError(f"Group '{group_name}' data must be a dict")
                
                if "value" not in group_data or "sigma" not in group_data:
                    raise ValueError(f"Group '{group_name}' must have 'value' and 'sigma' keys")
                
                g_value = np.asarray(group_data["value"], dtype=float)
                g_sigma = np.asarray(group_data["sigma"], dtype=float)
                
                # Apply nan_policy to group
                if g_value.shape != ():
                    finite_mask = np.isfinite(g_value)
                    
                    if nan_policy == "raise" and not np.all(finite_mask):
                        raise ValueError(f"Group '{group_name}' contains NaN or infinite values")
                    
                    if nan_policy == "drop":
                        g_value = g_value[finite_mask]
                        if g_sigma.ndim > 0:
                            g_sigma = g_sigma[finite_mask]
                else:
                    if nan_policy == "raise" and not np.isfinite(g_value):
                        raise ValueError(f"Group '{group_name}' contains NaN or infinite values")
                
                # Validate sigma
                if np.any(g_sigma < 0):
                    raise ValueError(f"Group '{group_name}' has negative sigma")
                
                # Ensure compatible shapes
                info = _Uncertainties.checker(g_value, g_sigma)
                sigma_out = info["sigma_vec"] if info["kind"] == "vector" else g_sigma
                
                processed_groups[group_name] = {
                    "value": g_value,
                    "sigma": sigma_out
                }
            
            # Build result dict with groups
            symbol_value = symbol
            if symbol_value is None and UNIT_CONVERSION_AVAILABLE and unit is not None:
                try:
                    symbol_value = unit_converter._converter.get_unit_symbol(unit)
                except Exception:
                    symbol_value = None
            
            result_dict = {
                "measure": None,
                "measure_si": None,
                "result": None,
                "expr": None,
                "unit": unit,
                "dimension": None,
                "symbol": symbol_value if symbol_value else unit,
                "_groups": processed_groups,
            }
            
            return Quantity(result_dict)

        # ================= ARGUMENT PARSING =================

        if len(args) == 4:
            value, sigma, unit, expr = args
            if not isinstance(expr, (str, sp.Expr, type(None))):
                raise TypeError("expr must be a string, sympy.Expr, or None")
            has_sigma = sigma is not None

        elif len(args) == 3:
            if isinstance(args[0], str):
                raise TypeError(
                    "quantity(...) with 3 args: use (value, sigma, unit)"
                )
            else:
                value, sigma, unit = args
                expr = None
                has_sigma = sigma is not None

        elif len(args) == 2:
            arg0, arg1 = args
            expr = None

            if isinstance(arg0, str):
                expr = arg0
                value = sigma = None
                unit = arg1
                has_sigma = False
            else:
                value = arg0
                sigma = None
                unit = arg1
                has_sigma = False
        else:
            raise TypeError(
                "quantity(...) expects (value, unit), (value, sigma, unit), "
                "(expr, unit), or (value, sigma, unit, expr)"
            )
        
        # If we get here with groups, it means positional args were passed
        # Groups only works in keyword-only mode (no positional args)
        if groups is not None:
            raise TypeError(
                "When using 'groups', call quantity with keywords only: "
                "quantity(groups={...}, unit='...', symbol='...')"
            )

        # ================= MEASUREMENT VALIDATION =================

        if value is not None:

            value_arr = np.asarray(value, dtype=float)

            if value_arr.shape != ():
                finite_mask = np.isfinite(value_arr)

                if nan_policy == "raise" and not np.all(finite_mask):
                    raise ValueError("value contains NaN or infinite values")

                if nan_policy == "drop":
                    value_arr = value_arr[finite_mask]
                    if has_sigma and sigma is not None:
                        sigma_arr = np.asarray(sigma, dtype=float)
                        # Only index if sigma is vectorial; keep scalar as is
                        if sigma_arr.ndim > 0:
                            sigma = sigma_arr[finite_mask]
                        else:
                            sigma = sigma_arr
            else:
                if nan_policy == "raise" and not np.isfinite(value_arr):
                    raise ValueError("value contains NaN or infinite values")

            if has_sigma and sigma is not None:
                sigma_arr = np.asarray(sigma, dtype=float)

                if np.any(sigma_arr < 0):
                    raise ValueError("sigma cannot be negative")

                if value_arr.shape != ():
                    if nan_policy == "drop":
                        sigma_arr = sigma_arr
                    else:
                        if sigma_arr.shape != value_arr.shape:
                            raise ValueError("sigma must have same shape as value")

                info = _Uncertainties.checker(value_arr, sigma_arr)
                sigma_out = info["sigma_vec"] if info["kind"] == "vector" else sigma_arr
                measure = (value_arr, sigma_out)
                dimension = info["shape"]

            else:
                info = _Uncertainties.checker(value_arr, None)
                sigma_out = np.zeros_like(value_arr, dtype=float)
                measure = (value_arr, sigma_out)
                dimension = info["shape"]

        else:
            measure = None
            dimension = None

        # ================= EXPR CHECK =================

        if expr is not None and not isinstance(expr, (str, sp.Expr)):
            raise TypeError("expr must be a string or sympy.Expr")

        # ================= UNIT CONVERSION =================

        measure_si = None
        unit_display = unit

        if UNIT_CONVERSION_AVAILABLE and measure is not None:
            try:
                value_orig, sigma_orig = measure

                value_si, sigma_si, unit_base = unit_converter._converter.normalize_value_with_uncertainty(
                    value_orig, sigma_orig, unit
                )

                if unit_base is not None:
                    measure_si = (value_si, sigma_si)
                    unit_si = unit_base
                else:
                    measure_si = measure
                    unit_si = unit

            except Exception as e:
                import warnings
                warnings.warn(
                    f"Unit conversion failed for '{unit}': {e}. Using original units.",
                    UserWarning
                )
                measure_si = measure
                unit_si = unit
        else:
            measure_si = measure
            unit_si = unit

        symbol_value = symbol
        if symbol_value is None and UNIT_CONVERSION_AVAILABLE and unit is not None:
            try:
                symbol_value = unit_converter._converter.get_unit_symbol(unit)
            except Exception:
                symbol_value = None

        if normalize and measure_si is not None:
            unit_display = unit_si
            measure = measure_si

            if UNIT_CONVERSION_AVAILABLE and symbol_value is None:
                try:
                    symbol_value = unit_converter._converter.get_unit_symbol(unit_si)
                except Exception:
                    symbol_value = None

            unit_display = symbol_value if symbol_value else unit_si
            unit = unit_display

        else:
            unit_display = unit

        return Quantity({
            "measure": measure,
            "measure_si": measure_si,
            "result": None,
            "expr": expr,
            "unit": unit,
            "dimension": dimension,
            "symbol": symbol_value if symbol_value else unit,
        })

            
    
    
    
    # --------- Uncertainty propagation ---------
    @staticmethod
    def uncertainty_propagation(
        f: sp.Expr,
        vars_: list[sp.Symbol], #What's the difference between vars_ and variables?
        values: dict[sp.Symbol, object],
        sigmas: dict[sp.Symbol, float],
        cov: sp.Matrix | None = None,
        simplify: bool = True
    ) -> dict:
        import sympy as sp
        import numpy as np

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

    @staticmethod
    def _propagate(expr, values: dict, sigmas: dict, simplify=True): #And unit??-> "symbols" is the list of variables in the expression.
        """
        ROBUST uncertainty propagation (no ordering errors).

        expr     : sympy.Expr
        values  : dict {Symbol: array | scalar}
        sigmas   : dict {Symbol: float}
        """
        import numpy as np

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

            res = _Uncertainties.uncertainty_propagation(
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
    @staticmethod
    def propagate_quantity(target, magnitudes=None, simplify=True, compact=False, group=None, **bindings):
        """
        High-level uncertainty propagation for a derived quantity.

        Parameters
        ----------
        target : dict or str
            Target quantity (with "symbol" key)
        magnitudes : dict or iterable, optional
            Dictionary or iterable of magnitude dicts
        simplify : bool, default True
            Whether to simplify symbolic expressions
        compact : bool, default False
            If True, converts result units to compact SI prefixes (e.g., 5000 mV → 5 V).
            If False, keeps units from quantity definition.
        group : str or None, default None
            Group mode for experimental subsets:
            - None (default): Global mode - use concatenated values from all groups
            - "group_name": Specific group mode - use only data from specified group
            - Auto-inheritance: If all quantities have identical groups and group=None,
              result inherits the group structure (evaluated per group)
        **bindings : dict
            Optional direct symbol bindings. Each keyword must be a symbol name present
            in expressions and each value must be a quantity dict.
            These bindings take priority over entries from magnitudes.

        Returns
        -------
        dict
            Updated quantity dictionary with added keys:
            - result : tuple (value, sigma) with propagated numeric values
            - expr_latex : str or None, LaTeX formula (None for base quantities)
            - sigma_latex : str or None, LaTeX uncertainty formula (None for base)
        
        Notes
        -----
        When compact=True, automatically applies to_compact() to result units, showing
        the most readable SI prefix (1e-9 m → 1 nm, 2.4e9 Hz → 2.4 GHz, etc.).
        
        Group modes:
        1. Global (group=None, default): Use concatenated global views
        2. Specific group (group="red"): Use only that group's data
        3. Auto-inheritance: If all quantities have same groups, result inherits structure
        
        Access symbolic expressions:
            result = propagate_quantity(R, magnitudes)
            print(result["expr_latex"])      # LaTeX formula
            print(result["sigma_latex"])     # LaTeX uncertainty formula
        """
        # 1) Normalize magnitudes and apply direct bindings (kwargs have priority)
        if magnitudes is None:
            registry = {}
        elif isinstance(magnitudes, dict):
            registry = dict(magnitudes)
        else:
            registry = {}
            for q in magnitudes:
                if not isinstance(q, dict):
                    raise TypeError("magnitudes must be a dict or an iterable of quantity dicts")
                symbol = q.get("symbol", None)
                if symbol is None:
                    raise ValueError("All magnitudes must define a non-empty 'symbol'")
                if symbol in registry:
                    raise ValueError(f"Duplicate magnitude symbol: {symbol}")
                registry[symbol] = q

        for symbol, quantity_dict in bindings.items():
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError("Binding names must be non-empty strings")
            if not isinstance(quantity_dict, dict):
                raise TypeError(
                    f"Binding '{symbol}' must be a quantity dict, got {type(quantity_dict).__name__}"
                )
            registry[symbol.strip()] = quantity_dict

        if isinstance(target, dict):
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
                    raise ValueError(
                        "Target magnitude must define a non-empty 'symbol' or be present in registry"
                    )
        else:
            name = target

        if name not in registry:
            raise ValueError(f"Missing quantity for {name}")

        # ========== GROUP MODE DETECTION ==========
        
        # Collect group information from all quantities (excluding target)
        quantities_with_groups = []
        quantities_without_groups = []
        all_group_names = []
        
        for sym, q in registry.items():
            if sym == name:
                # Skip the target itself
                continue
            if isinstance(q, Quantity) and q.has_groups():
                quantities_with_groups.append(sym)
                all_group_names.append(set(q.groups))
            elif isinstance(q, dict) and "_groups" in q and q["_groups"]:
                quantities_with_groups.append(sym)
                all_group_names.append(set(q["_groups"].keys()))
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
            
        elif quantities_with_groups and not quantities_without_groups:
            # Mode 3: Auto-inheritance only if ALL quantities have groups
            # and they all have identical groups
            if len(all_group_names) > 0:
                first_groups = all_group_names[0]
                if all(g == first_groups for g in all_group_names):
                    # All have the same groups - auto-inherit
                    inherit_groups = True
                    target_groups = sorted(first_groups)
        
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

            res = _Uncertainties._propagate(expr, values, sigmas, simplify=simplify)
            resolving.remove(cache_key)

            out = {
                "value": res["valor"],
                "sigma": res["sigma"],
                "expr_latex": res["expr_latex"],
                "sigma_latex": res["sigma_latex"],
            }
            cache[cache_key] = out

            # Cache computed numeric result without altering the definition.
            # Only cache to registry if not a group-specific calculation
            if for_group is None:
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

        # 4) Return the updated quantity dictionary with the result cached
        target_qty = registry[name]
        
        # 4.5) Add LaTeX expressions to the quantity dict
        target_qty["expr_latex"] = res.get("expr_latex", None)
        target_qty["sigma_latex"] = res.get("sigma_latex", None)
        
        # 5) Build result based on group mode
        if inherit_groups and "_groups" in res:
            # Mode 3: Build result with inherited groups
            result_dict = {
                "symbol": target_qty.get("symbol", None),
                "expr": target_qty.get("expr", None),
                "unit": target_qty.get("unit", None),
                "dimension": target_qty.get("dimension", None),
                "expr_latex": res["expr_latex"],
                "sigma_latex": res["sigma_latex"],
                "_groups": res["_groups"],
                "result": None,  # Quantity will build from groups
            }
            target_qty = Quantity(result_dict)
            registry[name] = target_qty
            
        elif group is not None:
            # Mode 2: Single group result
            target_qty["result"] = (res["value"], res["sigma"])
            
        else:
            # Mode 1: Global result
            target_qty["result"] = (res["value"], res["sigma"])
        
        # 6) Apply compact units if requested (only for non-grouped results)
        if compact and UNIT_CONVERSION_AVAILABLE and group is None and not inherit_groups:
            unit_str = target_qty.get("unit", None)
            if unit_str is not None:
                value, sigma = res["value"], res["sigma"]
                if value is not None and sigma is not None:
                    try:
                        compact_value, compact_sigma, compact_unit = unit_converter.get_compact_units(
                            value, sigma, unit_str
                        )
                        # Update the target quantity with compact units
                        target_qty["measure"] = (compact_value, compact_sigma)
                        target_qty["unit"] = compact_unit if compact_unit else unit_str
                        target_qty["result"] = (compact_value, compact_sigma)
                    except Exception as e:
                        # If compacting fails, keep original result
                        import warnings
                        warnings.warn(f"Could not apply compact units: {e}", UserWarning)
        
        # Ensure return value is always a Quantity object
        if not isinstance(target_qty, Quantity):
            target_qty = Quantity(target_qty)
        
        return target_qty


    # --------- Accessors ---------
    


class Quantity(dict):
    """
        Contenedor compatible con `dict` para magnitudes con o sin grupos.

        Reglas:
        - Si existe `_groups`, se construyen vistas globales concatenadas en
            `_value_global` y `_sigma_global`.
        - `value`/`sigma` exponen la vista global.
        - `q["nombre_grupo"]` devuelve una vista restringida a ese grupo,
            manteniendo símbolo y expresión de la magnitud original.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # If groups exist, create global views
        if "_groups" in self:
            self._build_global_views()
    
    def _build_global_views(self):
        """Build concatenated global views from groups."""
        if "_groups" not in self or not dict.__getitem__(self, "_groups"):
            return
        
        groups = dict.__getitem__(self, "_groups")
        all_values = []
        all_sigmas = []
        
        for group_name in sorted(groups.keys()):
            group_data = groups[group_name]
            all_values.append(np.asarray(group_data["value"]))
            all_sigmas.append(np.asarray(group_data["sigma"]))
        
        # Concatenate arrays
        if all_values:
            dict.__setitem__(self, "_value_global", np.concatenate(all_values))
            dict.__setitem__(self, "_sigma_global", np.concatenate(all_sigmas))
    
    @property
    def value(self):
        """
        Return the value array for this quantity.
        
        If groups exist, returns concatenated global view.
        Otherwise, returns value from measure or result.
        """
        if "_groups" in self:
            return self.get("_value_global", None)
        
        if "result" in self and self["result"] is not None:
            return self["result"][0]
        elif "measure" in self and self["measure"] is not None:
            return self["measure"][0]
        return None
    
    @property
    def sigma(self):
        """
        Return the uncertainty array for this quantity.
        
        If groups exist, returns concatenated global view.
        Otherwise, returns sigma from measure or result.
        """
        if "_groups" in self:
            return self.get("_sigma_global", None)
        
        if "result" in self and self["result"] is not None:
            return self["result"][1]
        elif "measure" in self and self["measure"] is not None:
            return self["measure"][1]
        return None
    
    def __getitem__(self, key):
        """
        Override __getitem__ to support group access.
        
        If key is a string and matches a group name, returns a view 
        restricted to that group. Otherwise behaves like normal dict.
        """
        # Check if this is a group access
        if isinstance(key, str) and "_groups" in self:
            groups = dict.__getitem__(self, "_groups")
            if key in groups:
                # Return a group view
                return self._create_group_view(key)
        
        # Standard dict access
        return super().__getitem__(key)
    
    def _create_group_view(self, group_name):
        """
        Create a restricted view for a specific group.
        
        The view is a new Quantity that shares the same symbol and expression
        but only contains data for the specified group.
        """
        if "_groups" not in self or group_name not in dict.__getitem__(self, "_groups"):
            raise KeyError(f"Group '{group_name}' not found")
        
        group_data = dict.__getitem__(self, "_groups")[group_name]
        
        # Create a new Quantity with only this group's data
        view = Quantity({
            "symbol": self.get("symbol", None),
            "expr": self.get("expr", None),
            "unit": self.get("unit", None),
            "dimension": self.get("dimension", None),
            "measure": (group_data["value"], group_data["sigma"]),
            "measure_si": (group_data["value"], group_data["sigma"]),
            "_is_group_view": True,
            "_group_name": group_name,
            "_parent": self,
        })
        
        return view
    
    @property
    def groups(self):
        """Return list of available group names."""
        if "_groups" in self:
            return list(dict.__getitem__(self, "_groups").keys())
        return []
    
    def has_groups(self):
        """Check if this quantity has experimental groups."""
        return "_groups" in self and bool(dict.__getitem__(self, "_groups"))


incertidumbres = _Uncertainties()


@functools.wraps(_Uncertainties.quantity)
def quantity(*args, symbol=None, normalize=True, nan_policy="keep", groups=None):
    """
    Constructor unificado de magnitudes con incertidumbre.

    Accepted signatures:
    1) quantity(value, unit)                -> measurement with sigma=0
    2) quantity(value, sigma, unit)         -> measurement only
    3) quantity(expr, unit)                 -> expression only
    4) quantity(value, sigma, unit, expr)   -> measurement + expression

    Keywords opcionales:
    - symbol: str | None
    - normalize: bool (default True) - If True, converts units to SI base.
                                      If False, keeps original units unchanged.
    - nan_policy: "keep" | "drop" | "raise" (default "keep")
                  Behavior when value contains NaN/inf:
                  keep = preserve data, 
                  drop = filter invalid entries (works with scalar or vector sigma),
                  raise = raise ValueError.
    - groups: dict | None
              Estructura de grupos experimentales:
              {"red": {"value": ..., "sigma": ...}, ...}
              Si se usa, la magnitud sigue siendo única (mismo `symbol`) y
              se habilita acceso global + por grupo.

    Nota importante:
    - Para magnitud con grupos, usa `quantity(None, unit, symbol=..., groups=...)`.

    Devuelve un `Quantity` (dict-like) con claves estables:
    - measure: (value, sigma) or None
    - result: (value, sigma) or None
    - expr:   None or sympy.Expr / str
    - unit:   str
    - dimension: shape tuple or None
    - symbol: str | None
    """
    return _Uncertainties.quantity(
        *args,
        symbol=symbol,
        normalize=normalize,
        nan_policy=nan_policy,
        groups=groups,
    )


@functools.wraps(_Uncertainties._propagate)
def _propagate(expr, values: dict, sigmas: dict, simplify=True):
    return _Uncertainties._propagate(expr, values, sigmas, simplify=simplify)


@functools.wraps(_Uncertainties.propagate_quantity)
def propagate_quantity(target, magnitudes=None, simplify=True, compact=False, group=None, **bindings):
    """
    Propagación de incertidumbre de alto nivel para magnitudes derivadas.
    
    Parameters
    ----------
    target : dict or str
        Target quantity (with "symbol" key)
    magnitudes : dict or iterable, optional
        Dictionary or iterable of magnitude dicts
    simplify : bool, default True
        Whether to simplify symbolic expressions
    compact : bool, default False
        If True, converts result units to compact SI prefixes (e.g., 5000 mV → 5 V).
        If False, keeps units from quantity definition.
    group : str or None, default None
        Selección de modo de grupos:
        - None: modo global (concatenado) o herencia automática
        - "group_name": usa solo ese subconjunto
    **bindings : dict
        Optional direct symbol bindings such as delta_m=rojo, alpha=alpha.
        Bindings have priority over entries in magnitudes.
    
    Modos implementados:
    1) Global (por defecto): usa `value/sigma` global concatenado.
    2) Grupo específico: `group="red"`.
    3) Herencia automática: si todas las dependencias comparten exactamente
       los mismos grupos, el resultado hereda esa estructura.
    """
    return _Uncertainties.propagate_quantity(
        target,
        magnitudes,
        simplify=simplify,
        compact=compact,
        group=group,
        **bindings,
    )


def value_quantity(q: dict):
    """
    Return numeric (value, sigma) from a quantity dict without mutation.

    If q is a list/tuple of quantity dicts, returns tuples of values and sigmas
    in the same order.

    Rule:
    - If result exists, return it
    - Else if measure exists, return it
    - Else raise ValueError
    """
    # Allow vectorized extraction from a list/tuple of quantity dicts.
    if isinstance(q, (list, tuple)):
        if len(q) == 0:
            raise ValueError("value_quantity(): empty list/tuple")
        values = []
        sigmas = []
        for i, item in enumerate(q):
            if not isinstance(item, dict):
                raise TypeError(
                    f"value_quantity(): expected dict at index {i}, got {type(item).__name__}"
                )
            value, sigma = value_quantity(item)
            values.append(value)
            sigmas.append(sigma)
        return tuple(values), tuple(sigmas)

    if not isinstance(q, dict):
        raise TypeError(
            f"value_quantity(): expected quantity dict, got {type(q).__name__}"
        )

    # Handle Quantity objects with groups
    if isinstance(q, Quantity) and q.has_groups():
        # Use properties to get global views
        value = q.value
        sigma = q.sigma
        if value is None or sigma is None:
            raise ValueError("No numeric value available in grouped quantity")
    elif q.get("result", None) is not None:
        value, sigma = q["result"]
    elif q.get("measure", None) is not None:
        value, sigma = q["measure"]
    elif "_groups" in q and q["_groups"]:
        # Plain dict with groups but not a Quantity object
        value = q.get("_value_global")
        sigma = q.get("_sigma_global")
        if value is None or sigma is None:
            raise ValueError("No numeric value available in grouped quantity dict")
    else:
        raise ValueError("No numeric value available")

    if np.any(np.asarray(sigma) < 0):
        raise ValueError("sigma cannot be negative")

    value_arr = np.asarray(value)
    sigma_arr = np.asarray(sigma)
    if value_arr.shape != () and sigma_arr.shape == ():
        sigma = np.full(value_arr.shape, float(sigma_arr), dtype=float)

    return value, sigma


def register(*magnitudes):
    """
    Build a magnitudes registry.

    Symbol resolution policy:
    - If a quantity defines a non-empty ``symbol``, it is used.
    - Otherwise, the caller-local variable name is used as fallback.

    Raises ValueError on duplicate symbols or unresolved unnamed quantities.
    """
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        raise RuntimeError("register() could not access caller frame")

    try:
        caller_locals = frame.f_back.f_locals
        registry = {}
        seen_ids = set()

        for q in magnitudes:
            if not isinstance(q, dict):
                raise TypeError("register() expects magnitude dicts from quantity()")

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

            q["symbol"] = symbol
            registry[symbol] = q

        return registry
    finally:
        del frame


@functools.wraps(_Uncertainties.uncertainty_propagation)
def uncertainty_propagation(
    f: sp.Expr,
    vars_: list[sp.Symbol],
    values: dict[sp.Symbol, object],
    sigmas: dict[sp.Symbol, float],
    cov: sp.Matrix | None = None,
    simplify: bool = True,
) -> dict:
    return _Uncertainties.uncertainty_propagation(
        f,
        vars_,
        values,
        sigmas,
        cov=cov,
        simplify=simplify,
    )

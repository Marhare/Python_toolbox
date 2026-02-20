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
    def quantity(*args, symbol=None, normalize=True):
        """
        Unified quantity constructor (positional-only).

        Accepted signatures:
        1) quantity(value, unit)              -> valor con sigma=zeros (sin incertidumbre)
        2) quantity(value, sigma, unit)       -> valor con incertidumbre
        3) quantity(expr, unit)               -> expresión solo (si expr es string)
        4) quantity(value, sigma, unit, expr) -> valor + expresión

        Optional keywords:
        - symbol: str | None
        - normalize: bool (default True) - If True, converts units to SI base.
                                          If False, keeps original units (e.g., "cm" stays "cm").

        Returns a dict with stable keys:
        - measure: (value, sigma) where sigma is always an array (never None)
        - result: (value, sigma) or None
        - expr:   None or sympy.Expr / str
        - unit:  str
        - dimension: shape tuple or None
        - symbol: str | None
        
        NOTE: If sigma is not provided, it's automatically set to zeros(value.shape)
        """

        if len(args) == 4:
            # quantity(value, sigma, unit, expr)
            value, sigma, unit, expr = args
            if not isinstance(expr, (str, sp.Expr, type(None))):
                raise TypeError("expr must be a string, sympy.Expr, or None")
            has_sigma = sigma is not None

        elif len(args) == 3:
            # Detectar: ¿es (value, sigma, unit) o (expr, unit, algo)?
            # Si args[0] es string, es expresión
            if isinstance(args[0], str):
                # quantity(expr, unit, ???) - inválido en este contexto
                raise TypeError(
                    "quantity(...) with 3 args: use (value, sigma, unit) or (value, unit)"
                )
            else:
                # quantity(value, sigma, unit)
                value, sigma, unit = args
                expr = None
                has_sigma = sigma is not None

        elif len(args) == 2:
            # quantity(value, unit) o quantity(expr, unit)
            arg0, arg1 = args
            expr = None
            
            if isinstance(arg0, str):
                # quantity(expr_str, unit)
                expr = arg0
                value = sigma = None
                unit = arg1
                has_sigma = False
            else:
                # quantity(value, unit) - sin sigma
                value = arg0
                sigma = None
                unit = arg1
                has_sigma = False

        else:
            raise TypeError(
                "quantity(...) expects (value, unit), (value, sigma, unit), "
                "(expr, unit), or (value, sigma, unit, expr)"
            )

        # Validar measurement
        if value is not None:
            # Se proporcionó value
            if has_sigma and sigma is not None:
                # Validar que sigma sea válido
                if np.any(np.asarray(sigma) < 0):
                    raise ValueError("sigma cannot be negative")
                info = _Uncertainties.checker(value, sigma)
                sigma_out = info["sigma_vec"] if info["kind"] == "vector" else sigma
                measure = (value, sigma_out)
                dimension = info["shape"]
            else:
                # Sin sigma: crear array de zeros automáticamente
                info = _Uncertainties.checker(value, None)
                value_arr = np.asarray(value)
                sigma_out = np.zeros_like(value_arr, dtype=float)
                measure = (value, sigma_out)
                dimension = info["shape"]
        else:
            measure = None
            dimension = None

        # keep expr as string or Expr (propagation will resolve symbols)
        if expr is not None and not isinstance(expr, (str, sp.Expr)):
            raise TypeError("expr must be a string or sympy.Expr")

        # ========== UNIT CONVERSION ==========
        # Strategy: always store measure_si (SI-normalized) for internal calculations
        # Then decide what to display (SI or original) based on normalize flag
        measure_si = None
        unit_display = unit  # Default: what to display
        
        # Convert to SI base units and store in measure_si
        if UNIT_CONVERSION_AVAILABLE and measure is not None:
            try:
                value_orig, sigma_orig = measure
                # Get SI-normalized version
                value_si, sigma_si, unit_base = unit_converter._converter.normalize_value_with_uncertainty(
                    value_orig, sigma_orig, unit
                )
                
                # Always store SI version for internal calculations
                if unit_base is not None:
                    measure_si = (value_si, sigma_si)
                    unit_si = unit_base
                else:
                    measure_si = measure
                    unit_si = unit
                    
            except Exception as e:
                # If conversion fails, use original as both
                import warnings
                warnings.warn(
                    f"Unit conversion failed for '{unit}': {e}. Using original units.",
                    UserWarning
                )
                measure_si = measure
                unit_si = unit
        else:
            # No conversion available: SI version is the same as original
            measure_si = measure
            unit_si = unit
        
        # Now decide what to display based on normalize flag
        if normalize and measure_si is not None:
            # normalize=True: display SI units with SI SYMBOLS (V not volt, A not ampere, etc.)
            unit_display = unit_si
            measure = measure_si
            # Convert SI unit name to symbol (volt → V, ampere → A, etc.)
            if UNIT_CONVERSION_AVAILABLE:
                try:
                    unit_display = unit_converter._converter.get_unit_symbol(unit_si) or unit_si
                except Exception:
                    # If symbol conversion fails, use full name
                    unit_display = unit_si
            unit = unit_display
        else:
            # normalize=False: display original units (keep as is)
            unit_display = unit
            # measure stays as original
            # unit stays as original

        return {
            "measure": measure,               # What to display (SI if normalize=True, else original)
            "measure_si": measure_si,         # Always SI for internal calculations
            "result": None,
            "expr": expr,
            "unit": unit,                      # Display unit
            "dimension": dimension,
            "symbol": symbol,
        }

        
    
    
    
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
    def propagate(expr, values: dict, sigmas: dict, simplify=True): #And unit??-> "symbols" is the list of variables in the expression.
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
    def propagate_quantity(target, magnitudes, simplify=True, compact=False):
        """
        High-level uncertainty propagation for a derived quantity.

        Parameters
        ----------
        target : dict or str
            Target quantity (with "symbol" key)
        magnitudes : dict or iterable
            Dictionary or iterable of magnitude dicts
        simplify : bool, default True
            Whether to simplify symbolic expressions
        compact : bool, default False
            If True, converts result units to compact SI prefixes (e.g., 5000 mV → 5 V).
            If False, keeps units from quantity definition.

        Returns:
            value
            uncertainty
            analytic expression (latex)
            analytic uncertainty expression (latex)
        
        Notes
        -----
        When compact=True, automatically applies to_compact() to result units, showing
        the most readable SI prefix (1e-9 m → 1 nm, 2.4e9 Hz → 2.4 GHz, etc.).
        """
        # 1) Normalize magnitudes and target
        if isinstance(magnitudes, dict):
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

        if isinstance(target, dict):
            name = target.get("symbol", None)
            if name is None:
                raise ValueError("Target magnitude must define a non-empty 'symbol'")
        else:
            name = target

        if name not in registry:
            raise ValueError(f"Missing quantity for {name}")

        # 2) Symbol registry
        symbols = {k: sp.Symbol(k) for k in registry}

        cache = {}
        resolving = set()

        def resolve_quantity(key: str) -> dict:
            if key in cache:
                return cache[key]

            if key not in registry:
                raise ValueError(f"Missing quantity for {key}")

            q = registry[key]
            expr = q.get("expr", None)
            # Use measure_si (SI-normalized) for calculations, fallback to measure
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
                cache[key] = res
                return res

            if key in resolving:
                raise ValueError(f"Circular dependency detected at {key}")

            resolving.add(key)
            expr = sp.sympify(expr, locals=symbols)
            expr_symbols = list(expr.free_symbols)

            values = {}
            sigmas = {}
            for sym in expr_symbols:
                dep = sym.name
                dep_res = resolve_quantity(dep)
                values[sym] = dep_res["value"]
                sigmas[sym] = dep_res["sigma"]

            res = _Uncertainties.propagate(expr, values, sigmas, simplify=simplify)
            resolving.remove(key)

            out = {
                "value": res["valor"],
                "sigma": res["sigma"],
                "expr_latex": res["expr_latex"],
                "sigma_latex": res["sigma_latex"],
            }
            cache[key] = out

            # Cache computed numeric result without altering the definition.
            registry[key]["result"] = (out["value"], out["sigma"])

            return out

        # 2) Resolve target
        res = resolve_quantity(name)

        # 3) Return the updated quantity dictionary with the result cached
        target_qty = registry[name]
        
        # 4) Apply compact units if requested
        if compact and UNIT_CONVERSION_AVAILABLE:
            unit_str = target_qty.get("unit", None)
            if unit_str is not None:
                value, sigma = res["value"], res["sigma"]
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
        
        return target_qty


    # --------- Accessors ---------
    


incertidumbres = _Uncertainties()


@functools.wraps(_Uncertainties.quantity)
def quantity(*args, symbol=None, normalize=True):
    """
    Unified quantity constructor (positional-only).

    Accepted signatures:
    1) quantity(value, sigma, unit)         -> measurement only
    2) quantity(expr, unit)                 -> expression only
    3) quantity(value, sigma, unit, expr)   -> measurement + expression

    Optional keywords:
    - symbol: str | None
    - normalize: bool (default True) - If True, converts units to SI base.
                                      If False, keeps original units unchanged.

    Returns a dict with stable keys:
    - measure: (value, sigma) or None
    - result: (value, sigma) or None
    - expr:   None or sympy.Expr / str
    - unit:   str
    - dimension: shape tuple or None
    - symbol: str | None
    """
    return _Uncertainties.quantity(*args, symbol=symbol, normalize=normalize)


@functools.wraps(_Uncertainties.propagate)
def propagate(expr, values: dict, sigmas: dict, simplify=True):
    return _Uncertainties.propagate(expr, values, sigmas, simplify=simplify)


@functools.wraps(_Uncertainties.propagate_quantity)
def propagate_quantity(target, magnitudes, simplify=True, compact=False):
    """
    High-level uncertainty propagation for a derived quantity.
    
    Parameters
    ----------
    target : dict or str
        Target quantity (with "symbol" key)
    magnitudes : dict or iterable
        Dictionary or iterable of magnitude dicts
    simplify : bool, default True
        Whether to simplify symbolic expressions
    compact : bool, default False
        If True, converts result units to compact SI prefixes (e.g., 5000 mV → 5 V).
        If False, keeps units from quantity definition.
    
    Returns
    -------
    dict
        Updated quantity dict with computed result
        
    Examples
    --------
    >>> V = quantity(5.0, 0.1, "V", symbol="V")
    >>> R = quantity(1000.0, 10.0, "ohm", symbol="R")
    >>> I = {"symbol": "I", "expr": "V/R", "unit": "A"}
    >>> result = propagate_quantity(I, [V, R])
    >>> # With compact=True, would convert to mA if result is in milliamperes
    >>> result_compact = propagate_quantity(I, [V, R], compact=True)
    """
    return _Uncertainties.propagate_quantity(target, magnitudes, simplify=simplify, compact=compact)


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

    if q.get("result", None) is not None:
        value, sigma = q["result"]
    elif q.get("measure", None) is not None:
        value, sigma = q["measure"]
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
    Build a magnitudes registry using caller-local variable names as symbols.

    Each provided object must be referenced by exactly one name in the caller's
    local namespace. Raises ValueError on missing bindings or aliasing.
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

            names = [name for name, val in caller_locals.items() if val is q]
            if len(names) == 0:
                raise ValueError("register(): magnitude not found in caller namespace")
            if len(names) > 1:
                raise ValueError(
                    "register(): magnitude has multiple names in caller namespace: "
                    + ", ".join(sorted(names))
                )

            symbol = names[0]
            if symbol in registry:
                raise ValueError(f"register(): duplicate symbol '{symbol}'")

            obj_id = id(q)
            if obj_id in seen_ids:
                raise ValueError("register(): duplicate magnitude object provided")
            seen_ids.add(obj_id)

            existing_symbol = q.get("symbol", None)
            if existing_symbol is not None and existing_symbol != symbol:
                raise ValueError(
                    f"register(): magnitude symbol mismatch for '{symbol}'"
                )
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

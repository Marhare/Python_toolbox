"""Core ``Quantity`` type and construction helpers.

This module defines:

- ``Quantity``: immutable value/sigma container with symbolic trace metadata,
- ``quantity(...)``: public factory for numeric or expression-based quantities,
- convenience helpers like ``weighted_quantity`` and value extraction.

Unit behavior depends on the ``normalize`` flag in ``quantity(...)``.
"""

import numpy as np
import sympy as sp
import warnings
from typing import Optional, Any, Dict

from . import units

class Quantity(np.lib.mixins.NDArrayOperatorsMixin):
    """
    Immutable physical magnitude with automatic analytical error propagation.
    
    Design Guarantees:
    - IMMUTABLE: Values cannot be modified after construction.
    - ANALYTICAL: NumPy operations map to SymPy expressions (CAS behavior).
    - TRACEABLE: Maintains history of base variables for `sigma_latex`.
    - FAST-MODE: Supports 'traceable=False' or auto-fallback to cut the symbolic graph.
    
    Scientific Assumptions:
    - INDEPENDENCE: Variables are strictly assumed to be statistically independent. 
    - IDENTITY: Base variables maintain symbolic identity across slices via `_var_id`.
    """
    
    __slots__ = (
        "_value", 
        "_sigma", 
        "_unit_raw",
        "_unit_internal", 
        "_unit_display", 
        "_symbol", 
        "_expr", 
        "_base_values", 
        "_base_sigmas", 
        "_expr_latex", 
        "_sigma_latex", 
        "_result_value",
        "_result_sigma",
        "_traceable",
        "_independent",
        "_var_id",        # <--- Priority 1: Strict physical identity tracking
        "_depth",         # <--- Priority 4: AST Graph depth protection
        "_initialized"
    )
    
    __array_priority__ = 20.0
    MAX_SYMBOLIC_DEPTH = 15  # Safety limit for SymPy expression trees

    def __init__(
        self, 
        value: Any, 
        sigma: Any, 
        unit: str, 
        symbol: Optional[str] = None, 
        traceable: bool = True,
        independent: bool = True,
        _expr: Optional[sp.Expr] = None,
        _base_values: Optional[Dict[sp.Symbol, np.ndarray]] = None,
        _base_sigmas: Optional[Dict[sp.Symbol, np.ndarray]] = None,
        _expr_latex: Optional[str] = None,
        _sigma_latex: Optional[str] = None,
        _var_id: Optional[str] = None,
        _depth: int = 0,
        _unit_raw: Optional[str] = None,
        _unit_display: Optional[str] = None,
        _result_value: Any = None,
        _result_sigma: Any = None,
    ):
        object.__setattr__(self, '_initialized', False)
        
        if not independent:
            raise NotImplementedError(
                "Covariance tracking (independent=False) is planned but not yet implemented."
            )
        self._independent = independent

        # 1. Array Conversion and Shape Validation
        if value is None and sigma is None:
            self._value = None
            self._sigma = None
        else:
            self._value = np.asanyarray(value, dtype=float)
            sigma_raw = np.asanyarray(sigma, dtype=float)

            if self._value.shape != sigma_raw.shape:
                try:
                    self._sigma = np.broadcast_to(sigma_raw, self._value.shape)
                except ValueError:
                    raise ValueError(f"Shape mismatch: value {self._value.shape} vs sigma {sigma_raw.shape}")
            else:
                self._sigma = sigma_raw

            if np.any(self._sigma < 0):
                raise ValueError("Uncertainty (sigma) must be non-negative.")

        # 2. Physical Identity & Tracing
        self._unit_raw = _unit_raw if _unit_raw is not None else unit
        self._unit_internal = unit
        self._unit_display = _unit_display if _unit_display is not None else unit
        self._symbol = symbol
        self._traceable = traceable
        self._depth = _depth
        self._result_value = _result_value
        self._result_sigma = _result_sigma
        
        # Priority 1: Generate or inherit a strict variable ID
        base_name = symbol if symbol else f"var_{abs(hash(id(self))) % 100000}"
        self._var_id = _var_id or f"{base_name}_{id(self)}"
        
        # 3. Analytical Traceability
        if _expr is None and self._traceable and self._value is not None:
            # Base quantities keep expr=None for propagation registry compatibility.
            self._expr = None
            clean_sym = sp.Symbol(base_name)
            self._base_values = {clean_sym: self._value}
            self._base_sigmas = {clean_sym: self._sigma}
            
            self._expr_latex = sp.latex(clean_sym)
            self._sigma_latex = sp.latex(sp.Symbol(f"sigma_{base_name}"))
            
        elif not self._traceable:
            # Graph cut/Fast mode
            self._expr = None
            self._base_values = {}
            self._base_sigmas = {}
            self._expr_latex = "\\text{Numeric}"
            self._sigma_latex = "\\text{Numeric}"
            
        else:
            # Derived variable
            self._expr = _expr
            self._base_values = _base_values or {}
            self._base_sigmas = _base_sigmas or {}
            self._expr_latex = _expr_latex
            self._sigma_latex = _sigma_latex
        
        object.__setattr__(self, '_initialized', True)

    def __setattr__(self, name, value):
        if getattr(self, '_initialized', False):
            raise AttributeError("Quantity is immutable. Reassign using mathematical operations.")
        object.__setattr__(self, name, value)

    @property
    def value(self) -> np.ndarray:
        return self._result_value if self._result_value is not None else self._value

    @property
    def sigma(self) -> np.ndarray:
        return self._result_sigma if self._result_sigma is not None else self._sigma

    @property
    def unit(self) -> str: return self._unit_display

    @property
    def symbol(self) -> Optional[str]: return self._symbol

    @property
    def expr(self) -> Optional[sp.Expr]: return self._expr

    @property
    def unit_internal(self) -> str: return self._unit_internal

    @property
    def unit_raw(self) -> Optional[str]: return self._unit_raw

    @property
    def unit_display(self) -> str: return self._unit_display

    @property
    def is_traceable(self) -> bool: return self._traceable

    @property
    def expr_latex(self) -> Optional[str]: return self._expr_latex

    @property
    def sigma_latex(self) -> Optional[str]: return self._sigma_latex

    def latex(self) -> Dict[str, str]:
        return {
            "expr_latex": self._expr_latex,
            "sigma_latex": self._sigma_latex
        }

    def uncertainty_budget(self, as_percent: bool = True) -> Dict[str, Any]:
        """Return uncertainty contributions from each base variable.

        Parameters
        ----------
        as_percent : bool, default True
            If True, include relative contributions (0-100) under
            ``"relative_contributions"``.

        Returns
        -------
        dict
            {
              "contributions": {name: (df/dx_i * sigma_i)^2},
              "total_variance": sum(contributions),
              "total_sigma": sqrt(total_variance),
              "relative_contributions": {name: percentage}  # optional
            }

        Notes
        -----
        - Works for scalar and vector quantities.
        - Requires traceable quantities with available symbolic/base metadata.
        """
        if not self._traceable:
            raise ValueError("uncertainty_budget requires traceable quantities")

        if not self._base_values or not self._base_sigmas:
            raise ValueError("uncertainty_budget requires base variable metadata")

        symbols = list(self._base_values.keys())
        values = [np.asarray(self._base_values[s], dtype=float) for s in symbols]
        sigmas = {s: np.asarray(self._base_sigmas[s], dtype=float) for s in symbols}

        expr = self._expr
        if expr is None:
            # Base quantity case: use the only tracked symbol as identity expression.
            if len(symbols) != 1:
                raise ValueError("Cannot build uncertainty budget without symbolic expression")
            expr = symbols[0]

        contributions = {}
        total_variance = None

        for s in symbols:
            dfdx = sp.diff(expr, s)
            dfdx_fn = sp.lambdify(symbols, dfdx, "numpy")
            sensitivity = np.asarray(dfdx_fn(*values), dtype=float)
            term = np.asarray((sensitivity * sigmas[s]) ** 2, dtype=float)
            contributions[s.name] = term.item() if term.shape == () else term

            if total_variance is None:
                total_variance = term
            else:
                total_variance = total_variance + term

        total_sigma = np.sqrt(total_variance)
        if np.asarray(total_variance).shape == ():
            total_variance_out = float(total_variance)
            total_sigma_out = float(total_sigma)
        else:
            total_variance_out = total_variance
            total_sigma_out = total_sigma

        out = {
            "contributions": contributions,
            "total_variance": total_variance_out,
            "total_sigma": total_sigma_out,
        }

        # Consistency diagnostic versus the quantity's numeric sigma.
        try:
            sigma_ref = np.asarray(self.sigma, dtype=float)
            sigma_calc = np.asarray(total_sigma, dtype=float)
            if sigma_ref.shape == sigma_calc.shape:
                abs_diff = np.abs(sigma_calc - sigma_ref)
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel_diff = np.where(np.abs(sigma_ref) > 0, abs_diff / np.abs(sigma_ref), 0.0)
                out["sigma_consistency"] = {
                    "max_abs_diff": float(np.max(abs_diff)),
                    "max_rel_diff": float(np.max(rel_diff)),
                }
        except Exception:
            # Non-fatal diagnostics path.
            pass

        if as_percent:
            rel = {}
            tv = np.asarray(total_variance, dtype=float)
            for name, term in contributions.items():
                t = np.asarray(term, dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    pct = np.where(tv > 0, 100.0 * t / tv, 0.0)
                rel[name] = float(pct) if np.asarray(pct).shape == () else pct
            out["relative_contributions"] = rel

        return out

    def dominant_uncertainty(self) -> Dict[str, Any]:
        """Return the dominant uncertainty source from ``uncertainty_budget``.

        Returns
        -------
        dict
            {
              "name": variable_name,
              "contribution": absolute_contribution,
              "percentage": relative_percentage,
            }

        Notes
        -----
        For vector quantities, dominance is ranked by mean percentage.
        """
        budget = self.uncertainty_budget(as_percent=True)
        rel = budget.get("relative_contributions", {})
        contrib = budget.get("contributions", {})
        if not rel:
            raise ValueError("No relative contributions available")

        def _score(v):
            arr = np.asarray(v, dtype=float)
            return float(np.mean(arr))

        name = max(rel.keys(), key=lambda k: _score(rel[k]))
        pct_val = rel[name]
        contrib_val = contrib[name]
        return {
            "name": name,
            "contribution": contrib_val,
            "percentage": pct_val,
        }

    def weighted(self, symbol: Optional[str] = None):
        """Return weighted mean as a scalar Quantity using this quantity's sigma."""
        if self.value is None or self.sigma is None:
            raise ValueError("Cannot compute weighted aggregate from expression-only quantity")

        from marhare.statistics import statistics as _statistics

        mean_val = _statistics.weighted_mean(self.value, sigma=self.sigma)
        mean_sig = _statistics.weighted_standard_error(self.value, sigma=self.sigma)
        return quantity(
            mean_val,
            mean_sig,
            self._unit_internal,
            symbol=symbol,
            normalize=False,
            nan_policy="raise",
        )

    def to(self, target_unit: str, *, symbol: Optional[str] = None, normalize: bool = False):
        """Return a new Quantity converted to ``target_unit``.

        Notes
        -----
        - Conversion is applied consistently to both value and sigma.
        - By default ``normalize=False`` to preserve the requested display unit.
        """
        if self.value is None or self.sigma is None:
            raise ValueError("Cannot convert units for expression-only quantity")

        value_conv, ok_value = units.convert_units(self.value, self._unit_internal, target_unit)
        sigma_conv, ok_sigma = units.convert_units(self.sigma, self._unit_internal, target_unit)

        if not (ok_value and ok_sigma):
            raise ValueError(
                f"Could not convert units from '{self._unit_internal}' to '{target_unit}'"
            )

        # Keep full trace metadata by default on unit conversion.
        if not normalize:
            return Quantity(
                value=value_conv,
                sigma=sigma_conv,
                unit=target_unit,
                symbol=self._symbol if symbol is None else symbol,
                traceable=self._traceable,
                _expr=self._expr,
                _base_values=dict(self._base_values),
                _base_sigmas=dict(self._base_sigmas),
                _expr_latex=self._expr_latex,
                _sigma_latex=self._sigma_latex,
                _var_id=self._var_id,
                _depth=self._depth,
                _unit_raw=self._unit_raw,
                _unit_display=target_unit,
            )

        # Preserve previous normalize=True behavior via factory path.
        return quantity(
            value_conv,
            sigma_conv,
            target_unit,
            symbol=self._symbol if symbol is None else symbol,
            normalize=True,
            nan_policy="raise",
        )

    def to_unit(self, target_unit: str, *, symbol: Optional[str] = None, normalize: bool = False):
        """Alias for ``to(...)`` kept for explicit readability in notebooks."""
        return self.to(target_unit, symbol=symbol, normalize=normalize)

    def evaluate(self, registry, simplify: bool = True, compact: bool = False, group=None, **bindings):
        from marhare.evaluation import evaluate_quantity
        return evaluate_quantity(self, registry, simplify=simplify, compact=compact, group=group, **bindings)

    def __getitem__(self, key):
        """
        Dict-like access for compatibility and numeric slicing for arrays.
        """
        if isinstance(key, str):
            if key == "measure":
                if self._value is None or self._sigma is None:
                    return None
                return (self._value, self._sigma)
            if key == "result":
                if self._result_value is None or self._result_sigma is None:
                    return None
                return (self._result_value, self._result_sigma)
            if key == "unit":
                return self.unit
            if key == "unit_internal":
                return self._unit_internal
            if key == "unit_raw":
                return self._unit_raw
            if key == "unit_display":
                return self._unit_display
            if key == "symbol":
                return self._symbol
            if key == "expr":
                return self._expr
            if key == "expr_latex":
                return self._expr_latex
            if key == "sigma_latex":
                return self._sigma_latex
            if key == "dimension":
                if self.value is None:
                    return None
                val_arr = np.asarray(self.value)
                if val_arr.ndim > 0 and val_arr.shape != ():
                    return val_arr.shape
                return None
            raise KeyError(f"'{key}'")

        # Numeric slicing path
        if self.value is None or self.sigma is None:
            raise ValueError("Cannot slice expression-only quantity without numeric measure")

        if not self._traceable:
            return Quantity(
                self.value[key], self.sigma[key], self._unit_internal,
                traceable=False, _var_id=self._var_id, _depth=self._depth,
                _unit_raw=self._unit_raw, _unit_display=self._unit_display
            )

        sliced_base_values = {k: v[key] for k, v in self._base_values.items()}
        sliced_base_sigmas = {k: v[key] for k, v in self._base_sigmas.items()}

        return Quantity(
            value=self.value[key],
            sigma=self.sigma[key],
            unit=self._unit_internal,
            symbol=self._symbol,
            traceable=True,
            _expr=self._expr,
            _base_values=sliced_base_values,
            _base_sigmas=sliced_base_sigmas,
            _expr_latex=self._expr_latex,
            _sigma_latex=self._sigma_latex,
            _var_id=self._var_id,
            _depth=self._depth,
            _unit_raw=self._unit_raw,
            _unit_display=self._unit_display,
            _result_value=self._result_value,
            _result_sigma=self._result_sigma,
        )

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False

    def keys(self):
        return [
            "measure",
            "result",
            "unit",
            "unit_raw",
            "unit_internal",
            "unit_display",
            "symbol",
            "expr",
            "dimension",
            "expr_latex",
            "sigma_latex",
        ]

    def items(self):
        return [(k, self.get(k)) for k in self.keys()]

    def values(self):
        return [self.get(k) for k in self.keys()]

    def as_dict(self) -> Dict[str, Any]:
        return {k: self.get(k) for k in self.keys()}

    # --- Unit Operations Hook (Priority 3: Decoupled) ---
    @staticmethod
    def _infer_derived_unit(ufunc, *inputs) -> str:
        """Fallback unit logic if external units module is missing or incomplete."""
        if ufunc in (np.sin, np.cos, np.tan, np.exp, np.log, np.log10):
            return "1"  
        if ufunc in (np.add, np.subtract):
            for inp in inputs:
                if isinstance(inp, Quantity): return inp._unit_internal
        if ufunc == np.multiply:
            u1 = inputs[0]._unit_internal if isinstance(inputs[0], Quantity) else "1"
            u2 = inputs[1]._unit_internal if isinstance(inputs[1], Quantity) else "1"
            if u1 == "1": return u2
            if u2 == "1": return u1
            return f"{u1}*{u2}"
        if ufunc in (np.divide, np.true_divide):
            u1 = inputs[0]._unit_internal if isinstance(inputs[0], Quantity) else "1"
            u2 = inputs[1]._unit_internal if isinstance(inputs[1], Quantity) else "1"
            if u2 == "1": return u1
            return f"{u1}/{u2}"
        if ufunc == np.power:
            u1 = inputs[0]._unit_internal if isinstance(inputs[0], Quantity) else "1"
            if isinstance(inputs[1], (int, float)):
                return f"{u1}^{inputs[1]}"
        return "derived_unit_pending"

    # --- The Analytical Engine: __array_ufunc__ ---
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != '__call__':
            return NotImplemented

        ufunc_map = {
            np.add: lambda a, b: a + b,
            np.subtract: lambda a, b: a - b,
            np.multiply: lambda a, b: a * b,
            np.divide: lambda a, b: a / b,
            np.true_divide: lambda a, b: a / b,
            np.power: lambda a, b: a ** b,
            np.square: lambda a: a ** 2,
            np.negative: lambda a: -a,
            np.sin: sp.sin,
            np.cos: sp.cos,
            np.tan: sp.tan,
            np.exp: sp.exp,
            np.log: sp.log,
            np.log10: lambda a: sp.log(a, 10),
            np.sqrt: sp.sqrt
        }

        if ufunc not in ufunc_map:
            raise NotImplementedError(f"Operation '{ufunc.__name__}' is not supported yet.")
        # Priority 2a: Dimensional Compatibility Check for Addition/Subtraction
        if ufunc in (np.add, np.subtract):
            if len(inputs) >= 2 and isinstance(inputs[0], Quantity) and isinstance(inputs[1], Quantity):
                u1 = inputs[0]._unit_internal
                u2 = inputs[1]._unit_internal
                # Check compatibility using units module
                if hasattr(units, '_converter') and hasattr(units._converter, 'check_dimensional_compatibility'):
                    if not units._converter.check_dimensional_compatibility(u1, u2):
                        msg = f"Dimensional mismatch in {ufunc.__name__}: cannot combine '{u1}' with '{u2}'. Units must be dimensionally compatible."
                        raise ValueError(msg)
        # Priority 2: Physical Validation for Transcendental Functions
        if ufunc in (np.sin, np.cos, np.tan, np.exp, np.log, np.log10):
            base_unit = inputs[0]._unit_internal if isinstance(inputs[0], Quantity) else "1"
            # Strict whitelist for dimensionless/angular arguments
            if base_unit not in ("1", "dimensionless", "unitless", "", "rad", "radian"):
                raise ValueError(
                    f"Physical violation: '{ufunc.__name__}' requires dimensionless or angular arguments, "
                    f"but received '{base_unit}'."
                )

        # Priority 4: Protect against symbolic graph explosion
        max_input_depth = max((getattr(inp, '_depth', 0) if isinstance(inp, Quantity) else 0) for inp in inputs)
        new_depth = max_input_depth + 1
        
        keep_trace = all(getattr(inp, '_traceable', True) if isinstance(inp, Quantity) else True for inp in inputs)
        
        if keep_trace and new_depth > self.MAX_SYMBOLIC_DEPTH:
            warnings.warn(
                f"Symbolic depth exceeded ({new_depth} > {self.MAX_SYMBOLIC_DEPTH}). "
                "Falling back to numeric FAST-MODE to prevent memory exhaustion.",
                UserWarning
            )
            keep_trace = False

        # Priority 3: Decoupled Unit Resolution
        unit_strs = [
            inp._unit_internal if isinstance(inp, Quantity) else str(inp) 
            for inp in inputs
        ]
        if hasattr(units, '_converter') and hasattr(units._converter, 'derive_operation_unit'):
            res_unit = units._converter.derive_operation_unit(ufunc.__name__, *unit_strs)
        else:
            res_unit = self._infer_derived_unit(ufunc, *inputs)

        # Numeric propagation without dependency on propagate_quantity/register.
        raw_vals = [i.value if isinstance(i, Quantity) else i for i in inputs]
        raw_sigs = [i.sigma if isinstance(i, Quantity) else np.zeros_like(i) for i in inputs]

        res_value = ufunc(*raw_vals, **kwargs)
        res_sigma_sq = np.zeros_like(res_value)

        for i, (val, sig) in enumerate(zip(raw_vals, raw_sigs)):
            if np.all(sig == 0):
                continue
            eps = np.abs(val) * 1e-6 + 1e-10

            args_plus = list(raw_vals)
            args_plus[i] = val + eps
            args_minus = list(raw_vals)
            args_minus[i] = val - eps

            deriv = (ufunc(*args_plus, **kwargs) - ufunc(*args_minus, **kwargs)) / (2 * eps)
            res_sigma_sq += (deriv * sig) ** 2

        expr_args = []
        merged_base_values = {}
        merged_base_sigmas = {}
        for inp in inputs:
            if isinstance(inp, Quantity):
                replace_map = {}
                inp_base_values = getattr(inp, "_base_values", {}) or {}
                inp_base_sigmas = getattr(inp, "_base_sigmas", {}) or {}

                if keep_trace:
                    for sym, val in inp_base_values.items():
                        sig = inp_base_sigmas.get(sym, None)
                        if sym in merged_base_values:
                            same_val = np.array_equal(np.asarray(merged_base_values[sym]), np.asarray(val))
                            same_sig = sig is not None and np.array_equal(
                                np.asarray(merged_base_sigmas[sym]), np.asarray(sig)
                            )
                            if same_val and same_sig:
                                continue

                            # Keep both variables by renaming the incoming symbol.
                            i = 2
                            new_sym = sp.Symbol(f"{sym.name}_{i}")
                            while new_sym in merged_base_values:
                                i += 1
                                new_sym = sp.Symbol(f"{sym.name}_{i}")

                            warnings.warn(
                                f"Symbol collision detected for '{sym.name}'. "
                                f"Auto-renaming incoming variable to '{new_sym.name}'.",
                                UserWarning,
                            )
                            replace_map[sym] = new_sym
                            merged_base_values[new_sym] = val
                            if sig is not None:
                                merged_base_sigmas[new_sym] = sig
                            continue

                        merged_base_values[sym] = val
                        if sig is not None:
                            merged_base_sigmas[sym] = sig

                if inp._expr is not None:
                    expr_arg = inp._expr
                elif getattr(inp, "_base_values", None):
                    first_sym = next(iter(inp._base_values.keys()))
                    expr_arg = first_sym
                else:
                    expr_arg = sp.Symbol(inp._var_id)

                if keep_trace and replace_map:
                    expr_arg = sp.sympify(expr_arg).subs(replace_map)

                expr_args.append(expr_arg)
            else:
                expr_args.append(sp.sympify(inp))
        new_expr = ufunc_map[ufunc](*expr_args)

        if keep_trace:
            try:
                sigma_expr = 0
                for sym in sorted(merged_base_sigmas.keys(), key=lambda s: s.name):
                    sigma_sym = sp.Symbol(f"sigma_{sym.name}")
                    sigma_expr += (sp.diff(new_expr, sym) * sigma_sym) ** 2

                expr_latex = sp.latex(new_expr)
                sigma_latex = sp.latex(sp.simplify(sp.sqrt(sigma_expr)))
            except Exception:
                expr_latex = sp.latex(new_expr)
                sigma_latex = None
        else:
            expr_latex = "\\text{Numeric}"
            sigma_latex = "\\text{Numeric}"

        return Quantity(
            value=res_value,
            sigma=np.sqrt(res_sigma_sq),
            unit=res_unit,
            traceable=keep_trace,
            _expr=new_expr if keep_trace else None,
            _base_values=merged_base_values if keep_trace else {},
            _base_sigmas=merged_base_sigmas if keep_trace else {},
            _expr_latex=expr_latex,
            _sigma_latex=sigma_latex,
            _depth=new_depth,
        )

    def __repr__(self):
        val = self.value
        sig = self.sigma
        if val is None or sig is None:
            name_str = f"'{self._symbol}' " if self._symbol else ""
            return f"<Quantity {name_str}(expr={self._expr}, unit={self._unit_internal})>"
        v_mean = np.mean(val) if np.asarray(val).ndim > 0 else val
        s_mean = np.mean(sig) if np.asarray(sig).ndim > 0 else sig
        name_str = f"'{self._symbol}' " if self._symbol else ""
        return f"<Quantity {name_str}({v_mean:.4g} ± {s_mean:.4g} {self._unit_internal})>"


class _Uncertainties:
    """Constructor/checker API for Quantity creation."""

    @staticmethod
    def _to_float_array(value, numeric_errors: str = "coerce"):
        """Best-effort numeric conversion for scalars/arrays/Series.

        - ``raise``: strict conversion, raises on non-numeric input.
        - ``coerce``: converts invalid entries to ``np.nan``.
        """
        if numeric_errors not in ("raise", "coerce"):
            raise ValueError("numeric_errors must be 'raise' or 'coerce'")

        try:
            return np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            if numeric_errors == "raise":
                raise

            # Optional pandas path for robust coercion on Series/object arrays.
            try:
                import pandas as pd
                orig = np.asarray(value, dtype=object)
                flat = pd.to_numeric(orig.ravel(), errors="coerce")
                coerced = np.asarray(flat, dtype=float).reshape(orig.shape)
                return coerced
            except Exception:
                # Fallback coercion without pandas dependency at runtime.
                orig = np.asarray(value, dtype=object)
                out = np.empty(orig.shape, dtype=float)
                it = np.nditer(orig, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
                for item in it:
                    try:
                        out[it.multi_index] = float(item.item())
                    except (TypeError, ValueError):
                        out[it.multi_index] = np.nan
                return out

    @staticmethod
    def checker(value, sigma):
        value_arr = np.asarray(value, dtype=float)
        if sigma is None:
            sigma_arr = np.zeros_like(value_arr, dtype=float)
        else:
            sigma_arr = np.asarray(sigma, dtype=float)

        if value_arr.shape != () and sigma_arr.shape == ():
            sigma_arr = np.full(value_arr.shape, float(sigma_arr), dtype=float)

        if value_arr.shape != sigma_arr.shape:
            raise ValueError(
                f"incompatible shapes: value={value_arr.shape}, sigma={sigma_arr.shape}"
            )

        if np.any(sigma_arr < 0):
            raise ValueError("sigma cannot be negative")

        kind = "vector" if value_arr.shape != () else "scalar"
        return {
            "shape": None if kind == "scalar" else value_arr.shape,
            "kind": kind,
            "sigma_vec": sigma_arr if kind == "vector" else None,
        }

    @staticmethod
    def quantity(*args, symbol=None, normalize=True, nan_policy="drop", unit=None, numeric_errors="coerce"):

        if nan_policy not in ("keep", "drop", "raise"):
            raise ValueError("nan_policy must be 'keep', 'drop', or 'raise'")

        if numeric_errors not in ("raise", "coerce"):
            raise ValueError("numeric_errors must be 'raise' or 'coerce'")

        expr = None
        value = None
        sigma = None

        if len(args) == 4:
            value, sigma, unit, expr = args
        elif len(args) == 3:
            value, sigma, unit = args
        elif len(args) == 2:
            a0, a1 = args
            if isinstance(a0, str):
                expr, unit = a0, a1
            else:
                value, unit = a0, a1
                sigma = 0.0
        elif len(args) == 1:
            if unit is None:
                raise TypeError(
                    "quantity(...) expects (value, unit), (value, sigma, unit), (expr, unit), "
                    "or (value, sigma, unit, expr)"
                )
            a0 = args[0]
            if isinstance(a0, str):
                expr = a0
            else:
                value = a0
                sigma = 0.0
        else:
            raise TypeError(
                "quantity(...) expects (value, unit), (value, sigma, unit), (expr, unit), "
                "or (value, sigma, unit, expr)"
            )

        if unit is None:
            raise TypeError("quantity(...): unit is required")

        unit_raw = unit
        unit_internal = unit

        if value is not None:
            value_arr = _Uncertainties._to_float_array(value, numeric_errors=numeric_errors)

            if nan_policy != "keep" and value_arr.shape != ():
                finite_mask = np.isfinite(value_arr)
                if nan_policy == "raise" and not np.all(finite_mask):
                    raise ValueError("value contains NaN or infinite values")
                if nan_policy == "drop":
                    value_arr = value_arr[finite_mask]
                    sigma_arr = _Uncertainties._to_float_array(sigma, numeric_errors=numeric_errors)
                    if sigma_arr.ndim > 0 and sigma_arr.shape == finite_mask.shape:
                        sigma_arr = sigma_arr[finite_mask]
                    sigma = sigma_arr

            info = _Uncertainties.checker(value_arr, sigma)
            if info["kind"] == "vector":
                sigma_arr = info["sigma_vec"]
            else:
                value_arr = float(np.asarray(value_arr, dtype=float))
                sigma_arr = float(np.asarray(sigma, dtype=float))

            if normalize and units.is_unit_conversion_enabled():
                value_si, sigma_si, base_unit = units.normalize_value_with_uncertainty(value_arr, sigma_arr, unit)
                if base_unit is not None:
                    value_arr = value_si
                    sigma_arr = sigma_si
                    unit_internal = base_unit

            q_expr = expr
            return Quantity(
                value=value_arr,
                sigma=sigma_arr,
                unit=unit_internal,
                symbol=symbol,
                traceable=True,
                _expr=q_expr,
                _unit_raw=unit_raw,
                _unit_display=unit_internal,
            )

        # Expression-only quantity
        return Quantity(
            value=None,
            sigma=None,
            unit=unit_internal,
            symbol=symbol,
            traceable=True,
            _expr=expr,
            _base_values={},
            _base_sigmas={},
            _expr_latex=None,
            _sigma_latex=None,
            _unit_raw=unit_raw,
            _unit_display=unit_internal,
        )


incertidumbres = _Uncertainties()


def quantity(*args, symbol=None, normalize=True, nan_policy="drop", unit=None, numeric_errors="coerce"):
    return _Uncertainties.quantity(
        *args,
        symbol=symbol,
        normalize=normalize,
        nan_policy=nan_policy,
        unit=unit,
        numeric_errors=numeric_errors,
    )


def value_quantity(q):
    """Return numeric ``(value, sigma)`` from a quantity-like object."""
    if hasattr(q, "value") and hasattr(q, "sigma"):
        return q.value, q.sigma
    if isinstance(q, dict):
        if q.get("result", None) is not None:
            return q["result"]
        if q.get("measure", None) is not None:
            return q["measure"]
        if "value" in q and "sigma" in q:
            return q["value"], q["sigma"]
    raise TypeError("Expected quantity-like object with value/sigma")


def weighted_quantity(q, symbol: Optional[str] = None):
    """Return weighted scalar Quantity from a quantity-like object."""
    if hasattr(q, "weighted"):
        return q.weighted(symbol=symbol)

    value, sigma = value_quantity(q)
    from marhare.statistics import statistics as _statistics

    mean_val = _statistics.weighted_mean(value, sigma=sigma)
    mean_sig = _statistics.weighted_standard_error(value, sigma=sigma)

    unit = "1"
    if hasattr(q, "unit"):
        unit = q.unit
    elif isinstance(q, dict):
        unit = q.get("unit", unit)

    return quantity(
        mean_val,
        mean_sig,
        unit,
        symbol=symbol,
        normalize=False,
        nan_policy="raise",
    )


__all__ = [
    "Quantity",
    "_Uncertainties",
    "incertidumbres",
    "quantity",
    "value_quantity",
    "weighted_quantity",
]
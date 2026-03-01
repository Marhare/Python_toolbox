"""
Quantity data model and constructor for uncertainties package.

This module defines:
- Quantity class: Immutable container for measurements with uncertainty
- quantity(): Unified constructor function
- value_quantity(): Value extraction function

Design principles:
- No imports from graphics, latex_tools, matplotlib
- Can import sympy for expression handling
- Uses units.py for unit conversion (no direct pint dependency)
- Encapsulated structure with __slots__
- Immutable: construction-only, no external mutation
- Formal unit separation: raw → internal (SI) → display (compact/preferred)
- Dict-like interface for backward compatibility

Version: 1.0 (Consolidated, immutable, unit-separated)
"""

from __future__ import annotations
import functools
import numpy as np
import sympy as sp
from typing import Tuple, Optional, Dict, Any, Union, List
from . import units


class Quantity:
    """
    Immutable magnitude (quantity) container with formal unit separation.
    
    Design guarantees (v1.0):
    - IMMUTABLE: No mutation after construction (no _set_* methods)
    - UNIT SEPARATION: _unit_raw → _unit_internal (SI) → _unit_display (compact)
    - VALIDATED: Invariants checked on construction and _with_result()
    - GROUPS: Always stored in _unit_internal (never compacted)
    
    Internal attributes (_*):
    - _symbol: str | None
    - _unit_raw: str | None (user's original unit)
    - _unit_internal: str | None (SI base unit, NEVER changes)
    - _unit_display: str | None (compact/preferred unit for display)
    - _expr: sympy.Expr | str | None (None for base quantities)
    - _measure_value, _measure_sigma: numeric (base measurement, in _unit_internal)
    - _result_value, _result_sigma: numeric | None (derived result, in _unit_internal)
    - _groups: dict[str, {"value": ..., "sigma": ...}] | None (always in _unit_internal)
    - _expr_latex, _sigma_latex: str | None (symbolic representation)
    
    Public interface:
    - value, sigma: properties that auto-select from active data layer
    - unit: returns _unit_display if set, else _unit_internal
    - symbol, expr: key properties
    - has_groups(), groups: group helpers
    - __getitem__: dict-like access + group access q["group_name"]
    - as_dict(): returns dict for legacy code
    - _with_result(): creates new Quantity with result (immutable update)
    
    Invariants (validated):
    - value.shape == sigma.shape (or sigma is scalar)
    - sigma >= 0 (all elements)
    - _unit_internal is not None (for base quantities with measure)
    - _measure_value and _measure_sigma are both set or both None
    - _result_value and _result_sigma are both set or both None
    """
    
    __slots__ = (
        "_symbol",
        "_unit_raw",
        "_unit_internal",
        "_unit_display",
        "_expr",
        "_measure_value",
        "_measure_sigma",
        "_result_value",
        "_result_sigma",
        "_groups",
        "_expr_latex",
        "_sigma_latex",
        "_initialized",  # Flag to enable immutability after __init__
    )
    
    def __init__(self, data_dict: Dict[str, Any]):
        """
        Initialize immutable Quantity from dict with validation.
        
        Required keys:
        - "symbol": str | None
        - "unit_raw": str | None (original unit from user)
        - "unit_internal": str | None (SI base unit, maintained for life)
        - "unit_display": str | None (display unit, e.g., compact)
        - "expr": sympy.Expr | str | None
        - "measure": (value, sigma) | None (in unit_internal)
        - "result": (value, sigma) | None (in unit_internal)
        - "_groups": dict | None (all values in unit_internal)
        - "expr_latex": str | None (optional)
        - "sigma_latex": str | None (optional)
        
        Backward compatibility:
        - If "unit" is provided instead of unit_internal, uses it for all three
        - If "measure_si" is provided, uses it as measure
        
        Validates:
        - value/sigma shapes match (or sigma is scalar)
        - sigma >= 0
        - unit_internal is set for base quantities with measure
        - measure and result tuple consistency
        """
        # Allow mutation during construction (disable immutability temporarily)
        object.__setattr__(self, '_initialized', False)
        # Symbol
        self._symbol = data_dict.get("symbol", None)
        
        # Unit separation with backward compatibility
        if "unit_internal" in data_dict:
            self._unit_raw = data_dict.get("unit_raw", None)
            self._unit_internal = data_dict.get("unit_internal", None)
            self._unit_display = data_dict.get("unit_display", None)
        else:
            # Backward compatibility: "unit" maps to all three
            unit_fallback = data_dict.get("unit", None)
            self._unit_raw = unit_fallback
            self._unit_internal = unit_fallback
            self._unit_display = data_dict.get("unit_display", None)
        
        # Expression
        self._expr = data_dict.get("expr", None)
        
        # Extract measure (prefer measure, fallback to measure_si for compat)
        measure = data_dict.get("measure", None)
        if measure is None:
            measure = data_dict.get("measure_si", None)
        
        if measure is not None:
            value, sigma = measure
            # Validate shapes
            value_arr = np.asarray(value)
            sigma_arr = np.asarray(sigma)
            
            # Allow sigma to be scalar for any value shape
            if value_arr.shape != () and sigma_arr.shape == ():
                sigma = np.full(value_arr.shape, float(sigma_arr), dtype=float)
                sigma_arr = sigma
            
            # Validate matching shapes
            if value_arr.shape != sigma_arr.shape:
                raise ValueError(
                    f"Quantity validation failed: value.shape={value_arr.shape} != sigma.shape={sigma_arr.shape}"
                )
            
            # Validate sigma >= 0
            if np.any(sigma_arr < 0):
                raise ValueError("Quantity validation failed: sigma must be >= 0")
            
            self._measure_value = value
            self._measure_sigma = sigma
            
            # Validate unit_internal exists for base quantities
            if self._expr is None and self._unit_internal is None:
                raise ValueError(
                    "Quantity validation failed: base quantity (no expr) must have unit_internal"
                )
        else:
            self._measure_value = None
            self._measure_sigma = None
        
        # Extract result
        result = data_dict.get("result", None)
        if result is not None:
            value, sigma = result
            # Validate shapes
            value_arr = np.asarray(value)
            sigma_arr = np.asarray(sigma)
            
            if value_arr.shape != () and sigma_arr.shape == ():
                sigma = np.full(value_arr.shape, float(sigma_arr), dtype=float)
                sigma_arr = sigma
            
            if value_arr.shape != sigma_arr.shape:
                raise ValueError(
                    f"Quantity validation failed: result value.shape={value_arr.shape} != sigma.shape={sigma_arr.shape}"
                )
            
            if np.any(sigma_arr < 0):
                raise ValueError("Quantity validation failed: result sigma must be >= 0")
            
            self._result_value = value
            self._result_sigma = sigma
        else:
            self._result_value = None
            self._result_sigma = None
        
        # Groups (always in unit_internal)
        self._groups = data_dict.get("_groups", None)
        if self._groups is not None:
            # Validate groups data
            for group_name, group_data in self._groups.items():
                if not isinstance(group_data, dict):
                    raise ValueError(f"Group '{group_name}' data must be a dict")
                if "value" not in group_data or "sigma" not in group_data:
                    raise ValueError(f"Group '{group_name}' must have 'value' and 'sigma' keys")
                
                # Validate group shapes
                g_value = np.asarray(group_data["value"])
                g_sigma = np.asarray(group_data["sigma"])
                
                if g_value.shape != () and g_sigma.shape == () and g_value.size > 0:
                    # Scalar sigma OK for vector value (will be broadcast)
                    pass
                elif g_value.shape != g_sigma.shape:
                    raise ValueError(
                        f"Group '{group_name}' validation failed: "
                        f"value.shape={g_value.shape} != sigma.shape={g_sigma.shape}"
                    )
                
                if np.any(g_sigma < 0):
                    raise ValueError(f"Group '{group_name}' validation failed: sigma must be >= 0")
            
            self._build_global_views()
        
        # LaTeX representations
        self._expr_latex = data_dict.get("expr_latex", None)
        self._sigma_latex = data_dict.get("sigma_latex", None)
        
        # Enable immutability after construction completes
        object.__setattr__(self, '_initialized', True)
    
    def __setattr__(self, name: str, value: Any) -> None:
        """
        Block attribute assignment after initialization (enforce immutability).
        
        Raises AttributeError if trying to modify after __init__ completes.
        """
        if getattr(self, '_initialized', False):
            raise AttributeError(
                f"Quantity is immutable: cannot assign to '{name}' after construction. "
                f"Use _with_result() or create a new Quantity instead."
            )
        object.__setattr__(self, name, value)
    
    def _build_global_views(self):
        """Build concatenated global views from groups (stored in measure layer)."""
        if self._groups is None or not self._groups:
            return
        
        all_values = []
        all_sigmas = []
        
        for group_name in sorted(self._groups.keys()):
            group_data = self._groups[group_name]
            all_values.append(np.asarray(group_data["value"]))
            all_sigmas.append(np.asarray(group_data["sigma"]))
        
        # Overwrite measure with concatenated views
        if all_values:
            self._measure_value = np.concatenate(all_values)
            self._measure_sigma = np.concatenate(all_sigmas)
    
    def _data_layer(self) -> tuple[str, np.ndarray, np.ndarray]:
        """
        Determine active data layer.
        
        Returns: (layer_type, value, sigma)
        where layer_type is "result" or "measure"
        """
        if self._result_value is not None:
            return ("result", self._result_value, self._result_sigma)
        elif self._measure_value is not None:
            return ("measure", self._measure_value, self._measure_sigma)
        else:
            return (None, None, None)
    
    @property
    def value(self):
        """Return value from active data layer (result if available, else measure)."""
        _, val, _ = self._data_layer()
        return val
    
    @property
    def sigma(self):
        """Return sigma from active data layer (result if available, else measure)."""
        _, _, sig = self._data_layer()
        return sig
    
    # --------- Dict-like interface ---------
    
    @property
    def unit(self) -> Optional[str]:
        """Return display unit if set, else internal unit (formal unit separation)."""
        if self._unit_display is not None:
            return self._unit_display
        return self._unit_internal
    
    @property
    def unit_internal(self) -> Optional[str]:
        """Return internal (SI base) unit (read-only, never changes after construction)."""
        return self._unit_internal
    
    @property
    def unit_raw(self) -> Optional[str]:
        """Return original raw unit from user input (read-only)."""
        return self._unit_raw
    
    @property
    def unit_display(self) -> Optional[str]:
        """Return display unit if set (compact mode), else None (read-only)."""
        return self._unit_display
    
    @property
    def symbol(self) -> Optional[str]:
        return self._symbol
    
    @property
    def expr(self) -> Optional[Union[str, sp.Expr]]:
        return self._expr
    
    def __getitem__(self, key: str):
        """
        Dict-like access with group view support.
        
        If key is a group name (and groups exist), returns a restricted view.
        Otherwise returns internal attributes for backward compatibility.
        """
        # Group access
        if isinstance(key, str) and self._groups is not None and key in self._groups:
            return self._create_group_view(key)
        
        # Dict-like attribute access (for backward compatibility)
        if key == "measure":
            if self._measure_value is not None:
                return (self._measure_value, self._measure_sigma)
            return None
        elif key == "measure_si":
            # For compat, return measure (measure_si was only an internal conversion step)
            if self._measure_value is not None:
                return (self._measure_value, self._measure_sigma)
            return None
        elif key == "result":
            if self._result_value is not None:
                return (self._result_value, self._result_sigma)
            return None
        elif key == "unit":
            # Return display unit if set, else internal (API consistency)
            return self.unit
        elif key == "unit_internal":
            return self._unit_internal
        elif key == "unit_raw":
            return self._unit_raw
        elif key == "unit_display":
            return self._unit_display
        elif key == "symbol":
            return self._symbol
        elif key == "expr":
            return self._expr
        elif key == "_groups":
            return self._groups
        elif key == "_value_global":
            # For backward compat with global views
            if self._measure_value is not None:
                return self._measure_value
            return None
        elif key == "_sigma_global":
            if self._measure_sigma is not None:
                return self._measure_sigma
            return None
        elif key == "expr_latex":
            return self._expr_latex
        elif key == "sigma_latex":
            return self._sigma_latex
        elif key == "dimension":
            # For backward compatibility
            if self._measure_value is not None:
                val_arr = np.asarray(self._measure_value)
                if val_arr.ndim > 0 and val_arr.shape != ():
                    return val_arr.shape
            return None
        else:
            raise KeyError(f"'{key}'")
    
    def get(self, key: str, default=None):
        """Dict-like get() method."""
        try:
            return self[key]
        except KeyError:
            return default
    
    def keys(self):
        """Return dict-like keys for iterating."""
        keys = ["measure", "result", "unit", "symbol", "expr", "dimension"]
        # Add unit tier keys
        if self._unit_raw is not None:
            keys.append("unit_raw")
        if self._unit_internal is not None:
            keys.append("unit_internal")
        if self._unit_display is not None:
            keys.append("unit_display")
        # Add groups and latex
        if self._groups is not None:
            keys.append("_groups")
        if self._expr_latex is not None:
            keys.append("expr_latex")
        if self._sigma_latex is not None:
            keys.append("sigma_latex")
        return keys
    
    def values(self):
        """Return dict-like values for iterating."""
        return [self[k] for k in self.keys()]
    
    def items(self):
        """Return dict-like items for iterating."""
        return [(k, self[k]) for k in self.keys()]
    
    def __contains__(self, key):
        """Support 'in' operator."""
        try:
            self[key]
            return True
        except KeyError:
            return False
    
    def __repr__(self):
        """Readable representation."""
        parts = []
        if self._symbol:
            parts.append(f"symbol={self._symbol!r}")
        if self.unit:  # Use property for display
            parts.append(f"unit={self.unit!r}")
        if self._measure_value is not None:
            parts.append(f"measure=(...)")
        if self._result_value is not None:
            parts.append(f"result=(...)")
        if self._groups is not None:
            parts.append(f"groups={list(self._groups.keys())}")
        return f"Quantity({', '.join(parts)})"
    
    # --------- Group support ---------
    
    def _create_group_view(self, group_name: str):
        """Create a restricted view for a specific group."""
        if self._groups is None or group_name not in self._groups:
            raise KeyError(f"Group '{group_name}' not found")
        
        group_data = self._groups[group_name]
        
        # Return a new Quantity viewing only this group's measure (in unit_internal)
        view_dict = {
            "symbol": self._symbol,
            "unit_raw": self._unit_raw,
            "unit_internal": self._unit_internal,
            "unit_display": self._unit_display,
            "expr": self._expr,
            "measure": (group_data["value"], group_data["sigma"]),
            "result": None,
            "_groups": None,  # Views don't have groups
            "expr_latex": self._expr_latex,
            "sigma_latex": self._sigma_latex,
        }
        
        return Quantity(view_dict)
    
    def has_groups(self) -> bool:
        """Check if this quantity has experimental groups."""
        return self._groups is not None and bool(self._groups)
    
    @property
    def groups(self) -> List[str]:
        """Return list of available group names."""
        if self._groups is not None:
            return list(self._groups.keys())
        return []
    
    # --------- Immutable updates (v1.0) ---------
    
    def _with_result(
        self, 
        value: np.ndarray, 
        sigma: np.ndarray, 
        expr_latex: Optional[str] = None, 
        sigma_latex: Optional[str] = None,
        unit_display: Optional[str] = None
    ) -> "Quantity":
        """
        Create new Quantity with result values (immutable update).
        
        This is the ONLY way to add result after construction.
        Called by propagate_quantity() to add derived values.
        
        Args:
            value: Result value array (in unit_internal)
            sigma: Result sigma array (in unit_internal)
            expr_latex: LaTeX expression (optional)
            sigma_latex: LaTeX sigma formula (optional)
            unit_display: Display unit (optional, for compact mode)
        
        Returns:
            New Quantity instance with result set
        
        Validates:
            - value/sigma shapes match
            - sigma >= 0
        """
        # Validation
        value_arr = np.asarray(value)
        sigma_arr = np.asarray(sigma)
        
        if value_arr.shape != () and sigma_arr.shape == ():
            sigma = np.full(value_arr.shape, float(sigma_arr), dtype=float)
            sigma_arr = sigma
        
        if value_arr.shape != sigma_arr.shape:
            raise ValueError(
                f"_with_result validation failed: value.shape={value_arr.shape} != sigma.shape={sigma_arr.shape}"
            )
        
        if np.any(sigma_arr < 0):
            raise ValueError("_with_result validation failed: sigma must be >= 0")
        
        # Build new dict preserving immutability
        new_dict = {
            "symbol": self._symbol,
            "unit_raw": self._unit_raw,
            "unit_internal": self._unit_internal,
            "unit_display": unit_display if unit_display is not None else self._unit_display,
            "expr": self._expr,
            "measure": (self._measure_value, self._measure_sigma) if self._measure_value is not None else None,
            "result": (value, sigma),
            "_groups": self._groups,
            "expr_latex": expr_latex if expr_latex is not None else self._expr_latex,
            "sigma_latex": sigma_latex if sigma_latex is not None else self._sigma_latex,
        }
        
        return Quantity(new_dict)
    
    def _with_groups(self, groups_dict: Dict[str, Dict[str, Any]]) -> "Quantity":
        """
        Create new Quantity with groups (immutable update).
        
        Args:
            groups_dict: dict[group_name, {"value": ..., "sigma": ...}]
                        All values MUST be in unit_internal
        
        Returns:
            New Quantity instance with groups set
        """
        new_dict = {
            "symbol": self._symbol,
            "unit_raw": self._unit_raw,
            "unit_internal": self._unit_internal,
            "unit_display": self._unit_display,
            "expr": self._expr,
            "measure": None,  # Will be built from groups
            "result": (self._result_value, self._result_sigma) if self._result_value is not None else None,
            "_groups": groups_dict,
            "expr_latex": self._expr_latex,
            "sigma_latex": self._sigma_latex,
        }
        
        return Quantity(new_dict)
    
    # --------- Legacy dict compatibility ---------
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Export as a legacy dict for compatibility with old code.
        
        Keys: "measure", "result", "unit", "symbol", "expr", "dimension",
              "unit_raw", "unit_internal", "unit_display",
              "_groups", "expr_latex", "sigma_latex"
        
        Note: "unit" returns display unit if set, else internal (backward compatibility)
        """
        result_dict = {
            "measure": (self._measure_value, self._measure_sigma) if self._measure_value is not None else None,
            "measure_si": (self._measure_value, self._measure_sigma) if self._measure_value is not None else None,
            "result": (self._result_value, self._result_sigma) if self._result_value is not None else None,
            "unit": self.unit,  # Use property (returns display or internal)
            "unit_raw": self._unit_raw,
            "unit_internal": self._unit_internal,
            "unit_display": self._unit_display,
            "symbol": self._symbol,
            "expr": self._expr,
            "dimension": None,
            "expr_latex": self._expr_latex,
            "sigma_latex": self._sigma_latex,
        }
        if self._groups is not None:
            result_dict["_groups"] = self._groups
            result_dict["_value_global"] = self._measure_value
            result_dict["_sigma_global"] = self._measure_sigma
        return result_dict


class _Uncertainties:
    """
    Internal class containing quantity construction logic.
    
    This class is kept for backward design compatibility, but all
    methods are static and could be refactored into standalone functions
    in future versions.
    """
    
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
        if value.dtype.kind in ("U", "S") or sigma.dtype.kind in ("U", "S"):
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

        if value_is_vec and not sigma_is_vec:       # Transform scalar sigma to vector
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
        - normalize: bool (default True) - If True, converts units to SI base.
                                          If False, keeps original units unchanged.
        - nan_policy: "keep" | "drop" | "raise"
        - groups: dict | None - Experimental groups structure. Each group can be:
                  Format 1 (tuple): {"red": (value, sigma), "blue": (value, sigma), ...}
                  Format 2 (dict):  {"red": {"value": array, "sigma": array}, ...}
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
            
            Groups support two formats:
            - Tuple/list: quantity(groups={"red": ([600, 605], [2, 2])})
            - Dict: quantity(groups={"red": {"value": [600, 605], "sigma": [2, 2]}})
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
            
            # Validate and process each group (v1.0: groups ALWAYS in unit_internal)
            processed_groups = {}
            
            # Normalize groups to unit_internal if requested
            unit_raw = unit
            unit_internal = unit
            unit_display = None
            
            if normalize and units.is_unit_conversion_available():
                # Try to get SI base unit for normalization
                # We'll use a dummy value to get the base unit
                try:
                    _, _, unit_base = units.normalize_value_with_uncertainty(1.0, 0.0, unit)
                    if unit_base is not None:
                        unit_internal = unit_base
                except:
                    # If normalization fails, keep original unit
                    unit_internal = unit
            
            for group_name, group_data in groups.items():
                if not isinstance(group_name, str):
                    raise TypeError(f"Group name must be string, got {type(group_name)}")
                
                # Support both tuple/list format (value, sigma) and dict format
                if isinstance(group_data, (tuple, list)):
                    if len(group_data) != 2:
                        raise ValueError(f"Group '{group_name}' tuple/list must have exactly 2 elements: (value, sigma)")
                    group_data = {"value": group_data[0], "sigma": group_data[1]}
                elif not isinstance(group_data, dict):
                    raise TypeError(f"Group '{group_name}' data must be a dict, tuple, or list (value, sigma)")
                
                if "value" not in group_data or "sigma" not in group_data:
                    raise ValueError(f"Group '{group_name}' must have 'value' and 'sigma' keys")
                
                g_value = np.asarray(group_data["value"], dtype=float)
                g_sigma = np.asarray(group_data["sigma"], dtype=float)
                
                # Normalize group data to unit_internal if requested
                if normalize and units.is_unit_conversion_available() and unit_internal != unit_raw:
                    try:
                        g_value_si, g_sigma_si, _ = units.normalize_value_with_uncertainty(
                            g_value, g_sigma, unit_raw
                        )
                        g_value = g_value_si
                        g_sigma = g_sigma_si
                    except:
                        # If normalization fails for this group, keep original
                        pass
                
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
                
                # CRITICAL: Store in unit_internal
                processed_groups[group_name] = {
                    "value": g_value,
                    "sigma": sigma_out
                }
            
            # Build result dict with groups (all in unit_internal)
            symbol_value = symbol
            if symbol_value is None and units.is_unit_conversion_available() and unit_internal is not None:
                symbol_value = units.get_unit_symbol(unit_internal)
            
            result_dict = {
                "measure": None,
                "measure_si": None,
                "result": None,
                "expr": None,
                "unit_raw": unit_raw,
                "unit_internal": unit_internal,
                "unit_display": unit_display,
                "dimension": None,
                "symbol": symbol_value if symbol_value else unit_internal,
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
                        # Allow broadcasting: scalar sigma for vector value
                        if sigma_arr.shape != () and sigma_arr.shape != value_arr.shape:
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

        # ================= UNIT CONVERSION (v1.0: formal separation) =================

        # Start with user's raw unit
        unit_raw = unit
        unit_internal = unit  # Will be SI base if normalization succeeds
        unit_display = None   # Only set if different from internal (e.g., compact)

        if units.is_unit_conversion_available() and measure is not None and normalize:
            value_orig, sigma_orig = measure
            value_si, sigma_si, unit_base = units.normalize_value_with_uncertainty(
                value_orig, sigma_orig, unit
            )

            if unit_base is not None:
                # Conversion succeeded: use SI base as internal
                measure_si = (value_si, sigma_si)
                unit_internal = unit_base
                measure = measure_si  # Store SI values
            else:
                # Conversion failed or not available: keep original
                measure_si = measure
                unit_internal = unit
        else:
            # No normalization: internal = raw
            measure_si = measure
            unit_internal = unit if normalize else unit

        symbol_value = symbol
        if symbol_value is None and units.is_unit_conversion_available() and unit is not None:
            symbol_value = units.get_unit_symbol(unit)

        if normalize and measure_si is not None and unit_internal is not None:
            # Get preferred symbol for internal unit
            if units.is_unit_conversion_available() and symbol_value is None:
                symbol_value = units.get_unit_symbol(unit_internal)
            
            # Set display unit to symbol if available, else internal
            if symbol_value and symbol_value != unit_internal:
                unit_display = symbol_value
        
        # Final symbol fallback
        if symbol_value is None:
            symbol_value = unit_internal if unit_internal else unit_raw

        return Quantity({
            "measure": measure,  # Always in unit_internal after normalization
            "measure_si": measure_si,
            "result": None,
            "expr": expr,
            "unit_raw": unit_raw,
            "unit_internal": unit_internal,
            "unit_display": unit_display,
            "dimension": dimension,
            "symbol": symbol_value,
        })


# Singleton instance for backward compatibility
incertidumbres = _Uncertainties()


@functools.wraps(_Uncertainties.quantity)
def quantity(*args, symbol=None, normalize=True, nan_policy="keep", groups=None, unit=None):
    """
    Constructor unificado de magnitudes con incertidumbre.

    Accepted signatures:
    1) quantity(value, unit)                -> measurement with sigma=0
    2) quantity(value, sigma, unit)         -> measurement only
    3) quantity(expr, unit)                 -> expression only
    4) quantity(value, sigma, unit, expr)   -> measurement + expression
    5) quantity(groups={...}, unit=..., symbol=...)  -> groups mode

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
              {"red": (value, sigma), ...} or {"red": {"value": ..., "sigma": ...}, ...}
              Si se usa, la magnitud sigue siendo única (mismo `symbol`) y
              se habilita acceso global + por grupo.
    - unit: str | None - Used with groups parameter

    Nota importante:
    - Para magnitud con grupos, usa `quantity(groups={...}, unit=..., symbol=...)`.

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
        unit=unit,
    )


def value_quantity(q: dict):
    """
    Return numeric (value, sigma) from a quantity dict/Quantity without mutation.

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
            if not isinstance(item, (dict, Quantity)):
                raise TypeError(
                    f"value_quantity(): expected dict or Quantity at index {i}, got {type(item).__name__}"
                )
            value, sigma = value_quantity(item)
            values.append(value)
            sigmas.append(sigma)
        return tuple(values), tuple(sigmas)

    if not isinstance(q, (dict, Quantity)):
        raise TypeError(
            f"value_quantity(): expected quantity dict or Quantity, got {type(q).__name__}"
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
    elif "_groups" in q and q.get("_groups"):
        # Dict or Quantity with groups but no result/measure
        if isinstance(q, Quantity):
            value = q.value
            sigma = q.sigma
        else:
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

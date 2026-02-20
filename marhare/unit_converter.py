"""
Unit conversion and normalization system for marhare quantities.

This module provides automatic unit conversion using pint as backend.
All units are normalized to SI base units internally for calculations,
while preserving the original unit strings for display and LaTeX formatting.

Features:
- Parse unit strings with SI prefixes (mV, GHz, mm^3, etc.)
- Normalize to SI base units (V, Hz, m^3)
- Convert values automatically during propagation
- Maintain original units for user-facing output
- Validate dimensional consistency
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Any
import warnings

try:
    import pint
    PINT_AVAILABLE = True
except ImportError:
    PINT_AVAILABLE = False
    warnings.warn(
        "pint is not installed. Unit conversion will be disabled. "
        "Install with: pip install pint",
        ImportWarning
    )


class UnitConverter:
    """
    Manages unit conversion and normalization for quantities.
    
    Uses pint's UnitRegistry for parsing and conversion.
    Caches parsed units for performance.
    """
    
    def __init__(self):
        if PINT_AVAILABLE:
            self.ureg = pint.UnitRegistry()
            # Configure to handle LaTeX-style units
            self.ureg.define('ohm = volt / ampere = Ω')
            self._unit_cache = {}
        else:
            self.ureg = None
            self._unit_cache = None
    
    def is_enabled(self) -> bool:
        """Check if unit conversion is available."""
        return PINT_AVAILABLE and self.ureg is not None
    
    def parse_unit(self, unit_str: str) -> Optional[pint.Unit]:
        """
        Parse a unit string into a pint Unit object.
        
        Handles common LaTeX and text formats:
        - "m/s²" or "m/s^2" → meter/second²
        - "µΩ·m" or "uohm*m" → microohm·meter
        - "GHz" → gigahertz
        
        Parameters
        ----------
        unit_str : str
            Unit string to parse
            
        Returns
        -------
        pint.Unit or None
            Parsed unit, or None if parsing fails or pint unavailable
        """
        if not self.is_enabled():
            return None
        
        if not unit_str or unit_str.strip() == "":
            return None
        
        # Check cache
        if unit_str in self._unit_cache:
            return self._unit_cache[unit_str]
        
        try:
            # Preprocess common LaTeX patterns
            processed = unit_str.replace('²', '**2').replace('³', '**3')
            processed = processed.replace('·', '*')
            processed = processed.replace('µ', 'u')  # micro prefix
            processed = processed.replace('Ω', 'ohm')
            
            # Parse with pint
            quantity = self.ureg(processed)
            unit = quantity.units
            
            # Cache for future use
            self._unit_cache[unit_str] = unit
            return unit
            
        except (pint.UndefinedUnitError, pint.DimensionalityError, AttributeError) as e:
            warnings.warn(
                f"Could not parse unit '{unit_str}': {e}. "
                f"Unit conversion will be disabled for this quantity.",
                UserWarning
            )
            return None
    
    def normalize_value(
        self, 
        value: Any, 
        unit_str: str
    ) -> Tuple[Any, Optional[str]]:
        """
        Convert value to SI base units.
        
        Parameters
        ----------
        value : scalar or array-like
            Numeric value(s) to convert
        unit_str : str
            Original unit string
            
        Returns
        -------
        normalized_value : same type as input
            Value(s) converted to SI base units
        base_unit_str : str or None
            SI base unit string, or None if conversion unavailable
            
        Examples
        --------
        >>> converter = UnitConverter()
        >>> val, unit = converter.normalize_value(5.0, "mV")
        >>> val, unit
        (0.005, "volt")
        
        >>> val, unit = converter.normalize_value(2.4, "GHz")
        >>> val, unit
        (2400000000.0, "hertz")
        """
        if not self.is_enabled():
            return value, None
        
        unit = self.parse_unit(unit_str)
        if unit is None:
            return value, None
        
        try:
            # Create quantity with original unit
            q = value * unit
            
            # Convert to base units
            q_base = q.to_base_units()
            
            # Extract numeric value and unit string
            value_base = q_base.magnitude
            unit_base_str = str(q_base.units)
            
            return value_base, unit_base_str
            
        except Exception as e:
            warnings.warn(
                f"Error converting {value} {unit_str} to base units: {e}",
                UserWarning
            )
            return value, None
    
    def normalize_value_with_uncertainty(
        self,
        value: Any,
        sigma: Any,
        unit_str: str
    ) -> Tuple[Any, Any, Optional[str]]:
        """
        Convert value AND uncertainty to SI base units using the same conversion factor.
        
        This ensures that value ± sigma always have exactly the same units.
        
        Parameters
        ----------
        value : scalar or array-like
            Numeric value(s) to convert
        sigma : scalar or array-like
            Uncertainty value(s) to convert (must have same shape as value)
        unit_str : str
            Original unit string
            
        Returns
        -------
        normalized_value : same type as input
            Value(s) converted to SI base units
        normalized_sigma : same type as input
            Uncertainty(ies) converted to SI base units with SAME factor
        base_unit_str : str or None
            SI base unit string, or None if conversion unavailable
            
        Examples
        --------
        >>> converter = UnitConverter()
        >>> val, sig, unit = converter.normalize_value_with_uncertainty(5000.0, 100.0, "mV")
        >>> val, sig, unit
        (5.0, 0.1, "volt")
        """
        if not self.is_enabled():
            return value, sigma, None
        
        unit = self.parse_unit(unit_str)
        if unit is None:
            return value, sigma, None
        
        try:
            # Create quantity with original unit for value
            q_value = value * unit
            
            # Convert value to base units
            q_value_base = q_value.to_base_units()
            
            # Extract conversion factor and unit
            value_base = q_value_base.magnitude
            unit_base_str = str(q_value_base.units)
            
            # Apply SAME conversion to sigma
            # This ensures value and sigma always have identical units
            q_sigma = sigma * unit
            q_sigma_base = q_sigma.to(q_value_base.units)  # Convert to exact same unit
            sigma_base = q_sigma_base.magnitude
            
            return value_base, sigma_base, unit_base_str
            
        except Exception as e:
            warnings.warn(
                f"Error converting {value}±{sigma} {unit_str} to base units: {e}",
                UserWarning
            )
            return value, sigma, None
    
    def convert(
        self, 
        value: Any, 
        from_unit: str, 
        to_unit: str
    ) -> Tuple[Any, bool]:
        """
        Convert value from one unit to another.
        
        Parameters
        ----------
        value : scalar or array-like
            Value(s) to convert
        from_unit : str
            Source unit
        to_unit : str
            Target unit
            
        Returns
        -------
        converted_value : same type as input
            Converted value(s)
        success : bool
            True if conversion succeeded
            
        Examples
        --------
        >>> converter = UnitConverter()
        >>> val, ok = converter.convert(1000.0, "mV", "V")
        >>> val, ok
        (1.0, True)
        """
        if not self.is_enabled():
            return value, False
        
        from_u = self.parse_unit(from_unit)
        to_u = self.parse_unit(to_unit)
        
        if from_u is None or to_u is None:
            return value, False
        
        try:
            q = value * from_u
            q_converted = q.to(to_u)
            return q_converted.magnitude, True
        except pint.DimensionalityError:
            warnings.warn(
                f"Cannot convert between incompatible units: {from_unit} → {to_unit}",
                UserWarning
            )
            return value, False
        except Exception as e:
            warnings.warn(
                f"Error during conversion {from_unit} → {to_unit}: {e}",
                UserWarning
            )
            return value, False
    
    def are_compatible(self, unit1: str, unit2: str) -> bool:
        """
        Check if two units have compatible dimensions.
        
        Parameters
        ----------
        unit1, unit2 : str
            Unit strings to compare
            
        Returns
        -------
        bool
            True if units are dimensionally compatible
            
        Examples
        --------
        >>> converter = UnitConverter()
        >>> converter.are_compatible("mV", "V")
        True
        >>> converter.are_compatible("V", "A")
        False
        """
        if not self.is_enabled():
            return True  # Can't check, assume compatible
        
        u1 = self.parse_unit(unit1)
        u2 = self.parse_unit(unit2)
        
        if u1 is None or u2 is None:
            return True  # Can't check, assume compatible
        
        try:
            # Try to convert - if it works, they're compatible
            test = (1.0 * u1).to(u2)
            return True
        except pint.DimensionalityError:
            return False
    
    def get_normalized_and_original(
        self,
        value: Any,
        sigma: Any,
        unit_str: str
    ) -> Tuple[Any, Any, Any, Optional[str]]:
        """
        Return both original and normalized versions of value/sigma.
        
        This is used when quantity() needs to decide what to store based on normalize flag.
        
        Parameters
        ----------
        value : scalar or array-like
            Original value
        sigma : scalar or array-like
            Original uncertainty  
        unit_str : str
            Unit string
            
        Returns
        -------
        value_si : same type as input
            Value converted to SI base units
        sigma_si : same type as input
            Sigma converted to SI base units (with same factor as value_si)
        base_unit_str : str or None
            SI base unit string
        
        Notes
        -----
        value and sigma returned are ALWAYS the originals (unchanged).
        The SI-normalized versions are also returned for internal calculations.
        """
        if not self.is_enabled():
            # No conversion available: all versions are the same
            return value, sigma, unit_str
        
        # Get normalized version  
        value_si, sigma_si, base_unit = self.normalize_value_with_uncertainty(value, sigma, unit_str)
        
        # Return SI version and base unit for internal use
        return value_si, sigma_si, base_unit
    
    def get_base_unit(self, unit_str: str) -> Optional[str]:
        """
        Get the SI base unit for a given unit string.
        
        Parameters
        ----------
        unit_str : str
            Unit string
            
        Returns
        -------
        str or None
            SI base unit string, or None if unavailable
            
        Examples
        --------
        >>> converter = UnitConverter()
        >>> converter.get_base_unit("mV")
        "volt"
        >>> converter.get_base_unit("mm^3")
        "meter ** 3"
        """
        if not self.is_enabled():
            return None
        
        _, base_unit = self.normalize_value(1.0, unit_str)
        return base_unit
    
    def get_unit_symbol(self, unit_str: str) -> Optional[str]:
        """
        Get the standard symbol for a unit using pint's internal definitions.
        
        Pint maintains symbol metadata for all registered units. This method
        extracts the proper symbol without needing manual mapping.
        
        Parameters
        ----------
        unit_str : str
            Unit string (e.g., "volt", "mV", "hertz", "GHz")
            
        Returns
        -------
        str or None
            The standard symbol for the unit (e.g., "V", "mV", "Hz", "GHz")
            or the original unit_str if conversion fails
            
        Examples
        --------
        >>> converter = UnitConverter()
        >>> converter.get_unit_symbol("volt")
        "V"
        >>> converter.get_unit_symbol("mV")
        "mV"  # Preserves prefixes
        >>> converter.get_unit_symbol("hertz")
        "Hz"
        """
        if not self.is_enabled():
            return unit_str  # Can't convert, return original
        
        unit = self.parse_unit(unit_str)
        if unit is None:
            return unit_str  # Can't parse, return original
        
        try:
            # Create a quantity with the parsed unit
            q = 1.0 * unit
            
            # For simple units, get symbol directly from the unit definition
            # Use pint's internal _units dictionary which has symbol metadata
            unit_str_parsed = str(q.units)
            
            # Try to get symbol from pint's unit definitions
            try:
                unit_def = self.ureg._units[unit_str_parsed]
                if hasattr(unit_def, 'symbol'):
                    return unit_def.symbol
            except (KeyError, AttributeError):
                # If not found in _units, it's a compound unit
                pass
            
            # For compound units (e.g., "m / s", "kg * m ** 2 / s ** 3"),
            # try to build the symbol from components
            if any(op in unit_str_parsed for op in [' / ', ' * ', '**']):
                # Get reduced form and try that
                q_reduced = q.to_reduced_units()
                unit_reduced_str = str(q_reduced.units)
                
                try:
                    unit_def = self.ureg._units[unit_reduced_str]
                    if hasattr(unit_def, 'symbol'):
                        return unit_def.symbol
                except (KeyError, AttributeError):
                    pass
            
            # Return the string representation as fallback
            return unit_str_parsed
            
        except Exception as e:
            warnings.warn(
                f"Error getting symbol for '{unit_str}': {e}",
                UserWarning
            )
            return unit_str  # Fall back to original
    
    def to_compact(
        self,
        value: Any,
        sigma: Any,
        unit_str: str
    ) -> Tuple[Any, Any, Optional[str]]:
        """
        Convert value and sigma to compact, human-readable units with SI prefixes.
        
        Automatically selects the best SI prefix to keep the magnitude between 1-999.
        Examples: 1e-9 m → 1 nm, 2.4e9 Hz → 2.4 GHz, 5000 mV → 5 V, 0.001 A → 1 mA
        
        This method:
        1. Parses the original unit string
        2. Creates a quantity with the original unit
        3. Uses pint's to_compact() to find the best prefix
        4. Converts sigma to the SAME compact unit
        5. Returns the compact representation
        
        Parameters
        ----------
        value : scalar or array-like
            Value to compact (can be in any SI unit)
        sigma : scalar or array-like
            Uncertainty to compact (must have same shape as value)
        unit_str : str
            Original unit string
            
        Returns
        -------
        compact_value : same type as input
            Value with optimized SI prefix
        compact_sigma : same type as input
            Sigma converted to the SAME unit as compact value
        compact_unit_str : str or None
            Compact unit string with proper SI prefix (e.g., "nanoampere"), 
            or None if conversion unavailable
            
        Examples
        --------
        >>> converter = UnitConverter()
        
        # Nanoseconds to nanoseconds (already compact)
        >>> val, sig, unit = converter.to_compact(1e-9, 1e-12, "s")
        >>> val, sig, unit
        (1.0, 0.001, "nanosecond")
        
        # Millivolts to volts (5000 mV = 5 V)
        >>> val, sig, unit = converter.to_compact(5000, 100, "mV")
        >>> val, sig, unit
        (5.0, 0.1, "volt")
        
        # Hertz to gigahertz
        >>> val, sig, unit = converter.to_compact(2400000000, 100000000, "Hz")
        >>> rounded = (round(val, 1), val, unit)
        >>> rounded
        (2.4, 2.4, "gigahertz")
        
        Notes
        -----
        - Arrays are supported but must have consistent shapes for value and sigma
        - Zero values are handled gracefully (return as-is)
        - If pint is unavailable or conversion fails, returns original values
        - The sigma is ALWAYS converted to the exact same unit as the compact value
        """
        if not self.is_enabled():
            return value, sigma, unit_str
        
        unit = self.parse_unit(unit_str)
        if unit is None:
            return value, sigma, unit_str
        
        try:
            # Create quantities with the original unit
            q_value = value * unit
            q_sigma = sigma * unit
            
            # Apply to_compact() to get best SI prefix for the value
            q_compact = q_value.to_compact()
            
            # Extract the compact unit that pint chose
            compact_unit = q_compact.units
            compact_value = q_compact.magnitude
            compact_unit_str = str(compact_unit)
            
            # Convert sigma to the EXACT SAME unit as the compact value
            # This ensures dimensional consistency: value and sigma have same units
            q_sigma_converted = q_sigma.to(compact_unit)
            compact_sigma = q_sigma_converted.magnitude
            
            return compact_value, compact_sigma, compact_unit_str
            
        except Exception as e:
            warnings.warn(
                f"Error applying to_compact for {value}±{sigma} {unit_str}: {e}. "
                f"Returning original values.",
                UserWarning
            )
            return value, sigma, unit_str


# Global converter instance
_converter = UnitConverter()


def normalize_quantity_units(value: Any, unit: str) -> Tuple[Any, Optional[str]]:
    """
    Convenience function to normalize a quantity's value and unit.
    
    DEPRECATED: Use normalize_value_with_uncertainty instead to ensure
    value and sigma are converted with the same factor.
    
    Parameters
    ----------
    value : scalar or array-like
        Value to normalize
    unit : str
        Unit string
        
    Returns
    -------
    normalized_value : same type as input
        Value in SI base units
    base_unit : str or None
        SI base unit string
    """
    return _converter.normalize_value(value, unit)


def convert_units(value: Any, from_unit: str, to_unit: str) -> Tuple[Any, bool]:
    """
    Convenience function to convert between units.
    
    Parameters
    ----------
    value : scalar or array-like
        Value to convert
    from_unit : str
        Source unit
    to_unit : str
        Target unit
        
    Returns
    -------
    converted_value : same type as input
        Converted value
    success : bool
        True if conversion succeeded
    """
    return _converter.convert(value, from_unit, to_unit)


def check_unit_compatibility(unit1: str, unit2: str) -> bool:
    """
    Check if two units are dimensionally compatible.
    
    Parameters
    ----------
    unit1, unit2 : str
        Units to compare
        
    Returns
    -------
    bool
        True if compatible
    """
    return _converter.are_compatible(unit1, unit2)


def is_unit_conversion_enabled() -> bool:
    """Check if unit conversion system is available."""
    return _converter.is_enabled()


def normalize_with_uncertainty(value: Any, sigma: Any, unit: str) -> Tuple[Any, Any, Optional[str]]:
    """
    Convenience function to normalize value and uncertainty with the same factor.
    
    This ensures value ± sigma always have identical units in the result.
    
    Parameters
    ----------
    value : scalar or array-like
        Value to normalize
    sigma : scalar or array-like
        Uncertainty to normalize (must match value shape)
    unit : str
        Unit string
        
    Returns
    -------
    normalized_value : same type as input
        Value in SI base units
    normalized_sigma : same type as input
        Uncertainty in SI base units (guaranteed same unit as value)
    base_unit : str or None
        SI base unit string
        
    Examples
    --------
    >>> val, sig, unit = normalize_with_uncertainty(5000.0, 100.0, "mV")
    >>> val, sig, unit
    (5.0, 0.1, "volt")
    """
    return _converter.normalize_value_with_uncertainty(value, sigma, unit)


def get_compact_units(value: Any, sigma: Any, unit: str) -> Tuple[Any, Any, Optional[str]]:
    """
    Convenience function to convert value and uncertainty to compact SI units.
    
    Automatically selects the best SI prefix for human-readable representation.
    
    Parameters
    ----------
    value : scalar or array-like
        Value to compact
    sigma : scalar or array-like
        Uncertainty to compact (must match value shape)
    unit : str
        Unit string (can be any valid SI unit)
        
    Returns
    -------
    compact_value : same type as input
        Value with optimized SI prefix
    compact_sigma : same type as input
        Uncertainty scaled consistently with value
    compact_unit : str or None
        Compact unit string (e.g., "nanosecond", "volt", "gigahertz")
        
    Examples
    --------
    >>> val, sig, unit = get_compact_units(1e-9, 1e-12, "s")
    >>> val, sig, unit
    (1.0, 0.001, "nanosecond")
    
    >>> val, sig, unit = get_compact_units(5000, 100, "mV")
    >>> val, sig, unit
    (5.0, 0.1, "volt")
    
    Notes
    -----
    If pint is unavailable, returns the original values unchanged.
    All exceptions are caught and warned about gracefully.
    """
    return _converter.to_compact(value, sigma, unit)

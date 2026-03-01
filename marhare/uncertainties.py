"""
Uncertainties module - Modular uncertainty propagation.

This module provides backward-compatible API while using internal modular structure:
- quantities: Quantity class and constructors  
- propagation: Symbolic propagation logic
- units: Unit conversion system

Public API remains stable and unchanged.
"""

# Re-export everything from internal modules
from .quantities import (
    # Main API
    Quantity,
    quantity,
    value_quantity,
    incertidumbres,  # Legacy singleton
    _Uncertainties,  # Internal, but exposed for advanced use
)

from .propagation import (
    propagate_quantity,
    register,
    uncertainty_propagation,
    _propagate,  # Internal but exposed
)

# Import units module for direct access
from . import units

__all__ = [
    "Quantity",
    "quantity",
    "propagate_quantity",
    "value_quantity",
    "register",
    "uncertainty_propagation",
    "_propagate",
    "incertidumbres",
    "_Uncertainties",
    "units",
]

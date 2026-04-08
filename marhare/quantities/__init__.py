"""Public computation-layer exports for quantities and propagation.

Re-exports the primary API used by end users:

- ``Quantity`` and ``quantity(...)`` constructors,
- numeric helpers (``value_quantity``, ``weighted_quantity``),
- symbolic propagation functions.
"""

from marhare.quantities.quantities import Quantity, quantity, value_quantity, weighted_quantity
from marhare.quantities.propagation import (
    propagate_quantity,
    register,
    uncertainty_propagation,
)

__all__ = [
    "Quantity",
    "quantity",
    "value_quantity",
    "weighted_quantity",
    "propagate_quantity",
    "register",
    "uncertainty_propagation",
]

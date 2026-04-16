"""Top-level public API for ``marhare``.

This module re-exports the main user-facing symbols from the internal layers:

- quantities and propagation,
- fitting,
- statistics,
- LaTeX helpers,
- symbolic Function utilities,
- Dataset container.

It also installs legacy module aliases for backwards compatibility.
"""

# marhare/__init__.py

import importlib
import sys
import types

# --- Functions ---
from .functions import (
    Function,
    D,
    I,
    dp,
)

# --- Quantities ---
from .quantities import (
    Quantity,
    quantity,
    value_quantity,
    weighted_quantity,
    propagate_quantity,
    register,
    uncertainty_propagation,
)

from .evaluation import evaluate_quantity

# --- Dataset ---
from .dataset import Dataset

 # --- LaTeX ---
from .latex import (
    valor_pm,
    tabla_latex,
    exportar,
    latex_quantity,
)

# --- Fitting ---
from .fitting import (
    fit,
    fit_quantity,
    FitResult,
    errorbar,
    plot_fit,
)

# --- Statistics ---
from .statistics import statistics as _statistics

mean = _statistics.mean
variance = _statistics.variance
standard_deviation = _statistics.standard_deviation
standard_error = _statistics.standard_error
weighted_mean = _statistics.weighted_mean
weighted_standard_error = _statistics.weighted_standard_error
weighted_variance = _statistics.weighted_variance
confidence_interval = _statistics.confidence_interval
variance_interval = _statistics.variance_interval
mean_test = _statistics.mean_test
ks_test = _statistics.ks_test


def _install_legacy_module_aliases():
    """Install v1 compatibility module aliases without legacy shim files."""
    aliases = {
        "marhare.quantities2": "marhare.quantities.quantities",
        "marhare.propagation": "marhare.quantities.propagation",
        "marhare.latex_tools": "marhare.latex.latex_tools",
    }

    for legacy_name, target_name in aliases.items():
        if legacy_name not in sys.modules:
            sys.modules[legacy_name] = importlib.import_module(target_name)

    # Special legacy facade: marhare.uncertainties
    if "marhare.uncertainties" not in sys.modules:
        qmod = importlib.import_module("marhare.quantities")
        legacy = types.ModuleType("marhare.uncertainties")
        legacy.__doc__ = "Deprecated v1 compatibility facade. Use marhare.quantities."
        legacy.__package__ = "marhare"

        for name in getattr(qmod, "__all__", []):
            setattr(legacy, name, getattr(qmod, name))

        legacy.evaluate_quantity = evaluate_quantity
        legacy.__all__ = [*getattr(qmod, "__all__", []), "evaluate_quantity"]
        sys.modules["marhare.uncertainties"] = legacy


_install_legacy_module_aliases()


__all__ = [
    "Function",
    "D",
    "I",
    "dp",
    "Quantity",
    "quantity",
    "value_quantity",
    "weighted_quantity",
    "evaluate_quantity",
    "propagate_quantity",
    "register",
    "uncertainty_propagation",
    "valor_pm",
    "tabla_latex",
    "exportar",
    "latex_quantity",
    "fit",
    "fit_quantity",
    "FitResult",
    "errorbar",
    "plot_fit",
    "mean",
    "variance",
    "standard_deviation",
    "standard_error",
    "weighted_mean",
    "weighted_standard_error",
    "weighted_variance",
    "confidence_interval",
    "variance_interval",
    "mean_test",
    "ks_test",

]
# marhare/__init__.py

# --- Functions ---
from .functions import (
    Function,
    D,
    I,
    dp,
)

# --- Uncertainties ---
from .uncertainties import (
    quantity,
    propagate,
    propagate_quantity,
    register,
    uncertainty_propagation,
    value_quantity,
)

 # --- LaTeX ---
from .latex_tools import (
    valor_pm,
    latex_quantity,
)

# --- Fitting ---
from .fitting import (
    fit_quantity,
)

# --- Statistics ---
from .statistics import statistics as _statistics

# --- Graphics ---
from .graphics import graphics, plot, Figure

mean = _statistics.mean
variance = _statistics.variance
standard_deviation = _statistics.standard_deviation
standard_error = _statistics.standard_error
weighted_mean = _statistics.weighted_mean
weighted_variance = _statistics.weighted_variance
confidence_interval = _statistics.confidence_interval
variance_interval = _statistics.variance_interval
mean_test = _statistics.mean_test
ks_test = _statistics.ks_test


__all__ = [
    "Function",
    "D",
    "I",
    "dp",
    "quantity",
    "propagate",
    "propagate_quantity",
    "register",
    "uncertainty_propagation",
    "value_quantity",
    "valor_pm",
    "latex_quantity",
    "fit_quantity",
    "mean",
    "variance",
    "standard_deviation",
    "standard_error",
    "weighted_mean",
    "weighted_variance",
    "confidence_interval",
    "variance_interval",
    "mean_test",
    "ks_test",
    "graphics",
    "plot",
    "Figure",

]
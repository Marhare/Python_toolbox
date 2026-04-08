"""Curve fitting tools for quantity-based workflows.

This module provides weighted least-squares fitting and helper wrappers used by
the public API:

- ``fit_quantity``: fit directly from Quantity-like inputs (value + sigma).
- ``fit``: generic fit from arrays/tuples/Quantity-like inputs.
- ``FitResult``: convenience wrapper around the raw fit dictionary.

Core assumptions:

- ``sy`` is treated as absolute uncertainty in ``y``.
- Residual diagnostics are reported as ``chi2``, ``chi2_red``, and ``p``.
- Statistical validity depends on user-side assumptions (model form, residual
  independence, realistic uncertainties).

Notes on units:

- This module fits numeric arrays only.
- If your input quantities were created with ``normalize=True`` (default),
  fitting runs on SI-normalized values.
- Parameter units should therefore be assigned consistently when using
  ``FitResult.parameter_quantity(...)``.
"""

import numpy as np
from scipy import stats, optimize


def _extract_value_sigma(obj):
    if hasattr(obj, "value") and hasattr(obj, "sigma"):
        return np.asarray(obj.value, dtype=float), np.asarray(obj.sigma, dtype=float)
    if isinstance(obj, dict):
        if obj.get("result", None) is not None:
            return np.asarray(obj["result"][0], dtype=float), np.asarray(obj["result"][1], dtype=float)
        if obj.get("measure", None) is not None:
            return np.asarray(obj["measure"][0], dtype=float), np.asarray(obj["measure"][1], dtype=float)
        if "value" in obj and "sigma" in obj:
            return np.asarray(obj["value"], dtype=float), np.asarray(obj["sigma"], dtype=float)
    val = np.asarray(obj, dtype=float)
    return val, np.zeros_like(val)

class _Fitting:

    @staticmethod
    def _validate_data(x, y, sy=None):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
        if x.ndim != 1:
            raise ValueError("x and y must be 1D arrays")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError("x and y must contain only finite values")
        if sy is None:
            sy = np.ones_like(y)
        else:
            sy = np.asarray(sy, dtype=float)
            if sy.ndim == 0:
                sy = np.full_like(y, float(sy), dtype=float)
            if sy.shape != y.shape:
                raise ValueError("sy must have the same shape as y")
            if np.any(sy <= 0):
                raise ValueError("sy must be positive at all points")
            if not np.all(np.isfinite(sy)):
                raise ValueError("sy must contain only finite values")
        return x, y, sy

    @staticmethod
    def _safe_fit_stats(chi2, ndof):
        chi2 = float(chi2)
        ndof = int(ndof)
        if ndof > 0:
            return float(chi2 / ndof), float(stats.chi2.sf(chi2, ndof))
        return float("nan"), float("nan")

    @staticmethod
    def _curve_fit(f, x, y, sy=None, p0=None):
        x, y, sy = _Fitting._validate_data(x, y, sy)

        popt, pcov = optimize.curve_fit(
            f, x, y, sigma=sy, absolute_sigma=True, p0=p0
        )
        perr = np.sqrt(np.diag(pcov))
        yfit = f(x, *popt)
        chi2 = np.sum(((y - yfit) / sy)**2)
        ndof = len(x) - len(popt)
        chi2_red, p = _Fitting._safe_fit_stats(chi2, ndof)

        return {
            "parameters": popt,
            "errors": perr,
            "covariance": pcov,
            "yfit": yfit,
            "chi2": chi2,
            "ndof": ndof,
            "chi2_red": chi2_red,
            "p": p,
        }

    # ---------- Linear ----------
    @staticmethod
    def linear_fit(x, y, sy=None):
        """Analytic weighted linear fit for ``y = a + b*x``.

        Parameters
        ----------
        x, y : array_like
            1D arrays with matching shape.
        sy : array_like | float | None
            Absolute uncertainty in ``y``. If ``None``, uses ones.

        Returns
        -------
        dict
            Keys: ``parameters`` (``{"a", "b"}``), ``errors`` (``{"sa", "sb"}``),
            ``covariance``, ``yfit``, ``chi2``, ``ndof``, ``chi2_red``, ``p``.
        """
        x, y, sy = _Fitting._validate_data(x, y, sy)
        w = 1 / sy**2

        S = np.sum(w)
        Sx = np.sum(w * x)
        Sy = np.sum(w * y)
        Sxx = np.sum(w * x * x)
        Sxy = np.sum(w * x * y)

        denom = S * Sxx - Sx**2
        a = (Sxx * Sy - Sx * Sxy) / denom
        b = (S * Sxy - Sx * Sy) / denom

        # Uncertainties (covariance) when absolute sigma is known
        # Parameter variances in weighted linear fit:
        # var(a) = Sxx / denom, var(b) = S / denom, cov(a,b) = -Sx / denom
        var_a = Sxx / denom
        var_b = S / denom
        cov_ab = -Sx / denom
        sa = float(np.sqrt(var_a))
        sb = float(np.sqrt(var_b))
        cov = np.array([[var_a, cov_ab], [cov_ab, var_b]], dtype=float)

        yfit = a + b * x
        chi2 = np.sum(((y - yfit) / sy)**2)
        ndof = len(x) - 2
        chi2_red, p = _Fitting._safe_fit_stats(chi2, ndof)

        return {
            "parameters": {"a": float(a), "b": float(b)},
            "errors": {"sa": sa, "sb": sb},
            "covariance": cov,
            "chi2": float(chi2),
            "ndof": ndof,
            "chi2_red": chi2_red,
            "p": float(p),
            "yfit": yfit,
        }

    # ---------- Polynomial ----------
    @staticmethod
    def polynomial_fit(x, y, degree, sy=None):
        """Weighted polynomial fit using ``numpy.polyfit``.

        Parameters
        ----------
        x, y : array_like
            1D arrays with matching shape.
        degree : int
            Polynomial degree.
        sy : array_like | float | None
            Absolute uncertainty in ``y``. If ``None``, uses ones.

        Returns
        -------
        dict
            Keys: ``parameters`` (coefficients in descending order), ``errors``,
            ``covariance``, ``yfit``, ``chi2``, ``ndof``, ``chi2_red``, ``p``.
        """
        x, y, sy = _Fitting._validate_data(x, y, sy)
        coef, cov = np.polyfit(x, y, degree, w=1 / sy, cov="unscaled")
        errors_arr = np.sqrt(np.diag(cov))
        yfit = np.polyval(coef, x)
        chi2 = np.sum(((y - yfit) / sy)**2)
        ndof = len(x) - (degree + 1)
        chi2_red, p = _Fitting._safe_fit_stats(chi2, ndof)
        return {
            "parameters": coef,
            "errors": errors_arr,
            "covariance": cov,
            "yfit": yfit,
            "chi2": chi2,
            "ndof": ndof,
            "chi2_red": chi2_red,
            "p": p,
        }

    # ---------- Unified ----------
    @staticmethod
    def fit(model, x, y, sy=None, p0=None, *, variable="x"):
        """Generic weighted fit for callable models.

        Parameters
        ----------
        model : callable
            Callable with signature ``f(x, *params)``.
        x, y : array_like
            1D arrays with matching shape.
        sy : array_like | float | None
            Absolute uncertainty in ``y``. If ``None``, uses ones.
        p0 : sequence | None
            Optional initial parameter guess.
        variable : str
            Reserved compatibility argument (currently ignored).

        Returns
        -------
        dict
            Keys: ``parameters``, ``errors``, ``covariance``, ``yfit``, ``chi2``,
            ``ndof``, ``chi2_red``, ``p``.
        """
        if not callable(model):
            raise TypeError("model must be callable")
        return _Fitting._curve_fit(model, x, y, sy, p0)

    # ---------- A.1 Parameter confidence intervals ----------
    @staticmethod
    def parameter_confidence_interval(fit_result, level=0.95):
        """Compute confidence intervals for fitted parameters.

        Parameters
        ----------
        fit_result : dict
            Result dictionary from ``fit``, ``linear_fit`` or ``polynomial_fit``.
        level : float, default 0.95
            Confidence level in ``(0, 1)``.

        Returns
        -------
        ConfidenceIntervalResult
            Dict-like wrapper containing ``{"parameters": [...]}`` where each
            entry has ``name``, ``estimate``, ``error``, ``lower_bound``,
            ``upper_bound``, ``level``, and ``distribution``.
        """
        if level <= 0 or level >= 1:
            raise ValueError("level must be in (0, 1)")
        
        params = fit_result.get("parameters")
        errors_dict = fit_result.get("errors")
        ndof = fit_result.get("ndof")
        
        if params is None or errors_dict is None:
            raise ValueError(
                "fit_result must contain 'parameters' and 'errors'"
            )
        
        # Extract values according to structure (dict or array)
        if isinstance(params, dict):
            param_names = list(params.keys())
            param_vals = np.array([params[k] for k in param_names])
            
            # Handle error dict: try direct keys first, then try 's' + key
            if isinstance(errors_dict, dict):
                error_vals = []
                for k in param_names:
                    if k in errors_dict:
                        error_vals.append(errors_dict[k])
                    elif f's{k}' in errors_dict:
                        error_vals.append(errors_dict[f's{k}'])
                    else:
                        raise KeyError(f"Cannot find error for parameter '{k}'")
                error_vals = np.array(error_vals)
            else:
                error_vals = np.asarray(errors_dict)
        else:
            param_names = [f"p{i}" for i in range(len(params))]
            param_vals = np.asarray(params)
            error_vals = np.asarray(errors_dict)
        
        alpha = 1 - level
        
        # Determine quantile (Student‑t or normal)
        if ndof is not None and ndof > 0 and ndof <= 30:
            cuantil = stats.t.ppf(1 - alpha / 2, ndof)
            dist = "t"
        else:
            cuantil = stats.norm.ppf(1 - alpha / 2)
            dist = "normal"
        
        ic_list = []
        for name, val, err in zip(param_names, param_vals, error_vals):
            margin = cuantil * err
            ic_list.append({
                "name": name,
                "estimate": float(val),
                "error": float(err),
                "lower_bound": float(val - margin),
                "upper_bound": float(val + margin),
                "level": level,
                "distribution": dist,
            })
        
        return ConfidenceIntervalResult({"parameters": ic_list})

    # ---------- A.2 Model prediction uncertainty ----------
    @staticmethod
    def prediction_uncertainty(fit_result, model, x0):
        """Compute model prediction uncertainty from parameter covariance.

        Parameters
        ----------
        fit_result : dict
            Result dictionary from ``fit``, ``linear_fit`` or ``polynomial_fit``.
        model : callable
            Callable ``f(x, *params)`` used for fitting.
        x0 : float | array_like
            Evaluation point(s).

        Returns
        -------
        dict
            ``{"x", "y", "sigma_model"}`` as scalars or arrays.

        Notes
        -----
        ``sigma_model`` includes only parameter-propagated uncertainty
        (confidence-band style), not measurement noise of new observations.
        """
        params = fit_result.get("parameters")
        covariance = fit_result.get("covariance")
        
        if params is None or covariance is None:
            raise ValueError(
                "fit_result must contain 'parameters' and 'covariance'"
            )
        
        # Convert params to array if dict
        if isinstance(params, dict):
            param_vals = np.array([params[k] for k in sorted(params.keys())])
        else:
            param_vals = np.asarray(params)
        
        x0 = np.atleast_1d(x0)
        is_scalar = np.isscalar(x0[0]) and len(x0) == 1
        if len(x0) == 1:
            x0_arr = x0
        else:
            x0_arr = x0
        
        # Numerical case: numerical derivative
        y_pred = model(x0_arr, *param_vals)

        # Numerical gradient by finite differences
        eps = np.sqrt(np.finfo(float).eps)
        grad_vals = np.zeros((len(param_vals), len(x0_arr)))

        for i in range(len(param_vals)):
            p_plus = param_vals.copy()
            p_plus[i] += eps
            p_minus = param_vals.copy()
            p_minus[i] -= eps

            grad_vals[i] = (
                model(x0_arr, *p_plus) - model(x0_arr, *p_minus)
            ) / (2 * eps)
        
        # Error propagation: Var(f) = grad_f^T · Cov · grad_f
        sigma_model = np.sqrt(
            np.sum(grad_vals * (covariance @ grad_vals), axis=0)
        )
        
        result = {
            "x": float(x0_arr[0]) if is_scalar else x0_arr,
            "y": float(y_pred[0]) if is_scalar else y_pred,
            "sigma_model": float(sigma_model[0]) if is_scalar else sigma_model,
        }
        
        return result
    
    ####Connection with uncertainties module for later propagation####


class ConfidenceIntervalResult:
    """Wrapper for confidence interval results with automatic formatting."""
    
    def __init__(self, data):
        self._data = data
    
    def __getitem__(self, key):
        """Allow dict-like access for backward compatibility."""
        return self._data[key]
    
    def __contains__(self, key):
        """Allow membership testing (in operator)."""
        return key in self._data
    
    def __str__(self):
        """Formatted string representation for easy printing."""
        params = self._data["parameters"]
        if not params:
            return "No parameters"
        
        level = params[0].get("level", 0.95)
        dist = params[0].get("distribution", "unknown")
        
        lines = [f"\n=== {level*100:.0f}% Confidence Intervals ({dist} distribution) ==="]
        
        for param in params:
            name = param["name"]
            est = param["estimate"]
            err = param["error"]
            lb = param["lower_bound"]
            ub = param["upper_bound"]
            
            lines.append(f"{name}: {est:.6g} ± {err:.6g}")
            lines.append(f"  CI: [{lb:.6g}, {ub:.6g}]")
        
        return "\n".join(lines)
    
    def __repr__(self):
        return f"ConfidenceIntervalResult({len(self._data['parameters'])} parameters)"
    
    def get(self, key, default=None):
        """Dict-like get method."""
        return self._data.get(key, default)


class FitResult:
    def __init__(self, raw, *, model, x, y, sy):
        self._raw = raw
        self.model = model
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self._sy = np.asarray(sy, dtype=float)

        raw_params = raw.get("parameters")
        if isinstance(raw_params, dict):
            self._param_names = list(raw_params.keys())
            self.params = np.asarray([raw_params[k] for k in self._param_names], dtype=float)
        else:
            self._param_names = [f"p{i}" for i in range(len(raw_params))]
            self.params = np.asarray(raw_params, dtype=float)

        self.cov = np.asarray(raw.get("covariance"), dtype=float)
        self.dof = len(self._x) - len(self.params)

    @property
    def raw(self):
        return self._raw

    @property
    def parameters(self):
        return self.params

    @property
    def covariance(self):
        return self.cov

    @property
    def sigma(self):
        return np.sqrt(np.diag(self.cov))

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        return self.model(x, *self.params)

    @property
    def residuals(self):
        return self._y - self.predict(self._x)

    @property
    def chi2(self):
        return float(np.sum((self.residuals / self._sy) ** 2))

    @property
    def reduced_chi2(self):
        if self.dof <= 0:
            return float("nan")
        return float(self.chi2 / self.dof)

    def params_quantity(self, unit="1"):
        """Return all fitted parameters as a list of ``Quantity``.

        Parameters
        ----------
        unit : str, default "1"
            Unit assigned to all returned parameters. Use ``"1"`` for
            dimensionless parameters.
        """
        # Local import to avoid circular imports at module load time.
        from marhare.quantities import Quantity

        return [
            Quantity(float(v), float(s), unit=unit, symbol=name, traceable=False)
            for name, v, s in zip(self._param_names, self.params, self.sigma)
        ]

    def parameter_quantity(self, key, *, unit="1", symbol=None):
        """Return a single fitted parameter as ``Quantity``.

        Parameters
        ----------
        key : str | int
            Parameter name (e.g. ``"b"``) or index (e.g. ``1``).
        unit : str, default "1"
            Unit assigned to the returned parameter quantity.
            Use ``"1"`` when the fit was performed on normalized SI values and
            the parameter is dimensionless in that representation.
        symbol : str | None
            Optional symbol override. If omitted, the parameter name is used.
        """
        # Local import to avoid circular imports at module load time.
        from marhare.quantities import Quantity

        if isinstance(key, str):
            if key not in self._param_names:
                raise KeyError(f"Unknown parameter name: {key}")
            idx = self._param_names.index(key)
            name = key
        else:
            try:
                idx = int(key)
            except (TypeError, ValueError):
                raise TypeError("key must be a parameter name (str) or index (int)")

            if idx < 0 or idx >= len(self.params):
                raise IndexError(f"Parameter index out of range: {idx}")
            name = self._param_names[idx]

        return Quantity(
            float(self.params[idx]),
            float(self.sigma[idx]),
            unit=unit,
            symbol=name if symbol is None else symbol,
            traceable=False,
        )

    def as_dict(self):
        return dict(self._raw)

    def confidence_interval(self, level=0.95):
        return _Fitting.parameter_confidence_interval(self._raw, level=level)

    def prediction(self, x0):
        return _Fitting.prediction_uncertainty(self._raw, self.model, x0)


def _is_quantity_like(obj):
    return (
        hasattr(obj, "value") and hasattr(obj, "sigma")
    ) or (
        isinstance(obj, dict)
        and (
            "measure" in obj
            or "result" in obj
            or ("value" in obj and "sigma" in obj)
        )
    )


def fit(model, X, Y, p0=None):
    """
    Generic curve fit for Quantity-like inputs or raw numpy arrays.

    Parameters
    ----------
    model : callable
        Model function with signature ``f(x, *params)``.
    X : array-like | Quantity-like
        Independent variable data.
    Y : array-like | Quantity-like | tuple(array-like, array-like)
        Dependent variable data. If a tuple/list of length 2 is provided,
        it is interpreted as ``(y, sy)`` for raw numpy workflows.
    p0 : sequence | None
        Optional initial parameter guess.

    Returns
    -------
    FitResult
        Encapsulated fit result.
    """
    if not callable(model):
        raise TypeError("model must be a callable with signature f(x, *params)")

    if _is_quantity_like(X):
        x, _ = _extract_value_sigma(X)
        x = np.asarray(x, dtype=float)
    elif isinstance(X, (tuple, list)) and len(X) == 2:
        x = np.asarray(X[0], dtype=float)
    else:
        x = np.asarray(X, dtype=float)

    sy = None
    if _is_quantity_like(Y):
        y, sy = _extract_value_sigma(Y)
        y = np.asarray(y, dtype=float)
        sy = np.asarray(sy, dtype=float)
    elif isinstance(Y, (tuple, list)) and len(Y) == 2:
        y = np.asarray(Y[0], dtype=float)
        sy = np.asarray(Y[1], dtype=float)
    else:
        y = np.asarray(Y, dtype=float)

    if x.shape != y.shape:
        raise ValueError("X and Y must have the same shape")
    if x.ndim != 1:
        raise ValueError("X and Y must be 1D arrays")

    if sy is not None:
        if sy.ndim == 0:
            sy = np.full_like(y, float(sy), dtype=float)
        if sy.shape != y.shape:
            raise ValueError("sy must have the same shape as Y")
        if np.any(sy <= 0):
            raise ValueError("sy must be positive at all points")

    popt, pcov = optimize.curve_fit(model, x, y, sigma=sy, p0=p0)
    yfit = model(x, *popt)

    # Keep chi2-based diagnostics consistent, even when no sy is provided.
    sy_eff = np.ones_like(y) if sy is None else sy
    chi2 = np.sum(((y - yfit) / sy_eff) ** 2)
    ndof = len(x) - len(popt)
    chi2_red, p = _Fitting._safe_fit_stats(chi2, ndof)

    raw = {
        "parameters": np.asarray(popt, dtype=float),
        "errors": np.sqrt(np.diag(pcov)),
        "covariance": np.asarray(pcov, dtype=float),
        "yfit": np.asarray(yfit, dtype=float),
        "chi2": float(chi2),
        "ndof": int(ndof),
        "chi2_red": float(chi2_red),
        "p": float(p),
    }

    return FitResult(raw, model=model, x=x, y=y, sy=sy_eff)


def fit_quantity(model, xq, yq, *, degree=None, p0=None, variable="x"):
    """
    Fit ``yq`` versus ``xq`` and return a ``FitResult`` wrapper.

    Parameters
    ----------
        model : "linear" | "polynomial" | callable
        Model to fit.
        - ``"linear"`` uses ``y = a + b*x``.
        - ``"polynomial"`` uses ``degree`` with weighted ``np.polyfit``.
        - ``callable`` must follow ``f(x, *params)`` signature.
          Example: ``def f(x, a, b): return a*x + b``.
    xq, yq : quantity-like
        Independent and dependent quantities (value + uncertainty).
    degree : int | None, optional
        Required only when ``model == "polynomial"``.
    p0 : sequence | None, optional
        Initial parameter guess for non-linear callable models.
    variable : str, default "x"
        Reserved for compatibility; ignored for numeric models.

    Returns
    -------
    FitResult
        Wrapper exposing:
        - ``raw`` (fit dictionary)
        - ``confidence_interval(level=...)``
        - ``prediction(x0)``

    Notes
    -----
    - Uncertainties are taken from ``yq`` and used as absolute sigma weights.
    - Callables without fit parameters (signature ``f(x)``) are not suitable for
      parameter estimation with this API.
    """
    x, _ = _extract_value_sigma(xq)
    y, sy = _extract_value_sigma(yq)

    # Quantity objects are only accepted at the API boundary.
    # Internal fitting always runs on plain numpy float arrays.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sy = np.asarray(sy, dtype=float)

    if sy.ndim == 0:
        sy = np.full_like(y, float(sy), dtype=float)

    # Convert model string to actual callable for prediction
    actual_model = model
    if isinstance(model, str):
        if model == "linear":
            raw = _Fitting.linear_fit(x, y, sy=sy)
            # Create linear model function: y = a + b*x
            actual_model = lambda x, a, b: a + b * x
        elif model == "polynomial":
            if degree is None:
                raise ValueError("polynomial fit requires degree=")
            raw = _Fitting.polynomial_fit(x, y, degree, sy=sy)
            # Create polynomial model function using numpy polyval
            # polyval expects coefficients in descending order
            actual_model = lambda x, *coeffs: np.polyval(coeffs, x)
        else:
            raise ValueError(f"Unknown model shortcut: {model}")
    elif callable(model):
        raw = _Fitting.fit(
            model,
            x,
            y,
            sy=sy,
            p0=p0,
            variable=variable,
        )
        actual_model = model
    else:
        raise ValueError("model must be a string or callable")

    return FitResult(
        raw,
        model=actual_model,
        x=x,
        y=y,
        sy=sy,
    )


def errorbar(X, Y, ax=None, **kwargs):
    """Plot data with uncertainties using matplotlib error bars."""
    import matplotlib.pyplot as plt

    x, sx = _extract_value_sigma(X)
    y, sy = _extract_value_sigma(Y)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sx = np.asarray(sx, dtype=float)
    sy = np.asarray(sy, dtype=float)

    if ax is None:
        ax = plt.gca()

    if "xerr" not in kwargs:
        kwargs["xerr"] = sx
    if "yerr" not in kwargs:
        kwargs["yerr"] = sy

    ax.errorbar(x, y, **kwargs)
    return ax


def plot_fit(fit, ax=None, n=200, **kwargs):
    """Plot fitted model curve over the x-range used in fitting."""
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    xdata = np.asarray(fit._x, dtype=float)
    xmin = float(np.min(xdata))
    xmax = float(np.max(xdata))
    xline = np.linspace(xmin, xmax, int(n))
    yline = fit.predict(xline)

    ax.plot(xline, yline, **kwargs)
    return ax

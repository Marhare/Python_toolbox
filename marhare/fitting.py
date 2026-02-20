"""
fitting.py
==========

PURPOSE AND DIFFERENCES:
- Curve fitting module using weighted least squares (WLS).
- fitting.py: fits models to data (linear, polynomial, generic, symbolic).
- estadistica.py: hypothesis tests, p‑values, confidence intervals for distributions.
- incertidumbres.py: propagates parameter uncertainties to observables (derivatives, etc.).
- This module COMBINES WLS + a covariance matrix to allow later propagation via incertidumbres.py.

DEFAULT ASSUMPTIONS:
- sy errors are interpreted as known absolute uncertainties in y.
- absolute_sigma=True is assumed (no automatic error rescaling).
- residuals are assumed independent and Gaussian.
- statistical validity (normality, homoscedasticity) is the user's responsibility.


PUBLIC FUNCTIONS
================

1) linear_fit(x, y, sy=None)
   Exact analytic WLS fit for y = a + b·x.
   
   INPUT:
       x: array_like (n,) -> independent variable
       y: array_like (n,) -> dependent variable
       sy: array_like (n,) | None -> absolute uncertainty of y; if None → sy=1 (all equal)
   
   OUTPUT: dict with
       "parameters": {"a": float, "b": float}  -> intercept and slope
       "errors": {"sa": float, "sb": float}   -> parameter standard errors
       "covariance": array (2,2)               -> covariance matrix (a,b)
       "yfit": array (n,)                      -> fitted values
       "chi2": float                           -> sum of residuals²/sy²
       "ndof": int                             -> degrees of freedom = n - 2
       "chi2_red": float                       -> chi2 / ndof
       "p": float                              -> p‑value (1 - CDF_chi2(chi2,ndof))
   
   NOTES:
       - Uses an analytic formulation (no numeric optimization).
       - absolute_sigma=True implied.
   
   ERRORS:
       - ValueError if len(x) != len(y)
       - ValueError if sy contains values <= 0 or shape differs from y
       - ValueError if x,y are not 1D arrays

2) polynomial_fit(x, y, degree, sy=None)
   Polynomial fit y = p0·x^n + ... + pn by WLS.
   
   INPUT:
       x: array_like (n,) -> independent variable
       y: array_like (n,) -> dependent variable
       grado: int         -> polynomial degree (grado >= 0)
       sy: array_like (n,) | None -> absolute uncertainty of y; if None → sy=1
   
   OUTPUT: dict with
       "coefficients": array (degree+1,)       -> coefficients in descending order (x^n, ..., x^0)
       "errors": array (degree+1,)            -> standard error of each coefficient
       "covariance": array (degree+1,degree+1)  -> covariance matrix
       "yfit": array (n,)                     -> fitted values
       "chi2": float                          -> sum of residuals²/sy²
       "ndof": int                            -> degrees of freedom = n - (grado+1)
       "chi2_red": float                      -> chi2 / ndof
       "p": float                             -> p‑value
   
   NOTES:
       - Based on np.polyfit with weights 1/sy².
       - Coefficients in descending order (standard polynomial).
       - absolute_sigma=True implied (unscaled covariance).
   
   ERRORS:
       - ValueError if len(x) != len(y)
       - ValueError if sy <= 0 or shape differs from y
       - ValueError if x,y are not 1D arrays
       - Error if grado < 0 or grado >= n

3) fit(model, x, y, sy=None, p0=None, *, variable="x")
   Unified generic fit: accepts callable OR sympy expression.
   
   INPUT:
       model: callable f(x, *params) | sympy.Expr
               If callable: function that returns f(x, p1, p2, ...).
               If sympy.Expr: symbolic expression in "x" and parameters (p, a, k, etc.).
       x: array_like (n,)  -> independent variable
       y: array_like (n,)  -> dependent variable
       sy: array_like (n,) | None -> absolute uncertainty of y
       p0: array_like (m,) | None -> initial values for m parameters
       variable: str -> independent variable name in expr (default "x")
   
   OUTPUT: dict with
       "parameters": array (m,)                -> fitted parameter values
       "errors": array (m,)                   -> standard error of each parameter
       "covariance": array (m,m)               -> covariance matrix
       "yfit": array (n,)                      -> fitted values
       "chi2": float                           -> sum of residuals²/sy²
       "ndof": int                             -> degrees of freedom = n - m
       "chi2_red": float                       -> chi2 / ndof
       "p": float                              -> p‑value
       [if model is sympy.Expr, adds:]
       "expression": sympy.Expr                 -> original expression
       "symbolic_parameters": list[sympy.Symbol] -> parameter symbols ordered by name
   
   NOTES:
       - Based on scipy.optimize.curve_fit with absolute_sigma=True.
       - If model is sympy.Expr: automatically lambdified.
       - Symbolic parameters detected and ordered alphabetically.
       - If p0 is None: tries p0=ones(m) (may fail with poor initialization).
   
   ERRORS:
       - TypeError if model is neither callable nor sympy.Expr
       - ValueError if variable is not in expr
       - ValueError if len(p0) != number of parameters
       - ValueError if len(x) != len(y) or sy shape/values are invalid
       - RuntimeError if curve_fit does not converge

4) parameter_confidence_interval(fit_result, level=0.95)
   Computes confidence intervals (CI) for fitted parameters.
   
   INPUT:
       fit_result: dict -> output of fit()/linear_fit()/polynomial_fit()
       level: float in (0,1) -> confidence level (default 0.95 → 95%)
   
   OUTPUT: dict with
       "parameters": list[dict] with one entry per parameter:
           {
               "name": str,           -> parameter name
               "estimate": float,     -> fitted value
               "error": float,        -> standard error
               "lower_bound": float,       -> lower CI bound
               "upper_bound": float,       -> upper CI bound
               "level": float,          -> confidence level used
               "distribution": "t" | "normal" -> quantile used
           }
   
   NOTES:
       - Uses Student‑t if ndof<=30, normal if ndof>30 or missing.
       - CI = parameter ± quantile * error
       - DOES NOT modify the original fit.
       - NOT a prediction interval (that would include experimental sy).
   
   ERRORS:
       - ValueError if level is not in (0,1)
       - ValueError if fit_result lacks 'parameters' or 'errors'

5) prediction_uncertainty(fit_result, model, x0)
   Computes statistical uncertainty of the model prediction at x0.
   
   INPUT:
       fit_result: dict -> output of fit()/linear_fit()/polynomial_fit()
       model: callable f(x, *params) | sympy.Expr -> same model used in fit
       x0: float | array_like -> point(s) to evaluate the prediction
   
   ERROR: dict with
       "x": float | array -> evaluation point(s)
       "y": float | array -> model prediction
       "sigma_model": float | array -> statistical uncertainty of the prediction
       (x and sigma_model are scalars if x0 is scalar, arrays if x0 is array)
   
   NOTES:
       - Computes ONLY parameter‑propagated uncertainty.
       - DOES NOT include experimental instrument error sy.
       - Formula: Var(f) = ∇f^T · Cov · ∇f
       - Symbolic gradient if model is sympy.Expr, numeric if callable.
       - WARNING: this is a confidence band (uncertainty of the mean).
         For a prediction band, you must add experimental sy.
   
   ERRORS:
       - ValueError if fit_result lacks 'parameters' or 'covariance'
       - ValueError if model is sympy.Expr but 'symbolic_parameters' is missing
       - RuntimeError if lambdify or differentiation fails


CONVENTIONS AND TYPICAL FLOW
============================

1. Fit data:
    res = _Fitting.linear_fit(x, y, sy=sy_data)
   or
    res = _Fitting.fit(model, x, y, sy=sy_data, p0=p0_inicial)

2. Interpret results:
   params = res["parameters"]
   p_valor = res["p"]  -> if p > 0.05, fit is "acceptable"
   chi2_red = res["chi2_red"]  -> if ~1, fit is good

3. Build CI for parameters:
    ic = _Fitting.parameter_confidence_interval(res, level=0.95)
   for param_ic in ic["parameters"]:
       print(f"{param_ic['name']}: {param_ic['lower_bound']} - {param_ic['upper_bound']}")

4. Evaluate prediction uncertainty:
    pred = _Fitting.prediction_uncertainty(res, model, x_nuevo)
   print(f"y({x_nuevo}) = {pred['y']} ± {pred['sigma_model']}")

5. Propagate to other observables (use incertidumbres.py):
   # Combine fitted parameters + covariances → new observable
   # (See incertidumbres.py module for details)
"""

import numpy as np
from scipy import stats, optimize
import sympy as sp

from marhare.uncertainties import value_quantity

class _Fitting:

    @staticmethod
    def _validate_data(x, y, sy=None):
        x = np.asarray(x)
        y = np.asarray(y)
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
        if x.ndim != 1:
            raise ValueError("x and y must be 1D arrays")
        if sy is None:
            sy = np.ones_like(y)
        else:
            sy = np.asarray(sy)
            if sy.shape != y.shape:
                raise ValueError("sy must have the same shape as y")
            if np.any(sy <= 0):
                raise ValueError("sy must be positive at all points")
        return x, y, sy

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
        p = stats.chi2.sf(chi2, ndof)

        return {
            "parameters": popt,
            "errors": perr,
            "covariance": pcov,
            "yfit": yfit,
            "chi2": chi2,
            "ndof": ndof,
            "chi2_red": chi2 / ndof,
            "p": p,
        }

    # ---------- Linear ----------
    @staticmethod
    def linear_fit(x, y, sy=None):
        """
        INPUT:
            x: array_like (n,)
            y: array_like (n,)
            sy: array_like (n,) | None  -> absolute errors in y
        OUTPUT:
            dict with:
                - parameters: {"a": intercept, "b": slope}
                - errors: {"sa": error_a, "sb": error_b}
                - covariance: 2x2 matrix
                - chi2, ndof, chi2_red, p
                - yfit
        NOTAS:
            - Analytic weighted linear fit (WLS)
            - Assumes absolute sigma is known (no error rescaling)
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
        p = stats.chi2.sf(chi2, ndof)

        return {
            "parameters": {"a": float(a), "b": float(b)},
            "errors": {"sa": sa, "sb": sb},
            "covariance": cov,
            "chi2": float(chi2),
            "ndof": ndof,
            "chi2_red": float(chi2 / ndof),
            "p": float(p),
            "yfit": yfit,
        }

    # ---------- Polynomial ----------
    @staticmethod
    def polynomial_fit(x, y, degree, sy=None):
        """
        INPUT:
            x: array_like (n,)
            y: array_like (n,)
            degree: int
            sy: array_like (n,) | None  -> absolute errors in y
        OUTPUT:
            dict with:
                - coefficients (descending order)
                - errors (sqrt(diag(cov)))
                - covariance
                - chi2, ndof, chi2_red, p
                - yfit
        NOTAS:
            - Based on np.polyfit with weights
            - Assumes absolute sigma is known (no error rescaling)
        """
        x, y, sy = _Fitting._validate_data(x, y, sy)
        coef, cov = np.polyfit(x, y, degree, w=1 / sy, cov="unscaled")
        errors_arr = np.sqrt(np.diag(cov))
        yfit = np.polyval(coef, x)
        chi2 = np.sum(((y - yfit) / sy)**2)
        ndof = len(x) - (degree + 1)
        p = stats.chi2.sf(chi2, ndof)
        return {
            "parameters": coef,
            "errors": errors_arr,
            "covariance": cov,
            "yfit": yfit,
            "chi2": chi2,
            "ndof": ndof,
            "chi2_red": chi2 / ndof,
            "p": p,
        }

    # ---------- Unified ----------
    @staticmethod
    def fit(model, x, y, sy=None, p0=None, *, variable="x"):
        """
        INPUT:
            model: callable f(x, *params) | sympy.Expr
            x: array_like (n,)
            y: array_like (n,)
            sy: array_like (n,) | None  -> absolute errors in y
            p0: initial values | None
            variable: str -> independent variable name
        OUTPUT:
            dict with:
                - parameters, errors, covariance, yfit
                - chi2, ndof, chi2_red, p
            If the model is symbolic, adds:
                - expression
                - symbolic_parameters
        NOTAS:
            - Weighted least squares fit (curve_fit)
            - absolute_sigma=True always
        """
        if callable(model) and not isinstance(model, sp.Expr):
            return _Fitting._curve_fit(model, x, y, sy, p0)

        if isinstance(model, sp.Expr):
            expr = model
            var_symbol = None
            for s in expr.free_symbols:
                if s.name == variable:
                    var_symbol = s
                    break
            if var_symbol is None:
                if len(expr.free_symbols) == 1:
                    var_symbol = list(expr.free_symbols)[0]
                else:
                    raise ValueError(
                        "Could not identify the independent variable; "
                        "specify the name with 'variable'"
                    )

            params = sorted(expr.free_symbols - {var_symbol}, key=lambda s: s.name)
            if p0 is not None and len(p0) != len(params):
                raise ValueError("p0 must have the same length as the parameters")
            if p0 is None:
                p0 = np.ones(len(params))

            f = sp.lambdify((var_symbol, *params), expr, "numpy")

            def f_num(x, *p):
                return f(x, *p)

            res = _Fitting._curve_fit(f_num, x, y, sy, p0)
            res["expression"] = expr
            res["symbolic_parameters"] = params
            return res

        raise TypeError("model must be callable or sympy.Expr")

    # ---------- A.1 Parameter confidence intervals ----------
    @staticmethod
    def parameter_confidence_interval(fit_result, level=0.95):
        """
        Compute confidence intervals for fitted parameters.
        
        INPUT:
            fit_result: dict result from fit()/linear_fit()/polynomial_fit()
            level: float [0, 1] -> confidence level (default 0.95)
        
        OUTPUT:
            dict with interval list:
            {
                "parameters": [
                    {
                        "name": str,
                        "estimate": float,
                        "lower_bound": float,
                        "upper_bound": float,
                        "error": float,
                        "level": float,
                        "distribution": "t" | "normal"
                    },
                    ...
                ]
            }
        
        NOTAS:
            - Uses Student‑t if ndof is available and small
            - Uses normal distribution if ndof is large (>30) or missing
            - Does NOT modify the fit
            - Parameter CI, NOT a prediction interval
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
        """
        Compute the statistical uncertainty of the model prediction at x0.
        
        INPUT:
            fit_result: dict result from fit()/linear_fit()/polynomial_fit()
            model: callable f(x, *params) | sympy.Expr
            x0: float | array_like -> prediction point(s)
        
        OUTPUT:
            If x0 is scalar:
                dict with:
                    "x": float,
                    "y": float,
                    "sigma_model": float,
            If x0 is array:
                dict with:
                    "x": array,
                    "y": array,
                    "sigma_model": array
        
        NOTAS:
                        - Computes ONLY parameter uncertainty
                        - DOES NOT include experimental instrument error (sy)
                        - Uses error propagation: Var(f) = grad_f^T · Cov · grad_f
                        - Gradient is computed symbolically (if expr) or numerically
                        - WARNING: this is uncertainty of the MEAN (confidence band),
                            NOT a prediction interval (prediction band)
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
        
        # Compute prediction and gradient
        if isinstance(model, sp.Expr):
            # Symbolic case: analytic derivative
            expr = model
            param_symbols = fit_result.get("symbolic_parameters")
            
            if param_symbols is None:
                raise ValueError(
                    "fit_result must contain 'symbolic_parameters' "
                    "if the model is symbolic"
                )
            
            # Detect independent variable
            var_symbol = None
            for s in expr.free_symbols:
                if s.name == "x":
                    var_symbol = s
                    break
            if var_symbol is None:
                if len(expr.free_symbols) == 1:
                    var_symbol = list(expr.free_symbols)[0]
            
            # Lambdify for prediction
            f_eval = sp.lambdify((var_symbol, *param_symbols), expr, "numpy")
            y_pred = np.atleast_1d(f_eval(x0_arr, *param_vals))
            
            # Gradient with respect to parameters
            # Evaluate each gradient directly and ensure consistent array shape
            grad_vals = []
            for p in param_symbols:
                grad_func = sp.lambdify((var_symbol, *param_symbols), sp.diff(expr, p), "numpy")
                grad_result = np.atleast_1d(grad_func(x0_arr, *param_vals))
                grad_vals.append(grad_result)
            grad_vals = np.array(grad_vals)
        else:
            # Numeric case: numerical derivative
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
    def __init__(self, raw, *, model, xq=None, yq=None, degree=None):
        self._raw = raw
        self.model = model
        self.degree = degree

        # metadatos opcionales (para plots, LaTeX futuro)
        self.x_symbol = getattr(xq, "symbol", None) if xq is not None else None
        self.y_symbol = getattr(yq, "symbol", None) if yq is not None else None
        self.x_unit = getattr(xq, "unit", None) if xq is not None else None
        self.y_unit = getattr(yq, "unit", None) if yq is not None else None

    # acceso directo a los resultados numéricos
    @property
    def raw(self):
        return self._raw

    # --- Métodos de fitting.py ---

    def confidence_interval(self, level=0.95):
        return _Fitting.parameter_confidence_interval(self._raw, level=level)

    def prediction(self, x0):
        return _Fitting.prediction_uncertainty(self._raw, self.model, x0)


def fit_quantity(model, xq, yq, *, degree=None, p0=None, variable="x"):
    """
    Fit yq vs xq to a model and return a FitResult wrapper.

    INPUT:
        model: "linear" | "polynomial" | callable | sympy.Expr
        xq: quantity dict for the independent variable
        yq: quantity dict for the dependent variable
        degree: int | None -> required if model is "polynomial"
        p0: initial guess for parameters (optional)
        variable: str -> independent variable name for symbolic models

    OUTPUT:
        FitResult with access to raw fit results and helpers.
    """
    x, sx = value_quantity(xq)
    y, sy = value_quantity(yq)

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
    elif callable(model) or isinstance(model, sp.Expr):
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
        raise ValueError("model must be a string, callable, or sympy expression")

    return FitResult(
        raw,
        model=actual_model,
        xq=xq,
        yq=yq,
        degree=degree,
    )

"""
ajustes.py
==========

PURPOSE AND DIFFERENCES:
- Curve fitting module using weighted least squares (WLS).
- ajustes.py: fits models to data (linear, polynomial, generic, symbolic).
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

1) ajuste_lineal(x, y, sy=None)
   Exact analytic WLS fit for y = a + b·x.
   
   INPUT:
       x: array_like (n,) -> independent variable
       y: array_like (n,) -> dependent variable
       sy: array_like (n,) | None -> absolute uncertainty of y; if None → sy=1 (all equal)
   
   OUTPUT: dict with
       "parametros": {"a": float, "b": float}  -> intercept and slope
       "errores": {"sa": float, "sb": float}   -> parameter standard errors
       "covarianza": array (2,2)               -> covariance matrix (a,b)
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

2) ajuste_polinomico(x, y, grado, sy=None)
   Polynomial fit y = p0·x^n + ... + pn by WLS.
   
   INPUT:
       x: array_like (n,) -> independent variable
       y: array_like (n,) -> dependent variable
       grado: int         -> polynomial degree (grado >= 0)
       sy: array_like (n,) | None -> absolute uncertainty of y; if None → sy=1
   
   OUTPUT: dict with
       "coeficientes": array (grado+1,)       -> coefficients in descending order (x^n, ..., x^0)
       "errores": array (grado+1,)            -> standard error of each coefficient
       "covarianza": array (grado+1,grado+1)  -> covariance matrix
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

3) ajuste(modelo, x, y, sy=None, p0=None, *, variable="x")
   Unified generic fit: accepts callable OR sympy expression.
   
   INPUT:
       modelo: callable f(x, *params) | sympy.Expr
               If callable: function that returns f(x, p1, p2, ...).
               If sympy.Expr: symbolic expression in "x" and parameters (p, a, k, etc.).
       x: array_like (n,)  -> independent variable
       y: array_like (n,)  -> dependent variable
       sy: array_like (n,) | None -> absolute uncertainty of y
       p0: array_like (m,) | None -> initial values for m parameters
       variable: str -> independent variable name in expr (default "x")
   
   OUTPUT: dict with
       "parametros": array (m,)                -> fitted parameter values
       "errores": array (m,)                   -> standard error of each parameter
       "covarianza": array (m,m)               -> covariance matrix
       "yfit": array (n,)                      -> fitted values
       "chi2": float                           -> sum of residuals²/sy²
       "ndof": int                             -> degrees of freedom = n - m
       "chi2_red": float                       -> chi2 / ndof
       "p": float                              -> p‑value
       [if modelo is sympy.Expr, adds:]
       "expresion": sympy.Expr                 -> original expression
       "parametros_simbolicos": list[sympy.Symbol] -> parameter symbols ordered by name
   
   NOTES:
       - Based on scipy.optimize.curve_fit with absolute_sigma=True.
       - If modelo is sympy.Expr: automatically lambdified.
       - Symbolic parameters detected and ordered alphabetically.
       - If p0 is None: tries p0=ones(m) (may fail with poor initialization).
   
   ERRORS:
       - TypeError if modelo is neither callable nor sympy.Expr
       - ValueError if variable is not in expr
       - ValueError if len(p0) != number of parameters
       - ValueError if len(x) != len(y) or sy shape/values are invalid
       - RuntimeError if curve_fit does not converge

4) intervalo_confianza_parametros(resultado_ajuste, nivel=0.95)
   Computes confidence intervals (CI) for fitted parameters.
   
   INPUT:
       resultado_ajuste: dict -> output of ajuste()/ajuste_lineal()/ajuste_polinomico()
       nivel: float in (0,1) -> confidence level (default 0.95 → 95%)
   
   OUTPUT: dict with
       "parametros": list[dict] with one entry per parameter:
           {
               "nombre": str,           -> parameter name
               "estimacion": float,     -> fitted value
               "error": float,          -> standard error
               "inferior": float,       -> lower CI bound
               "superior": float,       -> upper CI bound
               "nivel": float,          -> confidence level used
               "distribucion": "t" | "normal" -> quantile used
           }
   
   NOTES:
       - Uses Student‑t if ndof<=30, normal if ndof>30 or missing.
       - CI = parameter ± quantile * error
       - DOES NOT modify the original fit.
       - NOT a prediction interval (that would include experimental sy).
   
   ERRORS:
       - ValueError if nivel is not in (0,1)
       - ValueError if resultado_ajuste lacks 'parametros' or 'errores'

5) incertidumbre_prediccion(resultado_ajuste, modelo, x0)
   Computes statistical uncertainty of the model prediction at x0.
   
   INPUT:
       resultado_ajuste: dict -> output of ajuste()/ajuste_lineal()/ajuste_polinomico()
       modelo: callable f(x, *params) | sympy.Expr -> same model used in fit
       x0: float | array_like -> point(s) to evaluate the prediction
   
   OUTPUT: dict with
       "x": float | array -> evaluation point(s)
       "y": float | array -> model prediction
       "sigma_modelo": float | array -> statistical uncertainty of the prediction
       (x and sigma_modelo are scalars if x0 is scalar, arrays if x0 is array)
   
   NOTES:
       - Computes ONLY parameter‑propagated uncertainty.
       - DOES NOT include experimental instrument error sy.
       - Formula: Var(f) = ∇f^T · Cov · ∇f
       - Symbolic gradient if modelo is sympy.Expr, numeric if callable.
       - WARNING: this is a confidence band (uncertainty of the mean).
         For a prediction band, you must add experimental sy.
   
   ERRORS:
       - ValueError if resultado_ajuste lacks 'parametros' or 'covarianza'
       - ValueError if modelo is sympy.Expr but 'parametros_simbolicos' is missing
       - RuntimeError if lambdify or differentiation fails


CONVENTIONS AND TYPICAL FLOW
============================

1. Fit data:
   res = ajustes.ajuste_lineal(x, y, sy=sy_data)
   or
   res = ajustes.ajuste(modelo, x, y, sy=sy_data, p0=p0_inicial)

2. Interpret results:
   params = res["parametros"]
   p_valor = res["p"]  -> if p > 0.05, fit is "acceptable"
   chi2_red = res["chi2_red"]  -> if ~1, fit is good

3. Build CI for parameters:
   ic = ajustes.intervalo_confianza_parametros(res, nivel=0.95)
   for param_ic in ic["parametros"]:
       print(f"{param_ic['nombre']}: {param_ic['inferior']} - {param_ic['superior']}")

4. Evaluate prediction uncertainty:
   pred = ajustes.incertidumbre_prediccion(res, modelo, x_nuevo)
   print(f"y({x_nuevo}) = {pred['y']} ± {pred['sigma_modelo']}")

5. Propagate to other observables (use incertidumbres.py):
   # Combine fitted parameters + covariances → new observable
   # (See incertidumbres.py module for details)
"""

import numpy as np
from scipy import stats, optimize
import sympy as sp

class _Ajustes:

    @staticmethod
    def _validar_datos(x, y, sy=None):
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
    def _ajuste_curvefit(f, x, y, sy=None, p0=None):
        x, y, sy = _Ajustes._validar_datos(x, y, sy)

        popt, pcov = optimize.curve_fit(
            f, x, y, sigma=sy, absolute_sigma=True, p0=p0
        )
        perr = np.sqrt(np.diag(pcov))
        yfit = f(x, *popt)
        chi2 = np.sum(((y - yfit) / sy)**2)
        ndof = len(x) - len(popt)
        p = stats.chi2.sf(chi2, ndof)

        return {
            "parametros": popt,
            "errores": perr,
            "covarianza": pcov,
            "yfit": yfit,
            "chi2": chi2,
            "ndof": ndof,
            "chi2_red": chi2 / ndof,
            "p": p,
        }

    # ---------- Linear ----------
    @staticmethod
    def ajuste_lineal(x, y, sy=None):
        """
        INPUT:
            x: array_like (n,)
            y: array_like (n,)
            sy: array_like (n,) | None  -> absolute errors in y
        OUTPUT:
            dict with:
                - parametros: {"a": intercept, "b": slope}
                - errores: {"sa": error_a, "sb": error_b}
                - covarianza: 2x2 matrix
                - chi2, ndof, chi2_red, p
                - yfit
        NOTAS:
            - Analytic weighted linear fit (WLS)
            - Assumes absolute sigma is known (no error rescaling)
        """
        x, y, sy = _Ajustes._validar_datos(x, y, sy)
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
            "parametros": {"a": float(a), "b": float(b)},
            "errores": {"sa": sa, "sb": sb},
            "covarianza": cov,
            "chi2": float(chi2),
            "ndof": ndof,
            "chi2_red": float(chi2 / ndof),
            "p": float(p),
            "yfit": yfit,
        }

    # ---------- Polynomial ----------
    @staticmethod
    def ajuste_polinomico(x, y, grado, sy=None):
        """
        INPUT:
            x: array_like (n,)
            y: array_like (n,)
            grado: int
            sy: array_like (n,) | None  -> absolute errors in y
        OUTPUT:
            dict with:
                - coeficientes (descending order)
                - errores (sqrt(diag(cov)))
                - covarianza
                - chi2, ndof, chi2_red, p
                - yfit
        NOTAS:
            - Based on np.polyfit with weights
            - Assumes absolute sigma is known (no error rescaling)
        """
        x, y, sy = _Ajustes._validar_datos(x, y, sy)
        coef, cov = np.polyfit(x, y, grado, w=1 / sy, cov="unscaled")
        errores = np.sqrt(np.diag(cov))
        yfit = np.polyval(coef, x)
        chi2 = np.sum(((y - yfit) / sy)**2)
        ndof = len(x) - (grado + 1)
        p = stats.chi2.sf(chi2, ndof)
        return {
            "coeficientes": coef,
            "errores": errores,
            "covarianza": cov,
            "yfit": yfit,
            "chi2": chi2,
            "ndof": ndof,
            "chi2_red": chi2 / ndof,
            "p": p,
        }

    # ---------- Unified ----------
    @staticmethod
    def ajuste(modelo, x, y, sy=None, p0=None, *, variable="x"):
        """
        INPUT:
            modelo: callable f(x, *params) | sympy.Expr
            x: array_like (n,)
            y: array_like (n,)
            sy: array_like (n,) | None  -> absolute errors in y
            p0: initial values | None
            variable: str -> independent variable name
        OUTPUT:
            dict with:
                - parametros, errores, covarianza, yfit
                - chi2, ndof, chi2_red, p
            If the model is symbolic, adds:
                - expresion
                - parametros_simbolicos
        NOTAS:
            - Weighted least squares fit (curve_fit)
            - absolute_sigma=True always
        """
        if callable(modelo) and not isinstance(modelo, sp.Expr):
            return _Ajustes._ajuste_curvefit(modelo, x, y, sy, p0)

        if isinstance(modelo, sp.Expr):
            expr = modelo
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

            res = _Ajustes._ajuste_curvefit(f_num, x, y, sy, p0)
            res["expresion"] = expr
            res["parametros_simbolicos"] = params
            return res

        raise TypeError("modelo must be callable or sympy.Expr")

    # ---------- A.1 Parameter confidence intervals ----------
    @staticmethod
    def intervalo_confianza_parametros(resultado_ajuste, nivel=0.95):
        """
        Compute confidence intervals for fitted parameters.
        
        INPUT:
            resultado_ajuste: dict result from ajuste()/ajuste_lineal()/ajuste_polinomico()
            nivel: float [0, 1] -> confidence level (default 0.95)
        
        OUTPUT:
            dict with interval list:
            {
                "parametros": [
                    {
                        "nombre": str,
                        "estimacion": float,
                        "inferior": float,
                        "superior": float,
                        "error": float,
                        "nivel": float,
                        "distribucion": "t" | "normal"
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
        if nivel <= 0 or nivel >= 1:
            raise ValueError("nivel must be in (0, 1)")
        
        params = resultado_ajuste.get("parametros")
        errores = resultado_ajuste.get("errores")
        ndof = resultado_ajuste.get("ndof")
        
        if params is None or errores is None:
            raise ValueError(
                "resultado_ajuste must contain 'parametros' and 'errores'"
            )
        
        # Extract values according to structure (dict or array)
        if isinstance(params, dict):
            param_names = list(params.keys())
            param_vals = np.array([params[k] for k in param_names])
            error_vals = np.array([errores[k] for k in param_names])
        else:
            param_names = [f"p{i}" for i in range(len(params))]
            param_vals = np.asarray(params)
            error_vals = np.asarray(errores)
        
        alpha = 1 - nivel
        
        # Determine quantile (Student‑t or normal)
        if ndof is not None and ndof > 0 and ndof <= 30:
            cuantil = stats.t.ppf(1 - alpha / 2, ndof)
            dist = "t"
        else:
            cuantil = stats.norm.ppf(1 - alpha / 2)
            dist = "normal"
        
        ic_list = []
        for nombre, val, err in zip(param_names, param_vals, error_vals):
            margen = cuantil * err
            ic_list.append({
                "nombre": nombre,
                "estimacion": float(val),
                "error": float(err),
                "inferior": float(val - margen),
                "superior": float(val + margen),
                "nivel": nivel,
                "distribucion": dist,
            })
        
        return {"parametros": ic_list}

    # ---------- A.2 Model prediction uncertainty ----------
    @staticmethod
    def incertidumbre_prediccion(resultado_ajuste, modelo, x0):
        """
        Compute the statistical uncertainty of the model prediction at x0.
        
        INPUT:
            resultado_ajuste: dict result from ajuste()/ajuste_lineal()/ajuste_polinomico()
            modelo: callable f(x, *params) | sympy.Expr
            x0: float | array_like -> prediction point(s)
        
        OUTPUT:
            If x0 is scalar:
                dict with:
                    "x": float,
                    "y": float,
                    "sigma_modelo": float,
            If x0 is array:
                dict with:
                    "x": array,
                    "y": array,
                    "sigma_modelo": array
        
        NOTAS:
                        - Computes ONLY parameter uncertainty
                        - DOES NOT include experimental instrument error (sy)
                        - Uses error propagation: Var(f) = grad_f^T · Cov · grad_f
                        - Gradient is computed symbolically (if expr) or numerically
                        - WARNING: this is uncertainty of the MEAN (confidence band),
                            NOT a prediction interval (prediction band)
        """
        params = resultado_ajuste.get("parametros")
        covarianza = resultado_ajuste.get("covarianza")
        
        if params is None or covarianza is None:
            raise ValueError(
                "resultado_ajuste must contain 'parametros' and 'covarianza'"
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
        if isinstance(modelo, sp.Expr):
            # Symbolic case: analytic derivative
            expr = modelo
            param_symbols = resultado_ajuste.get("parametros_simbolicos")
            
            if param_symbols is None:
                raise ValueError(
                    "resultado_ajuste must contain 'parametros_simbolicos' "
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
            y_pred = f_eval(x0_arr, *param_vals)
            
            # Gradient with respect to parameters
            grad_f = np.array([
                sp.lambdify((var_symbol, *param_symbols), sp.diff(expr, p), "numpy")
                for p in param_symbols
            ])
            grad_vals = np.array([
                grad_f[i](x0_arr, *param_vals) for i in range(len(param_symbols))
            ])
        else:
            # Numeric case: numerical derivative
            y_pred = modelo(x0_arr, *param_vals)
            
            # Numerical gradient by finite differences
            eps = np.sqrt(np.finfo(float).eps)
            grad_vals = np.zeros((len(param_vals), len(x0_arr)))
            
            for i in range(len(param_vals)):
                p_plus = param_vals.copy()
                p_plus[i] += eps
                p_minus = param_vals.copy()
                p_minus[i] -= eps
                
                grad_vals[i] = (
                    modelo(x0_arr, *p_plus) - modelo(x0_arr, *p_minus)
                ) / (2 * eps)
        
        # Error propagation: Var(f) = grad_f^T · Cov · grad_f
        sigma_modelo = np.sqrt(
            np.sum(grad_vals * (covarianza @ grad_vals), axis=0)
        )
        
        result = {
            "x": float(x0_arr[0]) if is_scalar else x0_arr,
            "y": float(y_pred[0]) if is_scalar else y_pred,
            "sigma_modelo": float(sigma_modelo[0]) if is_scalar else sigma_modelo,
        }
        
        return result

ajustes = _Ajustes()

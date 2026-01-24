import numpy as np
from scipy import stats, optimize
import sympy as sp

class _Ajustes:

    # ---------- Lineal ----------
    @staticmethod
    def ajuste_lineal(x, y, sy=None):
        x = np.asarray(x)
        y = np.asarray(y)
        if sy is None:
            sy = np.ones_like(y)
        w = 1 / sy**2

        S = np.sum(w)
        Sx = np.sum(w * x)
        Sy = np.sum(w * y)
        Sxx = np.sum(w * x * x)
        Sxy = np.sum(w * x * y)

        denom = S * Sxx - Sx**2
        a = (Sxx * Sy - Sx * Sxy) / denom
        b = (S * Sxy - Sx * Sy) / denom

        # Incertidumbres (covarianza) en el caso de sigma absoluta conocida
        # Varianzas de los parámetros en ajuste lineal ponderado:
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
            "a": float(a),
            "b": float(b),
            "sa": sa,
            "sb": sb,
            "covarianza": cov,
            "chi2": float(chi2),
            "ndof": ndof,
            "chi2_red": float(chi2 / ndof),
            "p": float(p),
            "yfit": yfit,
        }

    # ---------- Polinómico ----------
    @staticmethod
    def ajuste_polinomico(x, y, grado, sy=None):
        if sy is None:
            sy = np.ones_like(y)
        coef, cov = np.polyfit(x, y, grado, w=1/sy, cov=True)
        yfit = np.polyval(coef, x)
        chi2 = np.sum(((y - yfit) / sy)**2)
        ndof = len(x) - (grado + 1)
        p = stats.chi2.sf(chi2, ndof)
        return {
            "coeficientes": coef,
            "covarianza": cov,
            "yfit": yfit,
            "chi2": chi2,
            "ndof": ndof,
            "p": p,
        }

    # ---------- Genérico ----------
    @staticmethod
    def ajuste_generico(f, x, y, sy=None, p0=None):
        x = np.asarray(x)
        y = np.asarray(y)
        if sy is None:
            sy = np.ones_like(y)

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

    # ---------- Simbólico ----------
    @staticmethod
    def ajuste_simbolico(expr_str, x, y, sy=None, p0=None):
        x = np.asarray(x)
        y = np.asarray(y)
        if sy is None:
            sy = np.ones_like(y)

        xs = sp.symbols("x")
        expr = sp.sympify(expr_str)

        params = sorted(expr.free_symbols - {xs}, key=lambda s: s.name)

        if p0 is None:
            p0 = np.ones(len(params))

        f = sp.lambdify((xs, *params), expr, "numpy")
        def f_num(x, *p):
            return f(x, *p)

        res = _Ajustes.ajuste_generico(f_num, x, y, sy, p0)
        res["parametros_simbolicos"] = params
        res["expresion"] = expr
        return res

ajustes = _Ajustes()

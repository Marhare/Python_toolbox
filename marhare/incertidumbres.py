"""
RESUMEN RÁPIDO (Funciones públicas)
----------------------------------
u
    INPUT:
        x: number | array-like -> valores nominales
        sigmax: number | array-like -> incertidumbre(s) (desviación típica)
    OUTPUT:
        ufloat | unp.uarray -> magnitud con incertidumbre
    ERRORES:
        ValueError -> sigmax incompatible con forma de x
    NOTAS:
        si x es escalar devuelve ufloat; si es array devuelve unp.uarray

propagacion_incertidumbre_sympy
    INPUT:
        f: sympy.Expr -> expresión simbólica
        vars_: list[sympy.Symbol] -> variables
        valores: dict[Symbol, object] -> valores nominales
        sigmas: dict[Symbol, float] -> incertidumbres
        cov: sympy.Matrix | None -> covarianzas (opcional)
        simplify: bool -> simplificar expresiones
    OUTPUT:
        dict -> expresiones simbólicas, valores numéricos y LaTeX
        claves: f, grad, Sigma, var_f, sigma_f, valor, incertidumbre
    ERRORES:
        ValueError -> datos faltantes, tamaños incompatibles, sigma negativa
        TypeError -> cov no es sympy.Matrix
    NOTAS:
        devuelve expr simbólica, valores numéricos y representación LaTeX si aplica
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

# Dependencia obligatoria: uncertainties SIEMPRE instalada
from uncertainties import ufloat
from uncertainties.core import AffineScalarFunc
import uncertainties.unumpy as unp

from .latex_tools import expr_to_latex, latex_tools
from .estadistica import estadistica
import sympy as sp






class _Incertidumbres:
    """Fachada de utilidades para magnitudes con incertidumbre.

    Objetivos del diseño:
    - Encapsular la posible dependencia con `uncertainties`.
    - Ofrecer una API mínima, clara y en español.
    - Integración directa con `latex_tools` (formato metrológico y tablas).
    """

    # --------- Construcción ---------
    @staticmethod
    def u(x, sigmax=0.0):
        """
        Construye una magnitud con incertidumbre (caso escalar o array) usando `uncertainties`.
        INPUT:
            x: number | array-like -> valores nominales
            sigmax: number | array-like -> incertidumbre(s) (desviación típica)
        OUTPUT:
            ufloat | unp.uarray -> magnitud con incertidumbre
        ERRORES:
            ValueError -> sigmax incompatible con forma de x
        NOTAS:
            si x es escalar devuelve ufloat; si es array devuelve unp.uarray

        Reglas:
        - Si x es escalar (0-D) -> devuelve `ufloat(x, sigmax)`.
        - Si x es array-like (>=1-D) -> devuelve `unp.uarray(x, sigmax)` (mismo shape).
        - `sigmax` puede ser escalar (se “broadcastea”) o array con la misma forma que `x`.

        Parámetros
        ----------
        x : number | array-like
            Valor(es) nominal(es).
        sigmax : number | array-like, default 0.0
            Incertidumbre(s) (desviación típica) asociada(s).

        Devuelve
        --------
        ufloat | unumpy.uarray (ndarray de objetos uncertainties)
        """
        # Convertimos x a ndarray para inspeccionar dimensionalidad (0-D vs N-D)
        x_arr = np.asarray(x)

        # Caso escalar (0-D): devolvemos ufloat
        if x_arr.ndim == 0:
            # Asegurar que sigmax es escalar numérico
            s = np.asarray(sigmax)
            if s.ndim != 0:
                raise ValueError("Para x escalar, sigmax debe ser escalar.")
            return ufloat(float(x_arr), float(s))

        # Caso array (>=1-D): construimos uarray con broadcasting controlado
        x_arr = x_arr.astype(float, copy=False)

        s_arr = np.asarray(sigmax)
        if s_arr.ndim == 0:
            # Broadcast escalar a la forma de x
            s_arr = np.full_like(x_arr, float(s_arr), dtype=float)
        else:
            s_arr = s_arr.astype(float, copy=False)
            if s_arr.shape != x_arr.shape:
                # Intentamos broadcasting estándar; si no, error claro
                try:
                    s_arr = np.broadcast_to(s_arr, x_arr.shape).astype(float, copy=False)
                except Exception as e:
                    raise ValueError(
                        f"sigmax no es compatible con la forma de x: "
                        f"x.shape={x_arr.shape}, sigmax.shape={np.asarray(sigmax).shape}"
                    ) from e

        return unp.uarray(x_arr, s_arr)
    # --------- Propagación de incertidumbres ---------
    @staticmethod
    def propagacion_incertidumbre_sympy(
        f: sp.Expr,
        vars_: list[sp.Symbol],
        valores: dict[sp.Symbol, object],
        sigmas: dict[sp.Symbol, float],
        cov: sp.Matrix | None = None,
        simplify: bool = True
    ) -> dict:
        import sympy as sp
        import numpy as np

        for v in vars_:
            if v not in valores:
                raise ValueError(f"Falta valor para {v}")
            if v not in sigmas:
                raise ValueError(f"Falta sigma para {v}")
            if sigmas[v] < 0:
                raise ValueError(f"Sigma negativa para {v}")

        # Gradiente
        grad = sp.Matrix([sp.diff(f, v) for v in vars_])

        # Covarianzas
        if cov is None:
            Sigma = sp.diag(*[sigmas[v]**2 for v in vars_])
        else:
            if cov.shape != (len(vars_), len(vars_)):
                raise ValueError("Dimensiones incorrectas de cov")
            Sigma = cov

        var_f = (grad.T * Sigma * grad)[0]
        sigma_f = sp.sqrt(var_f)

        if simplify:
            var_f = sp.simplify(var_f)
            sigma_f = sp.simplify(sigma_f)

        f_num = sp.lambdify(vars_, f, "numpy")
        s_num = sp.lambdify(vars_, sigma_f, "numpy")

        args = [valores[v] for v in vars_]

        return {
            "valor": f_num(*args),
            "sigma": s_num(*args),
            "expr_latex": sp.latex(f),
            "sigma_latex": sp.latex(sigma_f),
        }

def propagar(expr, valores: dict, sigmas: dict, simplify=True):
    """
    Propagación de incertidumbres ROBUSTA (sin errores de orden).

    expr     : sympy.Expr
    valores  : dict {Symbol: array | escalar}
    sigmas   : dict {Symbol: float}
    """
    import numpy as np

    symbols = list(expr.free_symbols)

    # Validaciones
    for s in symbols:
        if s not in valores:
            raise ValueError(f"Falta valor para {s}")
        if s not in sigmas:
            raise ValueError(f"Falta sigma para {s}")

    # Vectorialidad: si alguna entrada es array
    vectorial = any(np.ndim(v) > 0 for v in valores.values())
    if vectorial:
        longitudes = [len(v) for v in valores.values() if np.ndim(v) > 0]
        N = max(longitudes) if longitudes else 1
    else:
        N = 1

    f_vals = []
    s_vals = []

    # Bucle principal (aquí vive i)
    for i in range(N):
        vals_i = {}
        for s in symbols:
            v = valores[s]
            if vectorial and np.ndim(v) > 0:
                vals_i[s] = v[i]
            else:
                vals_i[s] = v

        res = incertidumbres.propagacion_incertidumbre_sympy(
            expr,
            symbols,
            vals_i,
            sigmas,
            simplify=simplify
        )

        f_vals.append(res["valor"])
        s_vals.append(res["sigma"])

    return {
        "valor": np.array(f_vals) if vectorial else f_vals[0],
        "sigma": np.array(s_vals) if vectorial else s_vals[0],
        "expr_latex": res["expr_latex"],
        "sigma_latex": res["sigma_latex"],
    }



    # --------- Accesores ---------
    


incertidumbres = _Incertidumbres()

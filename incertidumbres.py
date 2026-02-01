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
    
    @staticmethod
    def propagacion_incertidumbre_sympy(
        f: sp.Expr,
        vars_: List[sp.Symbol],
        valores: Dict[sp.Symbol, object],
        sigmas: Dict[sp.Symbol, float],
        cov: Optional[sp.Matrix] = None,
        simplify: bool = True
    ) -> Dict[str, object]:
        """
        Propagación simbólica de incertidumbres con SymPy.
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
        import numpy as np
        import sympy as sp

        # --------------------------------------------------
        # Validaciones
        # --------------------------------------------------
        n = len(vars_)
        if n == 0:
            raise ValueError("vars_ no puede estar vacío.")

        for v in vars_:
            if v not in valores:
                raise ValueError(f"Falta valor nominal para {v}.")
            if v not in sigmas:
                raise ValueError(f"Falta sigma para {v}.")
            if sigmas[v] < 0:
                raise ValueError(f"Sigma negativa para {v}.")

        # --------------------------------------------------
        # Gradiente simbólico
        # --------------------------------------------------
        grad = sp.Matrix([sp.diff(f, v) for v in vars_])

        # --------------------------------------------------
        # Matriz de covarianzas
        # --------------------------------------------------
        if cov is None:
            sigma_syms = [sp.Symbol(f"sigma_{v}") for v in vars_]
            Sigma = sp.diag(*[s**2 for s in sigma_syms])
            sigma_vals = {s: float(sigmas[v]) for s, v in zip(sigma_syms, vars_)}
        else:
            if not isinstance(cov, sp.MatrixBase):
                raise TypeError("cov debe ser sympy.Matrix (o None).")
            if cov.shape != (n, n):
                raise ValueError(f"cov debe ser de tamaño {n}x{n}.")
            Sigma = cov
            sigma_syms = []
            sigma_vals = {}

        # --------------------------------------------------
        # Varianza propagada
        # --------------------------------------------------
        var_f = (grad.T * Sigma * grad)[0, 0]

        if simplify:
            grad_s = sp.simplify(grad)
            var_f_s = sp.simplify(var_f)
        else:
            grad_s, var_f_s = grad, var_f

        sigma_f_s = sp.sqrt(var_f_s)
        if simplify:
            sigma_f_s = sp.simplify(sigma_f_s)

        # --------------------------------------------------
        # Evaluación numérica (ESCALAR O VECTORIAL)
        # --------------------------------------------------
        f_num = sp.lambdify(vars_, f, modules="numpy")
        sigma_f_num = sp.lambdify(
            [*vars_, *sigma_syms],
            sigma_f_s,
            modules="numpy"
        )

        args_vals = [valores[v] for v in vars_]
        args_sigmas = [sigma_vals[s] for s in sigma_syms]

        f_val = f_num(*args_vals)
        sigma_f_val = sigma_f_num(*args_vals, *args_sigmas)

        # --------------------------------------------------
        # LaTeX (solo simbólico + valores escalares)
        # --------------------------------------------------
        latex = {
            "f": expr_to_latex(f, simplify=simplify),
            "grad": expr_to_latex(grad_s, simplify=False),
            "Sigma": expr_to_latex(Sigma, simplify=False),
            "var_f": expr_to_latex(var_f_s, simplify=simplify),
            "sigma_f": expr_to_latex(sigma_f_s, simplify=simplify),
        }

        latex_vals = None
        if np.isscalar(f_val):
            latex_vals = {
                "f_val": sp.latex(sp.N(f_val)),
                "sigma_f_val": sp.latex(sp.N(sigma_f_val)),
            }

        # --------------------------------------------------
        # Return
        # --------------------------------------------------
        return {
            "f": {"expr": f, "latex": latex["f"]},
            "grad": {"expr": grad_s, "latex": latex["grad"]},
            "Sigma": {"expr": Sigma, "latex": latex["Sigma"]},
            "var_f": {"expr": var_f_s, "latex": latex["var_f"]},
            "sigma_f": {"expr": sigma_f_s, "latex": latex["sigma_f"]},
            "valor": {"value": f_val, "latex": latex_vals["f_val"] if latex_vals else None},
            "incertidumbre": {
                "value": sigma_f_val,
                "latex": latex_vals["sigma_f_val"] if latex_vals else None,
            },
        }
def propagar(expr, valores, sigmas, simplify=True):
    """
    Propagación de incertidumbres directa y minimalista.

    expr     : expresión simbólica f(x1, x2, ..., xk)
    valores  : tupla (x1, x2, ..., xk)  [arrays o escalares]
    sigmas   : tupla (sx1, sx2, ..., sxk)

    Devuelve:
      dict con:
        - valor
        - sigma
        - expr
        - sigma_expr
        - sigma_latex
    """
    import numpy as np
    import sympy as sp

    symbols = sorted(expr.free_symbols, key=lambda s: s.name)

    if len(symbols) != len(valores):
        raise ValueError("Número de variables y valores no coincide")

    if len(sigmas) < len(valores):
        sigmas = list(sigmas) + [0.0] * (len(valores) - len(sigmas))

    first = valores[0]
    vectorial = np.ndim(first) > 0
    N = len(first) if vectorial else 1

    f_vals = []
    s_vals = []

    for i in range(N):
        vals_i = {
            s: v[i] if np.ndim(v) > 0 else v
            for s, v in zip(symbols, valores)
        }
        sigs_i = {
            s: sig
            for s, sig in zip(symbols, sigmas)
        }

        res = incertidumbres.propagacion_incertidumbre_sympy(
            expr,
            symbols,
            vals_i,
            sigs_i,
            simplify=simplify
        )

        f_vals.append(res["valor"])
        s_vals.append(res["incertidumbre"])

    return {
        "valor": np.array(f_vals) if vectorial else f_vals[0],
        "sigma": np.array(s_vals) if vectorial else s_vals[0],
        "expr": expr,
        "sigma_expr": res["sigma_f"]["expr"],
        "sigma_latex": res["sigma_f"]["latex"],
    }


    # --------- Accesores ---------
    


incertidumbres = _Incertidumbres()

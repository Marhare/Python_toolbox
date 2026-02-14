import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import newton


class _Numericos:
    """
    Unified scientific calculator with an auto‑detectable API.

    Provides basic mathematical operations (differentiation, integration,
    equation solving) that automatically choose between symbolic and numeric
    methods depending on the input type.
    """

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    @staticmethod
    def _to_symbol(var):
        """Convert var to sympy.Symbol if needed."""
        if isinstance(var, sp.Symbol):
            return var
        return sp.Symbol(str(var))

    @staticmethod
    def _to_expr(expr):
        """Convert expr to sympy.Expr when possible; None if callable."""
        if isinstance(expr, sp.Lambda):
            return expr.expr
        if isinstance(expr, sp.Expr):
            return expr
        if isinstance(expr, sp.Equality):
            return expr.lhs - expr.rhs
        if isinstance(expr, str):
            try:
                return sp.sympify(expr)
            except Exception:
                return None
        return None

    
    # =========================================================================
    # EVALUATION AND UTILITIES
    # =========================================================================

    @staticmethod
    def evaluar(expr, valores):
        """
        Evaluate an expression or function with specific values.

        INPUT:
            expr (sympy.Expr | str | callable): expression, string, or function.
            valores (dict | tuple | list): values for variables.
                - For sympy.Expr: dict with Symbol or str keys.
                - For callable: dict (by name) or tuple/list (positional).

        OUTPUT:
            float | ndarray: evaluated value.

        NOTES:
            For symbolic expressions with no variables, returns float.
            For callables, attempts name‑based or positional interpretation.
        """
        expr_sym = _Numericos._to_expr(expr)

        if expr_sym is not None:
            if not isinstance(valores, dict):
                raise TypeError(
                    "For sympy expressions, valores must be a dict"
                )
            mapping = {}
            for k, v in valores.items():
                key = _Numericos._to_symbol(k) if not isinstance(k, sp.Symbol) else k
                mapping[key] = v

            symbols = sorted(expr_sym.free_symbols, key=lambda s: s.name)
            if not symbols:
                return float(sp.N(expr_sym))

            f_num = sp.lambdify(symbols, expr_sym, "numpy")
            return f_num(*[mapping[s] for s in symbols])

        elif callable(expr):
            if isinstance(valores, dict):
                try:
                    return expr(**valores)
                except TypeError as exc:
                    raise TypeError(
                        "Could not evaluate the callable with kwargs. "
                        "Pass a positional list/tuple in 'valores'."
                    ) from exc
            elif isinstance(valores, (list, tuple)):
                return expr(*valores)
            else:
                return expr(valores)
        else:
            raise TypeError(
                "expr must be sympy.Expr, str, or callable"
            )

    # =========================================================================
    # ORDINARY DIFFERENTIAL EQUATIONS
    # =========================================================================

    


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

numericos = _Numericos()

# Module aliases: allow "from numericos import derivar" without friction
# All of these are static methods of _Numericos exposed at module level
derivar = _Numericos.derivar
derivada = _Numericos.derivada

integrar_indefinida = _Numericos.integrar_indefinida
integrar_definida = _Numericos.integrar_definida
integrar = _Numericos.integrar

resolver_ecuacion = _Numericos.resolver_ecuacion
raiz_numerica = _Numericos.raiz_numerica

evaluar = _Numericos.evaluar
rk4 = _Numericos.rk4

# Clear aliases for the user
__all__ = [
    "numericos",
    # Differentiation
    "derivar",
    "derivada",
    # Integration
    "integrar_definida",
    "integrar_indefinida",
    "integrar",
    # Solving
    "resolver_ecuacion",
    "raiz_numerica",
    # Evaluation
    "evaluar",
    # ODEs
    "rk4",
]

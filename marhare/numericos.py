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

    @staticmethod
    def _derivada_numerica(f, x, h=1e-5):
        """
        Centered numerical derivative (second‑order finite differences).

        INPUT:
            f (callable): evaluable function.
            x (float): evaluation point.
            h (float): differentiation step.

        OUTPUT:
            float: approximate derivative at x.
        """
        return (f(x + h) - f(x - h)) / (2 * h)

    @staticmethod
    def _integral_numerica(f, a, b):
        """
        Definite numerical integration via scipy.integrate.quad.

        INPUT:
            f (callable): function to integrate.
            a (float): lower limit.
            b (float): upper limit.

        OUTPUT:
            float: value of the definite integral.
        """
        val, _ = quad(f, a, b)
        return val

    # =========================================================================
    # DIFFERENTIATION (UNIFIED)
    # =========================================================================

    @staticmethod
    def derivar(expr, var, *, metodo="auto", h=1e-5):
        """
        Derivative with auto‑detected method (symbolic or numeric).

        INPUT:
            expr (sympy.Expr | str | callable): expression, string, or function.
            var (sympy.Symbol | str): differentiation variable.
            metodo (str): "auto" | "simbolico" | "numerico" (default: "auto").
            h (float): step for numerical differentiation (default: 1e-5).

        OUTPUT:
            sympy.Expr (symbolic) | callable (numeric).

        NOTES:
            metodo="auto":
            - If expr is sympy.Expr or str → symbolic derivative.
            - If expr is callable → numerical derivative (returns callable).

            metodo="simbolico":
            - Uses sympy.diff exclusively.
            - Raises error if expr is not symbolic.

            metodo="numerico":
            - Uses numerical differentiation only.
            - Requires expr callable.
        """
        var_sym = _Numericos._to_symbol(var)
        expr_sym = _Numericos._to_expr(expr)

        if metodo == "auto":
            if expr_sym is not None:
                return sp.diff(expr_sym, var_sym)
            elif callable(expr):
                return lambda x: _Numericos._derivada_numerica(expr, x, h)
            else:
                raise TypeError(
                    "expr must be sympy.Expr, str, or callable"
                )

        elif metodo == "simbolico":
            if expr_sym is None:
                raise ValueError(
                    f"metodo='simbolico' requires a symbolic expr, "
                    f"but expr is {type(expr).__name__}"
                )
            return sp.diff(expr_sym, var_sym)

        elif metodo == "numerico":
            if not callable(expr):
                raise ValueError(
                    f"metodo='numerico' requires callable expr, "
                    f"but expr is {type(expr).__name__}"
                )
            return lambda x: _Numericos._derivada_numerica(expr, x, h)

        else:
            raise ValueError(
                f"metodo must be 'auto', 'simbolico' or 'numerico', "
                f"no {metodo}"
            )

    @staticmethod
    def derivada(f, x, h=1e-5):
        """
        [DEPRECATED] Use derivar(f, "x", metodo="numerico") instead.

        Centered numerical derivative at a point.

        INPUT:
            f (callable): function to differentiate.
            x (float): evaluation point.
            h (float): differentiation step.

        OUTPUT:
            float: approximate derivative at x.

        NOTES:
            Centered finite‑difference method of order O(h²).
            Keeps compatibility with legacy code.
        """
        return _Numericos._derivada_numerica(f, x, h)

    # =========================================================================
    # INTEGRATION
    # =========================================================================

    @staticmethod
    def integrar_indefinida(expr, var):
        """
        Indefinite integral: symbolic if possible, accumulated if numeric.

        INPUT:
            expr (sympy.Expr | str | callable): expression, string, or function.
            var (sympy.Symbol | str): integration variable.

        OUTPUT:
            sympy.Expr (symbolic) | callable (numeric).

                NOTES:
                        If expr is sympy.Expr or str:
                        - Returns the symbolic antiderivative F(var).

                        If expr is callable:
                        - Returns a function F(x) = ∫₀ˣ f(t) dt (accumulated integral from 0).
                        - WARNING: this is NOT a general antiderivative, but a definite
                            integral from 0. For a true antiderivative, use integrar_definida.
        """
        var_sym = _Numericos._to_symbol(var)
        expr_sym = _Numericos._to_expr(expr)

        if expr_sym is not None:
            return sp.integrate(expr_sym, var_sym)
        elif callable(expr):
            return lambda x: _Numericos._integral_numerica(expr, 0.0, x)
        else:
            raise TypeError(
                "expr must be sympy.Expr, str, or callable"
            )

    @staticmethod
    def integrar_definida(expr, var, a, b):
        """
        Definite integral: symbolic if possible, numeric otherwise.

        INPUT:
            expr (sympy.Expr | str | callable): expression, string, or function.
            var (sympy.Symbol | str): integration variable.
            a (float | sympy.Expr): lower limit (must be numerically evaluable).
            b (float | sympy.Expr): upper limit (must be numerically evaluable).

        OUTPUT:
            float | sympy.Expr: integral value.

        NOTES:
            If expr is symbolic and the integral is solvable → returns sympy.Expr.
            If the symbolic integral does not converge:
            - If a, b are numerically evaluable → use numerical integration.
            - If a, b contain symbols → return unresolved sp.Integral.
            If expr is callable → use numerical integration directly.
        """
        var_sym = _Numericos._to_symbol(var)
        expr_sym = _Numericos._to_expr(expr)

        if expr_sym is not None:
            res = sp.integrate(expr_sym, (var_sym, a, b))
            if isinstance(res, sp.Integral):
                # Only numeric fallback if a and b are evaluable numbers
                a_is_evaluable = (a.is_number if isinstance(a, sp.Expr) else True)
                b_is_evaluable = (b.is_number if isinstance(b, sp.Expr) else True)
                
                if a_is_evaluable and b_is_evaluable:
                    try:
                        a_num = float(sp.N(a))
                        b_num = float(sp.N(b))
                        f_num = sp.lambdify(var_sym, expr_sym, "numpy")
                        return _Numericos._integral_numerica(f_num, a_num, b_num)
                    except (TypeError, ValueError):
                        # If evaluation fails, return the symbolic Integral
                        return res
                # If a or b are non‑evaluable symbols, return unresolved Integral
                return res
            return res
        elif callable(expr):
            return _Numericos._integral_numerica(expr, a, b)
        else:
            raise TypeError(
                "expr must be sympy.Expr, str, or callable"
            )

    @staticmethod
    def integrar(f, a, b):
        """
        [DEPRECATED] Use integrar_definida(f, "x", a, b) instead.

        Definite numerical integration.

        INPUT:
            f (callable): function to integrate.
            a (float): lower limit.
            b (float): upper limit.

        OUTPUT:
            float: value of the integral.

        NOTES:
            Wrapper around scipy.integrate.quad.
            Keeps compatibility with legacy code.
        """
        return _Numericos._integral_numerica(f, a, b)

    # =========================================================================
    # EQUATION SOLVING
    # =========================================================================

    @staticmethod
    def resolver_ecuacion(expr, var, x0=0.0):
        """
        Solve expr(var) = 0 symbolically or numerically.

        INPUT:
            expr (sympy.Expr | sympy.Eq | str | callable): equation.
            var (sympy.Symbol | str): main variable.
            x0 (float): initial guess for numeric methods (default: 0.0).

        OUTPUT:
            list (symbolic) | float (numeric): solution(s).

        NOTES:
            Method hierarchy:
            1. sympy.solve → exact symbolic solutions.
            2. sympy.nsolve → symbolic numeric from x0.
            3. Numeric Newton method → fallback.

            If expr is callable, uses Newton only.
        """
        var_sym = _Numericos._to_symbol(var)
        expr_sym = _Numericos._to_expr(expr)

        if expr_sym is not None:
            # Try exact symbolic solution
            try:
                sols = sp.solve(expr_sym, var_sym)
                if sols:
                    return sols
            except Exception:
                pass

            # Try symbolic numeric solution
            try:
                return float(sp.nsolve(expr_sym, x0))
            except Exception:
                pass

            # Fallback to Newton method
            try:
                return _Numericos.raiz_numerica(expr_sym, x0)
            except Exception as exc:
                raise ValueError(
                    f"Could not solve the equation with x0={x0}"
                ) from exc

        elif callable(expr):
            return _Numericos.raiz_numerica(expr, x0)
        else:
            raise TypeError(
                "expr must be sympy.Expr, sympy.Eq, str, or callable"
            )

    @staticmethod
    def raiz_numerica(f, x0):
        """
        Numeric root via Newton's method with numerical derivative.

        INPUT:
            f (callable | sympy.Expr | str): function or expression.
            x0 (float): initial guess.

        OUTPUT:
            float: root found.

        NOTES:
            Uses scipy.optimize.newton.
            Derivative computed numerically.
        """
        expr_sym = _Numericos._to_expr(f)

        if expr_sym is not None:
            symbols = sorted(expr_sym.free_symbols, key=lambda s: s.name)
            if len(symbols) != 1:
                raise ValueError(
                    "The expression must be univariate (a single variable)"
                )
            f_num = sp.lambdify(symbols[0], expr_sym, "numpy")
        elif callable(f):
            f_num = f
        else:
            raise TypeError(
                "f must be sympy.Expr, str, or callable"
            )

        fprime = lambda x: _Numericos._derivada_numerica(f_num, x)
        return float(newton(f_num, x0, fprime=fprime))

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

    @staticmethod
    def rk4(f, t_span, y0, dt):
        """
        Explicit 4th‑order Runge‑Kutta integrator for ODE systems.

        INPUT:
            f (callable): derivative function f(t, y).
            t_span (tuple): (t_start, t_end).
            y0 (array-like): initial condition (scalar or vector).
            dt (float): time step.

        OUTPUT:
            dict: {"t": time array, "y": state array}.

        NOTES:
            dy/dt = f(t, y)
            4th‑order method with local error O(dt⁵).
        """
        t0, tf = t_span
        t_vals = np.arange(t0, tf + dt, dt)
        y = np.array(y0, float)
        ys = []

        for t in t_vals:
            ys.append(y.copy())
            k1 = f(t, y)
            k2 = f(t + dt / 2, y + dt * k1 / 2)
            k3 = f(t + dt / 2, y + dt * k2 / 2)
            k4 = f(t + dt, y + dt * k3)
            y += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return {"t": t_vals, "y": np.array(ys)}


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

import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import newton


class _Numericos:
    """
    Calculadora científica unificada con API auto-detectable.

    Proporciona operaciones matemáticas básicas (derivación, integración,
    resolución de ecuaciones) que automáticamente eligen entre métodos
    simbólicos y numéricos según el tipo de entrada.
    """

    # =========================================================================
    # HELPERS PRIVADOS
    # =========================================================================

    @staticmethod
    def _to_symbol(var):
        """Convierte var a sympy.Symbol si es necesario."""
        if isinstance(var, sp.Symbol):
            return var
        return sp.Symbol(str(var))

    @staticmethod
    def _to_expr(expr):
        """Convierte expr a sympy.Expr si es posible, None si es callable."""
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
        Derivada numérica centrada (diferencias finitas de orden 2).

        INPUT:
            f (callable): función evaluable.
            x (float): punto de evaluación.
            h (float): paso de derivación.

        OUTPUT:
            float: derivada aproximada en x.
        """
        return (f(x + h) - f(x - h)) / (2 * h)

    @staticmethod
    def _integral_numerica(f, a, b):
        """
        Integración numérica definida mediante scipy.integrate.quad.

        INPUT:
            f (callable): función a integrar.
            a (float): límite inferior.
            b (float): límite superior.

        OUTPUT:
            float: valor de la integral definida.
        """
        val, _ = quad(f, a, b)
        return val

    # =========================================================================
    # DERIVACIÓN (UNIFICADA)
    # =========================================================================

    @staticmethod
    def derivar(expr, var, *, metodo="auto", h=1e-5):
        """
        Derivada con auto-detección de método (simbólico o numérico).

        INPUT:
            expr (sympy.Expr | str | callable): expresión, string o función.
            var (sympy.Symbol | str): variable de derivación.
            metodo (str): "auto" | "simbolico" | "numerico" (default: "auto").
            h (float): paso para derivación numérica (default: 1e-5).

        OUTPUT:
            sympy.Expr (simbólico) | callable (numérico).

        NOTAS:
            metodo="auto":
            - Si expr es sympy.Expr o str → derivada simbólica.
            - Si expr es callable → derivada numérica (retorna callable).

            metodo="simbolico":
            - Usa exclusivamente sympy.diff.
            - Lanza error si expr no es simbólica.

            metodo="numerico":
            - Usa exclusivamente derivación numérica.
            - Requiere expr callable.
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
                    "expr debe ser sympy.Expr, str o callable"
                )

        elif metodo == "simbolico":
            if expr_sym is None:
                raise ValueError(
                    f"metodo='simbolico' requiere expr simbólica, "
                    f"pero expr es {type(expr).__name__}"
                )
            return sp.diff(expr_sym, var_sym)

        elif metodo == "numerico":
            if not callable(expr):
                raise ValueError(
                    f"metodo='numerico' requiere expr callable, "
                    f"pero expr es {type(expr).__name__}"
                )
            return lambda x: _Numericos._derivada_numerica(expr, x, h)

        else:
            raise ValueError(
                f"metodo debe ser 'auto', 'simbolico' o 'numerico', "
                f"no {metodo}"
            )

    @staticmethod
    def derivada(f, x, h=1e-5):
        """
        [DEPRECATED] Usa derivar(f, "x", metodo="numerico") en su lugar.

        Derivada numérica centrada en un punto.

        INPUT:
            f (callable): función a derivar.
            x (float): punto de evaluación.
            h (float): paso de derivación.

        OUTPUT:
            float: derivada aproximada en x.

        NOTAS:
            Método de diferencias centradas de orden O(h²).
            Mantiene compatibilidad con código legado.
        """
        return _Numericos._derivada_numerica(f, x, h)

    # =========================================================================
    # INTEGRACIÓN
    # =========================================================================

    @staticmethod
    def integrar_indefinida(expr, var):
        """
        Integral indefinida: simbólica si es posible, acumulada si numérica.

        INPUT:
            expr (sympy.Expr | str | callable): expresión, string o función.
            var (sympy.Symbol | str): variable de integración.

        OUTPUT:
            sympy.Expr (simbólica) | callable (numérica).

        NOTAS:
            Si expr es sympy.Expr o str:
            - Retorna la primitiva simbólica F(var).

            Si expr es callable:
            - Retorna una función F(x) = ∫₀ˣ f(t) dt (integral acumulada desde 0).
            - ADVERTENCIA: esto NO es una primitiva general, sino una integral
              definida desde 0. Para una primitiva real, usa integrar_definida.
        """
        var_sym = _Numericos._to_symbol(var)
        expr_sym = _Numericos._to_expr(expr)

        if expr_sym is not None:
            return sp.integrate(expr_sym, var_sym)
        elif callable(expr):
            return lambda x: _Numericos._integral_numerica(expr, 0.0, x)
        else:
            raise TypeError(
                "expr debe ser sympy.Expr, str o callable"
            )

    @staticmethod
    def integrar_definida(expr, var, a, b):
        """
        Integral definida: simbólica si es posible, numérica en caso contrario.

        INPUT:
            expr (sympy.Expr | str | callable): expresión, string o función.
            var (sympy.Symbol | str): variable de integración.
            a (float | sympy.Expr): límite inferior (debe ser evaluable numéricamente).
            b (float | sympy.Expr): límite superior (debe ser evaluable numéricamente).

        OUTPUT:
            float | sympy.Expr: valor de la integral.

        NOTAS:
            Si expr es simbólica y la integral es resoluble → retorna sympy.Expr.
            Si la integral simbólica no converge:
            - Si a, b son numéricamente evaluables → usa integración numérica.
            - Si a, b contienen símbolos → retorna sp.Integral sin resolver.
            Si expr es callable → usa integración numérica directamente.
        """
        var_sym = _Numericos._to_symbol(var)
        expr_sym = _Numericos._to_expr(expr)

        if expr_sym is not None:
            res = sp.integrate(expr_sym, (var_sym, a, b))
            if isinstance(res, sp.Integral):
                # Solo fallback numérico si a y b son números evaluables
                a_is_evaluable = (a.is_number if isinstance(a, sp.Expr) else True)
                b_is_evaluable = (b.is_number if isinstance(b, sp.Expr) else True)
                
                if a_is_evaluable and b_is_evaluable:
                    try:
                        a_num = float(sp.N(a))
                        b_num = float(sp.N(b))
                        f_num = sp.lambdify(var_sym, expr_sym, "numpy")
                        return _Numericos._integral_numerica(f_num, a_num, b_num)
                    except (TypeError, ValueError):
                        # Si la evaluación falla, retorna la Integral simbólica
                        return res
                # Si a o b son simbólicos no-evaluables, retorna Integral sin resolver
                return res
            return res
        elif callable(expr):
            return _Numericos._integral_numerica(expr, a, b)
        else:
            raise TypeError(
                "expr debe ser sympy.Expr, str o callable"
            )

    @staticmethod
    def integrar(f, a, b):
        """
        [DEPRECATED] Usa integrar_definida(f, "x", a, b) en su lugar.

        Integración numérica definida.

        INPUT:
            f (callable): función a integrar.
            a (float): límite inferior.
            b (float): límite superior.

        OUTPUT:
            float: valor de la integral.

        NOTAS:
            Envoltorio de scipy.integrate.quad.
            Mantiene compatibilidad con código legado.
        """
        return _Numericos._integral_numerica(f, a, b)

    # =========================================================================
    # RESOLUCIÓN DE ECUACIONES
    # =========================================================================

    @staticmethod
    def resolver_ecuacion(expr, var, x0=0.0):
        """
        Resuelve expr(var) = 0 simbólicamente o numéricamente.

        INPUT:
            expr (sympy.Expr | sympy.Eq | str | callable): ecuación.
            var (sympy.Symbol | str): variable principal.
            x0 (float): aproximación inicial para métodos numéricos (default: 0.0).

        OUTPUT:
            list (simbólica) | float (numérica): solución(es).

        NOTAS:
            Jerarquía de métodos:
            1. sympy.solve → soluciones simbólicas exactas.
            2. sympy.nsolve → numérica simbólica desde x0.
            3. Método de Newton numérico → fallback.

            Si expr es callable, usa únicamente método de Newton.
        """
        var_sym = _Numericos._to_symbol(var)
        expr_sym = _Numericos._to_expr(expr)

        if expr_sym is not None:
            # Intenta solución simbólica exacta
            try:
                sols = sp.solve(expr_sym, var_sym)
                if sols:
                    return sols
            except Exception:
                pass

            # Intenta solución simbólica numérica
            try:
                return float(sp.nsolve(expr_sym, x0))
            except Exception:
                pass

            # Fallback a método de Newton
            try:
                return _Numericos.raiz_numerica(expr_sym, x0)
            except Exception as exc:
                raise ValueError(
                    f"No se pudo resolver la ecuación con x0={x0}"
                ) from exc

        elif callable(expr):
            return _Numericos.raiz_numerica(expr, x0)
        else:
            raise TypeError(
                "expr debe ser sympy.Expr, sympy.Eq, str o callable"
            )

    @staticmethod
    def raiz_numerica(f, x0):
        """
        Raíz numérica mediante método de Newton con derivada numérica.

        INPUT:
            f (callable | sympy.Expr | str): función o expresión.
            x0 (float): aproximación inicial.

        OUTPUT:
            float: raíz encontrada.

        NOTAS:
            Usa scipy.optimize.newton.
            Derivada calculada numéricamente.
        """
        expr_sym = _Numericos._to_expr(f)

        if expr_sym is not None:
            symbols = sorted(expr_sym.free_symbols, key=lambda s: s.name)
            if len(symbols) != 1:
                raise ValueError(
                    "La expresión debe ser univariada (una única variable)"
                )
            f_num = sp.lambdify(symbols[0], expr_sym, "numpy")
        elif callable(f):
            f_num = f
        else:
            raise TypeError(
                "f debe ser sympy.Expr, str o callable"
            )

        fprime = lambda x: _Numericos._derivada_numerica(f_num, x)
        return float(newton(f_num, x0, fprime=fprime))

    # =========================================================================
    # EVALUACIÓN Y UTILIDADES
    # =========================================================================

    @staticmethod
    def evaluar(expr, valores):
        """
        Evalúa una expresión o función con valores específicos.

        INPUT:
            expr (sympy.Expr | str | callable): expresión, string o función.
            valores (dict | tuple | list): valores para variables.
                - Para sympy.Expr: dict con claves Symbol o str.
                - Para callable: dict (por nombre) o tuple/list (posicional).

        OUTPUT:
            float | ndarray: valor evaluado.

        NOTAS:
            Para expresiones simbólicas sin variables, retorna float.
            Para callables, intenta interpretación por nombre o posición.
        """
        expr_sym = _Numericos._to_expr(expr)

        if expr_sym is not None:
            if not isinstance(valores, dict):
                raise TypeError(
                    "Para expresiones sympy, valores debe ser dict"
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
                        "No se pudo evaluar el callable con kwargs. "
                        "Pasa una lista/tupla posicional en 'valores'."
                    ) from exc
            elif isinstance(valores, (list, tuple)):
                return expr(*valores)
            else:
                return expr(valores)
        else:
            raise TypeError(
                "expr debe ser sympy.Expr, str o callable"
            )

    # =========================================================================
    # ECUACIONES DIFERENCIALES ORDINARIAS
    # =========================================================================

    @staticmethod
    def rk4(f, t_span, y0, dt):
        """
        Integrador explícito de Runge-Kutta orden 4 para sistemas EDOs.

        INPUT:
            f (callable): función derivada f(t, y).
            t_span (tuple): (t_inicial, t_final).
            y0 (array-like): condición inicial (escalar o vector).
            dt (float): paso de tiempo.

        OUTPUT:
            dict: {"t": array de tiempos, "y": array de estados}.

        NOTAS:
            dy/dt = f(t, y)
            Método de orden 4 con error local O(dt⁵).
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
# INTERFAZ PÚBLICA
# =============================================================================

numericos = _Numericos()

# Aliases de módulo: permiten "from numericos import derivar" sin fricción
# Todos estos son métodos estáticos de _Numericos expuestos a nivel de módulo
derivar = _Numericos.derivar
derivada = _Numericos.derivada

integrar_indefinida = _Numericos.integrar_indefinida
integrar_definida = _Numericos.integrar_definida
integrar = _Numericos.integrar

resolver_ecuacion = _Numericos.resolver_ecuacion
raiz_numerica = _Numericos.raiz_numerica

evaluar = _Numericos.evaluar
rk4 = _Numericos.rk4

# Aliases claros para el usuario
__all__ = [
    "numericos",
    # Derivación
    "derivar",
    "derivada",
    # Integración
    "integrar_definida",
    "integrar_indefinida",
    "integrar",
    # Resolución
    "resolver_ecuacion",
    "raiz_numerica",
    # Evaluación
    "evaluar",
    # EDOs
    "rk4",
]

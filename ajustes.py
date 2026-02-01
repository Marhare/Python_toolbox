"""
ajustes.py
==========

PROPÓSITO Y DIFERENCIAS:
- Módulo de AJUSTE DE CURVAS por mínimos cuadrados ponderados (WLS).
- ajustes.py: ajusta modelos a datos (lineal, polinómico, genérico, simbólico).
- estadistica.py: test de hipótesis, p-valores, intervalos de confianza sobre distribuciones.
- incertidumbres.py: propaga incertidumbres de parámetros hacia observables (derivadas, etc.).
- Este módulo COMBINA WLS + matriz de covarianzas para permitir posterior propagación via incertidumbres.py.

HIPÓTESIS POR DEFECTO:
- Los errores sy se interpretan como incertidumbres absolutas conocidas en y.
- Se asume absolute_sigma=True (no reescalar errores automáticamente).
- Los residuos se asumen independientes y gaussianos.
- Validez estadística (normalidad, homocedasticidad) es responsabilidad del usuario.


FUNCIONES PÚBLICAS
==================

1) ajuste_lineal(x, y, sy=None)
   Ajuste y = a + b·x por WLS analítico exacto.
   
   INPUT:
       x: array_like (n,) -> variable independiente
       y: array_like (n,) -> variable dependiente
       sy: array_like (n,) | None -> incertidumbre absoluta de y; si None → sy=1 (todas iguales)
   
   OUTPUT: dict con
       "parametros": {"a": float, "b": float}  -> intercepto y pendiente
       "errores": {"sa": float, "sb": float}   -> err. estándar de parámetros
       "covarianza": array (2,2)               -> matriz covarianzas(a,b)
       "yfit": array (n,)                      -> valores ajustados
       "chi2": float                           -> suma residuos²/sy²
       "ndof": int                             -> grados libertad = n - 2
       "chi2_red": float                       -> chi2 / ndof
       "p": float                              -> p-valor (1 - CDF_chi2(chi2,ndof))
   
   NOTAS:
       - Usa formulación analítica (no optimización numérica).
       - absolute_sigma=True implícito.
   
   ERRORES:
       - ValueError si len(x) != len(y)
       - ValueError si sy contiene valores <= 0 o forma distinta a y
       - ValueError si x,y no son arrays 1D

2) ajuste_polinomico(x, y, grado, sy=None)
   Ajuste polinómico y = p0·x^n + ... + pn por WLS.
   
   INPUT:
       x: array_like (n,) -> variable independiente
       y: array_like (n,) -> variable dependiente
       grado: int         -> grado del polinomio (grado >= 0)
       sy: array_like (n,) | None -> incertidumbre absoluta de y; si None → sy=1
   
   OUTPUT: dict con
       "coeficientes": array (grado+1,)       -> coef. en orden descendente (x^n, ..., x^0)
       "errores": array (grado+1,)            -> err. estándar de cada coef.
       "covarianza": array (grado+1,grado+1)  -> matriz covarianzas
       "yfit": array (n,)                     -> valores ajustados
       "chi2": float                          -> suma residuos²/sy²
       "ndof": int                            -> grados libertad = n - (grado+1)
       "chi2_red": float                      -> chi2 / ndof
       "p": float                             -> p-valor
   
   NOTAS:
       - Basado en np.polyfit con pesos 1/sy².
       - Coeficientes en orden descendente (polinomio estándar).
       - absolute_sigma=True implícito (covarianza sin reescalar).
   
   ERRORES:
       - ValueError si len(x) != len(y)
       - ValueError si sy <= 0 o forma distinta a y
       - ValueError si x,y no son arrays 1D
       - Error si grado < 0 o grado >= n

3) ajuste(modelo, x, y, sy=None, p0=None, *, variable="x")
   Ajuste genérico unificado: acepta callable O expresión sympy.
   
   INPUT:
       modelo: callable f(x, *params) | sympy.Expr
               Si callable: función que devuelve f(x, p1, p2, ...).
               Si sympy.Expr: expresión simbólica en "x" y parámetros (p, a, k, etc.).
       x: array_like (n,)  -> variable independiente
       y: array_like (n,)  -> variable dependiente
       sy: array_like (n,) | None -> incertidumbre absoluta de y
       p0: array_like (m,) | None -> valores iniciales de m parámetros
       variable: str -> nombre de variable independiente en expr (por defecto "x")
   
   OUTPUT: dict con
       "parametros": array (m,)                -> valores ajustados de parámetros
       "errores": array (m,)                   -> err. estándar de cada parámetro
       "covarianza": array (m,m)               -> matriz covarianzas
       "yfit": array (n,)                      -> valores ajustados
       "chi2": float                           -> suma residuos²/sy²
       "ndof": int                             -> grados libertad = n - m
       "chi2_red": float                       -> chi2 / ndof
       "p": float                              -> p-valor
       [si modelo es sympy.Expr, añade:]
       "expresion": sympy.Expr                 -> expresión original
       "parametros_simbolicos": list[sympy.Symbol] -> símbolos parámetros ordenados por nombre
   
   NOTAS:
       - Basado en scipy.optimize.curve_fit con absolute_sigma=True.
       - Si modelo es sympy.Expr: se lambdifica automáticamente.
       - Parámetros simbólicos detectados y ordenados alfabéticamente.
       - Si p0 es None: se intenta con p0=ones(m) (puede fallar si inicial pobre).
   
   ERRORES:
       - TypeError si modelo no es callable ni sympy.Expr
       - ValueError si variable no está en expr
       - ValueError si len(p0) != número de parámetros
       - ValueError si len(x) != len(y) o forma/valores de sy inválidos
       - RuntimeError si curve_fit no converge

4) intervalo_confianza_parametros(resultado_ajuste, nivel=0.95)
   Calcula intervalos de confianza (IC) para los parámetros ajustados.
   
   INPUT:
       resultado_ajuste: dict -> salida de ajuste()/ajuste_lineal()/ajuste_polinomico()
       nivel: float in (0,1) -> nivel de confianza (por defecto 0.95 → 95%)
   
   OUTPUT: dict con
       "parametros": list[dict] con una entrada por parámetro:
           {
               "nombre": str,           -> nombre del parámetro
               "estimacion": float,     -> valor ajustado
               "error": float,          -> err. estándar
               "inferior": float,       -> límite inferior del IC
               "superior": float,       -> límite superior del IC
               "nivel": float,          -> nivel de confianza usado
               "distribucion": "t" | "normal" -> cuantil empleado
           }
   
   NOTAS:
       - Usa dist. t-Student si ndof<=30, normal si ndof>30 o no existe.
       - IC = parametro ± cuantil * error
       - NO modifica el ajuste original.
       - NO es intervalo de predicción (eso incluiría sy del experimento).
   
   ERRORES:
       - ValueError si nivel no está en (0,1)
       - ValueError si resultado_ajuste no tiene 'parametros' o 'errores'

5) incertidumbre_prediccion(resultado_ajuste, modelo, x0)
   Calcula la incertidumbre estadística de la predicción del modelo en x0.
   
   INPUT:
       resultado_ajuste: dict -> salida de ajuste()/ajuste_lineal()/ajuste_polinomico()
       modelo: callable f(x, *params) | sympy.Expr -> mismo modelo usado en ajuste
       x0: float | array_like -> punto(s) donde evaluar la predicción
   
   OUTPUT: dict con
       "x": float | array -> punto(s) de evaluación
       "y": float | array -> predicción del modelo
       "sigma_modelo": float | array -> incertidumbre estadística de la predicción
       (x y sigma_modelo escalares si x0 es escalar, arrays si x0 es array)
   
   NOTAS:
       - Calcula SOLO la incertidumbre propagada de parámetros.
       - NO incluye el error experimental sy del instrumento.
       - Fórmula: Var(f) = ∇f^T · Cov · ∇f
       - Gradiente simbólico si modelo es sympy.Expr, numérico si es callable.
       - ADVERTENCIA: esto es "confidence band" (incertidumbre de la media).
         Para "prediction band" (intervalo de predicción), hay que sumar sy experimental.
   
   ERRORES:
       - ValueError si resultado_ajuste no tiene 'parametros' o 'covarianza'
       - ValueError si modelo es sympy.Expr pero falta 'parametros_simbolicos'
       - RuntimeError si hay problemas en lambdify o derivación


CONVENCIONES Y FLUJO TÍPICO
============================

1. Ajustar datos:
   res = ajustes.ajuste_lineal(x, y, sy=sy_data)
   o
   res = ajustes.ajuste(modelo, x, y, sy=sy_data, p0=p0_inicial)

2. Interpretar resultados:
   params = res["parametros"]
   p_valor = res["p"]  -> si p > 0.05, ajuste es "aceptable"
   chi2_red = res["chi2_red"]  -> si ~1, ajuste es bueno

3. Construir IC sobre parámetros:
   ic = ajustes.intervalo_confianza_parametros(res, nivel=0.95)
   for param_ic in ic["parametros"]:
       print(f"{param_ic['nombre']}: {param_ic['inferior']} - {param_ic['superior']}")

4. Evaluar incertidumbre en predicción:
   pred = ajustes.incertidumbre_prediccion(res, modelo, x_nuevo)
   print(f"y({x_nuevo}) = {pred['y']} ± {pred['sigma_modelo']}")

5. Propagar a otros observables (usar incertidumbres.py):
   # Combinar parámetros ajustados + covarianzas → observable nuevo
   # (Ver módulo incertidumbres.py para detalles)
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
            raise ValueError("x e y deben tener la misma forma")
        if x.ndim != 1:
            raise ValueError("x e y deben ser arreglos 1D")
        if sy is None:
            sy = np.ones_like(y)
        else:
            sy = np.asarray(sy)
            if sy.shape != y.shape:
                raise ValueError("sy debe tener la misma forma que y")
            if np.any(sy <= 0):
                raise ValueError("sy debe ser positivo en todos los puntos")
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

    # ---------- Lineal ----------
    @staticmethod
    def ajuste_lineal(x, y, sy=None):
        """
        INPUT:
            x: array_like (n,)
            y: array_like (n,)
            sy: array_like (n,) | None  -> errores absolutos en y
        OUTPUT:
            dict con:
                - parametros: {"a": intercepto, "b": pendiente}
                - errores: {"sa": error_a, "sb": error_b}
                - covarianza: matriz 2x2
                - chi2, ndof, chi2_red, p
                - yfit
        NOTAS:
            - Ajuste lineal ponderado (WLS) analítico
            - Se asume sigma absoluta conocida (no reescalar errores)
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
            "parametros": {"a": float(a), "b": float(b)},
            "errores": {"sa": sa, "sb": sb},
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
        """
        INPUT:
            x: array_like (n,)
            y: array_like (n,)
            grado: int
            sy: array_like (n,) | None  -> errores absolutos en y
        OUTPUT:
            dict con:
                - coeficientes (orden descendente)
                - errores (sqrt(diag(cov)))
                - covarianza
                - chi2, ndof, chi2_red, p
                - yfit
        NOTAS:
            - Basado en np.polyfit con pesos
            - Se asume sigma absoluta conocida (no reescalar errores)
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

    # ---------- Unificado ----------
    @staticmethod
    def ajuste(modelo, x, y, sy=None, p0=None, *, variable="x"):
        """
        INPUT:
            modelo: callable f(x, *params) | sympy.Expr
            x: array_like (n,)
            y: array_like (n,)
            sy: array_like (n,) | None  -> errores absolutos en y
            p0: valores iniciales | None
            variable: str -> nombre de la variable independiente
        OUTPUT:
            dict con:
                - parametros, errores, covarianza, yfit
                - chi2, ndof, chi2_red, p
            Si el modelo es simbólico, añade:
                - expresion
                - parametros_simbolicos
        NOTAS:
            - Ajuste por mínimos cuadrados ponderados (curve_fit)
            - absolute_sigma=True siempre
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
                        "No se pudo identificar la variable independiente; "
                        "especifique el nombre con 'variable'"
                    )

            params = sorted(expr.free_symbols - {var_symbol}, key=lambda s: s.name)
            if p0 is not None and len(p0) != len(params):
                raise ValueError("p0 debe tener la misma longitud que los parámetros")
            if p0 is None:
                p0 = np.ones(len(params))

            f = sp.lambdify((var_symbol, *params), expr, "numpy")

            def f_num(x, *p):
                return f(x, *p)

            res = _Ajustes._ajuste_curvefit(f_num, x, y, sy, p0)
            res["expresion"] = expr
            res["parametros_simbolicos"] = params
            return res

        raise TypeError("modelo debe ser callable o sympy.Expr")

    # ---------- A.1 Intervalos de confianza de parámetros ----------
    @staticmethod
    def intervalo_confianza_parametros(resultado_ajuste, nivel=0.95):
        """
        Calcula intervalos de confianza para los parámetros ajustados.
        
        INPUT:
            resultado_ajuste: dict resultado de ajuste()/ajuste_lineal()/ajuste_polinomico()
            nivel: float [0, 1] -> nivel de confianza (por defecto 0.95)
        
        OUTPUT:
            dict con lista de intervalos:
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
            - Usa distribución t-Student si ndof está disponible y es pequeño
            - Usa distribución normal si ndof es muy grande (>30) o no existe
            - NO modifica el ajuste
            - IC de los parámetros, NO intervalo de predicción
        """
        if nivel <= 0 or nivel >= 1:
            raise ValueError("nivel debe estar en (0, 1)")
        
        params = resultado_ajuste.get("parametros")
        errores = resultado_ajuste.get("errores")
        ndof = resultado_ajuste.get("ndof")
        
        if params is None or errores is None:
            raise ValueError(
                "resultado_ajuste debe contener 'parametros' y 'errores'"
            )
        
        # Extraer valores según estructura (dict o array)
        if isinstance(params, dict):
            param_names = list(params.keys())
            param_vals = np.array([params[k] for k in param_names])
            error_vals = np.array([errores[k] for k in param_names])
        else:
            param_names = [f"p{i}" for i in range(len(params))]
            param_vals = np.asarray(params)
            error_vals = np.asarray(errores)
        
        alpha = 1 - nivel
        
        # Determinar cuantil (t-Student o normal)
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

    # ---------- A.2 Incertidumbre de predicción del modelo ----------
    @staticmethod
    def incertidumbre_prediccion(resultado_ajuste, modelo, x0):
        """
        Calcula la incertidumbre estadística de la predicción del modelo en x0.
        
        INPUT:
            resultado_ajuste: dict resultado de ajuste()/ajuste_lineal()/ajuste_polinomico()
            modelo: callable f(x, *params) | sympy.Expr
            x0: float | array_like -> punto(s) de predicción
        
        OUTPUT:
            Si x0 es escalar:
                dict con:
                    "x": float,
                    "y": float,
                    "sigma_modelo": float,
            Si x0 es array:
                dict con:
                    "x": array,
                    "y": array,
                    "sigma_modelo": array
        
        NOTAS:
            - Calcula SOLO la incertidumbre de los parámetros
            - NO incluye error experimental (sy) del instrumento
            - Usa propagación de errores: Var(f) = grad_f^T · Cov · grad_f
            - El gradiente se calcula simbólicamente (si hay expr) o numéricamente
            - ADVERTENCIA: esto es incertidumbre de la MEDIA (confidence band),
              NO intervalo de predicción (prediction band)
        """
        params = resultado_ajuste.get("parametros")
        covarianza = resultado_ajuste.get("covarianza")
        
        if params is None or covarianza is None:
            raise ValueError(
                "resultado_ajuste debe contener 'parametros' y 'covarianza'"
            )
        
        # Convertir params a array si es dict
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
        
        # Calcular predicción y gradiente
        if isinstance(modelo, sp.Expr):
            # Caso simbólico: derivada analítica
            expr = modelo
            param_symbols = resultado_ajuste.get("parametros_simbolicos")
            
            if param_symbols is None:
                raise ValueError(
                    "resultado_ajuste debe contener 'parametros_simbolicos' "
                    "si el modelo es simbólico"
                )
            
            # Detectar variable independiente
            var_symbol = None
            for s in expr.free_symbols:
                if s.name == "x":
                    var_symbol = s
                    break
            if var_symbol is None:
                if len(expr.free_symbols) == 1:
                    var_symbol = list(expr.free_symbols)[0]
            
            # Lambdify para predicción
            f_eval = sp.lambdify((var_symbol, *param_symbols), expr, "numpy")
            y_pred = f_eval(x0_arr, *param_vals)
            
            # Gradiente respecto a parámetros
            grad_f = np.array([
                sp.lambdify((var_symbol, *param_symbols), sp.diff(expr, p), "numpy")
                for p in param_symbols
            ])
            grad_vals = np.array([
                grad_f[i](x0_arr, *param_vals) for i in range(len(param_symbols))
            ])
        else:
            # Caso numérico: derivada numérica
            y_pred = modelo(x0_arr, *param_vals)
            
            # Gradiente numérico por diferencias finitas
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
        
        # Propagación de errores: Var(f) = grad_f^T · Cov · grad_f
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

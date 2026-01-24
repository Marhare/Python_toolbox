import numpy as np
from scipy import stats
from sympy import symbols, sympify, lambdify
from typing import Union, Tuple, Dict, Optional
import math

# Intenta importar uncertainties; si no está disponible, usa None
try:
    import uncertainties.unumpy as unp
    from uncertainties.core import AffineScalarFunc
    HAS_UNCERTAINTIES = True
except ImportError:
    HAS_UNCERTAINTIES = False
    unp = None
    AffineScalarFunc = None


def _es_ufloat_array(x: np.ndarray) -> bool:
    """Detecta si el array contiene ufloat (type=object y elementos son AffineScalarFunc)."""
    if not HAS_UNCERTAINTIES:
        return False
    return x.dtype == object and len(x) > 0 and isinstance(x.flat[0], AffineScalarFunc)


def _std_universal(x: np.ndarray, ddof: int = 1) -> Union[float, object]:
    """Desviación típica que funciona con float y ufloat arrays."""
    x = np.asarray(x)
    n = len(x)
    if n <= ddof:
        raise ValueError(f"Tamaño ({n}) debe ser > ddof ({ddof})")
    
    # Usar np.mean que funciona con ufloat
    media = np.mean(x)
    # Calcular suma de cuadrados de desviaciones
    # Esto propaga incertidumbres automáticamente
    suma_sq = np.sum((x - media) ** 2)
    # Retornar sqrt(suma / (n - ddof)) usando potenciación (funciona con ufloat)
    return (suma_sq / (n - ddof)) ** 0.5


def _var_universal(x: np.ndarray, ddof: int = 1) -> Union[float, object]:
    """Varianza que funciona con float y ufloat arrays."""
    x = np.asarray(x)
    n = len(x)
    if n <= ddof:
        raise ValueError(f"Tamaño ({n}) debe ser > ddof ({ddof})")
    
    # Usar np.mean que funciona con ufloat
    media = np.mean(x)
    # Calcular suma de cuadrados de desviaciones
    suma_sq = np.sum((x - media) ** 2)
    # Retornar suma / (n - ddof)
    return suma_sq / (n - ddof)


def _mean_universal(x: np.ndarray) -> Union[float, object]:
    """Media que funciona con float y ufloat arrays."""
    # np.mean propaga incertidumbres automáticamente en arrays de ufloat
    return np.mean(x)


class _Estadistica:
    """
    Herramientas estadísticas fundamentales para análisis físico.
    
    GUÍA RÁPIDA DE USO
    ==================
    
    from mi_toolbox.estadistica import estadistica
    import numpy as np
    import uncertainties.unumpy as unp
    
    # 1) Estadística descriptiva (compatible con ufloat)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    media = estadistica.media(x)
    sigma = estadistica.desviacion_tipica(x)
    
    # Con incertidumbres (propagación automática):
    x_u = unp.uarray([1.0, 2.0, 3.0], [0.1, 0.1, 0.1])
    media_u = estadistica.media(x_u)  # ufloat con incertidumbre
    
    # 2) Intervalos de confianza
    ic = estadistica.intervalo_media_sigma_desconocida(x, nivel=0.95)
    
    # 3) Tests estadísticos
    res = estadistica.test_media_t(x, mu0=3.0, alternativa="dos_colas")
    
    # 4) Propagación analítica de incertidumbres
    res = estadistica.propagacion_analitica(
        "sqrt(a*b)",
        valores={"a": 10, "b": 5},
        incertidumbres={"a": 0.5, "b": 0.2}
    )
    """

    # ---------- Descriptiva ----------
    @staticmethod
    def media(x: Union[list, np.ndarray]) -> Union[float, object]:
        """
        Calcula la media aritmética.
        
        Preserva ufloat si la entrada es array de ufloat (propagación automática).
        """
        x = np.asarray(x)
        if len(x) == 0:
            raise ValueError("Array vacío")
        return _mean_universal(x)

    @staticmethod
    def varianza(x: Union[list, np.ndarray], ddof: int = 1) -> Union[float, object]:
        """
        Calcula la varianza muestral.
        
        Preserva ufloat si la entrada es array de ufloat.
        ddof: grados de libertad (1 para muestra, 0 para población).
        """
        x = np.asarray(x)
        if len(x) == 0:
            raise ValueError("Array vacío")
        if len(x) <= ddof:
            raise ValueError(f"Tamaño de muestra ({len(x)}) debe ser > ddof ({ddof})")
        return _var_universal(x, ddof=ddof)

    @staticmethod
    def desviacion_tipica(x: Union[list, np.ndarray], ddof: int = 1) -> Union[float, object]:
        """
        Calcula la desviación típica (estándar) muestral.
        
        Preserva ufloat si la entrada es array de ufloat (propagación automática).
        ddof: grados de libertad (1 para muestra, 0 para población).
        """
        x = np.asarray(x)
        if len(x) == 0:
            raise ValueError("Array vacío")
        if len(x) <= ddof:
            raise ValueError(f"Tamaño de muestra ({len(x)}) debe ser > ddof ({ddof})")
        return _std_universal(x, ddof=ddof)

    @staticmethod
    def covarianza(x: Union[list, np.ndarray], y: Union[list, np.ndarray], ddof: int = 1) -> float:
        """Calcula la covarianza entre dos vectores."""
        x = np.asarray(x)
        y = np.asarray(y)
        if len(x) != len(y):
            raise ValueError("x e y deben tener la misma longitud")
        if len(x) == 0:
            raise ValueError("Arrays vacíos")
        cov_matrix = np.cov(x, y, ddof=ddof)
        return float(cov_matrix[0, 1])

    @staticmethod
    def correlacion(x: Union[list, np.ndarray], y: Union[list, np.ndarray]) -> float:
        """Calcula el coeficiente de correlación de Pearson."""
        x = np.asarray(x)
        y = np.asarray(y)
        if len(x) != len(y):
            raise ValueError("x e y deben tener la misma longitud")
        if len(x) == 0:
            raise ValueError("Arrays vacíos")
        corr_matrix = np.corrcoef(x, y)
        return float(corr_matrix[0, 1])

    @staticmethod
    def error_estandar(x: Union[list, np.ndarray]) -> Union[float, object]:
        """
        Calcula el error estándar de la media: σ / sqrt(n).
        
        Preserva ufloat si la entrada es array de ufloat.
        """
        x = np.asarray(x)
        n = len(x)
        if n == 0:
            raise ValueError("Array vacío")
        sigma = _std_universal(x, ddof=1)
        return sigma / (n ** 0.5)

    # ---------- Intervalos ----------
    @staticmethod
    def intervalo_media_sigma_conocida(
        x: Union[list, np.ndarray],
        sigma: float,
        nivel: float = 0.95
    ) -> Tuple[float, float]:
        """
        Intervalo de confianza para la media con σ conocida (distribución normal).
        
        Devuelve: (límite_inferior, límite_superior)
        """
        x = np.asarray(x)
        n = len(x)
        if n == 0:
            raise ValueError("Array vacío")
        if sigma <= 0:
            raise ValueError("sigma debe ser positivo")
        if not (0 < nivel < 1):
            raise ValueError("nivel debe estar entre 0 y 1")
        
        mu = float(np.mean(x))
        z = stats.norm.ppf(1 - (1 - nivel) / 2)
        Δ = z * sigma / np.sqrt(n)
        return float(mu - Δ), float(mu + Δ)

    @staticmethod
    def intervalo_media_sigma_desconocida(
        x: Union[list, np.ndarray],
        nivel: float = 0.95
    ) -> Tuple[float, float]:
        """
        Intervalo de confianza para la media con σ desconocida (distribución t-Student).
        
        Devuelve: (límite_inferior, límite_superior)
        """
        x = np.asarray(x)
        n = len(x)
        if n < 2:
            raise ValueError("Se necesitan al menos 2 observaciones")
        if not (0 < nivel < 1):
            raise ValueError("nivel debe estar entre 0 y 1")
        
        mu = float(_mean_universal(x))
        s = float(_std_universal(x, ddof=1))
        tcrit = stats.t.ppf(1 - (1 - nivel) / 2, n - 1)
        Δ = tcrit * s / (n ** 0.5)
        return float(mu - Δ), float(mu + Δ)

    @staticmethod
    def intervalo_varianza(
        x: Union[list, np.ndarray],
        nivel: float = 0.95
    ) -> Tuple[float, float]:
        """
        Intervalo de confianza para la varianza (distribución chi-cuadrado).
        
        Devuelve: (límite_inferior, límite_superior)
        """
        x = np.asarray(x)
        n = len(x)
        if n < 2:
            raise ValueError("Se necesitan al menos 2 observaciones")
        if not (0 < nivel < 1):
            raise ValueError("nivel debe estar entre 0 y 1")
        
        s2 = float(_var_universal(x, ddof=1))
        α = 1 - nivel
        χ2_sup = stats.chi2.ppf(α / 2, n - 1)
        χ2_inf = stats.chi2.ppf(1 - α / 2, n - 1)
        return float((n - 1) * s2 / χ2_inf), float((n - 1) * s2 / χ2_sup)

    # ---------- Tests ----------
    @staticmethod
    def test_media_t(
        x: Union[list, np.ndarray],
        mu0: float,
        alternativa: str = "dos_colas"
    ) -> Dict[str, Union[float, int]]:
        """
        Test t-Student para la media: H₀: μ = μ₀
        
        Parámetros:
        -----------
        x : datos
        mu0 : media hipotética
        alternativa : "dos_colas", "mayor" (μ > μ₀), o "menor" (μ < μ₀)
        
        Devuelve:
        ---------
        dict con claves: "t" (estadístico), "p" (p-valor), "df" (grados de libertad)
        """
        x = np.asarray(x)
        n = len(x)
        if n < 2:
            raise ValueError("Se necesitan al menos 2 observaciones")
        if alternativa not in ["dos_colas", "mayor", "menor"]:
            raise ValueError('alternativa debe ser "dos_colas", "mayor" o "menor"')
        
        mu = float(_mean_universal(x))
        s = float(_std_universal(x, ddof=1))
        t = (mu - mu0) / (s / (n ** 0.5))

        if alternativa == "dos_colas":
            p = 2 * stats.t.sf(abs(t), n - 1)
        elif alternativa == "mayor":
            p = stats.t.sf(t, n - 1)
        else:  # "menor"
            p = stats.t.cdf(t, n - 1)

        return {"t": float(t), "p": float(p), "df": n - 1}

    @staticmethod
    def test_ks(x: Union[list, np.ndarray], distribucion: str = "normal") -> Dict[str, float]:
        """
        Test de Kolmogórov-Smirnov para bondad de ajuste.
        
        Parámetros:
        -----------
        x : datos
        distribucion : "normal" (estima μ y σ), "uniforme", etc.
        
        Devuelve:
        ---------
        dict con "estadistico" y "p_valor"
        """
        x = np.asarray(x)
        if len(x) < 2:
            raise ValueError("Se necesitan al menos 2 observaciones")
        
        if distribucion == "normal":
            mu, sigma = np.mean(x), np.std(x)
            d, p = stats.kstest(x, "norm", args=(mu, sigma))
        elif distribucion == "uniforme":
            d, p = stats.kstest(x, "uniform")
        else:
            raise ValueError(f"Distribución {distribucion} no soportada")
        
        return {"estadistico": float(d), "p_valor": float(p)}

    # ---------- Propagación de incertidumbres ----------
    @staticmethod
    def propagacion_analitica(
        expr_str: str,
        valores: Dict[str, float],
        incertidumbres: Dict[str, float],
        covarianzas: Optional[Dict[Tuple[str, str], float]] = None
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Propagación analítica de incertidumbres mediante derivadas (1er orden).
        
        Útil para expresiones simbólicas simples o cuando quieres traza visible.
        Para operaciones numéricas complejas, considera usar `uncertainties.unumpy` directamente.
        
        Parámetros:
        -----------
        expr_str : string de la función, ej. "sqrt(x*y + sin(z))"
        valores : dict {"x": 2.0, "y": 3.5, "z": 1.2}
        incertidumbres : dict {"x": 0.05, "y": 0.10, "z": 0.02}
        covarianzas : dict opcional {("x", "y"): cov_xy, ...}
        
        Devuelve:
        ---------
        {"valor": float, "incertidumbre": float}
        
        Ejemplo:
        --------
        res = estadistica.propagacion_analitica(
            "sqrt(2*V/(B**2*r**2))",
            valores={"V": 100, "B": 0.5, "r": 0.01},
            incertidumbres={"V": 1, "B": 0.01, "r": 0.0001}
        )
        """
        # 1) Definir símbolos y parsear expresión
        nombres = list(valores.keys())
        simbolos_list = symbols(nombres)
        try:
            expr = sympify(expr_str)
        except Exception as e:
            raise ValueError(f"Error parsing expression '{expr_str}': {e}")

        # 2) Crear función numérica de f y sus derivadas parciales
        try:
            f = lambdify(simbolos_list, expr, "numpy")
            derivadas = [expr.diff(s) for s in simbolos_list]
            d_funcs = [lambdify(simbolos_list, d, "numpy") for d in derivadas]
        except Exception as e:
            raise ValueError(f"Error creating lambdify functions: {e}")

        # 3) Evaluar valor central
        vals = [valores[n] for n in nombres]
        try:
            y = float(f(*vals))
        except Exception as e:
            raise ValueError(f"Error evaluating function at {vals}: {e}")

        # 4) Evaluar derivadas en punto central
        try:
            dvals = np.array([df(*vals) for df in d_funcs], dtype=float)
        except Exception as e:
            raise ValueError(f"Error evaluating derivatives: {e}")

        # 5) Construir matriz de covarianzas
        n = len(nombres)
        C = np.zeros((n, n), dtype=float)
        for i, ni in enumerate(nombres):
            if ni not in incertidumbres:
                raise ValueError(f"Incertidumbre no especificada para {ni}")
            C[i, i] = incertidumbres[ni] ** 2
        
        if covarianzas is not None:
            for (ni, nj), cov in covarianzas.items():
                try:
                    i = nombres.index(ni)
                    j = nombres.index(nj)
                    C[i, j] = C[j, i] = cov
                except ValueError as e:
                    raise ValueError(f"Variable {ni} o {nj} no encontrada en valores: {e}")

        # 6) Propagación: u²(y) = ∇f^T C ∇f
        u2 = float(dvals @ C @ dvals.T)
        u = np.sqrt(max(u2, 0))  # Evita raíz de número negativo por errores numéricos

        return {"valor": y, "incertidumbre": u, "derivadas": dict(zip(nombres, dvals))}


estadistica = _Estadistica()

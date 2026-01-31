import numpy as np
from scipy import stats
from typing import Union, Tuple, Dict


# ============================================================================
# CLASE PRINCIPAL: ESTADISTICA
# ============================================================================

class _Estadistica:
    """
    Herramientas estadísticas fundamentales para análisis físico.
    
    ALCANCE Y ARQUITECTURA
    ======================
    
    Este módulo implementa cálculo estadístico puro con float/ndarray.
    Se enfoca EXCLUSIVAMENTE en incertidumbre Tipo A según ISO GUM.
    
    Clasificación de incertidumbres (ISO GUM):
    
    - **Tipo A**: Evaluada mediante análisis estadístico de observaciones repetidas.
      Ejemplos: error estándar de la media, incertidumbre de parámetros ajustados
      por mínimos cuadrados, desviación típica experimental.
      
    - **Tipo B**: Evaluada por información externa (no estadística).
      Ejemplos: resolución del instrumento, certificados de calibración,
      especificaciones del fabricante, juicio científico.
    
    **Este módulo (`estadistica.py`) calcula únicamente Tipo A.**
    
    La combinación de Tipo A + Tipo B, la propagación de incertidumbres totales,
    y la representación "valor ± incertidumbre" se realiza en `incertidumbres.py`
    usando la librería `uncertainties`.
    
    GUÍA RÁPIDA DE USO
    ==================
    
    from mi_toolbox.estadistica import estadistica
    import numpy as np
    
    # 1) Estadística descriptiva (Tipo A)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    media = estadistica.media(x)  # float
    sigma = estadistica.desviacion_tipica(x)  # float
    error_std = estadistica.error_estandar(x)  # float (incertidumbre Tipo A)
    
    # 2) Intervalos de confianza
    li, ls = estadistica.intervalo_media_sigma_desconocida(x, nivel=0.95)  # (float, float)
    
    # 3) Tests estadísticos
    res = estadistica.test_media_t(x, mu0=3.0, alternativa="dos_colas")  # Dict[str, float]
    """

    # ========================================================================
    # SECCIÓN 1: ESTADÍSTICA DESCRIPTIVA
    # ========================================================================
    
    @staticmethod
    def media(x: Union[list, np.ndarray]) -> float:
        """
        Calcula la media aritmética muestral.
        
        Parámetros
        ----------
        x : array_like
            Muestra de datos.
        
        Devuelve
        --------
        float
            Media aritmética.
        """
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            raise ValueError("Array vacío")
        return float(np.mean(x))

    @staticmethod
    def varianza(x: Union[list, np.ndarray], ddof: int = 1) -> float:
        """
        Calcula la varianza muestral.
        
        Parámetros
        ----------
        x : array_like
            Muestra de datos.
        ddof : int, default 1
            Grados de libertad (1 para muestra, 0 para población).
        
        Devuelve
        --------
        float
            Varianza muestral.
        """
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            raise ValueError("Array vacío")
        if len(x) <= ddof:
            raise ValueError(f"Tamaño de muestra ({len(x)}) debe ser > ddof ({ddof})")
        return float(np.var(x, ddof=ddof))

    @staticmethod
    def desviacion_tipica(x: Union[list, np.ndarray], ddof: int = 1) -> float:
        """
        Calcula la desviación típica (estándar) muestral.
        
        Parámetros
        ----------
        x : array_like
            Muestra de datos.
        ddof : int, default 1
            Grados de libertad (1 para muestra, 0 para población).
        
        Devuelve
        --------
        float
            Desviación típica muestral.
        """
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            raise ValueError("Array vacío")
        if len(x) <= ddof:
            raise ValueError(f"Tamaño de muestra ({len(x)}) debe ser > ddof ({ddof})")
        return float(np.std(x, ddof=ddof))

    @staticmethod
    def error_estandar(x: Union[list, np.ndarray]) -> float:
        """
        Calcula el error estándar de la media: σ / sqrt(n).
        
        Parámetros
        ----------
        x : array_like
            Muestra de datos.
        
        Devuelve
        --------
        float
            Error estándar de la media.
        """
        x = np.asarray(x, dtype=float)
        n = len(x)
        if n == 0:
            raise ValueError("Array vacío")
        sigma = float(np.std(x, ddof=1))
        return sigma / np.sqrt(n)

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

    # ========================================================================
    # SECCIÓN 2: INTERVALOS DE CONFIANZA
    # ========================================================================

    @staticmethod
    def intervalo_media_sigma_conocida(
        x: Union[list, np.ndarray],
        sigma: float,
        nivel: float = 0.95
    ) -> Tuple[float, float]:
        """
        Intervalo de confianza para la media con σ conocida (distribución normal).
        
        Parámetros
        ----------
        x : array_like
            Muestra de datos.
        sigma : float
            Desviación típica poblacional conocida.
        nivel : float, default 0.95
            Nivel de confianza (entre 0 y 1).
        
        Devuelve
        --------
        Tuple[float, float]
            (límite_inferior, límite_superior)
        """
        x = np.asarray(x, dtype=float)
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
        
        Parámetros
        ----------
        x : array_like
            Muestra de datos.
        nivel : float, default 0.95
            Nivel de confianza (entre 0 y 1).
        
        Devuelve
        --------
        Tuple[float, float]
            (límite_inferior, límite_superior)
        """
        x = np.asarray(x, dtype=float)
        n = len(x)
        if n < 2:
            raise ValueError("Se necesitan al menos 2 observaciones")
        if not (0 < nivel < 1):
            raise ValueError("nivel debe estar entre 0 y 1")
        
        mu = float(np.mean(x))
        s = float(np.std(x, ddof=1))
        tcrit = stats.t.ppf(1 - (1 - nivel) / 2, n - 1)
        Δ = tcrit * s / np.sqrt(n)
        
        return float(mu - Δ), float(mu + Δ)

    @staticmethod
    def intervalo_varianza(
        x: Union[list, np.ndarray],
        nivel: float = 0.95
    ) -> Tuple[float, float]:
        """
        Intervalo de confianza para la varianza (distribución chi-cuadrado).
        
        Parámetros
        ----------
        x : array_like
            Muestra de datos.
        nivel : float, default 0.95
            Nivel de confianza (entre 0 y 1).
        
        Devuelve
        --------
        Tuple[float, float]
            (límite_inferior, límite_superior)
        """
        x = np.asarray(x, dtype=float)
        n = len(x)
        if n < 2:
            raise ValueError("Se necesitan al menos 2 observaciones")
        if not (0 < nivel < 1):
            raise ValueError("nivel debe estar entre 0 y 1")
        
        s2 = float(np.var(x, ddof=1))
        α = 1 - nivel
        χ2_sup = stats.chi2.ppf(α / 2, n - 1)
        χ2_inf = stats.chi2.ppf(1 - α / 2, n - 1)
        return float((n - 1) * s2 / χ2_inf), float((n - 1) * s2 / χ2_sup)

    # ========================================================================
    # SECCIÓN 3: TESTS ESTADÍSTICOS
    # ========================================================================

    @staticmethod
    def test_media_t(
        x: Union[list, np.ndarray],
        mu0: float,
        alternativa: str = "dos_colas"
    ) -> Dict[str, Union[float, int]]:
        """
        Test t-Student para la media: H₀: μ = μ₀
        
        Parámetros
        ----------
        x : array_like
            Muestra de datos.
        mu0 : float
            Media hipotética bajo H₀.
        alternativa : str, default "dos_colas"
            Tipo de test: "dos_colas", "mayor" (μ > μ₀), o "menor" (μ < μ₀).
        
        Devuelve
        --------
        Dict[str, Union[float, int]]
            Diccionario con claves:
            - "t": estadístico t (float)
            - "p": p-valor (float)
            - "df": grados de libertad (int)
        """
        x = np.asarray(x, dtype=float)
        n = len(x)
        if n < 2:
            raise ValueError("Se necesitan al menos 2 observaciones")
        if alternativa not in ["dos_colas", "mayor", "menor"]:
            raise ValueError('alternativa debe ser "dos_colas", "mayor" o "menor"')
        
        mu = float(np.mean(x))
        s = float(np.std(x, ddof=1))
        t = (mu - mu0) / (s / np.sqrt(n))

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
        
        Parámetros
        ----------
        x : array_like
            Muestra de datos.
        distribucion : str, default "normal"
            Distribución a contrastar: "normal" (estima μ y σ) o "uniforme".
        
        Devuelve
        --------
        Dict[str, float]
            Diccionario con claves:
            - "estadistico": estadístico D de K-S (float)
            - "p_valor": p-valor del test (float)
        """
        x = np.asarray(x, dtype=float)
        if len(x) < 2:
            raise ValueError("Se necesitan al menos 2 observaciones")
        
        if distribucion == "normal":
            mu, sigma = float(np.mean(x)), float(np.std(x, ddof=0))
            d, p = stats.kstest(x, "norm", args=(mu, sigma))
        elif distribucion == "uniforme":
            d, p = stats.kstest(x, "uniform")
        else:
            raise ValueError(f"Distribución {distribucion} no soportada")
        
        return {"estadistico": float(d), "p_valor": float(p)}


# ============================================================================
# INSTANCIA SINGLETON
# ============================================================================

estadistica = _Estadistica()

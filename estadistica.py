"""
RESUMEN RÁPIDO (Funciones públicas)
----------------------------------
media
    INPUT:
        x: array_like (n,) -> datos numéricos (list/np.ndarray)
    OUTPUT:
        float -> media aritmética
    ERRORES:
        ValueError -> array vacío

varianza
    INPUT:
        x: array_like (n,) -> datos numéricos
        ddof: int -> grados de libertad
    OUTPUT:
        float -> varianza muestral
    ERRORES:
        ValueError -> array vacío, n <= ddof

desviacion_tipica
    INPUT:
        x: array_like (n,) -> datos numéricos
        ddof: int -> grados de libertad
    OUTPUT:
        float -> desviación típica muestral
    ERRORES:
        ValueError -> array vacío, n <= ddof

error_estandar
    INPUT:
        x: array_like (n,) -> datos numéricos
    OUTPUT:
        float -> error estándar de la media
    ERRORES:
        ValueError -> array vacío

media_ponderada
    INPUT:
        x: array_like (n,) -> datos numéricos
        w: array_like (n,) | None -> pesos positivos
        sigma: array_like (n,) | None -> incertidumbres positivas (w=1/sigma^2)
    OUTPUT:
        float -> media ponderada
    ERRORES:
        ValueError -> array vacío, longitudes no coinciden, valores no finitos

varianza_ponderada
    INPUT:
        x: array_like (n,) -> datos numéricos
        w: array_like (n,) | None -> pesos positivos
        sigma: array_like (n,) | None -> incertidumbres positivas (w=1/sigma^2)
        ddof: int -> grados de libertad (solo tipo="frecuentista")
        tipo: str -> "frecuentista" | "mle"
    OUTPUT:
        float -> varianza ponderada
    ERRORES:
        ValueError -> array vacío, n_eff <= ddof, tipo no soportado, pesos inválidos

intervalo_confianza
    INPUT:
        x: array_like (n,) -> datos numéricos
        nivel: float -> nivel de confianza (0,1)
        distribucion: str -> "normal" | "poisson" | "binomial"
        sigma: float | None -> σ conocida (solo normal)
    OUTPUT:
        dict -> limite_inferior, limite_superior, nivel, metodo, parametro_estimado, n
    NOTAS:
        incluye grados_libertad si aplica
    ERRORES:
        ValueError -> supuestos no válidos, datos incompatibles

test_media
    INPUT:
        x: array_like (n,) -> datos numéricos
        mu0: float -> valor bajo H0
        alternativa: str -> "dos_colas" | "mayor" | "menor"
        distribucion: str -> "normal" | "poisson" | "binomial"
        sigma: float | None -> σ conocida (solo normal)
    OUTPUT:
        dict -> estadistico, p_valor, metodo, parametro_nulo, parametro_estimado, n
    NOTAS:
        incluye grados_libertad si aplica
    ERRORES:
        ValueError -> supuestos no válidos, datos incompatibles

test_ks
    INPUT:
        x: array_like (n,) -> datos numéricos
        distribucion: str -> "normal" | "uniforme"
    OUTPUT:
        dict -> estadistico (float), p_valor (float)
    ERRORES:
        ValueError -> n < 2, distribución no soportada
"""

import numpy as np
from scipy import stats
from typing import Union, Tuple, Dict


# ============================================================================
# CLASE PRINCIPAL: ESTADISTICA
# ============================================================================

class _Estadistica:
    """
    Para hacer análisis estadístico de datos. Primero es necesario responder a las siguientes preguntas:
    
    - ¿Qué tipo de datos tengo? (Correspondientes al mismo observable A) o a distintas medidas B))
    
    - ¿Hay algún error en las mediciones? (Errores sistemáticos o del instrumento)
    
    - ¿Qué distribución tienen los datos? (Normal, Poisson, Binomial, etc.)
    
    - ¿Quiero estimar parámetros o contrastar hipótesis?
    
    
    """

    # ========================================================================
    # SECCIÓN 1: ESTADÍSTICA DESCRIPTIVA
    # ========================================================================
    
    @staticmethod
    def media(x: Union[list, np.ndarray]) -> float:
        """
        Calcula la media aritmética muestral.
        INPUT:
            x: array_like (n,) -> datos numéricos (list/np.ndarray)
        OUTPUT:
            float -> media aritmética
        ERRORES:
            ValueError -> array vacío
        
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
        Calcula la varianza muestral (s^2).
        INPUT:
            x: array_like (n,) -> datos numéricos
            ddof: int -> grados de libertad
        OUTPUT:
            float -> varianza muestral
        ERRORES:
            ValueError -> array vacío, n <= ddof
        
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
        Calcula la desviación típica (estándar) muestral (s).
        INPUT:
            x: array_like (n,) -> datos numéricos
            ddof: int -> grados de libertad
        OUTPUT:
            float -> desviación típica muestral
        ERRORES:
            ValueError -> array vacío, n <= ddof
        
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
        INPUT:
            x: array_like (n,) -> datos numéricos
        OUTPUT:
            float -> error estándar de la media
        ERRORES:
            ValueError -> array vacío
        
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
    def _calcular_pesos(
        x: np.ndarray,
        w: Union[list, np.ndarray, None],
        sigma: Union[list, np.ndarray, None]
    ) -> np.ndarray:
        """
        Helper interno: calcula pesos a partir de w o sigma (mutuamente excluyentes).
        
        Parámetros
        ----------
        x : np.ndarray
            Array de valores.
        w : array_like o None
            Pesos explícitos.
        sigma : array_like o None
            Incertidumbres (se convierte a w = 1/sigma^2).
        
        Devuelve
        --------
        np.ndarray
            Array de pesos (normalizado internamente si procede).
        
        Errores
        -------
        ValueError
            Si se pasan w y sigma a la vez, o si hay problemas de validación.
        """
        n = len(x)
        
        if w is not None and sigma is not None:
            raise ValueError("No se pueden especificar w y sigma simultáneamente")
        
        if w is None and sigma is None:
            # Pesos uniformes
            return np.ones(n, dtype=float)
        
        if sigma is not None:
            sigma = np.asarray(sigma, dtype=float)
            if len(sigma) != n:
                raise ValueError(f"Longitud de sigma ({len(sigma)}) no coincide con x ({n})")
            if np.any(sigma <= 0):
                raise ValueError("Todas las incertidumbres (sigma) deben ser positivas")
            if not np.all(np.isfinite(sigma)):
                raise ValueError("sigma contiene valores no finitos (inf/nan)")
            w = 1.0 / (sigma ** 2)
        else:
            w = np.asarray(w, dtype=float)
            if len(w) != n:
                raise ValueError(f"Longitud de w ({len(w)}) no coincide con x ({n})")
            if np.any(w <= 0):
                raise ValueError("Todos los pesos (w) deben ser positivos")
            if not np.all(np.isfinite(w)):
                raise ValueError("w contiene valores no finitos (inf/nan)")
        
        return w

    @staticmethod
    def media_ponderada(
        x: Union[list, np.ndarray],
        w: Union[list, np.ndarray, None] = None,
        sigma: Union[list, np.ndarray, None] = None
    ) -> float:
        """
        Calcula la media ponderada: μ_w = Σ(w_i·x_i) / Σ(w_i)
        INPUT:
            x: array_like (n,) -> datos numéricos
            w: array_like (n,) | None -> pesos positivos
            sigma: array_like (n,) | None -> incertidumbres positivas (w=1/sigma^2)
        OUTPUT:
            float -> media ponderada
        ERRORES:
            ValueError -> array vacío, longitudes no coinciden, valores no finitos
        
        Si no se especifican pesos, devuelve la media aritmética simple.
        Si se proporciona sigma, los pesos se calculan como w_i = 1/σ_i².
        
        Parámetros
        ----------
        x : array_like
            Muestra de datos.
        w : array_like, opcional
            Pesos asociados a cada dato. Deben ser positivos y finitos.
        sigma : array_like, opcional
            Incertidumbres de cada dato. Se convierten a pesos w = 1/σ².
            No se puede especificar w y sigma simultáneamente.
        
        Devuelve
        --------
        float
            Media ponderada.
        
        Errores
        -------
        ValueError
            Si el array está vacío, las longitudes no coinciden, se pasan w y sigma
            a la vez, o los pesos/sigmas no son válidos.
        
        Ejemplos
        --------
        >>> x = [1.0, 2.0, 3.0]
        >>> estadistica.media_ponderada(x)  # Media simple
        2.0
        >>> estadistica.media_ponderada(x, w=[1, 2, 1])  # Media ponderada
        2.0
        >>> estadistica.media_ponderada(x, sigma=[0.1, 0.2, 0.1])  # Pesos por incertidumbre
        2.0
        """
        x = np.asarray(x, dtype=float)
        n = len(x)
        
        if n == 0:
            raise ValueError("Array vacío")
        if not np.all(np.isfinite(x)):
            raise ValueError("x contiene valores no finitos (inf/nan)")
        
        w = _Estadistica._calcular_pesos(x, w, sigma)
        
        return float(np.sum(w * x) / np.sum(w))

    @staticmethod
    def varianza_ponderada(
        x: Union[list, np.ndarray],
        w: Union[list, np.ndarray, None] = None,
        sigma: Union[list, np.ndarray, None] = None,
        ddof: int = 1,
        tipo: str = "frecuentista"
    ) -> float:
        """
        Calcula la varianza ponderada de una muestra.
        INPUT:
            x: array_like (n,) -> datos numéricos
            w: array_like (n,) | None -> pesos positivos
            sigma: array_like (n,) | None -> incertidumbres positivas (w=1/sigma^2)
            ddof: int -> grados de libertad (solo tipo="frecuentista")
            tipo: str -> "frecuentista" | "mle"
        OUTPUT:
            float -> varianza ponderada
        ERRORES:
            ValueError -> array vacío, n_eff <= ddof, tipo no soportado, pesos inválidos
        
        Dos tipos de estimación:
        - "frecuentista": varianza muestral no sesgada con corrección por tamaño
          efectivo de muestra (n_eff = (Σw)²/Σw²). Usa ddof para corrección.
        - "mle": varianza tipo máxima verosimilitud sin corrección de sesgo.
        
        Parámetros
        ----------
        x : array_like
            Muestra de datos.
        w : array_like, opcional
            Pesos asociados a cada dato. Deben ser positivos y finitos.
        sigma : array_like, opcional
            Incertidumbres de cada dato. Se convierten a pesos w = 1/σ².
            No se puede especificar w y sigma simultáneamente.
        ddof : int, default 1
            Grados de libertad para corrección de sesgo (solo en tipo="frecuentista").
            Típicamente 1 para varianza muestral, 0 para poblacional.
        tipo : str, default "frecuentista"
            Tipo de estimación: "frecuentista" (con corrección) o "mle" (MLE sin corrección).
        
        Devuelve
        --------
        float
            Varianza ponderada.
        
        Errores
        -------
        ValueError
            Si el array está vacío, n_eff <= ddof (tipo frecuentista), tipo no soportado,
            o problemas de validación de pesos/sigmas.
        
        Notas
        -----
        Para tipo="frecuentista":
            n_eff = (Σw)² / Σ(w²)
            s²_w = Σ(w_i·(x_i - μ_w)²) / Σw
            var = s²_w · n_eff / (n_eff - ddof)
        
        Para tipo="mle":
            var = Σ(w_i·(x_i - μ_w)²) / Σw
        
        Ejemplos
        --------
        >>> x = [1.0, 2.0, 3.0]
        >>> estadistica.varianza_ponderada(x)  # Varianza simple (ddof=1)
        1.0
        >>> estadistica.varianza_ponderada(x, w=[1, 2, 1], tipo="mle")
        0.5
        """
        x = np.asarray(x, dtype=float)
        n = len(x)
        
        if n == 0:
            raise ValueError("Array vacío")
        if not np.all(np.isfinite(x)):
            raise ValueError("x contiene valores no finitos (inf/nan)")
        
        tipo = tipo.lower()
        if tipo not in ["frecuentista", "mle"]:
            raise ValueError('tipo debe ser "frecuentista" o "mle"')
        
        w = _Estadistica._calcular_pesos(x, w, sigma)
        
        # Media ponderada
        mu_w = np.sum(w * x) / np.sum(w)
        
        # Varianza ponderada base
        sum_w = np.sum(w)
        s2_w = np.sum(w * (x - mu_w) ** 2) / sum_w
        
        if tipo == "mle":
            return float(s2_w)
        
        # tipo == "frecuentista": corrección por tamaño efectivo
        sum_w2 = np.sum(w ** 2)
        n_eff = (sum_w ** 2) / sum_w2
        
        if n_eff <= ddof:
            raise ValueError(
                f"Tamaño efectivo de muestra ({n_eff:.2f}) debe ser > ddof ({ddof}). "
                f"Considera ddof=0 o tipo='mle'."
            )
        
        var_unbiased = s2_w * n_eff / (n_eff - ddof)
        return float(var_unbiased)

    

    # ========================================================================
    # SECCIÓN 2: ESTIMACIÓN DE PARÁMETROS Y CONSTRUCCIÓN DE INTERVALOS
    # ========================================================================
    
    
    @staticmethod
    def intervalo_confianza(
        x: Union[list, np.ndarray],
        *,
        nivel: float = 0.95,
        distribucion: str = "normal",
        sigma: float = None
    ) -> Dict[str, Union[float, int, str]]:
        """
        Estimación por intervalo (frecuentista) de un parámetro poblacional.
        INPUT:
            x: array_like (n,) -> datos numéricos
            nivel: float -> nivel de confianza (0,1)
            distribucion: str -> "normal" | "poisson" | "binomial"
            sigma: float | None -> σ conocida (solo normal)
        OUTPUT:
            dict -> limite_inferior, limite_superior, nivel, metodo, parametro_estimado, n
        NOTAS:
            incluye grados_libertad si aplica
        ERRORES:
            ValueError -> supuestos no válidos, datos incompatibles

        La cobertura se refiere al método, no al parámetro: el intervalo es
        aleatorio y, en repetición de muestreos, contiene al verdadero valor
        con probabilidad aproximadamente igual a ``nivel`` bajo las hipótesis
        asumidas.

        Hipótesis estadísticas según ``distribucion``
        -----------------------------------------------
        - ``"normal"``:
          * Si ``sigma`` está definido: población normal con desviación típica
            conocida, se usa intervalo z exacto.
          * Si ``sigma`` es ``None``: población normal con σ desconocida,
            se usa intervalo t-Student exacto.

        - ``"poisson"``:
          * ``x`` se interpreta como conteos Poisson independientes.
          * Se estima el parámetro de tasa λ por intervalo exacto (chi-cuadrado).

        - ``"binomial"``:
          * ``x`` se interpreta como observaciones Bernoulli (0/1).
          * Se usa intervalo exacto de Clopper–Pearson para p.

        Parámetros
        ----------
        x : array_like
            Muestra de datos.
        nivel : float, default 0.95
            Nivel de confianza (entre 0 y 1).
        distribucion : str, default "normal"
            Distribución asumida: "normal", "poisson", o "binomial".
        sigma : float, optional
            Desviación típica poblacional (solo para distribucion="normal").
            Si se proporciona, se usa intervalo z.
            Si es None, se usa intervalo t-Student.

        Devuelve
        --------
        Dict[str, Union[float, int, str]]
            Diccionario con claves:
            - "limite_inferior": límite inferior del intervalo (float)
            - "limite_superior": límite superior del intervalo (float)
            - "nivel": nivel de confianza (float)
            - "metodo": método utilizado ("z", "t", "poisson_exacto", "binomial_exacto")
            - "parametro_estimado": estimación puntual (float)
            - "n": tamaño de muestra (int)
            - "grados_libertad": df si aplica (int)

        Ejemplos de uso
        ---------------
        >>> x = [2.1, 2.4, 2.0, 2.3]
        >>> # Normal con σ desconocida (t-Student exacto)
        >>> resultado = estadistica.intervalo_confianza(x, distribucion="normal")
        >>> print(f"IC[μ]: ({resultado['limite_inferior']:.3f}, {resultado['limite_superior']:.3f})")

        >>> # Normal con σ conocida (z exacto)
        >>> resultado = estadistica.intervalo_confianza(x, sigma=0.2, distribucion="normal")

        >>> # Poisson (exacto chi-cuadrado)
        >>> conteos = [3, 1, 4, 2, 0]
        >>> resultado = estadistica.intervalo_confianza(conteos, distribucion="poisson")

        >>> # Binomial (Clopper-Pearson exacto)
        >>> ensayos = [1, 0, 1, 1, 0, 1]
        >>> resultado = estadistica.intervalo_confianza(ensayos, distribucion="binomial")

        Notas
        -----
        - Esta función NO adivina supuestos a partir de los datos.
        - Si falta un supuesto necesario, se lanza ``ValueError`` con mensaje claro.
        - Todos los métodos son exactos, no asintóticos.
        """
        x = np.asarray(x, dtype=float)
        n = len(x)

        # Validaciones generales
        if n == 0:
            raise ValueError("Array vacío")
        if not (0 < nivel < 1):
            raise ValueError("nivel debe estar entre 0 y 1")

        # Normalización case-insensitive
        distribucion = distribucion.lower()

        if distribucion == "normal":
            mu = float(np.mean(x))
            
            # Caso: σ conocida (z exacto)
            if sigma is not None:
                if sigma <= 0:
                    raise ValueError("sigma debe ser positivo")
                z = stats.norm.ppf(1 - (1 - nivel) / 2)
                delta = z * sigma / np.sqrt(n)
                return {
                    "limite_inferior": float(mu - delta),
                    "limite_superior": float(mu + delta),
                    "nivel": float(nivel),
                    "metodo": "z",
                    "parametro_estimado": float(mu),
                    "n": n,
                    "sigma_conocida": float(sigma)
                }

            # Caso: σ desconocida (t-Student exacto)
            if n < 2:
                raise ValueError("Se necesitan al menos 2 observaciones para estimar σ")
            s = float(np.std(x, ddof=1))
            tcrit = stats.t.ppf(1 - (1 - nivel) / 2, n - 1)
            delta = tcrit * s / np.sqrt(n)
            return {
                "limite_inferior": float(mu - delta),
                "limite_superior": float(mu + delta),
                "nivel": float(nivel),
                "metodo": "t",
                "parametro_estimado": float(mu),
                "desv_tipica_muestral": float(s),
                "n": n,
                "grados_libertad": n - 1
            }

        elif distribucion == "poisson":
            if np.any(x < 0):
                raise ValueError("Para Poisson, los conteos deben ser no negativos")
            if not np.all(np.isclose(x, np.round(x))):
                raise ValueError("Para Poisson, x debe contener conteos enteros")

            k = float(np.sum(x))
            lambda_est = k / n
            alpha = 1 - nivel
            
            if k == 0:
                li = 0.0
            else:
                # Límite inferior usando chi-cuadrado
                li = 0.5 * stats.chi2.ppf(alpha / 2, 2 * k) / n
            
            # Límite superior usando chi-cuadrado
            ls = 0.5 * stats.chi2.ppf(1 - alpha / 2, 2 * k + 2) / n
            
            return {
                "limite_inferior": float(li),
                "limite_superior": float(ls),
                "nivel": float(nivel),
                "metodo": "poisson_exacto",
                "parametro_estimado": float(lambda_est),
                "n": n
            }

        elif distribucion == "binomial":
            if np.any((x < 0) | (x > 1)):
                raise ValueError("Para binomial, x debe estar en el rango [0, 1]")

            # Número de éxitos en n ensayos Bernoulli
            k = int(np.round(np.sum(x)))
            p_est = k / n
            
            # Intervalo exacto de Clopper-Pearson usando distribución F
            alpha = 1 - nivel
            
            if k == 0:
                li = 0.0
            else:
                # Límite inferior: k / (k + (n-k+1)*F_{α/2}(2(n-k+1), 2k))
                F_lower = stats.f.ppf(alpha / 2, 2 * (n - k + 1), 2 * k)
                li = k / (k + (n - k + 1) * F_lower)
            
            if k == n:
                ls = 1.0
            else:
                # Límite superior: (k+1)*F_{1-α/2}(2(k+1), 2(n-k)) / (n-k + (k+1)*F_{1-α/2}(2(k+1), 2(n-k)))
                F_upper = stats.f.ppf(1 - alpha / 2, 2 * (k + 1), 2 * (n - k))
                ls = (k + 1) * F_upper / (n - k + (k + 1) * F_upper)
            
            return {
                "limite_inferior": float(li),
                "limite_superior": float(ls),
                "nivel": float(nivel),
                "metodo": "binomial_exacto",
                "parametro_estimado": float(p_est),
                "n": n,
                "exitos": int(k)
            }

        else:
            raise ValueError(f"Distribución '{distribucion}' no soportada")


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
        alpha = 1 - nivel
        chi2_sup = stats.chi2.ppf(alpha / 2, n - 1)
        chi2_inf = stats.chi2.ppf(1 - alpha / 2, n - 1)
        return float((n - 1) * s2 / chi2_inf), float((n - 1) * s2 / chi2_sup)

    # ========================================================================
    # SECCIÓN 3: TESTS ESTADÍSTICOS
    # ========================================================================

    @staticmethod
    def test_media(
        x: Union[list, np.ndarray],
        mu0: float,
        *,
        alternativa: str = "dos_colas",
        distribucion: str = "normal",
        sigma: float = None
    ) -> Dict[str, Union[float, int, str]]:
        """
        Test frecuentista exacto para la media de una población.
        INPUT:
            x: array_like (n,) -> datos numéricos
            mu0: float -> valor bajo H0
            alternativa: str -> "dos_colas" | "mayor" | "menor"
            distribucion: str -> "normal" | "poisson" | "binomial"
            sigma: float | None -> σ conocida (solo normal)
        OUTPUT:
            dict -> estadistico, p_valor, metodo, parametro_nulo, parametro_estimado, n
        NOTAS:
            incluye grados_libertad si aplica
        ERRORES:
            ValueError -> supuestos no válidos, datos incompatibles

        El test contrastea H₀: μ = μ₀ contra una hipótesis alternativa especificada.
        Se usa SIEMPRE el método exacto (no asintótico) disponible para la distribución.

        Hipótesis estadísticas según ``distribucion``
        -----------------------------------------------
        - ``"normal"``:
          * Si ``sigma`` está definido: población normal con σ conocida → z-test exacto.
          * Si ``sigma`` es ``None``: población normal con σ desconocida → t-test exacto.

        - ``"poisson"``:
          * ``x`` se interpreta como conteos Poisson independientes.
          * Se realiza test exacto sobre la tasa λ (media Poisson).
          * Bajo H₀: λ = μ₀, se usa distribución chi-cuadrado.

        - ``"binomial"``:
          * ``x`` se interpreta como ensayos Bernoulli (0/1).
          * Se realiza test exacto sobre la probabilidad p = μ₀ ∈ [0, 1].
          * Usa distribución binomial exacta bajo H₀.

        Parámetros
        ----------
        x : array_like
            Muestra de datos.
        mu0 : float
            Media (o tasa λ, o probabilidad p) hipotética bajo H₀.
        alternativa : str, default "dos_colas"
            Tipo de test: "dos_colas" (μ ≠ μ₀), "mayor" (μ > μ₀), o "menor" (μ < μ₀).
        distribucion : str, default "normal"
            Distribución asumida: "normal", "poisson", o "binomial".
        sigma : float, optional
            Desviación típica poblacional conocida (solo para distribucion="normal").
            Si se proporciona, se usa z-test.
            Si es None, se usa t-test estimando σ de la muestra.

        Devuelve
        --------
        Dict[str, Union[float, int, str]]
            Diccionario con claves:
            - "estadistico": valor del estadístico (z, t, o binomial count)
            - "p_valor": p-valor del test bilateral o unilateral (float)
            - "metodo": tipo de test ("z", "t", "poisson_exacto", "binomial_exacto")
            - "parametro_nulo": valor de H₀ (μ₀)
            - "parametro_estimado": estimación puntual del parámetro (float)
            - "n": tamaño de muestra (int)
            - "grados_libertad": df (int, solo si aplica)

        Validaciones y errores
        ----------------------
        - Lanza ValueError si se proporcionan supuestos contradictorios.
        - Lanza ValueError si los datos no cumplen las hipótesis (ej: negativos en Poisson).
        - NO infiere automáticamente la distribución a partir de los datos.

        Ejemplos
        --------
        >>> x = [2.1, 2.4, 2.0, 2.3]
        >>> # z-test con σ conocida = 0.2
        >>> resultado = estadistica.test_media(x, mu0=2.0, distribucion="normal", sigma=0.2)

        >>> # t-test con σ desconocida
        >>> resultado = estadistica.test_media(x, mu0=2.0, distribucion="normal")

        >>> # Test exacto Poisson: ¿es λ = 2?
        >>> conteos = [3, 1, 4, 2, 0]
        >>> resultado = estadistica.test_media(conteos, mu0=2.0, distribucion="poisson")

        >>> # Test exacto Binomial: ¿es p = 0.5?
        >>> ensayos = [1, 0, 1, 1, 0, 1]
        >>> resultado = estadistica.test_media(ensayos, mu0=0.5, distribucion="binomial")
        """
        x = np.asarray(x, dtype=float)
        n = len(x)

        # Validaciones generales
        if n == 0:
            raise ValueError("Array vacío")
        if alternativa not in ["dos_colas", "mayor", "menor"]:
            raise ValueError('alternativa debe ser "dos_colas", "mayor" o "menor"')

        distribucion = distribucion.lower()

        if distribucion == "normal":
            if n < 2 and sigma is None:
                raise ValueError("Se necesitan al menos 2 observaciones para estimar σ en t-test")

            mu = float(np.mean(x))
            
            if sigma is not None:
                # z-test con σ conocida
                if sigma <= 0:
                    raise ValueError("sigma debe ser positivo")
                z_stat = (mu - mu0) / (sigma / np.sqrt(n))
                
                if alternativa == "dos_colas":
                    p_valor = 2 * stats.norm.sf(abs(z_stat))
                elif alternativa == "mayor":
                    p_valor = stats.norm.sf(z_stat)
                else:  # "menor"
                    p_valor = stats.norm.cdf(z_stat)
                
                return {
                    "estadistico": float(z_stat),
                    "p_valor": float(p_valor),
                    "metodo": "z",
                    "parametro_nulo": float(mu0),
                    "parametro_estimado": float(mu),
                    "n": n,
                    "sigma_conocida": float(sigma)
                }
            else:
                # t-test con σ desconocida
                s = float(np.std(x, ddof=1))
                t_stat = (mu - mu0) / (s / np.sqrt(n))
                df = n - 1
                
                if alternativa == "dos_colas":
                    p_valor = 2 * stats.t.sf(abs(t_stat), df)
                elif alternativa == "mayor":
                    p_valor = stats.t.sf(t_stat, df)
                else:  # "menor"
                    p_valor = stats.t.cdf(t_stat, df)
                
                return {
                    "estadistico": float(t_stat),
                    "p_valor": float(p_valor),
                    "metodo": "t",
                    "parametro_nulo": float(mu0),
                    "parametro_estimado": float(mu),
                    "desv_tipica_muestral": float(s),
                    "n": n,
                    "grados_libertad": df
                }

        elif distribucion == "poisson":
            # Validar que x contiene conteos no-negativos enteros
            if np.any(x < 0):
                raise ValueError("Para Poisson, los conteos deben ser no negativos")
            if not np.all(np.isclose(x, np.round(x))):
                raise ValueError("Para Poisson, x debe contener conteos enteros")
            
            if mu0 <= 0:
                raise ValueError("Para Poisson, μ₀ (λ) debe ser positivo")
            
            # Conteo total K = sum(x_i) es el estadístico suficiente
            K = int(np.round(np.sum(x)))
            lambda_est = float(np.mean(x))
            
            # Test exacto Poisson: bajo H₀ cada x_i ~ Poisson(μ₀)
            # K ~ Poisson(n·μ₀)
            # Calculamos p-valor usando la distribución Poisson exacta
            lambda_total = n * mu0  # parámetro Poisson para K bajo H₀
            
            if alternativa == "dos_colas":
                # Dos colas: acumular probabilidades tan extremas o más que K observado
                prob_K = stats.poisson.pmf(K, lambda_total)
                
                # Valores de X con probabilidad ≤ prob(K)
                p_valor = 0.0
                max_x = max(K + 50, int(lambda_total + 10 * np.sqrt(lambda_total)))
                for i in range(max_x + 1):
                    if stats.poisson.pmf(i, lambda_total) <= prob_K:
                        p_valor += stats.poisson.pmf(i, lambda_total)
                
            elif alternativa == "mayor":
                # H₁: λ > λ₀, P(X ≥ K | H₀)
                p_valor = 1 - stats.poisson.cdf(K - 1, lambda_total)
                
            else:  # "menor"
                # H₁: λ < λ₀, P(X ≤ K | H₀)
                p_valor = stats.poisson.cdf(K, lambda_total)
            
            return {
                "estadistico": int(K),
                "p_valor": float(min(1.0, p_valor)),
                "metodo": "poisson_exacto",
                "parametro_nulo": float(mu0),
                "parametro_estimado": float(lambda_est),
                "n": n
            }

        elif distribucion == "binomial":
            # Validar que x contiene Bernoulli (0/1)
            if np.any((x < 0) | (x > 1)):
                raise ValueError("Para binomial, x debe estar en el rango [0, 1]")
            
            if not (0 <= mu0 <= 1):
                raise ValueError("Para binomial, μ₀ (p) debe estar en [0, 1]")
            
            # Número de éxitos
            k = int(np.round(np.sum(x)))
            p_est = float(np.mean(x))
            
            # Test exacto binomial usando la distribución binomial
            if alternativa == "dos_colas":
                # Dos colas: probabilidad de observar k o más extremo bajo H₀
                prob_k = stats.binom.pmf(k, n, mu0)
                
                # Para dos colas, considerar valores con probabilidad ≤ prob_k
                p_lower = stats.binom.cdf(k, n, mu0)
                p_upper = 1 - stats.binom.cdf(k - 1, n, mu0) if k > 0 else 1.0
                
                # Acumular probabilidades desde ambas colas
                p_valor = 0.0
                for i in range(n + 1):
                    if stats.binom.pmf(i, n, mu0) <= prob_k:
                        p_valor += stats.binom.pmf(i, n, mu0)
                
            elif alternativa == "mayor":
                # Una cola derecha: P(X ≥ k | H₀)
                p_valor = 1 - stats.binom.cdf(k - 1, n, mu0) if k > 0 else 1.0
            else:  # "menor"
                # Una cola izquierda: P(X ≤ k | H₀)
                p_valor = stats.binom.cdf(k, n, mu0)
            
            return {
                "estadistico": int(k),
                "p_valor": float(p_valor),
                "metodo": "binomial_exacto",
                "parametro_nulo": float(mu0),
                "parametro_estimado": float(p_est),
                "exitos": int(k),
                "n": n
            }

        else:
            raise ValueError(f"Distribución '{distribucion}' no soportada para test_media")

    @staticmethod
    def test_ks(x: Union[list, np.ndarray], distribucion: str = "normal") -> Dict[str, float]:
        """
        Test de Kolmogórov-Smirnov para bondad de ajuste.
        INPUT:
            x: array_like (n,) -> datos numéricos
            distribucion: str -> "normal" | "uniforme"
        OUTPUT:
            dict -> estadistico (float), p_valor (float)
        ERRORES:
            ValueError -> n < 2, distribución no soportada
        
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

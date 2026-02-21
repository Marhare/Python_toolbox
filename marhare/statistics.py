"""
QUICK SUMMARY (Public functions)
--------------------------------
mean
    INPUT:
        x: array_like (n,) -> numeric data (list/np.ndarray)
    OUTPUT:
        float -> arithmetic mean
    ERRORS:
        ValueError -> empty array

variance
    INPUT:
        x: array_like (n,) -> numeric data
        ddof: int -> degrees of freedom
    OUTPUT:
        float -> sample variance
    ERRORS:
        ValueError -> empty array, n <= ddof

standard_deviation
    INPUT:
        x: array_like (n,) -> numeric data
        ddof: int -> degrees of freedom
    OUTPUT:
        float -> sample standard deviation
    ERRORS:
        ValueError -> empty array, n <= ddof

standard_error
    INPUT:
        x: array_like (n,) -> numeric data
    OUTPUT:
        float -> standard error of the mean
    ERRORS:
        ValueError -> empty array

weighted_mean
    INPUT:
        x: array_like (n,) -> numeric data
        w: array_like (n,) | None -> positive weights
        sigma: array_like (n,) | None -> positive uncertainties (w=1/sigma^2)
    OUTPUT:
        float -> weighted mean
    ERRORS:
        ValueError -> empty array, mismatched lengths, non‑finite values

weighted_standard_error
    INPUT:
        x: array_like (n,) -> numeric data (not used, only for validation)
        w: array_like (n,) | None -> positive weights
        sigma: array_like (n,) | None -> positive uncertainties (w=1/sigma^2)
    OUTPUT:
        float -> standard error of weighted mean: sqrt(1/Σw)
    ERRORS:
        ValueError -> empty array, mismatched lengths, non‑finite values

weighted_variance
    INPUT:
        x: array_like (n,) -> numeric data
        w: array_like (n,) | None -> positive weights
        sigma: array_like (n,) | None -> positive uncertainties (w=1/sigma^2)
        ddof: int -> degrees of freedom (only tipo="frecuentista")
        tipo: str -> "frecuentista" | "mle"
    OUTPUT:
        float -> weighted variance
    ERRORS:
        ValueError -> empty array, n_eff <= ddof, unsupported tipo, invalid weights

confidence_interval
    INPUT:
        x: array_like (n,) -> numeric data
        nivel: float -> confidence level (0,1)
        distribucion: str -> "normal" | "poisson" | "binomial"
        sigma: float | None -> known σ (normal only)
    OUTPUT:
        dict -> lower_bound, upper_bound, level, method, estimated_parameter, n
    NOTES:
        includes degrees_of_freedom if applicable
    ERRORS:
        ValueError -> invalid assumptions, incompatible data

mean_test
    INPUT:
        x: array_like (n,) -> numeric data
        mu0: float -> value under H0
        alternativa: str -> "dos_colas" | "mayor" | "menor"
        distribucion: str -> "normal" | "poisson" | "binomial"
        sigma: float | None -> known σ (normal only)
    OUTPUT:
        dict -> estadistico, p_valor, metodo, parametro_nulo, parametro_estimado, n
    NOTES:
        includes grados_libertad if applicable
    ERRORS:
        ValueError -> invalid assumptions, incompatible data

ks_test
    INPUT:
        x: array_like (n,) -> numeric data
        distribucion: str -> "normal" | "uniforme"
    OUTPUT:
        dict -> estadistico (float), p_valor (float)
    ERRORS:
        ValueError -> n < 2, unsupported distribution
"""

import numpy as np
from scipy import stats
from typing import Union, Tuple, Dict


# ============================================================================
# MAIN CLASS: ESTADISTICA
# ============================================================================

class _Statistics:
    """
    To perform statistical analysis of data, first answer:
    
    - What type of data do I have? (Same observable A or different measurements B)
    - Are there measurement errors? (Systematic or instrument errors)
    - What distribution do the data have? (Normal, Poisson, Binomial, etc.)
    - Do I want to estimate parameters or test hypotheses?
    
    """

    # ========================================================================
    # SECTION 1: DESCRIPTIVE STATISTICS
    # ========================================================================
    
    @staticmethod
    def mean(x: Union[list, np.ndarray]) -> float:
        """
        Compute the sample arithmetic mean.
        INPUT:
            x: array_like (n,) -> numeric data (list/np.ndarray)
        OUTPUT:
            float -> arithmetic mean
        ERRORS:
            ValueError -> empty array
        
        Parameters
        ----------
        x : array_like
            Data sample.
        
        Returns
        -------
        float
            Arithmetic mean.
        """
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            raise ValueError("Empty array")
        return float(np.mean(x))

    @staticmethod
    def variance(x: Union[list, np.ndarray], ddof: int = 1) -> float:
        """
        Compute the sample variance (s^2).
        INPUT:
            x: array_like (n,) -> numeric data
            ddof: int -> degrees of freedom
        OUTPUT:
            float -> sample variance
        ERRORS:
            ValueError -> empty array, n <= ddof
        
        Parameters
        ----------
        x : array_like
            Data sample.
        ddof : int, default 1
            Degrees of freedom (1 for sample, 0 for population).
        
        Returns
        -------
        float
            Sample variance.
        """
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            raise ValueError("Empty array")
        if len(x) <= ddof:
            raise ValueError(f"Sample size ({len(x)}) must be > ddof ({ddof})")
        return float(np.var(x, ddof=ddof))

    @staticmethod
    def standard_deviation(x: Union[list, np.ndarray], ddof: int = 1) -> float:
        """
        Compute the sample standard deviation (s).
        INPUT:
            x: array_like (n,) -> numeric data
            ddof: int -> degrees of freedom
        OUTPUT:
            float -> sample standard deviation
        ERRORS:
            ValueError -> empty array, n <= ddof
        
        Parameters
        ----------
        x : array_like
            Data sample.
        ddof : int, default 1
            Degrees of freedom (1 for sample, 0 for population).
        
        Returns
        -------
        float
            Sample standard deviation.
        """
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            raise ValueError("Empty array")
        if len(x) <= ddof:
            raise ValueError(f"Sample size ({len(x)}) must be > ddof ({ddof})")
        return float(np.std(x, ddof=ddof))

    @staticmethod
    def standard_error(x: Union[list, np.ndarray]) -> float:
        """
        Compute the standard error of the mean: σ / sqrt(n).
        INPUT:
            x: array_like (n,) -> numeric data
        OUTPUT:
            float -> standard error of the mean
        ERRORS:
            ValueError -> empty array
        
        Parameters
        ----------
        x : array_like
            Data sample.
        
        Returns
        -------
        float
            Standard error of the mean.
        """
        x = np.asarray(x, dtype=float)
        n = len(x)
        if n == 0:
            raise ValueError("Empty array")
        sigma = float(np.std(x, ddof=1))
        return sigma / np.sqrt(n)

    @staticmethod
    def _compute_weights(
        x: np.ndarray,
        w: Union[list, np.ndarray, None],
        sigma: Union[list, np.ndarray, None]
    ) -> np.ndarray:
        """
        Internal helper: compute weights from w or sigma (mutually exclusive).
        
        Parameters
        ----------
        x : np.ndarray
            Array of values.
        w : array_like or None
            Explicit weights.
        sigma : array_like or None
            Uncertainties (converted to w = 1/sigma^2).
        
        Returns
        -------
        np.ndarray
            Weight array (normalized internally if needed).
        
        Errors
        ------
        ValueError
            If w and sigma are both provided, or validation fails.
        """
        n = len(x)
        
        if w is not None and sigma is not None:
            raise ValueError("w and sigma cannot be specified simultaneously")
        
        if w is None and sigma is None:
            # Uniform weights
            return np.ones(n, dtype=float)
        
        if sigma is not None:
            sigma = np.asarray(sigma, dtype=float)
            if len(sigma) != n:
                raise ValueError(f"Length of sigma ({len(sigma)}) does not match x ({n})")
            if np.any(sigma <= 0):
                raise ValueError("All uncertainties (sigma) must be positive")
            if not np.all(np.isfinite(sigma)):
                raise ValueError("sigma contains non‑finite values (inf/nan)")
            w = 1.0 / (sigma ** 2)
        else:
            w = np.asarray(w, dtype=float)
            if len(w) != n:
                raise ValueError(f"Length of w ({len(w)}) does not match x ({n})")
            if np.any(w <= 0):
                raise ValueError("All weights (w) must be positive")
            if not np.all(np.isfinite(w)):
                raise ValueError("w contains non‑finite values (inf/nan)")
        
        return w

    @staticmethod
    def weighted_mean(
        x: Union[list, np.ndarray],
        w: Union[list, np.ndarray, None] = None,
        sigma: Union[list, np.ndarray, None] = None
    ) -> float:
        """
        Compute the weighted mean: μ_w = Σ(w_i·x_i) / Σ(w_i)
        INPUT:
            x: array_like (n,) -> numeric data
            w: array_like (n,) | None -> positive weights
            sigma: array_like (n,) | None -> positive uncertainties (w=1/sigma^2)
        OUTPUT:
            float -> weighted mean
        ERRORS:
            ValueError -> empty array, mismatched lengths, non‑finite values
        
        If no weights are specified, returns the simple arithmetic mean.
        If sigma is provided, weights are computed as w_i = 1/σ_i².
        
        Parameters
        ----------
        x : array_like
            Data sample.
        w : array_like, optional
            Weights for each data point. Must be positive and finite.
        sigma : array_like, optional
            Uncertainties for each data point. Converted to weights w = 1/σ².
            w and sigma cannot be specified simultaneously.
        
        Returns
        -------
        float
            Weighted mean.
        
        Errors
        ------
        ValueError
            If the array is empty, lengths mismatch, w and sigma are both provided,
            or weights/sigmas are invalid.
        
        Examples
        --------
        >>> x = [1.0, 2.0, 3.0]
        >>> statistics.weighted_mean(x)  # Simple mean
        2.0
        >>> statistics.weighted_mean(x, w=[1, 2, 1])  # Weighted mean
        2.0
        >>> statistics.weighted_mean(x, sigma=[0.1, 0.2, 0.1])  # Uncertainty weights
        2.0
        """
        x = np.asarray(x, dtype=float)
        n = len(x)
        
        if n == 0:
            raise ValueError("Empty array")
        if not np.all(np.isfinite(x)):
            raise ValueError("x contains non‑finite values (inf/nan)")
        
        w = _Statistics._compute_weights(x, w, sigma)
        
        return float(np.sum(w * x) / np.sum(w))

    @staticmethod
    def weighted_standard_error(
        x: Union[list, np.ndarray],
        w: Union[list, np.ndarray, None] = None,
        sigma: Union[list, np.ndarray, None] = None
    ) -> float:
        """
        Compute the standard error of the weighted mean: σ_w = sqrt(1/Σw_i).
        INPUT:
            x: array_like (n,) -> numeric data (used only for validation)
            w: array_like (n,) | None -> positive weights
            sigma: array_like (n,) | None -> positive uncertainties (w=1/sigma^2)
        OUTPUT:
            float -> standard error of weighted mean
        ERRORS:
            ValueError -> empty array, mismatched lengths, non‑finite values
        
        When weights are inverse variances (w_i = 1/σ_i²), the uncertainty of
        the weighted mean is σ_w = sqrt(1/Σw_i).
        
        If no weights are specified, returns the simple standard error.
        If sigma is provided, weights are computed as w_i = 1/σ_i².
        
        Parameters
        ----------
        x : array_like
            Data sample (used for validation of length).
        w : array_like, optional
            Weights for each data point. Must be positive and finite.
        sigma : array_like, optional
            Uncertainties for each data point. Converted to weights w = 1/σ².
            w and sigma cannot be specified simultaneously.
        
        Returns
        -------
        float
            Standard error of the weighted mean.
        
        Errors
        ------
        ValueError
            If the array is empty, lengths mismatch, w and sigma are both provided,
            or weights/sigmas are invalid.
        
        Examples
        --------
        >>> x = [9.78, 9.81, 9.79, 9.82, 9.80]
        >>> s = [0.04, 0.04, 0.05, 0.04, 0.04]
        >>> statistics.weighted_standard_error(x, sigma=s)
        0.018...
        """
        x = np.asarray(x, dtype=float)
        n = len(x)
        
        if n == 0:
            raise ValueError("Empty array")
        if not np.all(np.isfinite(x)):
            raise ValueError("x contains non‑finite values (inf/nan)")
        
        # If no weights specified, return simple standard error
        if w is None and sigma is None:
            return _Statistics.standard_error(x)
        
        w = _Statistics._compute_weights(x, w, sigma)
        
        # Standard error of weighted mean: sqrt(1/Σw)
        return float(np.sqrt(1.0 / np.sum(w)))

    @staticmethod
    def weighted_variance(
        x: Union[list, np.ndarray],
        w: Union[list, np.ndarray, None] = None,
        sigma: Union[list, np.ndarray, None] = None,
        ddof: int = 1,
        tipo: str = "frecuentista"
    ) -> float:
        """
        Compute the weighted variance of a sample.
        INPUT:
            x: array_like (n,) -> numeric data
            w: array_like (n,) | None -> positive weights
            sigma: array_like (n,) | None -> positive uncertainties (w=1/sigma^2)
            ddof: int -> degrees of freedom (only tipo="frecuentista")
            tipo: str -> "frecuentista" | "mle"
        OUTPUT:
            float -> weighted variance
        ERRORS:
            ValueError -> empty array, n_eff <= ddof, unsupported tipo, invalid weights
        
        Two estimation types:
        - "frecuentista": unbiased sample variance with effective‑size correction
          (n_eff = (Σw)²/Σw²). Uses ddof for correction.
        - "mle": maximum‑likelihood‑type variance without bias correction.
        
        Parameters
        ----------
        x : array_like
            Data sample.
        w : array_like, optional
            Weights for each data point. Must be positive and finite.
        sigma : array_like, optional
            Uncertainties for each data point. Converted to weights w = 1/σ².
            w and sigma cannot be specified simultaneously.
        ddof : int, default 1
            Degrees of freedom for bias correction (only tipo="frecuentista").
            Typically 1 for sample variance, 0 for population.
        tipo : str, default "frecuentista"
            Estimation type: "frecuentista" (with correction) or "mle" (MLE without correction).
        
        Returns
        -------
        float
            Weighted variance.
        
        Errors
        ------
        ValueError
            If the array is empty, n_eff <= ddof (frecuentista), tipo unsupported,
            or weight/sigma validation fails.
        
        Notes
        -----
        For tipo="frecuentista":
            n_eff = (Σw)² / Σ(w²)
            s²_w = Σ(w_i·(x_i - μ_w)²) / Σw
            var = s²_w · n_eff / (n_eff - ddof)
        
        For tipo="mle":
            var = Σ(w_i·(x_i - μ_w)²) / Σw
        
        Examples
        --------
        >>> x = [1.0, 2.0, 3.0]
        >>> statistics.weighted_variance(x)  # Simple variance (ddof=1)
        1.0
        >>> statistics.weighted_variance(x, w=[1, 2, 1], tipo="mle")
        0.5
        """
        x = np.asarray(x, dtype=float)
        n = len(x)
        
        if n == 0:
            raise ValueError("Empty array")
        if not np.all(np.isfinite(x)):
            raise ValueError("x contains non‑finite values (inf/nan)")
        
        tipo = tipo.lower()
        if tipo not in ["frecuentista", "mle"]:
            raise ValueError('tipo must be "frecuentista" or "mle"')
        
        w = _Statistics._compute_weights(x, w, sigma)
        
        # Weighted mean
        mu_w = np.sum(w * x) / np.sum(w)
        
        # Base weighted variance
        sum_w = np.sum(w)
        s2_w = np.sum(w * (x - mu_w) ** 2) / sum_w
        
        if tipo == "mle":
            return float(s2_w)
        
        # tipo == "frecuentista": effective‑size correction
        sum_w2 = np.sum(w ** 2)
        n_eff = (sum_w ** 2) / sum_w2
        
        if n_eff <= ddof:
            raise ValueError(
                f"Effective sample size ({n_eff:.2f}) must be > ddof ({ddof}). "
                f"Consider ddof=0 or tipo='mle'."
            )
        
        var_unbiased = s2_w * n_eff / (n_eff - ddof)
        return float(var_unbiased)

    

    # ========================================================================
    # SECTION 2: PARAMETER ESTIMATION AND INTERVAL CONSTRUCTION
    # ========================================================================
    
    
    @staticmethod
    def confidence_interval(
        x: Union[list, np.ndarray],
        *,
        nivel: float = 0.95,
        distribucion: str = "normal",
        sigma: float = None
    ) -> Dict[str, Union[float, int, str]]:
        """
        Interval (frequentist) estimation of a population parameter.
        INPUT:
            x: array_like (n,) -> numeric data
            nivel: float -> confidence level (0,1)
            distribucion: str -> "normal" | "poisson" | "binomial"
            sigma: float | None -> known σ (normal only)
        OUTPUT:
            dict -> lower_bound, upper_bound, level, method, estimated_parameter, n
        NOTES:
            includes degrees_of_freedom if applicable
        ERRORS:
            ValueError -> invalid assumptions, incompatible data

        Coverage refers to the method, not the parameter: the interval is
        random and, under repeated sampling, contains the true value with
        probability approximately equal to ``nivel`` under the assumptions.

        Statistical assumptions by ``distribucion``
        -------------------------------------------
        - ``"normal"``:
          * If ``sigma`` is defined: normal population with known standard
            deviation, use exact z interval.
          * If ``sigma`` is ``None``: normal population with unknown σ,
            use exact Student‑t interval.

        - ``"poisson"``:
          * ``x`` is interpreted as independent Poisson counts.
          * Estimate rate parameter λ by exact interval (chi‑square).

        - ``"binomial"``:
          * ``x`` is interpreted as Bernoulli observations (0/1).
          * Use exact Clopper–Pearson interval for p.

        Parameters
        ----------
        x : array_like
            Data sample.
        nivel : float, default 0.95
            Confidence level (between 0 and 1).
        distribucion : str, default "normal"
            Assumed distribution: "normal", "poisson", or "binomial".
        sigma : float, optional
            Population standard deviation (only for distribucion="normal").
            If provided, use z interval.
            If None, use Student‑t interval.

        Returns
        -------
        Dict[str, Union[float, int, str]]
            Dictionary with keys:
            - "lower_bound": lower interval bound (float)
            - "upper_bound": upper interval bound (float)
            - "level": confidence level (float)
            - "method": method used ("z", "t", "poisson_exact", "binomial_exact")
            - "estimated_parameter": point estimate (float)
            - "n": sample size (int)
            - "degrees_of_freedom": df if applicable (int)

        Usage examples
        -------------
        >>> x = [2.1, 2.4, 2.0, 2.3]
        >>> # Normal con σ desconocida (t-Student exacto)
        >>> resultado = statistics.confidence_interval(x, distribucion="normal")
        >>> print(f"IC[μ]: ({resultado['lower_bound']:.3f}, {resultado['upper_bound']:.3f})")

        >>> # Normal con σ conocida (z exacto)
        >>> resultado = statistics.confidence_interval(x, sigma=0.2, distribucion="normal")

        >>> # Poisson (exacto chi-cuadrado)
        >>> conteos = [3, 1, 4, 2, 0]
        >>> resultado = statistics.confidence_interval(conteos, distribucion="poisson")

        >>> # Binomial (Clopper-Pearson exacto)
        >>> ensayos = [1, 0, 1, 1, 0, 1]
        >>> resultado = statistics.confidence_interval(ensayos, distribucion="binomial")

        Notes
        -----
        - This function does NOT infer assumptions from the data.
        - If a required assumption is missing, raises ``ValueError`` with a clear message.
        - All methods are exact, not asymptotic.
        """
        x = np.asarray(x, dtype=float)
        n = len(x)

        # Validaciones generales
        if n == 0:
            raise ValueError("Empty array")
        if not (0 < nivel < 1):
            raise ValueError("nivel must be between 0 and 1")

        # Case-insensitive normalization
        distribucion = distribucion.lower()

        if distribucion == "normal":
            mu = float(np.mean(x))
            
            # Case: known σ (exact z)
            if sigma is not None:
                if sigma <= 0:
                    raise ValueError("sigma must be positive")
                z = stats.norm.ppf(1 - (1 - nivel) / 2)
                delta = z * sigma / np.sqrt(n)
                return {
                    "lower_bound": float(mu - delta),
                    "upper_bound": float(mu + delta),
                    "level": float(nivel),
                    "method": "z",
                    "estimated_parameter": float(mu),
                    "n": n,
                    "known_sigma": float(sigma)
                }

            # Case: unknown σ (exact Student‑t)
            if n < 2:
                raise ValueError("At least 2 observations are required to estimate σ")
            s = float(np.std(x, ddof=1))
            tcrit = stats.t.ppf(1 - (1 - nivel) / 2, n - 1)
            delta = tcrit * s / np.sqrt(n)
            return {
                "lower_bound": float(mu - delta),
                "upper_bound": float(mu + delta),
                "level": float(nivel),
                "method": "t",
                "estimated_parameter": float(mu),
                "sample_std": float(s),
                "n": n,
                "degrees_of_freedom": n - 1
            }

        elif distribucion == "poisson":
            if np.any(x < 0):
                raise ValueError("For Poisson, counts must be non‑negative")
            if not np.all(np.isclose(x, np.round(x))):
                raise ValueError("For Poisson, x must contain integer counts")

            k = float(np.sum(x))
            lambda_est = k / n
            alpha = 1 - nivel
            
            if k == 0:
                li = 0.0
            else:
                # Lower bound using chi‑square
                li = 0.5 * stats.chi2.ppf(alpha / 2, 2 * k) / n
            
            # Upper bound using chi‑square
            ls = 0.5 * stats.chi2.ppf(1 - alpha / 2, 2 * k + 2) / n
            
            return {
                "lower_bound": float(li),
                "upper_bound": float(ls),
                "level": float(nivel),
                "method": "poisson_exact",
                "estimated_parameter": float(lambda_est),
                "n": n
            }

        elif distribucion == "binomial":
            if np.any((x < 0) | (x > 1)):
                raise ValueError("For binomial, x must be in the range [0, 1]")

            # Number of successes in n Bernoulli trials
            k = int(np.round(np.sum(x)))
            p_est = k / n
            
            # Exact Clopper‑Pearson interval using F distribution
            alpha = 1 - nivel
            
            if k == 0:
                li = 0.0
            else:
                # Lower bound: k / (k + (n-k+1)*F_{α/2}(2(n-k+1), 2k))
                F_lower = stats.f.ppf(alpha / 2, 2 * (n - k + 1), 2 * k)
                li = k / (k + (n - k + 1) * F_lower)
            
            if k == n:
                ls = 1.0
            else:
                # Upper bound: (k+1)*F_{1-α/2}(2(k+1), 2(n-k)) / (n-k + (k+1)*F_{1-α/2}(2(k+1), 2(n-k)))
                F_upper = stats.f.ppf(1 - alpha / 2, 2 * (k + 1), 2 * (n - k))
                ls = (k + 1) * F_upper / (n - k + (k + 1) * F_upper)
            
            return {
                "lower_bound": float(li),
                "upper_bound": float(ls),
                "level": float(nivel),
                "method": "binomial_exact",
                "estimated_parameter": float(p_est),
                "n": n,
                "successes": int(k)
            }

        else:
            raise ValueError(f"Distribution '{distribucion}' is not supported")


    @staticmethod
    def variance_interval(
        x: Union[list, np.ndarray],
        nivel: float = 0.95
    ) -> Tuple[float, float]:
        """
        Confidence interval for variance (chi‑square distribution).
        
        Parameters
        ----------
        x : array_like
            Data sample.
        nivel : float, default 0.95
            Confidence level (between 0 and 1).
        
        Returns
        -------
        Tuple[float, float]
            (lower_bound, upper_bound)
        """
        x = np.asarray(x, dtype=float)
        n = len(x)
        if n < 2:
            raise ValueError("At least 2 observations are required")
        if not (0 < nivel < 1):
            raise ValueError("nivel must be between 0 and 1")

        s2 = float(np.var(x, ddof=1))
        alpha = 1 - nivel
        chi2_sup = stats.chi2.ppf(alpha / 2, n - 1)
        chi2_inf = stats.chi2.ppf(1 - alpha / 2, n - 1)
        return float((n - 1) * s2 / chi2_inf), float((n - 1) * s2 / chi2_sup)

    # ========================================================================
    # SECTION 3: STATISTICAL TESTS
    # ========================================================================

    @staticmethod
    def mean_test(
        x: Union[list, np.ndarray],
        mu0: float,
        *,
        alternativa: str = "dos_colas",
        distribucion: str = "normal",
        sigma: float = None
    ) -> Dict[str, Union[float, int, str]]:
        """
        Exact frequentist test for a population mean.
        INPUT:
            x: array_like (n,) -> numeric data
            mu0: float -> value under H0
            alternativa: str -> "dos_colas" | "mayor" | "menor"
            distribucion: str -> "normal" | "poisson" | "binomial"
            sigma: float | None -> known σ (normal only)
        OUTPUT:
            dict -> estadistico, p_valor, metodo, parametro_nulo, parametro_estimado, n
        NOTES:
            includes grados_libertad if applicable
        ERRORS:
            ValueError -> invalid assumptions, incompatible data

        The test contrasts H₀: μ = μ₀ against the specified alternative hypothesis.
        It ALWAYS uses the exact (non‑asymptotic) method available for the distribution.

        Statistical assumptions by ``distribucion``
        -----------------------------------------------
        - ``"normal"``:
          * If ``sigma`` is defined: normal population with known σ → exact z-test.
          * If ``sigma`` is ``None``: normal population with unknown σ → exact t-test.

        - ``"poisson"``:
                    * ``x`` is interpreted as independent Poisson counts.
                    * Exact test on the rate λ (Poisson mean).
          * Under H₀: λ = μ₀, uses chi‑square distribution.

        - ``"binomial"``:
                    * ``x`` is interpreted as Bernoulli trials (0/1).
                    * Exact test on probability p = μ₀ ∈ [0, 1].
          * Uses exact binomial distribution under H₀.

        Parameters
        ----------
        x : array_like
            Data sample.
        mu0 : float
            Hypothetical mean (or rate λ, or probability p) under H₀.
        alternativa : str, default "dos_colas"
            Test type: "dos_colas" (μ ≠ μ₀), "mayor" (μ > μ₀), or "menor" (μ < μ₀).
        distribucion : str, default "normal"
            Assumed distribution: "normal", "poisson", or "binomial".
        sigma : float, optional
            Known population standard deviation (only for distribucion="normal").
            If provided, uses z-test.
            If None, uses t-test estimating σ from the sample.

        Returns
        -------
        Dict[str, Union[float, int, str]]
            Dictionary with keys:
            - "estadistico": value of the statistic (z, t, or binomial count)
            - "p_valor": p-value of the two‑sided or one‑sided test (float)
            - "metodo": test type ("z", "t", "poisson_exacto", "binomial_exacto")
            - "parametro_nulo": value under H₀ (μ₀)
            - "parametro_estimado": point estimate of the parameter (float)
            - "n": sample size (int)
            - "grados_libertad": df (int, only if applicable)

        Validations and errors
        ----------------------
        - Raises ValueError if contradictory assumptions are provided.
        - Raises ValueError if data do not meet assumptions (e.g., negatives in Poisson).
        - Does NOT infer the distribution automatically from data.

        Examples
        --------
        >>> x = [2.1, 2.4, 2.0, 2.3]
        >>> # z-test with known σ = 0.2
        >>> resultado = statistics.mean_test(x, mu0=2.0, distribucion="normal", sigma=0.2)

        >>> # t-test with unknown σ
        >>> resultado = statistics.mean_test(x, mu0=2.0, distribucion="normal")

        >>> # Exact Poisson test: is λ = 2?
        >>> conteos = [3, 1, 4, 2, 0]
        >>> resultado = statistics.mean_test(conteos, mu0=2.0, distribucion="poisson")

        >>> # Exact Binomial test: is p = 0.5?
        >>> ensayos = [1, 0, 1, 1, 0, 1]
        >>> resultado = statistics.mean_test(ensayos, mu0=0.5, distribucion="binomial")
        """
        x = np.asarray(x, dtype=float)
        n = len(x)

        # Validaciones generales
        if n == 0:
            raise ValueError("Empty array")
        if alternativa not in ["dos_colas", "mayor", "menor"]:
            raise ValueError('alternativa must be "dos_colas", "mayor" or "menor"')

        distribucion = distribucion.lower()

        if distribucion == "normal":
            if n < 2 and sigma is None:
                raise ValueError("At least 2 observations are required to estimate σ for t-test")

            mu = float(np.mean(x))
            
            if sigma is not None:
                # z-test with known σ
                if sigma <= 0:
                    raise ValueError("sigma must be positive")
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
                # t-test with unknown σ
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
            # Validate that x contains non‑negative integer counts
            if np.any(x < 0):
                raise ValueError("For Poisson, counts must be non‑negative")
            if not np.all(np.isclose(x, np.round(x))):
                raise ValueError("For Poisson, x must contain integer counts")
            
            if mu0 <= 0:
                raise ValueError("For Poisson, μ₀ (λ) must be positive")
            
            # Total count K = sum(x_i) is the sufficient statistic
            K = int(np.round(np.sum(x)))
            lambda_est = float(np.mean(x))
            
            # Exact Poisson test: under H₀ each x_i ~ Poisson(μ₀)
            # K ~ Poisson(n·μ₀)
            # Compute p-value using exact Poisson distribution
            lambda_total = n * mu0  # Poisson parameter for K under H₀
            
            if alternativa == "dos_colas":
                # Two-tailed: accumulate probabilities as extreme or more than observed K
                prob_K = stats.poisson.pmf(K, lambda_total)
                
                # X values with probability ≤ prob(K)
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
            # Validate that x contains Bernoulli (0/1)
            if np.any((x < 0) | (x > 1)):
                raise ValueError("For binomial, x must be in the range [0, 1]")
            
            if not (0 <= mu0 <= 1):
                raise ValueError("For binomial, μ₀ (p) must be in [0, 1]")
            
            # Number of successes
            k = int(np.round(np.sum(x)))
            p_est = float(np.mean(x))
            
            # Exact binomial test using the binomial distribution
            if alternativa == "dos_colas":
                # Two-tailed: probability of observing k or more extreme under H₀
                prob_k = stats.binom.pmf(k, n, mu0)
                
                # For two-tailed, consider values with probability ≤ prob_k
                p_lower = stats.binom.cdf(k, n, mu0)
                p_upper = 1 - stats.binom.cdf(k - 1, n, mu0) if k > 0 else 1.0
                
                # Accumulate probabilities from both tails
                p_valor = 0.0
                for i in range(n + 1):
                    if stats.binom.pmf(i, n, mu0) <= prob_k:
                        p_valor += stats.binom.pmf(i, n, mu0)
                
            elif alternativa == "mayor":
                # Right tail: P(X ≥ k | H₀)
                p_valor = 1 - stats.binom.cdf(k - 1, n, mu0) if k > 0 else 1.0
            else:  # "menor"
                # Left tail: P(X ≤ k | H₀)
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
            raise ValueError(f"Distribution '{distribucion}' not supported for mean_test")

    @staticmethod
    def ks_test(x: Union[list, np.ndarray], distribucion: str = "normal") -> Dict[str, float]:
        """
        Kolmogorov‑Smirnov test for goodness of fit.
        INPUT:
            x: array_like (n,) -> numeric data
            distribucion: str -> "normal" | "uniforme"
        OUTPUT:
            dict -> estadistico (float), p_valor (float)
        ERRORS:
            ValueError -> n < 2, unsupported distribution
        
        Parameters
        ----------
        x : array_like
            Data sample.
        distribucion : str, default "normal"
            Distribution to test: "normal" (estimates μ and σ) or "uniforme".
        
        Returns
        -------
        Dict[str, float]
            Dictionary with keys:
            - "estadistico": K‑S D statistic (float)
            - "p_valor": test p‑value (float)
        """
        x = np.asarray(x, dtype=float)
        if len(x) < 2:
            raise ValueError("At least 2 observations are required")
        
        if distribucion == "normal":
            mu, sigma = float(np.mean(x)), float(np.std(x, ddof=0))
            d, p = stats.kstest(x, "norm", args=(mu, sigma))
        elif distribucion == "uniforme":
            d, p = stats.kstest(x, "uniform")
        else:
            raise ValueError(f"Distribution {distribucion} not supported")
        
        return {"estadistico": float(d), "p_valor": float(p)}


# ============================================================================
# INSTANCIA SINGLETON
# ============================================================================

statistics = _Statistics()

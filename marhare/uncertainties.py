from __future__ import annotations
import numpy as np

# Required dependency: uncertainties ALWAYS installed
from .latex_tools import expr_to_latex, latex_tools
import sympy as sp






class _Uncertainties:
    '''
    Docstring for _Uncertainties
    '''
    # --------- Construction ---------
    @staticmethod
    def checker(value, sigma):
        """
        Safety check for measurement inputs.

        - Ensures value and sigma are numeric (scalar or array-like)
        - Classifies scalar vs vector
        - Checks shape compatibility
        - Returns inherited type and shape
        """

        value = np.asarray(value, dtype=object)
        sigma = np.asarray(sigma, dtype=object)

        # --- numeric check ---
        if value.dtype.kind in ("U", "S") or sigma.dtype.kind in ("U", "S"):#What's "U" and "S"?
            raise TypeError("value or sigma is not numeric (string)")

        try:
            if not np.issubdtype(value.dtype, np.number):
                np.asarray(value, dtype=float)
            if not np.issubdtype(sigma.dtype, np.number):
                np.asarray(sigma, dtype=float)
        except Exception:
            raise TypeError("value or sigma is not numeric")

        # --- scalar vs vector ---
        value_is_vec = value.ndim >= 1 and value.shape != ()
        sigma_is_vec = sigma.ndim >= 1 and sigma.shape != ()

        # --- compatibility ---
        if not value_is_vec and sigma_is_vec:
            raise ValueError("sigma is vector but value is scalar")

        if not value_is_vec and not sigma_is_vec:
            return {"shape": None, "kind": "scalar", "sigma_vec": None}

        if value_is_vec and not sigma_is_vec:       #Transform scalar sigma to vector
            sigma_scalar = float(np.asarray(sigma, dtype=float))
            sigma_vec = np.full(value.shape, sigma_scalar, dtype=float)
            return {"shape": value.shape, "kind": "vector", "sigma_vec": sigma_vec}

        if value.shape != sigma.shape:
            raise ValueError(
                f"incompatible shapes: value={value.shape}, sigma={sigma.shape}"
            )

        sigma_vec = np.asarray(sigma, dtype=float)
        return {"shape": value.shape, "kind": "vector", "sigma_vec": sigma_vec}
    
    @staticmethod
    def quantity(*args):
        """
        Unified quantity constructor (positional-only).

        Accepted signatures:
        1) quantity(value, sigma, unit)   -> input quantity (measurement/constant)
        2) quantity(expr, unit)           -> derived quantity (expr as string or sympy.Expr)

        Returns a dict with stable keys:
        - valor: (value, sigma) or None
        - expr:  None or sympy.Expr / str
        - unit:  str
        - dimension: shape tuple or None
        """

        if len(args) == 3:
            value, sigma, unit = args

            # negative sigma check (works for scalars and arrays)
            if np.any(np.asarray(sigma) < 0):
                raise ValueError("sigma cannot be negative")

            info = _Uncertainties.checker(value, sigma)
            sigma_out = info["sigma_vec"] if info["kind"] == "vector" else sigma

            return {
                "valor": (value, sigma_out),
                "expr": None,
                "unit": unit,
                "dimension": info["shape"],
            }

        if len(args) == 2:
            expr, unit = args

            # keep expr as string or Expr (propagation will resolve symbols)
            if not isinstance(expr, (str, sp.Expr)):
                raise TypeError("expr must be a string or sympy.Expr")

            return {
                "valor": None,
                "expr": expr,
                "unit": unit,
                "dimension": None,
            }

        raise TypeError("quantity(...) expects (value, sigma, unit) or (expr, unit)")

        
    
    
    
    # --------- Uncertainty propagation ---------
    @staticmethod
    def uncertainty_propagation(
        f: sp.Expr,
        vars_: list[sp.Symbol], #What's the difference between vars_ and variables?
        values: dict[sp.Symbol, object],
        sigmas: dict[sp.Symbol, float],
        cov: sp.Matrix | None = None,
        simplify: bool = True
    ) -> dict:
        import sympy as sp
        import numpy as np

        for v in vars_:
            if v not in values:
                raise ValueError(f"Missing value for {v}")
            if v not in sigmas:
                raise ValueError(f"Missing sigma for {v}")
            if sigmas[v] < 0:
                raise ValueError(f"Negative sigma for {v}")

        # Gradient
        grad = sp.Matrix([sp.diff(f, v) for v in vars_])

        # Covariances (symbolic sigmas by default)
        sigma_symbols = {v: sp.Symbol(f"sigma_{v.name}") for v in vars_}
        if cov is None:
            Sigma = sp.diag(*[sigma_symbols[v]**2 for v in vars_])
        else:
            if cov.shape != (len(vars_), len(vars_)):
                raise ValueError("Incorrect dimensions for cov")
            Sigma = cov

        var_f = (grad.T * Sigma * grad)[0]
        sigma_f_expr = sp.sqrt(var_f)

        if simplify:
            var_f = sp.simplify(var_f)
            sigma_f_expr = sp.simplify(sigma_f_expr)

        f_num = sp.lambdify(vars_, f, "numpy")
        sigma_syms = [sigma_symbols[v] for v in vars_]
        s_num = sp.lambdify(vars_ + sigma_syms, sigma_f_expr, "numpy")

        args = [values[v] for v in vars_]
        s_args = [sigmas[v] for v in vars_]

        return {
            "valor": f_num(*args),
            "sigma": s_num(*args, *s_args),
            "expr_latex": sp.latex(f),
            "sigma_latex": sp.latex(sigma_f_expr),
        }

    @staticmethod
    def propagate(expr, values: dict, sigmas: dict, simplify=True): #And unit??-> "symbols" is the list of variables in the expression.
        """
        ROBUST uncertainty propagation (no ordering errors).

        expr     : sympy.Expr
        values  : dict {Symbol: array | scalar}
        sigmas   : dict {Symbol: float}
        """
        import numpy as np

        symbols = list(expr.free_symbols)

        # Validations
        for s in symbols:
            if s not in values:
                raise ValueError(f"Missing value for {s}")
            if s not in sigmas:
                raise ValueError(f"Missing sigma for {s}")

        # Vectorized: if any input is array
        vectorial = any(np.ndim(v) > 0 for v in values.values())
        if vectorial:
            longitudes = [len(v) for v in values.values() if np.ndim(v) > 0]
            N = max(longitudes) if longitudes else 1
        else:
            N = 1

        f_vals = []
        s_vals = []

        # Main loop (i lives here)
        for i in range(N):
            vals_i = {}
            for s in symbols:
                v = values[s]
                if vectorial and np.ndim(v) > 0:
                    vals_i[s] = v[i]
                else:
                    vals_i[s] = v

            res = _Uncertainties.uncertainty_propagation(
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
    @staticmethod
    def propagate_quantity(name: str, magnitudes: dict, simplify=True):
        """
        High-level uncertainty propagation for a derived quantity.

        Returns:
            value
            uncertainty
            analytic expression (latex)
            analytic uncertainty expression (latex)
        """

        # 1) Symbol registry
        symbols = {k: sp.Symbol(k) for k in magnitudes}

        # 2) Expression
        q = magnitudes[name]
        if q["expr"] is None:
            raise ValueError(f"{name} is not a derived quantity")

        expr = sp.sympify(q["expr"], locals=symbols)

        # 3) Extract values and sigmas
        values = {}
        sigmas = {}
        for k, v in magnitudes.items():
            if v["valor"] is None:
                continue
            val, sig = v["valor"]
            sym = symbols[k]
            values[sym] = val
            sigmas[sym] = sig

        # 4) Propagate
        res = _Uncertainties.propagate(expr, values, sigmas, simplify=simplify)

        # 5) Return ONLY what you care about
        return {
            "value": res["valor"],
            "uncertainty": res["sigma"],
            "expr": res["expr_latex"],
            "sigma_expr": res["sigma_latex"],
        }


    # --------- Accessors ---------
    


incertidumbres = _Uncertainties()

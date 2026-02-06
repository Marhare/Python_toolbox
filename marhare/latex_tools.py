"""
latex_tools.py
Tools to generate scientific LaTeX from Python.

Typical workflow: Excel → Python → LaTeX

Features:
- Standard metrological rounding (1–2 significant digits in uncertainty)
- Formatting values with uncertainties: (x ± σ)
- LaTeX table generation with configurable style
- Scientific notation and siunitx support
- Export to .tex files
"""

import math
from typing import Optional

import numpy as np
import sympy as sp

from .uncertainties import value_quantity
import uncertainties.unumpy as unp
TABLA_CONFIG = {
    "centrar": True,
    "tamano": None,
    "lineas": "hline",
    "envolver": True,
    "posicion": "htbp",
}

def _orden_magnitud(x: float) -> int:
    """
    Returns the decimal order of magnitude of x.
    
    Example: _orden_magnitud(0.0123) → -2
    """
    if x == 0:
        return 0
    return int(math.floor(math.log10(abs(x))))

def redondeo_incertidumbre(
    valor: float,
    sigma: float,
    cifras: int = 2,
):
    """
    Rounds value and uncertainty following standard metrological rules.
    
    Parameters
    ----------
    valor : float
        Central value to round.
    sigma : float
        Uncertainty (must be > 0).
    cifras : int, default 2
        Number of significant digits for the uncertainty (1 or 2).
    
    Returns
    -------
    tuple[float, float, int]
        (valor_redondeado, sigma_redondeada, decimales)
    
    Rules:
    - Uncertainty is given with 1 or 2 significant digits
    - Value is rounded to the same number of decimals as sigma
    """
    if sigma <= 0:
        raise ValueError("Uncertainty must be positive")

    decimales = -(orden - (cifras - 1))

    sigma_r = round(sigma, decimales)
    valor_r = round(valor, decimales)

    return valor_r, sigma_r, decimales

def _to_latex_sci(valor: float, cifras: int = 2) -> str:
    """
    Converts a number in scientific notation to LaTeX format.
    
    Example: 5.3e-04 → "5.3 \\times 10^{-4}"
    """
    fmt = f"{{:.{cifras}e}}"
    s = fmt.format(valor)
    mantisa_str, exp_str = s.split('e')
    mantisa = float(mantisa_str)
    exponente = int(exp_str)
    
    return f"{mantisa:.{cifras}f} \\times 10^{{{exponente}}}"

def _valor_pm_escalar(
    valor: float,
    sigma: float,
    unidad: Optional[str] = None,
    cifras: int = 2,
    siunitx: bool = False,
):
    """
    Formats a scalar value with uncertainty in LaTeX.
    
    Returns LaTeX string of the form:
        (x ± s)
    or with siunitx:
        \\SI{x \\pm s}{unidad}
    
    WITHOUT external math delimiters (\\[ \\]).
    Automatically uses scientific notation for very large or small values.
    """
    v, s, d = redondeo_incertidumbre(valor, sigma, cifras)

    use_sci = False
    abs_v, abs_s = abs(v), abs(s)
    if (abs_v != 0 and (abs_v >= 1e5 or abs_v < 1e-4)) or (abs_s != 0 and (abs_s >= 1e5 or abs_s < 1e-4)):
        use_sci = True

    if use_sci:
        v_str = _to_latex_sci(v, cifras)
        s_str = _to_latex_sci(s, cifras)
    else:
        fmt_precision = max(0, d)
        fmt = f"{{:.{fmt_precision}f}}"
        v_str = fmt.format(v)
        s_str = fmt.format(s)

    if siunitx and unidad:
        return f"\\SI{{{v_str} \\pm {s_str}}}{{{unidad}}}"

    if unidad:
        return f"({v_str} \\pm {s_str})\\,\\mathrm{{{unidad}}}"

    return f"({v_str} \\pm {s_str})"

def valor_pm(
    valor,
    sigma=None,
    *,
    unidad: Optional[str] = None,
    cifras: int = 2,
    siunitx: bool = False,
    caption: Optional[str] = None,
    label: Optional[str] = None,
    centrar: Optional[bool] = None,
    tamano: Optional[str] = None,
    lineas: Optional[str] = None,
    envolver: Optional[bool] = None,
    posicion: Optional[str] = None,
    orientacion: str = "vertical",
    headers: Optional[list[str]] = None,
    row_headers: Optional[list[str]] = None,
):
    """
    Main function to format values with uncertainties in LaTeX.
    
    Accepts scalars, vectors, and matrices. For scalars returns a plain
    LaTeX fragment without display math delimiters; for vectors/matrices
    generates a LaTeX table.
    
    Parameters
    ----------
    valor : float, array_like, or uncertainties.uarray
        Value(s) to format.
    sigma : float, array_like, or None
        Uncertainty(ies). If None, assumes valor is already a uarray.
    unidad : str, optional
        Physical unit in LaTeX format (e.g., "m/s^2").
    cifras : int, default 2
        Significant digits for the uncertainty (1 or 2).
    siunitx : bool, default False
        If True, uses \\SI{...}{...} format (requires unit).
    caption : str, optional
        Caption for the table (vectors/matrices only).
    label : str, optional
        LaTeX label for cross‑references.
    centrar : bool, optional
        If True, centers the table. Defaults to TABLA_CONFIG.
    tamano : str, optional
        Font size ("small", "footnotesize", etc.).
    lineas : str, optional
        Line style: "booktabs", "hline" or None.
    envolver : bool, optional
        If True, wraps in a floating table environment.
    posicion : str, optional
        Float position ("htbp", "H", etc.).
    orientacion : str, default "vertical"
        For 1D vectors: "vertical" (column) or "horizontal" (row).
    headers : list[str], optional
        Column headers.
    row_headers : list[str], optional
        Row headers.
    
    Returns
    -------
    str
        Formatted LaTeX code.
    
    Examples
    --------
    >>> # Scalar
    >>> valor_pm(3.14159, 0.0123, cifras=2)
    '(3.14 ± 0.01)'
    
    >>> # With unit
    >>> valor_pm(9.81, 0.05, unidad="m/s^2", cifras=2)
    '(9.81 ± 0.05)\\,\\mathrm{m/s^2}'
    
    >>> # Vector (table)
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> valor_pm(x, 0.1, cifras=1, caption="Measures")
    # Generates a 3×1 LaTeX table
    """
    if centrar is None:
        centrar = TABLA_CONFIG["centrar"]
    if tamano is None:
        tamano = TABLA_CONFIG["tamano"]
    if lineas is None:
        lineas = TABLA_CONFIG["lineas"]
    if envolver is None:
        envolver = TABLA_CONFIG["envolver"]
    if posicion is None:
        posicion = TABLA_CONFIG["posicion"]

    if sigma is None:
        obj = valor
    else:
        obj = unp.uarray(valor, sigma)

    v = unp.nominal_values(obj)
    s = unp.std_devs(obj)

    if not isinstance(v, np.ndarray) or v.shape == ():
        v0 = float(np.asarray(v).item())
        s0 = float(np.asarray(s).item())
        contenido = _valor_pm_escalar(
            v0, s0, unidad=unidad, cifras=cifras, siunitx=siunitx
        )
        return contenido

    v = np.asarray(v)
    s = np.asarray(s)

    if v.ndim == 1:
        o = orientacion.lower().strip()
        if o in ("vertical", "v", "col", "columna"):
            v = v[:, None]
            s = s[:, None]
        elif o in ("horizontal", "h", "row", "fila"):
            v = v[None, :]
            s = s[None, :]
        else:
            raise ValueError("orientacion must be 'vertical' or 'horizontal'.")

    filas, cols = v.shape

    if headers is not None and len(headers) != cols:
        raise ValueError("headers must have the same length as the number of columns.")
    if row_headers is not None and len(row_headers) != filas:
        raise ValueError("row_headers must have the same length as the number of rows.")

    col_fmt = ""
    if row_headers:
        col_fmt += "l"
    col_fmt += "c" * cols

    tabular = [rf"\begin{{tabular}}{{{col_fmt}}}"]

    if lineas == "booktabs":
        tabular.append(r"\toprule")
    elif lineas == "hline":
        tabular.append(r"\hline")

    if headers:
        header_line = []
        if row_headers:
            header_line.append("")
        header_line.extend(headers)
        tabular.append(" & ".join(header_line) + r" \\")
        if lineas in ("booktabs", "hline"):
            tabular.append(r"\midrule" if lineas == "booktabs" else r"\hline")

    for i in range(filas):
        fila = []
        if row_headers:
            fila.append(row_headers[i])
        for j in range(cols):
            fila.append(
                _valor_pm_escalar(
                    float(v[i, j]),
                    float(s[i, j]),
                    unidad=None,
                    cifras=cifras,
                    siunitx=False,
                )
            )
        tabular.append(" & ".join(fila) + r" \\")

    if lineas == "booktabs":
        tabular.append(r"\bottomrule")
    elif lineas == "hline":
        tabular.append(r"\hline")

    tabular.append(r"\end{tabular}")
    latex_tabular = "\n".join(tabular)

    inserciones = ""
    if centrar:
        inserciones += "\\centering\n"
    if tamano:
        inserciones += f"\\{tamano}\n"

    unidad_global = rf"\,\mathrm{{{unidad}}}" if unidad else ""

    if envolver:
        out = f"\\begin{{table}}[{posicion}]\n"
        out += inserciones
        if caption:
            out += f"\\caption{{{caption}}}\n"
        if label:
            out += f"\\label{{{label}}}\n"
        out += latex_tabular + "\n"
        if unidad_global:
            out += unidad_global + "\n"
        out += "\\end{table}\n"
        return out

    out = inserciones + latex_tabular
    if unidad_global:
        out += "\n" + unidad_global
    return out


def tabla_pm(
    columnas,
    valores,
    sigmas,
    **kw
):
    valores = np.column_stack(valores)
    sigmas = np.column_stack(
        [np.full_like(valores[:, i], s) if np.isscalar(s) else s
         for i, s in enumerate(sigmas)]
    )
    return valor_pm(valores, sigmas, headers=columnas, **kw)


def latex_quantity(
    mag: dict,
    *,
    cifras: int = 2,
    siunitx: bool = False,
    **kw
) -> str:
    """
    Format a magnitude/constant from the magnitudes registry.

        Expects mag to be a dict like:
            {
                "measure": (value, sigma) | None,
                "result": (value, sigma) | None,
                "unit": "m",
                "dimension": None | tuple,
                "expr": None | str | sp.Expr
            }

    Notes:
    - Uses mag["dimension"] (shape) to decide scalar vs vector/matrix.
    - Assumes sigma is already vectorized when needed (checker does that).
    - Delegates actual formatting to `valor_pm`.
    """

    if mag.get("measure", None) is None and mag.get("result", None) is None:
        expr = mag.get("expr", None)
        if expr is not None:
            if isinstance(expr, sp.Expr):
                return sp.latex(expr)
            try:
                return sp.latex(sp.sympify(expr))
            except Exception:
                # Assume expr is already LaTeX if SymPy cannot parse it
                return str(expr)
        raise ValueError("latex_quantity(): mag has no numeric value to format")

    value, sigma = value_quantity(mag)
    unit = mag.get("unit", None)

    return valor_pm(
        value,
        sigma,
        unidad=unit,
        cifras=cifras,
        siunitx=siunitx,
        **kw
    )
def exportar(
    filename: str,
    contenido: str,
    modo: str = "w",
):
    """
    Write LaTeX content to a .tex file.
    
    Parameters
    ----------
    filename : str
        File path to create/modify.
    contenido : str
        LaTeX code to write.
    modo : str, default "w"
        Open mode: "w" (overwrite) or "a" (append).
    """
    with open(filename, modo, encoding="utf-8") as f:
        f.write(contenido)

class _LatexTools:
    """
    LaTeX tools facade for experimental analysis.
    
    Provides unified access to formatting, rounding,
    and export of scientific results in LaTeX.
    """

    redondeo_incertidumbre = staticmethod(redondeo_incertidumbre)
    valor_pm = staticmethod(valor_pm)
    exportar = staticmethod(exportar)
    tabla_pm = staticmethod(tabla_pm)
    latex_quantity = staticmethod(latex_quantity)


latex_tools = _LatexTools()

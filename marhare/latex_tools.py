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

    orden = _orden_magnitud(sigma)
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


def tabla_latex(
    filas,
    *,
    headers=None,
    row_headers=None,
    caption=None,
    label=None,
    centrar=True,
    tamano=None,
    lineas=None,
    envolver=True,
    posicion="htbp",
    post=None,
):
    """
    Build a LaTeX table from preformatted rows.

    This is the only place where tabular environments are constructed.
    """
    if lineas is None:
        lineas = TABLA_CONFIG["lineas"]

    if not isinstance(filas, (list, tuple)) or len(filas) == 0:
        raise ValueError("tabla_latex(): filas must be a non-empty list/tuple")

    filas_norm = []
    for i, fila in enumerate(filas):
        if not isinstance(fila, (list, tuple)):
            raise TypeError(
                f"tabla_latex(): expected row list/tuple at index {i}, got {type(fila).__name__}"
            )
        if i == 0:
            cols = len(fila)
            if cols == 0:
                raise ValueError("tabla_latex(): rows must have at least one column")
        elif len(fila) != cols:
            raise ValueError("tabla_latex(): all rows must have the same length")
        filas_norm.append([str(celda) for celda in fila])

    if headers is not None and len(headers) != cols:
        raise ValueError("headers must have the same length as the number of columns.")
    if row_headers is not None and len(row_headers) != len(filas_norm):
        raise ValueError("row_headers must have the same length as the number of rows.")

    col_fmt = ""
    if row_headers:
        col_fmt += "l"
    col_fmt += "c" * cols

    tabular = [rf"\\begin{{tabular}}{{{col_fmt}}}"]

    if lineas == "booktabs":
        tabular.append(r"\\toprule")
    elif lineas == "hline":
        tabular.append(r"\\hline")

    if headers:
        header_line = []
        if row_headers:
            header_line.append("")
        header_line.extend(headers)
        tabular.append(" & ".join(header_line) + r" \\")
        if lineas in ("booktabs", "hline"):
            tabular.append(r"\\midrule" if lineas == "booktabs" else r"\\hline")

    for i, fila in enumerate(filas_norm):
        fila_out = []
        if row_headers:
            fila_out.append(row_headers[i])
        fila_out.extend(fila)
        tabular.append(" & ".join(fila_out) + r" \\")

    if lineas == "booktabs":
        tabular.append(r"\\bottomrule")
    elif lineas == "hline":
        tabular.append(r"\\hline")

    tabular.append(r"\\end{tabular}")
    latex_tabular = "\n".join(tabular)

    inserciones = ""
    if centrar:
        inserciones += "\\centering\n"
    if tamano:
        inserciones += f"\\{tamano}\n"

    if envolver:
        out = f"\\begin{{table}}[{posicion}]\n"
        out += inserciones
        if caption:
            out += f"\\caption{{{caption}}}\n"
        if label:
            out += f"\\label{{{label}}}\n"
        out += latex_tabular + "\n"
        if post:
            out += str(post) + "\n"
        out += "\\end{table}\n"
        return out

    out = inserciones + latex_tabular
    if post:
        out += "\n" + str(post)
    return out

def valor_pm(
    valor,
    sigma=None,
    *magnitudes,
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
    
    Accepts scalars, vectors, matrices, or quantity dicts. For scalars returns
    a plain LaTeX fragment without display math delimiters; for vectors/matrices
    generates a LaTeX table. For quantity dicts, generates a magnitudes table.
    
    Parameters
    ----------
    valor : float, array_like, uncertainties.uarray, or quantity dict
        Value(s) to format or a quantity dict.
    sigma : float, array_like, dict, or None
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

    >>> # Magnitudes table
    >>> valor_pm(q1, q2, q3)
    # Generates a table with symbols and per-row units
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

    # Flow: detect quantity dicts here, normalize via value_quantity, then render.
    mags = None
    if isinstance(valor, (list, tuple)) and not magnitudes:
        if any(isinstance(item, dict) for item in valor) and not all(
            isinstance(item, dict) for item in valor
        ):
            raise TypeError("valor_pm(): cannot mix dict and non-dict magnitudes")
    if isinstance(valor, dict) and not magnitudes:
        if sigma is None:
            # Single quantity dict: normalize first, then format as scalar or 1D table.
            v, s = value_quantity(valor)
            unit = valor.get("unit", None)
            symbol = valor.get("symbol", "") or ""
            v_arr = np.asarray(v)
            s_arr = np.asarray(s)

            if v_arr.shape == () and s_arr.shape == ():
                return valor_pm(
                    v,
                    s,
                    unidad=unit,
                    cifras=cifras,
                    siunitx=siunitx,
                )

            if v_arr.ndim != 1 or s_arr.ndim != 1 or v_arr.shape != s_arr.shape:
                raise ValueError(
                    "valor_pm(): single-quantity tables require 1D value/sigma arrays"
                )
            if siunitx:
                raise ValueError(
                    "valor_pm(): siunitx is not supported for magnitude tables"
                )

            header = f"{symbol} ({unit})" if unit else symbol
            filas = [
                [
                    _valor_pm_escalar(
                        float(v_arr[i]),
                        float(s_arr[i]),
                        unidad=None,
                        cifras=cifras,
                        siunitx=False,
                    )
                ]
                for i in range(v_arr.shape[0])
            ]

            return tabla_latex(
                filas,
                headers=[header],
                row_headers=None,
                caption=caption,
                label=label,
                centrar=centrar,
                tamano=tamano,
                lineas=lineas,
                envolver=envolver,
                posicion=posicion,
            )
        raise TypeError("valor_pm(): sigma must be None when passing a quantity dict")

    if isinstance(valor, (list, tuple)) and not magnitudes:
        if len(valor) == 0:
            raise ValueError("valor_pm(): empty list/tuple of magnitudes")
        if all(isinstance(item, dict) for item in valor):
            if sigma is not None:
                raise TypeError("valor_pm(): sigma must be None when passing a list of magnitudes")
            mags = list(valor)

    if mags is None and isinstance(valor, dict) and magnitudes:
        # Support valor_pm(q1, q2, q3) by treating dict sigma as a magnitude.
        if sigma is not None and not isinstance(sigma, dict):
            raise TypeError("valor_pm(): sigma must be None for magnitudes tables")
        mags = [valor]
        if isinstance(sigma, dict):
            mags.append(sigma)
        for m in magnitudes:
            mags.append(m)

    if mags is not None:
        # Multiple quantities: build a magnitudes table via tabla_latex.
        for i, mag in enumerate(mags):
            if not isinstance(mag, dict):
                raise TypeError(
                    f"valor_pm(): expected dict at index {i}, got {type(mag).__name__}"
                )

        if headers is not None or row_headers is not None:
            raise ValueError("valor_pm(): headers are not allowed for magnitudes tables")

        values = []
        sigmas = []
        symbols = []
        units = []
        any_vector = False

        for mag in mags:
            # value_quantity centralizes numeric normalization for quantity dicts.
            v, s = value_quantity(mag)
            v_arr = np.asarray(v)
            s_arr = np.asarray(s)
            if v_arr.shape != () or s_arr.shape != ():
                any_vector = True
            values.append(v_arr)
            sigmas.append(s_arr)
            symbols.append(mag.get("symbol", "") or "")
            units.append(mag.get("unit", None))

        if any_vector:
            # Vector magnitudes: build a data table with one column per quantity.
            if siunitx:
                raise ValueError(
                    "valor_pm(): siunitx is not supported for magnitude tables"
                )
            lengths = []
            for v_arr, s_arr in zip(values, sigmas):
                if v_arr.ndim != 1 or s_arr.ndim != 1:
                    raise ValueError(
                        "valor_pm(): vector magnitudes must be 1D arrays"
                    )
                if v_arr.shape != s_arr.shape:
                    raise ValueError(
                        "valor_pm(): vector magnitudes must have matching value/sigma shapes"
                    )
                lengths.append(v_arr.shape[0])

            if len(set(lengths)) != 1:
                raise ValueError(
                    "valor_pm(): vector magnitudes must share the same length"
                )

            headers = []
            for sym, unit in zip(symbols, units):
                if unit:
                    headers.append(f"{sym} ({unit})" if sym else f"({unit})")
                else:
                    headers.append(sym)

            filas = []
            for i in range(lengths[0]):
                fila = []
                for v_arr, s_arr in zip(values, sigmas):
                    # Units live in the header for vector tables.
                    fila.append(
                        _valor_pm_escalar(
                            float(v_arr[i]),
                            float(s_arr[i]),
                            unidad=None,
                            cifras=cifras,
                            siunitx=False,
                        )
                    )
                filas.append(fila)

            return tabla_latex(
                filas,
                headers=headers,
                row_headers=None,
                caption=caption,
                label=label,
                centrar=centrar,
                tamano=tamano,
                lineas=lineas,
                envolver=envolver,
                posicion=posicion,
            )

        filas = []
        for v_arr, s_arr, unit, symbol in zip(values, sigmas, units, symbols):
            # Always format the numeric cell through valor_pm to keep rendering centralized.
            formatted = valor_pm(
                float(v_arr.item()),
                float(s_arr.item()),
                unidad=unit,
                cifras=cifras,
                siunitx=siunitx,
                centrar=False,
                tamano=None,
                lineas=lineas,
                envolver=False,
                posicion=posicion,
                orientacion=orientacion,
            )
            filas.append([symbol, formatted])

        # Headers are fixed for magnitudes tables.
        headers = ["Magnitud", "Valor"]
        # Delegate the table layout to the shared table helper.
        return tabla_latex(
            filas,
            headers=headers,
            row_headers=None,
            caption=caption,
            label=label,
            centrar=centrar,
            tamano=tamano,
            lineas=lineas,
            envolver=envolver,
            posicion=posicion,
        )

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

    if siunitx and unidad is None:
        raise ValueError("valor_pm(): siunitx requires unidad for tables")

    filas_fmt = []
    for i in range(filas):
        fila = []
        for j in range(cols):
            fila.append(
                _valor_pm_escalar(
                    float(v[i, j]),
                    float(s[i, j]),
                    unidad=unidad if siunitx else None,
                    cifras=cifras,
                    siunitx=siunitx,
                )
            )
        filas_fmt.append(fila)

    unidad_global = rf"\,\mathrm{{{unidad}}}" if unidad and not siunitx else ""

    # Delegate the table layout to the shared table helper.
    return tabla_latex(
        filas_fmt,
        headers=headers,
        row_headers=row_headers,
        caption=caption,
        label=label,
        centrar=centrar,
        tamano=tamano,
        lineas=lineas,
        envolver=envolver,
        posicion=posicion,
        post=unidad_global or None,
    )


def latex_quantity(
    mag: dict | list[dict] | tuple[dict, ...],
    *,
    cifras: int = 2,
    siunitx: bool = False,
    **kw
) -> str:
    """
    Format a magnitude/constant or a list/tuple of magnitudes.

        Expects each magnitude to be a dict like:
            {
                "measure": (value, sigma) | None,
                "result": (value, sigma) | None,
                "unit": "m",
                "dimension": None | tuple,
                "expr": None
            }

    Notes:
    - Expressions are resolved elsewhere; this function assumes numeric data.
    - Uses value_quantity for numeric normalization only.
    - Delegates all formatting to `valor_pm`.
    """
    if isinstance(mag, (list, tuple)):
        # Multiple quantities: delegate table rendering to valor_pm.
        return valor_pm(mag, cifras=cifras, siunitx=siunitx, **kw)

    # Single quantity: delegate rendering to valor_pm.
    return valor_pm(mag, cifras=cifras, siunitx=siunitx, **kw)


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
    tabla_latex = staticmethod(tabla_latex)
    latex_quantity = staticmethod(latex_quantity)


latex_tools = _LatexTools()

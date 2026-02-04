"""
latex_tools.py
Herramientas para generar LaTeX científico desde Python.

Flujo de trabajo típico: Excel → Python → LaTeX

Funcionalidades:
- Redondeo metrológico estándar (1-2 cifras significativas en incertidumbre)
- Formato de valores con incertidumbres: (x ± σ)
- Generación de tablas LaTeX con estilo configurable
- Soporte para notación científica y siunitx
- Exportación a archivos .tex
"""

import math
from typing import Optional

import numpy as np
import sympy as sp
import uncertainties.unumpy as unp



# ============================================================
#  GUÍA RÁPIDA DE USO (latex_tools.py)
# ============================================================
#
# Import recomendado (fachada estilo objeto):
#   from mi_toolbox.latex_tools import latex_tools
#
# 1) redondeo_incertidumbre(valor, sigma, cifras=2)
# ------------------------------------------------------------
# Qué hace:
#   Redondea valor e incertidumbre con reglas metrológicas:
#   - sigma con 1 o 2 cifras significativas (según 'cifras')
#   - valor redondeado al mismo número de decimales que sigma
# Devuelve:
#   (valor_redondeado, sigma_redondeada, decimales)
# Ejemplo:
#   v_r, s_r, dec = latex_tools.redondeo_incertidumbre(3.14159, 0.0123, cifras=2)
#
#
# 2) valor_pm(valor, sigma=None, *, unidad=None, cifras=2, siunitx=False,
#             caption=None, label=None, centrar=None, tamano=None, lineas=None,
#             envolver=None, posicion=None, orientacion="vertical")
# ------------------------------------------------------------
# Qué hace:
#   Fachada versátil para presentar valores con incertidumbres en LaTeX.
#
#   ENTRADA escalar o uncertainties.uarray:
#     - Devuelve: \[ (v ± s) \]  (display math)
#   ENTRADA vector/matriz:
#     - Devuelve: tabla LaTeX con estilo desde TABLA_CONFIG
#
#   Parámetros principales:
#     - valor: float, array, o uncertainties.uarray
#     - sigma: incertidumbre (None si valor ya es uarray)
#     - unidad: str opcional con unidad LaTeX
#     - cifras: 1 o 2 sig. figs. en la incertidumbre (keyword-only)
#     - siunitx: bool, si True usa \SI{...}{...} (requiere unidad)
#     - orientacion: "vertical" o "horizontal" (solo para vectores 1D)
#
#   Parámetros de tabla (tomados de TABLA_CONFIG si no se pasan):
#     - caption, label, centrar, tamano, lineas, envolver, posicion
#
# Devuelve:
#   str (LaTeX)
#
# Ejemplos:
#   # Escalar
#   latex_tools.valor_pm(3.14159, 0.0123, cifras=2)
#   → \[(3.14 ± 0.01)\]
#
#   latex_tools.valor_pm(9.81, 0.05, unidad="m/s^2", cifras=2)
#   → \[(9.81 ± 0.05)\\,\\mathrm{m/s^2}\]
#
#   # Vector (columna por defecto)
#   x = np.array([1.0, 2.0, 3.0])
#   sx = 0.1
#   latex_tools.valor_pm(x, sx, cifras=1)
#   → tabla vertical (1 columna, 3 filas)
#
#   # Vector horizontal
#   latex_tools.valor_pm(x, sx, cifras=1, orientacion="horizontal")
#   → tabla horizontal (1 fila, 3 columnas)
#
#   # Matriz
#   V = np.array([[1.0, 2.0], [3.0, 4.0]])
#   sV = 0.1
#   latex_tools.valor_pm(V, sV, cifras=1, caption="Datos", label="tab:datos")
#   → tabla LaTeX 2×2 con caption/label
#
# Nota importante:
#   - 'cifras' es KEYWORD-ONLY: valor_pm(x, sx, cifras=1)
#   - Para tablas, se respetan las opciones de TABLA_CONFIG
#   - Sin unidad, no se usa siunitx (se ignora parámetro)
#
#
# 3) exportar(filename, contenido, modo="w")
# ------------------------------------------------------------
# Qué hace:
#   Escribe el string LaTeX en un archivo (por defecto sobrescribe).
# Devuelve:
#   None
# Ejemplo:
#   latex_tools.exportar("tablas/medidas.tex", tex)
#
#
# Configuración de tablas (TABLA_CONFIG):
#   TABLA_CONFIG["lineas"] = "hline"     # o "booktabs" o None
#   TABLA_CONFIG["centrar"] = True       # centra la tabla
#   TABLA_CONFIG["tamano"] = "small"     # o None, "footnotesize", etc.
#   TABLA_CONFIG["envolver"] = True      # activa \begin{table}...\end{table}
#   TABLA_CONFIG["posicion"] = "htbp"    # posición del flotante


# ============================================================================
# CONFIGURACIÓN GLOBAL DE TABLAS
# ============================================================================

TABLA_CONFIG = {
    "centrar": True,
    "tamano": None,          # None | "small" | "footnotesize" | ...
    "lineas": "hline",       # "booktabs" | "hline" | None
    "envolver": True,        # Usar entorno table flotante
    "posicion": "htbp",      # Posición del flotante
}


# ============================================================================
# UTILIDADES DE REDONDEO Y CIFRAS SIGNIFICATIVAS
# ============================================================================

def _orden_magnitud(x: float) -> int:
    """
    Devuelve el orden de magnitud decimal de x.
    
    Ejemplo: _orden_magnitud(0.0123) → -2
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
    Redondea valor e incertidumbre siguiendo reglas metrológicas estándar.
    
    Parámetros
    ----------
    valor : float
        Valor central a redondear.
    sigma : float
        Incertidumbre (debe ser > 0).
    cifras : int, default 2
        Número de cifras significativas para la incertidumbre (1 o 2).
    
    Devuelve
    --------
    tuple[float, float, int]
        (valor_redondeado, sigma_redondeada, decimales)
    
    Reglas:
    - La incertidumbre se da con 1 o 2 cifras significativas
    - El valor se redondea al mismo número de decimales que sigma
    """
    if sigma <= 0:
        raise ValueError("La incertidumbre debe ser positiva")

    orden = _orden_magnitud(sigma)
    decimales = -(orden - (cifras - 1))

    sigma_r = round(sigma, decimales)
    valor_r = round(valor, decimales)

    return valor_r, sigma_r, decimales


# ============================================================================
# FORMATO LATEX CON INCERTIDUMBRES
# ============================================================================

def _to_latex_sci(valor: float, cifras: int = 2) -> str:
    """
    Convierte un número en notación científica a formato LaTeX.
    
    Ejemplo: 5.3e-04 → "5.3 \\times 10^{-4}"
    """
    fmt = f"{{:.{cifras}e}}"
    s = fmt.format(valor)
    # Parsear mantisa y exponente
    # Formato típico: "5.30e-04"
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
    Formatea un valor escalar con incertidumbre en LaTeX.
    
    Devuelve string LaTeX del tipo:
        (x ± s)
    o con siunitx:
        \\SI{x \\pm s}{unidad}
    
    SIN delimitadores matemáticos externos (\\[ \\]).
    Usa notación científica automáticamente para valores muy grandes o pequeños.
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

    # --- configuración de tabla ---
    caption: Optional[str] = None,
    label: Optional[str] = None,
    centrar: Optional[bool] = None,
    tamano: Optional[str] = None,
    lineas: Optional[str] = None,     # "booktabs" | "hline" | None
    envolver: Optional[bool] = None,
    posicion: Optional[str] = None,

    orientacion: str = "vertical",    # solo para vectores 1D

    # --- encabezados ---
    headers: Optional[list[str]] = None,       # encabezados de columnas
    row_headers: Optional[list[str]] = None,   # encabezados de filas
):
    """
    Función principal para formatear valores con incertidumbres en LaTeX.
    
    Admite escalares, vectores y matrices. Para escalares devuelve expresión
    matemática \\[...\\], para vectores/matrices genera tabla LaTeX.
    
    Parámetros
    ----------
    valor : float, array_like, o uncertainties.uarray
        Valor(es) a formatear.
    sigma : float, array_like, o None
        Incertidumbre(s). Si None, se asume que valor ya es uarray.
    unidad : str, optional
        Unidad física en formato LaTeX (ej: "m/s^2").
    cifras : int, default 2
        Cifras significativas para la incertidumbre (1 o 2).
    siunitx : bool, default False
        Si True, usa formato \\SI{...}{...} (requiere unidad).
    caption : str, optional
        Caption para la tabla (solo vectores/matrices).
    label : str, optional
        Label LaTeX para referencia cruzada.
    centrar : bool, optional
        Si True, centra la tabla. Por defecto usa TABLA_CONFIG.
    tamano : str, optional
        Tamaño de fuente ("small", "footnotesize", etc.).
    lineas : str, optional
        Estilo de líneas: "booktabs", "hline" o None.
    envolver : bool, optional
        Si True, envuelve en entorno table flotante.
    posicion : str, optional
        Posición del flotante ("htbp", "H", etc.).
    orientacion : str, default "vertical"
        Para vectores 1D: "vertical" (columna) o "horizontal" (fila).
    headers : list[str], optional
        Encabezados de columnas.
    row_headers : list[str], optional
        Encabezados de filas.
    
    Devuelve
    --------
    str
        Código LaTeX formateado.
    
    Ejemplos
    --------
    >>> # Escalar
    >>> valor_pm(3.14159, 0.0123, cifras=2)
    '\\[(3.14 ± 0.01)\\]'
    
    >>> # Con unidad
    >>> valor_pm(9.81, 0.05, unidad="m/s^2", cifras=2)
    '\\[(9.81 ± 0.05)\\,\\mathrm{m/s^2}\\]'
    
    >>> # Vector (tabla)
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> valor_pm(x, 0.1, cifras=1, caption="Medidas")
    # Genera tabla LaTeX 3×1
    """

    # Defaults desde configuración global
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

    # Normalización de entrada
    if sigma is None:
        obj = valor
    else:
        obj = unp.uarray(valor, sigma)

    v = unp.nominal_values(obj)
    s = unp.std_devs(obj)

    # -------------------------------
    # CASO ESCALAR
    # -------------------------------
    if not isinstance(v, np.ndarray) or v.shape == ():
        v0 = float(np.asarray(v).item())
        s0 = float(np.asarray(s).item())
        contenido = _valor_pm_escalar(
            v0, s0, unidad=unidad, cifras=cifras, siunitx=siunitx
        )
        return rf"\[{contenido}\]"

    # -------------------------------
    # CASO VECTOR / MATRIZ
    # -------------------------------
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
            raise ValueError("orientacion debe ser 'vertical' o 'horizontal'.")

    filas, cols = v.shape

    # Validación de headers
    if headers is not None and len(headers) != cols:
        raise ValueError("headers debe tener la misma longitud que el número de columnas.")
    if row_headers is not None and len(row_headers) != filas:
        raise ValueError("row_headers debe tener la misma longitud que el número de filas.")

    # Formato columnas
    col_fmt = ""
    if row_headers:
        col_fmt += "l"
    col_fmt += "c" * cols

    tabular = [rf"\begin{{tabular}}{{{col_fmt}}}"]

    # Línea superior
    if lineas == "booktabs":
        tabular.append(r"\toprule")
    elif lineas == "hline":
        tabular.append(r"\hline")

    # Encabezados de columnas
    if headers:
        header_line = []
        if row_headers:
            header_line.append("")  # esquina vacía
        header_line.extend(headers)
        tabular.append(" & ".join(header_line) + r" \\")
        if lineas in ("booktabs", "hline"):
            tabular.append(r"\midrule" if lineas == "booktabs" else r"\hline")

    # Cuerpo
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

    # Línea inferior
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


# ============================================================================
# EXPORTACIÓN A ARCHIVO
# ============================================================================

def exportar(
    filename: str,
    contenido: str,
    modo: str = "w",
):
    """
    Escribe contenido LaTeX en un archivo .tex.
    
    Parámetros
    ----------
    filename : str
        Ruta del archivo a crear/modificar.
    contenido : str
        Código LaTeX a escribir.
    modo : str, default "w"
        Modo de apertura: "w" (sobrescribir) o "a" (añadir).
    """
    with open(filename, modo, encoding="utf-8") as f:
        f.write(contenido)

# ============================================================================
# Conexión Sympy-LaTeX
# ============================================================================

def expr_to_latex(expr: sp.Expr, *, simplify: bool = True) -> str:
    """
    Convierte una expresión simbólica de SymPy a LaTeX.

    Parámetros
    ----------
    expr : sympy.Expr
        Expresión simbólica.
    simplify : bool, default True
        Si True, simplifica antes de convertir.

    Devuelve
    --------
    str
        Cadena LaTeX.
    """
    if simplify:
        expr = sp.simplify(expr)

    return sp.latex(expr)



# ============================================================================
# FACHADA (INSTANCIA SINGLETON)
# ============================================================================

class _LatexTools:
    """
    Fachada de herramientas LaTeX para análisis experimental.
    
    Proporciona acceso unificado a funciones de formateo, redondeo
    y exportación de resultados científicos en LaTeX.
    """

    redondeo_incertidumbre = staticmethod(redondeo_incertidumbre)
    valor_pm = staticmethod(valor_pm)
    exportar = staticmethod(exportar)
    expr_to_latex = staticmethod(expr_to_latex)
    tabla_pm = staticmethod(tabla_pm)


latex_tools = _LatexTools()

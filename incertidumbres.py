"""
incertidumbres.py
Herramientas prácticas para trabajar con magnitudes con incertidumbre
de forma cómoda y consistente con el flujo Excel → Python → LaTeX.

Supuesto clave: la librería `uncertainties` SIEMPRE está instalada.
Puedes usar NumPy directamente (`np.sqrt`, `np.cos`, etc.) sobre ufloats
o arrays de ufloats (dtype=object) y la propagación se aplicará vía
`__array_ufunc__` de uncertainties.

Diseño:
- Fachada estilo objeto (singleton exportado: `incertidumbres`)
- Integra con `latex_tools` para salida metrológica en LaTeX
- Usa siempre la librería `uncertainties` (dependencia obligatoria)

GUÍA RÁPIDA DE USO
===================

from mi_toolbox.incertidumbres import incertidumbres
from mi_toolbox.latex_tools import latex_tools

# 1) Crear magnitudes con incertidumbre
uL = incertidumbres.u(1.250, 0.020)
ug = incertidumbres.u(9.81, 0.05)

# 2) Formatear para LaTeX con redondeo metrológico
tex_g = incertidumbres.pm(ug, unidad="m/s^2", cifras=2)
# → "(9.81 ± 0.05) \mathrm{m/s^2}" (según reglas de latex_tools)

# 3) Construir una tabla de datos a partir de arrays de ufloat/tuplas
#    (ajusta a latex_tools.tabla_datos_con_incertidumbres)
#    x_u y V_u son listas/arrays de ufloat/tuplas (valor, sigma)
tex_tabla = incertidumbres.tabla_desde_u(
    ["x (cm)", "V (V)"],
    [x_u, V_u],
    cifras=1,
    caption="Medidas",
    label="tab:medidas"
)

# 4) Propagación analítica (sin uncertainties):
res = incertidumbres.propagacion(
    "x*y",
    valores={"x": 2.0, "y": 3.5},
    incertidumbres={"x": 0.05, "y": 0.10}
)
# res → {"valor": 7.0, "incertidumbre": ...}

"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np

# Dependencia obligatoria: uncertainties SIEMPRE instalada
from uncertainties import ufloat
from uncertainties.core import AffineScalarFunc
import uncertainties.unumpy as unp

from .latex_tools import latex_tools
from .estadistica import estadistica





class _Incertidumbres:
    """Fachada de utilidades para magnitudes con incertidumbre.

    Objetivos del diseño:
    - Encapsular la posible dependencia con `uncertainties`.
    - Ofrecer una API mínima, clara y en español.
    - Integración directa con `latex_tools` (formato metrológico y tablas).
    """

    # --------- Construcción ---------
    @staticmethod
    def u(x, sigmax=0.0):
        """
        Construye una magnitud con incertidumbre (caso escalar o array) usando `uncertainties`.

        Reglas:
        - Si x es escalar (0-D) -> devuelve `ufloat(x, sigmax)`.
        - Si x es array-like (>=1-D) -> devuelve `unp.uarray(x, sigmax)` (mismo shape).
        - `sigmax` puede ser escalar (se “broadcastea”) o array con la misma forma que `x`.

        Parámetros
        ----------
        x : number | array-like
            Valor(es) nominal(es).
        sigmax : number | array-like, default 0.0
            Incertidumbre(s) (desviación típica) asociada(s).

        Devuelve
        --------
        ufloat | unumpy.uarray (ndarray de objetos uncertainties)
        """
        # Convertimos x a ndarray para inspeccionar dimensionalidad (0-D vs N-D)
        x_arr = np.asarray(x)

        # Caso escalar (0-D): devolvemos ufloat
        if x_arr.ndim == 0:
            # Asegurar que sigmax es escalar numérico
            s = np.asarray(sigmax)
            if s.ndim != 0:
                raise ValueError("Para x escalar, sigmax debe ser escalar.")
            return ufloat(float(x_arr), float(s))

        # Caso array (>=1-D): construimos uarray con broadcasting controlado
        x_arr = x_arr.astype(float, copy=False)

        s_arr = np.asarray(sigmax)
        if s_arr.ndim == 0:
            # Broadcast escalar a la forma de x
            s_arr = np.full_like(x_arr, float(s_arr), dtype=float)
        else:
            s_arr = s_arr.astype(float, copy=False)
            if s_arr.shape != x_arr.shape:
                # Intentamos broadcasting estándar; si no, error claro
                try:
                    s_arr = np.broadcast_to(s_arr, x_arr.shape).astype(float, copy=False)
                except Exception as e:
                    raise ValueError(
                        f"sigmax no es compatible con la forma de x: "
                        f"x.shape={x_arr.shape}, sigmax.shape={np.asarray(sigmax).shape}"
                    ) from e

        return unp.uarray(x_arr, s_arr)
    # --------- Accesores ---------
    


incertidumbres = _Incertidumbres()

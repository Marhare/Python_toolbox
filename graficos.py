"""
graficos.py
Gráficas serias (pero bonitas) en Matplotlib, estilo "libro de física".

Objetivo:
- Tipografía limpia, ticks finos, rejilla suave, márgenes correctos
- Exportación directa a PDF/SVG (ideal para LaTeX)
- API sencilla: graficos.linea(...), graficos.errorbar(...), etc.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union, Iterable, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from cycler import cycler


# ============================================================
#  USO RÁPIDO (mini guía)
# ============================================================
#
# from mi_toolbox.graficos import graficos
# import matplotlib.pyplot as plt
#
# ax = graficos.linea(x, y, marker="o", label="Datos",
#                     xlabel="x (cm)", ylabel="V (V)", title="Título")
# plt.show()
#
# fig, ax = graficos.figura(figsize=(6.0, 3.6))
# graficos.errorbar(x, y, sy=sy, ax=ax, label="Datos ± error", marker="o")
# graficos.guardar(fig, "figuras/nombre", formatos=("pdf","png"))
# ============================================================


# ============================================================
#  CONFIG GLOBAL DE GRÁFICAS
# ============================================================

GRAFICOS_CONFIG = {
    # Tamaños típicos para LaTeX (aprox): 1 columna ~ 3.3in, 2 columnas ~ 6.9in
    "figsize": (4.8, 3.2),     # (ancho, alto) en pulgadas
    "dpi": 160,

    "grid": True,
    "grid_alpha": 0.14,
    "grid_lw": 0.6,

    "lw": 1.8,                 # grosor de líneas
    "ms": 5.0,                 # tamaño de markers
    "capsize": 2.8,            # capsize en errorbars

    "legend": True,
    "legend_frame": False,

    "tight": True,
    "minor_ticks": True,
    "spines": ("left", "bottom"),  # ejes visibles

    "save_formats": ("pdf",),      # recomendado: pdf; añade "png" si quieres
    "transparent": False,

    # Tipografía/Math: estilo clásico científico sin depender de TeX instalado
    "font_family": "STIXGeneral",
    "math_fontset": "stix",
    "font_size": 11,
    "label_size": 12,
    "title_size": 12,
    "tick_size": 10,

    # Si activas usetex=True necesitas LaTeX instalado en el sistema
    "usetex": False,

    # -------- Paleta sobria (sin naranja/azul "tab" por defecto) --------
    # idea: datos en gris oscuro, ajuste/interpolación en casi negro o azul muy profundo
    "palette": {
        "ink":   "#1b1f24",   # casi negro (líneas principales)
        "data":  "#2f343a",   # gris oscuro (puntos)
        "error": "#7a828a",   # gris medio (barras de error)
        "accent":"#1f4e79",   # azul muy profundo (si quieres destacar algo)
    },
    # ciclo de colores serio si dibujas varias curvas
    "cycle": ("#1b1f24", "#1f4e79", "#2d6a4f", "#6c2c2c", "#5b5f97"),

    # datos con marcador hueco por defecto (más “paper”)
    "hollow_markers": True,
}


# ============================================================
#  UTILS
# ============================================================

def _split_kwargs(kwargs: Dict[str, Any]):
    """
    Divide kwargs en:
    - style_kwargs: claves que pertenecen a GRAFICOS_CONFIG (afectan rcParams)
    - plot_kwargs: el resto (se pasan a plot/scatter/errorbar)
    """
    style = {}
    plot = {}
    for k, v in kwargs.items():
        if k in GRAFICOS_CONFIG:
            style[k] = v
        else:
            plot[k] = v
    return style, plot


# ============================================================
#  ESTILO BASE (rcParams)
# ============================================================

def aplicar_estilo(**overrides):
    """
    Aplica estilo global. Puedes sobreescribir cualquier clave de GRAFICOS_CONFIG.

    Ej:
        aplicar_estilo(usetex=True, figsize=(6.5,4))
    """
    cfg = dict(GRAFICOS_CONFIG)
    cfg.update(overrides)

    plt.rcParams.update({
        "figure.dpi": cfg["dpi"],
        "savefig.dpi": cfg["dpi"],

        "font.family": cfg["font_family"],
        "mathtext.fontset": cfg["math_fontset"],
        "font.size": cfg["font_size"],

        "axes.titlesize": cfg["title_size"],
        "axes.labelsize": cfg["label_size"],
        "xtick.labelsize": cfg["tick_size"],
        "ytick.labelsize": cfg["tick_size"],

        "axes.spines.top": False,
        "axes.spines.right": False,

        "axes.grid": cfg["grid"],
        "grid.alpha": cfg["grid_alpha"],
        "grid.linewidth": cfg["grid_lw"],

        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,

        "lines.linewidth": cfg["lw"],
        "lines.markersize": cfg["ms"],

        "text.usetex": cfg["usetex"],

        # ciclo de colores serio (evitas tab:blue/tab:orange)
        "axes.prop_cycle": cycler(color=list(cfg["cycle"])),
    })

    return cfg


def _post_ax(ax, *, cfg, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, legend=None):
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Minor ticks finos
    if cfg.get("minor_ticks", True):
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    # Spines visibles
    sp = set(cfg.get("spines", ("left", "bottom")))
    ax.spines["left"].set_visible("left" in sp)
    ax.spines["bottom"].set_visible("bottom" in sp)
    ax.spines["top"].set_visible("top" in sp)
    ax.spines["right"].set_visible("right" in sp)

    # Leyenda (solo si hay labels reales)
    if legend is None:
        legend = cfg.get("legend", True)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        labels_validos = [lab for lab in labels if lab and not lab.startswith("_")]
        if len(labels_validos) > 0:
            ax.legend(frameon=cfg.get("legend_frame", False))


# ============================================================
#  API DE GRÁFICAS
# ============================================================

def figura(
    figsize: Optional[Tuple[float, float]] = None,
    nrows: int = 1,
    ncols: int = 1,
    sharex: bool = False,
    sharey: bool = False,
    **kwargs
):
    """
    Crea figura + ejes con el estilo aplicado.
    Devuelve (fig, ax) o (fig, axs) si hay varios subplots.

    kwargs: puedes pasar claves de GRAFICOS_CONFIG (dpi, grid_alpha, lw, etc.)
    """
    style_kwargs, _ = _split_kwargs(kwargs)
    cfg = aplicar_estilo(**style_kwargs)

    if figsize is None:
        figsize = cfg["figsize"]

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        constrained_layout=cfg.get("tight", True)
    )

    return fig, axs


def guardar(fig, filename: Union[str, Path], formatos: Optional[Iterable[str]] = None,
            transparent: Optional[bool] = None, close: bool = True):
    """
    Guarda la figura en uno o varios formatos (PDF recomendado para LaTeX).
    """
    cfg = aplicar_estilo()
    if formatos is None:
        formatos = cfg["save_formats"]
    if transparent is None:
        transparent = cfg["transparent"]

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    for ext in formatos:
        out = filename.with_suffix("." + ext)
        fig.savefig(out, bbox_inches="tight", transparent=transparent)

    if close:
        plt.close(fig)


def linea(x, y, *, ax=None, label=None, xlabel=None, ylabel=None, title=None,
          xlim=None, ylim=None, marker=None, legend=None, **kwargs):
    """
    Línea clásica x-y.
    kwargs:
      - claves de GRAFICOS_CONFIG: afectan estilo global (lw, grid_alpha, etc.)
      - resto: se pasa a ax.plot (color=..., linestyle=..., alpha=..., etc.)
    """
    style_kwargs, plot_kwargs = _split_kwargs(kwargs)
    cfg = aplicar_estilo(**style_kwargs)

    if ax is None:
        fig, ax = figura(**style_kwargs)
        ax = ax if not isinstance(ax, np.ndarray) else ax.ravel()[0]

    # defaults sobrios si no pasas color explícito
    if "color" not in plot_kwargs:
        plot_kwargs["color"] = cfg["palette"]["ink"]

    ax.plot(x, y, marker=marker, label=label, **plot_kwargs)
    _post_ax(ax, cfg=cfg, xlabel=xlabel, ylabel=ylabel, title=title, xlim=xlim, ylim=ylim, legend=legend)
    return ax


def dispersion(x, y, *, ax=None, label=None, xlabel=None, ylabel=None, title=None,
              xlim=None, ylim=None, legend=None, **kwargs):
    """
    Scatter limpio.
    kwargs:
      - claves de GRAFICOS_CONFIG: estilo global
      - resto: se pasa a ax.scatter (color=..., s=..., alpha=..., etc.)
    """
    style_kwargs, sc_kwargs = _split_kwargs(kwargs)
    cfg = aplicar_estilo(**style_kwargs)

    if ax is None:
        fig, ax = figura(**style_kwargs)
        ax = ax if not isinstance(ax, np.ndarray) else ax.ravel()[0]

    sc_kwargs.setdefault("color", cfg["palette"]["data"])
    if cfg.get("hollow_markers", True):
        sc_kwargs.setdefault("facecolors", "none")
        sc_kwargs.setdefault("edgecolors", cfg["palette"]["data"])
        sc_kwargs.setdefault("linewidths", 1.0)

    ax.scatter(x, y, label=label, **sc_kwargs)
    _post_ax(ax, cfg=cfg, xlabel=xlabel, ylabel=ylabel, title=title, xlim=xlim, ylim=ylim, legend=legend)
    return ax


def errorbar(x, y, sx=None, sy=None, *, ax=None, label=None, xlabel=None, ylabel=None, title=None,
             xlim=None, ylim=None, marker="o", legend=None, **kwargs):
    """
    Puntos con barras de error (xerr/yerr).
    kwargs:
      - claves de GRAFICOS_CONFIG: estilo global
      - resto: se pasa a ax.errorbar (color=..., ecolor=..., linestyle=..., etc.)
    """
    style_kwargs, eb_kwargs = _split_kwargs(kwargs)
    cfg = aplicar_estilo(**style_kwargs)

    if ax is None:
        fig, ax = figura(**style_kwargs)
        ax = ax if not isinstance(ax, np.ndarray) else ax.ravel()[0]

    eb_kwargs.setdefault("color", cfg["palette"]["data"])
    eb_kwargs.setdefault("ecolor", cfg["palette"]["error"])
    eb_kwargs.setdefault("capsize", cfg.get("capsize", 2.8))

    # marcador hueco por defecto
    if cfg.get("hollow_markers", True):
        eb_kwargs.setdefault("markerfacecolor", "none")
        eb_kwargs.setdefault("markeredgecolor", eb_kwargs.get("color", cfg["palette"]["data"]))

    ax.errorbar(
        x, y,
        xerr=sx, yerr=sy,
        fmt=marker,
        label=label,
        **eb_kwargs
    )
    _post_ax(ax, cfg=cfg, xlabel=xlabel, ylabel=ylabel, title=title, xlim=xlim, ylim=ylim, legend=legend)
    return ax


def ajuste_lineal(x, y, *, ax=None, label_datos=None, label_ajuste=None,
                  xlabel=None, ylabel=None, title=None, legend=True,
                  dibujar_datos=True, dibujar_ajuste=True,
                  **kwargs):
    """
    Ajuste lineal rápido (mínimos cuadrados) y dibuja recta.

    Devuelve: (ax, m, b)
      y = m x + b
    kwargs:
      - claves de GRAFICOS_CONFIG: estilo global
      - resto: se usa para el plot de la recta (color=..., linestyle=..., etc.)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m, b = np.polyfit(x, y, 1)

    style_kwargs, fit_kwargs = _split_kwargs(kwargs)
    cfg = aplicar_estilo(**style_kwargs)

    if ax is None:
        fig, ax = figura(**style_kwargs)
        ax = ax if not isinstance(ax, np.ndarray) else ax.ravel()[0]

    if dibujar_datos:
        # datos: gris oscuro, huecos
        sc_kwargs = {}
        sc_kwargs["color"] = cfg["palette"]["data"]
        if cfg.get("hollow_markers", True):
            sc_kwargs["facecolors"] = "none"
            sc_kwargs["edgecolors"] = cfg["palette"]["data"]
            sc_kwargs["linewidths"] = 1.0
        ax.scatter(x, y, label=label_datos, **sc_kwargs)

    if dibujar_ajuste:
        xx = np.linspace(np.min(x), np.max(x), 400)
        yy = m * xx + b

        fit_kwargs.setdefault("color", cfg["palette"]["ink"])
        fit_kwargs.setdefault("linewidth", cfg["lw"])

        if label_ajuste is None:
            label_ajuste = rf"Ajuste: $y = {m:.3g}x + {b:.3g}$"

        ax.plot(xx, yy, label=label_ajuste, **fit_kwargs)

    _post_ax(ax, cfg=cfg, xlabel=xlabel, ylabel=ylabel, title=title, legend=legend)
    return ax, float(m), float(b)


# ============================================================
#  FACHADA (MISMO ESTILO QUE TUS OTROS MÓDULOS)
# ============================================================

class _Graficos:
    """Fachada: graficos.linea(...), graficos.errorbar(...), etc."""
    aplicar_estilo = staticmethod(aplicar_estilo)
    figura = staticmethod(figura)
    guardar = staticmethod(guardar)

    linea = staticmethod(linea)
    dispersion = staticmethod(dispersion)
    errorbar = staticmethod(errorbar)
    ajuste_lineal = staticmethod(ajuste_lineal)


graficos = _Graficos()

"""
graphics.py
============================================================
High‑level, low‑friction scientific visualization module.

PHILOSOPHY:
- The user declares INTENT (what they have: data, errors, fits, bands)
- The module automatically decides HOW to draw it

DOES NOT COMPUTE DATA. It only plots already computed results.

LAYERED DESIGN:
    1. Semantic objects (Series, SeriesWithError, Histogram, Fit, Band, Series3D, Panel, Scene)
    2. Core engine: plot(*objetos, layout=None, dims="2D", show=True)
    3. Automatic layout
    4. Future extensibility (animations)

ANIMATION PREPARATION:
    - Scene encapsulates the full graphic structure (panels + metadata)
    - Scene is the natural unit for animations: defines WHAT is drawn
    - updater defines HOW the scene evolves over time
    - Clean separation: graphics.py (static) + animations.py (temporal)

============================================================
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from cycler import cycler
from dataclasses import dataclass, field

try:
    from .uncertainties import value_quantity
except ImportError:
    from uncertainties import value_quantity


# ============================================================
#  GLOBAL CONFIG
# ============================================================

PLOT_DEFAULTS = {
    # Typical sizes for LaTeX (approx): 1 column ~ 3.3in, 2 columns ~ 6.9in
    "figsize": (4.8, 3.2),     # (width, height) in inches
    "dpi": 160,

    "grid": True,
    "grid_alpha": 0.14,
    "grid_lw": 0.6,

    "lw": 1.8,                 # line width
    "ms": 5.0,                 # marker size
    "capsize": 2.8,            # capsize in errorbars

    "legend": True,
    "legend_frame": False,

    "tight": True,
    "minor_ticks": True,
    "spines": ("left", "bottom"),  # visible spines

    "save_formats": ("pdf",),      # recommended: pdf; add "png" if desired
    "transparent": False,

    # Typography/Math: classic scientific style without depending on installed TeX
    "font_family": "STIXGeneral",
    "math_fontset": "stix",
    "font_size": 11,
    "label_size": 12,
    "title_size": 12,
    "tick_size": 10,

    # If you enable usetex=True you need LaTeX installed on the system
    "usetex": False,

    # -------- Subdued palette (no default "tab" orange/blue) --------
    # idea: data in dark gray, fit/interpolation in near‑black or deep blue
    "palette": {
        "ink":   "#1b1f24",   # near black (main lines)
        "data":  "#2f343a",   # gris oscuro (puntos)
        "error": "#7a828a",   # gris medio (barras de error)
        "accent":"#1f4e79",   # azul muy profundo (si quieres destacar algo)
    },
    # sober color cycle if you draw multiple curves
    "cycle": ("#1b1f24", "#1f4e79", "#2d6a4f", "#6c2c2c", "#5b5f97"),

    # hollow markers by default (more “paper‑like”)
    "hollow_markers": True,
}


# ============================================================
#  LAYER 1: SEMANTIC OBJECTS (NO calculations, NO plotting)
# ============================================================

@dataclass
class Series:
    """
    Represents a simple data series: x, y.
    
    INPUT:
        x: array_like -> independent variable
        y: array_like -> dependent variable
        label: str | None -> legend label
        marker: str | None -> marker type ('o', 's', '^', 'D', etc.) or None for hollow circles
    
    PURPOSE:
        Store intent: "I want to plot these points"
        No calculations or decorations.
    """
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    label: Optional[str] = None
    marker: Optional[str] = None

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("Series: 'x' and 'y' must have the same length")
    
    def __hash__(self):
        return id(self)


@dataclass
class SeriesWithError:
    """
    Series with (symmetric) error bars.
    
    INPUT:
        x: array_like -> independent variable
        y: array_like -> dependent variable
        sy: array_like | None -> error in y (optional)
        sx: array_like | None -> error in x (optional)
        label: str | None -> legend label
    
    PURPOSE:
        "I have experimental points with uncertainties"
        The engine draws error bars automatically.
    """
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    sy: Optional[np.ndarray] = None
    sx: Optional[np.ndarray] = None
    label: Optional[str] = None

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.sy is not None:
            self.sy = np.asarray(self.sy, dtype=float)
        if self.sx is not None:
            self.sx = np.asarray(self.sx, dtype=float)
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("SeriesWithError: 'x' and 'y' must have the same length")
        if self.sy is not None and self.sy.shape[0] != self.y.shape[0]:
            raise ValueError("SeriesWithError: 'sy' must have the same length as 'y'")
        if self.sx is not None and self.sx.shape[0] != self.x.shape[0]:
            raise ValueError("SeriesWithError: 'sx' must have the same length as 'x'")
    
    def __hash__(self):
        return id(self)


@dataclass
class Histogram:
    """
    Histogram of a variable.
    
    INPUT:
        data: array_like -> raw data
        bins: int | array_like -> number of bins or edges (Matplotlib standard)
        label: str | None -> label
    
    PURPOSE:
        "I want to see the distribution of these data"
        The engine draws automatically with a clean style.
    """
    data: np.ndarray = field(default_factory=lambda: np.array([]))
    bins: Union[int, np.ndarray] = 30
    label: Optional[str] = None

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=float)
    
    def __hash__(self):
        return id(self)


@dataclass
class Fit:
    """
    Fitted curve (already computed in another module).
    
    INPUT:
        x: array_like -> points where the fit is evaluated
        yfit: array_like -> fitted values
        label: str | None -> label (e.g., "Linear fit", "Spline")
    
    PURPOSE:
        "Here is the fitted curve, draw it over the data"
        Does not compute anything, only draws the line.
    """
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    yfit: np.ndarray = field(default_factory=lambda: np.array([]))
    label: Optional[str] = None

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.yfit = np.asarray(self.yfit, dtype=float)
        if self.x.shape[0] != self.yfit.shape[0]:
            raise ValueError("Fit: 'x' and 'yfit' must have the same length")
    
    def __hash__(self):
        return id(self)


@dataclass
class Band:
    """
    Confidence/prediction band (already computed).
    
    INPUT:
        x: array_like -> x points
        y_low: array_like -> lower bound of the band
        y_high: array_like -> upper bound of the band
        label: str | None -> label (e.g., "95% band")
    
    PURPOSE:
        "Draw this shaded region between y_low and y_high"
        Typically used together with Fit.
    """
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y_low: np.ndarray = field(default_factory=lambda: np.array([]))
    y_high: np.ndarray = field(default_factory=lambda: np.array([]))
    label: Optional[str] = None

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.y_low = np.asarray(self.y_low, dtype=float)
        self.y_high = np.asarray(self.y_high, dtype=float)
        if self.x.shape[0] != self.y_low.shape[0] or self.x.shape[0] != self.y_high.shape[0]:
            raise ValueError("Band: 'x', 'y_low' and 'y_high' must have the same length")
    
    def __hash__(self):
        return id(self)


@dataclass
class Series3D:
    """
    Represents a 3D data series: x, y, z.
    
    INPUT:
        x: array_like -> independent variable 1
        y: array_like -> independent variable 2
        z: array_like -> dependent variable
        label: str | None -> legend label
    
    PURPOSE:
        "Draw this trajectory or point cloud in 3D"
        Does NOT compute anything.
        Requires dims="3D" in plot().
    """
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    z: np.ndarray = field(default_factory=lambda: np.array([]))
    label: Optional[str] = None

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        self.z = np.asarray(self.z, dtype=float)
        if self.x.shape[0] != self.y.shape[0] or self.x.shape[0] != self.z.shape[0]:
            raise ValueError("Series3D: 'x', 'y' and 'z' must have the same length")
    
    def __hash__(self):
        return id(self)


@dataclass
class Panel:
    """
    Composition panel: groups objects that should be drawn on the same axis.

    INPUT:
        *objetos: semantic objects (Series, SeriesWithError, Histogram, Fit, Band, etc.)

    PURPOSE:
        Make explicit what should be overlaid in the same subplot.
        Does not compute or draw; only stores intent.
    """
    objetos: List[Any] = field(default_factory=list)

    def __init__(self, *objetos: Any):
        self.objetos = list(objetos)


@dataclass
class Scene:
    """
    Complete graphic scene: represents a reusable figure.
    
    INPUT:
        *paneles: Panel or semantic objects (normalized to Panel)
        layout: str | None -> explicit layout ("2x3") or None (auto)
        dims: str -> "2D" or "3D"
        figsize: tuple | None -> figure size
        xlabel, ylabel, title: str | None -> global labels
    
    PURPOSE:
        Encapsulate graphic STRUCTURE (what objects, how to arrange them).
        Does NOT encapsulate AESTHETICS (style kwargs are passed to plot()).
        Does NOT compute or draw anything.
        Natural base for future animations.
    
    DESIGN DECISIONS:
        - Scene is MUTABLE by design (allows modifying panels after creation)
        - Early validation: checks dims-object compatibility on construction
        - Must be passed as a SINGLE ARGUMENT to plot(): plot(scene)
        - Style is controlled externally: plot(scene, lw=2, dpi=200)
        
    EXAMPLES:
        scene = Scene(Series(x, y), Histogram(data))
        scene = Scene(Panel(Series(...), Fit(...)), layout="1x2", dims="2D")
        plot(scene)  # draw the scene with default style
        plot(scene, dpi=300)  # same content, different style
    """
    paneles: List[Panel] = field(default_factory=list)
    layout: Optional[str] = None
    dims: str = "2D"
    figsize: Optional[Tuple[float, float]] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    title: Optional[str] = None
    
    def __init__(
        self,
        *paneles,
        layout: Optional[str] = None,
        dims: str = "2D",
        figsize: Optional[Tuple[float, float]] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None
    ):
        # Normalize loose objects to Panel (same as plot())
        self.paneles = []
        for obj in paneles:
            if isinstance(obj, Panel):
                self.paneles.append(obj)
            else:
                self.paneles.append(Panel(obj))
        
        # Early validation: dims-object compatibility
        if dims == "3D":
            for panel in self.paneles:
                for obj in panel.objetos:
                    if not isinstance(obj, Series3D):
                        raise TypeError(
                            f"Scene con dims='3D' solo acepta objetos Series3D. "
                            f"Encontrado: {type(obj).__name__}"
                        )
        elif dims == "2D":
            for panel in self.paneles:
                for obj in panel.objetos:
                    if isinstance(obj, Series3D):
                        raise TypeError(
                            f"Scene con dims='2D' no acepta objetos Series3D. "
                            f"Usa dims='3D' o cambia a objetos 2D."
                        )
        
        self.layout = layout
        self.dims = dims
        self.figsize = figsize
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title


# ============================================================
#  UTILIDADES INTERNAS
# ============================================================

def _apply_style(**overrides):
    """
    Aplica estilo global a matplotlib rcParams.
    """
    cfg = dict(PLOT_DEFAULTS)
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
        "axes.prop_cycle": cycler(color=list(cfg["cycle"])),
    })

    return cfg


def _layout_shape(n: int) -> Tuple[int, int]:
    """
    Automatically decide grid shape (nrows, ncols) for n plots.
    
    INPUT:
        n: int -> number of subplots
    
    OUTPUT:
        (nrows, ncols)
    
    RULE:
        1  -> 1x1
        2  -> 1x2
        3  -> 2x2  (leaves 1 empty)
        4  -> 2x2
        5  -> 2x3  (leaves 1 empty)
        6  -> 2x3
        7  -> 3x3  (leaves 2 empty)
        etc.
    """
    if n == 1:
        return 1, 1
    elif n == 2:
        return 1, 2
    elif n <= 4:
        return 2, 2
    elif n <= 6:
        return 2, 3
    else:
        # for n >= 7: use ceil(sqrt(n))
        side = int(np.ceil(np.sqrt(n)))
        return side, side


def _parse_layout(layout: str) -> Tuple[int, int]:
    parts = layout.split("x")
    if len(parts) != 2:
        raise ValueError(f"layout must be 'NxM' (e.g., '2x3'), received: {layout}")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(f"layout must be 'NxM' with integer N,M, received: {layout}")


def _create_axis(fig, idx: int, nrows: int, ncols: int, dims: str) -> plt.Axes:
    """
    Create a 2D or 3D axis depending on dims.
    
    INPUT:
        fig: matplotlib figure
        idx: subplot index (1-based)
        nrows, ncols: subplot layout
        dims: "2D" or "3D"
    
    OUTPUT:
        ax: matplotlib Axes (2D) or Axes3D (3D)
    
    NOTES:
        - If dims="3D", create with projection="3d"
        - If dims="2D", create a normal axis
    """
    if dims == "3D":
        return fig.add_subplot(nrows, ncols, idx, projection="3d")
    else:
        return fig.add_subplot(nrows, ncols, idx)


def _is_axis_3d(ax) -> bool:
    """
    Detect whether an axis is 3D.
    
    INPUT:
        ax: matplotlib Axes
    
    OUTPUT:
        bool: True if the axis has 3D projection
    
    NOTES:
        Centralizes detection to avoid inconsistencies.
        3D axes have the 'zaxis' attribute.
    """
    return hasattr(ax, 'zaxis')


def _post_process_axis(ax, cfg):
    """
    Apply standard post‑processing to an axis.
    Used internally after each plot.
    
    NOTES:
        - For 3D axes, skip minor ticks and spines (not applicable)
        - For 2D axes, apply standard configuration
    """
    # 3D axes: do not apply 2D logic
    if _is_axis_3d(ax):
        return
    
    # Minor ticks (2D only)
    if cfg.get("minor_ticks", True):
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    # Spines visibility (2D only)
    sp = set(cfg.get("spines", ("left", "bottom")))
    ax.spines["left"].set_visible("left" in sp)
    ax.spines["bottom"].set_visible("bottom" in sp)
    ax.spines["top"].set_visible("top" in sp)
    ax.spines["right"].set_visible("right" in sp)


def _apply_legend(ax, cfg):
    """
    Apply legend only if there are valid labels.
    """
    if not cfg.get("legend", True):
        return

    handles, labels = ax.get_legend_handles_labels()
    valid_labels = [lab for lab in labels if lab and not lab.startswith("_")]
    if len(valid_labels) > 0:
        ax.legend(frameon=cfg.get("legend_frame", False))


# ============================================================
#  CAPA 2: MOTOR CENTRAL DE PLOTTING
# ============================================================

def _is_scene_like(obj: Any) -> bool:
    return isinstance(obj, Scene) or (
        hasattr(obj, "paneles") and hasattr(obj, "layout") and hasattr(obj, "dims")
    )


def _is_panel_like(obj: Any) -> bool:
    return isinstance(obj, Panel) or hasattr(obj, "objetos")


def _is_series3d_like(obj: Any) -> bool:
    return hasattr(obj, "x") and hasattr(obj, "y") and hasattr(obj, "z")


def _is_series_with_error_like(obj: Any) -> bool:
    return hasattr(obj, "x") and hasattr(obj, "y") and (hasattr(obj, "sy") or hasattr(obj, "sx"))


def _is_histogram_like(obj: Any) -> bool:
    return hasattr(obj, "data") and hasattr(obj, "bins")


def _is_fit_like(obj: Any) -> bool:
    return hasattr(obj, "x") and hasattr(obj, "yfit")


def _is_band_like(obj: Any) -> bool:
    return hasattr(obj, "x") and hasattr(obj, "y_low") and hasattr(obj, "y_high")


def _is_series_like(obj: Any) -> bool:
    return hasattr(obj, "x") and hasattr(obj, "y")


def _is_semantic_like(obj: Any) -> bool:
    return (
        _is_scene_like(obj)
        or _is_panel_like(obj)
        or isinstance(obj, (Series, SeriesWithError, Histogram, Fit, Band, Series3D))
        or _is_series_with_error_like(obj)
        or _is_histogram_like(obj)
        or _is_fit_like(obj)
        or _is_band_like(obj)
        or _is_series3d_like(obj)
        or _is_series_like(obj)
    )


def _looks_like_data(obj: Any) -> bool:
    return isinstance(obj, (list, tuple, np.ndarray))


def _looks_like_data_or_quantity(obj: Any) -> bool:
    return _looks_like_data(obj) or _is_quantity_like(obj)


def _parse_histogram(hist) -> Histogram:
    if isinstance(hist, Histogram) or _is_histogram_like(hist):
        return hist
    if isinstance(hist, (list, tuple)) and len(hist) == 2:
        data, bins = hist
        return Histogram(data, bins)
    return Histogram(hist)


def _parse_band(x, bands) -> Band:
    if isinstance(bands, Band) or _is_band_like(bands):
        return bands
    if isinstance(bands, dict):
        if "y_low" not in bands or "y_high" not in bands:
            raise ValueError("bands dict must contain 'y_low' and 'y_high'")
        return Band(x, bands["y_low"], bands["y_high"])
    if isinstance(bands, (list, tuple)) and len(bands) == 2:
        y_low, y_high = bands
        return Band(x, y_low, y_high)
    raise ValueError("bands must be a Band or a (y_low, y_high) tuple")


def _is_quantity_like(obj: Any) -> bool:
    return isinstance(obj, dict) and "unit" in obj and ("measure" in obj or "result" in obj)


def _quantity_axis_label(q: dict) -> Optional[str]:
    symbol = q.get("symbol")
    unit = q.get("unit")
    if symbol and unit:
        return f"{symbol} [{unit}]"
    if symbol:
        return str(symbol)
    if unit:
        return str(unit)
    return None

def plot(
    *objetos,
    layout: Optional[str] = None,
    dims: str = "2D",
    show: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    y_fit: Optional[np.ndarray] = None,
    yerr: Optional[np.ndarray] = None,
    bands: Optional[Any] = None,
    hist: Optional[Any] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> Union[Scene, Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]]:
    """
    Core engine and fast Scene constructor.
    
    INPUT:
        *objetos: Series, SeriesWithError, Histogram, Fit, Band, Panel, Scene
              or quantity dicts for x/y (constructor mode)
        layout: str | None
            None -> auto (1→1x1, 2→1x2, etc.)
            "2x3" -> force specific layout
            Ignored if Scene is passed (uses scene.layout)
        dims: str -> "2D" (default) or "3D"
            Ignored if Scene is passed (uses scene.dims)
        show: bool -> if True, calls plt.show()
        figsize: tuple | None -> figure size; if None uses PLOT_DEFAULTS
            Ignored if Scene is passed (uses scene.figsize)
        xlabel, ylabel, title: str | None -> global labels (if applicable)
            Ignored if Scene is passed (uses scene.xlabel, etc.)
        y_fit: array_like | Fit | quantity dict | None -> fitted curve (constructor mode)
        yerr: array_like | None -> symmetric y errors (constructor mode)
        bands: Band | (y_low, y_high) | dict | None -> band (constructor mode)
        hist: array_like | (data, bins) | Histogram | quantity dict | None -> histogram (constructor mode)
        ax: matplotlib.axes.Axes | None -> draw into a provided axis (single panel)
        **kwargs: style options (dpi, grid_alpha, lw, etc.)
            ALWAYS applied (even with Scene)
    
    OUTPUT:
        Scene when used as constructor (plot(x, y, ...) or plot(hist=...))
        (fig, ax) si n_objetos == 1
        (fig, axs) si n_objetos > 1
    
    EXAMPLES:
        # Fast constructor
        s1 = plot(x, y)
        s2 = plot(x, y, y_fit=yfit)
        s3 = plot(x, y, yerr=sy, bands=(y_low, y_high))
        s4 = plot(hist=data)
        s5 = plot(xq, yq)  # quantity dicts

        # With Scene (recommended for reuse)
        scene = Scene(Series(x, y), Histogram(data), layout="1x2")
        plot(scene)  # Scene structure
        plot(scene, dpi=300)  # same content, different style
    
    NOTES ABOUT Scene:
        - Scene must be passed as a SINGLE ARGUMENT: plot(scene)
        - Scene defines STRUCTURE (what objects, layout, dims, labels)
        - kwargs define AESTHETICS (dpi, colors, line widths, etc.)
        - layout/dims/figsize/labels parameters are ignored when Scene is provided
    
    PURPOSE:
        The user only declares what they have; this engine:
        - Builds a Scene when called with raw data
        - Plots existing Scene/Panel/semantic objects
        - Applies consistent styling
    """
    # Constructor mode: build a Scene from raw data
    return_scene = False
    constructor_mode = False
    if hist is not None or y_fit is not None or yerr is not None or bands is not None:
        constructor_mode = True
    elif len(objetos) == 2 and not _is_semantic_like(objetos[0]) and not _is_semantic_like(objetos[1]):
        if _looks_like_data_or_quantity(objetos[0]) and _looks_like_data_or_quantity(objetos[1]):
            constructor_mode = True

    if constructor_mode:
        show = False
        if hist is not None:
            if len(objetos) > 0 or y_fit is not None or yerr is not None or bands is not None:
                raise ValueError("Histogram must be plotted as its own Scene")
            if _is_quantity_like(hist):
                xlabel = xlabel or _quantity_axis_label(hist)
                hist = value_quantity(hist)[0]
            scene = Scene(_parse_histogram(hist))
        else:
            if len(objetos) != 2:
                raise ValueError("plot(x, y, ...) requires both x and y")
            x, y = objetos
            series_label = None
            if _is_quantity_like(x):
                x_value, _ = value_quantity(x)
                xlabel = xlabel or _quantity_axis_label(x)
                x = x_value
            if _is_quantity_like(y):
                y_value, y_sigma = value_quantity(y)
                ylabel = ylabel or _quantity_axis_label(y)
                series_label = y.get("symbol") or None
                y = y_value
                if yerr is None:
                    yerr = y_sigma
            if yerr is not None:
                series_obj = SeriesWithError(x, y, sy=yerr, label=series_label)
            else:
                series_obj = Series(x, y, label=series_label)
            objetos_scene = [series_obj]
            if y_fit is not None:
                if isinstance(y_fit, Fit) or _is_fit_like(y_fit):
                    objetos_scene.append(y_fit)
                elif _is_quantity_like(y_fit):
                    y_fit_value, _ = value_quantity(y_fit)
                    objetos_scene.append(Fit(x, y_fit_value))
                else:
                    objetos_scene.append(Fit(x, y_fit))
            if bands is not None:
                objetos_scene.append(_parse_band(x, bands))
            scene = Scene(*objetos_scene)
        if xlabel is not None:
            scene.xlabel = xlabel
        if ylabel is not None:
            scene.ylabel = ylabel
        if title is not None:
            scene.title = title
        objetos = (scene,)
        return_scene = True

    # Special case: if a single Scene is passed, use its properties
    scene_ref = None
    if len(objetos) == 1 and _is_scene_like(objetos[0]):
        scene = objetos[0]
        scene_ref = scene
        # Extract Scene properties (do not overwrite if explicitly provided)
        if layout is None:
            layout = scene.layout
        if dims == "2D":  # only if not explicitly passed
            dims = scene.dims
        if figsize is None:
            figsize = scene.figsize
        if xlabel is None:
            xlabel = scene.xlabel
        if ylabel is None:
            ylabel = scene.ylabel
        if title is None:
            title = scene.title
        # Use Scene panels directly
        objetos = tuple(scene.paneles)
    
    if len(objetos) == 0:
        raise ValueError("plot() needs at least one object (Series, Histogram, etc.)")

    if dims not in ("2D", "3D"):
        raise ValueError(f"dims must be '2D' or '3D', received: {dims}")

    cfg = _apply_style(**kwargs)

    if figsize is None:
        figsize = cfg["figsize"]

    # Normalize to Panels (each Panel corresponds to a subplot)
    paneles: List[Panel] = []
    for obj in objetos:
        if _is_panel_like(obj):
            paneles.append(obj)
        else:
            paneles.append(Panel(obj))
    if len(paneles) == 0:
        raise ValueError("plot() needs at least one object (Series, Histogram, etc.)")

    # Plot with a provided axis (single panel only)
    if ax is not None:
        if len(paneles) != 1:
            raise ValueError("ax can only be used with a single panel/Scene")
        if dims == "3D" and not _is_axis_3d(ax):
            raise TypeError("dims='3D' requires a 3D axis")
        if dims == "2D" and _is_axis_3d(ax):
            raise TypeError("dims='2D' cannot be drawn on a 3D axis")
        if scene_ref is not None:
            scene_ref._artist_map_by_id = {}
        for obj in paneles[0].objetos:
            artist = _plot_object(obj, ax, cfg)
            if scene_ref is not None:
                scene_ref._artist_map_by_id[id(obj)] = artist
        _post_process_axis(ax, cfg)
        _apply_legend(ax, cfg)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        if show:
            plt.show()
        if return_scene:
            if not show:
                plt.close(ax.figure)
            return scene
        return (ax.figure, ax)

    # Compute layout
    n_grupos = len(paneles)
    if layout is None:
        nrows, ncols = _layout_shape(n_grupos)
    else:
        nrows, ncols = _parse_layout(layout)

    # Architectural decision: multipanel uses manual control, single panel uses constrained_layout
    # constrained_layout and subplots_adjust are incompatible (Matplotlib ignores adjust if constrained=True)
    is_multipanel = n_grupos > 1
    use_constrained = cfg.get("tight", True) and not is_multipanel

    # Create figure
    fig = plt.figure(figsize=figsize, constrained_layout=use_constrained)

    # Create axes manually (2D or 3D depending on dims)
    axs = []
    for i in range(nrows * ncols):
        ax = _create_axis(fig, i + 1, nrows, ncols, dims)
        axs.append(ax)
    axs = np.array(axs)

    # For multipanel, apply visual composition with manual control
    if is_multipanel:
        fig.subplots_adjust(
            left=0.10,      # 10% left margin
            right=0.95,     # 5% right margin
            top=0.92,       # 8% top margin
            bottom=0.10,    # 10% bottom margin
            wspace=0.35,    # 35% relative horizontal spacing
            hspace=0.40     # 40% relative vertical spacing
        )

    # If a Scene was provided, keep a stable mapping from semantic objects to artists
    if scene_ref is not None:
        scene_ref._artist_map_by_id = {}

    # Plot each panel
    for idx, panel in enumerate(paneles):
        if idx < len(axs):
            ax = axs[idx]
            for obj in panel.objetos:
                artist = _plot_object(obj, ax, cfg)
                if scene_ref is not None:
                    scene_ref._artist_map_by_id[id(obj)] = artist
            _post_process_axis(ax, cfg)
            _apply_legend(ax, cfg)

    # Hide extra axes
    for idx in range(len(paneles), len(axs)):
        axs[idx].set_visible(False)

    # Global labels (if only one subplot)
    if n_grupos == 1:
        ax = axs[0]
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)

    if show:
        plt.show()

    # Return consistent format
    if return_scene:
        if not show:
            plt.close(fig)
        return scene
    if n_grupos == 1:
        return fig, axs[0]
    else:
        return fig, axs[:n_grupos]


def _plot_object(obj: Any, ax: plt.Axes, cfg: Dict[str, Any]):
    """
    Internal dispatcher: draw object on axis by type.
    
    INPUT:
        obj: Series, SeriesWithError, Histogram, Fit, Band, Series3D
        ax: matplotlib.axes.Axes or Axes3D
        cfg: configuration dictionary
    
    NOTES:
        - Does NOT modify obj
        - NO calcula nada
        - Solo dibuja lo que obj contiene
        - Valida compatibilidad 2D/3D
    """
    # Detectar si el eje es 3D (usando helper centralizado)
    is_3d_axis = _is_axis_3d(ax)
    
    # Validar compatibilidad 2D/3D
    if isinstance(obj, Series3D) or _is_series3d_like(obj):
        if not is_3d_axis:
            raise TypeError("Series3D requires dims='3D'. Use plot(..., dims='3D')")
        return _plot_series3d(obj, ax, cfg)
    else:
        # 2D objects (Series, SeriesWithError, etc.)
        if is_3d_axis:
            tipos_2d = "Series, SeriesWithError, Histogram, Fit, Band"
            raise TypeError(f"2D objects ({tipos_2d}) cannot be drawn on 3D axes. Use dims='2D' or Series3D")

        if isinstance(obj, SeriesWithError) or _is_series_with_error_like(obj):
            return _plot_series_with_error(obj, ax, cfg)
        elif isinstance(obj, Histogram) or _is_histogram_like(obj):
            return _plot_histogram(obj, ax, cfg)
        elif isinstance(obj, Fit) or _is_fit_like(obj):
            return _plot_fit(obj, ax, cfg)
        elif isinstance(obj, Band) or _is_band_like(obj):
            return _plot_band(obj, ax, cfg)
        elif isinstance(obj, Series) or _is_series_like(obj):
            return _plot_series(obj, ax, cfg)
        else:
            raise TypeError(f"Unsupported object type: {type(obj).__name__}")


def _plot_series(obj: Series, ax: plt.Axes, cfg: Dict[str, Any]):
    """
    Draw a simple series (clean scatter plot).
    """
    color = cfg["palette"]["data"]
    marker = obj.marker if obj.marker is not None else "o"
    
    if cfg.get("hollow_markers", True) and obj.marker is None:
        # Hollow circles only if marker is not specified
        artist = ax.scatter(
            obj.x, obj.y,
            label=obj.label,
            marker=marker,
            facecolors="none",
            edgecolors=color,
            linewidths=1.0,
            s=36
        )
        return artist
    else:
        # Specific marker or filled
        artist = ax.scatter(
            obj.x, obj.y,
            label=obj.label,
            marker=marker,
            color=color,
            s=36
        )
        return artist


def _plot_series_with_error(obj: SeriesWithError, ax: plt.Axes, cfg: Dict[str, Any]):
    """
    Draw a series with error bars (errorbar).
    """
    artist = ax.errorbar(
        obj.x, obj.y,
        xerr=obj.sx, yerr=obj.sy,
        fmt="o",
        label=obj.label,
        color=cfg["palette"]["data"],
        ecolor=cfg["palette"]["error"],
        capsize=cfg.get("capsize", 2.8),
        markerfacecolor="none" if cfg.get("hollow_markers", True) else None,
        markeredgecolor=cfg["palette"]["data"] if cfg.get("hollow_markers", True) else None,
    )
    return artist


def _plot_histogram(obj: Histogram, ax: plt.Axes, cfg: Dict[str, Any]):
    """
    Draw a clean histogram.
    """
    _, _, artist = ax.hist(
        obj.data,
        bins=obj.bins,
        label=obj.label,
        color=cfg["palette"]["data"],
        edgecolor=cfg["palette"]["ink"],
        alpha=0.7,
    )
    return artist


def _plot_fit(obj: Fit, ax: plt.Axes, cfg: Dict[str, Any]):
    """
    Draw a fitted curve (smooth line).
    """
    (artist,) = ax.plot(
        obj.x, obj.yfit,
        label=obj.label,
        color=cfg["palette"]["ink"],
        linewidth=cfg["lw"],
    )
    return artist


def _plot_band(obj: Band, ax: plt.Axes, cfg: Dict[str, Any]):
    """
    Draw a band (filled between y_low and y_high).
    """
    artist = ax.fill_between(
        obj.x,
        obj.y_low, obj.y_high,
        label=obj.label,
        color=cfg["palette"]["accent"],
        alpha=0.2,
        linewidth=0,
    )
    return artist


def _plot_series3d(obj: Series3D, ax, cfg: Dict[str, Any]):
    """
    Draw a 3D series (line or points in 3D space).
    
    INPUT:
        obj: Series3D
        ax: Axes3D (axis with projection="3d")
        cfg: configuration dictionary
    
    NOTES:
        - Uses ax.plot(x, y, z) from mpl_toolkits.mplot3d
        - Subdued color consistent with 2D palette
    """
    (artist,) = ax.plot(
        obj.x, obj.y, obj.z,
        label=obj.label,
        color=cfg["palette"]["ink"],
        linewidth=cfg["lw"],
        marker='o',
        markersize=cfg["ms"]
    )
    return artist


# ============================================================
#  CLASSIC API (compatibility with legacy code)
# ============================================================

def figure(
    figsize: Optional[Tuple[float, float]] = None,
    nrows: int = 1,
    ncols: int = 1,
    sharex: bool = False,
    sharey: bool = False,
    **kwargs
):
    """
    Create figure + axes with the applied style.
    Returns (fig, ax) or (fig, axs) if there are multiple subplots.

    kwargs: you can pass keys from PLOT_DEFAULTS (dpi, grid_alpha, lw, etc.)
    """
    cfg = _apply_style(**kwargs)

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
    
    # Adjust visual composition for multipanel (margins and scientific spacing)
    # Apply the same logic as plot(): multipanel disables constrained_layout
    if nrows > 1 or ncols > 1:
        fig.subplots_adjust(
            left=0.10,      # 10% left margin
            right=0.95,     # 5% right margin
            top=0.92,       # 8% top margin
            bottom=0.10,    # 10% bottom margin
            wspace=0.35,    # 35% relative horizontal spacing
            hspace=0.40     # 40% relative vertical spacing
        )

    return fig, axs


def save(
    fig,
    filename: Union[str, Path],
    formatos: Optional[List[str]] = None,
    transparent: Optional[bool] = None,
    close: bool = True
):
    """
    Save figure in one or more formats (PDF recommended for LaTeX).
    
    INPUT:
        fig: matplotlib.figure.Figure
        filename: str | Path -> path without extension
        formatos: list[str] | None -> ["pdf", "png"]; None uses PLOT_DEFAULTS
        transparent: bool | None -> transparent background
        close: bool -> close figure after saving
    
    OUTPUT:
        None (side effect: saves file(s))
    """
    if formatos is None:
        formatos = PLOT_DEFAULTS["save_formats"]
    if transparent is None:
        transparent = PLOT_DEFAULTS["transparent"]
    
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    for fmt in formatos:
        out = filename.with_suffix("." + fmt)
        fig.savefig(out, bbox_inches="tight", transparent=transparent)
    
    if close:
        plt.close(fig)


def line(
    x, y,
    *,
    ax=None,
    label=None,
    xlabel=None,
    ylabel=None,
    title=None,
    xlim=None,
    ylim=None,
    marker=None,
    legend=None,
    **kwargs
):
    """
    Classic x‑y line (traditional API).
    
    Uses the Series class internally to keep consistency.
    """
    cfg = _apply_style(**kwargs)
    if ax is None:
        fig, ax = plt.subplots(figsize=cfg["figsize"], constrained_layout=cfg.get("tight", True))
    
    kwargs_plot = {k: v for k, v in kwargs.items() if k not in PLOT_DEFAULTS}
    kwargs_plot.setdefault("color", cfg["palette"]["ink"])
    
    ax.plot(x, y, marker=marker, label=label, **kwargs_plot)
    
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
    
    _post_process_axis(ax, cfg)
    
    if legend is None:
        legend = cfg.get("legend", True)
    if legend:
        _apply_legend(ax, cfg)
    
    return ax


def scatter(
    x, y,
    *,
    ax=None,
    label=None,
    xlabel=None,
    ylabel=None,
    title=None,
    xlim=None,
    ylim=None,
    legend=None,
    **kwargs
):
    """
    Scatter limpio (API tradicional).
    """
    cfg = _apply_style(**kwargs)
    if ax is None:
        fig, ax = plt.subplots(figsize=cfg["figsize"], constrained_layout=cfg.get("tight", True))
    
    kwargs_sc = {k: v for k, v in kwargs.items() if k not in PLOT_DEFAULTS}
    kwargs_sc.setdefault("color", cfg["palette"]["data"])
    
    if cfg.get("hollow_markers", True):
        kwargs_sc["facecolors"] = "none"
        kwargs_sc["edgecolors"] = cfg["palette"]["data"]
        kwargs_sc["linewidths"] = 1.0
        s = kwargs_sc.pop("s", 36)
        ax.scatter(x, y, label=label, s=s, **kwargs_sc)
    else:
        ax.scatter(x, y, label=label, **kwargs_sc)
    
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
    
    _post_process_axis(ax, cfg)
    
    if legend is None:
        legend = cfg.get("legend", True)
    if legend:
        _apply_legend(ax, cfg)
    
    return ax


def errorbar(
    x, y,
    sx=None, sy=None,
    *,
    ax=None,
    label=None,
    xlabel=None,
    ylabel=None,
    title=None,
    xlim=None,
    ylim=None,
    marker="o",
    legend=None,
    **kwargs
):
    """
    Puntos con barras de error (API tradicional).
    
    Uses SeriesWithError internally.
    """
    cfg = _apply_style(**kwargs)
    if ax is None:
        fig, ax = plt.subplots(figsize=cfg["figsize"], constrained_layout=cfg.get("tight", True))
    
    kwargs_eb = {k: v for k, v in kwargs.items() if k not in PLOT_DEFAULTS}
    kwargs_eb.setdefault("color", cfg["palette"]["data"])
    kwargs_eb.setdefault("ecolor", cfg["palette"]["error"])
    kwargs_eb.setdefault("capsize", cfg.get("capsize", 2.8))
    
    ax.errorbar(
        x, y,
        xerr=sx, yerr=sy,
        fmt=marker,
        label=label,
        markerfacecolor="none" if cfg.get("hollow_markers", True) else None,
        markeredgecolor=cfg["palette"]["data"] if cfg.get("hollow_markers", True) else None,
        **kwargs_eb
    )
    
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
    
    _post_process_axis(ax, cfg)
    
    if legend is None:
        legend = cfg.get("legend", True)
    if legend:
        _apply_legend(ax, cfg)
    
    return ax


# ============================================================
#  SCENE COMPOSITION (FIGURE OF SCENES)
# ============================================================

class Figure:
    """
    Lightweight container for composing multiple Scene objects.

    RULES:
        - One Scene corresponds to one axis
        - Composition happens only at this level
        - Each Scene must contain exactly one Panel

    EXAMPLE:
        s1 = plot(x, y)
        s2 = plot(hist=data)
        fig = Figure(s1, s2)
        fig.show()
    """

    def __init__(self, *scenes: Scene):
        if len(scenes) == 0:
            raise ValueError("Figure requires at least one Scene")
        for scene in scenes:
            if not _is_scene_like(scene):
                raise TypeError("Figure only accepts Scene objects")
            if len(scene.paneles) != 1:
                raise ValueError("Each Scene in Figure must contain exactly one Panel")
        self.scenes = list(scenes)

    def show(
        self,
        *,
        layout: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
        **kwargs
    ) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
        cfg = _apply_style(**kwargs)
        if figsize is None:
            figsize = cfg["figsize"]

        n_scenes = len(self.scenes)
        if layout is None:
            nrows, ncols = _layout_shape(n_scenes)
        else:
            nrows, ncols = _parse_layout(layout)

        is_multipanel = n_scenes > 1
        use_constrained = cfg.get("tight", True) and not is_multipanel
        fig = plt.figure(figsize=figsize, constrained_layout=use_constrained)

        axs = []
        for i in range(nrows * ncols):
            if i < n_scenes:
                dims = self.scenes[i].dims
                ax = _create_axis(fig, i + 1, nrows, ncols, dims)
                plot(self.scenes[i], show=False, ax=ax, **kwargs)
            else:
                ax = _create_axis(fig, i + 1, nrows, ncols, "2D")
                ax.set_visible(False)
            axs.append(ax)
        axs = np.array(axs)

        if is_multipanel:
            fig.subplots_adjust(
                left=0.10,
                right=0.95,
                top=0.92,
                bottom=0.10,
                wspace=0.35,
                hspace=0.40
            )

        if show:
            plt.show()

        if n_scenes == 1:
            return fig, axs[0]
        return fig, axs[:n_scenes]


# ============================================================
#  FACADE (SAME STYLE AS YOUR OTHER MODULES)
# ============================================================

class _Graphics:
    """Facade: graphics.plot(...), graphics.line(...), graphics.errorbar(...), etc."""
    
    # Core engine (NEW - Layer 2)
    plot = staticmethod(plot)
    
    # Semantic classes (NEW - Layer 1)
    Series = Series
    SeriesWithError = SeriesWithError
    Histogram = Histogram
    Fit = Fit
    Band = Band
    Series3D = Series3D
    Panel = Panel
    Scene = Scene
    Figure = Figure
    
    # Utilities and configuration
    _apply_style = staticmethod(_apply_style)
    figure = staticmethod(figure)
    save = staticmethod(save)
    
    # Classic API (compatibility with legacy code)
    line = staticmethod(line)
    scatter = staticmethod(scatter)
    errorbar = staticmethod(errorbar)


# Global instance
graphics = _Graphics()
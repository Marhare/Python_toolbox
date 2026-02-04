"""
graficos.py
============================================================
Módulo de visualización científica de ALTO NIVEL y BAJA FRICCIÓN.

FILOSOFÍA:
- El usuario declara INTENCIÓN (qué tiene: datos, errores, ajustes, bandas)
- El módulo decide automáticamente CÓMO dibujarlo

NO CALCULA NADA. Solo grafica resultados ya calculados.

DISEÑO POR CAPAS:
  1. Objetos semánticos (Serie, SerieConError, Histograma, Ajuste, Banda, Serie3D, Panel, Scene)
  2. Motor central: plot(*objetos, layout=None, dims="2D", show=True)
  3. Layout automático
  4. Extensibilidad futura (animaciones)

PREPARACIÓN PARA ANIMACIONES:
  - Scene encapsula estructura gráfica completa (paneles + metadatos)
  - Motor temporal futuro: animate(scene, updater, frames) [NO implementado]
  - Scene es la unidad natural para animaciones: define QUÉ se dibuja
  - updater define CÓMO evoluciona la escena en el tiempo
  - Separación limpia: graficos.py (estático) + animaciones.py (temporal)

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


# ============================================================
#  CONFIG GLOBAL
# ============================================================

PLOT_DEFAULTS = {
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
#  CAPA 1: OBJETOS SEMÁNTICOS (NO cálculos, NO gráficos)
# ============================================================

@dataclass
class Serie:
    """
    Representa una serie de datos simple: x, y.
    
    INPUT:
        x: array_like -> variable independiente
        y: array_like -> variable dependiente
        label: str | None -> etiqueta para leyenda
        marker: str | None -> tipo de punto ('o', 's', '^', 'D', etc.) o None para hollow circles
    
    PROPÓSITO:
        Almacenar intención "quiero graficar estos puntos"
        Sin cálculos ni decoraciones.
    """
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    label: Optional[str] = None
    marker: Optional[str] = None

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("Serie: 'x' e 'y' deben tener la misma longitud")
    
    def __hash__(self):
        return id(self)


@dataclass
class SerieConError:
    """
    Serie con barras de error (simétricas).
    
    INPUT:
        x: array_like -> variable independiente
        y: array_like -> variable dependiente
        sy: array_like | None -> error en y (opcional)
        sx: array_like | None -> error en x (opcional)
        label: str | None -> etiqueta para leyenda
    
    PROPÓSITO:
        "Tengo puntos experimentales con incertidumbres"
        Motor dibuja automáticamente errorbar.
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
            raise ValueError("SerieConError: 'x' e 'y' deben tener la misma longitud")
        if self.sy is not None and self.sy.shape[0] != self.y.shape[0]:
            raise ValueError("SerieConError: 'sy' debe tener la misma longitud que 'y'")
        if self.sx is not None and self.sx.shape[0] != self.x.shape[0]:
            raise ValueError("SerieConError: 'sx' debe tener la misma longitud que 'x'")
    
    def __hash__(self):
        return id(self)


@dataclass
class Histograma:
    """
    Histograma de una variable.
    
    INPUT:
        data: array_like -> datos brutos
        bins: int | array_like -> número de bins o bordes (Matplotlib standard)
        label: str | None -> etiqueta
    
    PROPÓSITO:
        "Quiero ver la distribución de estos datos"
        Motor dibuja automáticamente con estilo limpio.
    """
    data: np.ndarray = field(default_factory=lambda: np.array([]))
    bins: Union[int, np.ndarray] = 30
    label: Optional[str] = None

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=float)
    
    def __hash__(self):
        return id(self)


@dataclass
class Ajuste:
    """
    Curva ajustada (ya calculada en otro módulo).
    
    INPUT:
        x: array_like -> puntos donde evaluar el ajuste
        yfit: array_like -> valores del ajuste
        label: str | None -> etiqueta (ej: "Ajuste lineal", "Spline")
    
    PROPÓSITO:
        "Aquí está la curva de ajuste, dibújala sobre los datos"
        No calcula nada, solo dibuja la línea.
    """
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    yfit: np.ndarray = field(default_factory=lambda: np.array([]))
    label: Optional[str] = None

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.yfit = np.asarray(self.yfit, dtype=float)
        if self.x.shape[0] != self.yfit.shape[0]:
            raise ValueError("Ajuste: 'x' e 'yfit' deben tener la misma longitud")
    
    def __hash__(self):
        return id(self)


@dataclass
class Banda:
    """
    Banda de confianza/predicción (ya calculada).
    
    INPUT:
        x: array_like -> puntos x
        y_low: array_like -> límite inferior de la banda
        y_high: array_like -> límite superior de la banda
        label: str | None -> etiqueta (ej: "Banda 95%")
    
    PROPÓSITO:
        "Dibuja esta región sombreada entre y_low e y_high"
        Típicamente se usa junto con Ajuste.
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
            raise ValueError("Banda: 'x', 'y_low' e 'y_high' deben tener la misma longitud")
    
    def __hash__(self):
        return id(self)


@dataclass
class Serie3D:
    """
    Representa una serie de datos en 3D: x, y, z.
    
    INPUT:
        x: array_like -> variable independiente 1
        y: array_like -> variable independiente 2
        z: array_like -> variable dependiente
        label: str | None -> etiqueta para leyenda
    
    PROPÓSITO:
        "Dibuja esta trayectoria o nube de puntos en 3D"
        NO calcula nada.
        Requiere dims="3D" en plot().
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
            raise ValueError("Serie3D: 'x', 'y' y 'z' deben tener la misma longitud")
    
    def __hash__(self):
        return id(self)


@dataclass
class Panel:
    """
    Panel de composición: agrupa objetos que deben dibujarse en el mismo eje.

    INPUT:
        *objetos: objetos semánticos (Serie, SerieConError, Histograma, Ajuste, Banda, etc.)

    PROPÓSITO:
        Explicitar qué debe ir superpuesto en un mismo subplot.
        No calcula ni dibuja; solo almacena la intención.
    """
    objetos: List[Any] = field(default_factory=list)

    def __init__(self, *objetos: Any):
        self.objetos = list(objetos)


@dataclass
class Scene:
    """
    Escena gráfica completa: representa una figura reutilizable.
    
    INPUT:
        *paneles: Panel o objetos semánticos (se normalizan a Panel)
        layout: str | None -> layout explícito ("2x3") o None (automático)
        dims: str -> "2D" o "3D"
        figsize: tuple | None -> tamaño de figura
        xlabel, ylabel, title: str | None -> etiquetas globales
    
    PROPÓSITO:
        Encapsular ESTRUCTURA gráfica (qué objetos, cómo organizarlos).
        NO encapsula ESTÉTICA (los kwargs de estilo se pasan a plot()).
        NO calcula ni dibuja nada.
        Base natural para futuras animaciones.
    
    DECISIONES DE DISEÑO:
        - Scene es MUTABLE por diseño (permite modificar paneles después de creación)
        - Validación temprana: verifica compatibilidad dims-objetos en construcción
        - Debe pasarse como ARGUMENTO ÚNICO a plot(): plot(scene)
        - Estilo se controla externamente: plot(scene, lw=2, dpi=200)
        
    EJEMPLOS:
        scene = Scene(Serie(x, y), Histograma(data))
        scene = Scene(Panel(Serie(...), Ajuste(...)), layout="1x2", dims="2D")
        plot(scene)  # dibuja la escena con estilo por defecto
        plot(scene, dpi=300)  # mismo contenido, distinto estilo
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
        # Normalizar objetos sueltos a Panel (igual que plot())
        self.paneles = []
        for obj in paneles:
            if isinstance(obj, Panel):
                self.paneles.append(obj)
            else:
                self.paneles.append(Panel(obj))
        
        # Validación temprana: compatibilidad dims-objetos
        if dims == "3D":
            for panel in self.paneles:
                for obj in panel.objetos:
                    if not isinstance(obj, Serie3D):
                        raise TypeError(
                            f"Scene con dims='3D' solo acepta objetos Serie3D. "
                            f"Encontrado: {type(obj).__name__}"
                        )
        elif dims == "2D":
            for panel in self.paneles:
                for obj in panel.objetos:
                    if isinstance(obj, Serie3D):
                        raise TypeError(
                            f"Scene con dims='2D' no acepta objetos Serie3D. "
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

def _aplicar_estilo(**overrides):
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
    Decide automáticamente forma de grid (nrows, ncols) para n gráficos.
    
    INPUT:
        n: int -> número de subplots
    
    OUTPUT:
        (nrows, ncols)
    
    REGLA:
        1  -> 1x1
        2  -> 1x2
        3  -> 2x2  (deja 1 vacío)
        4  -> 2x2
        5  -> 2x3  (deja 1 vacío)
        6  -> 2x3
        7  -> 3x3  (deja 2 vacíos)
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
        # para n >= 7: usa ceil(sqrt(n))
        side = int(np.ceil(np.sqrt(n)))
        return side, side


def _crear_eje(fig, idx: int, nrows: int, ncols: int, dims: str) -> plt.Axes:
    """
    Crea un eje 2D o 3D según dims.
    
    INPUT:
        fig: figura matplotlib
        idx: índice del subplot (1-based)
        nrows, ncols: layout del subplot
        dims: "2D" o "3D"
    
    OUTPUT:
        ax: matplotlib Axes (2D) o Axes3D (3D)
    
    NOTAS:
        - Si dims="3D", crea con projection="3d"
        - Si dims="2D", crea eje normal
    """
    if dims == "3D":
        return fig.add_subplot(nrows, ncols, idx, projection="3d")
    else:
        return fig.add_subplot(nrows, ncols, idx)


def _es_eje_3d(ax) -> bool:
    """
    Detecta si un eje es 3D.
    
    INPUT:
        ax: matplotlib Axes
    
    OUTPUT:
        bool: True si el eje tiene proyección 3D
    
    NOTAS:
        Centraliza la detección para evitar inconsistencias.
        Los ejes 3D tienen atributo 'zaxis'.
    """
    return hasattr(ax, 'zaxis')


def _post_process_ax(ax, cfg):
    """
    Aplica post-processing estándar a un eje.
    Usado internamente después de cada gráfico.
    
    NOTAS:
        - Para ejes 3D, omite minor ticks y spines (no aplicables)
        - Para ejes 2D, aplica configuración estándar
    """
    # Ejes 3D: no aplicar lógica 2D
    if _es_eje_3d(ax):
        return
    
    # Minor ticks (solo 2D)
    if cfg.get("minor_ticks", True):
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    # Spines visibility (solo 2D)
    sp = set(cfg.get("spines", ("left", "bottom")))
    ax.spines["left"].set_visible("left" in sp)
    ax.spines["bottom"].set_visible("bottom" in sp)
    ax.spines["top"].set_visible("top" in sp)
    ax.spines["right"].set_visible("right" in sp)


def _apply_legend(ax, cfg):
    """
    Aplica leyenda solo si hay labels válidos.
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

def _es_scene_like(obj: Any) -> bool:
    return isinstance(obj, Scene) or (
        hasattr(obj, "paneles") and hasattr(obj, "layout") and hasattr(obj, "dims")
    )


def _es_panel_like(obj: Any) -> bool:
    return isinstance(obj, Panel) or hasattr(obj, "objetos")


def _es_serie3d_like(obj: Any) -> bool:
    return hasattr(obj, "x") and hasattr(obj, "y") and hasattr(obj, "z")


def _es_serie_con_error_like(obj: Any) -> bool:
    return hasattr(obj, "x") and hasattr(obj, "y") and (hasattr(obj, "sy") or hasattr(obj, "sx"))


def _es_histograma_like(obj: Any) -> bool:
    return hasattr(obj, "data") and hasattr(obj, "bins")


def _es_ajuste_like(obj: Any) -> bool:
    return hasattr(obj, "x") and hasattr(obj, "yfit")


def _es_banda_like(obj: Any) -> bool:
    return hasattr(obj, "x") and hasattr(obj, "y_low") and hasattr(obj, "y_high")


def _es_serie_like(obj: Any) -> bool:
    return hasattr(obj, "x") and hasattr(obj, "y")

def plot(
    *objetos,
    layout: Optional[str] = None,
    dims: str = "2D",
    show: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    **kwargs
) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
    """
    Motor central: grafica automáticamente una o más series.
    
    INPUT:
        *objetos: Serie, SerieConError, Histograma, Ajuste, Banda, Panel, Scene
        layout: str | None
                None -> calcula automáticamente (1→1x1, 2→1x2, etc.)
                "2x3" -> fuerza layout específico
                Ignorado si se pasa Scene (usa scene.layout)
        dims: str -> "2D" (defecto) o "3D"
                Ignorado si se pasa Scene (usa scene.dims)
        show: bool -> si True, ejecuta plt.show()
        figsize: tuple | None -> tamaño figura; si None usa PLOT_DEFAULTS
                Ignorado si se pasa Scene (usa scene.figsize)
        xlabel, ylabel, title: str | None -> etiquetas globales (si aplica)
                Ignorado si se pasa Scene (usa scene.xlabel, etc.)
        **kwargs: opciones estilo (dpi, grid_alpha, lw, etc.)
                Aplicables SIEMPRE (incluso con Scene)
    
    OUTPUT:
        (fig, ax) si n_objetos == 1
        (fig, axs) si n_objetos > 1
    
    EJEMPLOS:
        # Modo directo
        plot(Serie(x, y))
        plot(SerieConError(x, y, sy), Ajuste(x, yfit))
        plot(Histograma(data1), Histograma(data2), layout="1x2")
        
        # Con Scene (recomendado para reutilización)
        scene = Scene(Serie(x, y), Histograma(data), layout="1x2")
        plot(scene)  # estructura de Scene
        plot(scene, dpi=300)  # mismo contenido, distinto estilo
    
    NOTAS SOBRE Scene:
        - Scene debe pasarse como ARGUMENTO ÚNICO: plot(scene)
        - Scene define ESTRUCTURA (qué objetos, layout, dims, labels)
        - kwargs definen ESTÉTICA (dpi, colores, grosores, etc.)
        - Parámetros layout/dims/figsize/labels se ignoran si viene Scene
    
    PROPÓSITO:
        Usuario solo declara qué tiene, este motor:
        - Detecta número y tipos de objetos
        - Crea figura y ejes automáticamente
        - Llama funciones internas según tipo
        - Aplica estilos coherentes
        - Minimiza decisiones del usuario
    """
    # Caso especial: si se pasa un único Scene, usar sus propiedades
    if len(objetos) == 1 and _es_scene_like(objetos[0]):
        scene = objetos[0]
        # Extraer propiedades de Scene (no sobreescribir si ya vienen en kwargs)
        if layout is None:
            layout = scene.layout
        if dims == "2D":  # solo si no se pasó explícitamente
            dims = scene.dims
        if figsize is None:
            figsize = scene.figsize
        if xlabel is None:
            xlabel = scene.xlabel
        if ylabel is None:
            ylabel = scene.ylabel
        if title is None:
            title = scene.title
        # Usar paneles de Scene directamente
        objetos = tuple(scene.paneles)
    
    if len(objetos) == 0:
        raise ValueError("plot() necesita al menos un objeto (Serie, Histograma, etc.)")

    if dims not in ("2D", "3D"):
        raise ValueError(f"dims debe ser '2D' o '3D', recibido: {dims}")

    cfg = _aplicar_estilo(**kwargs)

    if figsize is None:
        figsize = cfg["figsize"]

    # Normalizar a Paneles (cada Panel corresponde a un subplot)
    paneles: List[Panel] = []
    for obj in objetos:
        if _es_panel_like(obj):
            paneles.append(obj)
        else:
            paneles.append(Panel(obj))
    if len(paneles) == 0:
        raise ValueError("plot() necesita al menos un objeto (Serie, Histograma, etc.)")

    # Calcular layout
    n_grupos = len(paneles)
    if layout is None:
        nrows, ncols = _layout_shape(n_grupos)
    else:
        # parsear "2x3" -> (2, 3)
        parts = layout.split("x")
        if len(parts) != 2:
            raise ValueError(f"layout debe ser 'NxM' (ej: '2x3'), recibido: {layout}")
        try:
            nrows, ncols = int(parts[0]), int(parts[1])
        except ValueError:
            raise ValueError(f"layout debe ser 'NxM' con N,M enteros, recibido: {layout}")

    # Decisión arquitectural: multipanel usa control manual, monoPanel usa constrained_layout
    # constrained_layout y subplots_adjust son incompatibles (Matplotlib ignora adjust si constrained=True)
    is_multipanel = n_grupos > 1
    use_constrained = cfg.get("tight", True) and not is_multipanel
    
    # Crear figura
    fig = plt.figure(figsize=figsize, constrained_layout=use_constrained)

    # Crear ejes manualmente (2D o 3D según dims)
    axs = []
    for i in range(nrows * ncols):
        ax = _crear_eje(fig, i + 1, nrows, ncols, dims)
        axs.append(ax)
    axs = np.array(axs)
    
    # Para multipanel, aplicar composición visual con control manual
    if is_multipanel:
        fig.subplots_adjust(
            left=0.10,      # 10% margen izquierdo
            right=0.95,     # 5% margen derecho
            top=0.92,       # 8% margen superior
            bottom=0.10,    # 10% margen inferior
            wspace=0.35,    # 35% espaciado horizontal relativo
            hspace=0.40     # 40% espaciado vertical relativo
        )

    # Graficar cada panel
    for idx, panel in enumerate(paneles):
        if idx < len(axs):
            ax = axs[idx]
            for obj in panel.objetos:
                _plot_objeto(obj, ax, cfg)
            _post_process_ax(ax, cfg)
            _apply_legend(ax, cfg)

    # Ocultar ejes sobrantes
    for idx in range(len(paneles), len(axs)):
        axs[idx].set_visible(False)

    # Etiquetas globales (si hay un solo subplot)
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

    # Devolver formato consistente
    if n_grupos == 1:
        return fig, axs[0]
    else:
        return fig, axs[:n_grupos]


def _plot_objeto(obj: Any, ax: plt.Axes, cfg: Dict[str, Any]):
    """
    Dispatcher interno: dibuja objeto en eje según su tipo.
    
    INPUT:
        obj: Serie, SerieConError, Histograma, Ajuste, Banda, Serie3D
        ax: matplotlib.axes.Axes o Axes3D
        cfg: diccionario de configuración
    
    NOTAS:
        - NO modifica obj
        - NO calcula nada
        - Solo dibuja lo que obj contiene
        - Valida compatibilidad 2D/3D
    """
    # Detectar si el eje es 3D (usando helper centralizado)
    is_3d_axis = _es_eje_3d(ax)
    
    # Validar compatibilidad 2D/3D
    if isinstance(obj, Serie3D) or _es_serie3d_like(obj):
        if not is_3d_axis:
            raise TypeError("Serie3D requiere dims='3D'. Usa plot(..., dims='3D')")
        _plot_serie3d(obj, ax, cfg)
    else:
        # Objetos 2D (Serie, SerieConError, etc.)
        if is_3d_axis:
            tipos_2d = "Serie, SerieConError, Histograma, Ajuste, Banda"
            raise TypeError(f"Objetos 2D ({tipos_2d}) no pueden dibujarse en ejes 3D. Usa dims='2D' o Serie3D")

        if isinstance(obj, SerieConError) or _es_serie_con_error_like(obj):
            _plot_serie_con_error(obj, ax, cfg)
        elif isinstance(obj, Histograma) or _es_histograma_like(obj):
            _plot_histograma(obj, ax, cfg)
        elif isinstance(obj, Ajuste) or _es_ajuste_like(obj):
            _plot_ajuste(obj, ax, cfg)
        elif isinstance(obj, Banda) or _es_banda_like(obj):
            _plot_banda(obj, ax, cfg)
        elif isinstance(obj, Serie) or _es_serie_like(obj):
            _plot_serie(obj, ax, cfg)
        else:
            raise TypeError(f"Tipo de objeto no soportado: {type(obj).__name__}")


def _plot_serie(obj: Serie, ax: plt.Axes, cfg: Dict[str, Any]):
    """
    Dibuja una serie simple (scatter plot limpio).
    """
    color = cfg["palette"]["data"]
    marker = obj.marker if obj.marker is not None else "o"
    
    if cfg.get("hollow_markers", True) and obj.marker is None:
        # Hollow circles solo si no se especifica marker
        ax.scatter(
            obj.x, obj.y,
            label=obj.label,
            marker=marker,
            facecolors="none",
            edgecolors=color,
            linewidths=1.0,
            s=36
        )
    else:
        # Marker específico o filled
        ax.scatter(
            obj.x, obj.y,
            label=obj.label,
            marker=marker,
            color=color,
            s=36
        )


def _plot_serie_con_error(obj: SerieConError, ax: plt.Axes, cfg: Dict[str, Any]):
    """
    Dibuja serie con barras de error (errorbar).
    """
    ax.errorbar(
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


def _plot_histograma(obj: Histograma, ax: plt.Axes, cfg: Dict[str, Any]):
    """
    Dibuja histograma limpio.
    """
    ax.hist(
        obj.data,
        bins=obj.bins,
        label=obj.label,
        color=cfg["palette"]["data"],
        edgecolor=cfg["palette"]["ink"],
        alpha=0.7,
    )


def _plot_ajuste(obj: Ajuste, ax: plt.Axes, cfg: Dict[str, Any]):
    """
    Dibuja curva de ajuste (línea suave).
    """
    ax.plot(
        obj.x, obj.yfit,
        label=obj.label,
        color=cfg["palette"]["ink"],
        linewidth=cfg["lw"],
    )


def _plot_banda(obj: Banda, ax: plt.Axes, cfg: Dict[str, Any]):
    """
    Dibuja banda (relleno entre y_low e y_high).
    """
    ax.fill_between(
        obj.x,
        obj.y_low, obj.y_high,
        label=obj.label,
        color=cfg["palette"]["accent"],
        alpha=0.2,
        linewidth=0,
    )


def _plot_serie3d(obj: Serie3D, ax, cfg: Dict[str, Any]):
    """
    Dibuja una serie 3D (línea o puntos en espacio 3D).
    
    INPUT:
        obj: Serie3D
        ax: Axes3D (eje con projection="3d")
        cfg: diccionario de configuración
    
    NOTAS:
        - Usa ax.plot(x, y, z) de mpl_toolkits.mplot3d
        - Color sobrio coherente con paleta 2D
    """
    ax.plot(
        obj.x, obj.y, obj.z,
        label=obj.label,
        color=cfg["palette"]["ink"],
        linewidth=cfg["lw"],
        marker='o',
        markersize=cfg["ms"]
    )


# ============================================================
#  API CLÁSICA (compatibilidad con código antiguo)
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

    kwargs: puedes pasar claves de PLOT_DEFAULTS (dpi, grid_alpha, lw, etc.)
    """
    cfg = _aplicar_estilo(**kwargs)

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
    
    # Ajustar composición visual para multipanel (márgenes y espaciado científico)
    # Aplicar la misma lógica que en plot(): multipanel desactiva constrained_layout
    if nrows > 1 or ncols > 1:
        fig.subplots_adjust(
            left=0.10,      # 10% margen izquierdo
            right=0.95,     # 5% margen derecho
            top=0.92,       # 8% margen superior
            bottom=0.10,    # 10% margen inferior
            wspace=0.35,    # 35% espaciado horizontal relativo
            hspace=0.40     # 40% espaciado vertical relativo
        )

    return fig, axs


def guardar(
    fig,
    filename: Union[str, Path],
    formatos: Optional[List[str]] = None,
    transparent: Optional[bool] = None,
    close: bool = True
):
    """
    Guarda figura en uno o varios formatos (PDF recomendado para LaTeX).
    
    INPUT:
        fig: matplotlib.figure.Figure
        filename: str | Path -> ruta sin extensión
        formatos: list[str] | None -> ["pdf", "png"]; None usa PLOT_DEFAULTS
        transparent: bool | None -> fondo transparente
        close: bool -> cerrar figura después de guardar
    
    OUTPUT:
        None (efecto: guarda archivo(s))
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


def linea(
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
    Línea clásica x-y (API tradicional).
    
    Usa la clase Serie internamente para mantener consistencia.
    """
    cfg = _aplicar_estilo(**kwargs)
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
    
    _post_process_ax(ax, cfg)
    
    if legend is None:
        legend = cfg.get("legend", True)
    if legend:
        _apply_legend(ax, cfg)
    
    return ax


def dispersion(
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
    cfg = _aplicar_estilo(**kwargs)
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
    
    _post_process_ax(ax, cfg)
    
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
    
    Usa SerieConError internamente.
    """
    cfg = _aplicar_estilo(**kwargs)
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
    
    _post_process_ax(ax, cfg)
    
    if legend is None:
        legend = cfg.get("legend", True)
    if legend:
        _apply_legend(ax, cfg)
    
    return ax


# ============================================================
#  FACHADA (MISMO ESTILO QUE TUS OTROS MÓDULOS)
# ============================================================

class _Graficos:
    """Fachada: graficos.plot(...), graficos.linea(...), graficos.errorbar(...), etc."""
    
    # Motor central (NUEVO - Capa 2)
    plot = staticmethod(plot)
    
    # Clases semánticas (NUEVO - Capa 1)
    Serie = Serie
    SerieConError = SerieConError
    Histograma = Histograma
    Ajuste = Ajuste
    Banda = Banda
    Serie3D = Serie3D
    Panel = Panel
    Scene = Scene
    
    # Utilidades y configuración
    _aplicar_estilo = staticmethod(_aplicar_estilo)
    figura = staticmethod(figura)
    guardar = staticmethod(guardar)
    
    # API clásica (compatibilidad con código antiguo)
    linea = staticmethod(linea)
    dispersion = staticmethod(dispersion)
    errorbar = staticmethod(errorbar)


# Instancia global
graficos = _Graficos()
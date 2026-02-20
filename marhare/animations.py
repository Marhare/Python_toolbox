"""
animaciones.py
============================================================
Declarative time engine for graficos.py

PHILOSOPHY:
- Scene defines WHAT is drawn (static structure)
- evolve defines HOW it evolves in time (time laws)
- animate maps physical time t -> frames and updates artists

DOES NOT COMPUTE DATA. It only animates already computed results.

STRICT SEPARATION:
- graficos.py: static engine (plot, Scene, semantic objects)
- animaciones.py: time engine (animate)

============================================================
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

try:
    from .graphics import graphics, Scene, Series, Series3D, Band, Fit
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from graphics import graphics, Scene, Series, Series3D, Band, Fit


# ============================================================
#  MATCHING HELPERS BETWEEN OBJECTS AND ARTISTS
# ============================================================

def _es_serie_like(obj) -> bool:
    return hasattr(obj, "x") and hasattr(obj, "y") and not hasattr(obj, "yfit")


def _es_ajuste_like(obj) -> bool:
    return hasattr(obj, "x") and hasattr(obj, "yfit")


def _es_banda_like(obj) -> bool:
    return hasattr(obj, "x") and hasattr(obj, "y_low") and hasattr(obj, "y_high")


def _es_serie3d_like(obj) -> bool:
    return hasattr(obj, "x") and hasattr(obj, "y") and hasattr(obj, "z")

def _es_serie_con_error_like(obj) -> bool:
    return hasattr(obj, "x") and hasattr(obj, "y") and (hasattr(obj, "sy") or hasattr(obj, "sx"))

def _build_band_verts(x: np.ndarray, y_low: np.ndarray, y_high: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    y_low = np.asarray(y_low)
    y_high = np.asarray(y_high)
    upper = np.column_stack([x, y_high])
    lower = np.column_stack([x[::-1], y_low[::-1]])
    return np.vstack([upper, lower])


def _collect_scene_objects(scene: Scene):
    objetos = []
    for panel in scene.paneles:
        for obj in panel.objetos:
            objetos.append(obj)
    return objetos


def _get_artist_map(scene: Scene) -> Dict[int, Any]:
    artist_map = getattr(scene, "_artist_map_by_id", None)
    if not isinstance(artist_map, dict):
        raise TypeError(
            "Scene has no artist map. Ensure the scene was plotted with graficos.plot(scene, show=False) "
            "from a version that supports animations."
        )
    return artist_map


# ============================================================
#  ANIMATION ENGINE (DECLARATIVE API)
# ============================================================

def animate(
    scene: Scene,
    evolve: Dict[Any, Callable[[float], Any]],
    *,
    duration: float,
    fps: int = 30,
    speed: float = 1.0,
    loop: bool = False,
    show: bool = True,
    blit: bool = False
) -> animation.FuncAnimation:
    """
    Declarative time engine.

    INPUT:
        scene: Scene -> graphic structure to animate (defines WHAT)
        evolve: dict {semantic_object: f(t)} -> time laws
            - Serie -> f(t) returns y(t)
            - Serie3D -> f(t) returns (x, y, z)
            - Banda -> f(t) returns (y_low, y_high)
            - Ajuste -> f(t) returns yfit(t)
        duration: total duration in seconds
        fps: frames per second
        speed: time scale factor (t = speed * frame / fps)
        loop: repeat animation when finished
        show: display interactive animation
        blit: enable blitting (only if all artists support it)

    OUTPUT:
        matplotlib.animation.FuncAnimation

    ERRORS:
        - evolve references object not present in scene -> ValueError
        - incompatible data returned -> TypeError
    """
    if not isinstance(scene, Scene):
        # Accept equivalent scenes (e.g., imported from another namespace)
        if not (hasattr(scene, "paneles") and hasattr(scene, "layout")):
            raise TypeError(f"scene must be Scene, received: {type(scene).__name__}")
    if not isinstance(evolve, dict):
        raise TypeError("evolve must be dict {obj: function(t)}")
    if duration <= 0:
        raise ValueError("duration must be > 0")
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if speed <= 0:
        raise ValueError("speed must be > 0")

    # Validate that evolve objects are in the scene (by identity)
    scene_objects = _collect_scene_objects(scene)
    for obj in evolve.keys():
        if not any(obj is scene_obj for scene_obj in scene_objects):
            raise ValueError(f"evolve contains object not present in scene: {type(obj).__name__}")
        if _es_serie_con_error_like(obj):
            raise TypeError("SeriesWithError cannot be animated (static only)")

    # Initial render
    fig, axs = graficos.plot(scene, show=False)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    
    # Fix legend position to avoid moving during animation
    for ax in axs.flat if isinstance(axs, np.ndarray) else [axs]:
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) > 0:
            ax.legend(loc='upper left', frameon=False)

    artist_map = _get_artist_map(scene)
    obj_to_artist: Dict[Any, Any] = {}
    for obj in evolve.keys():
        artist = artist_map.get(id(obj))
        if artist is None:
            raise TypeError(
                f"Could not associate an artist for {type(obj).__name__}. "
                "Replot the scene before calling animate."
            )
        obj_to_artist[obj] = artist

    total_frames = int(np.ceil(duration * fps))
    frames = range(total_frames)

    def _update_frame(frame: int):
        t = speed * frame / fps
        modified = []
        for obj, func in evolve.items():
            artist = obj_to_artist[obj]
            try:
                new_data = func(t)
            except Exception as exc:
                raise TypeError(f"Error evaluating evolve for {type(obj).__name__}: {exc}") from exc

            if isinstance(obj, Series) or _es_serie_like(obj):
                y = np.asarray(new_data, dtype=float)
                if y.shape != obj.y.shape:
                    raise TypeError("Series: evolve must return y with the same length as x")
                obj.y = y
                if hasattr(artist, "set_offsets"):
                    artist.set_offsets(np.column_stack([obj.x, obj.y]))
                elif hasattr(artist, "set_data"):
                    artist.set_data(obj.x, obj.y)
                elif hasattr(artist, "set_xdata") and hasattr(artist, "set_ydata"):
                    artist.set_xdata(obj.x)
                    artist.set_ydata(obj.y)
                else:
                    raise TypeError("Series artist does not support data updates")
                modified.append(artist)
            elif isinstance(obj, Seriess3D) or _es_serie3d_like(obj):
                if not isinstance(new_data, Tuple) and not isinstance(new_data, list):
                    raise TypeError("Series3D: evolve must return (x, y, z)")
                if len(new_data) != 3:
                    raise TypeError("Series3D: evolve must return (x, y, z)")
                x, y, z = [np.asarray(v, dtype=float) for v in new_data]
                if x.shape != y.shape or x.shape != z.shape:
                    raise TypeError("Series3D: x, y, z must have the same length")
                obj.x, obj.y, obj.z = x, y, z
                if hasattr(artist, "set_data_3d"):
                    artist.set_data_3d(x, y, z)
                else:
                    artist.set_data(x, y)
                    artist.set_3d_properties(z)
                modified.append(artist)
            elif isinstance(obj, Band) or _es_banda_like(obj):
                if not isinstance(new_data, Tuple) and not isinstance(new_data, list):
                    raise TypeError("Band: evolve must return (y_low, y_high)")
                if len(new_data) != 2:
                    raise TypeError("Band: evolve must return (y_low, y_high)")
                y_low, y_high = [np.asarray(v, dtype=float) for v in new_data]
                if y_low.shape != obj.y_low.shape or y_high.shape != obj.y_high.shape:
                    raise TypeError("Band: y_low/y_high must have the same length as x")
                obj.y_low, obj.y_high = y_low, y_high
                artist.set_verts([_build_band_verts(obj.x, obj.y_low, obj.y_high)])
                modified.append(artist)
            elif isinstance(obj, Fit) or _es_ajuste_like(obj):
                yfit = np.asarray(new_data, dtype=float)
                if yfit.shape != obj.yfit.shape:
                    raise TypeError("Fit: evolve must return yfit with the same length as x")
                obj.yfit = yfit
                if hasattr(artist, "set_ydata"):
                    artist.set_ydata(obj.yfit)
                elif hasattr(artist, "set_data"):
                    artist.set_data(obj.x, obj.yfit)
                else:
                    raise TypeError("Fit artist does not support data updates")
                modified.append(artist)
            else:
                raise TypeError(f"Unsupported type in evolve: {type(obj).__name__}")
        return modified

    anim = animation.FuncAnimation(
        fig=fig,
        func=_update_frame,
        frames=frames,
        interval=int(1000 / fps),
        repeat=loop,
        blit=blit
    )

    if show:
        plt.show()

    return anim


# ============================================================
#  FACADE (consistency with graficos.py)
# ============================================================

class _Animaciones:
    """Facade: animaciones.animate(...)"""

    animate = staticmethod(animate)


animaciones = _Animaciones()

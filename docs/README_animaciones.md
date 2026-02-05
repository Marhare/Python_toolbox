# animaciones.py

## Purpose
Declarative time engine to animate objects defined in `graficos.py`. It does not compute data; it only updates artists from time‑evolution functions.

## Key concepts
- `Scene` defines **what** is drawn.
- `evolve` defines **how** each object evolves in time.
- `animate` maps physical time to frames and updates artists.

## Main API
- `animate(scene, evolve, duration, fps=30, speed=1.0, loop=False, show=True)`
  - `scene`: `Scene` with panels and objects.
  - `evolve`: dictionary `{object: f(t)}`.
  - Supports `Serie`, `Serie3D`, `Banda`, `Ajuste`.

## `evolve` rules
- `Serie` → `f(t)` returns `y(t)`.
- `Serie3D` → `f(t)` returns `(x, y, z)`.
- `Banda` → `f(t)` returns `(y_low, y_high)`.
- `Ajuste` → `f(t)` returns `yfit(t)`.

## Output
Returns a `matplotlib.animation.FuncAnimation`. If `show=True`, shows the interactive animation.

## Notebook notes
- In inline backends the animation may appear as a static frame.
- Saving to GIF/MP4 with `anim.save(...)` is recommended for playback outside the notebook.

## Typical errors
- Invalid `scene`.
- Objects in `evolve` not present in the scene.
- Incompatible data format returned by `f(t)`.

## Example
```python
import numpy as np
from graficos import graficos
from animaciones import animaciones

x = np.linspace(0, 2*np.pi, 200)
y = np.sin(x)
serie = graficos.Serie(x, y, label="sin")
scene = graficos.Scene(serie, title="Animation")
anim = animaciones.animate(scene, {serie: lambda t: y*np.cos(t)}, duration=2.0)
```
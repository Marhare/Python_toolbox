# animations.py

## Purpose
Declarative time engine to animate objects defined in `graphics.py`. It does not compute data; it only updates artists from time‑evolution functions.

## Key concepts
- `Scene` defines **what** is drawn.
- `evolve` defines **how** each object evolves in time.
- `animate` maps physical time to frames and updates artists.

## Main API
- `animate(scene, evolve, duration, fps=30, speed=1.0, loop=False, show=True, blit=False)`
  - `scene`: `Scene` with panels and objects.
  - `evolve`: dictionary `{object: f(t)}`.
  - Supports `Series`, `Series3D`, `Band`, `Fit`.

## `evolve` rules
- `Series` → `f(t)` returns `y(t)`.
- `Series3D` → `f(t)` returns `(x, y, z)`.
- `Band` → `f(t)` returns `(y_low, y_high)`.
- `Fit` → `f(t)` returns `yfit(t)`.

## Output
Returns a `matplotlib.animation.FuncAnimation`. If `show=True`, shows the interactive animation.

## Notebook notes
- In inline backends the animation may appear as a static frame.
- Saving to GIF/MP4 with `anim.save(...)` is recommended for playback outside the notebook.
- Use `%matplotlib widget` in Jupyter for interactive animations.

## Typical errors
- Invalid `scene`.
- Objects in `evolve` not present in the scene.
- Incompatible data format returned by `f(t)`.

## Example
```python
import numpy as np
import marhare as mh
from marhare.graphics import Series, Scene
from marhare.animations import animate

x = np.linspace(0, 2*np.pi, 200)
y = np.sin(x)
serie = Series(x, y, label="sin")
scene = Scene([serie], title="Animation")

# Animate: y evolves as y * cos(t)
anim = animate(scene, {serie: lambda t: y*np.cos(t)}, duration=2.0, show=True)
```
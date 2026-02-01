# animaciones.py

## Propósito
Motor temporal declarativo para animar objetos definidos en `graficos.py`. No calcula datos, solo actualiza artistas a partir de funciones de evolución temporal.

## Conceptos clave
- `Scene` define **qué** se dibuja.
- `evolve` define **cómo** evoluciona cada objeto en el tiempo.
- `animate` transforma tiempo físico en frames y actualiza artistas.

## API principal
- `animate(scene, evolve, duration, fps=30, speed=1.0, loop=False, show=True)`
  - `scene`: `Scene` con paneles y objetos.
  - `evolve`: diccionario `{objeto: f(t)}`.
  - Soporta `Serie`, `Serie3D`, `Banda`, `Ajuste`.

## Reglas de `evolve`
- `Serie` → `f(t)` devuelve `y(t)`.
- `Serie3D` → `f(t)` devuelve `(x, y, z)`.
- `Banda` → `f(t)` devuelve `(y_low, y_high)`.
- `Ajuste` → `f(t)` devuelve `yfit(t)`.

## Salida
Devuelve un `matplotlib.animation.FuncAnimation`. Si `show=True`, muestra la animación interactiva.

## Notas de notebook
- En backends inline la animación puede mostrarse como frame estático.
- Se recomienda guardar a GIF/MP4 con `anim.save(...)` para reproducir fuera del notebook.

## Errores típicos
- `scene` no válido.
- Objetos en `evolve` no presentes en la escena.
- Formato de datos devuelto por `f(t)` incompatible.

## Ejemplo
```python
import numpy as np
from graficos import graficos
from animaciones import animaciones

x = np.linspace(0, 2*np.pi, 200)
y = np.sin(x)
serie = graficos.Serie(x, y, label="sin")
scene = graficos.Scene(serie, title="Animación")
anim = animaciones.animate(scene, {serie: lambda t: y*np.cos(t)}, duration=2.0)
```
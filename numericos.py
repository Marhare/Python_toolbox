import numpy as np
from scipy.integrate import quad

class _Numericos:

    @staticmethod
    def derivada(f, x, h=1e-5):
        return (f(x+h) - f(x-h)) / (2*h)

    @staticmethod
    def integrar(f, a, b):
        val, _ = quad(f, a, b)
        return val

    @staticmethod
    def rk4(f, t_span, y0, dt):
        t0, tf = t_span
        t_vals = np.arange(t0, tf+dt, dt)
        y = np.array(y0, float)
        ys = []

        for t in t_vals:
            ys.append(y.copy())
            k1 = f(t, y)
            k2 = f(t+dt/2, y+dt*k1/2)
            k3 = f(t+dt/2, y+dt*k2/2)
            k4 = f(t+dt, y+dt*k3)
            y += dt*(k1+2*k2+2*k3+k4)/6

        return {"t": t_vals, "y": np.array(ys)}

numericos = _Numericos()

import numpy as np

class _MonteCarlo:

    @staticmethod
    def integral_1d(f, a, b, n=10000):
        x = np.random.uniform(a, b, n)
        fx = f(x)
        val = (b - a) * np.mean(fx)
        err = (b - a) * np.std(fx, ddof=1) / np.sqrt(n)
        return {"valor": val, "error": err}

    @staticmethod
    def propagacion(fun, generadores, n=10000):
        muestras = [g(n) for g in generadores]
        vals = fun(*muestras)
        return {
            "media": float(np.mean(vals)),
            "sigma": float(np.std(vals, ddof=1)),
        }

montecarlo = _MonteCarlo()

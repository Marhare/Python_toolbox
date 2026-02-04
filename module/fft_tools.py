import numpy as np
from scipy.fft import rfft, rfftfreq

class _FFTTools:

    @staticmethod
    def fft(signal, dt):
        s = np.asarray(signal)
        freqs = rfftfreq(len(s), dt)
        fft_vals = rfft(s)
        return {"freqs": freqs, "fft": fft_vals}

    @staticmethod
    def espectro_potencia(signal, dt):
        s = np.asarray(signal)
        freqs = rfftfreq(len(s), dt)
        fft_vals = np.abs(rfft(s))**2
        return {"freqs": freqs, "psd": fft_vals}

fft_tools = _FFTTools()

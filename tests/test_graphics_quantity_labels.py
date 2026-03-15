import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import marhare as mh


def test_dimensionless_unit_not_shown_in_axis_label():
    x = mh.quantity([1.0, 2.0, 3.0], [0.1, 0.1, 0.1], "m", symbol="x")
    n = mh.quantity([1.4, 1.5, 1.6], [0.02, 0.02, 0.02], "1", symbol="n")

    fig, ax = mh.plot(x, n, show=False)
    try:
        ylabel = ax.get_ylabel()
        assert ylabel == "n"
        assert "[1]" not in ylabel
    finally:
        plt.close(fig)


def test_latex_symbol_is_rendered_as_mathtext():
    wl = mh.quantity([500.0, 510.0, 520.0], [1.0, 1.0, 1.0], "nm", symbol=r"\lambda", normalize=False)
    y = mh.quantity([1.0, 1.1, 1.2], [0.02, 0.02, 0.02], "1", symbol="n")

    fig, ax = mh.plot(wl, y, show=False)
    try:
        xlabel = ax.get_xlabel()
        assert "$\\lambda$" in xlabel
    finally:
        plt.close(fig)


def test_degrees_label_reflects_normalized_internal_unit():
    theta = mh.quantity([0.0, 45.0, 90.0], [0.5, 0.5, 0.5], "degrees", symbol=r"\theta")
    y = mh.quantity([0.0, 1.0, 2.0], [0.1, 0.1, 0.1], "m", symbol="y")

    fig, ax = mh.plot(theta, y, show=False)
    try:
        xlabel = ax.get_xlabel().lower()
        assert "degree" not in xlabel
        assert ("rad" in xlabel) or ("radian" in xlabel)
    finally:
        plt.close(fig)

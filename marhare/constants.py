"""Physical constants as Quantity objects.

Import examples:
- from marhare.constants import h, e, m_e
- import marhare as mh; mh.h
"""

import numpy as np

from .quantities import Quantity

# Fundamental constants (SI)
h = Quantity(6.62607015e-34, 0.0, "J*s", symbol="h")
e = Quantity(1.602176634e-19, 0.0, "C", symbol="e")
m_e = Quantity(9.10938356e-31, 0.0, "kg", symbol="m_e")
c = Quantity(299792458.0, 0.0, "m/s", symbol="c")
k_B = Quantity(1.380649e-23, 0.0, "J/K", symbol="k_B")
N_A = Quantity(6.02214076e23, 0.0, "1/mol", symbol="N_A")
R = Quantity(8.31446261815324, 0.0, "J/mol/K", symbol="R")
epsilon_0 = Quantity(8.8541878128e-12, 0.0, "F/m", symbol="epsilon_0")
mu_0 = Quantity(1.25663706212e-6, 0.0, "N/A**2", symbol="mu_0")
G = Quantity(6.67430e-11, 0.0, "m**3/(kg*s**2)", symbol="G")
g0 = Quantity(9.80665, 0.0, "m/s**2", symbol="g_0")

# Derived constants
hbar = h / (2.0 * np.pi)

# Optional descriptive aliases
planck_constant = h
elementary_charge = e
electron_mass = m_e
speed_of_light = c
boltzmann_constant = k_B
avogadro_constant = N_A
gas_constant = R
vacuum_permittivity = epsilon_0
vacuum_permeability = mu_0
gravitational_constant = G
standard_gravity = g0
reduced_planck_constant = hbar

__all__ = [
    "h",
    "e",
    "m_e",
    "c",
    "hbar",
    "k_B",
    "N_A",
    "R",
    "epsilon_0",
    "mu_0",
    "G",
    "g0",
    "planck_constant",
    "elementary_charge",
    "electron_mass",
    "speed_of_light",
    "boltzmann_constant",
    "avogadro_constant",
    "gas_constant",
    "vacuum_permittivity",
    "vacuum_permeability",
    "gravitational_constant",
    "standard_gravity",
    "reduced_planck_constant",
]

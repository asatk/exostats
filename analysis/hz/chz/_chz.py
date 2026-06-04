import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Kopparapu+ 14 Recent Venus Limit (inner optimistic CHZ)
s_rv = 1.776
a_rv = 2.136e-4
b_rv = 2.533e-8
c_rv = -1.332e-11
d_rv = -3.097e-15

# Kopparapu+ 14 Runaway Greenhouse Limit (inner conservative CHZ)
s_rg = np.array([0.990, 1.107, 1.188])
a_rg = np.array([1.209e-4, 1.332e-4, 1.433e-4])
b_rg = np.array([1.404e-8, 1.580e-8, 1.707e-8])
c_rg = np.array([-7.418e-12, -8.308e-12, -8.968e-12])
d_rg = np.array([-1.713e-15, -1.931e-15, -2.084e-15])

# Kopparapu+ 14 Maximum Greenhouse Limit (outer conservative CHZ)
s_mg = 0.356
a_mg = 6.171e-5
b_mg = 1.698e-9
c_mg = -3.198e-12
d_mg = -5.575e-16

# Kopparapu+ 14 Early Mars Limit (outer optimistic CHZ)
s_em = 0.32
a_em = 5.547e-5
b_em = 1.526e-9
c_em = -2.874e-12
d_em = -5.011e-16

# Kopparapu+ 14 reference temperature (Sun's surface temperature)
t0 = 5780

def create_interpolator_k14_rg(ntemps: int=1000, is_bounded: bool=False):
    temps = np.linspace(2600, 7200, ntemps, endpoint=True)
    t = temps - t0
    masses = np.array([0.1, 1.0, 5.0])

    params = np.stack([s_rg, a_rg, b_rg, c_rg, d_rg])
    t_pows = np.power.outer(t, np.arange(len(params)))
    fluxes = t_pows @ params

    interp = RegularGridInterpolator((t, masses), fluxes, bounds_error=is_bounded, fill_value=np.nan)
    return interp

# Recent Venus
def k14_rv(temp: np.ndarray, lumi: np.ndarray, mass_pl: np.ndarray) -> np.ndarray:
    t = temp - t0
    flux = s_rv + \
        a_rv * t + \
        b_rv * t ** 2 + \
        c_rv * t ** 3 + \
        d_rv * t ** 4
    dist = np.sqrt(lumi / flux)
    return dist

# Runaway Greenhouse
def k14_rg(temp: np.ndarray, lumi: np.ndarray, mass_pl: np.ndarray, interp) -> np.ndarray:
    temp = np.clip(temp, a_min=2601, a_max=7199)
    mass_pl = np.clip(mass_pl, a_min=0.11, a_max=4.9)

    t = (temp - t0)

    interp_input = np.stack([t, mass_pl]).T
    flux = interp(interp_input)
    dist = np.sqrt(lumi / flux)
    return dist

# Maximum Greenhouse
def k14_mg(temp: np.ndarray, lumi: np.ndarray, mass_pl: np.ndarray) -> np.ndarray:
    t = temp - t0
    flux = s_mg + \
        a_mg * t + \
        b_mg * t ** 2 + \
        c_mg * t ** 3 + \
        d_mg * t ** 4
    dist = np.sqrt(lumi / flux)
    return dist

# Early Mars
def k14_em(temp: np.ndarray, lumi: np.ndarray, mass_pl: np.ndarray) -> np.ndarray:
    t = temp - t0
    flux = s_em + \
        a_em * t + \
        b_em * t ** 2 + \
        c_em * t ** 3 + \
        d_em * t ** 4
    dist = np.sqrt(lumi / flux)
    return dist
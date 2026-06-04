import numpy as np
import pandas as pd
from scipy.interpolate import CloughTocher2DInterpolator

# Wright+ 18 Convective Turnover Time (Mass)
w18_sm_c0 = 2.33
w18_sm_c1 = -1.50
w18_sm_c2 = 0.31

def tauc_w18_sm(mass: np.ndarray, is_bounded: bool=False):
    if is_bounded:
        oob = (mass < 0.08) | (mass > 1.36)
        mass = np.copy(mass)
        mass[oob] = np.nan
    log_tauc = w18_sm_c0 + w18_sm_c1 * mass + w18_sm_c2 * mass ** 2
    return 10. ** log_tauc



# Wright+ 18 Convective Turnover Time (V-K Color)
w18_vk_c0 = 0.64
w18_vk_c1 = 0.25

def tauc_w18_vk(vk: np.ndarray, is_bounded: bool=False):
    if is_bounded:
        oob = (vk < 1.1) | (vk > 7.0)
        vk = np.copy(vk)
        vk[oob] = np.nan

    log_tauc = w18_vk_c0 + w18_vk_c1 * vk
    return 10. ** log_tauc



# Interpolate temp + lumi class for V-K color
def fetch_spline_data(fname: str):
    ms = pd.read_csv(fname, delimiter=r"\s+", na_values="--")
    ms.sort_values(by="Te(K)", ascending=True, inplace=True)
    bad_data = np.isnan(ms["Te(K)"])
    vk_ms = np.asarray(ms.loc[~bad_data, "V-K"], dtype=np.float64)
    teff_ms = np.asarray(ms.loc[~bad_data, "Te(K)"], dtype=np.float64)
    return teff_ms, vk_ms



def create_interpolator_temp_vk(is_bounded: bool=False):
    teff_supg, vk_supg = fetch_spline_data("../db/analysis/teff-color-supg.txt")
    teff_gnt, vk_gnt = fetch_spline_data("../db/analysis/teff-color-gnt.txt")
    teff_ms, vk_ms = fetch_spline_data("../db/analysis/teff-color-ms.txt")

    coord_supg = np.stack([teff_supg, np.full_like(teff_supg, 1)], axis=1)
    coord_gnt = np.stack([teff_gnt, np.full_like(teff_gnt, 3)], axis=1)
    coord_ms = np.stack([teff_ms, np.full_like(teff_ms, 5)], axis=1)

    coords = np.r_[coord_supg, coord_gnt, coord_ms]
    vals = np.r_[vk_supg, vk_gnt, vk_ms]

    interp = CloughTocher2DInterpolator(coords, vals, fill_value=np.nan)
    return interp



# Schrijver+ 03 Alfven Radius
r_sun = 6.957e8
au = 1.496e11

s = -1.38
r = -0.16
Ro_sun = 1.85
ra_sun = 20 * r_sun / au

def a24_asurf(teff: np.ndarray,
              lumclass: np.ndarray,
              prot: np.ndarray,
              interp,
              is_bounded: bool=False):
    vk = interp(teff, lumclass)
    tauc = tauc_w18_vk(vk, is_bounded=is_bounded)
    Ro = prot / tauc
    ra = ra_sun * np.real(np.power(Ro / Ro_sun, s * r))
    return ra


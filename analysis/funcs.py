from astropy.modeling import Model
from astropy.modeling.models import Spline1D
from astropy.modeling.fitting import SplineInterpolateFitter
import numpy as np


########### ASHZ

w18_sm_c0 = 2.33
w18_sm_c1 = -1.50
w18_sm_c2 = 0.31

def tauc_w18_sm(mass: np.ndarray):

    oob = (0.08 > mass) | (mass > 1.36)
    mass[oob] = np.nan

    log_tauc = w18_sm_c0 + w18_sm_c1 * mass + w18_sm_c2 * mass ** 2

    return 10. ** log_tauc



w18_vk_c0 = 0.64
w18_vk_c1 = 0.25

def tauc_w18_vk(color: np.ndarray):

    oob = (1.1 > color) | (color > 7.0)
    color[oob] = np.nan

    log_tauc = w18_vk_c0 + w18_vk_c1 * color
    return 10. ** log_tauc




r_sun = 6.957e8
au = 1.496e11

s = -1.38
r = -0.16
Ro_sun = 1.85
ra_sun = 20 * r_sun / au

def ra_s03(Ro: np.ndarray):
    return ra_sun * np.real(np.power(Ro / Ro_sun, s * r))



def ashz_inner_a24(temp: np.ndarray, prot: np.ndarray, vk: np.ndarray):
    tauc = tauc_w18_vk(vk)
    Ro = prot / tauc
    ra = ra_s03(Ro)
    return ra




######### UHZ

def mag_to_lumi_nuv(dist: np.ndarray, mag_nuv: np.ndarray):
    Mag_nuv = mag_nuv - 5 * np.log10(dist) + 5

s23_smin = 45
s23_smax = 2 * 5.2 * 1e3

def uhz_inner_s23(temp: np.ndarray, f_nuv: np.ndarray, lumi_nuv: np.ndarray):
    return np.sqrt(lumi_nuv * f_nuv / s23_smax) / au / 1e2

def uhz_outer_s23(temp: np.ndarray, f_nuv: np.ndarray, lumi_nuv: np.ndarray):
    return np.sqrt(lumi_nuv * f_nuv / s23_smin) / au / 1e2





########## CHZ

# runaway greenhouse (inner CHZ)
chz_spl = Spline1D(degree=1)
chz_fit = SplineInterpolateFitter()
k14_mpl = np.array([0.1, 1, 5])
k14_rg_s = np.array([0.990, 1.107, 1.188])
k14_rg_a = np.array([1.209e-4, 1.332e-4, 1.433e-4])
k14_rg_b = np.array([1.404e-8, 1.580e-8, 1.707e-8])
k14_rg_c = np.array([-7.418e-12, -8.308e-12, -8.968e-12])
k14_rg_d = np.array([-1.713e-15, -1.931e-15, -2.084e-15])

k14_rg_s_model = chz_fit(chz_spl, k14_mpl, k14_rg_s)
k14_rg_a_model = chz_fit(chz_spl, k14_mpl, k14_rg_a)
k14_rg_b_model = chz_fit(chz_spl, k14_mpl, k14_rg_b)
k14_rg_c_model = chz_fit(chz_spl, k14_mpl, k14_rg_c)
k14_rg_d_model = chz_fit(chz_spl, k14_mpl, k14_rg_d)

# maximum greenhouse (outer CHZ)
k14_mg_s = 0.356
k14_mg_a = 6.171e-5
k14_mg_b = 1.698e-9
k14_mg_c = -3.198e-12
k14_mg_d = -5.575e-16


def chz_inner_k14(temp: np.ndarray, lumi: np.ndarray, mpl: np.ndarray):
    t = temp - 5780
    oob = (0.1 > mpl) | (mpl > 5.0)
    if isinstance(mpl, np.ndarray):
        mpl[oob] = np.nan
    elif isinstance(mpl, float) and oob:
        return np.nan

    rg_s = k14_rg_s_model(mpl)
    rg_a = k14_rg_a_model(mpl)
    rg_b = k14_rg_b_model(mpl)
    rg_c = k14_rg_c_model(mpl)
    rg_d = k14_rg_d_model(mpl)
    seff = rg_s + rg_a * t + rg_b * t**2 + rg_c * t**3 + rg_d * t**4
    return np.sqrt(lumi / seff)

def chz_outer_k14(temp: np.ndarray, lumi: np.ndarray):
    t = temp - 5780
    seff = k14_mg_s + k14_mg_a * t + k14_mg_b * t**2 + k14_mg_c * t**3 + k14_mg_d * t**4
    return np.sqrt(lumi / seff)

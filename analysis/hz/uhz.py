import numpy as np

from hz import HabitableZone

######### UHZ

au = 1.496e11

fit_const = -48.22
fit_power = 21.12

def lnuv_fit(temp: np.ndarray):
    log_lumi = fit_power * np.log10(temp) + fit_const
    return 10. ** log_lumi

# def mag_to_lumi_nuv(dist: np.ndarray, mag_nuv: np.ndarray):
#     Mag_nuv = mag_nuv - 5 * np.log10(dist) + 5
#     return Mag_nuv

flux_min = 45
flux_max = 2 * 5.2 * 1e3

def s23_rnasyn(temp: np.ndarray, f_nuv: np.ndarray):
    lumi_nuv = lnuv_fit(temp)
    return np.sqrt(lumi_nuv * f_nuv / flux_max) / au / 1e2


def s23_maxtol(temp: np.ndarray, f_nuv: np.ndarray):
    lumi_nuv = lnuv_fit(temp)
    return np.sqrt(lumi_nuv * f_nuv / flux_min) / au / 1e2



class UltravioletHabitableZone(HabitableZone):

    def __init__(self,
                 is_bounded: bool=True):
        self._is_bounded = is_bounded

    def limits(self, *data):
        temp = data[0]
        f_nuv = data[1]
        lo = s23_rnasyn(temp, f_nuv)
        hi = s23_maxtol(temp, f_nuv)
        return np.array([lo, hi])

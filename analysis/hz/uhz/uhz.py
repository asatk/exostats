import numpy as np

from ..hz import UniformHabitableZone
from ._uhz import s23_rnasyn, s23_maxtol

class UltravioletHabitableZone(UniformHabitableZone):

    def __init__(self,
                 teff: np.ndarray,
                 fnuv: np.ndarray,
                 outer_boundary: float|np.ndarray,
                 is_bounded: bool=True):
        super().__init__(teff, fnuv)
        self._teff = teff
        self._fnuv = fnuv
        self._outer_boundary = outer_boundary
        self._is_bounded = is_bounded

    @property
    def teff(self):
        return self._teff

    @teff.setter
    def teff(self, value):
        self._teff = value

    @property
    def fnuv(self):
        return self._fnuv

    @fnuv.setter
    def fnuv(self, value):
        self._fnuv = value

    @property
    def outer_boundary(self):
        return self._outer_boundary

    @outer_boundary.setter
    def outer_boundary(self, value):
        self._outer_boundary = value

    def inner_rad(self):
        return s23_maxtol(self._teff, self._fnuv)

    def outer_rad(self):
        return np.full(self._nrows, self._outer_boundary)

class UltravioletHabitableZoneABG(UltravioletHabitableZone):

    def __init__(self,
                 teff: np.ndarray,
                 fnuv: np.ndarray,
                 is_bounded: bool=True):
        super().__init__(teff,
                         fnuv,
                         outer_boundary=np.nan,
                         is_bounded=is_bounded)

    def outer_rad(self):
        return s23_rnasyn(self._teff, self._fnuv)
import numpy as np

from ..hz import UniformHabitableZone
from ._ashz import create_interpolator_temp_vk
from ._ashz import a24_asurf

class AlfvenSurfaceHabitableZone(UniformHabitableZone):

    vk_bound_lo = 1.1
    vk_bound_hi = 7.0

    def __init__(self,
                 teff: np.ndarray,
                 lumclass: np.ndarray,
                 prot: np.ndarray,
                 outer_boundary: float|np.ndarray,
                 is_bounded: bool=True):
        super().__init__(teff, lumclass, prot)
        self._teff = teff
        self._lumclass = lumclass
        self._prot = prot
        self._is_bounded = is_bounded
        self._outer_boundary = outer_boundary
        self._interp = create_interpolator_temp_vk(is_bounded=is_bounded)

    @property
    def teff(self):
        return self._teff

    @teff.setter
    def teff(self, value):
        self._teff = value

    @property
    def lumclass(self):
        return self._lumclass

    @lumclass.setter
    def lumclass(self, value):
        self._lumclass = value

    @property
    def prot(self):
        return self._prot

    @prot.setter
    def prot(self, value):
        self._prot = value

    @property
    def outer_boundary(self):
        return self._outer_boundary

    @outer_boundary.setter
    def outer_boundary(self, value):
        self._outer_boundary = value

    def inner_rad(self):
        return a24_asurf(self.teff, self.lumclass, self.prot, self._interp)

    def outer_rad(self):
        return np.full(self._nrows, self._outer_boundary)
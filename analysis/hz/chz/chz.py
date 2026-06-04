import numpy as np
from ..hz import UniformHabitableZone

from ._chz import create_interpolator_k14_rg
from ._chz import k14_em, k14_rg, k14_mg, k14_rv

class CircumstellarHabitableZone(UniformHabitableZone):
    """
    CHZ limits (au) for a system given the host's effective temperature (K) and
    total luminosity (L_Sun) and a planet's mass. These limits are considered
    'optimistic'.
    """

    teff_bound_lo = 2600.0
    teff_bound_hi = 7200.0

    mass_pl_bound_lo = 0.1
    mass_pl_bound_hi = 5.0

    def __init__(self,
                 teff: np.ndarray,
                 lumi: np.ndarray,
                 plmass: np.ndarray,
                 is_bounded: bool=True):
        """
        is_bounded : bool
            Ability to calculate HZ is determined by bounds on inputs decided
            by the HZ models used. For CHZ, the temperatures are limited to
            Teff in [2600K, 7200K].
        """
        super().__init__(teff, lumi, plmass)
        self._teff = teff
        self._lumi = lumi
        self._plmass = plmass
        self._is_bounded = is_bounded

    @property
    def teff(self):
        return self._teff

    @teff.setter
    def teff(self, value):
        self._teff = value

    @property
    def lumi(self):
        return self._lumi

    @lumi.setter
    def lumi(self, value):
        self._lumi = value

    @property
    def plmass(self):
        return self._plmass

    @plmass.setter
    def plmass(self, value):
        self._plmass = value

    def inner_rad(self):
        return k14_rv(self.teff, self.lumi, self.plmass)

    def outer_rad(self):
        return k14_em(self.teff, self.lumi, self.plmass)


class ConservativeCircumstellarHabitableZone(CircumstellarHabitableZone):
    """
    CHZ limits (au) for a system given the host's effective temperature (K) and
    total luminosity (L_Sun) and a planet's mass. These limits are considered
    'conservative'.
    """

    def __init__(self,
                 teff: np.ndarray,
                 lumi: np.ndarray,
                 plmass: np.ndarray,
                 is_bounded: bool = True):
        """
        is_bounded : bool
            Ability to calculate HZ is determined by bounds on inputs decided
            by the HZ models used. For CHZ, the temperatures are limited to
            Teff in [2600K, 7200K].
        """
        super().__init__(teff, lumi, plmass, is_bounded)
        self._interp = create_interpolator_k14_rg(is_bounded=is_bounded)

    def inner_rad(self):
        return k14_rg(self.teff, self.lumi, self.plmass, self._interp)

    def outer_rad(self):
        return k14_mg(self.teff, self.lumi, self.plmass)


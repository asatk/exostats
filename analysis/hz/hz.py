import abc

import numpy as np

class HabitableZone(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def limits(self, *data):
        return NotImplementedError

    def zone(self, pts: np.ndarray, *data):
        """
        Creates 2D annular zone assuming uniform inner and outer HZ radii.
        """
        npts = pts.shape[0]
        lims = self.limits(*data)
        annulus = np.repeat(lims, repeats=npts).reshape(2, npts)
        return annulus

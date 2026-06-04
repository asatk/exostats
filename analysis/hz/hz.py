import abc
import numpy as np

class HabitableZone(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def zone(self, theta: np.ndarray, phi: np.ndarray):
        return NotImplementedError

class UniformHabitableZone(HabitableZone):

    def __init__(self, *args):
        nrows = _check_inputs(*args)
        self._nrows = nrows

    @property
    def nrows(self) -> int:
        return self._nrows

    @abc.abstractmethod
    def inner_rad(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def outer_rad(self) -> np.ndarray:
        pass

    def zone(self, theta: float|np.ndarray, phi: float|np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Creates 2D annular zone assuming uniform inner and outer HZ radii.
        """
        if isinstance(theta, float|np.float64):
            ncoords = 1
        else:
            ncoords = len(theta)

        sphere_coords = [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ]

        # lo = self.inner_rad().reshape(self._nrows, 1)
        lo = self.inner_rad()
        lo_coords = np.multiply.outer(lo, sphere_coords)

        # hi = self.outer_rad().reshape(self._nrows, 1)
        hi = self.outer_rad()
        hi_coords = np.multiply.outer(hi, sphere_coords)

        # lo_ring = np.repeat(lo, repeats=ncoords, axis=1)
        # hi_ring = np.repeat(hi, repeats=ncoords, axis=1)

        # zone_boundary = np.stack([lo_ring, hi_ring], axis=1)

        # dimensions: [input rows, cartesian axis, input coordinate]
        return (lo_coords, hi_coords)

def _check_inputs(*args) -> int:
    nrows_list = [len(np.ravel(a)) for a in args]
    unique_lens = len(set(nrows_list))
    if unique_lens != 1:
        raise ValueError(f"Input arrays are not the same lengths: {nrows_list}")
    return nrows_list[0]

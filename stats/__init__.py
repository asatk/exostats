from .kmeans import learnkNN, predictLabel
from .mlr import OLSFit, RidgeFit
from .plot import diag_plot
from .utils import load

__all__ = [
    "learnkNN",
    "predictLabel",
    "OLSFit",
    "RidgeFit",
    "diag_plot",
    "load"
]
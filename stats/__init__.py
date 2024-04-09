from .knn import learnkNN, predictLabel
from .mlr import learnLM, predictLM
from .plot import diag_plot

__all__ = [
    "learnkNN",
    "predictLabel",
    "learnLM",
    "predictLM",
    "diag_plot"
]
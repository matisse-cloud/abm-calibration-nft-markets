"""Calibration toolkit for a modified Chiarella agent-based model.

Modules
-------
chiarella : Numba-compiled simulator of the modified Chiarella model.
moments   : Numba-compiled stylised-fact moments used as GMM targets.
gmm       : Simulated Method of Moments (SMM/GMM) machinery: distance,
            optimiser benchmark and iterative optimal-weighting GMM.
plotting  : Shared dark plotting theme.
"""
from .chiarella import chiarella_model, PARAM_NAMES
from .moments import compute_moments, MOMENT_NAMES
from . import gmm, plotting

__all__ = ["chiarella_model", "PARAM_NAMES", "compute_moments", "MOMENT_NAMES",
           "gmm", "plotting"]

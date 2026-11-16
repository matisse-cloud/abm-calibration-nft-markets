"""Stylised-fact moments, re-implemented for Numba (scipy.stats is not
supported in nopython mode). Six moments for six parameters."""
import numpy as np
from numba import njit

MOMENT_NAMES = ["Var", "Kurtosis", "Q99|r|", "ACF(r)", "ACF(r²)", "Skew"]


@njit(cache=True)
def _kurtosis(r):
    mu, std = np.mean(r), np.std(r)
    if std == 0.0:
        return 0.0
    return np.mean(((r - mu) / std) ** 4) - 3.0   # excess (Fisher) kurtosis


@njit(cache=True)
def _skew(r):
    mu, std = np.mean(r), np.std(r)
    if std == 0.0:
        return 0.0
    return np.mean(((r - mu) / std) ** 3)


@njit(cache=True)
def _corrcoef(x, y):
    mx, my = np.mean(x), np.mean(y)
    num = np.mean((x - mx) * (y - my))
    denom = np.std(x) * np.std(y)
    if denom == 0.0:
        return 0.0
    return num / denom


@njit(cache=True)
def _quantile(arr, q):
    s = np.sort(arr)
    idx = q * (len(s) - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= len(s):
        return s[lo]
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


@njit(cache=True)
def compute_moments(r):
    """Return the 6 target moments of a return series, in MOMENT_NAMES order."""
    return np.array([
        np.var(r),                           # 1. volatility level
        _kurtosis(r),                        # 2. fat tails
        _quantile(np.abs(r), 0.99),          # 3. size of extremes
        _corrcoef(r[:-1], r[1:]),            # 4. linear predictability (lag-1 ACF)
        _corrcoef(r[:-1] ** 2, r[1:] ** 2),  # 5. volatility clustering
        _skew(r),                            # 6. asymmetry
    ])

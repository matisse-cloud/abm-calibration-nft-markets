"""Modified Chiarella model (fundamentalists vs chartists), JIT-compiled.

    p_{t+1} = p_t + kappa * [(v_t - p_t)^3 - (v_{t-1} - p_{t-1})^3]
                  + beta  * [tanh(gamma m_t) - tanh(gamma m_{t-1})]
                  + sigma_N * eta_t
    m_t = (1 - alpha) m_{t-1} + alpha (p_t - p_{t-1})     # trend memory
    v_t = (1 - lambda) v_{t-1} + lambda p_t               # fundamental value = EMA of prices

The modification with respect to the textbook model is that the fundamental
value v_t is no longer exogenous but an exponential moving average of prices.
"""
import numpy as np
from numba import njit

PARAM_NAMES = ["kappa", "beta", "gamma", "alpha", "lmbda", "sigma_N"]


@njit(cache=True)
def chiarella_model(eta, kappa, beta, gamma, alpha, lmbda, sigma_N):
    """Simulate one log-price path.

    Parameters
    ----------
    eta : ndarray
        Pre-drawn standard normal shocks; its length sets the horizon. Passing
        the shocks in (rather than drawing them inside) lets the caller reuse
        the same draws across parameter values (common random numbers).
    kappa, beta, gamma, alpha, lmbda, sigma_N : float
        Model parameters (see module docstring).

    Returns
    -------
    p : ndarray, log-prices
    returns : ndarray, r_t = p_{t+1} - p_t
    """
    nit = len(eta)
    v = np.zeros(nit)
    m = np.zeros(nit)
    p = np.zeros(nit)

    # The loop is recursive and non-linear, so it cannot be vectorised.
    for t in range(1, nit - 1):
        v[t] = (1 - lmbda) * v[t - 1] + lmbda * p[t]
        m[t] = (1 - alpha) * m[t - 1] + alpha * (p[t] - p[t - 1])
        p[t + 1] = (
            p[t]
            + kappa * ((v[t] - p[t]) ** 3 - (v[t - 1] - p[t - 1]) ** 3)   # fundamentalists
            + beta * (np.tanh(gamma * m[t]) - np.tanh(gamma * m[t - 1]))  # chartists
            + sigma_N * eta[t]                                            # exogenous noise
        )

    returns = np.diff(p)
    return p, returns

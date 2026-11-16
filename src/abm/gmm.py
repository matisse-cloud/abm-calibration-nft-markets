"""Simulated Method of Moments / GMM machinery.

Distance (Hansen, 1982), estimated by simulation:
    D_W(xi) = d(xi)^T W d(xi),   d_k = (mean_sim c_k(xi) - c_k_target) / s_k
where s_k is the inter-simulation std of moment k at the target, so that all
six moments contribute on an O(1) scale.
"""
import time

import numpy as np
from scipy.optimize import minimize, differential_evolution

from .chiarella import chiarella_model
from .moments import compute_moments

# Parameter order: kappa, beta, gamma, alpha, lmbda, sigma_N
BOUNDS = [
    (0.001, 0.5),   # kappa
    (0.001, 0.5),   # beta
    (0.05, 2.0),    # gamma
    (0.01, 0.5),    # alpha
    (0.01, 0.5),    # lmbda
    (0.001, 0.1),   # sigma_N
]
LO = np.array([b[0] for b in BOUNDS])
HI = np.array([b[1] for b in BOUNDS])

NIT_SIM = 1500   # length of each simulated path
N_SIM = 6        # paths averaged per distance evaluation

METHODS_FREE = ['Nelder-Mead', 'Powell', 'COBYLA', 'COBYQA']        # derivative-free
METHODS_BOUNDED = ['L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr']      # finite-diff gradient + bounds


def target_scale(xi_true, n_rep=150, nit=NIT_SIM, seed=7):
    """Target moments at xi_true and their inter-simulation std (normalisation)."""
    rng = np.random.default_rng(seed)
    ml = np.empty((n_rep, 6))
    for i in range(n_rep):
        eta = rng.normal(0, 1, nit)
        _p, r = chiarella_model(eta, *xi_true)
        ml[i] = compute_moments(r)
    return ml.mean(axis=0), ml.std(axis=0) + 1e-9


def make_distance(moments_target, scale, n_sim=N_SIM, nit=NIT_SIM, crn_seed=None):
    """Build the normalised distance D_W(x, W).

    crn_seed=None (default, used throughout the main study): fresh shocks are
    drawn at every call, so D_W is a *noisy* function of x.
    crn_seed=int: the same bank of shocks is reused at every call (common
    random numbers), which makes D_W a deterministic function of x.
    """
    eta_bank = None
    if crn_seed is not None:
        eta_bank = np.random.default_rng(crn_seed).normal(0, 1, (n_sim, nit))

    def distance(x, W):
        x = np.clip(x, LO, HI)   # guard for unbounded methods
        ml = np.empty((n_sim, 6))
        for i in range(n_sim):
            eta = eta_bank[i] if eta_bank is not None else np.random.normal(0, 1, nit)
            _p, r = chiarella_model(eta, x[0], x[1], x[2], x[3], x[4], x[5])
            ml[i] = compute_moments(r)
        diff = (np.mean(ml, axis=0) - moments_target) / scale
        return float(diff @ W @ diff)

    return distance


def run_multistart(method, dist_fn, W, n_start=10, seed=123, maxiter=200):
    """Run one scipy method from n_start random points; return summary metrics."""
    rng = np.random.default_rng(seed)
    funs, succ, t0 = [], [], time.time()
    for _ in range(n_start):
        x0 = LO + rng.random(6) * (HI - LO)
        try:
            kw = dict(args=(W,), method=method, options={'maxiter': maxiter})
            if method in METHODS_BOUNDED:
                kw['bounds'] = BOUNDS
            res = minimize(dist_fn, x0, **kw)
            funs.append(res.fun)
            succ.append(bool(res.success))
        except Exception:
            funs.append(np.inf)
            succ.append(False)
    funs = np.array(funs, dtype=float)
    return {"conv": float(np.mean(succ)), "best": float(np.nanmin(funs)),
            "med": float(np.nanmedian(funs)), "time": time.time() - t0}


def run_differential_evolution(dist_fn, W, n_start=10, maxiter=15, popsize=8):
    """Global optimiser baseline, same summary metrics as run_multistart."""
    funs, succ, t0 = [], [], time.time()
    for s in range(n_start):
        de = differential_evolution(dist_fn, BOUNDS, args=(W,), maxiter=maxiter,
                                    popsize=popsize, tol=0.01, seed=s, polish=True,
                                    workers=1, updating='immediate')
        funs.append(de.fun)
        succ.append(bool(de.success))
    funs = np.array(funs)
    return {"conv": float(np.mean(succ)), "best": float(np.nanmin(funs)),
            "med": float(np.nanmedian(funs)), "time": time.time() - t0}


def select_method(bench):
    """Convergence rate first; among the best, lowest median loss."""
    max_conv = max(v['conv'] for v in bench.values())
    finalists = {k: v for k, v in bench.items() if v['conv'] >= max_conv - 1e-9}
    winner = min(finalists, key=lambda k: finalists[k]['med'])
    return winner, max_conv, finalists


def gmm_optimal(moments_target, scale, method='COBYQA', n_iter=4, n_start=5,
                n_cov=60, tol=1e-5, seed=None):
    """Iterative optimal GMM: W0 = I, then W = pinv(V + 1e-6 I) where V is the
    covariance of normalised moments simulated at the current estimate.

    Returns xi_hat, final loss, success flag, final W, history.
    """
    rng = np.random.default_rng(seed)
    dist_fn = make_distance(moments_target, scale)
    W = np.eye(6)
    xi_prev = np.zeros(6)
    history = []

    for _ in range(n_iter):
        # Step 1: minimise D_W by multi-start
        best_fun, best_x, best_ok = np.inf, None, False
        for _ in range(n_start):
            x0 = LO + rng.random(6) * (HI - LO)
            try:
                res = minimize(dist_fn, x0, args=(W,), method=method,
                               bounds=BOUNDS if method in METHODS_BOUNDED else None,
                               options={'maxiter': 250})
                if res.fun < best_fun:
                    best_fun, best_x, best_ok = res.fun, res.x, bool(res.success)
            except Exception:
                continue
        xi_hat = np.clip(best_x, LO, HI)

        # Step 2: update W = V^{-1} on normalised moments (ridge + pinv for stability)
        runs = np.empty((n_cov, 6))
        for k in range(n_cov):
            eta = rng.normal(0, 1, NIT_SIM)
            _p, rr = chiarella_model(eta, *xi_hat)
            runs[k] = compute_moments(rr) / scale
        V = np.cov(runs.T)
        W = np.linalg.pinv(V + 1e-6 * np.eye(6))

        history.append((xi_hat.copy(), best_fun, best_ok))
        if np.linalg.norm(xi_hat - xi_prev) < tol:
            break
        xi_prev = xi_hat

    return xi_hat, best_fun, best_ok, W, history

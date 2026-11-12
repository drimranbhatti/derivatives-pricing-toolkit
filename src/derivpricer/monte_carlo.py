"""Monte Carlo pricing under geometric Brownian motion.

All pricers return ``MCResult(price, std_error)`` so estimates can be reported
with a confidence interval.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MCResult:
    price: float
    std_error: float

    @property
    def ci95(self):
        return (self.price - 1.96 * self.std_error, self.price + 1.96 * self.std_error)


def _summary(discounted_payoffs: np.ndarray) -> MCResult:
    n = discounted_payoffs.size
    return MCResult(float(discounted_payoffs.mean()), float(discounted_payoffs.std(ddof=1) / np.sqrt(n)))


def gbm_paths(S0, r, sigma, T, n_steps, n_paths, q=0.0, antithetic=True, seed=None):
    """Simulate GBM paths, shape (n_steps + 1, n_paths). Exact log-Euler scheme."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    half = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal((n_steps, half))
    if antithetic:
        z = np.concatenate([z, -z], axis=1)
    increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.vstack([np.zeros(z.shape[1]), np.cumsum(increments, axis=0)])
    return S0 * np.exp(log_paths)


def _payoff(ST, K, option):
    if option == "call":
        return np.maximum(ST - K, 0.0)
    if option == "put":
        return np.maximum(K - ST, 0.0)
    raise ValueError("option must be 'call' or 'put'")


def european(S0, K, T, r, sigma, option="call", n_paths=200_000, q=0.0, seed=42) -> MCResult:
    """European option by terminal-value sampling (one step is exact under GBM)."""
    ST = gbm_paths(S0, r, sigma, T, 1, n_paths, q, seed=seed)[-1]
    return _summary(np.exp(-r * T) * _payoff(ST, K, option))


def asian_arithmetic(S0, K, T, r, sigma, option="call", n_steps=252, n_paths=100_000, q=0.0, seed=42) -> MCResult:
    """Arithmetic-average-price Asian option (average over monitoring dates excl. t=0)."""
    S = gbm_paths(S0, r, sigma, T, n_steps, n_paths, q, seed=seed)
    return _summary(np.exp(-r * T) * _payoff(S[1:].mean(axis=0), K, option))


def barrier(S0, K, H, T, r, sigma, option="call", kind="up-and-in",
            n_steps=252, n_paths=100_000, q=0.0, seed=42, paths=None) -> MCResult:
    """Discretely monitored knock-in / knock-out barrier option.

    kind: 'up-and-in', 'up-and-out', 'down-and-in', 'down-and-out'.
    ``paths`` may be supplied (e.g. Heston or Merton paths) to price under another model.
    """
    S = paths if paths is not None else gbm_paths(S0, r, sigma, T, n_steps, n_paths, q, seed=seed)
    direction, knock = kind.split("-and-")
    hit = (S.max(axis=0) >= H) if direction == "up" else (S.min(axis=0) <= H)
    alive = hit if knock == "in" else ~hit
    return _summary(np.exp(-r * T) * _payoff(S[-1], K, option) * alive)


def american_lsm(S0, K, T, r, sigma, option="put", n_steps=50, n_paths=100_000,
                 degree=3, q=0.0, seed=42, paths=None) -> MCResult:
    """Longstaff-Schwartz least-squares Monte Carlo for American options.

    Regresses continuation values on in-the-money paths only (as in the original
    paper) using a polynomial basis in moneyness.
    """
    S = paths if paths is not None else gbm_paths(S0, r, sigma, T, n_steps, n_paths, q, seed=seed)
    n_steps = S.shape[0] - 1
    dt = T / n_steps
    disc = np.exp(-r * dt)
    cash = _payoff(S[-1], K, option)
    for t in range(n_steps - 1, 0, -1):
        cash *= disc
        exercise = _payoff(S[t], K, option)
        itm = exercise > 0
        if itm.sum() > degree + 1:
            x = S[t, itm] / K
            coef = np.polynomial.polynomial.polyfit(x, cash[itm], degree)
            continuation = np.polynomial.polynomial.polyval(x, coef)
            ex_now = exercise[itm] > continuation
            idx = np.flatnonzero(itm)[ex_now]
            cash[idx] = exercise[idx]
    cash *= disc
    # Exercise at t=0 is allowed if immediately optimal
    value = max(cash.mean(), float(_payoff(np.array([S0]), K, option)[0]))
    return MCResult(float(value), float(cash.std(ddof=1) / np.sqrt(cash.size)))


def bump_greeks(pricer, S0, h=0.01, **kwargs):
    """Finite-difference delta & gamma with common random numbers.

    ``pricer`` must accept S0 as its first argument and return an MCResult or float.
    """
    up, mid, dn = (_as_float(pricer(S0 * (1 + k * h), **kwargs)) for k in (1, 0, -1))
    dS = S0 * h
    return (up - dn) / (2 * dS), (up - 2 * mid + dn) / dS**2


def _as_float(x):
    return float(x.price) if hasattr(x, "price") else float(x)

"""Heston (1993) stochastic-volatility model.

    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW^S_t
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW^v_t,   d<W^S, W^v> = rho dt
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._fourier import lewis_call, put_from_call
from .monte_carlo import MCResult, _payoff, _summary


@dataclass(frozen=True)
class HestonParams:
    v0: float      # initial variance
    kappa: float   # mean-reversion speed
    theta: float   # long-run variance
    xi: float      # vol of vol
    rho: float     # spot/vol correlation

    @property
    def feller(self) -> bool:
        """True if 2*kappa*theta > xi^2 (variance stays strictly positive)."""
        return 2 * self.kappa * self.theta > self.xi**2


def char_func(u, T, p: HestonParams):
    """CF of ln(S_T/F_T) - 'little Heston trap' form (Albrecher et al., 2007)."""
    iu = 1j * u
    b = p.kappa - p.rho * p.xi * iu
    d = np.sqrt(b**2 + p.xi**2 * (iu + u**2))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    C = p.kappa * p.theta / p.xi**2 * ((b - d) * T - 2.0 * np.log((1 - g * e) / (1 - g)))
    D = (b - d) / p.xi**2 * (1 - e) / (1 - g * e)
    return np.exp(C + D * p.v0)


def price(S0, K, T, r, p: HestonParams, option="call", q=0.0):
    """Semi-analytic European price (vectorised over K)."""
    call = lewis_call(lambda u: char_func(u, T, p), S0, K, T, r, q)
    return call if option == "call" else put_from_call(call, S0, K, T, r, q)


def simulate(S0, T, r, p: HestonParams, n_steps=252, n_paths=100_000, q=0.0, seed=42, return_var=False):
    """Full-truncation Euler scheme (Lord et al., 2010). Returns paths (n_steps+1, n_paths)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    S = np.empty((n_steps + 1, n_paths)); S[0] = S0
    v = np.empty_like(S); v[0] = p.v0
    sq = np.sqrt(1 - p.rho**2)
    for t in range(1, n_steps + 1):
        z1 = rng.standard_normal(n_paths)
        z2 = p.rho * z1 + sq * rng.standard_normal(n_paths)
        vp = np.maximum(v[t - 1], 0.0)
        v[t] = v[t - 1] + p.kappa * (p.theta - vp) * dt + p.xi * np.sqrt(vp * dt) * z2
        S[t] = S[t - 1] * np.exp((r - q - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1)
    return (S, np.maximum(v, 0.0)) if return_var else S


def price_mc(S0, K, T, r, p: HestonParams, option="call", n_steps=252, n_paths=100_000, q=0.0, seed=42) -> MCResult:
    S = simulate(S0, T, r, p, n_steps, n_paths, q, seed)
    return _summary(np.exp(-r * T) * _payoff(S[-1], K, option))

"""Merton (1976) jump-diffusion model.

    dS_t / S_t- = (r - q - lam * kbar) dt + sigma dW_t + (J - 1) dN_t,
    ln J ~ N(mu_j, delta^2),  N_t ~ Poisson(lam),  kbar = E[J] - 1
"""
from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np

from . import black_scholes as bs
from ._fourier import lewis_call, put_from_call
from .monte_carlo import MCResult, _payoff, _summary


@dataclass(frozen=True)
class MertonParams:
    sigma: float   # diffusion volatility
    lam: float     # jump intensity (jumps / year)
    mu_j: float    # mean log jump size
    delta: float   # std of log jump size

    @property
    def kbar(self):
        return np.exp(self.mu_j + 0.5 * self.delta**2) - 1


def jump_cf(u, T, lam, mu_j, delta):
    """Compensated log-normal compound-Poisson CF (used by Merton and Bates)."""
    kbar = np.exp(mu_j + 0.5 * delta**2) - 1
    return np.exp(lam * T * (np.exp(1j * u * mu_j - 0.5 * delta**2 * u**2) - 1 - 1j * u * kbar))


def char_func(u, T, p: MertonParams):
    """CF of ln(S_T/F_T)."""
    diffusion = np.exp(-0.5 * p.sigma**2 * T * (1j * u + u**2))
    return diffusion * jump_cf(u, T, p.lam, p.mu_j, p.delta)


def price_series(S0, K, T, r, p: MertonParams, option="call", q=0.0, n_terms=60):
    """Merton's original closed form: Poisson-weighted sum of Black-Scholes prices."""
    lam_p = p.lam * (1 + p.kbar)
    total = 0.0
    for n in range(n_terms):
        w = np.exp(-lam_p * T) * (lam_p * T) ** n / factorial(n)
        sigma_n = np.sqrt(p.sigma**2 + n * p.delta**2 / T)
        r_n = r - p.lam * p.kbar + n * np.log1p(p.kbar) / T
        total = total + w * bs.price(S0, K, T, r_n, sigma_n, option, q)
    return total


def price(S0, K, T, r, p: MertonParams, option="call", q=0.0):
    """Fourier (Lewis) price - fast and vectorised over K."""
    call = lewis_call(lambda u: char_func(u, T, p), S0, K, T, r, q)
    return call if option == "call" else put_from_call(call, S0, K, T, r, q)


def simulate(S0, T, r, p: MertonParams, n_steps=252, n_paths=100_000, q=0.0, seed=42):
    """Exact simulation on a time grid (sum of N log-normal jumps per step)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_steps, n_paths))
    n_jumps = rng.poisson(p.lam * dt, (n_steps, n_paths))
    jump = p.mu_j * n_jumps + p.delta * np.sqrt(n_jumps) * rng.standard_normal((n_steps, n_paths))
    drift = (r - q - p.lam * p.kbar - 0.5 * p.sigma**2) * dt
    log_inc = drift + p.sigma * np.sqrt(dt) * z + jump
    return S0 * np.exp(np.vstack([np.zeros(n_paths), np.cumsum(log_inc, axis=0)]))


def price_mc(S0, K, T, r, p: MertonParams, option="call", n_steps=50, n_paths=200_000, q=0.0, seed=42) -> MCResult:
    S = simulate(S0, T, r, p, n_steps, n_paths, q, seed)
    return _summary(np.exp(-r * T) * _payoff(S[-1], K, option))

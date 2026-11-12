"""Vasicek and Cox-Ingersoll-Ross (CIR) short-rate models.

    Vasicek: dr = a (b - r) dt + sigma dW
    CIR:     dr = a (b - r) dt + sigma sqrt(r) dW
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class ShortRateParams:
    r0: float
    a: float      # mean-reversion speed
    b: float      # long-run mean
    sigma: float


def vasicek_zcb(T, p: ShortRateParams):
    T = np.asarray(T, dtype=float)
    B = (1 - np.exp(-p.a * T)) / p.a
    lnA = (p.b - p.sigma**2 / (2 * p.a**2)) * (B - T) - p.sigma**2 * B**2 / (4 * p.a)
    return np.exp(lnA - B * p.r0)


def cir_zcb(T, p: ShortRateParams):
    T = np.asarray(T, dtype=float)
    h = np.sqrt(p.a**2 + 2 * p.sigma**2)
    denom = 2 * h + (p.a + h) * (np.exp(h * T) - 1)
    A = (2 * h * np.exp((p.a + h) * T / 2) / denom) ** (2 * p.a * p.b / p.sigma**2)
    B = 2 * (np.exp(h * T) - 1) / denom
    return A * np.exp(-B * p.r0)


def zero_yield(prices, T):
    """Continuously-compounded zero yield from discount factors."""
    return -np.log(prices) / np.asarray(T)


def calibrate(T, market_yields, model="cir", x0=(0.03, 0.5, 0.04, 0.05)):
    """Fit (r0, a, b, sigma) to a zero-coupon yield curve by least squares."""
    T, y = np.asarray(T, float), np.asarray(market_yields, float)
    zcb = cir_zcb if model == "cir" else vasicek_zcb

    def resid(x):
        return zero_yield(zcb(T, ShortRateParams(*x)), T) - y

    sol = least_squares(resid, x0, bounds=([1e-5, 1e-3, 1e-5, 1e-4], [0.25, 10, 0.25, 0.5]))
    return ShortRateParams(*sol.x), float(np.sqrt(np.mean(sol.fun**2)))


def simulate_cir(p: ShortRateParams, T, n_steps=252, n_paths=10_000, seed=42):
    """Full-truncation Euler paths of the CIR short rate."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    r = np.empty((n_steps + 1, n_paths)); r[0] = p.r0
    for t in range(1, n_steps + 1):
        rp = np.maximum(r[t - 1], 0.0)
        r[t] = r[t - 1] + p.a * (p.b - rp) * dt + p.sigma * np.sqrt(rp * dt) * rng.standard_normal(n_paths)
    return np.maximum(r, 0.0)

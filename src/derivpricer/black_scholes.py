"""Black-Scholes-Merton closed-form pricing, Greeks and implied volatility."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q=0.0):
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def price(S, K, T, r, sigma, option="call", q=0.0):
    """European option price under Black-Scholes-Merton (vectorised).

    Parameters
    ----------
    S, K : spot and strike
    T    : time to maturity in years
    r    : continuously-compounded risk-free rate
    sigma: volatility
    option: "call" or "put"
    q    : continuous dividend yield
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_s, disc_k = S * np.exp(-q * T), K * np.exp(-r * T)
    if option == "call":
        return disc_s * norm.cdf(d1) - disc_k * norm.cdf(d2)
    if option == "put":
        return disc_k * norm.cdf(-d2) - disc_s * norm.cdf(-d1)
    raise ValueError("option must be 'call' or 'put'")


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float   # per 1.00 change in volatility
    theta: float  # per year
    rho: float    # per 1.00 change in rate


def greeks(S, K, T, r, sigma, option="call", q=0.0) -> Greeks:
    """Analytic Black-Scholes Greeks."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    pdf_d1 = norm.pdf(d1)
    eq, er = np.exp(-q * T), np.exp(-r * T)
    gamma = eq * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * eq * pdf_d1 * np.sqrt(T)
    common_theta = -S * eq * pdf_d1 * sigma / (2 * np.sqrt(T))
    if option == "call":
        delta = eq * norm.cdf(d1)
        theta = common_theta - r * K * er * norm.cdf(d2) + q * S * eq * norm.cdf(d1)
        rho = K * T * er * norm.cdf(d2)
    elif option == "put":
        delta = -eq * norm.cdf(-d1)
        theta = common_theta + r * K * er * norm.cdf(-d2) - q * S * eq * norm.cdf(-d1)
        rho = -K * T * er * norm.cdf(-d2)
    else:
        raise ValueError("option must be 'call' or 'put'")
    return Greeks(delta, gamma, vega, theta, rho)


def implied_vol(market_price, S, K, T, r, option="call", q=0.0, lo=1e-4, hi=5.0):
    """Implied volatility by Brent root-finding. Returns NaN if no solution exists."""
    f = lambda s: price(S, K, T, r, s, option, q) - market_price
    try:
        return brentq(f, lo, hi, xtol=1e-10)
    except ValueError:
        return np.nan


def put_call_parity_gap(call, put, S, K, T, r, q=0.0):
    """C - P - (S e^{-qT} - K e^{-rT}); zero when parity holds."""
    return call - put - (S * np.exp(-q * T) - K * np.exp(-r * T))

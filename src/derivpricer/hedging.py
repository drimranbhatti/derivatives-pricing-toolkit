"""Discrete delta hedging of a short European option under geometric Brownian motion.

The writer sells one option at the Black-Scholes price computed with `sigma_price`,
then rebalances a stock position to the Black-Scholes delta (computed with
`sigma_hedge`) `n_rebal` times until expiry, financing at the risk-free rate.
The stock actually follows GBM with volatility `sigma_true`. The terminal
profit and loss of the hedged book is returned per path.
"""
from __future__ import annotations

import numpy as np

from . import black_scholes as bs


def delta_hedge_pnl(S0, K, T, r, sigma_true, n_rebal, n_paths=20_000, sigma_price=None, sigma_hedge=None,
                    option="call", cost=0.0, mu=None, seed=0):
    """Simulate the P&L of selling an option and delta hedging it.

    cost : proportional transaction cost on the value of shares traded
    mu   : real-world drift of the stock (defaults to r)
    Returns an array of terminal P&L (positive = profit to the option writer).
    """
    sigma_price = sigma_true if sigma_price is None else sigma_price
    sigma_hedge = sigma_price if sigma_hedge is None else sigma_hedge
    mu = r if mu is None else mu
    rng = np.random.default_rng(seed)
    dt = T / n_rebal
    S = np.full(n_paths, float(S0))
    premium = bs.price(S0, K, T, r, sigma_price, option)
    delta = bs.greeks(S0, K, T, r, sigma_hedge, option).delta
    cash = premium - delta * S - cost * np.abs(delta) * S
    for i in range(1, n_rebal + 1):
        z = rng.standard_normal(n_paths)
        S = S * np.exp((mu - 0.5 * sigma_true ** 2) * dt + sigma_true * np.sqrt(dt) * z)
        cash = cash * np.exp(r * dt)
        tau = T - i * dt
        new_delta = bs.greeks(S, K, tau, r, sigma_hedge, option).delta if tau > 1e-12 else (
            (S > K).astype(float) if option == "call" else -(S < K).astype(float))
        trade = new_delta - delta
        cash -= trade * S + cost * np.abs(trade) * S
        delta = new_delta
    payoff = np.maximum(S - K, 0) if option == "call" else np.maximum(K - S, 0)
    return cash + delta * S - payoff


def leland_vol(sigma, cost, dt):
    """Leland (1985) volatility adjustment for a hedger who pays round-trip proportional cost `cost` and rebalances every `dt` years."""
    return sigma * np.sqrt(1 + np.sqrt(2 / np.pi) * cost / (sigma * np.sqrt(dt)))

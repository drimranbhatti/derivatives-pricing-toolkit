"""Cox-Ross-Rubinstein binomial tree for European and American options."""
from __future__ import annotations

import numpy as np


def crr_price(S, K, T, r, sigma, n_steps=500, option="call", american=False, q=0.0):
    """Price an option on a recombining CRR tree (vectorised backward induction)."""
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError("Arbitrage in tree: increase n_steps")
    disc = np.exp(-r * dt)

    j = np.arange(n_steps + 1)
    ST = S * u ** (n_steps - j) * d**j
    payoff = (lambda s: np.maximum(s - K, 0.0)) if option == "call" else (lambda s: np.maximum(K - s, 0.0))
    V = payoff(ST)
    for i in range(n_steps - 1, -1, -1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if american:
            Si = S * u ** (i - np.arange(i + 1)) * d ** np.arange(i + 1)
            V = np.maximum(V, payoff(Si))
    return float(V[0])


def early_exercise_premium(S, K, T, r, sigma, option="put", n_steps=500, q=0.0):
    """American minus European value on the same tree."""
    am = crr_price(S, K, T, r, sigma, n_steps, option, american=True, q=q)
    eu = crr_price(S, K, T, r, sigma, n_steps, option, american=False, q=q)
    return am - eu

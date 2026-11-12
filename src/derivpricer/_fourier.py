"""Shared Fourier pricing engine (Lewis 2001 formula).

Every model supplies the characteristic function of X = ln(S_T / F_T), where
F_T = S0 * exp((r - q) T) is the forward. Because X is a martingale log-return,
psi(-1j) == 1. European calls then follow from a single real integral:

    C = e^{-rT} [ F - sqrt(F K)/pi * int_0^inf Re(e^{iu ln(F/K)} psi(u - i/2)) / (u^2 + 1/4) du ]
"""
from __future__ import annotations

import numpy as np

# Fixed integration grid: vectorises over strikes and is accurate to ~1e-7
# for typical equity parameters. Increase U_MAX/N_U for very short maturities.
U_MAX, N_U = 250.0, 4000
_U = np.linspace(1e-8, U_MAX, N_U)


def lewis_call(cf, S0, K, T, r, q=0.0):
    """Call prices for an array of strikes given ``cf(u) -> psi(u)`` (complex array)."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    F = S0 * np.exp((r - q) * T)
    psi = cf(_U - 0.5j)                      # shape (N_U,)
    k = np.log(F / K)[:, None]               # shape (nK, 1)
    integrand = np.real(np.exp(1j * _U * k) * psi) / (_U**2 + 0.25)
    integral = np.trapezoid(integrand, _U, axis=1)
    calls = np.exp(-r * T) * (F - np.sqrt(F * K) / np.pi * integral)
    return calls if calls.size > 1 else float(calls[0])


def put_from_call(call, S0, K, T, r, q=0.0):
    """Put via put-call parity."""
    return call - S0 * np.exp(-q * T) + np.asarray(K) * np.exp(-r * T)

"""Bates (1996) model: Heston stochastic volatility + Merton log-normal jumps."""
from __future__ import annotations

from dataclasses import dataclass

from . import heston
from ._fourier import lewis_call, put_from_call
from .merton import jump_cf


@dataclass(frozen=True)
class BatesParams:
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    lam: float
    mu_j: float
    delta: float

    @property
    def heston(self) -> heston.HestonParams:
        return heston.HestonParams(self.v0, self.kappa, self.theta, self.xi, self.rho)


def char_func(u, T, p: BatesParams):
    return heston.char_func(u, T, p.heston) * jump_cf(u, T, p.lam, p.mu_j, p.delta)


def price(S0, K, T, r, p: BatesParams, option="call", q=0.0):
    call = lewis_call(lambda u: char_func(u, T, p), S0, K, T, r, q)
    return call if option == "call" else put_from_call(call, S0, K, T, r, q)

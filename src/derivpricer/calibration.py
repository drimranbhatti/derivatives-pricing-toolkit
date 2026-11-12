"""Least-squares calibration of Heston and Bates models to option quotes.

Quotes are passed as a DataFrame-like object with columns
``strike``, ``maturity`` (years) and ``price`` (call mid prices).
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from scipy.optimize import least_squares

from . import bates, heston


@dataclass
class CalibrationResult:
    params: object
    rmse: float
    model_prices: np.ndarray
    success: bool
    message: str


def _model_prices(model_price, S0, quotes, r, q, p):
    out = np.empty(len(quotes))
    for T, idx in quotes.groupby("maturity").indices.items():
        out[idx] = model_price(S0, quotes["strike"].to_numpy()[idx], T, r, p, "call", q)
    return out


def _calibrate(param_cls, model_price, S0, quotes, r, q, x0, lb, ub, weights, feller_penalty):
    quotes = quotes.reset_index(drop=True)
    mkt = quotes["price"].to_numpy()
    w = np.ones_like(mkt) if weights is None else np.asarray(weights)

    def residuals(x):
        p = param_cls(*x)
        res = w * (_model_prices(model_price, S0, quotes, r, q, p) - mkt)
        if feller_penalty:
            h = p.heston if hasattr(p, "heston") else p
            res = np.append(res, feller_penalty * max(0.0, h.xi**2 - 2 * h.kappa * h.theta))
        return res

    # Multi-start: the Heston objective is non-convex, so try several starting points
    starts = [x0] if np.ndim(x0[0]) == 0 else list(x0)
    sol = None
    for start in starts:
        cand = least_squares(residuals, np.clip(start, lb, ub), bounds=(lb, ub), method="trf",
                             x_scale="jac", max_nfev=500)
        if sol is None or cand.cost < sol.cost:
            sol = cand
    p = param_cls(*sol.x)
    fitted = _model_prices(model_price, S0, quotes, r, q, p)
    rmse = float(np.sqrt(np.mean((fitted - mkt) ** 2)))
    return CalibrationResult(p, rmse, fitted, bool(sol.success), sol.message)


HESTON_STARTS = [(0.04, 2.0, 0.04, 0.5, -0.6), (0.03, 8.0, 0.05, 2.0, -0.7), (0.02, 1.0, 0.08, 1.0, -0.8),
                 (0.06, 4.0, 0.03, 0.8, -0.5)]


def calibrate_heston(S0, quotes, r, q=0.0, x0=HESTON_STARTS, weights=None, feller_penalty=0.0):
    """Calibrate Heston. ``x0`` is one start (v0, kappa, theta, xi, rho) or a list of starts."""
    lb = [1e-4, 1e-3, 1e-4, 1e-3, -0.999]
    ub = [1.0, 15.0, 0.25, 3.0, 0.999]
    return _calibrate(heston.HestonParams, heston.price, S0, quotes, r, q, x0, lb, ub, weights, feller_penalty)


def calibrate_bates(S0, quotes, r, q=0.0, x0=(0.04, 2.0, 0.04, 0.5, -0.6, 0.3, -0.1, 0.1), weights=None, feller_penalty=0.0):
    """Calibrate Bates. Tip: start from a calibrated Heston solution plus jump guesses."""
    lb = [1e-4, 1e-3, 1e-4, 1e-3, -0.999, 0.0, -1.0, 1e-3]
    ub = [1.0, 15.0, 0.25, 3.0, 0.999, 3.0, 0.5, 1.0]
    return _calibrate(bates.BatesParams, bates.price, S0, quotes, r, q, x0, lb, ub, weights, feller_penalty)


def as_dict(params) -> dict:
    return {f.name: getattr(params, f.name) for f in fields(params)}

"""derivpricer - option pricing models for quantitative finance.

Modules
-------
black_scholes : closed-form prices, Greeks and implied volatility
trees         : Cox-Ross-Rubinstein binomial trees (European & American)
monte_carlo   : GBM simulation, Asian / barrier options, Longstaff-Schwartz
heston        : Heston stochastic-volatility model (semi-analytic + Monte Carlo)
merton        : Merton jump-diffusion model (series formula + Monte Carlo)
bates         : Bates (Heston + jumps) model
calibration   : least-squares calibration of Heston / Bates to option quotes
rates         : Vasicek and CIR short-rate models and curve calibration
hedging       : discrete delta-hedging simulation and the Leland cost adjustment
"""

from . import black_scholes, trees, monte_carlo, heston, merton, bates, calibration, rates, hedging

__all__ = [
    "black_scholes", "trees", "monte_carlo", "heston",
    "merton", "bates", "calibration", "rates", "hedging",
]
__version__ = "1.0.0"

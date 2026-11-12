import numpy as np
import pandas as pd
import pytest

from derivpricer import bates, black_scholes as bs, calibration, heston, merton, monte_carlo as mc, rates, trees

S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2


def test_black_scholes_reference_value():
    # Hull, Options Futures & Other Derivatives - textbook reference
    assert bs.price(42, 40, 0.5, 0.10, 0.20, "call") == pytest.approx(4.7594, abs=1e-4)
    assert bs.price(42, 40, 0.5, 0.10, 0.20, "put") == pytest.approx(0.8086, abs=1e-4)


def test_put_call_parity():
    c, p = bs.price(S0, K, T, r, sigma, "call"), bs.price(S0, K, T, r, sigma, "put")
    assert bs.put_call_parity_gap(c, p, S0, K, T, r) == pytest.approx(0, abs=1e-12)


def test_greeks_match_finite_differences():
    g, h = bs.greeks(S0, K, T, r, sigma, "call"), 1e-3
    up, mid, dn = (bs.price(S0 + k * h, K, T, r, sigma) for k in (1, 0, -1))
    assert g.delta == pytest.approx((up - dn) / (2 * h), rel=1e-6)
    assert g.gamma == pytest.approx((up - 2 * mid + dn) / h**2, rel=1e-4)
    assert g.vega == pytest.approx((bs.price(S0, K, T, r, sigma + h) - bs.price(S0, K, T, r, sigma - h)) / (2 * h), rel=1e-6)


def test_implied_vol_roundtrip():
    c = bs.price(S0, 110, T, r, 0.27)
    assert bs.implied_vol(c, S0, 110, T, r) == pytest.approx(0.27, abs=1e-8)


def test_crr_converges_and_american_put_premium():
    assert trees.crr_price(S0, K, T, r, sigma, 2000) == pytest.approx(bs.price(S0, K, T, r, sigma), abs=5e-3)
    assert trees.early_exercise_premium(S0, K, T, r, sigma, "put") > 0
    # American call on non-dividend stock = European call
    assert trees.crr_price(S0, K, T, r, sigma, 500, "call", american=True) == pytest.approx(
        trees.crr_price(S0, K, T, r, sigma, 500, "call"), abs=1e-10)


def test_monte_carlo_european_within_ci():
    res = mc.european(S0, K, T, r, sigma)
    lo, hi = res.ci95
    assert lo < bs.price(S0, K, T, r, sigma) < hi


def test_lsm_close_to_tree():
    tree = trees.crr_price(S0, K, T, r, sigma, 1000, "put", american=True)
    lsm = mc.american_lsm(S0, K, T, r, sigma, "put", n_steps=50, n_paths=100_000)
    assert lsm.price == pytest.approx(tree, abs=0.08)


def test_barrier_in_out_parity():
    kw = dict(S0=S0, K=K, H=120, T=T, r=r, sigma=sigma, seed=1, n_paths=50_000)
    vanilla = mc.barrier(kind="up-and-in", **{**kw, "H": 0.0}).price  # always knocked in
    assert mc.barrier(kind="up-and-in", **kw).price + mc.barrier(kind="up-and-out", **kw).price == pytest.approx(vanilla, rel=1e-10)


def test_heston_reduces_to_black_scholes():
    p = heston.HestonParams(v0=sigma**2, kappa=1.0, theta=sigma**2, xi=1e-4, rho=0.0)
    assert heston.price(S0, K, T, r, p) == pytest.approx(bs.price(S0, K, T, r, sigma), abs=1e-4)


def test_heston_fourier_vs_monte_carlo():
    p = heston.HestonParams(0.04, 2.0, 0.04, 0.5, -0.7)
    mcres = heston.price_mc(S0, K, T, r, p, n_paths=100_000)
    assert abs(heston.price(S0, K, T, r, p) - mcres.price) < 4 * mcres.std_error + 0.02


def test_merton_fourier_vs_series_vs_mc():
    p = merton.MertonParams(sigma=0.2, lam=0.75, mu_j=-0.5, delta=0.22)
    four, ser = merton.price(S0, K, T, r, p), merton.price_series(S0, K, T, r, p)
    assert four == pytest.approx(ser, abs=1e-4)
    mcres = merton.price_mc(S0, K, T, r, p)
    assert abs(four - mcres.price) < 4 * mcres.std_error


def test_bates_nests_heston():
    hp = heston.HestonParams(0.04, 2.0, 0.04, 0.5, -0.7)
    bp = bates.BatesParams(*vars(hp).values(), lam=0.0, mu_j=0.0, delta=0.1)
    assert bates.price(S0, K, T, r, bp) == pytest.approx(heston.price(S0, K, T, r, hp), abs=1e-10)


def test_heston_calibration_recovers_parameters():
    true = heston.HestonParams(0.05, 1.5, 0.06, 0.6, -0.65)
    rows = []
    for mat in (0.25, 0.5, 1.0, 2.0):
        strikes = np.linspace(80, 120, 9)
        for k_, px in zip(strikes, heston.price(S0, strikes, mat, r, true)):
            rows.append((k_, mat, px))
    quotes = pd.DataFrame(rows, columns=["strike", "maturity", "price"])
    res = calibration.calibrate_heston(S0, quotes, r)
    assert res.rmse < 1e-3
    assert res.params.rho == pytest.approx(true.rho, abs=0.05)


def test_short_rate_models():
    p = rates.ShortRateParams(0.03, 0.8, 0.045, 0.06)
    assert rates.cir_zcb(1e-9, p) == pytest.approx(1.0)
    Ts = np.array([0.25, 1, 2, 5, 10, 30])
    fit, rmse = rates.calibrate(Ts, rates.zero_yield(rates.cir_zcb(Ts, p), Ts))
    assert rmse < 1e-6
    assert rates.simulate_cir(p, 1.0, n_paths=100).min() >= 0

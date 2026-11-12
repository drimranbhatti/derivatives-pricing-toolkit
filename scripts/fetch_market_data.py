"""Refresh the market-data snapshots used by the notebooks (data/*.csv).

    python scripts/fetch_market_data.py

Uses bid/ask mids during market hours, otherwise last trades from the most
recent session with volume > 10.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DATA = Path(__file__).resolve().parents[1] / "data"


def option_surface(symbol="SPY", n_expiries=4, min_days=30, max_days=400):
    tk = yf.Ticker(symbol)
    hist = tk.history(period="5d")["Close"]
    spot, asof = float(hist.iloc[-1]), hist.index[-1].date()
    today = pd.Timestamp.today().normalize()
    exps = [e for e in tk.options if min_days <= (pd.Timestamp(e) - today).days <= max_days]
    exps = exps[:: max(1, len(exps) // n_expiries)][:n_expiries]
    frames = []
    for e in exps:
        c = tk.option_chain(e).calls
        live = (c.bid > 0) & (c.ask > 0)
        px = np.where(live, (c.bid + c.ask) / 2, c.lastPrice)
        fresh = (c.volume > 10) & (live | (c.lastTradeDate >= c.lastTradeDate.max() - pd.Timedelta(days=1)))
        c = c.assign(price=px)[(c.strike > 0.85 * spot) & (c.strike < 1.15 * spot) & fresh & (px > 0.10)]
        frames.append(pd.DataFrame({"expiry": e, "strike": c.strike, "maturity": (pd.Timestamp(e) - today).days / 365,
                                    "price": c.price, "volume": c.volume}))
    surf = pd.concat(frames, ignore_index=True)
    # drop quotes below intrinsic value (no valid implied volatility) - usually stale prints
    intrinsic = np.maximum(spot - surf.strike * np.exp(-0.04 * surf.maturity), 0)
    surf = surf[surf.price > intrinsic * 1.0005].reset_index(drop=True)
    surf.insert(0, "spot", spot)
    surf.insert(0, "asof", asof)
    return surf


def treasury_curve():
    tick = {"^IRX": 0.25, "^FVX": 5, "^TNX": 10, "^TYX": 30}
    rows = [(T, float(yf.Ticker(t).history(period="5d")["Close"].dropna().iloc[-1]) / 100) for t, T in tick.items()]
    df = pd.DataFrame(rows, columns=["maturity", "yield"]).sort_values("maturity")
    df.insert(0, "asof", pd.Timestamp.today().date())
    return df


def spy_vix_history(start="2006-01-01", end="2026-11-01"):
    px = yf.download(["SPY", "^VIX", "^IRX"], start=start, end=end, auto_adjust=True, progress=False)["Close"]
    px.columns = ["spy" if c == "SPY" else "vix" if c == "^VIX" else "tbill" for c in px.columns]
    return px.dropna().round(4)


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    option_surface().to_csv(DATA / "spy_call_surface.csv", index=False)
    treasury_curve().to_csv(DATA / "treasury_curve.csv", index=False)
    spy_vix_history().to_csv(DATA / "spy_vix_daily.csv")
    print("Snapshots written to", DATA)

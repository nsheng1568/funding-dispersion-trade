"""
Live signal pipeline for the funding dispersion trade.

Incrementally refreshes funding data, then computes the composite signal:
  - EWMA at 168h and 72h (top-2 half-lives from in-sample calibration)
  - Direct regression: rolling OLS of past cumulative funding on current rate

Output: ranked signal table + top long/short candidates.

Usage:
    python -m src.models.signal
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import SIGNAL_HORIZON_DAYS
from src.data import hl_client
from src.data.loader import load_betas
from src.models.betas import estimate_betas

DATA_DIR = Path("data")

BEST_HLS         = [168, 72]   # EWMA half-lives calibrated in notebook 03 (hours)
ROLL_WINDOW      = 24 * 30     # 30-day rolling OLS window (hours)
HORIZON_H        = SIGNAL_HORIZON_DAYS * 24
TARGET_ANN_VOL   = 0.40
ANN_FACTOR       = np.sqrt(3 * 365)   # 8h periods per year
MAX_LEVERAGE     = 4.0


def _direct_forecast_now(funding_col: pd.Series, target_col: pd.Series) -> float:
    """
    Fit OLS on the most recent ROLL_WINDOW observations where the causal target
    is non-NaN, then return the prediction at the current (last) time step.

    target_col must already be causally lagged (see compute_signal).
    """
    # Only need enough history to fill the regression window
    lookback = ROLL_WINDOW + HORIZON_H + 10
    f = funding_col.iloc[-lookback:].values
    t = target_col.iloc[-lookback:].values

    x = f[:-1]   # features up to t-1
    y = t[:-1]   # targets up to t-1 (causal)
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 10:
        return np.nan

    # Use only the most recent ROLL_WINDOW valid pairs
    valid_x = x[mask][-ROLL_WINDOW:]
    valid_y = y[mask][-ROLL_WINDOW:]
    b = np.cov(valid_x, valid_y)[0, 1] / (np.var(valid_x) + 1e-12)
    a = valid_y.mean() - b * valid_x.mean()
    return a + b * f[-1]


def compute_signal(funding: pd.DataFrame, betas: pd.DataFrame) -> pd.Series:
    """
    Compute the current composite risk-adjusted signal for each coin.

    Higher value → coin expected to have high future funding → SHORT candidate.
    Lower  value → coin expected to have low/negative funding → LONG candidate.

    Parameters
    ----------
    funding : hourly funding rates, index=timestamp, columns=coins
    betas   : DataFrame with columns 'beta', 'idio_vol', indexed by coin

    Returns
    -------
    pd.Series of signal values, indexed by coin, NaN coins dropped.
    """
    # EWMA components — causal, computed on full history
    ewma_now = {hl: funding.ewm(halflife=hl).mean().iloc[-1] for hl in BEST_HLS}

    # Direct regression target: causally-lagged realized cumulative funding.
    # realized_cumulative[t] = sum(funding[t+2 .. t+HORIZON_H+1]) — requires future data.
    # Shift back by HORIZON_H+1 so target[T] = realized_cumulative[T-HORIZON_H-1],
    # which is fully observable at T (all required future funding has settled).
    realized_cumulative = funding.shift(-(HORIZON_H + 1)).rolling(HORIZON_H).sum()
    target = realized_cumulative.shift(HORIZON_H + 1)

    direct_now = pd.Series({
        coin: _direct_forecast_now(funding[coin], target[coin])
        for coin in funding.columns
    })

    # Composite: raw average (no z-scoring) — values are in funding-rate units
    composite_now = sum(ewma_now[hl] for hl in BEST_HLS) + direct_now
    composite_now /= (len(BEST_HLS) + 1)

    idio_vol = betas["idio_vol"].reindex(composite_now.index)
    signal = composite_now / idio_vol
    return signal.dropna()


def size_position(
    long_coin: str, short_coin: str, betas: pd.DataFrame
) -> tuple[float, float]:
    """Beta-neutral, vol-targeted position sizes as fractions of equity."""
    beta_A = betas.loc[long_coin,  "beta"]
    beta_B = betas.loc[short_coin, "beta"]
    idio_A = betas.loc[long_coin,  "idio_vol"]
    idio_B = betas.loc[short_coin, "idio_vol"]

    ratio = abs(beta_A / beta_B) if abs(beta_B) > 1e-8 else 1.0
    idio_combined = np.sqrt(idio_A ** 2 + ratio ** 2 * idio_B ** 2)
    L = TARGET_ANN_VOL / (ANN_FACTOR * idio_combined)
    S = L * ratio

    max_leg = max(L, S)
    if max_leg > MAX_LEVERAGE:
        scale = MAX_LEVERAGE / max_leg
        L, S = L * scale, S * scale

    return L, S


def refresh_funding(coins: list[str]) -> pd.DataFrame:
    """Fetch any hourly funding records newer than what's in the parquet."""
    path = DATA_DIR / "funding_rates.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run pipeline.py first")

    existing = pd.read_parquet(path)
    last_ts  = existing.index.max()
    start_ms = int(last_ts.timestamp() * 1000) + 1

    frames = {col: existing[col] for col in existing.columns}
    for coin in coins:
        if coin not in existing.columns:
            continue
        try:
            df = hl_client.get_funding_history(coin, start_ms)
            if not df.empty:
                new_rows = df["fundingRate"]
                frames[coin] = pd.concat([existing[coin], new_rows])
            time.sleep(0.2)
        except Exception as e:
            print(f"  Warning: could not refresh {coin}: {e}")

    updated = pd.DataFrame(frames)
    updated.index = updated.index.floor("h")
    updated = updated[~updated.index.duplicated(keep="last")].sort_index()
    updated.to_parquet(path)
    return updated


def run(refresh_betas: bool = False) -> pd.Series:
    if refresh_betas:
        print("Estimating betas from current prices...")
        prices_all = pd.read_parquet(DATA_DIR / "prices.parquet")
        betas = estimate_betas(prices_all)
        betas.to_parquet(DATA_DIR / "coin_betas.parquet")
        betas = betas[(betas["beta"] > 0) & (betas["idio_vol"] > 0)]
    else:
        print("Loading betas...")
        betas = load_betas()

    valid_coins = list(betas.index)

    print(f"Refreshing funding data for {len(valid_coins)} coins...")
    raw = refresh_funding(valid_coins)
    funding = raw[[c for c in valid_coins if c in raw.columns]]

    print("Computing composite signal...")
    signal = compute_signal(funding, betas)
    signal = signal.sort_values()

    long_coin  = signal.index[0]
    short_coin = signal.index[-1]
    L, S = size_position(long_coin, short_coin, betas)
    spread = signal.iloc[-1] - signal.iloc[0]

    print(f"\nLatest funding timestamp: {funding.index.max()}")
    print(f"\n{'Rank':<5} {'Coin':<8} {'Signal':>10}  Side")
    print("-" * 38)
    n = len(signal)
    for rank, (coin, val) in enumerate(signal.items(), 1):
        if rank == 1:
            side = f"LONG  → {L:.2f}x"
        elif rank == n:
            side = f"SHORT → {S:.2f}x"
        else:
            side = ""
        print(f"{rank:<5} {coin:<8} {val:>10.4f}  {side}")

    print(f"\nSignal spread: {spread:.4f}")
    return signal


if __name__ == "__main__":
    run()

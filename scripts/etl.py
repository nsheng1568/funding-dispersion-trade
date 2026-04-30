"""
ETL snapshot script — run by GitHub Actions every 24 hours.

Appends one row per run to:
  data/history/equity.csv    — timestamp, equity
  data/history/positions.csv — timestamp, coin, notional
  data/history/signals.csv   — timestamp, coin, signal
  data/history/target.csv    — timestamp, long_coin, short_coin, long_usd, short_usd, spread

Also expects data/coin_betas.parquet to exist (committed to repo).
Run `python -m src.models.betas` locally and commit the output once.

No private key required — all reads are from the Hyperliquid public API.
"""

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Make repo root importable when invoked as `python scripts/etl.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.data import hl_client
from src.data.loader import load_betas
from src.models.signal import compute_signal, size_position
from src.trading.executor import get_equity, get_positions, make_info_client

DATA_DIR    = Path("data")
HISTORY_DIR = DATA_DIR / "history"
LOOKBACK_DAYS = 40   # enough for EWMA half-life warm-up (168h × 3 = 504h < 960h)


def _append(path: Path, df: pd.DataFrame) -> None:
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def fetch_recent_funding(coins: list[str]) -> pd.DataFrame:
    start_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).timestamp() * 1000
    )
    frames = {}
    for coin in coins:
        try:
            df = hl_client.get_funding_history(coin, start_ms)
            if not df.empty:
                frames[coin] = df["fundingRate"]
            time.sleep(0.05)
        except Exception as e:
            print(f"  Warning: could not fetch {coin}: {e}")
    if not frames:
        return pd.DataFrame()
    funding = pd.DataFrame(frames)
    funding.index = funding.index.floor("h")
    return funding[~funding.index.duplicated(keep="last")].sort_index()


def run() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    ts  = now.isoformat()
    print(f"ETL run: {ts}")

    # ------------------------------------------------------------------ account
    info    = make_info_client()
    equity  = get_equity(info)
    current = get_positions(info)
    print(f"  Equity: ${equity:,.2f}  Positions: {list(current.keys()) or 'none'}")

    _append(HISTORY_DIR / "equity.csv",
            pd.DataFrame([{"timestamp": ts, "equity": equity}]))

    if current:
        rows = [{"timestamp": ts, "coin": c, "notional": n} for c, n in current.items()]
    else:
        rows = [{"timestamp": ts, "coin": None, "notional": None}]
    _append(HISTORY_DIR / "positions.csv", pd.DataFrame(rows))

    # ------------------------------------------------------------------ signal
    betas_path = DATA_DIR / "coin_betas.parquet"
    if not betas_path.exists():
        print("  Skipping signal — data/coin_betas.parquet not found. "
              "Run `python -m src.models.betas` locally and commit the file.")
        return

    betas       = load_betas()
    valid_coins = list(betas.index)
    print(f"  Fetching {LOOKBACK_DAYS}-day funding for {len(valid_coins)} coins...")
    funding = fetch_recent_funding(valid_coins)

    if funding.empty or len(funding) < 100:
        print("  Skipping signal — not enough funding data.")
        return

    funding = funding[[c for c in valid_coins if c in funding.columns]]
    signal  = compute_signal(funding, betas)

    if signal.empty:
        print("  Signal is empty after compute.")
        return

    sig_rows = [{"timestamp": ts, "coin": c, "signal": float(v)}
                for c, v in signal.items()]
    _append(HISTORY_DIR / "signals.csv", pd.DataFrame(sig_rows))
    print(f"  Signal computed for {len(sig_rows)} coins.")

    # ------------------------------------------------------------------ target
    signal_sorted = signal.sort_values()
    long_coin     = signal_sorted.index[0]
    short_coin    = signal_sorted.index[-1]
    spread        = float(signal_sorted.iloc[-1] - signal_sorted.iloc[0])
    L, S          = size_position(long_coin, short_coin, betas)
    long_usd      = float(L * equity)
    short_usd     = float(S * equity)
    print(f"  Target: LONG {long_coin}  SHORT {short_coin}  spread={spread:.4f}")

    _append(HISTORY_DIR / "target.csv", pd.DataFrame([{
        "timestamp":  ts,
        "long_coin":  long_coin,
        "short_coin": short_coin,
        "long_usd":   round(long_usd, 2),
        "short_usd":  round(short_usd, 2),
        "spread":     round(spread, 4),
    }]))


if __name__ == "__main__":
    run()

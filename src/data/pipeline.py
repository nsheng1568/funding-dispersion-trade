"""
Fetches and stores historical funding rates and prices for the HL universe.

Outputs (written to data/):
  universe.parquet       - coin metadata + current market context
  funding_rates.parquet  - wide: index=timestamp, columns=coins, values=hourly funding rate
  prices.parquet         - wide: index=timestamp, columns=coins, values=close price (8h candles)

Re-running is safe: already-fetched coins are skipped.
"""

import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from src.data import hl_client, coingecko_client
from src.constants import INSAMPLE_START

DATA_DIR = Path("data")
HISTORY_DAYS = 730
MIN_24H_VOL_USD = 1_000_000
MAX_COINS = 30
CANDLE_INTERVAL = "8h"
INTER_COIN_SLEEP = 1.0  # seconds between coins to avoid 429s


def _start_ms(days_ago: int) -> int:
    return int((datetime.now(timezone.utc) - timedelta(days=days_ago)).timestamp() * 1000)


def build_universe() -> pd.DataFrame:
    """Filter HL universe to liquid coins, cross-referenced with CoinGecko."""
    print("Fetching HL universe...")
    hl = hl_client.get_universe()
    hl = hl[hl["dayNtlVlm"] >= MIN_24H_VOL_USD].copy()

    print("Fetching CoinGecko markets...")
    try:
        cg = coingecko_client.get_markets()
        cg_symbols = set(cg["symbol"].str.upper())
        hl = hl[hl.index.str.upper().isin(cg_symbols)]
    except Exception as e:
        print(f"  CoinGecko unavailable ({e}), skipping cross-reference filter")

    hl = hl.sort_values("dayNtlVlm", ascending=False).head(MAX_COINS)
    print(f"  {len(hl)} coins selected: {list(hl.index)}")
    return hl


def fetch_funding(coins: list[str], existing: pd.DataFrame) -> pd.DataFrame:
    """Fetch hourly funding rate history, skipping coins already in existing."""
    start = _start_ms(HISTORY_DAYS)
    frames = {col: existing[col] for col in existing.columns if col in coins}
    todo = [c for c in coins if c not in frames]

    for i, coin in enumerate(todo):
        print(f"  Funding [{i+1}/{len(todo)}] {coin}")
        try:
            df = hl_client.get_funding_history(coin, start)
            frames[coin] = df["fundingRate"]
        except Exception as e:
            print(f"    ERROR: {e}")
        time.sleep(INTER_COIN_SLEEP)

    wide = pd.DataFrame({c: frames[c] for c in coins if c in frames})
    wide.index.name = "time"
    return wide


def fetch_prices(coins: list[str], existing: pd.DataFrame) -> pd.DataFrame:
    """Fetch 8h OHLCV candles, skipping coins already in existing."""
    start = _start_ms(HISTORY_DAYS)
    end = int(datetime.now(timezone.utc).timestamp() * 1000)
    frames = {col: existing[col] for col in existing.columns if col in coins}
    todo = [c for c in coins if c not in frames]

    for i, coin in enumerate(todo):
        print(f"  Prices  [{i+1}/{len(todo)}] {coin}")
        try:
            df = hl_client.get_candles(coin, CANDLE_INTERVAL, start, end)
            frames[coin] = df["close"]
        except Exception as e:
            print(f"    ERROR: {e}")
        time.sleep(INTER_COIN_SLEEP)

    wide = pd.DataFrame({c: frames[c] for c in coins if c in frames})
    wide.index.name = "time"
    return wide


def _load_existing(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def run():
    DATA_DIR.mkdir(exist_ok=True)

    universe = build_universe()
    universe.to_parquet(DATA_DIR / "universe.parquet")
    print(f"Saved {DATA_DIR}/universe.parquet")

    coins = list(universe.index)

    print("\nFetching funding history...")
    funding = fetch_funding(coins, _load_existing(DATA_DIR / "funding_rates.parquet"))

    # Drop coins whose history doesn't reach back to INSAMPLE_START — survivorship filter.
    # Coins listed or liquid only after May 2024 would otherwise inflate backtest quality.
    insample_ts = pd.Timestamp(INSAMPLE_START)
    early_enough = [
        c for c in funding.columns
        if funding[c].first_valid_index() is not None
        and funding[c].first_valid_index() <= insample_ts
    ]
    dropped = [c for c in funding.columns if c not in early_enough]
    if dropped:
        print(f"  Dropped (listed after INSAMPLE_START): {dropped}")
    funding = funding[early_enough]

    funding.to_parquet(DATA_DIR / "funding_rates.parquet")
    print(f"Saved {DATA_DIR}/funding_rates.parquet  shape={funding.shape}")

    print("\nFetching price history...")
    prices = fetch_prices(early_enough, _load_existing(DATA_DIR / "prices.parquet"))
    prices.to_parquet(DATA_DIR / "prices.parquet")
    print(f"Saved {DATA_DIR}/prices.parquet  shape={prices.shape}")

    print("\nDone.")
    print(f"  Funding: {funding.index.min()} → {funding.index.max()}")
    print(f"  Prices:  {prices.index.min()} → {prices.index.max()}")
    print(f"  Coins:   {list(funding.columns)}")


if __name__ == "__main__":
    run()

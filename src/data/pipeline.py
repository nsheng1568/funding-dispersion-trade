"""
Fetches and stores historical funding rates and prices for the HL universe.

Outputs (written to data/):
  universe.parquet       - coin metadata + current market context
  funding_rates.parquet  - wide: index=timestamp, columns=coins, values=8h funding rate
  prices.parquet         - wide: index=timestamp, columns=coins, values=close price (8h candles)
"""

import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from src.data import hl_client, coingecko_client

DATA_DIR = Path("data")
HISTORY_DAYS = 90
MIN_24H_VOL_USD = 1_000_000
MAX_COINS = 30
CANDLE_INTERVAL = "8h"


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


def fetch_funding(coins: list[str]) -> pd.DataFrame:
    """Fetch 8h funding rate history for each coin and pivot to wide format."""
    start = _start_ms(HISTORY_DAYS)
    frames = {}
    for i, coin in enumerate(coins):
        print(f"  Funding [{i+1}/{len(coins)}] {coin}")
        try:
            df = hl_client.get_funding_history(coin, start)
            frames[coin] = df["fundingRate"]
        except Exception as e:
            print(f"    ERROR: {e}")
        time.sleep(0.1)
    wide = pd.DataFrame(frames)
    wide.index.name = "time"
    return wide


def fetch_prices(coins: list[str]) -> pd.DataFrame:
    """Fetch 8h OHLCV candles for each coin and pivot close prices to wide format."""
    start = _start_ms(HISTORY_DAYS)
    end = int(datetime.now(timezone.utc).timestamp() * 1000)
    frames = {}
    for i, coin in enumerate(coins):
        print(f"  Prices  [{i+1}/{len(coins)}] {coin}")
        try:
            df = hl_client.get_candles(coin, CANDLE_INTERVAL, start, end)
            frames[coin] = df["close"]
        except Exception as e:
            print(f"    ERROR: {e}")
        time.sleep(0.1)
    wide = pd.DataFrame(frames)
    wide.index.name = "time"
    return wide


def run():
    DATA_DIR.mkdir(exist_ok=True)

    universe = build_universe()
    universe.to_parquet(DATA_DIR / "universe.parquet")
    print(f"Saved {DATA_DIR}/universe.parquet")

    coins = list(universe.index)

    print("\nFetching funding history...")
    funding = fetch_funding(coins)
    funding.to_parquet(DATA_DIR / "funding_rates.parquet")
    print(f"Saved {DATA_DIR}/funding_rates.parquet  shape={funding.shape}")

    print("\nFetching price history...")
    prices = fetch_prices(coins)
    prices.to_parquet(DATA_DIR / "prices.parquet")
    print(f"Saved {DATA_DIR}/prices.parquet  shape={prices.shape}")

    print("\nDone.")
    print(f"  Funding: {funding.index.min()} → {funding.index.max()}")
    print(f"  Prices:  {prices.index.min()} → {prices.index.max()}")
    print(f"  Coins:   {list(funding.columns)}")


if __name__ == "__main__":
    run()

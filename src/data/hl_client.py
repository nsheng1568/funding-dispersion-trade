import requests
import pandas as pd
import time

_HL_URL = "https://api.hyperliquid.xyz/info"


def _post(payload: dict, retries: int = 5) -> any:
    delay = 2.0
    for _ in range(retries):
        resp = requests.post(_HL_URL, json=payload, timeout=15)
        if resp.status_code == 429:
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()


def get_universe() -> pd.DataFrame:
    """Returns all perps with current market context (funding, OI, volume, price)."""
    meta, ctxs = _post({"type": "metaAndAssetCtxs"})
    coins = [a["name"] for a in meta["universe"]]
    df = pd.DataFrame(ctxs, index=coins)
    for col in ["funding", "openInterest", "prevDayPx", "markPx", "midPx", "dayNtlVlm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_funding_history(coin: str, start_ms: int) -> pd.DataFrame:
    """Returns hourly funding rate history for a coin since start_ms, paginating past the 500-record API cap."""
    pages = []
    cursor = start_ms
    while True:
        data = _post({"type": "fundingHistory", "coin": coin, "startTime": cursor})
        if not data:
            break
        pages.append(data)
        if len(data) < 500:
            break
        # advance past the last record to avoid re-fetching it
        cursor = data[-1]["time"] + 1
        time.sleep(0.05)
    if not pages:
        return pd.DataFrame(columns=["fundingRate", "premium"])
    df = pd.DataFrame([row for page in pages for row in page])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["premium"] = df["premium"].astype(float)
    return df[["time", "fundingRate", "premium"]].drop_duplicates("time").set_index("time")


def get_candles(coin: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Returns OHLCV candles for a coin."""
    data = _post({
        "type": "candleSnapshot",
        "req": {"coin": coin, "interval": interval, "startTime": start_ms, "endTime": end_ms},
    })
    if not data:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    for src, dst in [("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"), ("v", "volume")]:
        df[dst] = df[src].astype(float)
    return df[["time", "open", "high", "low", "close", "volume"]].set_index("time")


def get_all_mids() -> pd.Series:
    """Returns live mid prices for all coins."""
    data = _post({"type": "allMids"})
    return pd.Series(data, dtype=float)

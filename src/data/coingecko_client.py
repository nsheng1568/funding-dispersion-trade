import time

import requests
import pandas as pd

_CG_URL = "https://api.coingecko.com/api/v3"
_RETRY_DELAYS = [10, 30, 60]


def get_markets(vs_currency: str = "usd", per_page: int = 250) -> pd.DataFrame:
    """Returns top coins by market cap with 24h volume and price data."""
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": 1,
        "sparkline": False,
    }
    for attempt, delay in enumerate(_RETRY_DELAYS + [None]):
        resp = requests.get(f"{_CG_URL}/coins/markets", params=params, timeout=15)
        if resp.status_code == 429 and delay is not None:
            time.sleep(delay)
            continue
        resp.raise_for_status()
        break
    df = pd.DataFrame(resp.json())
    return df[["id", "symbol", "name", "current_price", "market_cap", "total_volume"]].copy()

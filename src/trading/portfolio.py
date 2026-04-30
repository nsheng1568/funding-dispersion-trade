"""
Target portfolio construction and order delta computation.

Mirrors the signal → portfolio logic from notebooks/04_backtest.ipynb:
  - Pick idxmin (LONG) and idxmax (SHORT) from the composite signal
  - Beta-neutral, vol-targeted sizing via size_position()
  - Delta: same coin → resize, different coin → close old + open new
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.constants import MIN_ORDER_USD
from src.models.signal import compute_signal, size_position


@dataclass
class TargetPortfolio:
    long_coin:  str
    short_coin: str
    long_usd:   float   # magnitude (positive USD notional)
    short_usd:  float   # magnitude (positive USD notional)


@dataclass
class Order:
    coin:        str
    usd_delta:   float   # positive = buy, negative = sell
    reduce_only: bool
    reason:      str


def compute_target(
    funding: pd.DataFrame,
    betas: pd.DataFrame,
    equity: float,
) -> TargetPortfolio | None:
    """
    Derive the desired 1-long / 1-short portfolio from the current signal.
    """
    signal = compute_signal(funding, betas).sort_values()

    if len(signal) < 2:
        return None

    long_coin  = signal.index[0]
    short_coin = signal.index[-1]

    L, S = size_position(long_coin, short_coin, betas)
    return TargetPortfolio(
        long_coin  = long_coin,
        short_coin = short_coin,
        long_usd   = L * equity,
        short_usd  = S * equity,
    )


def compute_orders(
    current: dict[str, float],
    target: TargetPortfolio,
) -> list[Order]:
    """
    Compute the minimal set of orders to move from current positions to target.

    Reduce-only orders (closes / partial closes) are sorted first to free
    margin before opening new positions.

    Parameters
    ----------
    current : {coin: signed_usd_notional}  positive=long, negative=short
    target  : desired portfolio
    """
    target_map: dict[str, float] = {
        target.long_coin:  +target.long_usd,
        target.short_coin: -target.short_usd,
    }

    orders: list[Order] = []

    # 1. Coins held now but not wanted → close fully
    for coin, notional in current.items():
        if coin not in target_map and notional != 0:
            orders.append(Order(
                coin        = coin,
                usd_delta   = -notional,
                reduce_only = True,
                reason      = "close stale",
            ))

    # 2. Target coins → compute delta vs current
    for coin, target_notional in target_map.items():
        current_notional = current.get(coin, 0.0)
        delta = target_notional - current_notional

        if abs(delta) < MIN_ORDER_USD:
            continue

        # reduce_only when the order moves toward flat
        is_reducing = (
            (current_notional > 0 and delta < 0) or
            (current_notional < 0 and delta > 0)
        )
        orders.append(Order(
            coin        = coin,
            usd_delta   = delta,
            reduce_only = is_reducing,
            reason      = "rebalance",
        ))

    # Closes first, then opens (free margin before increasing exposure)
    orders.sort(key=lambda o: (not o.reduce_only))
    return orders

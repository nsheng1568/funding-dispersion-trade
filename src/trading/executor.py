"""
Hyperliquid exchange wrapper: equity, positions, market orders.

Loads HL_PRIVATE_KEY from .env. All order sizes are converted from
USD notional to coin units using the live mid price.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

from src.constants import MIN_ORDER_USD

load_dotenv()

# The master wallet whose funds and positions this strategy trades.
# The API wallet (HL_PRIVATE_KEY) signs orders on its behalf.
VAULT_ADDRESS = "0x6e48fcE48934317b52b6CcACe381b4548683D156"

SLIPPAGE = 0.002   # 20bps slippage for IOC limit acting as market order


def make_info_client() -> Info:
    """Read-only HL client — no private key required."""
    return Info(constants.MAINNET_API_URL, skip_ws=True)


def create_client() -> tuple[Info, Exchange]:
    """Authenticated client pair — requires HL_PRIVATE_KEY in .env."""
    key = os.environ.get("HL_PRIVATE_KEY")
    if not key:
        raise RuntimeError("HL_PRIVATE_KEY not set — copy .env.example to .env and fill it in")
    account  = Account.from_key(key)
    info     = Info(constants.MAINNET_API_URL, skip_ws=True)
    # account_address tells the SDK which account's positions to trade on behalf of
    exchange = Exchange(account, constants.MAINNET_API_URL, account_address=VAULT_ADDRESS)
    return info, exchange


def get_equity(info: Optional[Info] = None) -> float:
    """Total USDC account value of the vault (margin summary)."""
    state = (info or make_info_client()).user_state(VAULT_ADDRESS)
    return float(state["marginSummary"]["accountValue"])


def get_positions(info: Optional[Info] = None) -> dict[str, float]:
    """
    Returns {coin: signed_usd_notional} for the vault.
    Positive = long, negative = short.
    Notional is mark-to-market: sign(szi) * positionValue.
    """
    state = (info or make_info_client()).user_state(VAULT_ADDRESS)
    positions = {}
    for item in state["assetPositions"]:
        p = item["position"]
        szi = float(p["szi"])
        if szi == 0:
            continue
        sign = 1 if szi > 0 else -1
        positions[p["coin"]] = sign * float(p["positionValue"])
    return positions


def market_order(
    coin: str,
    usd_delta: float,
    reduce_only: bool = False,
    *,
    client: Optional[tuple[Info, Exchange]] = None,
    mids: Optional[dict] = None,
    meta: Optional[dict] = None,
) -> dict:
    """
    Place a market-like IOC limit order for the given USD notional.

    Parameters
    ----------
    coin        : coin name (e.g. "BTC")
    usd_delta   : positive = buy, negative = sell
    reduce_only : if True, order can only reduce an existing position
    client      : optional pre-built (Info, Exchange) pair; created fresh if omitted
    mids        : optional pre-fetched all_mids dict; fetched fresh if omitted
    meta        : optional pre-fetched meta dict; fetched fresh if omitted

    Returns
    -------
    SDK result dict, or {"status": "skipped", "reason": ...} if below threshold.
    """
    if abs(usd_delta) < MIN_ORDER_USD:
        return {"status": "skipped", "reason": f"|notional| ${abs(usd_delta):.2f} < ${MIN_ORDER_USD}"}

    info, exchange = client if client is not None else create_client()
    is_buy = bool(usd_delta > 0)          # ensure native bool for msgpack
    reduce_only = bool(reduce_only)

    _mids = mids if mids is not None else info.all_mids()
    if coin not in _mids:
        return {"status": "error", "reason": f"{coin} not found in all_mids"}
    mid = float(_mids[coin])

    # Convert USD → coin units, rounded to exchange lot size
    _meta  = meta if meta is not None else info.meta()
    sz_dec = next(
        (int(a["szDecimals"]) for a in _meta["universe"] if a["name"] == coin),
        3,
    )
    sz = round(abs(usd_delta) / mid, sz_dec)

    if sz == 0:
        return {"status": "skipped", "reason": "size rounds to 0 after lot-size rounding"}

    # Round price to 5 significant figures then to (6 - sz_dec) decimal places,
    # matching the HL exchange's own _slippage_price rounding (exchange.py:111-112).
    raw_px = mid * (1 + SLIPPAGE) if is_buy else mid * (1 - SLIPPAGE)
    px = round(float(f"{raw_px:.5g}"), 6 - sz_dec)

    result = exchange.order(
        coin,
        is_buy,
        sz,
        px,
        order_type={"limit": {"tif": "Ioc"}},
        reduce_only=reduce_only,
    )
    return result

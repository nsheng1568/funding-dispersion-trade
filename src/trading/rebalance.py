"""
One-shot rebalancing CLI for the funding dispersion trade.

Usage:
    python -m src.trading.rebalance              # dry run — show plan only
    python -m src.trading.rebalance --execute    # send live orders

Sequence:
  1. Load betas + funding from parquet
  2. Fetch equity and current positions from HL
  3. Compute target portfolio from signal
  4. Compute order delta
  5. Print plan; execute if --execute passed
"""

import argparse

from src.data.loader import load_betas
from src.models.signal import refresh_funding
from src.trading.executor import create_client, get_equity, get_positions, make_info_client, market_order
from src.trading.portfolio import compute_orders, compute_target


def run(execute: bool = False) -> None:
    print("=== Funding Dispersion Trade — Rebalance ===\n")

    # ------------------------------------------------------------------ data
    print("Loading betas and refreshing funding data...")
    betas = load_betas()
    valid_coins = list(betas.index)
    raw = refresh_funding(valid_coins)
    funding = raw[[c for c in valid_coins if c in raw.columns]]

    # --------------------------------------------------------- account state
    print("Fetching account state from HL...")
    info   = make_info_client()
    equity = get_equity(info)
    current = get_positions(info)

    print(f"\nEquity: ${equity:,.2f}")
    if current:
        print("Current positions:")
        for coin, notional in current.items():
            side = "LONG " if notional > 0 else "SHORT"
            print(f"  {side} {coin:<8}  ${abs(notional):>10,.2f}")
    else:
        print("Current positions: none")

    # -------------------------------------------------------- target + delta
    target = compute_target(funding, betas, equity)

    if target is None:
        print("\nNot enough coins with valid signals — no trade.")
        return

    print(f"\nTarget portfolio:")
    print(f"  LONG  {target.long_coin:<8}  ${target.long_usd:>10,.2f}")
    print(f"  SHORT {target.short_coin:<8}  ${target.short_usd:>10,.2f}")

    orders = compute_orders(current, target)

    if not orders:
        print("\nNo orders needed — already at target.")
        return

    print(f"\nOrders ({len(orders)}):")
    for o in orders:
        side = "BUY " if o.usd_delta > 0 else "SELL"
        ro   = "  [reduce_only]" if o.reduce_only else ""
        print(f"  {side} {o.coin:<8}  ${abs(o.usd_delta):>10,.2f}{ro}  ({o.reason})")

    # ------------------------------------------------------------ execution
    if not execute:
        print("\n[DRY RUN] Pass --execute to send orders.")
        return

    print("\nExecuting orders...")
    client = create_client()
    exec_info = client[0]
    mids = exec_info.all_mids()
    meta = exec_info.meta()
    for o in orders:
        result = market_order(o.coin, o.usd_delta, reduce_only=o.reduce_only,
                              client=client, mids=mids, meta=meta)
        status = result.get("status", "unknown")
        if status == "skipped":
            print(f"  SKIP  {o.coin}: {result.get('reason', '')}")
        elif status == "error":
            print(f"  ERROR {o.coin}: {result.get('reason', '')}")
        else:
            # SDK returns {"status": "ok", "response": {...}}
            filled = result.get("response", {}).get("data", {})
            print(f"  OK    {o.coin}: {filled}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebalance the funding dispersion portfolio")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Send live orders to HL (default: dry run)",
    )
    args = parser.parse_args()
    run(execute=args.execute)

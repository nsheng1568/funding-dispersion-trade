from unittest.mock import patch

import pandas as pd
import pytest

from src.models.signal import size_position
from src.trading.portfolio import Order, TargetPortfolio, compute_orders, compute_target


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _target(long_coin="APE", short_coin="LINK", long_usd=500.0, short_usd=400.0):
    return TargetPortfolio(long_coin, short_coin, long_usd, short_usd)


def _reduce_first(orders):
    """Assert all reduce_only orders come before non-reduce_only orders."""
    saw_open = False
    for o in orders:
        if not o.reduce_only:
            saw_open = True
        if saw_open and o.reduce_only:
            return False
    return True


# ---------------------------------------------------------------------------
# compute_orders — pure logic, no mocking needed
# ---------------------------------------------------------------------------

def test_flat_start_opens_both_legs():
    orders = compute_orders({}, _target())
    coins = {o.coin: o for o in orders}
    assert "APE"  in coins and coins["APE"].usd_delta  > 0
    assert "LINK" in coins and coins["LINK"].usd_delta < 0
    assert all(not o.reduce_only for o in orders)


def test_already_at_target_no_orders():
    current = {"APE": +500.0, "LINK": -400.0}
    assert compute_orders(current, _target()) == []


def test_same_pair_resize_both_legs():
    current = {"APE": +400.0, "LINK": -300.0}
    orders = compute_orders(current, _target())
    coins = {o.coin: o for o in orders}
    assert coins["APE"].usd_delta  == pytest.approx(+100.0)
    assert coins["LINK"].usd_delta == pytest.approx(-100.0)
    assert not coins["APE"].reduce_only
    assert not coins["LINK"].reduce_only


def test_long_rotation_closes_stale_and_opens_new():
    current = {"SOL": +500.0, "LINK": -400.0}  # SOL is stale long
    orders = compute_orders(current, _target())
    coins = {o.coin: o for o in orders}
    # SOL must be closed
    assert "SOL" in coins
    assert coins["SOL"].usd_delta == pytest.approx(-500.0)
    assert coins["SOL"].reduce_only
    # LINK delta is zero — no order
    assert "LINK" not in coins
    # APE must be opened
    assert "APE" in coins
    assert coins["APE"].usd_delta == pytest.approx(+500.0)


def test_full_rotation_four_orders():
    current = {"SOL": +500.0, "BTC": -400.0}
    orders = compute_orders(current, _target())
    coins = {o.coin: o for o in orders}
    assert set(coins) == {"SOL", "BTC", "APE", "LINK"}
    assert coins["SOL"].reduce_only
    assert coins["BTC"].reduce_only
    assert not coins["APE"].reduce_only
    assert not coins["LINK"].reduce_only


def test_closes_sorted_before_opens():
    current = {"SOL": +500.0, "BTC": -400.0}
    orders = compute_orders(current, _target())
    assert _reduce_first(orders)


def test_delta_below_min_order_skipped():
    current = {"APE": +495.0, "LINK": -400.0}  # APE delta = $5 < MIN_ORDER_USD
    orders = compute_orders(current, _target())
    assert not any(o.coin == "APE" for o in orders)


def test_partial_close_sets_reduce_only():
    current = {"APE": +800.0, "LINK": -600.0}   # both oversized vs target
    orders = compute_orders(current, _target())
    coins = {o.coin: o for o in orders}
    # Reducing a long → sell → reduce_only
    assert coins["APE"].usd_delta  == pytest.approx(-300.0)
    assert coins["APE"].reduce_only
    # Reducing a short → buy → reduce_only
    assert coins["LINK"].usd_delta == pytest.approx(+200.0)
    assert coins["LINK"].reduce_only


# ---------------------------------------------------------------------------
# compute_target — mock compute_signal to isolate portfolio logic
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_betas():
    return pd.DataFrame(
        {"beta": {"APE": 0.8, "LINK": 0.9}, "idio_vol": {"APE": 0.008, "LINK": 0.007}}
    )


def test_compute_target_returns_none_when_too_few_signals(minimal_betas):
    single_coin_signal = pd.Series({"APE": 0.5})   # only 1 coin — can't form a pair
    with patch("src.trading.portfolio.compute_signal", return_value=single_coin_signal):
        result = compute_target(None, minimal_betas, equity=1000.0)
    assert result is None


def test_compute_target_correct_coins_and_notionals(minimal_betas):
    signal = pd.Series({"APE": -0.5, "LINK": 0.5})
    with patch("src.trading.portfolio.compute_signal", return_value=signal):
        target = compute_target(None, minimal_betas, equity=1000.0)

    assert target is not None
    assert target.long_coin  == "APE"   # lowest signal
    assert target.short_coin == "LINK"  # highest signal
    # USD notionals must be L * equity and S * equity
    L, S = size_position("APE", "LINK", minimal_betas)
    assert target.long_usd  == pytest.approx(L * 1000.0)
    assert target.short_usd == pytest.approx(S * 1000.0)

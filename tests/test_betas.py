import numpy as np
import pandas as pd
import pytest

from src.models.betas import ANN_FACTOR, estimate_betas, fit_market_factor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_prices():
    """
    500 8h bars with a strong common market factor plus per-coin noise.
    BTC/ETH/SOL are the anchor coins used by fit_market_factor.
    """
    n = 500
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=n, freq="8h")

    market_shocks = rng.normal(0, 0.01, n)

    data = {}
    for coin, noise_scale in [("BTC", 0.002), ("ETH", 0.003), ("SOL", 0.004)]:
        rets = market_shocks + rng.normal(0, noise_scale, n)
        data[coin] = 100 * (1 + rets).cumprod()

    return pd.DataFrame(data, index=idx)


def _add_synthetic_coin(prices: pd.DataFrame, name: str, scale: float) -> pd.DataFrame:
    """Add a coin whose returns are exactly `scale * market_return`."""
    market = fit_market_factor(prices)
    ret_index = prices.pct_change().dropna().index
    coin_rets = (market * scale).reindex(ret_index)
    coin_prices = 100 * (1 + coin_rets).cumprod()
    out = prices.copy()
    out[name] = coin_prices.reindex(prices.index)
    return out


# ---------------------------------------------------------------------------
# fit_market_factor tests
# ---------------------------------------------------------------------------

def test_market_factor_positive_btc_correlation(synthetic_prices):
    """PC1 must have positive correlation with BTC returns (sign convention)."""
    market = fit_market_factor(synthetic_prices)
    ret_btc = synthetic_prices["BTC"].pct_change().dropna()
    common = market.index.intersection(ret_btc.index)
    corr = market.loc[common].corr(ret_btc.loc[common])
    assert corr > 0


def test_market_factor_length(synthetic_prices):
    """Output index should align with the anchor-coin return index."""
    market = fit_market_factor(synthetic_prices)
    ret_anchor = synthetic_prices[["BTC", "ETH", "SOL"]].pct_change().dropna()
    assert len(market) == len(ret_anchor)
    assert (market.index == ret_anchor.index).all()


# ---------------------------------------------------------------------------
# estimate_betas tests
# ---------------------------------------------------------------------------

def test_output_columns(synthetic_prices):
    betas = estimate_betas(synthetic_prices)
    expected = {"beta", "alpha_8h", "r_squared", "idio_vol", "total_vol", "idio_vol_ann"}
    assert expected.issubset(set(betas.columns))


def test_idio_vol_nonnegative(synthetic_prices):
    betas = estimate_betas(synthetic_prices)
    assert (betas["idio_vol"] >= 0).all()


def test_perfect_market_coin_unit_beta(synthetic_prices):
    """A coin with returns == market_return must get beta≈1 and idio_vol≈0."""
    prices = _add_synthetic_coin(synthetic_prices, "PERFECT", scale=1.0)
    betas = estimate_betas(prices)
    assert abs(betas.loc["PERFECT", "beta"] - 1.0) < 0.01
    assert betas.loc["PERFECT", "idio_vol"] < 1e-8


def test_scaled_market_coin_proportional_beta(synthetic_prices):
    """A coin with returns == 3 * market_return must get beta≈3 and idio_vol≈0."""
    prices = _add_synthetic_coin(synthetic_prices, "TRIPLE", scale=3.0)
    betas = estimate_betas(prices)
    assert abs(betas.loc["TRIPLE", "beta"] - 3.0) < 0.05
    assert betas.loc["TRIPLE", "idio_vol"] < 1e-8


def test_lookback_days_restricts_window(synthetic_prices):
    """Betas estimated on 30 days should differ from betas on the full history."""
    betas_full  = estimate_betas(synthetic_prices)
    betas_short = estimate_betas(synthetic_prices, lookback_days=30)
    assert len(betas_short) > 0
    # With a shorter window the point estimates won't be identical
    assert not betas_full["beta"].equals(betas_short["beta"])


def test_idio_vol_ann_consistent(synthetic_prices):
    """idio_vol_ann must equal idio_vol * ANN_FACTOR for every coin."""
    betas = estimate_betas(synthetic_prices)
    expected = betas["idio_vol"] * ANN_FACTOR
    pd.testing.assert_series_equal(betas["idio_vol_ann"], expected, check_names=False)

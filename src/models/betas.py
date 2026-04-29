"""
PCA market factor estimation and per-coin beta regression.

Mirrors the methodology from notebooks/02_beta_research.ipynb.

Usage:
    python -m src.models.betas              # estimate on full price history
    python -m src.models.betas --days 180   # use only the last 180 days
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

DATA_DIR    = Path("data")
ANCHOR_COINS = ["BTC", "ETH", "SOL"]
ANN_FACTOR  = np.sqrt(3 * 365)   # 1095 8h periods per year


def fit_market_factor(prices: pd.DataFrame) -> pd.Series:
    """
    Fit PCA on BTC/ETH/SOL returns and return PC1 scaled to BTC-return units.

    Standardising before PCA prevents high-vol ETH/SOL from dominating PC1,
    which would otherwise push BTC's beta artificially below 1. Scaling by
    btc_std keeps betas interpretable: beta=1 means the coin moves one
    BTC-sigma per one BTC-sigma of the market factor.

    Sign convention: PC1 is positive when BTC goes up.
    """
    anchor = [c for c in ANCHOR_COINS if c in prices.columns]
    if len(anchor) < 2:
        raise ValueError(
            f"Need at least 2 of {ANCHOR_COINS} in prices; found {anchor}"
        )

    ret_anchor = prices[anchor].pct_change().dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(ret_anchor)

    pca = PCA(n_components=1)
    pca.fit(X)

    # Sign-correct so PC1 loads positively on BTC
    btc_idx = anchor.index("BTC") if "BTC" in anchor else 0
    if pca.components_[0, btc_idx] < 0:
        pca.components_[0] *= -1

    btc_std = ret_anchor[anchor[btc_idx]].std()
    market = pd.Series(
        pca.transform(X)[:, 0] * btc_std,
        index=ret_anchor.index,
        name="market",
    )
    return market


def estimate_betas(
    prices: pd.DataFrame, lookback_days: int = None
) -> pd.DataFrame:
    """
    Estimate per-coin betas by regressing 8h returns on the PCA market factor.

    Parameters
    ----------
    prices        : 8h close prices, index=UTC timestamp, columns=coin names
    lookback_days : if set, restrict to the most recent N calendar days

    Returns
    -------
    DataFrame indexed by coin with columns:
        beta, alpha_8h, r_squared, idio_vol (per 8h), total_vol (per 8h), idio_vol_ann
    """
    if lookback_days is not None:
        cutoff = prices.index.max() - pd.Timedelta(days=lookback_days)
        prices = prices.loc[cutoff:]

    market = fit_market_factor(prices)
    ret = prices.pct_change()   # per-coin NaN handled in the loop below

    results = {}
    for coin in ret.columns:
        y_series = ret[coin].dropna()
        common = y_series.index.intersection(market.index)
        if len(common) < 30:
            continue

        y = y_series.loc[common].values
        x = market.loc[common].values.reshape(-1, 1)

        reg = LinearRegression().fit(x, y)
        resid = y - reg.predict(x)
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)

        results[coin] = {
            "beta":      reg.coef_[0],
            "alpha_8h":  reg.intercept_,
            "r_squared": 1.0 - ss_res / (ss_tot + 1e-30),
            "idio_vol":  resid.std(),
            "total_vol": y.std(),
        }

    if not results:
        raise ValueError(
            "No coins had enough overlapping observations with the market factor "
            f"(min 30 required). Check that prices contains {ANCHOR_COINS} and "
            "that the lookback window is large enough."
        )

    df = pd.DataFrame.from_dict(results, orient="index")
    df["idio_vol_ann"] = df["idio_vol"] * ANN_FACTOR
    return df


def run(lookback_days: int = None) -> pd.DataFrame:
    """Recompute betas from the current prices parquet and overwrite the file."""
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    betas = estimate_betas(prices, lookback_days=lookback_days)
    betas.to_parquet(DATA_DIR / "coin_betas.parquet")

    print(f"Estimated betas for {len(betas)} coins")
    print(
        betas[["beta", "r_squared", "idio_vol_ann"]]
        .sort_values("beta")
        .to_string(float_format="{:.4f}".format)
    )
    return betas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=None,
                        help="Lookback window in calendar days (default: full history)")
    args = parser.parse_args()
    run(lookback_days=args.days)

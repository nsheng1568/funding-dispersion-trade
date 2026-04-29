import numpy as np
import pandas as pd
import pytest

from src.models.signal import (
    _direct_forecast_now,
    size_position,
    ANN_FACTOR,
    MAX_LEVERAGE,
    ROLL_WINDOW,
    TARGET_ANN_VOL,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def betas_normal():
    """Two coins well below the leverage cap: beta_A=1, beta_B=2, idio_vol=0.005."""
    return pd.DataFrame(
        {"beta": {"A": 1.0, "B": 2.0}, "idio_vol": {"A": 0.005, "B": 0.005}}
    )


@pytest.fixture
def betas_low_vol():
    """Two coins whose uncapped sizing would far exceed MAX_LEVERAGE."""
    return pd.DataFrame(
        {"beta": {"A": 1.0, "B": 1.0}, "idio_vol": {"A": 0.0001, "B": 0.0001}}
    )


@pytest.fixture
def forecast_series():
    """Synthetic funding and target series long enough for ROLL_WINDOW."""
    n = ROLL_WINDOW + 500
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    f_vals = rng.normal(0.0001, 0.0002, n)
    return idx, f_vals


# ---------------------------------------------------------------------------
# size_position tests
# ---------------------------------------------------------------------------

def test_beta_neutrality(betas_normal):
    L, S = size_position("A", "B", betas_normal)
    assert abs(L * betas_normal.loc["A", "beta"] - S * betas_normal.loc["B", "beta"]) < 1e-10


def test_vol_target_reached(betas_normal):
    L, S = size_position("A", "B", betas_normal)

    # Analytical expected L
    ratio = betas_normal.loc["A", "beta"] / betas_normal.loc["B", "beta"]   # 0.5
    idio_combined = np.sqrt(0.005 ** 2 + ratio ** 2 * 0.005 ** 2)
    L_expected = TARGET_ANN_VOL / (ANN_FACTOR * idio_combined)
    assert abs(L - L_expected) < 1e-9

    # Annualised idio portfolio vol should equal TARGET_ANN_VOL
    ann_vol = np.sqrt(L ** 2 * 0.005 ** 2 + S ** 2 * 0.005 ** 2) * ANN_FACTOR
    assert abs(ann_vol - TARGET_ANN_VOL) < 1e-6


def test_leverage_cap_fires(betas_low_vol):
    L, S = size_position("A", "B", betas_low_vol)
    assert max(L, S) == pytest.approx(MAX_LEVERAGE, rel=1e-9)


def test_beta_neutrality_preserved_after_cap(betas_low_vol):
    L, S = size_position("A", "B", betas_low_vol)
    assert abs(L * betas_low_vol.loc["A", "beta"] - S * betas_low_vol.loc["B", "beta"]) < 1e-10


# ---------------------------------------------------------------------------
# _direct_forecast_now tests
# ---------------------------------------------------------------------------

def test_nan_when_no_valid_pairs(forecast_series):
    idx, f_vals = forecast_series
    f = pd.Series(f_vals, index=idx)
    t = pd.Series(np.nan, index=idx)   # all NaN — no valid pairs
    assert np.isnan(_direct_forecast_now(f, t))


def test_exact_linear_recovery(forecast_series):
    idx, f_vals = forecast_series
    a_true, b_true = 3.0, 2.0
    t_vals = a_true + b_true * f_vals

    f_series = pd.Series(f_vals, index=idx)
    t_series = pd.Series(t_vals, index=idx)
    t_series.iloc[-1] = np.nan   # last target intentionally missing (causal)

    pred = _direct_forecast_now(f_series, t_series)
    expected = a_true + b_true * f_vals[-1]
    assert abs(pred - expected) / abs(expected) < 1e-6


def test_causal_constraint(forecast_series):
    """Poisoning the last target value must not change the prediction."""
    idx, f_vals = forecast_series
    a_true, b_true = 3.0, 2.0
    t_vals = a_true + b_true * f_vals

    f_series = pd.Series(f_vals, index=idx)
    t_clean   = pd.Series(t_vals, index=idx)
    t_poisoned = t_clean.copy()
    t_poisoned.iloc[-1] = 1e9   # inject garbage into current-period target

    pred_clean    = _direct_forecast_now(f_series, t_clean)
    pred_poisoned = _direct_forecast_now(f_series, t_poisoned)
    assert pred_clean == pytest.approx(pred_poisoned)

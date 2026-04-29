from datetime import datetime, timezone

# Two-way train/test split.
# In-sample: PCA/beta estimation (uses prices) + signal calibration (uses funding rates).
# These are orthogonal data streams so sharing a window introduces no look-ahead bias.
# Out-of-sample (backtest): held out entirely until evaluation.

INSAMPLE_START = datetime(2024, 5, 1, tzinfo=timezone.utc)
INSAMPLE_END   = datetime(2025, 9, 1, tzinfo=timezone.utc)  # ~16 months in-sample
BACKTEST_START = datetime(2025, 9, 1, tzinfo=timezone.utc)  # ~8 months OOS
BACKTEST_END   = datetime(2026, 4, 29, tzinfo=timezone.utc)

# Rebalancing cadence and signal horizon.
# Transaction costs (~7bps round-trip) require holding long enough for funding to accumulate.
# At 5bps/day spread: 3-day hold = ~15bps gross / ~8bps net; 7-day = ~35bps gross / ~28bps net.
SIGNAL_HORIZON_DAYS = 7     # forecast cumulative funding over this many days

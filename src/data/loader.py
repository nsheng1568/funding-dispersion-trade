"""
Shared data-loading helpers used by signal.py and rebalance.py.

Centralises parquet schema assumptions so a schema change is a single edit.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")


def load_betas(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load coin betas, filtered to coins with positive beta and idio_vol."""
    betas = pd.read_parquet(data_dir / "coin_betas.parquet").astype(float)
    return betas[(betas["beta"] > 0) & (betas["idio_vol"] > 0)]

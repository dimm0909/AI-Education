from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.features.transform import BASE_FEATURE_COLUMNS

LOGGER = logging.getLogger(__name__)
TARGET_COLUMN = "cnt"
TIME_COLUMNS = ["dteday", "hr"]
REQUIRED_COLUMNS = BASE_FEATURE_COLUMNS + [TARGET_COLUMN] + TIME_COLUMNS


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load Bike Sharing data from CSV and validate expected columns."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset {csv_path} is missing columns: {missing_columns}")

    df["dteday"] = pd.to_datetime(df["dteday"], errors="coerce")
    if df["dteday"].isna().any():
        raise ValueError("Column 'dteday' contains invalid dates")

    df = df.sort_values(TIME_COLUMNS).reset_index(drop=True)
    LOGGER.info("Loaded dataset %s with shape %s", csv_path, df.shape)
    return df


def get_train_test_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by time to avoid leakage from future observations."""
    if not 0.05 <= test_size <= 0.5:
        raise ValueError("test_size should be in [0.05, 0.5]")

    split_idx = int(len(df) * (1.0 - test_size))
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError("Unable to split dataset with selected test_size")

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    LOGGER.info("Time split complete: train=%s test=%s", train_df.shape, test_df.shape)
    return train_df, test_df


def pick_training_data(preferred_path: str | Path, fallback_path: str | Path) -> Path:
    """Pick full dataset if available, otherwise fallback to sample data."""
    preferred = Path(preferred_path)
    fallback = Path(fallback_path)

    if preferred.exists():
        LOGGER.info("Using full dataset: %s", preferred)
        return preferred

    if fallback.exists():
        LOGGER.warning("Full dataset not found. Using sample dataset: %s", fallback)
        return fallback

    raise FileNotFoundError(
        "Neither full nor sample dataset exists. "
        f"Checked: {preferred} and {fallback}"
    )

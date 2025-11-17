"""Data loading utilities for the steel defect project."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from .config import PATHS, TRAINING


def load_dataset(
    path: Path | str | None = None,
    target_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, List[str]]:
    """Load the Steel.csv dataset and return features information.

    Parameters
    ----------
    path:
        Optional override for the CSV path. Defaults to :data:`PATHS.dataset`.
    target_column:
        Name of the target column. If ``None`` the value from
        :data:`TRAINING.target_column` is used.

    Returns
    -------
    df, target_name, numeric_features
        The loaded dataframe, the resolved target column name, and the list of
        numeric feature columns excluding the target.
    """

    csv_path = Path(path) if path else PATHS.dataset
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    target_name = target_column or TRAINING.target_column
    if target_name not in df.columns:
        raise ValueError(
            f"Target column '{target_name}' not found. Available columns: {list(df.columns)}"
        )

    _report_overview(df)
    _report_missing(df)

    numeric_features = [col for col in df.select_dtypes(include=["number"]).columns if col != target_name]
    return df, target_name, numeric_features


def _report_overview(df: pd.DataFrame) -> None:
    """Print high-level information about the dataset."""

    print("Dataset loaded:")
    print(f"  Shape: {df.shape}")
    print("  Dtypes:")
    print(df.dtypes)


def _report_missing(df: pd.DataFrame) -> None:
    """Report missing values per column."""

    missing = df.isna().sum()
    total_missing = int(missing.sum())
    if total_missing == 0:
        print("No missing values detected.")
    else:
        print("Missing values per column:")
        print(missing[missing > 0])


def describe_target_distribution(df: pd.DataFrame, target_column: str) -> pd.Series:
    """Return the value counts for the target column."""

    if target_column not in df.columns:
        raise KeyError(f"{target_column} is not a column in the dataframe")
    return df[target_column].value_counts().sort_index()


__all__ = [
    "load_dataset",
    "describe_target_distribution",
]

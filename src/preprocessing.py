"""Preprocessing utilities for the steel defect pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import TRAINING


@dataclass
class PreprocessedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train_scaled: np.ndarray
    X_test_scaled: np.ndarray
    scaler: StandardScaler
    label_encoder: Optional[LabelEncoder]


def split_and_scale(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    test_size: float | None = None,
    random_state: int | None = None,
) -> PreprocessedData:
    """Split the dataset and scale numeric features.

    Parameters
    ----------
    df:
        Input dataframe.
    feature_columns:
        Columns used as predictors.
    target_column:
        Name of the target column.
    test_size, random_state:
        Optional overrides for :data:`TRAINING` defaults.
    """

    size = test_size if test_size is not None else TRAINING.test_size
    seed = random_state if random_state is not None else TRAINING.random_state

    X = df[feature_columns]
    y = df[target_column]

    label_encoder: Optional[LabelEncoder] = None
    if y.dtype == "O" or y.dtype.name == "category":
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), index=y.index)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=size,
        stratify=y,
        random_state=seed,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return PreprocessedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        scaler=scaler,
        label_encoder=label_encoder,
    )


__all__ = ["PreprocessedData", "split_and_scale"]

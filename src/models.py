"""Model construction utilities."""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

from .config import TRAINING


def build_random_forest(class_weight: Dict[int, int] | None = None) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=1,
        min_samples_split=2,
        class_weight=class_weight or TRAINING.default_class_weight,
        random_state=TRAINING.random_state,
        n_jobs=-1,
    )


def build_xgboost() -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        learning_rate=0.01,
        max_depth=7,
        n_estimators=200,
        subsample=1.0,
        colsample_bytree=0.8,
        gamma=0,
        reg_lambda=1.0,
        random_state=TRAINING.random_state,
        use_label_encoder=False,
        n_jobs=-1,
    )


def build_svm_rbf(class_weight: Dict[int, int] | None = None) -> SVC:
    return SVC(
        kernel="rbf",
        probability=True,
        class_weight=class_weight or TRAINING.default_class_weight,
        random_state=TRAINING.random_state,
    )


def build_linear_svm(class_weight: Dict[int, int] | None = None) -> SVC:
    return SVC(
        kernel="linear",
        probability=True,
        class_weight=class_weight or TRAINING.default_class_weight,
        random_state=TRAINING.random_state,
    )


def build_mlp() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=TRAINING.random_state,
    )


def average_probabilities(probability_arrays: Iterable[np.ndarray]) -> np.ndarray:
    probs = list(probability_arrays)
    if not probs:
        raise ValueError("No probability arrays provided for averaging")
    stacked = np.stack(probs, axis=0)
    return stacked.mean(axis=0)


__all__ = [
    "build_random_forest",
    "build_xgboost",
    "build_svm_rbf",
    "build_linear_svm",
    "build_mlp",
    "average_probabilities",
]

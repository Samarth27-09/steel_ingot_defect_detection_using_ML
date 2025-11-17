"""Training and ensemble optimization utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from .config import TRAINING
from .evaluation import ClassificationMetrics, evaluate_model
from .models import (
    average_probabilities,
    build_linear_svm,
    build_mlp,
    build_random_forest,
    build_svm_rbf,
    build_xgboost,
)
from .preprocessing import PreprocessedData


@dataclass
class TrainingArtifacts:
    models: Dict[str, Any]
    metrics: Dict[str, ClassificationMetrics]
    best_class_weight: Dict[int, int]
    best_threshold: float
    ensemble_probabilities: np.ndarray
    ensemble_predictions: np.ndarray


def train_and_optimize(preprocessed: PreprocessedData) -> TrainingArtifacts:
    """Train base models and optimize ensemble threshold/class weights."""

    X_train = preprocessed.X_train
    X_test = preprocessed.X_test
    y_train = preprocessed.y_train.values
    y_test = preprocessed.y_test.values

    X_train_scaled = preprocessed.X_train_scaled
    X_test_scaled = preprocessed.X_test_scaled

    xgb = build_xgboost()
    xgb.fit(X_train, y_train)
    xgb_proba_test = xgb.predict_proba(X_test)[:, 1]

    mlp = build_mlp()
    mlp.fit(X_train_scaled, y_train)
    mlp_proba_test = mlp.predict_proba(X_test_scaled)[:, 1]

    best_result: TrainingArtifacts | None = None
    best_score: Tuple[float, float] = (-np.inf, -np.inf)

    for class_weight in TRAINING.class_weight_grid:
        rf = build_random_forest(class_weight)
        rf.fit(X_train, y_train)
        rf_proba_test = rf.predict_proba(X_test)[:, 1]

        svm = build_svm_rbf(class_weight)
        svm.fit(X_train_scaled, y_train)
        svm_proba_test = svm.predict_proba(X_test_scaled)[:, 1]

        base_probs = {
            "Random Forest": rf_proba_test,
            "XGBoost": xgb_proba_test,
            "SVM (RBF)": svm_proba_test,
            "MLP": mlp_proba_test,
        }

        base_metrics = {
            name: evaluate_model(y_test, (proba >= 0.5).astype(int), proba)
            for name, proba in base_probs.items()
        }

        ensemble_proba = average_probabilities(base_probs.values())

        for threshold in TRAINING.threshold_grid:
            ensemble_pred = (ensemble_proba >= threshold).astype(int)
            ensemble_metrics = evaluate_model(y_test, ensemble_pred, ensemble_proba)

            score = (ensemble_metrics.f1, ensemble_metrics.accuracy)
            if score > best_score:
                best_score = score
                metrics_map = dict(base_metrics)
                metrics_map["Ensemble"] = ensemble_metrics
                best_result = TrainingArtifacts(
                    models={
                        "Random Forest": rf,
                        "XGBoost": xgb,
                        "SVM (RBF)": svm,
                        "MLP": mlp,
                    },
                    metrics=metrics_map,
                    best_class_weight=class_weight,
                    best_threshold=threshold,
                    ensemble_probabilities=ensemble_proba,
                    ensemble_predictions=ensemble_pred,
                )

    if best_result is None:
        raise RuntimeError("Training failed to produce any ensemble configuration.")

    return best_result


__all__ = ["TrainingArtifacts", "train_and_optimize"]

"""Explainability helpers: SHAP + linear SVM decision boundary."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .config import TRAINING
from .models import build_linear_svm
from .preprocessing import PreprocessedData


@dataclass
class LinearSVMExplanation:
    model: Any
    coefficients: pd.DataFrame
    equation: str
    figure: plt.Figure


def compute_shap_values(
    model,
    X_background: np.ndarray,
    X_target: np.ndarray,
    feature_names: Iterable[str],
    max_background: int = 200,
):
    """Compute SHAP values for a tree-based model."""

    if X_background.shape[0] > max_background:
        background = shap.sample(X_background, max_background, random_state=TRAINING.random_state)
    else:
        background = X_background

    explainer = shap.TreeExplainer(model, data=background)
    shap_values = explainer.shap_values(X_target)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # probability for positive class
    return explainer, shap_values


def plot_shap_summary(shap_values, X_target: np.ndarray, feature_names: Iterable[str]) -> None:
    shap.summary_plot(shap_values, X_target, feature_names=feature_names, plot_type="dot")


def plot_shap_importance(shap_values, feature_names: Iterable[str]) -> None:
    shap.summary_plot(shap_values, features=None, feature_names=feature_names, plot_type="bar")


def explain_linear_svm(preprocessed: PreprocessedData, class_weight: Dict[int, int] | None = None) -> LinearSVMExplanation:
    """Train a linear SVM and return coefficient-based explanations."""

    model = build_linear_svm(class_weight)
    model.fit(preprocessed.X_train_scaled, preprocessed.y_train)

    coeffs = model.coef_.ravel()
    intercept = float(model.intercept_[0])
    feature_names = list(preprocessed.X_train.columns)

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coeffs,
            "importance": np.abs(coeffs),
        }
    ).sort_values("importance", ascending=False)

    terms = [f"({coef:+.4f})*{name}" for coef, name in zip(coeffs, feature_names)]
    equation = "f(x) = " + " + ".join(terms) + f" + ({intercept:+.4f})"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance["feature"], importance["coefficient"], color=["#d62728" if v < 0 else "#1f77b4" for v in importance["coefficient"]])
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Linear SVM Coefficient-Based Importance")
    ax.set_xlabel("Coefficient value")
    ax.set_ylabel("Feature")
    ax.invert_yaxis()

    return LinearSVMExplanation(model=model, coefficients=importance, equation=equation, figure=fig)


__all__ = [
    "compute_shap_values",
    "plot_shap_summary",
    "plot_shap_importance",
    "explain_linear_svm",
    "LinearSVMExplanation",
]

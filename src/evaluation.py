"""Model evaluation helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


@dataclass
class ClassificationMetrics:
    """Container for scalar metrics and curve data."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: np.ndarray = field(repr=False)
    fpr: np.ndarray = field(repr=False)
    tpr: np.ndarray = field(repr=False)
    roc_thresholds: np.ndarray = field(repr=False)
    precision_curve: np.ndarray = field(repr=False)
    recall_curve: np.ndarray = field(repr=False)
    pr_thresholds: np.ndarray = field(repr=False)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> ClassificationMetrics:
    fpr, tpr, roc_thresholds = metrics.roc_curve(y_true, y_proba)
    precision_curve, recall_curve, pr_thresholds = metrics.precision_recall_curve(y_true, y_proba)

    return ClassificationMetrics(
        accuracy=float(metrics.accuracy_score(y_true, y_pred)),
        precision=float(metrics.precision_score(y_true, y_pred, zero_division=0)),
        recall=float(metrics.recall_score(y_true, y_pred)),
        f1=float(metrics.f1_score(y_true, y_pred)),
        roc_auc=float(metrics.auc(fpr, tpr)),
        pr_auc=float(metrics.auc(recall_curve, precision_curve)),
        confusion_matrix=metrics.confusion_matrix(y_true, y_pred),
        fpr=fpr,
        tpr=tpr,
        roc_thresholds=roc_thresholds,
        precision_curve=precision_curve,
        recall_curve=recall_curve,
        pr_thresholds=pr_thresholds,
    )


def plot_roc_curves(curve_data: Mapping[str, ClassificationMetrics], figsize: tuple[int, int] = (8, 6)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    for label, metric in curve_data.items():
        ax.plot(metric.fpr, metric.tpr, label=f"{label} (AUC={metric.roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_title("ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    return fig


def plot_pr_curves(curve_data: Mapping[str, ClassificationMetrics], figsize: tuple[int, int] = (8, 6)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    for label, metric in curve_data.items():
        ax.plot(metric.recall_curve, metric.precision_curve, label=f"{label} (AUPRC={metric.pr_auc:.2f})")
    ax.set_title("Precision-Recall Curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    return fig


def plot_metric_comparison(metrics_map: Mapping[str, ClassificationMetrics], figsize: tuple[int, int] = (10, 6)) -> plt.Figure:
    labels = list(metrics_map.keys())
    metrics_arrays = {
        "Accuracy": [metrics_map[name].accuracy for name in labels],
        "Precision": [metrics_map[name].precision for name in labels],
        "Recall": [metrics_map[name].recall for name in labels],
        "F1": [metrics_map[name].f1 for name in labels],
        "ROC AUC": [metrics_map[name].roc_auc for name in labels],
        "PR AUC": [metrics_map[name].pr_auc for name in labels],
    }

    x = np.arange(len(labels))
    width = 0.12

    fig, ax = plt.subplots(figsize=figsize)
    for idx, (metric_name, values) in enumerate(metrics_arrays.items()):
        ax.bar(x + idx * width, values, width=width, label=metric_name)

    ax.set_xticks(x + width * (len(metrics_arrays) - 1) / 2)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison across Metrics")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    return fig


def metrics_to_dataframe(metrics_map: Mapping[str, ClassificationMetrics]) -> "pd.DataFrame":
    import pandas as pd

    rows = []
    for label, metric in metrics_map.items():
        rows.append(
            {
                "Model": label,
                "Accuracy": metric.accuracy,
                "Precision": metric.precision,
                "Recall": metric.recall,
                "F1": metric.f1,
                "ROC AUC": metric.roc_auc,
                "PR AUC": metric.pr_auc,
            }
        )
    return pd.DataFrame(rows).set_index("Model")


__all__ = [
    "ClassificationMetrics",
    "evaluate_model",
    "plot_roc_curves",
    "plot_pr_curves",
    "plot_metric_comparison",
    "metrics_to_dataframe",
]

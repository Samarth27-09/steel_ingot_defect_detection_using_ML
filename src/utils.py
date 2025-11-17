"""Utility helpers shared across modules."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import seaborn as sns


def configure_plotting(style: str = "whitegrid") -> None:
    sns.set_style(style)
    sns.set_context("talk")
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(matrix, labels: Iterable[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return fig


__all__ = ["configure_plotting", "ensure_directory", "plot_confusion_matrix"]

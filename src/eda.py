"""Exploratory data analysis utilities."""
from __future__ import annotations

from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_context("talk")


def plot_density_by_class(
    df: pd.DataFrame,
    features: Iterable[str],
    target_column: str,
    class_labels: List[str] | None = None,
    figsize: tuple[int, int] = (6, 4),
) -> List[plt.Figure]:
    """Plot overlapping KDE distributions for each feature split by class."""

    plot_df = df.copy()
    hue_column = target_column
    if class_labels is not None:
        unique_values = sorted(plot_df[target_column].unique())
        if len(class_labels) != len(unique_values):
            raise ValueError(
                "Number of class labels must match unique values in the target column."
            )
        mapping = {value: label for value, label in zip(unique_values, class_labels)}
        hue_column = f"{target_column}_label"
        plot_df[hue_column] = plot_df[target_column].map(mapping)

    figures: List[plt.Figure] = []
    for feature in features:
        fig, ax = plt.subplots(figsize=figsize)
        sns.kdeplot(
            data=plot_df,
            x=feature,
            hue=hue_column,
            ax=ax,
            common_norm=False,
            fill=True,
            palette="Set1",
        )
        ax.set_title(f"Density of {feature} by {target_column}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend(title=hue_column)
        figures.append(fig)
    return figures


def plot_correlation_heatmap(
    df: pd.DataFrame,
    features: Iterable[str],
    figsize: tuple[int, int] = (12, 10),
    cmap: str = "coolwarm",
) -> plt.Figure:
    """Plot the correlation heatmap for the provided features."""

    corr = df[list(features)].corr(method="pearson")
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        cmap=cmap,
        center=0.0,
        annot=False,
        square=True,
        cbar_kws={"label": "Pearson r"},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap of Alloying Elements & Casting Parameters")
    return fig


__all__ = ["plot_density_by_class", "plot_correlation_heatmap"]

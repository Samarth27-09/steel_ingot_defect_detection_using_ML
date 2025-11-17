"""Configuration utilities for the steel ingot defect project."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class Paths:
    """Centralized location of important project paths."""

    repo_root: Path = Path(__file__).resolve().parents[1]

    @property
    def data(self) -> Path:
        return self.repo_root / "data"

    @property
    def dataset(self) -> Path:
        return self.data / "Steel.csv"

    @property
    def notebooks(self) -> Path:
        return self.repo_root / "notebooks"


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters shared across modules."""

    random_state: int = 42
    test_size: float = 0.2
    target_column: str = "Defect"
    default_class_weight: Dict[int, int] = None  # type: ignore[assignment]
    decision_threshold: float = 0.62
    class_weight_grid: List[Dict[int, int]] = None  # type: ignore[assignment]
    threshold_grid: List[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "default_class_weight", {0: 62, 1: 12})
        object.__setattr__(self, "class_weight_grid", [
            {0: 50, 1: 10},
            {0: 56, 1: 10},
            {0: 62, 1: 12},
            {0: 70, 1: 14},
        ])
        object.__setattr__(self, "threshold_grid", [0.50, 0.55, 0.60, 0.62, 0.65, 0.70])


PATHS = Paths()
TRAINING = TrainingConfig()

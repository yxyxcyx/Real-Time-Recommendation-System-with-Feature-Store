"""Training utilities and trainers for recommendation models.

This module provides reusable training components extracted from scripts/
to enable clean separation between "Running" (scripts) and "Defining" (src).
"""

from .trainers import TwoTowerTrainer
from .datasets import MovieLensDataset, collate_fn
from .utils import create_two_tower_model_for_training

__all__ = [
    # Trainers
    "TwoTowerTrainer",
    # Datasets
    "MovieLensDataset",
    "collate_fn",
    # Utilities
    "create_two_tower_model_for_training",
]

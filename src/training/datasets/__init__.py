"""PyTorch datasets for training recommendation models."""

from .movielens import MovieLensDataset, collate_fn

__all__ = [
    "MovieLensDataset",
    "collate_fn",
]

"""PyTorch Dataset for MovieLens data.

This module provides a Dataset class for training recommendation models
on MovieLens data with support for negative sampling.
"""

from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ...data.movielens import (
    create_user_features,
    create_movie_features,
    get_user_positive_items,
    sample_negative_items,
)


class MovieLensDataset(Dataset):
    """PyTorch Dataset for MovieLens interactions.
    
    This dataset supports:
    - Precomputed feature matrices for efficiency
    - Negative sampling during training
    - Both training and evaluation modes
    """
    
    def __init__(
        self,
        interactions: pd.DataFrame,
        users: pd.DataFrame,
        movies: pd.DataFrame,
        num_negatives: int = 4,
        is_training: bool = True
    ):
        """Initialize dataset.
        
        Args:
            interactions: Interactions DataFrame with columns:
                - user_idx: int
                - movie_idx: int
                - label: float (0 or 1)
            users: Users DataFrame with user features
            movies: Movies DataFrame with movie features
            num_negatives: Number of negative samples per positive
            is_training: Whether this is for training (enables negative sampling)
        """
        self.interactions = interactions.reset_index(drop=True)
        self.users = users
        self.movies = movies
        self.num_negatives = num_negatives
        self.is_training = is_training
        
        # Precompute user positive items for negative sampling
        self.user_positive_items = get_user_positive_items(interactions)
        self.num_items = movies["movie_idx"].max() + 1
        
        # Precompute feature matrices for efficiency
        self.user_features = self._precompute_user_features()
        self.movie_features = self._precompute_movie_features()
        
    def _precompute_user_features(self) -> np.ndarray:
        """Precompute all user features.
        
        Returns:
            Numpy array of shape (num_users, feature_dim)
        """
        all_user_idx = np.arange(self.users["user_idx"].max() + 1)
        return create_user_features(self.users, all_user_idx, normalize=True)
    
    def _precompute_movie_features(self) -> np.ndarray:
        """Precompute all movie features.
        
        Returns:
            Numpy array of shape (num_movies, feature_dim)
        """
        all_movie_idx = np.arange(self.movies["movie_idx"].max() + 1)
        return create_movie_features(self.movies, all_movie_idx, normalize=True)
    
    def __len__(self) -> int:
        return len(self.interactions)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - user_idx: User index
                - user_features: User feature tensor
                - pos_item_idx: Positive item index
                - pos_item_features: Positive item feature tensor
                - label: Label (0 or 1)
                - neg_item_indices (training only): Negative item indices
                - neg_item_features (training only): Negative item features
        """
        row = self.interactions.iloc[idx]
        user_idx = int(row["user_idx"])
        pos_item_idx = int(row["movie_idx"])
        label = float(row["label"])
        
        # Get features
        user_feat = self.user_features[user_idx]
        pos_item_feat = self.movie_features[pos_item_idx]
        
        if self.is_training and self.num_negatives > 0:
            # Sample negative items
            neg_item_indices = sample_negative_items(
                user_idx, self.user_positive_items, self.num_items, self.num_negatives
            )
            neg_item_feats = self.movie_features[neg_item_indices]
            
            return {
                "user_idx": user_idx,
                "user_features": torch.tensor(user_feat, dtype=torch.float32),
                "pos_item_idx": pos_item_idx,
                "pos_item_features": torch.tensor(pos_item_feat, dtype=torch.float32),
                "neg_item_indices": torch.tensor(neg_item_indices, dtype=torch.long),
                "neg_item_features": torch.tensor(neg_item_feats, dtype=torch.float32),
                "label": torch.tensor(label, dtype=torch.float32),
            }
        else:
            return {
                "user_idx": user_idx,
                "user_features": torch.tensor(user_feat, dtype=torch.float32),
                "pos_item_idx": pos_item_idx,
                "pos_item_features": torch.tensor(pos_item_feat, dtype=torch.float32),
                "label": torch.tensor(label, dtype=torch.float32),
            }


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching MovieLens samples.
    
    This function properly stacks tensors and handles different tensor types.
    
    Args:
        batch: List of sample dictionaries from MovieLensDataset
        
    Returns:
        Collated batch dictionary with stacked tensors
    """
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        if "features" in key:
            # Stack feature tensors
            collated[key] = torch.stack([b[key] for b in batch])
        elif "indices" in key:
            # Stack index tensors
            collated[key] = torch.stack([b[key] for b in batch])
        else:
            # Convert scalars to tensor
            collated[key] = torch.tensor([b[key] for b in batch])
    
    return collated

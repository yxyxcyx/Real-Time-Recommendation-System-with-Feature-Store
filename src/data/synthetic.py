"""Synthetic data generation for training and testing.

This module consolidates synthetic data generation logic that was previously
duplicated across scripts/train_models.py and scripts/train_portfolio_model.py.
"""

import random
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
from loguru import logger

from ..constants import DEFAULT_FEATURE_PADDING_DIM


def generate_synthetic_data(
    num_users: int = 1000,
    num_items: int = 5000,
    num_interactions: int = 50000,
    include_numerical_features: bool = True,
    num_numerical_features: int = 10,
    random_seed: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate synthetic user-item interaction data for training.
    
    Args:
        num_users: Number of users to generate
        num_items: Number of items to generate
        num_interactions: Number of interactions to generate
        include_numerical_features: Whether to include additional numerical features
        num_numerical_features: Number of additional numerical features per entity
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (user_features_df, item_features_df, interactions_df)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)
    
    logger.info(f"Generating synthetic data: {num_users} users, {num_items} items, {num_interactions} interactions")
    
    # Generate user features
    user_features = pd.DataFrame({
        'user_id': [f'user_{i}' for i in range(num_users)],
        'age': np.random.randint(18, 65, num_users),
        'gender': np.random.choice(['M', 'F', 'O'], num_users),
        'location': np.random.choice(['US', 'UK', 'CA', 'AU'], num_users),
        'subscription_tier': np.random.choice(['free', 'basic', 'premium'], num_users),
        'income': np.random.normal(50000, 15000, num_users),
        'signup_days': np.random.randint(1, 365, num_users)
    })
    
    # Add numerical features if requested
    if include_numerical_features:
        for i in range(num_numerical_features):
            user_features[f'user_feat_{i}'] = np.random.randn(num_users)
    
    # Generate item features
    item_features = pd.DataFrame({
        'item_id': [f'item_{i}' for i in range(num_items)],
        'category': np.random.choice(
            ['tech', 'sports', 'entertainment', 'news', 'lifestyle', 
             'electronics', 'clothing', 'books', 'home'], 
            num_items
        ),
        'quality_score': np.random.uniform(0.5, 1.0, num_items),
        'freshness_score': np.random.uniform(0.3, 1.0, num_items),
        'price': np.random.uniform(10, 500, num_items),
        'rating': np.random.uniform(3.0, 5.0, num_items),
        'popularity': np.random.exponential(1.0, num_items)
    })
    
    # Add numerical features if requested
    if include_numerical_features:
        for i in range(num_numerical_features):
            item_features[f'item_feat_{i}'] = np.random.randn(num_items)
    
    # Generate interactions with preference patterns
    interactions = []
    for _ in range(num_interactions):
        user_idx = np.random.randint(num_users)
        item_idx = np.random.randint(num_items)
        
        # Simulate preference patterns
        user_age = user_features.iloc[user_idx]['age']
        item_category = item_features.iloc[item_idx]['category']
        
        # Young users prefer tech and entertainment
        if user_age < 30 and item_category in ['tech', 'entertainment', 'electronics']:
            label = np.random.choice([0, 1], p=[0.2, 0.8])
            rating = random.uniform(4.0, 5.0)
        else:
            label = np.random.choice([0, 1], p=[0.7, 0.3])
            rating = random.uniform(3.0, 4.5)
        
        interactions.append({
            'user_id': f'user_{user_idx}',
            'item_id': f'item_{item_idx}',
            'label': label,
            'rating': rating,
            'timestamp': pd.Timestamp.now()
        })
    
    interactions_df = pd.DataFrame(interactions)
    
    logger.info(f"Generated {len(user_features)} users, {len(item_features)} items, {len(interactions_df)} interactions")
    
    return user_features, item_features, interactions_df


def create_tensors_from_features(
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    target_dim: int = 50
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert feature DataFrames to tensors for model training.
    
    Args:
        user_features: User features DataFrame
        item_features: Item features DataFrame
        target_dim: Target dimension for feature tensors
        
    Returns:
        Tuple of (user_tensor, item_tensor)
    """
    logger.info("Converting features to tensors...")
    
    num_users = len(user_features)
    num_items = len(item_features)
    
    # Extract numerical user features only
    user_numerical_cols = ['age', 'income', 'signup_days']
    user_numerical_cols += [col for col in user_features.columns if 'user_feat_' in col]
    
    available_user_cols = [col for col in user_numerical_cols if col in user_features.columns]
    
    # Handle empty columns case
    if not available_user_cols:
        logger.warning("No numerical user columns found, using random features")
        user_numerical = torch.randn(num_users, target_dim)
    else:
        # Ensure numeric data and convert to tensor
        user_data = user_features[available_user_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        user_numerical = torch.tensor(user_data.values, dtype=torch.float32)
    
    # Normalize income if present
    if 'income' in available_user_cols:
        income_idx = available_user_cols.index('income')
        user_numerical[:, income_idx] = user_numerical[:, income_idx] / 1000
    
    # Pad to target dimension if needed
    current_user_dim = user_numerical.shape[1]
    if current_user_dim < target_dim:
        padding_dim = target_dim - current_user_dim
        user_numerical = torch.cat([
            user_numerical,
            torch.randn(num_users, padding_dim)
        ], dim=1)
    
    # Extract numerical item features only
    item_numerical_cols = ['price', 'rating', 'popularity', 'quality_score', 'freshness_score']
    item_numerical_cols += [col for col in item_features.columns if 'item_feat_' in col]
    
    available_item_cols = [col for col in item_numerical_cols if col in item_features.columns]
    
    # Handle empty columns case
    if not available_item_cols:
        logger.warning("No numerical item columns found, using random features")
        item_numerical = torch.randn(num_items, target_dim)
    else:
        # Ensure numeric data and convert to tensor
        item_data = item_features[available_item_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        item_numerical = torch.tensor(item_data.values, dtype=torch.float32)
    
    # Normalize price if present
    if 'price' in available_item_cols:
        price_idx = available_item_cols.index('price')
        item_numerical[:, price_idx] = item_numerical[:, price_idx] / 100
    
    # Pad to target dimension if needed
    current_item_dim = item_numerical.shape[1]
    if current_item_dim < target_dim:
        padding_dim = target_dim - current_item_dim
        item_numerical = torch.cat([
            item_numerical,
            torch.randn(num_items, padding_dim)
        ], dim=1)
    
    logger.info(f"Created tensors: users {user_numerical.shape}, items {item_numerical.shape}")
    
    return user_numerical, item_numerical


def prepare_two_tower_data(
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    interactions: pd.DataFrame
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare data for two-tower model training.
    
    Args:
        user_features: User features DataFrame
        item_features: Item features DataFrame
        interactions: Interactions DataFrame
        
    Returns:
        Tuple of (user_tensor, item_tensor, labels_tensor)
    """
    logger.info("Preparing data for two-tower model...")
    
    # Merge features with interactions
    data = interactions.merge(user_features, on='user_id').merge(item_features, on='item_id')
    
    # Extract numerical features
    user_num_cols = [col for col in user_features.columns if 'user_feat_' in col]
    item_num_cols = [col for col in item_features.columns if 'item_feat_' in col]
    
    # Handle case where numerical features might not exist
    if not user_num_cols:
        user_num_cols = ['age']  # Fallback to basic feature
    if not item_num_cols:
        item_num_cols = ['quality_score']  # Fallback to basic feature
    
    user_numerical = torch.tensor(data[user_num_cols].values, dtype=torch.float32)
    item_numerical = torch.tensor(data[item_num_cols].values, dtype=torch.float32)
    labels = torch.tensor(data['label'].values, dtype=torch.float32)
    
    logger.info(f"Prepared {len(data)} samples for training")
    
    return user_numerical, item_numerical, labels

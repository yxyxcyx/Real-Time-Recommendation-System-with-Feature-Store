"""Training utility functions.

This module provides helper functions for setting up training,
including model creation and configuration.
"""

from typing import Dict, Any, Optional

import torch

from ..models.two_tower import UserTower, ItemTower, TwoTowerModel


def create_two_tower_model_for_training(
    user_feature_dim: int,
    item_feature_dim: int,
    config: Optional[Dict[str, Any]] = None
) -> TwoTowerModel:
    """Create a Two-Tower model configured for training.
    
    This is a convenience function that creates a properly configured
    TwoTowerModel with sensible defaults.
    
    Args:
        user_feature_dim: Dimension of user input features
        item_feature_dim: Dimension of item input features
        config: Optional configuration dict with keys:
            - embedding_dim: int (default: 64)
            - hidden_layers: list[int] (default: [128, 64])
            - dropout_rate: float (default: 0.2)
            - temperature: float (default: 0.1)
            - activation: str (default: "relu")
            - use_bias: bool (default: True)
            
    Returns:
        Configured TwoTowerModel instance
    """
    config = config or {}
    
    embedding_dim = config.get("embedding_dim", 64)
    hidden_layers = config.get("hidden_layers", [128, 64])
    dropout_rate = config.get("dropout_rate", 0.2)
    activation = config.get("activation", "relu")
    temperature = config.get("temperature", 0.1)
    use_bias = config.get("use_bias", True)
    
    user_tower = UserTower(
        input_dim=user_feature_dim,
        embedding_dim=embedding_dim,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        activation=activation
    )
    
    item_tower = ItemTower(
        input_dim=item_feature_dim,
        embedding_dim=embedding_dim,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        activation=activation,
        use_content_embedding=False
    )
    
    model = TwoTowerModel(
        user_tower=user_tower,
        item_tower=item_tower,
        temperature=temperature,
        use_bias=use_bias
    )
    
    return model


def get_device(prefer_gpu: bool = True) -> str:
    """Get the best available device.
    
    Args:
        prefer_gpu: Whether to prefer GPU if available
        
    Returns:
        Device string ("cuda" or "cpu")
    """
    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

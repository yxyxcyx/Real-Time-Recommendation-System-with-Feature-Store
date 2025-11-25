#!/usr/bin/env python3
"""Simple training script to create a working model for portfolio demonstration."""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import random

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.two_tower import create_two_tower_model
from src.config import get_config
from src.data import generate_synthetic_data, create_tensors_from_features


# NOTE: generate_synthetic_data and create_tensors_from_features moved to src/data/synthetic.py
# Import them from src.data instead of defining here (DRY consolidation)


def train_two_tower_model(user_features, item_features, interactions, config):
    """Train the Two-Tower model."""
    print("Starting Two-Tower model training...")
    
    # Create tensors (interactions not needed for tensor creation)
    user_tensors, item_tensors = create_tensors_from_features(user_features, item_features)
    
    # Create model
    model = create_two_tower_model(config.get("model", {}))
    model.train()
    
    # Set BatchNorm layers to handle batch size 1 during inference
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.track_running_stats = True
            module.momentum = 0.1
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    num_epochs = 10
    batch_size = 64
    
    print(f"Training for {num_epochs} epochs with batch size {batch_size}")
    
    # Simple training loop
    # Convert interactions to list of dicts if it's a DataFrame (from consolidated synthetic.py)
    if hasattr(interactions, 'to_dict'):
        interactions_list = interactions.to_dict('records')
    else:
        interactions_list = interactions
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        # Generate positive pairs from interactions
        for i in range(0, len(interactions_list), batch_size):
            batch_interactions = interactions_list[i:i+batch_size]
            
            # Get user and item indices
            user_indices = [int(interaction['user_id'].split('_')[1]) for interaction in batch_interactions]
            item_indices = [int(interaction['item_id'].split('_')[1]) for interaction in batch_interactions]
            
            # Create batch features
            user_batch = user_tensors[user_indices]
            item_batch = item_tensors[item_indices]
            
            # Forward pass
            user_embeddings = model.get_user_embeddings({'numerical': user_batch, 'categorical': {}})
            item_embeddings = model.get_item_embeddings({'numerical': item_batch, 'categorical': {}})
            
            # Simple contrastive loss
            similarity = torch.sum(user_embeddings * item_embeddings, dim=1)
            loss = -torch.mean(similarity)  # Maximize similarity for positive pairs
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    return model, user_tensors, item_tensors


def save_model_and_artifacts(model, user_tensors, item_tensors, output_dir, config):
    """Save model and related artifacts."""
    print(f"Saving model and artifacts to {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Access the config attribute from ConfigLoader
    config_dict = config.config
    
    # Get two_tower config from the full config structure
    two_tower_config = config_dict.get("model", {}).get("two_tower", {})
    
    # Save model checkpoint
    checkpoint = {
        'user_tower_state': model.user_tower.state_dict(),
        'item_tower_state': model.item_tower.state_dict(),
        'embedding_dim': two_tower_config.get('embedding_dim', 128),
        'temperature': model.temperature,
        'user_bias': None,  # Will be initialized as zeros
        'item_bias': None   # Will be initialized as zeros
    }
    
    checkpoint_path = os.path.join(output_dir, "two_tower_latest.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")
    
    # Save embeddings for quick loading
    model.eval()
    with torch.no_grad():
        user_embeddings = model.get_user_embeddings({'numerical': user_tensors, 'categorical': {}})
        item_embeddings = model.get_item_embeddings({'numerical': item_tensors, 'categorical': {}})
        
        torch.save(user_embeddings, os.path.join(output_dir, "user_embeddings.pt"))
        torch.save(item_embeddings, os.path.join(output_dir, "item_embeddings.pt"))
    
    print("Model and artifacts saved successfully!")


def main():
    """Main training function."""
    print("Training Portfolio Model for RecSys")
    
    # Load config
    config = get_config()
    
    # Generate synthetic data
    user_features, item_features, interactions = generate_synthetic_data()
    
    # Train model
    model, user_tensors, item_tensors = train_two_tower_model(user_features, item_features, interactions, config)
    
    # Save model
    output_dir = "models/checkpoints"
    save_model_and_artifacts(model, user_tensors, item_tensors, output_dir, config)
    
    print("Portfolio model training completed!")
    print("Model is ready for recommendation inference")


if __name__ == "__main__":
    main()

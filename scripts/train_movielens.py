#!/usr/bin/env python3
"""Training script for Two-Tower model on MovieLens-1M dataset.

This script is a thin CLI entry point that uses the training utilities
from src/training/. The heavy lifting is done by:
- src.training.trainers.TwoTowerTrainer
- src.training.datasets.MovieLensDataset
- src.training.utils.create_two_tower_model_for_training
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.movielens import MovieLensLoader
from src.training import (
    TwoTowerTrainer,
    MovieLensDataset,
    collate_fn,
    create_two_tower_model_for_training,
)


# NOTE: MovieLensDataset, collate_fn, TwoTowerTrainer, and create_model have been 
# moved to src/training/ as part of the structural refactoring.
# See:
#   - src/training/datasets/movielens.py
#   - src/training/trainers/two_tower.py  
#   - src/training/utils.py


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Two-Tower model on MovieLens-1M")
    parser.add_argument("--data-path", type=str, default="ml-1m", help="Path to MovieLens-1M")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--num-negatives", type=int, default=16, help="Negative samples per positive")
    args = parser.parse_args()
    
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("logs/training.log", rotation="10 MB", level="DEBUG")
    
    logger.info("=" * 60)
    logger.info("Two-Tower Model Training on MovieLens-1M")
    logger.info("=" * 60)
    
    # Device setup
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")
    
    # Load and preprocess data
    logger.info("Loading MovieLens-1M dataset...")
    loader = MovieLensLoader(args.data_path)
    data = loader.load_and_preprocess(
        split_method="time",
        val_ratio=0.1,
        test_ratio=0.1,
        implicit_threshold=4.0,
        min_user_interactions=5,
        min_item_interactions=5
    )
    
    logger.info(f"Dataset statistics:")
    logger.info(f"  - Users: {data.num_users:,}")
    logger.info(f"  - Movies: {data.num_movies:,}")
    logger.info(f"  - Interactions: {data.num_interactions:,}")
    logger.info(f"  - User features: {data.user_feature_dim}")
    logger.info(f"  - Movie features: {data.movie_feature_dim}")
    
    # Create datasets
    train_dataset = MovieLensDataset(
        data.train_interactions,
        data.users,
        data.movies,
        num_negatives=args.num_negatives,
        is_training=True
    )
    
    val_dataset = MovieLensDataset(
        data.val_interactions,
        data.users,
        data.movies,
        num_negatives=0,
        is_training=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Get feature dimensions from actual data
    user_feature_dim = train_dataset.user_features.shape[1]
    movie_feature_dim = train_dataset.movie_features.shape[1]
    
    logger.info(f"Actual feature dimensions - User: {user_feature_dim}, Movie: {movie_feature_dim}")
    
    # Create model - larger architecture for better performance
    model_config = {
        "embedding_dim": args.embedding_dim,
        "hidden_layers": [256, 128],  # Larger hidden layers
        "dropout_rate": 0.2,
        "temperature": 0.05,  # Lower temperature for sharper similarity
    }
    
    model = create_two_tower_model_for_training(user_feature_dim, movie_feature_dim, model_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create trainer
    trainer_config = {
        "learning_rate": args.lr,
        "weight_decay": 1e-5,
        "early_stopping_patience": 5,
        "checkpoint_dir": "models/checkpoints",
    }
    
    trainer = TwoTowerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        device=device
    )
    
    # Train
    trainer.train(args.epochs)
    
    # Save final artifacts
    logger.info("Saving training artifacts...")
    
    # Save the MovieLens data metadata for evaluation
    import pickle
    metadata = {
        "user_encoder": data.user_encoder,
        "movie_encoder": data.movie_encoder,
        "num_users": data.num_users,
        "num_movies": data.num_movies,
        "user_feature_dim": user_feature_dim,
        "movie_feature_dim": movie_feature_dim,
    }
    
    with open("models/checkpoints/data_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    # Save test data for evaluation
    data.test_interactions.to_parquet("data/processed/test_interactions.parquet")
    data.users.to_parquet("data/processed/users.parquet")
    data.movies.to_parquet("data/processed/movies.parquet")
    
    logger.info("Training completed successfully!")
    logger.info(f"Model checkpoint: models/checkpoints/two_tower_best.pth")
    logger.info(f"Test data: data/processed/test_interactions.parquet")


if __name__ == "__main__":
    main()

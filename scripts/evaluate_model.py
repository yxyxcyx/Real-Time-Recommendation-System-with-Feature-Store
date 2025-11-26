#!/usr/bin/env python3
"""Evaluation script for Two-Tower model on MovieLens-1M test set.

This script:
1. Loads a trained model checkpoint
2. Computes recommendations for test users
3. Calculates comprehensive metrics (Recall, NDCG, Hit Rate, MRR, etc.)
4. Outputs results in a reproducible format
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd
import torch
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.movielens import (
    MovieLensLoader,
    create_user_features,
    create_movie_features,
    get_user_positive_items,
)
from src.models.two_tower import UserTower, ItemTower, TwoTowerModel
from src.evaluation.metrics import Evaluator, EvaluationMetrics, compute_diversity, compute_novelty


def load_model(checkpoint_path: str, user_dim: int, item_dim: int, device: str) -> TwoTowerModel:
    """Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        user_dim: User feature dimension
        item_dim: Item feature dimension
        device: Device to load model on
        
    Returns:
        Loaded TwoTowerModel
    """
    logger.info(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Infer architecture from checkpoint weights
    user_state = checkpoint["user_tower_state"]
    # Get hidden layers from weight shapes: mlp.0.weight shape is [first_hidden, input_dim]
    # mlp.4.weight shape is [second_hidden, first_hidden], etc.
    first_hidden = user_state["mlp.0.weight"].shape[0]  # 256
    second_hidden = user_state["mlp.4.weight"].shape[0]  # 128
    embedding_dim = user_state["mlp.8.weight"].shape[0]  # 128
    hidden_layers = [first_hidden, second_hidden]
    
    logger.info(f"Inferred architecture: hidden_layers={hidden_layers}, embedding_dim={embedding_dim}")
    
    user_tower = UserTower(
        input_dim=user_dim,
        embedding_dim=embedding_dim,
        hidden_layers=hidden_layers,
        dropout_rate=0.2,
        activation="relu"
    )
    
    item_tower = ItemTower(
        input_dim=item_dim,
        embedding_dim=embedding_dim,
        hidden_layers=hidden_layers,
        dropout_rate=0.2,
        activation="relu",
        use_content_embedding=False
    )
    
    model = TwoTowerModel(
        user_tower=user_tower,
        item_tower=item_tower,
        temperature=checkpoint.get("temperature", 0.1),
        use_bias=True
    )
    
    # Load weights
    model.user_tower.load_state_dict(checkpoint["user_tower_state"])
    model.item_tower.load_state_dict(checkpoint["item_tower_state"])
    
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    return model


def prepare_evaluation_data(
    data_path: str,
    train_path: str = None,
    test_path: str = None
) -> tuple:
    """Prepare data for evaluation.
    
    Args:
        data_path: Path to MovieLens-1M data
        train_path: Optional path to saved train interactions
        test_path: Optional path to saved test interactions
        
    Returns:
        Tuple of (train_items, test_ground_truth, users, movies, user_features, movie_features)
    """
    # Load data
    if test_path and Path(test_path).exists():
        logger.info(f"Loading preprocessed test data from {test_path}")
        test_interactions = pd.read_parquet(test_path)
        users = pd.read_parquet("data/processed/users.parquet")
        movies = pd.read_parquet("data/processed/movies.parquet")
        
        # We need to also load train data for exclusion
        loader = MovieLensLoader(data_path)
        data = loader.load_and_preprocess(split_method="time")
        train_interactions = data.train_interactions
    else:
        logger.info("Loading and preprocessing MovieLens-1M...")
        loader = MovieLensLoader(data_path)
        data = loader.load_and_preprocess(split_method="time")
        
        train_interactions = data.train_interactions
        test_interactions = data.test_interactions
        users = data.users
        movies = data.movies
    
    # Get training items per user (to exclude from evaluation)
    train_items = get_user_positive_items(train_interactions)
    
    # Get test ground truth
    test_ground_truth = {}
    for user_idx, group in test_interactions.groupby("user_idx"):
        # Only consider positive interactions (label == 1)
        positive_items = group[group["label"] == 1]["movie_idx"].tolist()
        if positive_items:
            test_ground_truth[user_idx] = set(positive_items)
    
    # Create feature matrices
    max_user_idx = max(users["user_idx"].max(), train_interactions["user_idx"].max())
    max_movie_idx = max(movies["movie_idx"].max(), train_interactions["movie_idx"].max())
    
    all_user_idx = np.arange(max_user_idx + 1)
    all_movie_idx = np.arange(max_movie_idx + 1)
    
    user_features = create_user_features(users, all_user_idx, normalize=True)
    movie_features = create_movie_features(movies, all_movie_idx, normalize=True)
    
    logger.info(f"Test users: {len(test_ground_truth)}")
    logger.info(f"User features shape: {user_features.shape}")
    logger.info(f"Movie features shape: {movie_features.shape}")
    
    return train_items, test_ground_truth, users, movies, user_features, movie_features


def generate_recommendations(
    model: TwoTowerModel,
    test_users: list,
    train_items: Dict[int, list],
    user_features: np.ndarray,
    movie_features: np.ndarray,
    top_k: int = 100,
    batch_size: int = 256,
    device: str = "cpu"
) -> Dict[int, list]:
    """Generate recommendations for test users.
    
    Args:
        model: TwoTowerModel instance
        test_users: List of test user indices
        train_items: Dict of user -> train items to exclude
        user_features: User feature matrix
        movie_features: Movie feature matrix
        top_k: Number of recommendations per user
        batch_size: Batch size for inference
        device: Device for inference
        
    Returns:
        Dict mapping user_idx -> list of recommended movie_idx
    """
    logger.info("Computing item embeddings...")
    
    # Compute all item embeddings once
    with torch.no_grad():
        movie_features_tensor = torch.tensor(movie_features, dtype=torch.float32).to(device)
        item_embeddings = model.get_item_embeddings({
            "numerical": movie_features_tensor,
            "categorical": {}
        }).cpu().numpy()
    
    logger.info(f"Generating recommendations for {len(test_users)} users...")
    recommendations = {}
    
    num_items = movie_features.shape[0]
    
    for i in range(0, len(test_users), batch_size):
        batch_users = test_users[i:i + batch_size]
        
        # Get user features
        batch_user_features = user_features[batch_users]
        
        with torch.no_grad():
            user_features_tensor = torch.tensor(
                batch_user_features, dtype=torch.float32
            ).to(device)
            user_embeddings = model.get_user_embeddings({
                "numerical": user_features_tensor,
                "categorical": {}
            }).cpu().numpy()
        
        # Compute scores (dot product)
        scores = np.dot(user_embeddings, item_embeddings.T)
        
        # Get top-K items for each user, excluding training items
        for j, user_idx in enumerate(batch_users):
            user_scores = scores[j].copy()
            
            # Mask training items
            if user_idx in train_items:
                for train_item in train_items[user_idx]:
                    if train_item < num_items:
                        user_scores[train_item] = -np.inf
            
            # Get top items
            top_indices = np.argsort(user_scores)[::-1][:top_k]
            recommendations[user_idx] = top_indices.tolist()
    
    return recommendations


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Two-Tower model")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/two_tower_best.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--data-path", type=str, default="ml-1m",
                        help="Path to MovieLens-1M data")
    parser.add_argument("--output", type=str, default="results/evaluation_results.json",
                        help="Path to save results")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cpu/cuda/auto)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for inference")
    parser.add_argument("--k-values", type=str, default="5,10,20,50,100",
                        help="Comma-separated K values for evaluation")
    args = parser.parse_args()
    
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    logger.info("=" * 60)
    logger.info("Two-Tower Model Evaluation on MovieLens-1M")
    logger.info("=" * 60)
    
    # Device setup
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")
    
    # Parse K values
    k_values = [int(k) for k in args.k_values.split(",")]
    logger.info(f"Evaluating at K = {k_values}")
    
    # Prepare data
    train_items, test_ground_truth, users, movies, user_features, movie_features = \
        prepare_evaluation_data(args.data_path)
    
    # Load model
    model = load_model(
        args.checkpoint,
        user_dim=user_features.shape[1],
        item_dim=movie_features.shape[1],
        device=device
    )
    
    # Generate recommendations
    test_users = list(test_ground_truth.keys())
    recommendations = generate_recommendations(
        model=model,
        test_users=test_users,
        train_items=train_items,
        user_features=user_features,
        movie_features=movie_features,
        top_k=max(k_values),
        batch_size=args.batch_size,
        device=device
    )
    
    # Evaluate
    evaluator = Evaluator(k_values=k_values, num_items=movie_features.shape[0])
    metrics = evaluator.evaluate(
        predictions=recommendations,
        ground_truth=test_ground_truth,
        exclude_items=train_items
    )
    
    # Print results
    print("\n" + str(metrics))
    
    # Compute additional metrics
    logger.info("Computing diversity and novelty...")
    
    # Item embeddings for diversity
    with torch.no_grad():
        movie_features_tensor = torch.tensor(movie_features, dtype=torch.float32).to(device)
        item_embeddings = model.get_item_embeddings({
            "numerical": movie_features_tensor,
            "categorical": {}
        }).cpu().numpy()
    
    diversity = compute_diversity(recommendations, item_embeddings, k=10)
    
    # Item popularity for novelty (use train data we already loaded)
    from src.data.movielens import MovieLensLoader as MLLoader
    ml_loader = MLLoader(args.data_path)
    data = ml_loader.load_and_preprocess()
    item_popularity = data.train_interactions['movie_idx'].value_counts().to_dict()
    
    novelty = compute_novelty(recommendations, item_popularity, k=10)
    
    logger.info(f"Diversity@10: {diversity:.4f}")
    logger.info(f"Novelty@10: {novelty:.4f}")
    
    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    results = metrics.to_dict()
    results["diversity@10"] = float(diversity)
    results["novelty@10"] = float(novelty)
    results["num_test_users"] = len(test_users)
    results["num_items"] = movie_features.shape[0]
    results["checkpoint"] = args.checkpoint
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {args.output}")
    
    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<20} {'Value':>10}")
    print("-" * 30)
    for k in [10, 20, 50]:
        if k in k_values:
            print(f"{'Recall@' + str(k):<20} {metrics.recall[k]:>10.4f}")
            print(f"{'NDCG@' + str(k):<20} {metrics.ndcg[k]:>10.4f}")
            print(f"{'Hit Rate@' + str(k):<20} {metrics.hit_rate[k]:>10.4f}")
    print(f"{'MRR':<20} {metrics.mrr:>10.4f}")
    print(f"{'Coverage':<20} {metrics.coverage:>10.4f}")
    print(f"{'Diversity@10':<20} {diversity:>10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

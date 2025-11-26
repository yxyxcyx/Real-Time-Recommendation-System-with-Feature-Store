"""Script to train recommendation models."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from loguru import logger

from src.config import get_config
from src.data import generate_synthetic_data, prepare_two_tower_data
from src.features import FeatureStore, FeatureEngineer
from src.models import TwoTowerModel, UserTower, ItemTower, create_two_tower_model, create_ranking_model


# NOTE: generate_synthetic_data and prepare_two_tower_data moved to src/data/synthetic.py
# Import them from src.data instead of defining here (DRY consolidation)


def train_two_tower_model(config):
    """Train the two-tower model."""
    logger.info("Training Two-Tower model...")
    
    # Generate synthetic data
    user_features, item_features, interactions = generate_synthetic_data()
    
    # Prepare data
    user_numerical, item_numerical, labels = prepare_two_tower_data(
        user_features, item_features, interactions
    )
    
    # Split data
    train_idx, val_idx = train_test_split(
        range(len(labels)), test_size=0.2, random_state=42
    )
    
    # Create model using config (matches what service.py loads)
    model = create_two_tower_model(config.get('model.two_tower', {}))
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    num_epochs = 10
    batch_size = 256
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, len(train_idx), batch_size):
            batch_idx = train_idx[i:i+batch_size]
            
            batch_user = user_numerical[batch_idx]
            batch_item = item_numerical[batch_idx]
            batch_labels = labels[batch_idx]
            
            # Forward pass
            outputs = model(
                {'numerical': batch_user},
                {'numerical': batch_item},
                compute_loss=False
            )
            
            similarity = outputs['similarity']
            loss = criterion(similarity, batch_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_user = user_numerical[val_idx]
            val_item = item_numerical[val_idx]
            val_labels = labels[val_idx]
            
            outputs = model(
                {'numerical': val_user},
                {'numerical': val_item},
                compute_loss=False
            )
            
            val_similarity = outputs['similarity']
            val_loss = criterion(val_similarity, val_labels)
            
            # Calculate accuracy
            predictions = (torch.sigmoid(val_similarity) > 0.5).float()
            accuracy = (predictions == val_labels).float().mean()
        
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {epoch_loss/num_batches:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {accuracy:.4f}"
        )
    
    # Save model
    os.makedirs("models/checkpoints", exist_ok=True)
    model.save_model("models/checkpoints/two_tower_latest.pth")
    logger.info("Two-Tower model training completed and saved")
    
    return model


def prepare_ranking_data(user_features, item_features, interactions):
    """Prepare data for ranking model training."""
    logger.info("Preparing data for ranking model...")
    
    # Create interaction features
    data = interactions.merge(user_features, on='user_id').merge(item_features, on='item_id')
    
    # Create ranking features
    ranking_features = pd.DataFrame()
    
    # User features
    ranking_features['user_age'] = data['age'] / 100
    ranking_features['user_is_premium'] = (data['subscription_tier'] == 'premium').astype(int)
    
    # Item features
    ranking_features['item_quality'] = data['quality_score']
    ranking_features['item_freshness'] = data['freshness_score']
    
    # Cross features
    ranking_features['age_quality_product'] = ranking_features['user_age'] * ranking_features['item_quality']
    
    # Add all numerical features
    for col in data.columns:
        if 'feat_' in col:
            ranking_features[col] = data[col]
    
    labels = data['label'].values
    
    return ranking_features, labels


def train_ranking_model(config):
    """Train the ranking model."""
    logger.info("Training ranking model...")
    
    # Check if ranking model is available (requires xgboost)
    if create_ranking_model is None:
        logger.warning("Ranking model not available (xgboost not installed)")
        logger.warning("Skipping ranking model training. Install with: pip install xgboost")
        return None
    
    # Generate synthetic data
    user_features, item_features, interactions = generate_synthetic_data()
    
    # Prepare data
    features, labels = prepare_ranking_data(user_features, item_features, interactions)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # Create and train XGBoost model
    model = create_ranking_model('xgboost', config.get('model.ranking.xgboost', {}))
    model.fit(X_train, y_train, X_val, y_val)
    
    # Save model
    os.makedirs("models/checkpoints", exist_ok=True)
    model.save("models/checkpoints/xgboost_ranker.pkl")
    logger.info("Ranking model training completed and saved")
    
    # Print feature importance
    importance = model.get_feature_importance()
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info(f"Top 10 important features: {top_features}")
    
    return model


# REMOVED: setup_feature_store() was calling generate_sample_data() which
# doesn't exist on SimpleFeatureStore. The function was broken and unused.
# See REFACTORING_BLUEPRINT.md for details.
#
# def setup_feature_store(config):
#     """Set up and populate feature store."""
#     logger.info("Setting up feature store...")
#     feature_store = FeatureStore(config.get('feature_store', {}))
#     user_data, item_data = feature_store.generate_sample_data()  # BROKEN
#     logger.info("Feature store setup completed")
#     return feature_store


def main():
    """Main training pipeline."""
    logger.info("Starting model training pipeline...")
    
    # Load configuration
    config = get_config()
    
    # NOTE: setup_feature_store() removed - was broken (see above)
    # Feature store initialization should be done separately if needed
    
    # Train Two-Tower model
    two_tower_model = train_two_tower_model(config)
    
    # Train ranking model
    ranking_model = train_ranking_model(config)
    
    logger.info("Training pipeline completed successfully!")
    logger.info("Models saved in models/checkpoints/")
    logger.info("You can now start the API server with: python -m src.serving.api")


if __name__ == "__main__":
    main()

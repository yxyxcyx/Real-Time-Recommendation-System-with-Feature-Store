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
from src.features import FeatureStore, FeatureEngineer
from src.models import TwoTowerModel, UserTower, ItemTower, create_ranking_model


def generate_synthetic_data(num_users=1000, num_items=5000, num_interactions=50000):
    """Generate synthetic data for training."""
    logger.info("Generating synthetic training data...")
    
    # Generate user features
    user_features = pd.DataFrame({
        'user_id': [f'user_{i}' for i in range(num_users)],
        'age': np.random.randint(18, 65, num_users),
        'gender': np.random.choice(['M', 'F', 'O'], num_users),
        'location': np.random.choice(['US', 'UK', 'CA', 'AU'], num_users),
        'subscription_tier': np.random.choice(['free', 'basic', 'premium'], num_users),
    })
    
    # Add numerical features
    for i in range(10):
        user_features[f'user_feat_{i}'] = np.random.randn(num_users)
    
    # Generate item features
    item_features = pd.DataFrame({
        'item_id': [f'item_{i}' for i in range(num_items)],
        'category': np.random.choice(['tech', 'sports', 'entertainment', 'news', 'lifestyle'], num_items),
        'quality_score': np.random.uniform(0.5, 1.0, num_items),
        'freshness_score': np.random.uniform(0.3, 1.0, num_items),
    })
    
    # Add numerical features
    for i in range(10):
        item_features[f'item_feat_{i}'] = np.random.randn(num_items)
    
    # Generate interactions (positive samples)
    interactions = []
    for _ in range(num_interactions):
        user_idx = np.random.randint(num_users)
        item_idx = np.random.randint(num_items)
        
        # Simulate some preference patterns
        user_age = user_features.iloc[user_idx]['age']
        item_category = item_features.iloc[item_idx]['category']
        
        # Young users prefer tech and entertainment
        if user_age < 30 and item_category in ['tech', 'entertainment']:
            label = np.random.choice([0, 1], p=[0.2, 0.8])
        else:
            label = np.random.choice([0, 1], p=[0.7, 0.3])
        
        interactions.append({
            'user_id': f'user_{user_idx}',
            'item_id': f'item_{item_idx}',
            'label': label,
            'timestamp': pd.Timestamp.now()
        })
    
    interactions_df = pd.DataFrame(interactions)
    
    return user_features, item_features, interactions_df


def prepare_two_tower_data(user_features, item_features, interactions):
    """Prepare data for two-tower model training."""
    logger.info("Preparing data for two-tower model...")
    
    # Merge features
    data = interactions.merge(user_features, on='user_id').merge(item_features, on='item_id')
    
    # Extract numerical features
    user_num_cols = [col for col in user_features.columns if 'user_feat_' in col]
    item_num_cols = [col for col in item_features.columns if 'item_feat_' in col]
    
    user_numerical = torch.tensor(data[user_num_cols].values, dtype=torch.float32)
    item_numerical = torch.tensor(data[item_num_cols].values, dtype=torch.float32)
    labels = torch.tensor(data['label'].values, dtype=torch.float32)
    
    return user_numerical, item_numerical, labels


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
    
    # Create model
    user_tower = UserTower(
        input_dim=10,  # Number of numerical features
        embedding_dim=128,
        hidden_layers=[256, 128],
        dropout_rate=0.2
    )
    
    item_tower = ItemTower(
        input_dim=10,  # Number of numerical features
        embedding_dim=128,
        hidden_layers=[256, 128],
        dropout_rate=0.2,
        use_content_embedding=False  # Simplified for demo
    )
    
    model = TwoTowerModel(user_tower, item_tower, temperature=0.05)
    
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


def setup_feature_store(config):
    """Set up and populate feature store."""
    logger.info("Setting up feature store...")
    
    # Initialize feature store
    feature_store = FeatureStore(config.get('feature_store', {}))
    
    # Generate and save sample data
    user_data, item_data = feature_store.generate_sample_data()
    
    logger.info("Feature store setup completed")
    return feature_store


def main():
    """Main training pipeline."""
    logger.info("Starting model training pipeline...")
    
    # Load configuration
    config = get_config()
    
    # Set up feature store
    feature_store = setup_feature_store(config)
    
    # Train Two-Tower model
    two_tower_model = train_two_tower_model(config)
    
    # Train ranking model
    ranking_model = train_ranking_model(config)
    
    logger.info("Training pipeline completed successfully!")
    logger.info("Models saved in models/checkpoints/")
    logger.info("You can now start the API server with: python -m src.serving.api")


if __name__ == "__main__":
    main()

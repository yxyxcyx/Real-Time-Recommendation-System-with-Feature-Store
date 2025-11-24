"""Feature Store implementation using Feast."""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from feast import FeatureStore as FeastStore
from feast import FeatureView, Entity
from feast.infra.offline_stores.file_source import FileSource
from feast.repo_config import RepoConfig
from loguru import logger

from .feature_definitions import (
    UserFeatures,
    ItemFeatures,
    InteractionFeatures,
    StreamingFeatures,
    RequestFeatures
)


class FeatureStore:
    """Manages feature store operations for the recommendation system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the feature store.
        
        Args:
            config: Feature store configuration
        """
        self.config = config
        self.project_name = config.get("project_name", "recsys_features")
        self.registry_path = config.get("registry_path", "feature_store/registry/registry.db")
        self.online_store_config = config.get("online_store", {})
        self.offline_store_config = config.get("offline_store", {})
        
        # Initialize Feast store
        self._init_feast_store()
        
        # Register feature definitions
        self._register_features()
        
    def _init_feast_store(self):
        """Initialize Feast feature store."""
        # Create directories if they don't exist
        Path(self.registry_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.offline_store_config.get("path", "feature_store/data/")).mkdir(
            parents=True, exist_ok=True
        )
        
        # Create Feast configuration
        repo_config = RepoConfig(
            project=self.project_name,
            registry=self.registry_path,
            provider="local",
            online_store=self.online_store_config,
            offline_store=self.offline_store_config,
        )
        
        # Initialize Feast store
        self.feast_store = FeastStore(repo_path=".", config=repo_config)
        logger.info(f"Initialized Feast store for project: {self.project_name}")
        
    def _register_features(self):
        """Register all feature definitions with Feast."""
        logger.info("Registering feature definitions...")
        
        # Get all feature definitions
        entities = [
            UserFeatures.get_entity(),
            ItemFeatures.get_entity(),
        ]
        
        feature_views = [
            UserFeatures.get_profile_features(),
            UserFeatures.get_activity_features(),
            ItemFeatures.get_content_features(),
            ItemFeatures.get_popularity_features(),
            InteractionFeatures.get_interaction_features(),
            StreamingFeatures.get_realtime_user_features(),
            StreamingFeatures.get_realtime_item_features(),
        ]
        
        # Apply to Feast
        self.feast_store.apply(entities + feature_views)
        logger.info(f"Registered {len(entities)} entities and {len(feature_views)} feature views")
        
    def get_online_features(
        self,
        entity_dict: Dict[str, List[Any]],
        features: List[str],
        request_features: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Get online features for given entities.
        
        Args:
            entity_dict: Dictionary of entity keys and values
            features: List of feature names to retrieve
            request_features: Optional request-time features
            
        Returns:
            DataFrame with requested features
        """
        logger.debug(f"Getting online features for {len(entity_dict)} entities")
        
        try:
            # Prepare feature request
            feature_vector = self.feast_store.get_online_features(
                features=features,
                entity_rows=[
                    {k: v[i] for k, v in entity_dict.items()}
                    for i in range(len(list(entity_dict.values())[0]))
                ],
            ).to_df()
            
            # Add request features if provided
            if request_features:
                for key, value in request_features.items():
                    feature_vector[key] = value
            
            logger.debug(f"Retrieved {len(feature_vector)} feature vectors")
            return feature_vector
            
        except Exception as e:
            logger.error(f"Error getting online features: {e}")
            raise
            
    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        features: List[str],
        timestamp_col: str = "event_timestamp"
    ) -> pd.DataFrame:
        """Get historical features for training.
        
        Args:
            entity_df: DataFrame with entity keys and timestamps
            features: List of feature names to retrieve
            timestamp_col: Name of timestamp column
            
        Returns:
            DataFrame with historical features
        """
        logger.info(f"Getting historical features for {len(entity_df)} entities")
        
        try:
            # Get historical features from Feast
            feature_df = self.feast_store.get_historical_features(
                entity_df=entity_df,
                features=features,
            ).to_df()
            
            logger.info(f"Retrieved {len(feature_df)} historical feature vectors")
            return feature_df
            
        except Exception as e:
            logger.error(f"Error getting historical features: {e}")
            raise
            
    def push_streaming_features(
        self,
        feature_data: pd.DataFrame,
        feature_view_name: str
    ):
        """Push real-time features to online store.
        
        Args:
            feature_data: DataFrame with feature values
            feature_view_name: Name of the feature view
        """
        logger.debug(f"Pushing {len(feature_data)} features to {feature_view_name}")
        
        try:
            # Get feature view
            feature_view = self.feast_store.get_feature_view(feature_view_name)
            
            # Push features to online store
            self.feast_store.push(
                push_source_name=f"{feature_view_name}_push",
                df=feature_data,
            )
            
            logger.debug(f"Successfully pushed features to {feature_view_name}")
            
        except Exception as e:
            logger.error(f"Error pushing streaming features: {e}")
            raise
            
    def materialize_features(
        self,
        start_date: datetime,
        end_date: datetime,
        feature_views: Optional[List[str]] = None
    ):
        """Materialize features to online store.
        
        Args:
            start_date: Start date for materialization
            end_date: End date for materialization
            feature_views: Optional list of feature view names
        """
        logger.info(f"Materializing features from {start_date} to {end_date}")
        
        try:
            self.feast_store.materialize(
                start_date=start_date,
                end_date=end_date,
                feature_views=feature_views,
            )
            
            logger.info("Feature materialization completed")
            
        except Exception as e:
            logger.error(f"Error materializing features: {e}")
            raise
            
    def update_feature_statistics(
        self,
        feature_data: pd.DataFrame,
        feature_view_name: str
    ):
        """Update feature statistics for monitoring.
        
        Args:
            feature_data: DataFrame with feature values
            feature_view_name: Name of the feature view
        """
        stats = {
            "feature_view": feature_view_name,
            "timestamp": datetime.now().isoformat(),
            "row_count": len(feature_data),
            "null_counts": feature_data.isnull().sum().to_dict(),
            "numeric_stats": feature_data.describe().to_dict(),
        }
        
        # Save statistics
        stats_path = Path(f"feature_store/stats/{feature_view_name}_stats.json")
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
            
        logger.debug(f"Updated statistics for {feature_view_name}")
        
    def validate_features(
        self,
        feature_data: pd.DataFrame,
        feature_view_name: str
    ) -> bool:
        """Validate feature data quality.
        
        Args:
            feature_data: DataFrame with feature values
            feature_view_name: Name of the feature view
            
        Returns:
            True if validation passes
        """
        logger.debug(f"Validating features for {feature_view_name}")
        
        # Basic validation checks
        checks = {
            "not_empty": len(feature_data) > 0,
            "has_required_columns": all(
                col in feature_data.columns
                for col in self._get_required_columns(feature_view_name)
            ),
            "no_excessive_nulls": (
                feature_data.isnull().sum() / len(feature_data)
            ).max() < 0.5,
        }
        
        # Log validation results
        for check, passed in checks.items():
            if not passed:
                logger.warning(f"Validation failed for {feature_view_name}: {check}")
                
        return all(checks.values())
        
    def _get_required_columns(self, feature_view_name: str) -> List[str]:
        """Get required columns for a feature view.
        
        Args:
            feature_view_name: Name of the feature view
            
        Returns:
            List of required column names
        """
        # Map feature views to required columns
        required_columns = {
            "user_profile_features": ["user_id", "age", "gender"],
            "user_activity_features": ["user_id", "clicks_24h", "views_24h"],
            "item_content_features": ["item_id", "category", "title"],
            "item_popularity_features": ["item_id", "views_24h", "ctr_24h"],
            "interaction_features": ["user_id", "item_id"],
            "realtime_user_features": ["user_id", "clicks_5min"],
            "realtime_item_features": ["item_id", "views_5min"],
        }
        
        return required_columns.get(feature_view_name, [])
        
    def generate_sample_data(self):
        """Generate sample data for testing."""
        logger.info("Generating sample feature data...")
        
        # Generate sample user data
        num_users = 1000
        user_data = pd.DataFrame({
            "user_id": [f"user_{i}" for i in range(num_users)],
            "age": np.random.randint(18, 65, num_users),
            "gender": np.random.choice(["M", "F", "O"], num_users),
            "location": np.random.choice(["US", "UK", "CA", "AU"], num_users),
            "signup_days": np.random.randint(1, 365, num_users),
            "total_interactions": np.random.randint(0, 1000, num_users),
            "avg_session_duration": np.random.uniform(60, 3600, num_users),
            "preferred_categories": [
                ",".join(np.random.choice(
                    ["tech", "sports", "entertainment", "news", "lifestyle"],
                    size=np.random.randint(1, 4)
                )) for _ in range(num_users)
            ],
            "device_type": np.random.choice(["mobile", "desktop", "tablet"], num_users),
            "subscription_tier": np.random.choice(["free", "basic", "premium"], num_users),
            "event_timestamp": datetime.now(),
            "created_timestamp": datetime.now() - timedelta(days=30),
        })
        
        # Generate sample item data
        num_items = 5000
        item_data = pd.DataFrame({
            "item_id": [f"item_{i}" for i in range(num_items)],
            "title": [f"Item Title {i}" for i in range(num_items)],
            "category": np.random.choice(
                ["tech", "sports", "entertainment", "news", "lifestyle"],
                num_items
            ),
            "subcategory": np.random.choice(
                ["latest", "trending", "featured", "recommended"],
                num_items
            ),
            "tags": [
                ",".join(np.random.choice(
                    ["breaking", "viral", "exclusive", "opinion", "analysis"],
                    size=np.random.randint(1, 3)
                )) for _ in range(num_items)
            ],
            "publish_timestamp": [
                datetime.now() - timedelta(hours=np.random.randint(1, 168))
                for _ in range(num_items)
            ],
            "content_length": np.random.randint(100, 5000, num_items),
            "language": np.random.choice(["en", "es", "fr"], num_items, p=[0.8, 0.1, 0.1]),
            "author": [f"author_{np.random.randint(1, 100)}" for _ in range(num_items)],
            "quality_score": np.random.uniform(0.5, 1.0, num_items),
            "event_timestamp": datetime.now(),
        })
        
        # Save sample data
        data_path = Path(self.offline_store_config.get("path", "feature_store/data/"))
        
        user_data.to_parquet(data_path / "user_profiles.parquet")
        item_data.to_parquet(data_path / "item_content.parquet")
        
        logger.info(f"Generated sample data with {num_users} users and {num_items} items")
        
        return user_data, item_data

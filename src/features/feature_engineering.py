"""Feature engineering pipeline for recommendation system."""

import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm


class FeatureEngineer:
    """Handles feature engineering for the recommendation system."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize feature engineer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.scalers = {}
        self.encoders = {}
        self.feature_stats = {}
        
    def create_user_features(
        self, 
        user_data: pd.DataFrame,
        interaction_data: pd.DataFrame,
        timestamp: datetime
    ) -> pd.DataFrame:
        """Create user features from raw data.
        
        Args:
            user_data: User profile data
            interaction_data: User interaction history
            timestamp: Current timestamp for feature calculation
            
        Returns:
            DataFrame with user features
        """
        logger.info("Creating user features...")
        
        # Convert to Polars for faster processing
        user_df = pl.from_pandas(user_data)
        interaction_df = pl.from_pandas(interaction_data)
        
        # Calculate time-based features
        features = self._calculate_user_activity_features(
            interaction_df, timestamp
        )
        
        # Add profile features
        features = features.join(
            user_df.select([
                "user_id", "age", "gender", "location", 
                "signup_date", "device_type", "subscription_tier"
            ]),
            on="user_id",
            how="left"
        )
        
        # Calculate derived features
        features = features.with_columns([
            ((timestamp - pl.col("signup_date")).dt.days()).alias("signup_days"),
            self._calculate_diversity_score(pl.col("category_list")).alias("diversity_score"),
            self._calculate_engagement_score(
                pl.col("clicks_24h"), pl.col("views_24h")
            ).alias("engagement_score"),
        ])
        
        # Convert back to pandas
        return features.to_pandas()
    
    def create_item_features(
        self,
        item_data: pd.DataFrame,
        interaction_data: pd.DataFrame,
        timestamp: datetime
    ) -> pd.DataFrame:
        """Create item features from raw data.
        
        Args:
            item_data: Item content data
            interaction_data: Item interaction history
            timestamp: Current timestamp for feature calculation
            
        Returns:
            DataFrame with item features
        """
        logger.info("Creating item features...")
        
        # Convert to Polars
        item_df = pl.from_pandas(item_data)
        interaction_df = pl.from_pandas(interaction_data)
        
        # Calculate popularity features
        popularity_features = self._calculate_item_popularity_features(
            interaction_df, timestamp
        )
        
        # Add content features
        features = popularity_features.join(
            item_df.select([
                "item_id", "title", "category", "subcategory",
                "publish_timestamp", "content_length", "author"
            ]),
            on="item_id",
            how="left"
        )
        
        # Calculate derived features
        features = features.with_columns([
            ((timestamp - pl.col("publish_timestamp")).dt.hours()).alias("age_hours"),
            self._calculate_freshness_score(pl.col("age_hours")).alias("freshness_score"),
            self._calculate_trending_score(
                pl.col("views_1h"), pl.col("views_24h")
            ).alias("trending_score"),
            self._calculate_quality_score(features).alias("quality_score"),
        ])
        
        return features.to_pandas()
    
    def create_interaction_features(
        self,
        user_features: pd.DataFrame,
        item_features: pd.DataFrame,
        interaction_pairs: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        """Create user-item interaction features.
        
        Args:
            user_features: User features
            item_features: Item features
            interaction_pairs: List of (user_id, item_id) pairs
            
        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features...")
        
        # Create base DataFrame
        interaction_df = pd.DataFrame(
            interaction_pairs, columns=["user_id", "item_id"]
        )
        
        # Merge user and item features
        features = interaction_df.merge(
            user_features, on="user_id", how="left", suffixes=("", "_user")
        ).merge(
            item_features, on="item_id", how="left", suffixes=("", "_item")
        )
        
        # Calculate cross features
        features["user_category_match"] = (
            features["preferred_categories"].apply(
                lambda x: x if isinstance(x, list) else []
            ).apply(
                lambda cats: features["category"].isin(cats).astype(int)
            )
        )
        
        features["engagement_quality_product"] = (
            features["engagement_score"] * features["quality_score"]
        )
        
        features["freshness_engagement_ratio"] = (
            features["freshness_score"] / (features["engagement_score"] + 1e-6)
        )
        
        return features
    
    def create_streaming_features(
        self,
        event_stream: pd.DataFrame,
        window_minutes: int = 5
    ) -> pd.DataFrame:
        """Create real-time streaming features from event stream.
        
        Args:
            event_stream: Stream of user events
            window_minutes: Window size in minutes
            
        Returns:
            DataFrame with streaming features
        """
        logger.info(f"Creating streaming features with {window_minutes}min window...")
        
        current_time = datetime.now()
        window_start = current_time - timedelta(minutes=window_minutes)
        
        # Filter to window
        windowed_events = event_stream[
            event_stream["timestamp"] >= window_start
        ]
        
        # Aggregate by user
        user_features = windowed_events.groupby("user_id").agg({
            "event_type": lambda x: (x == "click").sum(),
            "item_id": "count",
            "category": lambda x: len(set(x)),
            "dwell_time": "mean",
            "session_id": lambda x: len(set(x)),
        }).rename(columns={
            "event_type": f"clicks_{window_minutes}min",
            "item_id": f"views_{window_minutes}min",
            "category": f"categories_{window_minutes}min",
            "dwell_time": f"avg_dwell_time_{window_minutes}min",
            "session_id": f"sessions_{window_minutes}min",
        })
        
        # Aggregate by item
        item_features = windowed_events.groupby("item_id").agg({
            "event_type": lambda x: (x == "click").sum(),
            "user_id": "count",
        }).rename(columns={
            "event_type": f"clicks_{window_minutes}min",
            "user_id": f"views_{window_minutes}min",
        })
        
        # Calculate CTR
        item_features[f"ctr_{window_minutes}min"] = (
            item_features[f"clicks_{window_minutes}min"] / 
            (item_features[f"views_{window_minutes}min"] + 1e-6)
        )
        
        return user_features, item_features
    
    def _calculate_user_activity_features(
        self,
        interaction_df: pl.DataFrame,
        timestamp: datetime
    ) -> pl.DataFrame:
        """Calculate user activity features for different time windows."""
        windows = {
            "1h": timedelta(hours=1),
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
        }
        
        features = []
        for window_name, window_delta in windows.items():
            window_start = timestamp - window_delta
            
            windowed = interaction_df.filter(
                pl.col("timestamp") >= window_start
            )
            
            window_features = windowed.group_by("user_id").agg([
                pl.col("event_type").filter(pl.col("event_type") == "click")
                    .count().alias(f"clicks_{window_name}"),
                pl.col("event_type").count().alias(f"views_{window_name}"),
                pl.col("dwell_time").mean().alias(f"avg_view_time_{window_name}"),
                pl.col("category").n_unique().alias(f"categories_{window_name}"),
            ])
            
            features.append(window_features)
        
        # Combine all window features
        result = features[0]
        for feat in features[1:]:
            result = result.join(feat, on="user_id", how="outer")
        
        return result.fill_null(0)
    
    def _calculate_item_popularity_features(
        self,
        interaction_df: pl.DataFrame,
        timestamp: datetime
    ) -> pl.DataFrame:
        """Calculate item popularity features for different time windows."""
        windows = {
            "1h": timedelta(hours=1),
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
        }
        
        features = []
        for window_name, window_delta in windows.items():
            window_start = timestamp - window_delta
            
            windowed = interaction_df.filter(
                pl.col("timestamp") >= window_start
            )
            
            window_features = windowed.group_by("item_id").agg([
                pl.col("event_type").filter(pl.col("event_type") == "click")
                    .count().alias(f"clicks_{window_name}"),
                pl.col("event_type").count().alias(f"views_{window_name}"),
            ])
            
            # Calculate CTR
            window_features = window_features.with_columns([
                (pl.col(f"clicks_{window_name}") / 
                 (pl.col(f"views_{window_name}") + 1e-6))
                .alias(f"ctr_{window_name}")
            ])
            
            features.append(window_features)
        
        # Combine all window features
        result = features[0]
        for feat in features[1:]:
            result = result.join(feat, on="item_id", how="outer")
        
        return result.fill_null(0)
    
    def _calculate_diversity_score(self, categories: pl.Series) -> pl.Series:
        """Calculate diversity score based on category distribution."""
        return categories.map_elements(
            lambda x: len(set(x)) / max(len(x), 1) if x else 0
        )
    
    def _calculate_engagement_score(
        self, 
        clicks: pl.Series, 
        views: pl.Series
    ) -> pl.Series:
        """Calculate engagement score from clicks and views."""
        return (clicks * 2 + views) / (views + 1e-6)
    
    def _calculate_freshness_score(self, age_hours: pl.Series) -> pl.Series:
        """Calculate freshness score based on content age."""
        return np.exp(-age_hours / 168)  # Decay over a week
    
    def _calculate_trending_score(
        self,
        recent_views: pl.Series,
        older_views: pl.Series
    ) -> pl.Series:
        """Calculate trending score based on view velocity."""
        return (recent_views * 24) / (older_views + 1e-6)
    
    def _calculate_quality_score(self, features: pl.DataFrame) -> pl.Series:
        """Calculate quality score from multiple signals."""
        # Simple quality score based on engagement metrics
        return (
            features["ctr_24h"] * 0.3 +
            features["trending_score"].clip(0, 1) * 0.3 +
            features["freshness_score"] * 0.2 +
            (features["views_24h"] / features["views_24h"].max()) * 0.2
        ).clip(0, 1)
    
    def encode_categorical_features(
        self,
        features: pd.DataFrame,
        categorical_columns: List[str]
    ) -> pd.DataFrame:
        """Encode categorical features.
        
        Args:
            features: DataFrame with features
            categorical_columns: List of categorical column names
            
        Returns:
            DataFrame with encoded features
        """
        encoded_features = features.copy()
        
        for col in categorical_columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                encoded_features[f"{col}_encoded"] = self.encoders[col].fit_transform(
                    features[col].fillna("unknown")
                )
            else:
                # Handle unseen categories
                known_categories = set(self.encoders[col].classes_)
                features[col] = features[col].apply(
                    lambda x: x if x in known_categories else "unknown"
                )
                encoded_features[f"{col}_encoded"] = self.encoders[col].transform(
                    features[col].fillna("unknown")
                )
        
        return encoded_features
    
    def scale_numerical_features(
        self,
        features: pd.DataFrame,
        numerical_columns: List[str],
        fit: bool = True
    ) -> pd.DataFrame:
        """Scale numerical features.
        
        Args:
            features: DataFrame with features
            numerical_columns: List of numerical column names
            fit: Whether to fit the scaler
            
        Returns:
            DataFrame with scaled features
        """
        scaled_features = features.copy()
        
        for col in numerical_columns:
            if col not in self.scalers:
                self.scalers[col] = StandardScaler()
            
            if fit:
                scaled_features[f"{col}_scaled"] = self.scalers[col].fit_transform(
                    features[[col]]
                )
            else:
                scaled_features[f"{col}_scaled"] = self.scalers[col].transform(
                    features[[col]]
                )
        
        return scaled_features
    
    def create_embedding_features(
        self,
        text_data: List[str],
        embedding_model: Optional[Any] = None
    ) -> np.ndarray:
        """Create text embedding features.
        
        Args:
            text_data: List of text strings
            embedding_model: Pre-trained embedding model
            
        Returns:
            Array of embeddings
        """
        if embedding_model is None:
            # Use simple TF-IDF as fallback
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=128)
            embeddings = vectorizer.fit_transform(text_data).toarray()
        else:
            # Use provided embedding model
            embeddings = embedding_model.encode(text_data)
        
        return embeddings
    
    def save_feature_stats(self, path: str):
        """Save feature statistics for monitoring.
        
        Args:
            path: Path to save statistics
        """
        import json
        
        stats = {
            "encoders": {
                col: list(encoder.classes_)
                for col, encoder in self.encoders.items()
            },
            "scalers": {
                col: {
                    "mean": float(scaler.mean_[0]),
                    "std": float(scaler.scale_[0])
                }
                for col, scaler in self.scalers.items()
            },
            "feature_stats": self.feature_stats
        }
        
        with open(path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved feature statistics to {path}")

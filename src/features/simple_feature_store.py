"""Simple feature store mock for demo purposes."""

import pandas as pd
from loguru import logger
from typing import Dict, List, Any, Optional


class SimpleFeatureStore:
    """Simple in-memory feature store for demo."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize simple feature store."""
        self.config = config
        self.project_name = config.get("project_name", "recsys_features")
        logger.info(f"Initialized SimpleFeatureStore for project: {self.project_name}")
        
    def get_online_features(
        self,
        entity_dict: Dict[str, List[Any]],
        features: List[str],
        request_features: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Get online features (mock implementation)."""
        logger.debug(f"Getting online features for {len(entity_dict)} entities")
        
        # Return dummy features
        num_entities = len(list(entity_dict.values())[0]) if entity_dict else 1
        
        # Create dummy feature data
        feature_data = {}
        for feature in features:
            if "age" in feature.lower():
                feature_data[feature] = [25] * num_entities
            elif "gender" in feature.lower():
                feature_data[feature] = ["M"] * num_entities
            elif "clicks" in feature.lower():
                feature_data[feature] = [5] * num_entities
            elif "views" in feature.lower():
                feature_data[feature] = [20] * num_entities
            elif "ctr" in feature.lower():
                feature_data[feature] = [0.25] * num_entities
            else:
                feature_data[feature] = [1.0] * num_entities
        
        # Add request features if provided
        if request_features:
            for key, value in request_features.items():
                feature_data[key] = [value] * num_entities
        
        return pd.DataFrame(feature_data)
    
    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        features: List[str],
        timestamp_col: str = "event_timestamp"
    ) -> pd.DataFrame:
        """Get historical features (mock implementation)."""
        logger.info(f"Getting historical features for {len(entity_df)} entities")
        
        # Return dummy features
        feature_data = {}
        for feature in features:
            if "age" in feature.lower():
                feature_data[feature] = [25] * len(entity_df)
            elif "gender" in feature.lower():
                feature_data[feature] = ["M"] * len(entity_df)
            else:
                feature_data[feature] = [1.0] * len(entity_df)
        
        return pd.DataFrame(feature_data)
    
    def push_streaming_features(
        self,
        feature_data: pd.DataFrame,
        feature_view_name: str
    ):
        """Push streaming features (mock implementation)."""
        logger.debug(f"Mock pushing {len(feature_data)} features to {feature_view_name}")
        pass
    
    def materialize_features(
        self,
        start_date,
        end_date,
        feature_views: Optional[List[str]] = None
    ):
        """Materialize features (mock implementation)."""
        logger.info(f"Mock materializing features from {start_date} to {end_date}")
        pass

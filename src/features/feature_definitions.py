"""Feature definitions for the recommendation system."""

from datetime import timedelta
from typing import List, Optional

from .simple_feast_mocks import (
    Entity,
    FeatureView,
    Field,
    FileSource,
    PushSource,
    RequestSource,
    ValueType,
    ParquetFormat,
    Float32,
    Float64,
    Int64,
    String,
    UnixTimestamp,
)


class UserFeatures:
    """User feature definitions."""
    
    @staticmethod
    def get_entity() -> Entity:
        """Get user entity definition."""
        return Entity(
            name="user",
            description="User entity for recommendations",
            value_type=ValueType.STRING,
        )
    
    @staticmethod
    def get_profile_features() -> FeatureView:
        """Get user profile feature view."""
        user_profile_source = FileSource(
            name="user_profile_source",
            path="feature_store/data/user_profiles.parquet",
            timestamp_field="event_timestamp",
            created_timestamp_column="created_timestamp",
        )
        
        return FeatureView(
            name="user_profile_features",
            entities=[UserFeatures.get_entity()],
            ttl=timedelta(days=30),
            schema=[
                Field(name="user_id", dtype=String),
                Field(name="age", dtype=Int64),
                Field(name="gender", dtype=String),
                Field(name="location", dtype=String),
                Field(name="signup_days", dtype=Int64),
                Field(name="total_interactions", dtype=Int64),
                Field(name="avg_session_duration", dtype=Float32),
                Field(name="preferred_categories", dtype=String),
                Field(name="device_type", dtype=String),
                Field(name="subscription_tier", dtype=String),
            ],
            source=user_profile_source,
            tags={"team": "ml", "type": "profile"},
        )
    
    @staticmethod
    def get_activity_features() -> FeatureView:
        """Get user activity feature view."""
        user_activity_source = FileSource(
            name="user_activity_source",
            path="feature_store/data/user_activities.parquet",
            timestamp_field="event_timestamp",
        )
        
        return FeatureView(
            name="user_activity_features",
            entities=[UserFeatures.get_entity()],
            ttl=timedelta(hours=24),
            schema=[
                Field(name="clicks_1h", dtype=Int64),
                Field(name="clicks_24h", dtype=Int64),
                Field(name="clicks_7d", dtype=Int64),
                Field(name="views_1h", dtype=Int64),
                Field(name="views_24h", dtype=Int64),
                Field(name="views_7d", dtype=Int64),
                Field(name="avg_view_time_1h", dtype=Float32),
                Field(name="avg_view_time_24h", dtype=Float32),
                Field(name="bounce_rate_24h", dtype=Float32),
                Field(name="diversity_score", dtype=Float32),
            ],
            source=user_activity_source,
            tags={"team": "ml", "type": "activity"},
        )


class ItemFeatures:
    """Item feature definitions."""
    
    @staticmethod
    def get_entity() -> Entity:
        """Get item entity definition."""
        return Entity(
            name="item",
            description="Item entity for recommendations",
            value_type=ValueType.STRING,
        )
    
    @staticmethod
    def get_content_features() -> FeatureView:
        """Get item content feature view."""
        item_content_source = FileSource(
            name="item_content_source",
            path="feature_store/data/item_content.parquet",
            timestamp_field="event_timestamp",
        )
        
        return FeatureView(
            name="item_content_features",
            entities=[ItemFeatures.get_entity()],
            ttl=timedelta(days=7),
            schema=[
                Field(name="item_id", dtype=String),
                Field(name="title", dtype=String),
                Field(name="category", dtype=String),
                Field(name="subcategory", dtype=String),
                Field(name="tags", dtype=String),
                Field(name="publish_timestamp", dtype=UnixTimestamp),
                Field(name="content_length", dtype=Int64),
                Field(name="language", dtype=String),
                Field(name="author", dtype=String),
                Field(name="quality_score", dtype=Float32),
            ],
            source=item_content_source,
            tags={"team": "ml", "type": "content"},
        )
    
    @staticmethod
    def get_popularity_features() -> FeatureView:
        """Get item popularity feature view."""
        item_popularity_source = FileSource(
            name="item_popularity_source",
            path="feature_store/data/item_popularity.parquet",
            timestamp_field="event_timestamp",
        )
        
        return FeatureView(
            name="item_popularity_features",
            entities=[ItemFeatures.get_entity()],
            ttl=timedelta(hours=6),
            schema=[
                Field(name="views_1h", dtype=Int64),
                Field(name="views_24h", dtype=Int64),
                Field(name="views_7d", dtype=Int64),
                Field(name="clicks_1h", dtype=Int64),
                Field(name="clicks_24h", dtype=Int64),
                Field(name="clicks_7d", dtype=Int64),
                Field(name="ctr_1h", dtype=Float32),
                Field(name="ctr_24h", dtype=Float32),
                Field(name="ctr_7d", dtype=Float32),
                Field(name="avg_rating", dtype=Float32),
                Field(name="rating_count", dtype=Int64),
                Field(name="trending_score", dtype=Float32),
            ],
            source=item_popularity_source,
            tags={"team": "ml", "type": "popularity"},
        )


class InteractionFeatures:
    """User-Item interaction features."""
    
    @staticmethod
    def get_interaction_features() -> FeatureView:
        """Get interaction feature view."""
        interaction_source = FileSource(
            name="interaction_source",
            path="feature_store/data/interactions.parquet",
            timestamp_field="event_timestamp",
        )
        
        return FeatureView(
            name="interaction_features",
            entities=[UserFeatures.get_entity(), ItemFeatures.get_entity()],
            ttl=timedelta(hours=1),
            schema=[
                Field(name="user_item_clicks", dtype=Int64),
                Field(name="user_item_views", dtype=Int64),
                Field(name="user_item_total_time", dtype=Float32),
                Field(name="user_item_last_interaction", dtype=UnixTimestamp),
                Field(name="user_category_affinity", dtype=Float32),
                Field(name="user_author_affinity", dtype=Float32),
            ],
            source=interaction_source,
            tags={"team": "ml", "type": "interaction"},
        )


class StreamingFeatures:
    """Real-time streaming features."""
    
    @staticmethod
    def get_streaming_source() -> PushSource:
        """Get streaming push source for real-time features."""
        return PushSource(
            name="streaming_features_push",
            batch_source=FileSource(
                name="streaming_features_batch",
                path="feature_store/data/streaming_features.parquet",
                timestamp_field="event_timestamp",
            ),
        )
    
    @staticmethod
    def get_realtime_user_features() -> FeatureView:
        """Get real-time user feature view."""
        streaming_source = StreamingFeatures.get_streaming_source()
        
        return FeatureView(
            name="realtime_user_features",
            entities=[UserFeatures.get_entity()],
            ttl=timedelta(minutes=5),
            schema=[
                Field(name="clicks_5min", dtype=Int64),
                Field(name="views_5min", dtype=Int64),
                Field(name="categories_5min", dtype=String),
                Field(name="avg_dwell_time_5min", dtype=Float32),
                Field(name="bounce_rate_5min", dtype=Float32),
                Field(name="session_depth", dtype=Int64),
                Field(name="current_context", dtype=String),
            ],
            source=streaming_source,
            tags={"team": "ml", "type": "streaming"},
        )
    
    @staticmethod
    def get_realtime_item_features() -> FeatureView:
        """Get real-time item feature view."""
        streaming_source = StreamingFeatures.get_streaming_source()
        
        return FeatureView(
            name="realtime_item_features",
            entities=[ItemFeatures.get_entity()],
            ttl=timedelta(minutes=5),
            schema=[
                Field(name="views_5min", dtype=Int64),
                Field(name="clicks_5min", dtype=Int64),
                Field(name="ctr_5min", dtype=Float32),
                Field(name="velocity_score", dtype=Float32),
                Field(name="freshness_score", dtype=Float32),
            ],
            source=streaming_source,
            tags={"team": "ml", "type": "streaming"},
        )


class RequestFeatures:
    """Request-time features."""
    
    @staticmethod
    def get_request_source() -> RequestSource:
        """Get request source for online features."""
        return RequestSource(
            name="request_features",
            schema=[
                Field(name="time_of_day", dtype=Int64),
                Field(name="day_of_week", dtype=Int64),
                Field(name="device_type", dtype=String),
                Field(name="location", dtype=String),
                Field(name="session_id", dtype=String),
                Field(name="referrer", dtype=String),
                Field(name="query", dtype=String),
            ],
        )

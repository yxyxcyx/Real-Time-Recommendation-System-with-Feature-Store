"""API module for the recommendation system.

This module provides a clean separation between the API layer and the
service/business logic. Routes are organized into submodules:

- routes/health.py: Health checks and metrics
- routes/recommendations.py: Core recommendation endpoints
- routes/management.py: Model and feature management

Usage:
    from src.api import create_app
    app = create_app()
"""

from .app import create_app, get_service
from .schemas import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendationItem,
    FeedbackRequest,
    HealthResponse,
    StatusResponse,
)

__all__ = [
    # App factory
    "create_app",
    "get_service",
    # Schemas
    "RecommendationRequest",
    "RecommendationResponse",
    "RecommendationItem",
    "FeedbackRequest",
    "HealthResponse",
    "StatusResponse",
]

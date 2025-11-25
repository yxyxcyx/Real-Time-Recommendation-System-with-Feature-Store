"""Pydantic schemas for API request/response models.

This module defines all the request and response models used by the API.
Separating schemas from routes makes the API easier to test and maintain.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    """Request model for recommendations."""
    
    user_id: str = Field(..., description="User ID")
    context: Optional[Dict[str, Any]] = Field(
        default={},
        description="Request context (device, location, etc.)"
    )
    num_recommendations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of recommendations to return"
    )
    filter_categories: Optional[List[str]] = Field(
        default=None,
        description="Categories to filter"
    )
    exclude_items: Optional[List[str]] = Field(
        default=None,
        description="Items to exclude from recommendations"
    )


class RecommendationItem(BaseModel):
    """Single recommendation item."""
    
    item_id: str
    score: float
    title: Optional[str] = None
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    
    user_id: str
    recommendations: List[RecommendationItem]
    model_version: str
    response_time_ms: float
    debug_info: Optional[Dict[str, Any]] = None


class FeedbackRequest(BaseModel):
    """Request model for user feedback."""
    
    user_id: str
    item_id: str
    event_type: str = Field(..., description="click, view, like, dislike, etc.")
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str
    timestamp: str
    initialized: bool


class StatusResponse(BaseModel):
    """Generic status response."""
    
    status: str
    message: Optional[str] = None

"""Recommendation endpoints.

This module contains the core recommendation API endpoints.
"""

from fastapi import APIRouter

from ..schemas import (
    RecommendationRequest,
    RecommendationResponse,
    FeedbackRequest,
)

router = APIRouter(tags=["Recommendations"])


def register_recommendation_routes(app, service) -> None:
    """Register recommendation-related routes.
    
    Args:
        app: FastAPI application instance
        service: RecommendationService instance
    """
    
    @app.post("/recommend", response_model=RecommendationResponse)
    async def get_recommendations(request: RecommendationRequest):
        """Get personalized recommendations for a user.
        
        Args:
            request: Recommendation request with user_id and options
            
        Returns:
            Personalized recommendations with scores and metadata
        """
        return await service.get_recommendations(request)
    
    @app.post("/feedback")
    async def submit_feedback(feedback: FeedbackRequest):
        """Submit user feedback for an item.
        
        Args:
            feedback: Feedback request with user_id, item_id, and event_type
            
        Returns:
            Status of the feedback submission
        """
        return await service.process_feedback(feedback)

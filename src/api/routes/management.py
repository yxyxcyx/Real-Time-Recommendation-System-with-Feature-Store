"""Management endpoints for models, features, and A/B testing.

These endpoints are used for operational management of the recommendation system.
"""

from datetime import datetime

from fastapi import APIRouter, Query

router = APIRouter(tags=["Management"])


def register_management_routes(app, service) -> None:
    """Register management routes for models and features.
    
    Args:
        app: FastAPI application instance
        service: RecommendationService instance
    """
    
    # Model management endpoints
    @app.post("/models/reload")
    async def reload_models():
        """Reload ML models from disk.
        
        Returns:
            Status of the reload operation
        """
        service._load_models()
        return {"status": "success", "message": "Models reloaded"}
    
    # Feature store endpoints
    @app.post("/features/materialize")
    async def materialize_features(
        start_date: datetime = Query(..., description="Start date for materialization"),
        end_date: datetime = Query(default=None, description="End date (defaults to now)")
    ):
        """Materialize features to online store.
        
        Args:
            start_date: Start date for feature materialization
            end_date: End date (defaults to current time)
            
        Returns:
            Status of the materialization
        """
        end_date = end_date or datetime.now()
        service.feature_store.materialize_features(start_date, end_date)
        return {"status": "success", "message": "Materialization started"}
    
    # A/B testing endpoints (shadow deployment removed)
    @app.get("/ab/status")
    async def ab_test_status():
        """Get A/B test status.
        
        Returns:
            Current A/B test configuration status
        """
        return {"enabled": False, "message": "Shadow deployment feature removed"}
    
    @app.post("/ab/configure")
    async def configure_ab_test(
        traffic_percentage: float = Query(..., ge=0, le=100, description="Traffic percentage for shadow model")
    ):
        """Configure A/B test traffic split.
        
        Args:
            traffic_percentage: Percentage of traffic to route to shadow model
            
        Returns:
            Status of the configuration change
        """
        return {"status": "error", "message": "Shadow deployment feature removed"}

"""Health and metrics endpoints.

These endpoints are used for monitoring and observability.
"""

from datetime import datetime

from fastapi import APIRouter
from prometheus_client import generate_latest

from ..schemas import HealthResponse

router = APIRouter(tags=["Health"])


def register_health_routes(app, service) -> None:
    """Register health check and metrics routes.
    
    Args:
        app: FastAPI application instance
        service: RecommendationService instance
    """
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "initialized": service.initialized
        }
    
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return generate_latest()

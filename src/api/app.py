"""FastAPI application factory.

This module provides the main application factory that assembles
all routes and middleware into a complete FastAPI application.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .routes import (
    register_health_routes,
    register_recommendation_routes,
    register_management_routes,
)

# Import service from serving module
from ..serving.service import RecommendationService


# Global service instance
_service = RecommendationService()


def get_service() -> RecommendationService:
    """Get the global service instance.
    
    Returns:
        RecommendationService instance
    """
    return _service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.
    
    Handles startup and shutdown events for the application.
    """
    # Startup
    logger.info("Starting recommendation service...")
    await _service.initialize()
    yield
    # Shutdown
    logger.info("Shutting down recommendation service...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    
    This factory function creates a new FastAPI instance with:
    - CORS middleware configured
    - All routes registered
    - Lifespan manager for startup/shutdown
    
    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="Real-Time Recommendation System",
        description="Production-grade recommendation system with feature store integration",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register all routes
    register_health_routes(app, _service)
    register_recommendation_routes(app, _service)
    register_management_routes(app, _service)
    
    logger.info("FastAPI application created with all routes registered")
    
    return app

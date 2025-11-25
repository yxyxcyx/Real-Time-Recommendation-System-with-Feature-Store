"""API routes module.

This module provides route registration functions that can be used
to modularly add endpoints to the FastAPI application.
"""

from .health import register_health_routes
from .recommendations import register_recommendation_routes
from .management import register_management_routes

__all__ = [
    "register_health_routes",
    "register_recommendation_routes",
    "register_management_routes",
]

"""Serving components for the recommendation system.

This module provides:
- RetrievalEngine: ANN-based item retrieval (Faiss/Milvus)
- RecommendationService: Core business logic for recommendations
- run_server: Server entry point

Note: The API layer is now at src/api/. Use `from src.api import create_app`.
"""

from .retrieval import RetrievalEngine, FaissIndex, MilvusIndex
from .service import RecommendationService, run_server

__all__ = [
    "RetrievalEngine",
    "FaissIndex",
    "MilvusIndex",
    "RecommendationService",
    "run_server",
]
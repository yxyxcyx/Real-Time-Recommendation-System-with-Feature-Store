"""Recommendation service business logic.

This module contains the core RecommendationService class that handles
all recommendation logic. The API layer (src/api/) calls this service.
"""

import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import HTTPException
from loguru import logger
from prometheus_client import Counter, Histogram, REGISTRY

from ..config import get_config
from ..features import FeatureStore
from ..models import TwoTowerModel, create_ranking_model, create_two_tower_model
from .retrieval import RetrievalEngine


# Prometheus Metrics - Handle duplicate registration gracefully
def _get_or_create_counter(name, description, labels):
    """Get existing counter or create new one."""
    try:
        return Counter(name, description, labels)
    except ValueError:
        # Already registered, find it
        for collector in REGISTRY._collector_to_names:
            if name in REGISTRY._collector_to_names.get(collector, []):
                return collector
        # Fallback: clear and recreate
        return Counter(name, description, labels)


def _get_or_create_histogram(name, description, labels):
    """Get existing histogram or create new one."""
    try:
        return Histogram(name, description, labels)
    except ValueError:
        for collector in REGISTRY._collector_to_names:
            if name in REGISTRY._collector_to_names.get(collector, []):
                return collector
        return Histogram(name, description, labels)


recommendation_counter = _get_or_create_counter(
    "recommendations_total",
    "Total number of recommendations served",
    ["model_version", "status"]
)
recommendation_latency = _get_or_create_histogram(
    "recommendation_latency_seconds",
    "Recommendation request latency",
    ["model_version", "stage"]
)


class RecommendationService:
    """Main recommendation service.
    
    This service handles:
    - Model loading and initialization
    - User embedding computation
    - Item retrieval via ANN search
    - Ranking and reranking
    - Response building
    
    Note: Uses in-memory caching for demo. In production, use Redis.
    """
    
    def __init__(self):
        """Initialize recommendation service."""
        self.config = get_config()
        self.initialized = False
        
        # Core components
        self.feature_store = None
        self.two_tower_model = None
        self.ranking_model = None
        self.retrieval_engine = None
        
        # In-memory cache (for demo - use Redis in production)
        self.item_metadata_cache = {}
        self.user_embedding_cache = {}
        self.cache_ttl = 300
        
    async def initialize(self):
        """Initialize all components."""
        if self.initialized:
            return
        
        logger.info("Initializing recommendation service...")
        
        try:
            # Initialize feature store
            self.feature_store = FeatureStore(
                self.config.get("feature_store")
            )
            
            # Load models
            self._load_models()
            
            # Initialize retrieval engine
            self.retrieval_engine = RetrievalEngine(
                self.config.get("retrieval")
            )
            
            # Pre-compute item embeddings
            await self._precompute_item_embeddings()
            
            self.initialized = True
            logger.info("Recommendation service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize service: {e}")
            raise
    
    def _load_models(self):
        """Load ML models."""
        logger.info("Loading models...")
        
        import torch
        
        model_config = self.config.get("model.two_tower")
        self.two_tower_model = create_two_tower_model(model_config)
        self.two_tower_model.eval()
        
        # Try to load checkpoint
        checkpoint_path = "models/checkpoints/two_tower_latest.pth"
        try:
            self.two_tower_model.load_model(checkpoint_path)
            logger.info(f"Loaded Two-Tower model from {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Could not load Two-Tower model: {e}")
            logger.warning("Using random initialization for Two-Tower model.")

        # Load ranking model
        ranking_config = self.config.get("model.ranking")
        model_type = ranking_config.get("model_type", "xgboost")
        self.ranking_model = create_ranking_model(model_type, ranking_config)

        ranking_checkpoint = f"models/checkpoints/{model_type}_ranker.pkl"
        try:
            self.ranking_model.load(ranking_checkpoint)
            logger.info(f"Loaded ranking model from {ranking_checkpoint}")
        except Exception as e:
            logger.warning(f"Could not load ranking model: {e}")
            self.ranking_model = None
    
    async def _precompute_item_embeddings(self):
        """Precompute and index all item embeddings."""
        logger.info("Precomputing item embeddings...")
        
        import torch
        
        # Generate sample items for demo
        num_items = 1000
        item_ids = [f"item_{i}" for i in range(num_items)]
        
        item_features = {
            "numerical": torch.randn(num_items, 50),
            "categorical": {},
            "content_embeddings": torch.randn(num_items, 768)
        }
        
        with torch.no_grad():
            item_embeddings = self.two_tower_model.get_item_embeddings(item_features)
        
        self.retrieval_engine.build_index(
            item_embeddings.numpy(),
            item_ids
        )
        
        logger.info(f"Indexed {num_items} items")
    
    async def get_recommendations(self, request) -> dict:
        """Get recommendations for a user.
        
        Args:
            request: RecommendationRequest with user_id, num_recommendations, etc.
            
        Returns:
            RecommendationResponse dict
        """
        start_time = time.time()
        model_version = "primary"
        
        try:
            # Get user embedding
            user_embedding = await self._get_user_embedding(request.user_id)
            
            # Retrieval stage
            with recommendation_latency.labels(
                model_version=model_version,
                stage="retrieval"
            ).time():
                candidates, scores, retrieval_metrics = self.retrieval_engine.retrieve(
                    user_embedding,
                    k=request.num_recommendations * 10,
                    filter_ids=None
                )
            
            # Get candidate features
            candidate_features = await self._get_candidate_features(
                request.user_id,
                candidates[0] if candidates else []
            )
            
            # Ranking stage
            with recommendation_latency.labels(
                model_version=model_version,
                stage="ranking"
            ).time():
                if self.ranking_model and candidate_features is not None:
                    ranking_scores = self.ranking_model.predict(candidate_features)
                    ranked_indices = np.argsort(ranking_scores)[::-1]
                    final_candidates = [candidates[0][i] for i in ranked_indices[:request.num_recommendations]]
                    final_scores = [float(ranking_scores[i]) for i in ranked_indices[:request.num_recommendations]]
                else:
                    final_candidates = candidates[0][:request.num_recommendations]
                    final_scores = scores[0][:request.num_recommendations]
            
            # Build response
            recommendations = []
            for item_id, score in zip(final_candidates, final_scores):
                metadata = await self._get_item_metadata(item_id)
                recommendations.append({
                    "item_id": item_id,
                    "score": score,
                    "title": metadata.get("title"),
                    "category": metadata.get("category"),
                    "metadata": metadata
                })
            
            response_time = (time.time() - start_time) * 1000
            
            recommendation_counter.labels(
                model_version=model_version,
                status="success"
            ).inc()
            
            return {
                "user_id": request.user_id,
                "recommendations": recommendations,
                "model_version": model_version,
                "response_time_ms": response_time,
                "debug_info": {
                    "retrieval_latency_ms": retrieval_metrics.get("latency_ms"),
                    "num_candidates": len(candidates[0]) if candidates else 0,
                } if self.config.get("system.debug") else None
            }
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            recommendation_counter.labels(
                model_version=model_version,
                status="error"
            ).inc()
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _get_user_embedding(self, user_id: str) -> np.ndarray:
        """Get user embedding with caching."""
        if user_id in self.user_embedding_cache:
            cache_entry = self.user_embedding_cache[user_id]
            if time.time() - cache_entry["timestamp"] < self.cache_ttl:
                return cache_entry["embedding"]
        
        import torch
        
        try:
            user_features = self.feature_store.get_online_features(
                {"user": [user_id]},
                ["user_profile_features:*", "user_activity_features:*"]
            )
        except:
            user_features = {
                "numerical": torch.randn(1, 50),
                "categorical": {}
            }
        
        with torch.no_grad():
            if isinstance(user_features, dict) and "numerical" in user_features:
                user_embedding = self.two_tower_model.get_user_embeddings(user_features)
            else:
                user_embedding = self.two_tower_model.get_user_embeddings({
                    "numerical": torch.randn(1, 50),
                    "categorical": {}
                })
        
        embedding_np = user_embedding.numpy()
        
        self.user_embedding_cache[user_id] = {
            "embedding": embedding_np,
            "timestamp": time.time()
        }
        
        return embedding_np
    
    async def _get_candidate_features(
        self,
        user_id: str,
        item_ids: List[str]
    ) -> Optional[Any]:
        """Get features for ranking candidates."""
        if not item_ids:
            return None
        
        try:
            import pandas as pd
            
            features = pd.DataFrame({
                "user_item_clicks": np.random.randint(0, 10, len(item_ids)),
                "user_item_views": np.random.randint(0, 50, len(item_ids)),
                "item_popularity": np.random.uniform(0, 1, len(item_ids)),
                "item_freshness": np.random.uniform(0, 1, len(item_ids)),
                "user_category_affinity": np.random.uniform(0, 1, len(item_ids)),
            })
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting candidate features: {e}")
            return None
    
    async def _get_item_metadata(self, item_id: str) -> Dict[str, Any]:
        """Get item metadata with caching."""
        if item_id in self.item_metadata_cache:
            return self.item_metadata_cache[item_id]
        
        metadata = {
            "title": f"Item {item_id}",
            "category": random.choice(["tech", "sports", "entertainment", "news"]),
            "publish_date": datetime.now().isoformat(),
            "author": f"Author {random.randint(1, 100)}"
        }
        
        self.item_metadata_cache[item_id] = metadata
        return metadata
    
    async def process_feedback(self, feedback) -> dict:
        """Process user feedback.
        
        Args:
            feedback: FeedbackRequest with user_id, item_id, event_type
            
        Returns:
            Status dict
        """
        logger.info(
            f"Feedback: User {feedback.user_id} {feedback.event_type} "
            f"item {feedback.item_id}"
        )
        return {"status": "success", "message": "Feedback recorded"}


def run_server():
    """Run the FastAPI server.
    
    This function starts the uvicorn server using the new modular
    API structure at src/api/.
    """
    import uvicorn
    
    config = get_config()
    workers = config.get("serving.api.workers", 4)
    
    if workers > 1:
        uvicorn.run(
            "src.api:create_app",
            factory=True,
            host=config.get("serving.api.host", "0.0.0.0"),
            port=config.get("serving.api.port", 8000),
            workers=workers,
            log_level="info"
        )
    else:
        from src.api import create_app
        app = create_app()
        uvicorn.run(
            app,
            host=config.get("serving.api.host", "0.0.0.0"),
            port=config.get("serving.api.port", 8000),
            log_level="info"
        )

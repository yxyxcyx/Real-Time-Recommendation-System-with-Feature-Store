"""FastAPI service for real-time recommendations with shadow deployment."""

import random
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest

from ..config import get_config
from ..features import FeatureStore
from ..models import TwoTowerModel, create_ranking_model
from .retrieval import RetrievalEngine
from .shadow_deployment import ShadowDeploymentManager


# Request/Response Models
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


# Metrics
recommendation_counter = Counter(
    "recommendations_total",
    "Total number of recommendations served",
    ["model_version", "status"]
)

recommendation_latency = Histogram(
    "recommendation_latency_seconds",
    "Recommendation request latency",
    ["model_version", "stage"]
)


class RecommendationService:
    """Main recommendation service."""
    
    def __init__(self):
        """Initialize recommendation service."""
        self.config = get_config()
        self.initialized = False
        
        # Core components
        self.feature_store = None
        self.two_tower_model = None
        self.ranking_model = None
        self.retrieval_engine = None
        self.shadow_deployment = None
        
        # Cache
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
            
            # Initialize shadow deployment
            if self.config.get("serving.shadow_deployment.enabled"):
                self.shadow_deployment = ShadowDeploymentManager(
                    self.config.get("serving.shadow_deployment")
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
        
        # Load Two-Tower model
        import torch
        from ..models import create_two_tower_model
        
        model_config = self.config.get("model.two_tower")
        self.two_tower_model = create_two_tower_model(model_config)
        
        # Try to load checkpoint
        checkpoint_path = "models/checkpoints/two_tower_latest.pth"
        try:
            self.two_tower_model.load_model(checkpoint_path)
            logger.info(f"Loaded Two-Tower model from {checkpoint_path}")
        except:
            logger.warning("No checkpoint found, using random initialization")
        
        # Load ranking model
        ranking_config = self.config.get("model.ranking")
        model_type = ranking_config.get("model_type", "xgboost")
        self.ranking_model = create_ranking_model(model_type, ranking_config)
        
        # Try to load ranking model
        ranking_checkpoint = f"models/checkpoints/{model_type}_ranker.pkl"
        try:
            self.ranking_model.load(ranking_checkpoint)
            logger.info(f"Loaded ranking model from {ranking_checkpoint}")
        except:
            logger.warning("No ranking model checkpoint found")
    
    async def _precompute_item_embeddings(self):
        """Precompute and index all item embeddings."""
        logger.info("Precomputing item embeddings...")
        
        # Get all items from feature store
        # In production, this would be from database
        import pandas as pd
        import torch
        
        # Generate sample items for demo
        num_items = 1000
        item_ids = [f"item_{i}" for i in range(num_items)]
        
        # Generate random features for demo
        item_features = {
            "numerical": torch.randn(num_items, 50),
            "categorical": {},
            "content_embeddings": torch.randn(num_items, 768)
        }
        
        # Get item embeddings
        with torch.no_grad():
            item_embeddings = self.two_tower_model.get_item_embeddings(item_features)
        
        # Build retrieval index
        self.retrieval_engine.build_index(
            item_embeddings.numpy(),
            item_ids
        )
        
        logger.info(f"Indexed {num_items} items")
    
    async def get_recommendations(
        self,
        request: RecommendationRequest
    ) -> RecommendationResponse:
        """Get recommendations for a user.
        
        Args:
            request: Recommendation request
            
        Returns:
            Recommendation response
        """
        start_time = time.time()
        
        # Determine which model to use (for shadow deployment)
        use_shadow = False
        if self.shadow_deployment:
            use_shadow = self.shadow_deployment.should_use_shadow(request.user_id)
        
        model_version = "shadow" if use_shadow else "primary"
        
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
                    k=request.num_recommendations * 10,  # Over-retrieve for ranking
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
                    
                    # Rerank candidates
                    ranked_indices = np.argsort(ranking_scores)[::-1]
                    final_candidates = [candidates[0][i] for i in ranked_indices[:request.num_recommendations]]
                    final_scores = [float(ranking_scores[i]) for i in ranked_indices[:request.num_recommendations]]
                else:
                    # Use retrieval scores
                    final_candidates = candidates[0][:request.num_recommendations]
                    final_scores = scores[0][:request.num_recommendations]
            
            # Build response
            recommendations = []
            for item_id, score in zip(final_candidates, final_scores):
                metadata = await self._get_item_metadata(item_id)
                recommendations.append(
                    RecommendationItem(
                        item_id=item_id,
                        score=score,
                        title=metadata.get("title"),
                        category=metadata.get("category"),
                        metadata=metadata
                    )
                )
            
            # Calculate response time
            response_time = (time.time() - start_time) * 1000
            
            # Log metrics
            recommendation_counter.labels(
                model_version=model_version,
                status="success"
            ).inc()
            
            # Log shadow deployment comparison if applicable
            if use_shadow and self.shadow_deployment:
                self.shadow_deployment.log_comparison(
                    request.user_id,
                    recommendations,
                    response_time
                )
            
            return RecommendationResponse(
                user_id=request.user_id,
                recommendations=recommendations,
                model_version=model_version,
                response_time_ms=response_time,
                debug_info={
                    "retrieval_latency_ms": retrieval_metrics.get("latency_ms"),
                    "num_candidates": len(candidates[0]) if candidates else 0,
                    "cache_hit": retrieval_metrics.get("cache_hit", False)
                } if self.config.get("system.debug") else None
            )
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            recommendation_counter.labels(
                model_version=model_version,
                status="error"
            ).inc()
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _get_user_embedding(self, user_id: str) -> np.ndarray:
        """Get user embedding.
        
        Args:
            user_id: User ID
            
        Returns:
            User embedding vector
        """
        # Check cache
        if user_id in self.user_embedding_cache:
            cache_entry = self.user_embedding_cache[user_id]
            if time.time() - cache_entry["timestamp"] < self.cache_ttl:
                return cache_entry["embedding"]
        
        # Get user features from feature store
        try:
            user_features = self.feature_store.get_online_features(
                {"user": [user_id]},
                ["user_profile_features:*", "user_activity_features:*"]
            )
        except:
            # Use default features if not found
            import torch
            user_features = {
                "numerical": torch.randn(1, 50),
                "categorical": {}
            }
        
        # Compute embedding
        import torch
        with torch.no_grad():
            if isinstance(user_features, dict) and "numerical" in user_features:
                user_embedding = self.two_tower_model.get_user_embeddings(user_features)
            else:
                # Convert DataFrame to tensor format
                user_embedding = self.two_tower_model.get_user_embeddings({
                    "numerical": torch.randn(1, 50),
                    "categorical": {}
                })
        
        embedding_np = user_embedding.numpy()
        
        # Update cache
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
        """Get features for ranking candidates.
        
        Args:
            user_id: User ID
            item_ids: List of candidate item IDs
            
        Returns:
            Features for ranking or None
        """
        if not item_ids:
            return None
        
        try:
            # Get features from feature store
            import pandas as pd
            
            # Create interaction pairs
            interaction_data = pd.DataFrame({
                "user_id": [user_id] * len(item_ids),
                "item_id": item_ids
            })
            
            # Get features (simplified for demo)
            # In production, would fetch from feature store
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
        """Get item metadata.
        
        Args:
            item_id: Item ID
            
        Returns:
            Item metadata
        """
        # Check cache
        if item_id in self.item_metadata_cache:
            return self.item_metadata_cache[item_id]
        
        # Fetch metadata (simplified for demo)
        metadata = {
            "title": f"Item {item_id}",
            "category": random.choice(["tech", "sports", "entertainment", "news"]),
            "publish_date": datetime.now().isoformat(),
            "author": f"Author {random.randint(1, 100)}"
        }
        
        # Update cache
        self.item_metadata_cache[item_id] = metadata
        
        return metadata
    
    async def process_feedback(self, feedback: FeedbackRequest):
        """Process user feedback.
        
        Args:
            feedback: Feedback request
        """
        logger.info(
            f"Feedback: User {feedback.user_id} {feedback.event_type} "
            f"item {feedback.item_id}"
        )
        
        # Send to Kafka for streaming processing
        # In production, would actually send to Kafka
        
        # Update metrics
        if feedback.event_type == "click":
            # Track CTR
            pass
        
        return {"status": "success", "message": "Feedback recorded"}


# Global service instance
service = RecommendationService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    await service.initialize()
    yield
    # Shutdown
    logger.info("Shutting down recommendation service...")


def create_app() -> FastAPI:
    """Create FastAPI application.
    
    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="Real-Time Recommendation System",
        description="SOTA recommendation system with feature store and shadow deployment",
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
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "initialized": service.initialized
        }
    
    # Metrics endpoint
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return generate_latest()
    
    # Main recommendation endpoint
    @app.post("/recommend", response_model=RecommendationResponse)
    async def get_recommendations(request: RecommendationRequest):
        """Get personalized recommendations for a user."""
        return await service.get_recommendations(request)
    
    # Feedback endpoint
    @app.post("/feedback")
    async def submit_feedback(feedback: FeedbackRequest):
        """Submit user feedback for an item."""
        return await service.process_feedback(feedback)
    
    # Model management endpoints
    @app.post("/models/reload")
    async def reload_models():
        """Reload ML models."""
        service._load_models()
        return {"status": "success", "message": "Models reloaded"}
    
    # Feature store endpoints
    @app.post("/features/materialize")
    async def materialize_features(
        start_date: datetime = Query(...),
        end_date: datetime = Query(default=None)
    ):
        """Materialize features to online store."""
        end_date = end_date or datetime.now()
        service.feature_store.materialize_features(start_date, end_date)
        return {"status": "success", "message": "Materialization started"}
    
    # A/B testing endpoints
    @app.get("/ab/status")
    async def ab_test_status():
        """Get A/B test status."""
        if service.shadow_deployment:
            return service.shadow_deployment.get_status()
        return {"enabled": False}
    
    @app.post("/ab/configure")
    async def configure_ab_test(traffic_percentage: float = Query(..., ge=0, le=100)):
        """Configure A/B test traffic split."""
        if service.shadow_deployment:
            service.shadow_deployment.set_traffic_percentage(traffic_percentage)
            return {"status": "success", "traffic_percentage": traffic_percentage}
        return {"status": "error", "message": "Shadow deployment not enabled"}
    
    return app


def run_server():
    """Run the FastAPI server."""
    config = get_config()
    app = create_app()
    
    uvicorn.run(
        app,
        host=config.get("serving.api.host", "0.0.0.0"),
        port=config.get("serving.api.port", 8000),
        workers=config.get("serving.api.workers", 4),
        log_level="info"
    )


if __name__ == "__main__":
    run_server()

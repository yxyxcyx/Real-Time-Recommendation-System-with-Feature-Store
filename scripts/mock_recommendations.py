#!/usr/bin/env python3
"""Mock recommendation service for performance testing."""

import json
import random
import time
from typing import List, Dict, Any


class MockRecommendationService:
    """Mock service for generating recommendations without model inference."""
    
    def __init__(self, num_items: int = 10000):
        self.num_items = num_items
        self.categories = ["electronics", "clothing", "books", "home", "sports", "toys"]
        
    def generate_recommendations(self, user_id: str, num_recommendations: int = 5) -> Dict[str, Any]:
        """Generate mock recommendations."""
        # Simulate some processing time
        time.sleep(random.uniform(0.001, 0.01))  # 1-10ms processing time
        
        # Generate random item recommendations
        recommendations = []
        item_ids = random.sample(range(1, self.num_items), num_recommendations)
        
        for i, item_id in enumerate(item_ids):
            recommendations.append({
                "item_id": f"item_{item_id}",
                "score": round(random.uniform(0.7, 0.99), 3),
                "title": f"Product Title {item_id}",
                "category": random.choice(self.categories),
                "metadata": {
                    "brand": f"Brand_{random.randint(1, 100)}",
                    "price": round(random.uniform(10.0, 500.0), 2)
                }
            })
        
        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "model_version": "mock_v1.0",
            "response_time_ms": round(random.uniform(5, 25), 2),
            "debug_info": {
                "cache_hit": random.choice([True, False]),
                "num_candidates": num_recommendations * 10
            }
        }


def create_mock_api_endpoint():
    """Create a simple mock API endpoint for testing."""
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    service = MockRecommendationService()
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({
            "status": "healthy",
            "timestamp": time.time(),
            "service": "mock_recommendation_api"
        })
    
    @app.route('/recommend', methods=['POST'])
    def recommend():
        data = request.get_json()
        user_id = data.get('user_id', 'unknown')
        num_recommendations = data.get('num_recommendations', 5)
        
        try:
            recommendations = service.generate_recommendations(user_id, num_recommendations)
            return jsonify(recommendations)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/metrics', methods=['GET'])
    def metrics():
        return jsonify({
            "mock_requests_total": random.randint(1000, 5000),
            "mock_latency_avg": random.uniform(10, 50)
        })
    
    return app


if __name__ == "__main__":
    # Run mock API for testing
    app = create_mock_api_endpoint()
    app.run(host='0.0.0.0', port=8001, debug=False)

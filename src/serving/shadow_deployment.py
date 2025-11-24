"""Shadow deployment manager for A/B testing and gradual rollout."""

import hashlib
import random
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger


class ShadowDeploymentManager:
    """Manages shadow deployment and A/B testing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize shadow deployment manager.
        
        Args:
            config: Shadow deployment configuration
        """
        self.config = config
        self.enabled = config.get("enabled", False)
        self.traffic_percentage = config.get("traffic_percentage", 5)
        self.comparison_metrics = config.get("comparison_metrics", ["ctr", "engagement_time"])
        
        # User assignment cache
        self.user_assignments = {}
        self.assignment_ttl = 3600  # 1 hour
        
        # Metrics collection
        self.metrics = {
            "primary": defaultdict(list),
            "shadow": defaultdict(list)
        }
        
        # Comparison results
        self.comparison_results = []
        self.max_comparisons = 10000
        
        logger.info(
            f"Shadow deployment initialized with {self.traffic_percentage}% traffic"
        )
    
    def should_use_shadow(self, user_id: str) -> bool:
        """Determine if user should use shadow model.
        
        Args:
            user_id: User ID
            
        Returns:
            True if shadow model should be used
        """
        if not self.enabled:
            return False
        
        # Check cached assignment
        if user_id in self.user_assignments:
            assignment = self.user_assignments[user_id]
            if time.time() - assignment["timestamp"] < self.assignment_ttl:
                return assignment["use_shadow"]
        
        # Consistent hashing for user assignment
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        use_shadow = (hash_value % 100) < self.traffic_percentage
        
        # Cache assignment
        self.user_assignments[user_id] = {
            "use_shadow": use_shadow,
            "timestamp": time.time()
        }
        
        return use_shadow
    
    def set_traffic_percentage(self, percentage: float):
        """Update traffic percentage for shadow deployment.
        
        Args:
            percentage: New traffic percentage (0-100)
        """
        self.traffic_percentage = min(max(percentage, 0), 100)
        self.user_assignments.clear()  # Clear cache to reassign users
        logger.info(f"Updated shadow traffic to {self.traffic_percentage}%")
    
    def log_comparison(
        self,
        user_id: str,
        recommendations: List[Any],
        response_time: float,
        model_type: str = "shadow"
    ):
        """Log results for comparison.
        
        Args:
            user_id: User ID
            recommendations: List of recommendations
            response_time: Response time in ms
            model_type: "primary" or "shadow"
        """
        comparison_entry = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "model_type": model_type,
            "num_recommendations": len(recommendations),
            "response_time_ms": response_time,
            "top_items": [r.item_id for r in recommendations[:5]]
        }
        
        # Store comparison
        if len(self.comparison_results) >= self.max_comparisons:
            self.comparison_results.pop(0)
        self.comparison_results.append(comparison_entry)
        
        # Update metrics
        self.metrics[model_type]["response_time"].append(response_time)
        self.metrics[model_type]["num_recommendations"].append(len(recommendations))
    
    def log_feedback(
        self,
        user_id: str,
        item_id: str,
        event_type: str,
        model_type: Optional[str] = None
    ):
        """Log user feedback for metrics calculation.
        
        Args:
            user_id: User ID
            item_id: Item ID
            event_type: Type of event (click, view, etc.)
            model_type: Model that served the recommendation
        """
        if model_type is None:
            # Determine model type from user assignment
            model_type = "shadow" if self.should_use_shadow(user_id) else "primary"
        
        # Log event
        event = {
            "user_id": user_id,
            "item_id": item_id,
            "event_type": event_type,
            "timestamp": datetime.now().isoformat()
        }
        
        if event_type == "click":
            self.metrics[model_type]["clicks"].append(event)
        elif event_type == "view":
            self.metrics[model_type]["views"].append(event)
    
    def calculate_metrics(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Calculate comparison metrics.
        
        Args:
            time_window_hours: Time window for metrics calculation
            
        Returns:
            Dictionary of calculated metrics
        """
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        results = {}
        
        for model_type in ["primary", "shadow"]:
            model_metrics = self.metrics[model_type]
            
            # Calculate CTR
            recent_clicks = [
                c for c in model_metrics.get("clicks", [])
                if datetime.fromisoformat(c["timestamp"]) > cutoff_time
            ]
            recent_views = [
                v for v in model_metrics.get("views", [])
                if datetime.fromisoformat(v["timestamp"]) > cutoff_time
            ]
            
            ctr = len(recent_clicks) / max(len(recent_views), 1)
            
            # Calculate average response time
            response_times = model_metrics.get("response_time", [])
            avg_response_time = np.mean(response_times) if response_times else 0
            
            # Calculate other metrics
            results[model_type] = {
                "ctr": ctr,
                "clicks": len(recent_clicks),
                "views": len(recent_views),
                "avg_response_time_ms": avg_response_time,
                "p95_response_time_ms": np.percentile(response_times, 95) if response_times else 0,
                "total_requests": len(response_times)
            }
        
        # Calculate relative improvements
        if results["primary"]["views"] > 0 and results["shadow"]["views"] > 0:
            results["comparison"] = {
                "ctr_improvement": (
                    (results["shadow"]["ctr"] - results["primary"]["ctr"]) / 
                    results["primary"]["ctr"] * 100
                    if results["primary"]["ctr"] > 0 else 0
                ),
                "response_time_improvement": (
                    (results["primary"]["avg_response_time_ms"] - 
                     results["shadow"]["avg_response_time_ms"]) /
                    results["primary"]["avg_response_time_ms"] * 100
                    if results["primary"]["avg_response_time_ms"] > 0 else 0
                )
            }
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get current shadow deployment status.
        
        Returns:
            Status dictionary
        """
        metrics = self.calculate_metrics()
        
        return {
            "enabled": self.enabled,
            "traffic_percentage": self.traffic_percentage,
            "active_users": len(self.user_assignments),
            "total_comparisons": len(self.comparison_results),
            "metrics": metrics,
            "last_updated": datetime.now().isoformat()
        }
    
    def should_promote_shadow(self, min_samples: int = 1000) -> bool:
        """Determine if shadow model should be promoted to primary.
        
        Args:
            min_samples: Minimum number of samples required
            
        Returns:
            True if shadow should be promoted
        """
        metrics = self.calculate_metrics()
        
        # Check if we have enough data
        shadow_samples = metrics["shadow"]["total_requests"]
        if shadow_samples < min_samples:
            logger.info(
                f"Not enough shadow samples ({shadow_samples}/{min_samples})"
            )
            return False
        
        # Check performance criteria
        comparison = metrics.get("comparison", {})
        
        # Criteria for promotion
        ctr_improvement = comparison.get("ctr_improvement", 0)
        response_time_improvement = comparison.get("response_time_improvement", 0)
        
        # Shadow should have better or similar CTR and not worse response time
        promote = (
            ctr_improvement >= -2 and  # Allow 2% CTR degradation
            response_time_improvement >= -10  # Allow 10% slower response
        )
        
        if promote:
            logger.info(
                f"Shadow model ready for promotion: "
                f"CTR improvement: {ctr_improvement:.2f}%, "
                f"Response time improvement: {response_time_improvement:.2f}%"
            )
        else:
            logger.info(
                f"Shadow model not ready for promotion: "
                f"CTR improvement: {ctr_improvement:.2f}%, "
                f"Response time improvement: {response_time_improvement:.2f}%"
            )
        
        return promote
    
    def promote_shadow(self):
        """Promote shadow model to primary."""
        logger.info("Promoting shadow model to primary...")
        
        # In a real system, this would:
        # 1. Save current primary as backup
        # 2. Copy shadow model to primary location
        # 3. Update model registry
        # 4. Clear metrics and reset
        
        self.metrics = {
            "primary": defaultdict(list),
            "shadow": defaultdict(list)
        }
        self.comparison_results = []
        self.user_assignments = {}
        
        logger.info("Shadow model promoted successfully")
    
    def rollback(self):
        """Rollback to previous primary model."""
        logger.warning("Rolling back to previous primary model...")
        
        # In a real system, this would restore the backup
        
        self.enabled = False
        logger.info("Rollback completed, shadow deployment disabled")


class ExperimentTracker:
    """Tracks A/B testing experiments."""
    
    def __init__(self):
        """Initialize experiment tracker."""
        self.experiments = {}
        self.active_experiment = None
    
    def create_experiment(
        self,
        name: str,
        description: str,
        variants: Dict[str, Any],
        metrics: List[str],
        duration_hours: int = 168  # 1 week default
    ) -> str:
        """Create a new experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
            variants: Dictionary of variant configurations
            metrics: List of metrics to track
            duration_hours: Experiment duration
            
        Returns:
            Experiment ID
        """
        experiment_id = hashlib.md5(
            f"{name}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:8]
        
        self.experiments[experiment_id] = {
            "id": experiment_id,
            "name": name,
            "description": description,
            "variants": variants,
            "metrics": metrics,
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(hours=duration_hours),
            "status": "running",
            "results": defaultdict(lambda: defaultdict(list))
        }
        
        self.active_experiment = experiment_id
        logger.info(f"Created experiment {experiment_id}: {name}")
        
        return experiment_id
    
    def log_event(
        self,
        experiment_id: str,
        variant: str,
        metric: str,
        value: float
    ):
        """Log an event for an experiment.
        
        Args:
            experiment_id: Experiment ID
            variant: Variant name
            metric: Metric name
            value: Metric value
        """
        if experiment_id not in self.experiments:
            return
        
        experiment = self.experiments[experiment_id]
        if experiment["status"] != "running":
            return
        
        # Check if experiment has ended
        if datetime.now() > experiment["end_time"]:
            self.end_experiment(experiment_id)
            return
        
        # Log the metric
        experiment["results"][variant][metric].append({
            "value": value,
            "timestamp": datetime.now().isoformat()
        })
    
    def end_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """End an experiment and calculate results.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment results
        """
        if experiment_id not in self.experiments:
            return {}
        
        experiment = self.experiments[experiment_id]
        experiment["status"] = "completed"
        experiment["end_time"] = datetime.now()
        
        # Calculate statistics
        results = {}
        for variant, metrics in experiment["results"].items():
            variant_results = {}
            for metric, values in metrics.items():
                metric_values = [v["value"] for v in values]
                variant_results[metric] = {
                    "mean": np.mean(metric_values) if metric_values else 0,
                    "std": np.std(metric_values) if metric_values else 0,
                    "count": len(metric_values),
                    "min": min(metric_values) if metric_values else 0,
                    "max": max(metric_values) if metric_values else 0
                }
            results[variant] = variant_results
        
        experiment["final_results"] = results
        
        # Determine winner (simplified)
        primary_metric = experiment["metrics"][0] if experiment["metrics"] else None
        if primary_metric and len(results) > 1:
            best_variant = max(
                results.keys(),
                key=lambda v: results[v].get(primary_metric, {}).get("mean", 0)
            )
            experiment["winner"] = best_variant
        
        logger.info(
            f"Experiment {experiment_id} completed. "
            f"Winner: {experiment.get('winner', 'No winner')}"
        )
        
        return experiment
    
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get current experiment status.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment status
        """
        if experiment_id not in self.experiments:
            return {"error": "Experiment not found"}
        
        experiment = self.experiments[experiment_id]
        
        # Calculate current statistics
        current_results = {}
        for variant, metrics in experiment["results"].items():
            variant_results = {}
            for metric, values in metrics.items():
                recent_values = [
                    v["value"] for v in values[-100:]  # Last 100 values
                ]
                variant_results[metric] = {
                    "current_mean": np.mean(recent_values) if recent_values else 0,
                    "total_samples": len(values)
                }
            current_results[variant] = variant_results
        
        return {
            "id": experiment_id,
            "name": experiment["name"],
            "status": experiment["status"],
            "start_time": experiment["start_time"].isoformat(),
            "end_time": experiment["end_time"].isoformat(),
            "current_results": current_results,
            "final_results": experiment.get("final_results"),
            "winner": experiment.get("winner")
        }

"""Metrics collection and monitoring for the recommendation system."""

import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    CollectorRegistry,
    generate_latest,
    push_to_gateway,
)


class MetricsCollector:
    """Collects and exposes metrics for monitoring."""
    
    def __init__(self, namespace: str = "recsys"):
        """Initialize metrics collector.
        
        Args:
            namespace: Prometheus namespace for metrics
        """
        self.namespace = namespace
        self.registry = CollectorRegistry()
        
        # Define metrics
        self._init_counters()
        self._init_gauges()
        self._init_histograms()
        self._init_summaries()
        
        # Internal tracking
        self.start_time = time.time()
        self.metric_buffer = defaultdict(list)
        
    def _init_counters(self):
        """Initialize counter metrics."""
        self.request_counter = Counter(
            f"{self.namespace}_requests_total",
            "Total number of requests",
            ["endpoint", "status"],
            registry=self.registry
        )
        
        self.model_prediction_counter = Counter(
            f"{self.namespace}_model_predictions_total",
            "Total number of model predictions",
            ["model_type", "model_version"],
            registry=self.registry
        )
        
        self.cache_hit_counter = Counter(
            f"{self.namespace}_cache_hits_total",
            "Total number of cache hits",
            ["cache_type"],
            registry=self.registry
        )
        
        self.error_counter = Counter(
            f"{self.namespace}_errors_total",
            "Total number of errors",
            ["error_type", "component"],
            registry=self.registry
        )
        
    def _init_gauges(self):
        """Initialize gauge metrics."""
        self.active_users_gauge = Gauge(
            f"{self.namespace}_active_users",
            "Number of active users",
            registry=self.registry
        )
        
        self.model_accuracy_gauge = Gauge(
            f"{self.namespace}_model_accuracy",
            "Current model accuracy",
            ["model_type"],
            registry=self.registry
        )
        
        self.feature_freshness_gauge = Gauge(
            f"{self.namespace}_feature_freshness_seconds",
            "Feature freshness in seconds",
            ["feature_type"],
            registry=self.registry
        )
        
        self.queue_size_gauge = Gauge(
            f"{self.namespace}_queue_size",
            "Current queue size",
            ["queue_name"],
            registry=self.registry
        )
        
    def _init_histograms(self):
        """Initialize histogram metrics."""
        self.latency_histogram = Histogram(
            f"{self.namespace}_request_latency_seconds",
            "Request latency in seconds",
            ["endpoint", "stage"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry
        )
        
        self.recommendation_count_histogram = Histogram(
            f"{self.namespace}_recommendations_per_request",
            "Number of recommendations per request",
            buckets=(1, 5, 10, 20, 50, 100),
            registry=self.registry
        )
        
        self.feature_computation_histogram = Histogram(
            f"{self.namespace}_feature_computation_seconds",
            "Feature computation time",
            ["feature_type"],
            registry=self.registry
        )
        
    def _init_summaries(self):
        """Initialize summary metrics."""
        self.ctr_summary = Summary(
            f"{self.namespace}_ctr",
            "Click-through rate",
            ["model_version"],
            registry=self.registry
        )
        
        self.engagement_summary = Summary(
            f"{self.namespace}_engagement_score",
            "User engagement score",
            registry=self.registry
        )
        
    def record_request(
        self,
        endpoint: str,
        status: str,
        latency: float
    ):
        """Record API request metrics.
        
        Args:
            endpoint: API endpoint
            status: Request status (success/error)
            latency: Request latency in seconds
        """
        self.request_counter.labels(endpoint=endpoint, status=status).inc()
        self.latency_histogram.labels(endpoint=endpoint, stage="total").observe(latency)
        
    def record_model_prediction(
        self,
        model_type: str,
        model_version: str,
        prediction_time: float
    ):
        """Record model prediction metrics.
        
        Args:
            model_type: Type of model (two_tower, xgboost, etc.)
            model_version: Model version
            prediction_time: Prediction time in seconds
        """
        self.model_prediction_counter.labels(
            model_type=model_type,
            model_version=model_version
        ).inc()
        
        self.latency_histogram.labels(
            endpoint="predict",
            stage=model_type
        ).observe(prediction_time)
        
    def record_cache_hit(self, cache_type: str):
        """Record cache hit.
        
        Args:
            cache_type: Type of cache (embedding, feature, etc.)
        """
        self.cache_hit_counter.labels(cache_type=cache_type).inc()
        
    def record_error(self, error_type: str, component: str):
        """Record error occurrence.
        
        Args:
            error_type: Type of error
            component: Component where error occurred
        """
        self.error_counter.labels(error_type=error_type, component=component).inc()
        
    def update_active_users(self, count: int):
        """Update active users gauge.
        
        Args:
            count: Number of active users
        """
        self.active_users_gauge.set(count)
        
    def update_model_accuracy(self, model_type: str, accuracy: float):
        """Update model accuracy gauge.
        
        Args:
            model_type: Type of model
            accuracy: Model accuracy
        """
        self.model_accuracy_gauge.labels(model_type=model_type).set(accuracy)
        
    def record_ctr(self, clicks: int, impressions: int, model_version: str = "default"):
        """Record CTR metrics.
        
        Args:
            clicks: Number of clicks
            impressions: Number of impressions
            model_version: Model version
        """
        if impressions > 0:
            ctr = clicks / impressions
            self.ctr_summary.labels(model_version=model_version).observe(ctr)
            
    def record_engagement(self, score: float):
        """Record engagement score.
        
        Args:
            score: Engagement score
        """
        self.engagement_summary.observe(score)
        
    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format.
        
        Returns:
            Metrics in Prometheus text format
        """
        return generate_latest(self.registry)
        
    def push_metrics(self, gateway: str, job: str):
        """Push metrics to Prometheus Pushgateway.
        
        Args:
            gateway: Pushgateway URL
            job: Job name
        """
        try:
            push_to_gateway(gateway, job=job, registry=self.registry)
        except Exception as e:
            logger.error(f"Failed to push metrics: {e}")


class PerformanceMonitor:
    """Monitors system performance and model quality."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize performance monitor.
        
        Args:
            metrics_collector: MetricsCollector instance
        """
        self.metrics = metrics_collector
        self.performance_history = defaultdict(list)
        self.alert_thresholds = {
            "latency_p95": 0.1,  # 100ms
            "error_rate": 0.01,  # 1%
            "ctr_drop": 0.2,  # 20% relative drop
        }
        
    def check_latency(self, latencies: List[float]) -> Dict[str, Any]:
        """Check latency metrics.
        
        Args:
            latencies: List of latency values
            
        Returns:
            Latency statistics
        """
        if not latencies:
            return {}
            
        stats = {
            "mean": np.mean(latencies),
            "median": np.median(latencies),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
            "max": np.max(latencies),
        }
        
        # Check if P95 exceeds threshold
        if stats["p95"] > self.alert_thresholds["latency_p95"]:
            logger.warning(
                f"High latency detected: P95={stats['p95']:.3f}s "
                f"(threshold: {self.alert_thresholds['latency_p95']}s)"
            )
            
        return stats
        
    def check_error_rate(self, errors: int, total: int) -> float:
        """Check error rate.
        
        Args:
            errors: Number of errors
            total: Total requests
            
        Returns:
            Error rate
        """
        if total == 0:
            return 0
            
        error_rate = errors / total
        
        if error_rate > self.alert_thresholds["error_rate"]:
            logger.warning(
                f"High error rate: {error_rate:.2%} "
                f"(threshold: {self.alert_thresholds['error_rate']:.2%})"
            )
            
        return error_rate
        
    def check_model_drift(
        self,
        current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Check for model drift.
        
        Args:
            current_metrics: Current model metrics
            baseline_metrics: Baseline metrics
            
        Returns:
            Drift statistics
        """
        drift_stats = {}
        
        for metric, current_value in current_metrics.items():
            if metric in baseline_metrics:
                baseline_value = baseline_metrics[metric]
                if baseline_value > 0:
                    relative_change = (current_value - baseline_value) / baseline_value
                    drift_stats[f"{metric}_drift"] = relative_change
                    
                    # Check for significant CTR drop
                    if metric == "ctr" and relative_change < -self.alert_thresholds["ctr_drop"]:
                        logger.warning(
                            f"Significant CTR drop detected: {relative_change:.2%} "
                            f"(threshold: -{self.alert_thresholds['ctr_drop']:.2%})"
                        )
                        
        return drift_stats
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate performance report.
        
        Returns:
            Performance report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "uptime_hours": (time.time() - self.metrics.start_time) / 3600,
            "metrics_summary": {},
            "alerts": [],
        }
        
        # Add recent performance stats
        for metric_name, values in self.performance_history.items():
            if values:
                recent_values = values[-1000:]  # Last 1000 measurements
                report["metrics_summary"][metric_name] = {
                    "mean": np.mean(recent_values),
                    "std": np.std(recent_values),
                    "min": np.min(recent_values),
                    "max": np.max(recent_values),
                }
                
        return report
        
    def log_performance(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None
    ):
        """Log performance metric.
        
        Args:
            metric_name: Name of metric
            value: Metric value
            timestamp: Optional timestamp
        """
        timestamp = timestamp or datetime.now()
        
        self.performance_history[metric_name].append({
            "value": value,
            "timestamp": timestamp.isoformat()
        })
        
        # Keep only recent history (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.performance_history[metric_name] = [
            entry for entry in self.performance_history[metric_name]
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
        ]


class ModelMetricsTracker:
    """Tracks model-specific metrics."""
    
    def __init__(self):
        """Initialize model metrics tracker."""
        self.model_metrics = defaultdict(lambda: defaultdict(list))
        
    def track_prediction(
        self,
        model_name: str,
        model_version: str,
        prediction_time: float,
        input_size: int,
        output_size: int
    ):
        """Track model prediction metrics.
        
        Args:
            model_name: Name of model
            model_version: Model version
            prediction_time: Time taken for prediction
            input_size: Size of input
            output_size: Size of output
        """
        key = f"{model_name}_{model_version}"
        
        self.model_metrics[key]["prediction_times"].append(prediction_time)
        self.model_metrics[key]["input_sizes"].append(input_size)
        self.model_metrics[key]["output_sizes"].append(output_size)
        self.model_metrics[key]["throughput"].append(input_size / prediction_time)
        
    def track_accuracy(
        self,
        model_name: str,
        model_version: str,
        accuracy: float,
        metric_type: str = "accuracy"
    ):
        """Track model accuracy metrics.
        
        Args:
            model_name: Name of model
            model_version: Model version
            accuracy: Accuracy value
            metric_type: Type of metric
        """
        key = f"{model_name}_{model_version}"
        self.model_metrics[key][metric_type].append({
            "value": accuracy,
            "timestamp": datetime.now().isoformat()
        })
        
    def get_model_stats(self, model_name: str, model_version: str) -> Dict[str, Any]:
        """Get statistics for a specific model.
        
        Args:
            model_name: Name of model
            model_version: Model version
            
        Returns:
            Model statistics
        """
        key = f"{model_name}_{model_version}"
        metrics = self.model_metrics.get(key, {})
        
        stats = {}
        for metric_name, values in metrics.items():
            if values and isinstance(values[0], (int, float)):
                stats[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values)
                }
                
        return stats

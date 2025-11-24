"""Streaming components for real-time feature processing."""

from .kafka_consumer import KafkaFeatureConsumer
from .kafka_producer import KafkaFeatureProducer
from .event_processor import EventProcessor
from .window_aggregator import WindowAggregator

__all__ = [
    "KafkaFeatureConsumer",
    "KafkaFeatureProducer",
    "EventProcessor",
    "WindowAggregator",
]

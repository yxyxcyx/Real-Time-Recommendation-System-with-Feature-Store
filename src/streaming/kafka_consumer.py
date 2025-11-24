"""Kafka consumer for real-time feature updates."""

import asyncio
import json
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from aiokafka import AIOKafkaConsumer
from kafka import KafkaConsumer
from loguru import logger


class KafkaFeatureConsumer:
    """Consumes events from Kafka for real-time feature processing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Kafka consumer.
        
        Args:
            config: Kafka configuration
        """
        self.config = config
        self.bootstrap_servers = config.get("bootstrap_servers", "localhost:9092")
        self.topics = config.get("topics", {})
        self.consumer_group = config.get("consumer_group", "recsys-consumer")
        self.auto_offset_reset = config.get("auto_offset_reset", "latest")
        
        self.consumer = None
        self.async_consumer = None
        self.handlers = {}
        self.running = False
        
        # Metrics
        self.messages_processed = 0
        self.messages_failed = 0
        self.last_message_time = None
        
    def register_handler(self, event_type: str, handler: Callable):
        """Register an event handler.
        
        Args:
            event_type: Type of event to handle
            handler: Handler function
        """
        self.handlers[event_type] = handler
        logger.info(f"Registered handler for event type: {event_type}")
    
    def start(self):
        """Start consuming messages."""
        logger.info("Starting Kafka consumer...")
        
        self.consumer = KafkaConsumer(
            *self.topics.values(),
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.consumer_group,
            auto_offset_reset=self.auto_offset_reset,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            enable_auto_commit=True,
            auto_commit_interval_ms=5000
        )
        
        self.running = True
        self._consume_loop()
    
    def _consume_loop(self):
        """Main consumption loop."""
        while self.running:
            try:
                # Poll for messages
                messages = self.consumer.poll(timeout_ms=1000)
                
                for topic_partition, records in messages.items():
                    for record in records:
                        self._process_message(record)
                        
            except Exception as e:
                logger.error(f"Error in consumer loop: {e}")
                self.messages_failed += 1
    
    def _process_message(self, record):
        """Process a single message.
        
        Args:
            record: Kafka record
        """
        try:
            # Extract event data
            event = record.value
            event_type = event.get("event_type", "unknown")
            timestamp = event.get("timestamp", datetime.now().isoformat())
            
            # Log message receipt
            logger.debug(f"Received {event_type} event from topic {record.topic}")
            
            # Call appropriate handler
            if event_type in self.handlers:
                self.handlers[event_type](event)
            else:
                logger.warning(f"No handler for event type: {event_type}")
            
            # Update metrics
            self.messages_processed += 1
            self.last_message_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.messages_failed += 1
    
    async def start_async(self):
        """Start async consumer."""
        logger.info("Starting async Kafka consumer...")
        
        self.async_consumer = AIOKafkaConsumer(
            *self.topics.values(),
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.consumer_group,
            auto_offset_reset=self.auto_offset_reset,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        await self.async_consumer.start()
        self.running = True
        
        try:
            await self._async_consume_loop()
        finally:
            await self.async_consumer.stop()
    
    async def _async_consume_loop(self):
        """Async consumption loop."""
        async for msg in self.async_consumer:
            try:
                await self._async_process_message(msg)
            except Exception as e:
                logger.error(f"Error processing async message: {e}")
                self.messages_failed += 1
    
    async def _async_process_message(self, msg):
        """Process message asynchronously.
        
        Args:
            msg: Kafka message
        """
        event = msg.value
        event_type = event.get("event_type", "unknown")
        
        if event_type in self.handlers:
            handler = self.handlers[event_type]
            
            # Run handler (async or sync)
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        
        self.messages_processed += 1
        self.last_message_time = datetime.now()
    
    def stop(self):
        """Stop the consumer."""
        logger.info("Stopping Kafka consumer...")
        self.running = False
        
        if self.consumer:
            self.consumer.close()
        
        logger.info(f"Consumer stopped. Processed {self.messages_processed} messages")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get consumer metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {
            "messages_processed": self.messages_processed,
            "messages_failed": self.messages_failed,
            "success_rate": self.messages_processed / max(self.messages_processed + self.messages_failed, 1),
            "last_message_time": self.last_message_time.isoformat() if self.last_message_time else None,
            "running": self.running
        }


class EventHandlers:
    """Collection of event handlers for different event types."""
    
    def __init__(self, feature_store, retrieval_engine):
        """Initialize event handlers.
        
        Args:
            feature_store: Feature store instance
            retrieval_engine: Retrieval engine instance
        """
        self.feature_store = feature_store
        self.retrieval_engine = retrieval_engine
        
    def handle_user_click(self, event: Dict[str, Any]):
        """Handle user click event.
        
        Args:
            event: Click event data
        """
        user_id = event.get("user_id")
        item_id = event.get("item_id")
        timestamp = event.get("timestamp")
        
        logger.debug(f"User {user_id} clicked item {item_id}")
        
        # Update real-time features
        self._update_user_streaming_features(user_id, "click", item_id, timestamp)
        self._update_item_streaming_features(item_id, "click", user_id, timestamp)
    
    def handle_user_view(self, event: Dict[str, Any]):
        """Handle user view event.
        
        Args:
            event: View event data
        """
        user_id = event.get("user_id")
        item_id = event.get("item_id")
        duration = event.get("duration", 0)
        timestamp = event.get("timestamp")
        
        logger.debug(f"User {user_id} viewed item {item_id} for {duration}s")
        
        # Update real-time features
        self._update_user_streaming_features(user_id, "view", item_id, timestamp, duration)
        self._update_item_streaming_features(item_id, "view", user_id, timestamp)
    
    def handle_item_update(self, event: Dict[str, Any]):
        """Handle item update event.
        
        Args:
            event: Item update event
        """
        item_id = event.get("item_id")
        updates = event.get("updates", {})
        
        logger.debug(f"Item {item_id} updated: {list(updates.keys())}")
        
        # Update item embeddings if needed
        if "embedding" in updates:
            self._update_item_embedding(item_id, updates["embedding"])
    
    def handle_user_update(self, event: Dict[str, Any]):
        """Handle user profile update.
        
        Args:
            event: User update event
        """
        user_id = event.get("user_id")
        updates = event.get("updates", {})
        
        logger.debug(f"User {user_id} updated: {list(updates.keys())}")
        
        # Update user features in feature store
        self._update_user_profile_features(user_id, updates)
    
    def _update_user_streaming_features(
        self,
        user_id: str,
        event_type: str,
        item_id: str,
        timestamp: str,
        duration: Optional[float] = None
    ):
        """Update user streaming features.
        
        Args:
            user_id: User ID
            event_type: Type of event (click, view)
            item_id: Item ID
            timestamp: Event timestamp
            duration: Optional duration for view events
        """
        # Create feature update
        import pandas as pd
        
        feature_data = pd.DataFrame([{
            "user_id": user_id,
            "event_timestamp": pd.to_datetime(timestamp),
            "clicks_5min": 1 if event_type == "click" else 0,
            "views_5min": 1,
            "avg_dwell_time_5min": duration or 0,
            "session_depth": 1,  # Would be calculated from session
            "current_context": item_id,
        }])
        
        # Push to feature store
        try:
            self.feature_store.push_streaming_features(
                feature_data,
                "realtime_user_features"
            )
        except Exception as e:
            logger.error(f"Failed to update user streaming features: {e}")
    
    def _update_item_streaming_features(
        self,
        item_id: str,
        event_type: str,
        user_id: str,
        timestamp: str
    ):
        """Update item streaming features.
        
        Args:
            item_id: Item ID
            event_type: Type of event
            user_id: User ID
            timestamp: Event timestamp
        """
        import pandas as pd
        
        feature_data = pd.DataFrame([{
            "item_id": item_id,
            "event_timestamp": pd.to_datetime(timestamp),
            "clicks_5min": 1 if event_type == "click" else 0,
            "views_5min": 1,
            "velocity_score": 1.0,  # Would be calculated
            "freshness_score": 1.0,  # Would be calculated
        }])
        
        # Push to feature store
        try:
            self.feature_store.push_streaming_features(
                feature_data,
                "realtime_item_features"
            )
        except Exception as e:
            logger.error(f"Failed to update item streaming features: {e}")
    
    def _update_item_embedding(self, item_id: str, embedding: List[float]):
        """Update item embedding in retrieval index.
        
        Args:
            item_id: Item ID
            embedding: New embedding vector
        """
        import numpy as np
        
        try:
            # Add to retrieval index
            self.retrieval_engine.update_index(
                np.array([embedding]),
                [item_id]
            )
            logger.debug(f"Updated embedding for item {item_id}")
        except Exception as e:
            logger.error(f"Failed to update item embedding: {e}")
    
    def _update_user_profile_features(self, user_id: str, updates: Dict[str, Any]):
        """Update user profile features.
        
        Args:
            user_id: User ID
            updates: Feature updates
        """
        # Would update user profile in feature store
        logger.debug(f"Updated user {user_id} profile features")

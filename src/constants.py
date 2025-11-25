"""Centralized constants and configuration values.

This module consolidates magic numbers and hardcoded strings that were
previously scattered across the codebase. Use environment variables
for values that may change between environments.

See REFACTORING_BLUEPRINT.md for details on the consolidation effort.
"""

import os


# =============================================================================
# Infrastructure Connection Defaults
# =============================================================================

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "recsys-consumer")
KAFKA_AUTO_OFFSET_RESET = os.getenv("KAFKA_AUTO_OFFSET_RESET", "latest")

# Redis Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_TTL_SECONDS = int(os.getenv("REDIS_TTL_SECONDS", "3600"))

# Milvus Configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "item_embeddings")


# =============================================================================
# Model & Feature Constants
# =============================================================================

# Default embedding dimensions
DEFAULT_EMBEDDING_DIM = 128
DEFAULT_FEATURE_PADDING_DIM = 47  # Used to pad features to reach 50 dims

# Model checkpoint paths
MODEL_CHECKPOINT_DIR = os.getenv("MODEL_CHECKPOINT_DIR", "models/checkpoints")
FEATURE_STORE_DATA_DIR = os.getenv("FEATURE_STORE_DATA_DIR", "feature_store/data/")

# Feature store constants
EVENT_TIMESTAMP_COLUMN = "event_timestamp"


# =============================================================================
# API & Serving Constants
# =============================================================================

# Cache settings
DEFAULT_CACHE_TTL = 300  # 5 minutes
USER_ASSIGNMENT_TTL = 3600  # 1 hour (for A/B testing)

# Rate limits and batch sizes
MAX_COMPARISONS = 10000
MIN_SAMPLES_FOR_PROMOTION = 1000
DEFAULT_BATCH_SIZE = 1024


# =============================================================================
# Synthetic Data Defaults
# =============================================================================

DEFAULT_NUM_USERS = 1000
DEFAULT_NUM_ITEMS = 5000
DEFAULT_NUM_INTERACTIONS = 50000
DEFAULT_NUM_NUMERICAL_FEATURES = 10

"""Simple mock classes for Feast types to avoid dependency issues."""

from datetime import timedelta
from typing import List, Any


class Entity:
    """Mock Entity class."""
    def __init__(self, name: str, description: str = "", value_type: str = "STRING"):
        self.name = name
        self.description = description
        self.value_type = value_type


class Field:
    """Mock Field class."""
    def __init__(self, name: str, dtype):
        self.name = name
        self.dtype = dtype


class FeatureView:
    """Mock FeatureView class."""
    def __init__(self, name: str, entities: List[Any], ttl: timedelta, schema: List[Field], source: Any, tags: dict = None):
        self.name = name
        self.entities = entities
        self.ttl = ttl
        self.schema = schema
        self.source = source
        self.tags = tags or {}


class FileSource:
    """Mock FileSource class."""
    def __init__(self, name: str, path: str, timestamp_field: str, created_timestamp_column: str = None):
        self.name = name
        self.path = path
        self.timestamp_field = timestamp_field
        self.created_timestamp_column = created_timestamp_column


class PushSource:
    """Mock PushSource class."""
    def __init__(self, name: str, batch_source: FileSource):
        self.name = name
        self.batch_source = batch_source


class RequestSource:
    """Mock RequestSource class."""
    def __init__(self, name: str, schema: List[Field]):
        self.name = name
        self.schema = schema


class ValueType:
    """Mock ValueType class."""
    STRING = "string"
    INT64 = "int64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"


# Mock types
class String:
    pass

class Int64:
    pass

class Float32:
    pass

class Float64:
    pass

class UnixTimestamp:
    pass


class ParquetFormat:
    """Mock ParquetFormat class."""
    pass

"""Retrieval layer for candidate generation using ANN search."""

import hashlib
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from loguru import logger
from annoy import AnnoyIndex


class IndexBase(ABC):
    """Abstract base class for ANN indices."""
    
    @abstractmethod
    def build(self, embeddings: np.ndarray, ids: List[str]):
        """Build the index from embeddings."""
        pass
    
    @abstractmethod
    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors."""
        pass
    
    @abstractmethod
    def add(self, embeddings: np.ndarray, ids: List[str]):
        """Add new embeddings to the index."""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save index to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load index from disk."""
        pass


class FaissIndex(IndexBase):
    """Faiss-based ANN index for fast retrieval."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Faiss index.
        
        Args:
            config: Index configuration
        """
        self.config = config or {}
        self.dimension = self.config.get("dimension", 128)
        self.index_factory = self.config.get("index_factory", "IVF1024,Flat")
        self.metric = self.config.get("metric", "cosine")
        self.nprobe = self.config.get("nprobe", 20)
        self.use_gpu = self.config.get("use_gpu", False)
        
        self.index = None
        self.id_map = {}  # Map from index position to item ID
        self.reverse_id_map = {}  # Map from item ID to index position
        self.current_size = 0
        
    def build(self, embeddings: np.ndarray, ids: List[str]):
        """Build Faiss index from embeddings.
        
        Args:
            embeddings: Item embeddings [n_items, dimension]
            ids: List of item IDs
        """
        logger.info(f"Building Faiss index for {len(embeddings)} items...")
        start_time = time.time()
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype(np.float32)
        
        # Normalize if using cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(embeddings)
        
        # Create index based on factory string
        if self.index_factory == "Flat":
            if self.metric == "cosine":
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine
            else:
                self.index = faiss.IndexFlatL2(self.dimension)  # L2 distance
        else:
            # Create composite index (e.g., IVF)
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.index_factory(
                self.dimension,
                self.index_factory,
                faiss.METRIC_INNER_PRODUCT if self.metric == "cosine" else faiss.METRIC_L2
            )
        
        # Move to GPU if requested and available
        if self.use_gpu and faiss.get_num_gpus() > 0:
            logger.info("Moving index to GPU...")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # Train index if needed (for IVF indices)
        if not self.index.is_trained:
            logger.info("Training index...")
            self.index.train(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Build ID mappings
        for i, item_id in enumerate(ids):
            self.id_map[i] = item_id
            self.reverse_id_map[item_id] = i
        
        self.current_size = len(embeddings)
        
        # Set search parameters
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.nprobe
        
        build_time = time.time() - start_time
        logger.info(f"Index built in {build_time:.2f} seconds")
        
        # Verify index
        self._verify_index(embeddings)
    
    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
        filter_ids: Optional[List[str]] = None
    ) -> Tuple[List[List[str]], List[List[float]]]:
        """Search for nearest neighbors.
        
        Args:
            query_embeddings: Query embeddings [batch_size, dimension]
            k: Number of neighbors to return
            filter_ids: Optional list of IDs to filter results
            
        Returns:
            Tuple of (item_ids, distances) for each query
        """
        if self.index is None:
            raise ValueError("Index not built yet")
        
        # Ensure embeddings are float32 and 2D
        query_embeddings = query_embeddings.astype(np.float32)
        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        # Normalize if using cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(query_embeddings)
        
        # Search
        k_search = min(k * 2, self.current_size) if filter_ids else k
        distances, indices = self.index.search(query_embeddings, k_search)
        
        # Convert indices to IDs
        batch_ids = []
        batch_distances = []
        
        for i in range(len(query_embeddings)):
            item_ids = []
            item_distances = []
            
            for j in range(k_search):
                idx = indices[i, j]
                if idx >= 0 and idx in self.id_map:  # Valid index
                    item_id = self.id_map[idx]
                    
                    # Apply filtering if specified
                    if filter_ids is None or item_id in filter_ids:
                        item_ids.append(item_id)
                        item_distances.append(float(distances[i, j]))
                        
                        if len(item_ids) >= k:
                            break
            
            batch_ids.append(item_ids)
            batch_distances.append(item_distances)
        
        return batch_ids, batch_distances
    
    def add(self, embeddings: np.ndarray, ids: List[str]):
        """Add new embeddings to existing index.
        
        Args:
            embeddings: New embeddings to add
            ids: IDs for new embeddings
        """
        if self.index is None:
            raise ValueError("Index not built yet")
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype(np.float32)
        
        # Normalize if using cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Update ID mappings
        for i, item_id in enumerate(ids):
            new_idx = self.current_size + i
            self.id_map[new_idx] = item_id
            self.reverse_id_map[item_id] = new_idx
        
        self.current_size += len(embeddings)
        logger.info(f"Added {len(embeddings)} items to index. Total size: {self.current_size}")
    
    def update(self, embeddings: np.ndarray, ids: List[str]):
        """Update existing embeddings in the index.
        
        Args:
            embeddings: Updated embeddings
            ids: IDs of items to update
        """
        # Faiss doesn't support direct updates, so we need to rebuild
        # In production, you might use a more sophisticated approach
        logger.warning("Faiss doesn't support direct updates. Consider periodic rebuilds.")
    
    def remove(self, ids: List[str]):
        """Remove items from the index.
        
        Args:
            ids: IDs of items to remove
        """
        # Faiss doesn't support removal in most index types
        logger.warning("Faiss doesn't support removal. Consider periodic rebuilds.")
    
    def save(self, path: str):
        """Save index to disk.
        
        Args:
            path: Path to save index
        """
        if self.index is None:
            raise ValueError("No index to save")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save Faiss index
        faiss.write_index(self.index, str(path.with_suffix('.faiss')))
        
        # Save ID mappings
        import pickle
        with open(path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump({
                'id_map': self.id_map,
                'reverse_id_map': self.reverse_id_map,
                'current_size': self.current_size,
                'config': self.config
            }, f)
        
        logger.info(f"Saved index to {path}")
    
    def load(self, path: str):
        """Load index from disk.
        
        Args:
            path: Path to index files
        """
        path = Path(path)
        
        # Load Faiss index
        self.index = faiss.read_index(str(path.with_suffix('.faiss')))
        
        # Load ID mappings
        import pickle
        with open(path.with_suffix('.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.id_map = data['id_map']
            self.reverse_id_map = data['reverse_id_map']
            self.current_size = data['current_size']
            self.config = data['config']
        
        # Set search parameters
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.nprobe
        
        logger.info(f"Loaded index from {path} with {self.current_size} items")
    
    def _verify_index(self, embeddings: np.ndarray, n_verify: int = 5):
        """Verify index quality with sample queries.
        
        Args:
            embeddings: Original embeddings
            n_verify: Number of samples to verify
        """
        if len(embeddings) < n_verify:
            return
        
        # Sample random embeddings
        sample_indices = np.random.choice(len(embeddings), n_verify, replace=False)
        sample_embeddings = embeddings[sample_indices]
        
        # Search for nearest neighbors
        ids, distances = self.search(sample_embeddings, k=1)
        
        # Check if each embedding finds itself as nearest neighbor
        correct = 0
        for i, idx in enumerate(sample_indices):
            expected_id = self.id_map[idx]
            if ids[i] and ids[i][0] == expected_id:
                correct += 1
        
        accuracy = correct / n_verify
        logger.info(f"Index verification: {accuracy:.2%} accuracy on self-retrieval")
        
        if accuracy < 0.9:
            logger.warning("Index quality might be low. Consider adjusting parameters.")


class AnnoyIndex(IndexBase):
    """Annoy-based ANN index (alternative to Faiss)."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Annoy index.
        
        Args:
            config: Index configuration
        """
        self.config = config or {}
        self.dimension = self.config.get("dimension", 128)
        self.n_trees = self.config.get("n_trees", 10)
        self.metric = self.config.get("metric", "angular")  # angular, euclidean
        
        self.index = None
        self.id_map = {}
        self.reverse_id_map = {}
        self.current_size = 0
    
    def build(self, embeddings: np.ndarray, ids: List[str]):
        """Build Annoy index from embeddings.
        
        Args:
            embeddings: Item embeddings
            ids: List of item IDs
        """
        logger.info(f"Building Annoy index for {len(embeddings)} items...")
        start_time = time.time()
        
        # Create index
        self.index = AnnoyIndex(self.dimension, self.metric)
        
        # Add embeddings
        for i, (embedding, item_id) in enumerate(zip(embeddings, ids)):
            self.index.add_item(i, embedding)
            self.id_map[i] = item_id
            self.reverse_id_map[item_id] = i
        
        # Build index
        self.index.build(self.n_trees)
        self.current_size = len(embeddings)
        
        build_time = time.time() - start_time
        logger.info(f"Annoy index built in {build_time:.2f} seconds")
    
    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 10
    ) -> Tuple[List[List[str]], List[List[float]]]:
        """Search for nearest neighbors.
        
        Args:
            query_embeddings: Query embeddings
            k: Number of neighbors
            
        Returns:
            Tuple of (item_ids, distances) for each query
        """
        if self.index is None:
            raise ValueError("Index not built yet")
        
        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        batch_ids = []
        batch_distances = []
        
        for query in query_embeddings:
            indices, distances = self.index.get_nns_by_vector(
                query, k, include_distances=True
            )
            
            item_ids = [self.id_map[idx] for idx in indices]
            batch_ids.append(item_ids)
            batch_distances.append(distances)
        
        return batch_ids, batch_distances
    
    def add(self, embeddings: np.ndarray, ids: List[str]):
        """Add new embeddings (requires rebuild)."""
        logger.warning("Annoy doesn't support incremental updates. Rebuild required.")
    
    def save(self, path: str):
        """Save index to disk."""
        if self.index is None:
            raise ValueError("No index to save")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save Annoy index
        self.index.save(str(path.with_suffix('.ann')))
        
        # Save mappings
        import pickle
        with open(path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump({
                'id_map': self.id_map,
                'reverse_id_map': self.reverse_id_map,
                'current_size': self.current_size,
                'config': self.config
            }, f)
        
        logger.info(f"Saved Annoy index to {path}")
    
    def load(self, path: str):
        """Load index from disk."""
        path = Path(path)
        
        # Load mappings first to get dimension
        import pickle
        with open(path.with_suffix('.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.id_map = data['id_map']
            self.reverse_id_map = data['reverse_id_map']
            self.current_size = data['current_size']
            self.config = data['config']
        
        # Create and load index
        self.dimension = self.config.get("dimension", 128)
        self.index = AnnoyIndex(self.dimension, self.metric)
        self.index.load(str(path.with_suffix('.ann')))
        
        logger.info(f"Loaded Annoy index from {path}")


class MilvusIndex(IndexBase):
    """Milvus-based vector database for ANN search."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Milvus client.
        
        Args:
            config: Milvus configuration
        """
        self.config = config or {}
        self.collection_name = self.config.get("collection_name", "item_embeddings")
        self.dimension = self.config.get("dimension", 128)
        self.host = self.config.get("host", "localhost")
        self.port = self.config.get("port", 19530)
        
        # Note: Actual Milvus implementation would require pymilvus
        logger.warning("Milvus index is a placeholder. Install pymilvus for full functionality.")
    
    def build(self, embeddings: np.ndarray, ids: List[str]):
        """Build Milvus collection and insert embeddings."""
        logger.info("Building Milvus index (placeholder)...")
        # Actual implementation would create collection and insert data
        pass
    
    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 10
    ) -> Tuple[List[List[str]], List[List[float]]]:
        """Search in Milvus."""
        # Placeholder implementation
        return [[]], [[]]
    
    def add(self, embeddings: np.ndarray, ids: List[str]):
        """Add to Milvus collection."""
        pass
    
    def save(self, path: str):
        """Milvus persists automatically."""
        pass
    
    def load(self, path: str):
        """Load Milvus collection."""
        pass


class RetrievalEngine:
    """High-level retrieval engine managing indices and caching."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize retrieval engine.
        
        Args:
            config: Retrieval configuration
        """
        self.config = config
        self.index_type = config.get("index_type", "faiss")
        self.top_k = config.get("top_k", 100)
        self.update_interval = config.get("update_interval_seconds", 300)
        
        # Initialize index
        self.index = self._create_index()
        
        # Cache for frequent queries
        self.cache = {}
        self.cache_ttl = config.get("cache_ttl", 300)
        self.last_cache_clear = time.time()
        
        # Metrics
        self.total_queries = 0
        self.cache_hits = 0
        self.total_latency = 0
    
    def _create_index(self) -> IndexBase:
        """Create index based on configuration."""
        index_config = self.config.get(self.index_type, {})
        index_config["dimension"] = self.config.get("embedding_dim", 128)
        
        if self.index_type == "faiss":
            return FaissIndex(index_config)
        elif self.index_type == "annoy":
            return AnnoyIndex(index_config)
        elif self.index_type == "milvus":
            return MilvusIndex(index_config)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def build_index(
        self,
        embeddings: np.ndarray,
        ids: List[str]
    ):
        """Build or rebuild the index.
        
        Args:
            embeddings: Item embeddings
            ids: Item IDs
        """
        self.index.build(embeddings, ids)
        self.cache.clear()
        logger.info(f"Built index with {len(embeddings)} items")
    
    def retrieve(
        self,
        query_embeddings: np.ndarray,
        k: Optional[int] = None,
        filter_ids: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> Tuple[List[List[str]], List[List[float]], Dict[str, Any]]:
        """Retrieve top-k candidates for queries.
        
        Args:
            query_embeddings: Query embeddings
            k: Number of candidates to retrieve
            filter_ids: Optional IDs to filter
            use_cache: Whether to use caching
            
        Returns:
            Tuple of (item_ids, scores, metrics)
        """
        start_time = time.time()
        k = k or self.top_k
        
        # Check cache
        cache_key = None
        if use_cache and filter_ids is None:
            cache_key = hashlib.md5(query_embeddings.tobytes()).hexdigest()
            if cache_key in self.cache:
                cache_entry = self.cache[cache_key]
                if time.time() - cache_entry["timestamp"] < self.cache_ttl:
                    self.cache_hits += 1
                    latency = time.time() - start_time
                    self.total_queries += 1
                    self.total_latency += latency
                    
                    return (
                        cache_entry["ids"],
                        cache_entry["scores"],
                        {"latency_ms": latency * 1000, "cache_hit": True}
                    )
        
        # Perform search
        item_ids, scores = self.index.search(query_embeddings, k, filter_ids)
        
        # Update cache
        if use_cache and cache_key:
            self.cache[cache_key] = {
                "ids": item_ids,
                "scores": scores,
                "timestamp": time.time()
            }
            
            # Clear old cache entries periodically
            if time.time() - self.last_cache_clear > self.cache_ttl:
                self._clear_expired_cache()
        
        # Calculate metrics
        latency = time.time() - start_time
        self.total_queries += 1
        self.total_latency += latency
        
        metrics = {
            "latency_ms": latency * 1000,
            "cache_hit": False,
            "num_results": sum(len(ids) for ids in item_ids),
            "avg_score": np.mean([s for scores_list in scores for s in scores_list])
        }
        
        return item_ids, scores, metrics
    
    def update_index(
        self,
        new_embeddings: np.ndarray,
        new_ids: List[str]
    ):
        """Update index with new items.
        
        Args:
            new_embeddings: New item embeddings
            new_ids: New item IDs
        """
        self.index.add(new_embeddings, new_ids)
        self.cache.clear()
    
    def _clear_expired_cache(self):
        """Clear expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time - entry["timestamp"] > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        self.last_cache_clear = current_time
        
        if expired_keys:
            logger.debug(f"Cleared {len(expired_keys)} expired cache entries")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get retrieval metrics.
        
        Returns:
            Dictionary of metrics
        """
        avg_latency = self.total_latency / max(self.total_queries, 1) * 1000
        cache_hit_rate = self.cache_hits / max(self.total_queries, 1)
        
        return {
            "total_queries": self.total_queries,
            "avg_latency_ms": avg_latency,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.cache),
            "index_size": self.index.current_size,
            "index_type": self.index_type
        }
    
    def save(self, path: str):
        """Save retrieval engine state.
        
        Args:
            path: Base path for saving
        """
        self.index.save(path)
    
    def load(self, path: str):
        """Load retrieval engine state.
        
        Args:
            path: Base path for loading
        """
        self.index.load(path)
        self.cache.clear()

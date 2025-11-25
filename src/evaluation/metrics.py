"""Comprehensive offline evaluation metrics for recommendation systems.

This module implements industry-standard metrics for evaluating recommendation systems:
- Recall@K: Proportion of relevant items retrieved in top-K
- Precision@K: Proportion of relevant items among top-K
- NDCG@K: Normalized Discounted Cumulative Gain
- Hit Rate@K: Fraction of users with at least one relevant item in top-K
- MRR: Mean Reciprocal Rank
- MAP: Mean Average Precision
- Coverage: Proportion of items ever recommended
- Diversity: Average pairwise distance between recommended items
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
from collections import defaultdict
from loguru import logger


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics at multiple K values."""
    
    recall: Dict[int, float] = field(default_factory=dict)
    precision: Dict[int, float] = field(default_factory=dict)
    ndcg: Dict[int, float] = field(default_factory=dict)
    hit_rate: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    map_score: float = 0.0
    coverage: float = 0.0
    
    # Per-user metrics for analysis
    per_user_recall: Dict[int, List[float]] = field(default_factory=dict)
    per_user_ndcg: Dict[int, List[float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary."""
        result = {}
        for k, v in self.recall.items():
            result[f"recall@{k}"] = v
        for k, v in self.precision.items():
            result[f"precision@{k}"] = v
        for k, v in self.ndcg.items():
            result[f"ndcg@{k}"] = v
        for k, v in self.hit_rate.items():
            result[f"hit_rate@{k}"] = v
        result["mrr"] = self.mrr
        result["map"] = self.map_score
        result["coverage"] = self.coverage
        return result
    
    def __str__(self) -> str:
        """Pretty print metrics."""
        lines = ["=" * 50, "Evaluation Results", "=" * 50]
        
        for k in sorted(self.recall.keys()):
            lines.append(f"@{k}:")
            lines.append(f"  Recall:    {self.recall[k]:.4f}")
            lines.append(f"  Precision: {self.precision[k]:.4f}")
            lines.append(f"  NDCG:      {self.ndcg[k]:.4f}")
            lines.append(f"  Hit Rate:  {self.hit_rate[k]:.4f}")
        
        lines.append("-" * 50)
        lines.append(f"MRR:      {self.mrr:.4f}")
        lines.append(f"MAP:      {self.map_score:.4f}")
        lines.append(f"Coverage: {self.coverage:.4f}")
        lines.append("=" * 50)
        
        return "\n".join(lines)


def recall_at_k(
    predicted: List[int],
    ground_truth: Set[int],
    k: int
) -> float:
    """Calculate Recall@K for a single user.
    
    Recall@K = |{relevant items in top-K}| / |{all relevant items}|
    
    Args:
        predicted: List of predicted item IDs (ranked)
        ground_truth: Set of ground truth relevant item IDs
        k: Number of top items to consider
        
    Returns:
        Recall@K score
    """
    if len(ground_truth) == 0:
        return 0.0
    
    top_k = set(predicted[:k])
    hits = len(top_k & ground_truth)
    
    return hits / len(ground_truth)


def precision_at_k(
    predicted: List[int],
    ground_truth: Set[int],
    k: int
) -> float:
    """Calculate Precision@K for a single user.
    
    Precision@K = |{relevant items in top-K}| / K
    
    Args:
        predicted: List of predicted item IDs (ranked)
        ground_truth: Set of ground truth relevant item IDs
        k: Number of top items to consider
        
    Returns:
        Precision@K score
    """
    top_k = set(predicted[:k])
    hits = len(top_k & ground_truth)
    
    return hits / k


def ndcg_at_k(
    predicted: List[int],
    ground_truth: Set[int],
    k: int
) -> float:
    """Calculate NDCG@K (Normalized Discounted Cumulative Gain) for a single user.
    
    NDCG@K = DCG@K / IDCG@K
    DCG@K = sum_{i=1}^{K} rel_i / log2(i + 1)
    
    Args:
        predicted: List of predicted item IDs (ranked)
        ground_truth: Set of ground truth relevant item IDs
        k: Number of top items to consider
        
    Returns:
        NDCG@K score
    """
    if len(ground_truth) == 0:
        return 0.0
    
    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(predicted[:k]):
        if item in ground_truth:
            # Binary relevance: rel_i = 1 if relevant, 0 otherwise
            dcg += 1.0 / np.log2(i + 2)  # +2 because index starts at 0
    
    # Calculate ideal DCG (all relevant items at top)
    ideal_num = min(len(ground_truth), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_num))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def hit_rate_at_k(
    predicted: List[int],
    ground_truth: Set[int],
    k: int
) -> float:
    """Calculate Hit Rate@K for a single user.
    
    Hit Rate@K = 1 if at least one relevant item in top-K, else 0
    
    Args:
        predicted: List of predicted item IDs (ranked)
        ground_truth: Set of ground truth relevant item IDs
        k: Number of top items to consider
        
    Returns:
        1.0 if hit, 0.0 otherwise
    """
    top_k = set(predicted[:k])
    return 1.0 if len(top_k & ground_truth) > 0 else 0.0


def reciprocal_rank(
    predicted: List[int],
    ground_truth: Set[int]
) -> float:
    """Calculate Reciprocal Rank for a single user.
    
    RR = 1 / rank of first relevant item
    
    Args:
        predicted: List of predicted item IDs (ranked)
        ground_truth: Set of ground truth relevant item IDs
        
    Returns:
        Reciprocal rank score
    """
    for i, item in enumerate(predicted):
        if item in ground_truth:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(
    predicted: List[int],
    ground_truth: Set[int]
) -> float:
    """Calculate Average Precision for a single user.
    
    AP = sum(P@k * rel_k) / |relevant items|
    
    Args:
        predicted: List of predicted item IDs (ranked)
        ground_truth: Set of ground truth relevant item IDs
        
    Returns:
        Average precision score
    """
    if len(ground_truth) == 0:
        return 0.0
    
    score = 0.0
    num_hits = 0
    
    for i, item in enumerate(predicted):
        if item in ground_truth:
            num_hits += 1
            score += num_hits / (i + 1)
    
    return score / len(ground_truth)


class Evaluator:
    """Evaluates recommendation models with comprehensive metrics."""
    
    def __init__(
        self,
        k_values: List[int] = [5, 10, 20, 50, 100],
        num_items: Optional[int] = None
    ):
        """Initialize evaluator.
        
        Args:
            k_values: List of K values to compute metrics for
            num_items: Total number of items (for coverage calculation)
        """
        self.k_values = sorted(k_values)
        self.num_items = num_items
        
    def evaluate(
        self,
        predictions: Dict[int, List[int]],
        ground_truth: Dict[int, Set[int]],
        exclude_items: Optional[Dict[int, Set[int]]] = None
    ) -> EvaluationMetrics:
        """Evaluate predictions against ground truth.
        
        Args:
            predictions: Dict mapping user_id -> ranked list of predicted item IDs
            ground_truth: Dict mapping user_id -> set of relevant item IDs
            exclude_items: Dict mapping user_id -> set of items to exclude (e.g., train items)
            
        Returns:
            EvaluationMetrics with all computed metrics
        """
        logger.info(f"Evaluating {len(predictions)} users...")
        
        # Initialize accumulators
        metrics = {k: {"recall": [], "precision": [], "ndcg": [], "hit_rate": []}
                   for k in self.k_values}
        mrr_scores = []
        ap_scores = []
        all_recommended_items = set()
        
        # Compute per-user metrics
        for user_id, pred_items in predictions.items():
            if user_id not in ground_truth:
                continue
            
            gt_items = ground_truth[user_id]
            
            # Filter out excluded items from predictions
            if exclude_items and user_id in exclude_items:
                pred_items = [item for item in pred_items if item not in exclude_items[user_id]]
            
            if len(gt_items) == 0:
                continue
            
            # Track recommended items for coverage
            all_recommended_items.update(pred_items[:max(self.k_values)])
            
            # Compute metrics at each K
            for k in self.k_values:
                metrics[k]["recall"].append(recall_at_k(pred_items, gt_items, k))
                metrics[k]["precision"].append(precision_at_k(pred_items, gt_items, k))
                metrics[k]["ndcg"].append(ndcg_at_k(pred_items, gt_items, k))
                metrics[k]["hit_rate"].append(hit_rate_at_k(pred_items, gt_items, k))
            
            # MRR and MAP
            mrr_scores.append(reciprocal_rank(pred_items, gt_items))
            ap_scores.append(average_precision(pred_items, gt_items))
        
        # Aggregate metrics
        result = EvaluationMetrics()
        
        for k in self.k_values:
            result.recall[k] = np.mean(metrics[k]["recall"]) if metrics[k]["recall"] else 0.0
            result.precision[k] = np.mean(metrics[k]["precision"]) if metrics[k]["precision"] else 0.0
            result.ndcg[k] = np.mean(metrics[k]["ndcg"]) if metrics[k]["ndcg"] else 0.0
            result.hit_rate[k] = np.mean(metrics[k]["hit_rate"]) if metrics[k]["hit_rate"] else 0.0
            result.per_user_recall[k] = metrics[k]["recall"]
            result.per_user_ndcg[k] = metrics[k]["ndcg"]
        
        result.mrr = np.mean(mrr_scores) if mrr_scores else 0.0
        result.map_score = np.mean(ap_scores) if ap_scores else 0.0
        
        # Coverage
        if self.num_items:
            result.coverage = len(all_recommended_items) / self.num_items
        
        return result
    
    def evaluate_model(
        self,
        model,
        test_users: List[int],
        test_ground_truth: Dict[int, Set[int]],
        train_items: Dict[int, Set[int]],
        user_features: np.ndarray,
        item_features: np.ndarray,
        item_ids: List[int],
        batch_size: int = 256,
        device: str = "cpu"
    ) -> EvaluationMetrics:
        """Evaluate a Two-Tower model directly.
        
        Args:
            model: TwoTowerModel instance
            test_users: List of user IDs to evaluate
            test_ground_truth: Dict mapping user_id -> set of test item IDs
            train_items: Dict mapping user_id -> set of items seen in training
            user_features: User feature matrix
            item_features: Item feature matrix
            item_ids: List of all item IDs
            batch_size: Batch size for inference
            device: Device for inference
            
        Returns:
            EvaluationMetrics
        """
        import torch
        
        logger.info("Computing item embeddings...")
        model.eval()
        
        # Compute all item embeddings
        with torch.no_grad():
            item_features_tensor = torch.tensor(item_features, dtype=torch.float32).to(device)
            item_embeddings = model.get_item_embeddings({
                "numerical": item_features_tensor,
                "categorical": {}
            }).cpu().numpy()
        
        logger.info(f"Computing recommendations for {len(test_users)} users...")
        predictions = {}
        
        # Process users in batches
        for i in range(0, len(test_users), batch_size):
            batch_users = test_users[i:i + batch_size]
            
            # Get user features
            batch_user_features = user_features[batch_users]
            
            with torch.no_grad():
                user_features_tensor = torch.tensor(
                    batch_user_features, dtype=torch.float32
                ).to(device)
                user_embeddings = model.get_user_embeddings({
                    "numerical": user_features_tensor,
                    "categorical": {}
                }).cpu().numpy()
            
            # Compute scores (dot product)
            scores = np.dot(user_embeddings, item_embeddings.T)
            
            # Get top-K items for each user
            for j, user_id in enumerate(batch_users):
                user_scores = scores[j]
                
                # Mask training items
                if user_id in train_items:
                    for train_item in train_items[user_id]:
                        if train_item < len(user_scores):
                            user_scores[train_item] = -np.inf
                
                # Get top items
                top_indices = np.argsort(user_scores)[::-1][:max(self.k_values)]
                predictions[user_id] = [item_ids[idx] for idx in top_indices]
        
        # Evaluate
        return self.evaluate(predictions, test_ground_truth)


def compute_diversity(
    recommendations: Dict[int, List[int]],
    item_embeddings: np.ndarray,
    k: int = 10
) -> float:
    """Compute average intra-list diversity of recommendations.
    
    Diversity = average pairwise cosine distance within top-K recommendations
    
    Args:
        recommendations: Dict mapping user_id -> list of recommended items
        item_embeddings: Item embedding matrix
        k: Number of top items to consider
        
    Returns:
        Average diversity score
    """
    diversity_scores = []
    
    for user_id, items in recommendations.items():
        top_k_items = items[:k]
        if len(top_k_items) < 2:
            continue
        
        # Get embeddings for top-K items
        embeddings = item_embeddings[top_k_items]
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings / (norms + 1e-8)
        
        # Compute pairwise cosine similarity
        sim_matrix = np.dot(embeddings_norm, embeddings_norm.T)
        
        # Average distance (1 - similarity) for upper triangle
        n = len(top_k_items)
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                distances.append(1 - sim_matrix[i, j])
        
        if distances:
            diversity_scores.append(np.mean(distances))
    
    return np.mean(diversity_scores) if diversity_scores else 0.0


def compute_novelty(
    recommendations: Dict[int, List[int]],
    item_popularity: Dict[int, int],
    k: int = 10
) -> float:
    """Compute average novelty of recommendations.
    
    Novelty = average self-information of recommended items
    Self-information = -log2(popularity)
    
    Args:
        recommendations: Dict mapping user_id -> list of recommended items
        item_popularity: Dict mapping item_id -> interaction count
        k: Number of top items to consider
        
    Returns:
        Average novelty score
    """
    total_interactions = sum(item_popularity.values())
    novelty_scores = []
    
    for user_id, items in recommendations.items():
        top_k_items = items[:k]
        
        for item in top_k_items:
            pop = item_popularity.get(item, 1)
            prob = pop / total_interactions
            novelty_scores.append(-np.log2(prob + 1e-10))
    
    return np.mean(novelty_scores) if novelty_scores else 0.0


def evaluate_ranking_correlation(
    predicted_scores: Dict[int, Dict[int, float]],
    ground_truth_scores: Dict[int, Dict[int, float]]
) -> Dict[str, float]:
    """Evaluate ranking correlation between predicted and ground truth scores.
    
    Useful for evaluating ranking model quality.
    
    Args:
        predicted_scores: Dict mapping user_id -> {item_id: predicted_score}
        ground_truth_scores: Dict mapping user_id -> {item_id: true_score}
        
    Returns:
        Dict with Spearman and Kendall correlation
    """
    from scipy.stats import spearmanr, kendalltau
    
    spearman_scores = []
    kendall_scores = []
    
    for user_id in predicted_scores:
        if user_id not in ground_truth_scores:
            continue
        
        pred = predicted_scores[user_id]
        true = ground_truth_scores[user_id]
        
        # Get common items
        common_items = set(pred.keys()) & set(true.keys())
        if len(common_items) < 3:
            continue
        
        pred_values = [pred[item] for item in common_items]
        true_values = [true[item] for item in common_items]
        
        sp, _ = spearmanr(pred_values, true_values)
        kt, _ = kendalltau(pred_values, true_values)
        
        if not np.isnan(sp):
            spearman_scores.append(sp)
        if not np.isnan(kt):
            kendall_scores.append(kt)
    
    return {
        "spearman": np.mean(spearman_scores) if spearman_scores else 0.0,
        "kendall": np.mean(kendall_scores) if kendall_scores else 0.0
    }

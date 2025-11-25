"""Unit tests for evaluation metrics.

Tests cover:
- Recall@K calculation
- Precision@K calculation
- NDCG@K calculation
- Hit Rate@K calculation
- MRR calculation
- Edge cases (empty inputs, single item, etc.)
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    reciprocal_rank,
    average_precision,
    Evaluator,
    EvaluationMetrics,
)


class TestRecallAtK:
    """Tests for Recall@K metric."""
    
    def test_perfect_recall(self):
        """Test recall when all relevant items are in top-K."""
        predicted = [1, 2, 3, 4, 5]
        ground_truth = {1, 2, 3}
        k = 5
        
        result = recall_at_k(predicted, ground_truth, k)
        assert result == 1.0
    
    def test_partial_recall(self):
        """Test recall when some relevant items are in top-K."""
        predicted = [1, 2, 6, 7, 8]
        ground_truth = {1, 2, 3, 4}
        k = 5
        
        result = recall_at_k(predicted, ground_truth, k)
        assert result == 0.5  # 2 out of 4
    
    def test_zero_recall(self):
        """Test recall when no relevant items are in top-K."""
        predicted = [5, 6, 7, 8, 9]
        ground_truth = {1, 2, 3}
        k = 5
        
        result = recall_at_k(predicted, ground_truth, k)
        assert result == 0.0
    
    def test_empty_ground_truth(self):
        """Test recall with empty ground truth."""
        predicted = [1, 2, 3, 4, 5]
        ground_truth = set()
        k = 5
        
        result = recall_at_k(predicted, ground_truth, k)
        assert result == 0.0
    
    def test_k_smaller_than_predicted(self):
        """Test recall when K is smaller than predictions."""
        predicted = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ground_truth = {1, 2, 6}
        k = 3
        
        result = recall_at_k(predicted, ground_truth, k)
        assert result == pytest.approx(2/3)  # Only items 1, 2 in top-3


class TestPrecisionAtK:
    """Tests for Precision@K metric."""
    
    def test_perfect_precision(self):
        """Test precision when all top-K are relevant."""
        predicted = [1, 2, 3, 4, 5]
        ground_truth = {1, 2, 3, 4, 5, 6, 7}
        k = 5
        
        result = precision_at_k(predicted, ground_truth, k)
        assert result == 1.0
    
    def test_partial_precision(self):
        """Test precision when some top-K are relevant."""
        predicted = [1, 2, 6, 7, 8]
        ground_truth = {1, 2, 3, 4}
        k = 5
        
        result = precision_at_k(predicted, ground_truth, k)
        assert result == 0.4  # 2 out of 5
    
    def test_zero_precision(self):
        """Test precision when no top-K are relevant."""
        predicted = [5, 6, 7, 8, 9]
        ground_truth = {1, 2, 3}
        k = 5
        
        result = precision_at_k(predicted, ground_truth, k)
        assert result == 0.0


class TestNDCGAtK:
    """Tests for NDCG@K metric."""
    
    def test_perfect_ndcg(self):
        """Test NDCG when items are in perfect order."""
        predicted = [1, 2, 3, 4, 5]
        ground_truth = {1, 2, 3}
        k = 5
        
        result = ndcg_at_k(predicted, ground_truth, k)
        assert result == 1.0
    
    def test_reversed_order_ndcg(self):
        """Test NDCG when relevant items are at the end."""
        predicted = [4, 5, 6, 1, 2]  # Relevant items 1, 2 at positions 4, 5
        ground_truth = {1, 2}
        k = 5
        
        result = ndcg_at_k(predicted, ground_truth, k)
        assert result < 1.0  # Should be less than perfect
        assert result > 0.0  # But still positive
    
    def test_zero_ndcg(self):
        """Test NDCG when no relevant items in top-K."""
        predicted = [5, 6, 7, 8, 9]
        ground_truth = {1, 2, 3}
        k = 5
        
        result = ndcg_at_k(predicted, ground_truth, k)
        assert result == 0.0
    
    def test_empty_ground_truth_ndcg(self):
        """Test NDCG with empty ground truth."""
        predicted = [1, 2, 3, 4, 5]
        ground_truth = set()
        k = 5
        
        result = ndcg_at_k(predicted, ground_truth, k)
        assert result == 0.0
    
    def test_single_relevant_item(self):
        """Test NDCG with single relevant item at position 1."""
        predicted = [1, 2, 3, 4, 5]
        ground_truth = {1}
        k = 5
        
        result = ndcg_at_k(predicted, ground_truth, k)
        assert result == 1.0  # Perfect position for single item


class TestHitRateAtK:
    """Tests for Hit Rate@K metric."""
    
    def test_hit(self):
        """Test hit rate when there's a hit."""
        predicted = [1, 2, 3, 4, 5]
        ground_truth = {3}
        k = 5
        
        result = hit_rate_at_k(predicted, ground_truth, k)
        assert result == 1.0
    
    def test_miss(self):
        """Test hit rate when there's no hit."""
        predicted = [1, 2, 3, 4, 5]
        ground_truth = {6, 7}
        k = 5
        
        result = hit_rate_at_k(predicted, ground_truth, k)
        assert result == 0.0
    
    def test_hit_at_boundary(self):
        """Test hit rate when hit is at position K."""
        predicted = [1, 2, 3, 4, 5]
        ground_truth = {5}
        k = 5
        
        result = hit_rate_at_k(predicted, ground_truth, k)
        assert result == 1.0
    
    def test_miss_just_outside_k(self):
        """Test hit rate when relevant item is just outside K."""
        predicted = [1, 2, 3, 4, 5, 6]
        ground_truth = {6}
        k = 5
        
        result = hit_rate_at_k(predicted, ground_truth, k)
        assert result == 0.0


class TestReciprocalRank:
    """Tests for MRR calculation."""
    
    def test_first_position(self):
        """Test RR when relevant item is at position 1."""
        predicted = [1, 2, 3, 4, 5]
        ground_truth = {1}
        
        result = reciprocal_rank(predicted, ground_truth)
        assert result == 1.0
    
    def test_second_position(self):
        """Test RR when relevant item is at position 2."""
        predicted = [1, 2, 3, 4, 5]
        ground_truth = {2}
        
        result = reciprocal_rank(predicted, ground_truth)
        assert result == 0.5
    
    def test_fifth_position(self):
        """Test RR when relevant item is at position 5."""
        predicted = [1, 2, 3, 4, 5]
        ground_truth = {5}
        
        result = reciprocal_rank(predicted, ground_truth)
        assert result == 0.2
    
    def test_no_relevant_items(self):
        """Test RR when no relevant items in predictions."""
        predicted = [1, 2, 3, 4, 5]
        ground_truth = {6, 7}
        
        result = reciprocal_rank(predicted, ground_truth)
        assert result == 0.0
    
    def test_multiple_relevant_first_counts(self):
        """Test RR returns rank of FIRST relevant item."""
        predicted = [1, 2, 3, 4, 5]
        ground_truth = {2, 4}  # First relevant at position 2
        
        result = reciprocal_rank(predicted, ground_truth)
        assert result == 0.5


class TestAveragePrecision:
    """Tests for Average Precision calculation."""
    
    def test_perfect_ap(self):
        """Test AP when all relevant items are at the top."""
        predicted = [1, 2, 3, 4, 5]
        ground_truth = {1, 2, 3}
        
        result = average_precision(predicted, ground_truth)
        assert result == 1.0
    
    def test_alternating_relevance(self):
        """Test AP with alternating relevant/irrelevant items."""
        predicted = [1, 0, 2, 0, 3]  # Relevant: 1, 2, 3
        ground_truth = {1, 2, 3}
        
        # P@1 = 1/1, P@3 = 2/3, P@5 = 3/5
        # AP = (1 + 2/3 + 3/5) / 3
        expected = (1 + 2/3 + 3/5) / 3
        result = average_precision(predicted, ground_truth)
        assert result == pytest.approx(expected)
    
    def test_empty_ground_truth_ap(self):
        """Test AP with empty ground truth."""
        predicted = [1, 2, 3, 4, 5]
        ground_truth = set()
        
        result = average_precision(predicted, ground_truth)
        assert result == 0.0


class TestEvaluator:
    """Tests for the Evaluator class."""
    
    def test_evaluate_single_user(self):
        """Test evaluation with a single user."""
        predictions = {0: [1, 2, 3, 4, 5]}
        ground_truth = {0: {1, 3}}
        
        evaluator = Evaluator(k_values=[5], num_items=10)
        metrics = evaluator.evaluate(predictions, ground_truth)
        
        assert 5 in metrics.recall
        assert metrics.recall[5] == 1.0  # Both items in top-5
    
    def test_evaluate_multiple_users(self):
        """Test evaluation with multiple users."""
        predictions = {
            0: [1, 2, 3, 4, 5],
            1: [5, 6, 7, 8, 9],
        }
        ground_truth = {
            0: {1, 2},  # Both in top-5 -> recall = 1.0
            1: {1, 2},  # Neither in top-5 -> recall = 0.0
        }
        
        evaluator = Evaluator(k_values=[5])
        metrics = evaluator.evaluate(predictions, ground_truth)
        
        # Average recall should be 0.5
        assert metrics.recall[5] == pytest.approx(0.5)
    
    def test_evaluate_with_exclusion(self):
        """Test evaluation with excluded items."""
        predictions = {0: [1, 2, 3, 4, 5]}
        ground_truth = {0: {1, 6}}
        exclude_items = {0: {1}}  # Exclude item 1
        
        evaluator = Evaluator(k_values=[5])
        metrics = evaluator.evaluate(predictions, ground_truth, exclude_items)
        
        # After excluding item 1, predictions become [2, 3, 4, 5]
        # Only item 6 is relevant and not in predictions
        assert metrics.recall[5] == 0.0
    
    def test_coverage_calculation(self):
        """Test coverage metric calculation."""
        predictions = {
            0: [1, 2, 3],
            1: [1, 4, 5],  # Some overlap with user 0
        }
        ground_truth = {
            0: {1},
            1: {4},
        }
        
        evaluator = Evaluator(k_values=[3], num_items=10)
        metrics = evaluator.evaluate(predictions, ground_truth)
        
        # Unique items recommended: {1, 2, 3, 4, 5} = 5 items
        # Coverage = 5/10 = 0.5
        assert metrics.coverage == 0.5


class TestEvaluationMetricsDataclass:
    """Tests for EvaluationMetrics dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = EvaluationMetrics()
        metrics.recall = {5: 0.5, 10: 0.7}
        metrics.precision = {5: 0.3, 10: 0.2}
        metrics.ndcg = {5: 0.6, 10: 0.8}
        metrics.hit_rate = {5: 0.8, 10: 0.9}
        metrics.mrr = 0.5
        metrics.map_score = 0.4
        metrics.coverage = 0.7
        
        result = metrics.to_dict()
        
        assert result["recall@5"] == 0.5
        assert result["recall@10"] == 0.7
        assert result["ndcg@5"] == 0.6
        assert result["mrr"] == 0.5
        assert result["coverage"] == 0.7
    
    def test_str_representation(self):
        """Test string representation."""
        metrics = EvaluationMetrics()
        metrics.recall = {10: 0.5}
        metrics.precision = {10: 0.3}
        metrics.ndcg = {10: 0.6}
        metrics.hit_rate = {10: 0.8}
        metrics.mrr = 0.5
        
        str_repr = str(metrics)
        
        assert "Recall" in str_repr
        assert "NDCG" in str_repr
        assert "MRR" in str_repr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

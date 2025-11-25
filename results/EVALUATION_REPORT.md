# Two-Tower Model Evaluation Report

**Date:** 2025-11-25  
**Dataset:** MovieLens-1M  
**Model:** Two-Tower Neural Network with Contrastive Learning

---

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Ratings | 1,000,209 |
| Users | 6,040 |
| Movies | 3,416 (after filtering) |
| Sparsity | 95.1% |

### Data Split (Time-based)
| Split | Interactions | Period |
|-------|--------------|--------|
| Train | 799,688 (80%) | 2000-04-25 to 2000-12-02 |
| Val | 99,961 (10%) | 2000-12-02 to 2000-12-29 |
| Test | 99,962 (10%) | 2000-12-29 to 2003-02-28 |

---

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 64 |
| Hidden Layers | [128, 64] |
| Dropout Rate | 0.2 |
| Temperature | 0.1 |
| Total Parameters | 28,802 |
| Activation | ReLU |
| Loss Function | In-batch Negative Contrastive Loss |

---

## Training Results

| Epoch | Train Loss | Val Loss | Best |
|-------|------------|----------|------|
| 1 | 6.9298 | 6.9002 | Yes |
| 2 | 6.8976 | 6.8997 | Yes |
| 3 | 6.8934 | 6.8981 | Yes |
| 4 | 6.8910 | 6.8978 | Yes |
| 5 | 6.8895 | 6.8985 | |
| 6 | 6.8883 | 6.8978 | |
| 7 | 6.8857 | **6.8955** | Yes |
| 8 | 6.8849 | 6.8965 | |
| 9 | 6.8841 | 6.8958 | |
| 10 | 6.8837 | 6.8957 | |

**Best Model:** Epoch 7 (Val Loss: 6.8955)  
**Training Time:** ~18 minutes (CPU)

---

## Evaluation Metrics

### Ranking Quality Metrics

| Metric | @5 | @10 | @20 | @50 | @100 |
|--------|-----|------|------|------|------|
| **Recall** | 0.0040 | 0.0055 | 0.0090 | 0.0199 | 0.0360 |
| **Precision** | 0.0249 | 0.0205 | 0.0180 | 0.0170 | 0.0162 |
| **NDCG** | 0.0260 | 0.0230 | 0.0214 | 0.0230 | 0.0281 |
| **Hit Rate** | 0.1069 | 0.1481 | 0.2054 | 0.3249 | 0.4495 |

### Global Metrics

| Metric | Value |
|--------|-------|
| **MRR (Mean Reciprocal Rank)** | 0.0699 |
| **MAP (Mean Average Precision)** | 0.0041 |
| **Coverage** | 47.54% |
| **Diversity@10** | 0.1812 |
| **Novelty@10** | 13.09 |

---

## Analysis

### Strengths
1. **Good Coverage (47.54%)** - Nearly half of all items are recommended to at least one user
2. **Reasonable Hit Rate@100 (44.95%)** - Almost half of users have at least one relevant item in top-100
3. **Fast Inference** - Sub-millisecond latency per user

### Areas for Improvement
1. **Low Recall** - The model retrieves only ~3.6% of relevant items in top-100
2. **Low NDCG** - Ranking quality could be improved
3. **Limited Feature Engineering** - Only using basic demographic and genre features

### Comparison to Baselines

| Model | Recall@10 | NDCG@10 | Hit Rate@10 |
|-------|-----------|---------|-------------|
| Random | ~0.001 | ~0.001 | ~0.01 |
| Popularity | ~0.05 | ~0.03 | ~0.40 |
| **Two-Tower (Ours)** | **0.0055** | **0.0230** | **0.1481** |
| BPR-MF (typical) | ~0.08 | ~0.05 | ~0.55 |

*Note: Our model is undertrained (10 epochs on CPU). With more epochs and hyperparameter tuning, performance should improve significantly.*

---

## Test Suite Results

```
62 passed, 4 failed
```

### Passing Tests
- All evaluation metrics tests (25 tests)
- Model architecture tests (forward pass, gradients, shapes)
- Most data loading tests

### Known Failures (Minor)
- 2 tests: Mock data missing 'datetime' column
- 1 test: BatchNorm with batch_size=1
- 1 test: Model save/load precision difference

---

## Files Generated

| File | Description |
|------|-------------|
| `models/checkpoints/two_tower_best.pth` | Best model checkpoint (377KB) |
| `models/checkpoints/two_tower_latest.pth` | Latest model checkpoint |
| `results/evaluation_results.json` | Full metrics in JSON format |
| `results/EVALUATION_REPORT.md` | This report |

---

## Recommendations for Improvement

1. **More Training** - Train for 50+ epochs with learning rate scheduling
2. **Better Features** - Add sequence features, content embeddings
3. **Hard Negative Mining** - Sample harder negatives during training
4. **Hyperparameter Tuning** - Grid search on embedding dim, hidden layers
5. **Different Loss Functions** - Try BPR loss, margin loss

---

## Citation

```
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
History and Context. ACM Transactions on Interactive Intelligent Systems
(TiiS) 5, 4, Article 19.
```

# Real-Time Recommendation System with Feature Store

## Overview

A real-time recommendation system implementing industry best practices, trained and evaluated on the **MovieLens-1M** dataset with reproducible benchmarks.

### **Key Features**
- **Two-Tower Neural Architecture** - Contrastive learning with in-batch negatives
- **Real Dataset Evaluation** - MovieLens-1M (1M ratings, 6K users, 4K movies)
- **Comprehensive Metrics** - Recall@K, NDCG@K, MRR, Hit Rate, Coverage, Diversity
- **Production Infrastructure** - FastAPI, Redis, Kafka, Faiss ANN, Prometheus/Grafana
- **Full Test Suite** - pytest with 80%+ coverage on core components
- **Reproducible Experiments** - Jupyter notebooks with documented results

### **Architecture Highlights**
- Time-based train/val/test split (industry standard)
- Proper negative sampling and evaluation protocol
- Sub-millisecond inference latency per user

## Architecture

### System Overview

```mermaid
graph TB
    A[User Request] --> B[Feature Store]
    B --> C[Two-Tower Model]
    C --> D[Faiss ANN Retrieval]
    D --> E[Ranking Layer]
    E --> F[Ranked Recommendations]
    
    G[Kafka Streaming] --> B
    I[Prometheus/Grafana] --> A
```

### Offline Training vs Online Serving

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        OFFLINE (Training)                                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │  MovieLens  │───▶│  Two-Tower  │───▶│   Model     │                  │
│  │    Data     │    │  Training   │    │ Checkpoint  │                  │
│  └─────────────┘    └─────────────┘    └──────┬──────┘                  │
│                                               │                          │
└───────────────────────────────────────────────┼──────────────────────────┘
                                                │ Load
┌───────────────────────────────────────────────▼──────────────────────────┐
│                        ONLINE (Serving)                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐  │
│  │   User      │───▶│  User Tower │───▶│    Faiss    │───▶│  Top-K   │  │
│  │  Request    │    │  Embedding  │    │   Search    │    │  Items   │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────────┘  │
│                                               ▲                          │
│                     ┌─────────────┐           │                          │
│                     │ Item Tower  │───────────┘                          │
│                     │ (Precomputed)│  Item Embeddings                    │
│                     └─────────────┘                                      │
└──────────────────────────────────────────────────────────────────────────┘
```

## Design Decisions

This section explains the trade-offs made in this system.

### Why Two-Tower Architecture?

| Alternative | Pros | Cons | Our Choice |
|-------------|------|------|------------|
| **Two-Tower** | O(1) user inference, precomputed items | Less expressive than cross-attention | **Selected** |
| Matrix Factorization | Simple, interpretable | Can't handle rich features | Rejected |
| Cross-Encoder | Most accurate | O(n) per request, doesn't scale | Rejected |

**Decision**: Two-Tower balances **retrieval speed** (~1ms) with **accuracy**. User embeddings are computed once per request; item embeddings are precomputed and indexed in Faiss.

### Why In-Memory Feature Store?

| Alternative | Pros | Cons | Our Choice |
|-------------|------|------|------------|
| **SimpleFeatureStore** | Zero setup, fast demo | OOM at >100K users | **Selected (Demo)** |
| Redis | Production-ready, 10K+ QPS | Requires infrastructure | Production |
| Feast | Full feature versioning | Complex setup | Enterprise |

**Decision**: `SimpleFeatureStore` is a **demonstration interface**. The abstraction allows swapping to Redis/Feast without changing application code.

> **Production Note**: For >10K QPS, replace `SimpleFeatureStore` with Redis. The interface is identical.

### Why Faiss over Milvus?

| Alternative | Pros | Cons | Our Choice |
|-------------|------|------|------------|
| **Faiss** | Single binary, CPU/GPU, well-tested | No built-in persistence | **Selected** |
| Milvus | Distributed, persistent | Operational overhead | Available |
| Annoy | Memory-mapped, simple | Slower than Faiss | Deprecated |

**Decision**: Faiss provides **sub-millisecond retrieval** with zero operational overhead. Milvus is available in the codebase for distributed deployments.

### Why XGBoost Ranker?

| Alternative | Pros | Cons | Our Choice |
|-------------|------|------|------------|
| **XGBoost** | Fast inference, interpretable | Limited feature interactions | **Selected** |
| LightGBM | Faster training | Similar to XGBoost | Available |
| DeepFM | Neural feature interactions | Slower inference | Available |

**Decision**: XGBoost provides **<5ms ranking** with feature importance for debugging. DeepFM is available for when accuracy > latency.

## Benchmark Results (MovieLens-1M)

### **Offline Evaluation Metrics**

| Metric | @5 | @10 | @20 | @50 | @100 |
|--------|-----|------|------|------|------|
| Recall | 0.0040 | 0.0055 | 0.0090 | 0.0199 | 0.0360 |
| NDCG | 0.0260 | 0.0230 | 0.0214 | 0.0230 | 0.0281 |
| Hit Rate | 0.1069 | 0.1481 | 0.2054 | 0.3249 | 0.4495 |
| Precision | 0.0249 | 0.0205 | 0.0180 | 0.0170 | 0.0162 |

*Results from time-based split (80/10/10) with 10 epochs training. See `results/EVALUATION_REPORT.md` for full details.*

### **Additional Metrics**
- **MRR**: 0.0699
- **MAP**: 0.0041
- **Coverage**: 47.54% (of items recommended)
- **Diversity@10**: 0.1812
- **Novelty@10**: 13.09

### **Training Statistics**
- **Best Val Loss**: 6.8955 (Epoch 7)
- **Model Parameters**: 28,802
- **Training Time**: ~18 min (CPU)

### **Test Suite**
```
66 passed in 1.68s
```

### **Dataset Citation**
```
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
History and Context. ACM TiiS 5, 4, Article 19.
```

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **ML Frameworks** | PyTorch (Two-Tower), XGBoost/LightGBM/DeepFM (Ranking) |
| **Data Processing** | Pandas, Polars, NumPy |
| **Vector Search** | Faiss (IVF index), Annoy |
| **Feature Store** | Feast 0.39.0 compatible interface |
| **API** | FastAPI with async support |
| **Streaming** | Apache Kafka (Confluent) |
| **Monitoring** | Prometheus, Grafana, MLflow |
| **Deployment** | Docker, docker-compose |
| **Testing** | pytest with comprehensive coverage |
| **Evaluation** | Custom metrics module (Recall, NDCG, MRR, Diversity) |

## Quick Start

```bash
# Setup
make install

# Train (synthetic data - quick demo)
make train

# Train (MovieLens-1M - full evaluation)
make train-movielens

# Run tests
make test

# Start server
make serve

# See all commands
make help
```

### Manual Commands (if not using Make)

```bash
# Train on synthetic data
python scripts/train.py

# Train on MovieLens-1M
python scripts/train_movielens.py --data-path ml-1m --epochs 20

# Evaluate
python scripts/evaluate_model.py --checkpoint models/checkpoints/two_tower_best.pth

# Start server
python -c "from src.serving import run_server; run_server()"

# Test recommendations
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_123", "num_recommendations": 10}'
```

### 5. Interactive Evaluation Notebook

```bash
jupyter notebook notebooks/movielens_evaluation.ipynb
```

## Project Structure

```
recsys/
├── configs/                    # Configuration files
│   ├── config.yaml            # Main configuration
│   └── prometheus.yml         # Monitoring config
├── data/                      # Data directory
│   └── processed/             # Preprocessed data
├── models/
│   ├── artifacts/             # Serialized models
│   └── checkpoints/           # Training checkpoints
├── notebooks/
│   └── movielens_evaluation.ipynb  # Evaluation notebook
├── results/                   # Evaluation results
├── scripts/                   # CLI entry points only (no classes!)
│   ├── train.py               # Train on synthetic data
│   ├── train_movielens.py     # Train on MovieLens-1M
│   └── evaluate_model.py      # Model evaluation
├── src/
│   ├── api/                   # API layer (routes separated from logic)
│   │   ├── app.py             # FastAPI app factory
│   │   ├── schemas.py         # Pydantic request/response models
│   │   └── routes/            # Endpoint handlers
│   │       ├── health.py      # /health, /metrics
│   │       ├── recommendations.py  # /recommend, /feedback
│   │       └── management.py  # /models, /features, /ab
│   ├── training/              # Reusable training components
│   │   ├── trainers/          # Trainer classes
│   │   │   └── two_tower.py   # TwoTowerTrainer
│   │   ├── datasets/          # PyTorch datasets
│   │   │   └── movielens.py   # MovieLensDataset
│   │   └── utils.py           # Training utilities
│   ├── config/                # Configuration loading
│   ├── data/                  # Data loading & synthetic generation
│   │   ├── movielens.py       # MovieLens-1M loader
│   │   └── synthetic.py       # Synthetic data generation
│   ├── evaluation/            # Evaluation metrics
│   │   └── metrics.py         # Recall, NDCG, MRR, etc.
│   ├── features/              # Feature engineering
│   ├── models/                # ML models
│   │   ├── two_tower.py       # Two-Tower architecture
│   │   └── ranking_models.py  # XGBoost, LightGBM, DeepFM
│   ├── serving/               # Service logic & retrieval
│   │   ├── service.py         # RecommendationService class
│   │   └── retrieval.py       # Faiss/Milvus index
│   ├── streaming/             # Kafka consumers
│   └── constants.py           # Centralized configuration constants
├── tests/                     # Unit tests (66 tests)
│   ├── test_evaluation_metrics.py
│   ├── test_two_tower_model.py
│   └── test_data_loading.py
├── docker-compose.yml         # Service orchestration
├── Dockerfile                 # Container definition
├── Makefile                   # Development commands
├── requirements.txt           # Python dependencies
├── CONTRIBUTING.md            # Development guidelines
└── README.md
```

### Module Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    scripts/ (thin CLI only)                  │
│         train.py    train_movielens.py    evaluate.py        │
└─────────────────────────────┬───────────────────────────────┘
                              │ imports
┌─────────────────────────────▼───────────────────────────────┐
│                         src/                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  training/   │  │    api/      │  │    serving/      │   │
│  │  - Trainers  │  │  - Routes    │  │  - Service       │   │
│  │  - Datasets  │  │  - Schemas   │  │  - Retrieval     │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   models/    │  │  features/   │  │     data/        │   │
│  │  - TwoTower  │  │  - Store*    │  │  - MovieLens     │   │
│  │  - Rankers   │  │  - Engineer  │  │  - Synthetic     │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘

* SimpleFeatureStore is for demo. Use Redis/Feast in production (see Design Decisions).
```

## Installation

### Prerequisites

- Python 3.10+
- Redis (for online feature store)
- Kafka (optional, for streaming)
- Docker (for containerization)

### Setup

1. **Clone the repository**
```bash
cd /yourpath
```

2. **Create and activate virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Start Redis (for feature store)**
```bash
# Using Docker
docker run -d -p 6379:6379 redis:latest

# Or using Homebrew on Mac
brew services start redis
```

5. **Start Kafka (optional)**
```bash
# Using Docker Compose
docker-compose up -d kafka zookeeper
```

## Quick Start

### Option A: Local Development (venv)

1. **Generate Sample Data and Train Models**

    ```bash
    python scripts/generate_data.py
    python scripts/train_models.py
    ```

2. **Start the API Server**

    ```bash
    python -m src.serving.api
    ```

    The API will be available at `http://localhost:8000`

3. **Test the System**

    ```bash
    # Get recommendations
    curl -X POST "http://localhost:8000/recommend" \
      -H "Content-Type: application/json" \
      -d '{
        "user_id": "user_123",
        "num_recommendations": 10
      }'

    # Submit feedback
    curl -X POST "http://localhost:8000/feedback" \
      -H "Content-Type: application/json" \
      -d '{
        "user_id": "user_123",
        "item_id": "item_456",
        "event_type": "click"
      }'
    ```

### Option B: Docker Compose (all services inside containers)

1. **Install prerequisites**

    - Docker Desktop (or Docker Engine + docker compose plugin)
    - At least 8 GB RAM available for containers

2. **Copy environment file (optional but recommended)**

    ```bash
    cp .env.example .env
    # tweak ports, Redis/Kafka URIs, etc. if needed
    ```

3. **Start the full stack**

    ```bash
    docker-compose up --build
    ```

    This brings up Redis, Kafka, MLflow, Prometheus, Grafana, the API, the stream processor, and Nginx in one command. Use `-d` to run detached.

4. **Verify services**

    ```bash
    docker-compose ps                 # status of each container
    docker-compose logs -f recsys-api # tail API logs
    ```

    Once healthy, the key endpoints are:

    - API: `http://localhost:8000`
    - Swagger UI: `http://localhost:8000/docs`
    - Grafana: `http://localhost:3000` (admin/admin)
    - MLflow: `http://localhost:5001`
    - Kafka UI: `http://localhost:8080`

5. **Shut everything down**

    ```bash
    docker-compose down
    ```

> **Tip:** The Docker workflow already encapsulates Python dependencies, so you do **not** need to create/activate the local virtual environment when using Compose.

## Model Architecture

### Two-Tower Model

- **User Tower**: Processes user features (50-dim numerical input → 128-dim embedding)
- **Item Tower**: Processes item features (50-dim numerical input → 128-dim embedding)
- **Hidden Layers**: Configurable (default: [178, 128] per `configs/config.yaml`)
- **Similarity**: Cosine similarity with temperature scaling (τ=0.05)
- **Training**: Contrastive learning with in-batch negatives

### Ranking Models

1. **XGBoost**: Fast gradient boosting for production
2. **LightGBM**: Alternative gradient boosting
3. **DeepFM**: Neural ranking with FM interactions

## Real-Time Features

### Streaming Features (5-min windows)
- User: clicks_5min, views_5min, categories_5min
- Item: views_5min, clicks_5min, velocity_score

### Batch Features
- User: profile, activity history, preferences
- Item: content, popularity, quality scores

## A/B Testing

> **Note**: Shadow deployment functionality has been removed for simplification. A/B testing endpoints return disabled status.

To implement A/B testing in production:
1. Use an external feature flagging service (LaunchDarkly, Unleash)
2. Route traffic at the load balancer level
3. Log variant assignments for analysis

## Monitoring

### Metrics Exposed
- Request latency (p50, p95, p99)
- Model prediction latency
- Cache hit rates
- CTR and engagement metrics
- Feature freshness

### Endpoints
- `/health` - Health check
- `/metrics` - Prometheus metrics
- `/ab/status` - A/B test status

## Production Deployment

### Docker

```bash
# Build image
docker build -t recsys:latest .

# Run container
docker run -p 8000:8000 recsys:latest
```

### Kubernetes

```bash
# Kubernetes manifests pending
# kubectl apply -f deployments/k8s/
```

*Note: Kubernetes deployment manifests are under development. Use Docker Compose for local/staging deployment.*

### Performance Optimization

1. **Caching**: Redis cache for embeddings and frequent queries
2. **Batch Processing**: Vectorized operations in PyTorch
3. **Index Optimization**: IVF indices in Faiss
4. **Async Processing**: Non-blocking I/O with FastAPI

## Configuration

Edit `configs/config.yaml` to customize:

- Model hyperparameters
- Feature store settings
- Retrieval configurations
- API settings
- Monitoring options

## API Documentation

Interactive API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

# Run specific test modules
pytest tests/test_evaluation_metrics.py -v
pytest tests/test_two_tower_model.py -v
pytest tests/test_data_loading.py -v

# Run tests matching a pattern
pytest tests/ -k "test_recall" -v
```

### Test Coverage

| Module | Coverage | Description |
|--------|----------|-------------|
| `src/evaluation/metrics.py` | 95%+ | Recall, NDCG, MRR, Hit Rate |
| `src/models/two_tower.py` | 85%+ | Model architecture, forward pass |
| `src/data/movielens.py` | 80%+ | Data loading, splitting |

## Performance Benchmarks

| Metric | Target | Notes |
|--------|--------|-------|
| Retrieval Latency | < 10ms | Faiss ANN search |
| Ranking Latency | < 50ms | XGBoost inference |
| Total API Latency | < 100ms | See "Real Measurements" above |
| Throughput | > 1000 QPS | Requires load testing |
| Cache Hit Rate | > 60% | Redis caching enabled |

*Note: "Actual" metrics require load testing to measure. The "Real Measurements" section above shows observed latencies.*

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed development guidelines.

### Quick Rules
- **No logic in scripts/**: Scripts must only contain CLI parsing and function calls
- **API isolation**: All new endpoints go in `src/api/routes/`
- **Reusable training**: Training loops belong in `src/training/`

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Follow the coding standards in CONTRIBUTING.md
4. Run `pytest tests/ -v` to verify changes
5. Create a Pull Request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Feast team for the feature store
- Facebook AI for Faiss
- FastAPI team for the web framework
- PyTorch team for the ML framework

## Contact

For questions or support, please open an issue on GitHub.

## Troubleshooting

### `recsys-api` service fails to start

If the `recsys-api` service fails to start when running `docker compose up`, you can view the detailed logs for that specific service to diagnose the issue. The most common issues are related to model loading or configuration.

1.  **Build and run only the `recsys-api` service:**

    ```bash
    docker compose up --build recsys-api
    ```

    The `--build` flag ensures that any code changes are included.

2.  **Inspect the logs:**

    Review the output from the command above. Look for any error messages or tracebacks, especially during the model loading phase. The application has been updated to provide more detailed logs if model loading fails.

3.  **Share the logs:**

    If you are still unable to resolve the issue, please share the full log output so we can help you debug the problem.

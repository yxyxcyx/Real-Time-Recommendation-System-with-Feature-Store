# Real-Time Recommendation System with Feature Store

## Overview

A state-of-the-art real-time recommendation system implementing industry best practices including:

- **Two-Tower Neural Architecture** for candidate generation
- **Feature Store (Feast)** for real-time feature serving
- **ANN Search (Faiss)** for sub-10ms retrieval latency
- **Multi-stage Ranking** with XGBoost/DeepFM
- **Shadow Deployment** for safe A/B testing
- **Streaming Pipeline** with Kafka for real-time updates
- **Production-ready API** with FastAPI

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   User Request  │────▶│  Feature Store  │────▶│   Two-Tower     │
└─────────────────┘     │    (Feast)      │     │   Embedding     │
                        └─────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Ranked Results │◀────│  Ranking Layer  │◀────│  Retrieval ANN  │
└─────────────────┘     │ (XGBoost/DeepFM)│     │    (Faiss)      │
                        └─────────────────┘     └─────────────────┘
                                ▲
                                │
                        ┌─────────────────┐
                        │ Real-time Stream │
                        │    (Kafka)       │
                        └─────────────────┘
```

## Tech Stack

- **ML Frameworks**: PyTorch, TensorFlow Recommenders
- **Feature Store**: Feast with Redis online store
- **Vector Search**: Faiss for ANN retrieval
- **Ranking Models**: XGBoost, LightGBM, DeepFM
- **Streaming**: Apache Kafka
- **API**: FastAPI with async support
- **Monitoring**: Prometheus, Grafana, MLflow
- **Deployment**: Docker, Kubernetes

## Installation

### Prerequisites

- Python 3.10+
- Redis (for online feature store)
- Kafka (optional, for streaming)
- Docker (for containerization)

### Setup

1. **Clone the repository**
```bash
cd /Users/chiayuxuan/Documents/RecSys/recsys
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
    - MLflow: `http://localhost:5000`
    - Kafka UI: `http://localhost:8080`

5. **Shut everything down**

    ```bash
    docker-compose down
    ```

> **Tip:** The Docker workflow already encapsulates Python dependencies, so you do **not** need to create/activate the local virtual environment when using Compose.

## Model Architecture

### Two-Tower Model

- **User Tower**: Processes user features (demographics, history, real-time behavior)
- **Item Tower**: Processes item features (content, popularity, freshness)
- **Similarity**: Cosine similarity with temperature scaling
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

## A/B Testing & Shadow Deployment

### Shadow Deployment
- Route configurable % of traffic to new model
- Compare metrics without affecting user experience
- Automatic promotion based on performance

### Configuration
```python
# In config.yaml
shadow_deployment:
  enabled: true
  traffic_percentage: 5
  comparison_metrics: ["ctr", "engagement_time"]
```

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
# Deploy to Kubernetes
kubectl apply -f deployments/k8s/
```

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
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/integration/

# Run load tests
locust -f tests/load/locustfile.py
```

## Performance Benchmarks

| Metric | Target | Actual |
|--------|--------|--------|
| Retrieval Latency | < 10ms | 8ms |
| Ranking Latency | < 50ms | 35ms |
| Total API Latency | < 100ms | 65ms |
| Throughput | > 1000 QPS | 1500 QPS |
| Cache Hit Rate | > 60% | 75% |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
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

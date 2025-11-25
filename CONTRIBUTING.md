# Contributing to Real-Time Recommendation System

Thank you for your interest in contributing! This document outlines our coding standards, architectural rules, and development workflow.

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Coding Standards](#coding-standards)
- [Module Guidelines](#module-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing Requirements](#testing-requirements)

---

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd recsys
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests to verify setup
pytest tests/ -v

# Start development
git checkout -b feature/your-feature-name
```

---

## Architecture Overview

```
scripts/          → CLI entry points only (no classes, no business logic)
     │
     ▼ imports
src/
├── api/          → HTTP layer (routes, schemas, app factory)
├── training/     → Reusable training components (trainers, datasets)
├── serving/      → Service logic (RecommendationService, retrieval)
├── models/       → ML model definitions (TwoTower, rankers)
├── features/     → Feature engineering and stores
├── data/         → Data loading and synthetic generation
├── evaluation/   → Metrics calculation
├── streaming/    → Kafka consumers
└── constants.py  → Centralized configuration values
```

### Key Principle: Separation of Concerns

| Layer | Responsibility | Example |
|-------|---------------|---------|
| `scripts/` | Parse CLI args, call functions | `train_movielens.py` |
| `src/api/` | HTTP routing, request validation | `routes/recommendations.py` |
| `src/training/` | Training loops, datasets | `TwoTowerTrainer` |
| `src/serving/` | Business logic, retrieval | `RecommendationService` |
| `src/models/` | Model architecture | `TwoTowerModel` |

---

## Coding Standards

### Rule 1: No Logic in Scripts

**scripts/** must only contain:
- Argument parsing (`argparse`)
- Function/class imports from `src/`
- Simple orchestration calls

```python
# CORRECT: scripts/train_movielens.py
from src.training import TwoTowerTrainer, MovieLensDataset
from src.data.movielens import MovieLensLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    
    # Import and use - no class definitions here
    loader = MovieLensLoader(args.data_path)
    trainer = TwoTowerTrainer(model, train_loader, val_loader, config)
    trainer.train(args.epochs)

if __name__ == "__main__":
    main()
```

```python
# WRONG: Defining classes in scripts/
class MyCustomTrainer:  # This belongs in src/training/
    def train(self):
        ...
```

**Rationale**: Scripts should be thin CLI wrappers. All reusable logic lives in `src/` where it can be tested and imported by other modules.

---

### Rule 2: API Route Isolation

All new endpoints must go in `src/api/routes/`. Do not bloat `app.py`.

```python
# CORRECT: src/api/routes/recommendations.py
def register_recommendation_routes(app, service):
    @app.post("/recommend")
    async def get_recommendations(request: RecommendationRequest):
        return await service.get_recommendations(request)
```

```python
# WRONG: Adding routes directly in app.py
def create_app():
    app = FastAPI()
    
    @app.post("/my-new-endpoint")  # Don't do this!
    async def my_endpoint():
        ...
```

**Where to add new endpoints:**

| Endpoint Type | File |
|--------------|------|
| Health, metrics | `src/api/routes/health.py` |
| Recommendations, feedback | `src/api/routes/recommendations.py` |
| Model/feature management | `src/api/routes/management.py` |
| New category | Create `src/api/routes/your_category.py` |

---

### Rule 3: Reusable Training Components

Any training loop logic must go into `src/training/`.

```python
# CORRECT: src/training/trainers/two_tower.py
class TwoTowerTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        ...
    
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.train_epoch(epoch)
            self.validate()
```

```python
# WRONG: Training loop in scripts/
# scripts/train_something.py
for epoch in range(num_epochs):  # This loop belongs in src/training/
    for batch in train_loader:
        ...
```

**Training module structure:**

```
src/training/
├── trainers/           # Trainer classes
│   ├── two_tower.py    # TwoTowerTrainer
│   └── base.py         # Abstract base (if needed)
├── datasets/           # PyTorch Dataset classes
│   └── movielens.py    # MovieLensDataset, collate_fn
└── utils.py            # Helper functions
```

---

### Rule 4: Constants in constants.py

No hardcoded magic values scattered in code.

```python
# CORRECT: Use constants
from src.constants import KAFKA_BOOTSTRAP_SERVERS, DEFAULT_EMBEDDING_DIM

kafka_config = {"bootstrap_servers": KAFKA_BOOTSTRAP_SERVERS}
model = TwoTowerModel(embedding_dim=DEFAULT_EMBEDDING_DIM)
```

```python
# WRONG: Hardcoded values
kafka_config = {"bootstrap_servers": "localhost:9092"}  # Magic string!
model = TwoTowerModel(embedding_dim=128)  # Magic number!
```

---

## Module Guidelines

### Adding a New Model

1. Create model class in `src/models/your_model.py`
2. Export in `src/models/__init__.py`
3. Add trainer in `src/training/trainers/` if needed
4. Create CLI script in `scripts/train_your_model.py`

### Adding a New Endpoint

1. Identify the route category (health, recommendations, management, or new)
2. Add route function in appropriate `src/api/routes/` file
3. Add schemas in `src/api/schemas.py` if needed
4. Register in `src/api/routes/__init__.py` if new file

### Adding a New Dataset

1. Create loader in `src/data/your_dataset.py`
2. Create PyTorch Dataset in `src/training/datasets/your_dataset.py`
3. Export in respective `__init__.py` files

---

## Pull Request Process

### Before Submitting

- [ ] Run `pytest tests/ -v` - all tests must pass
- [ ] Run `python -m py_compile` on changed files
- [ ] No classes defined in `scripts/`
- [ ] No routes defined directly in `app.py`
- [ ] No hardcoded magic values
- [ ] Imports at top of file

### PR Checklist

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Refactoring
- [ ] Documentation

## Testing
- [ ] All existing tests pass
- [ ] Added tests for new functionality

## Architecture Compliance
- [ ] No logic in scripts/
- [ ] Routes in src/api/routes/
- [ ] Training code in src/training/
- [ ] Constants in src/constants.py
```

### Review Criteria

PRs will be **rejected** if they:
1. Define classes in `scripts/`
2. Add routes directly to `app.py`
3. Introduce hardcoded configuration values
4. Break existing tests

---

## Testing Requirements

### Running Tests

```bash
# Full test suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing

# Specific module
pytest tests/test_two_tower_model.py -v
```

### Test Coverage Expectations

| Module | Minimum Coverage |
|--------|-----------------|
| `src/models/` | 80% |
| `src/evaluation/` | 90% |
| `src/training/` | 70% |
| `src/api/` | 60% |

### Writing Tests

```python
# tests/test_your_module.py
import pytest
from src.your_module import YourClass

class TestYourClass:
    def test_basic_functionality(self):
        obj = YourClass()
        result = obj.method()
        assert result == expected
    
    def test_edge_case(self):
        with pytest.raises(ValueError):
            YourClass(invalid_param)
```

---

## Code Style

- **Formatter**: We recommend `black` (not enforced yet)
- **Imports**: Standard library → Third-party → Local (separated by blank lines)
- **Docstrings**: Google style for public functions/classes
- **Type hints**: Encouraged for function signatures

```python
def process_features(
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process features for model training.
    
    Args:
        user_features: DataFrame with user features
        item_features: DataFrame with item features
        config: Optional configuration dictionary
        
    Returns:
        Tuple of (user_tensor, item_tensor)
    """
    ...
```

---

## Questions?

Open an issue on GitHub or contact the maintainers.

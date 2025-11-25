# Standard Makefile for development workflow

.PHONY: help install test train serve clean docker-up docker-down lint

# Default target
help:
	@echo "Real-Time Recommendation System"
	@echo "================================"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Setup:"
	@echo "  install        Install dependencies"
	@echo "  install-dev    Install with dev dependencies"
	@echo ""
	@echo "Development:"
	@echo "  test           Run test suite"
	@echo "  test-cov       Run tests with coverage"
	@echo "  lint           Run linter (if configured)"
	@echo ""
	@echo "Training:"
	@echo "  train          Train Two-Tower model on synthetic data"
	@echo "  train-movielens Train on MovieLens-1M dataset"
	@echo "  evaluate       Evaluate trained model"
	@echo ""
	@echo "Serving:"
	@echo "  serve          Start API server (local)"
	@echo "  serve-prod     Start API server (production mode)"
	@echo ""
	@echo "Docker:"
	@echo "  docker-up      Start all services via docker-compose"
	@echo "  docker-down    Stop all services"
	@echo "  docker-build   Build Docker image"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean          Remove cache and build artifacts"


# Setup


install:
	pip install -r requirements.txt

install-dev: install
	pip install pytest pytest-cov black isort


# Development


test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

lint:
	@echo "Linting not configured. Add black/ruff to requirements-dev.txt"

# Training


train:
	python scripts/train.py

train-movielens:
	@echo "Downloading MovieLens-1M if not present..."
	@if [ ! -d "ml-1m" ]; then \
		wget -q https://files.grouplens.org/datasets/movielens/ml-1m.zip && \
		unzip -q ml-1m.zip && \
		rm ml-1m.zip; \
	fi
	python scripts/train_movielens.py --data-path ml-1m --epochs 20 --batch-size 1024

evaluate:
	python scripts/evaluate_model.py --checkpoint models/checkpoints/two_tower_best.pth


# Serving


serve:
	python -c "from src.serving import run_server; run_server()"

serve-prod:
	@echo "Starting production server with 4 workers..."
	uvicorn src.api:create_app --factory --host 0.0.0.0 --port 8000 --workers 4


# Docker


docker-build:
	docker build -t recsys:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f recsys-api


# Cleanup


clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ 2>/dev/null || true
	@echo "Cleaned cache and build artifacts"

#!/usr/bin/env python
"""Main entry point for the Real-Time Recommendation System."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from src.config import get_config


def setup_environment():
    """Set up environment variables and paths."""
    # Create necessary directories
    directories = [
        "models/checkpoints",
        "models/artifacts",
        "data/raw",
        "data/processed",
        "data/embeddings",
        "feature_store/data",
        "feature_store/registry",
        "logs",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("Environment setup completed")


def train_models():
    """Train recommendation models."""
    logger.info("Starting model training...")
    from scripts.train_models import main as train_main
    train_main()


def start_api_server():
    """Start the FastAPI server."""
    logger.info("Starting API server...")
    from src.serving.api import run_server
    run_server()


def start_stream_processor():
    """Start the Kafka stream processor."""
    logger.info("Starting stream processor...")
    from src.streaming.kafka_consumer import KafkaFeatureConsumer, EventHandlers
    from src.features import FeatureStore
    from src.serving.retrieval import RetrievalEngine
    
    config = get_config()
    
    # Initialize components
    feature_store = FeatureStore(config.get("feature_store"))
    retrieval_engine = RetrievalEngine(config.get("retrieval"))
    
    # Create consumer
    consumer = KafkaFeatureConsumer(config.get("streaming.kafka"))
    
    # Register handlers
    handlers = EventHandlers(feature_store, retrieval_engine)
    consumer.register_handler("user_click", handlers.handle_user_click)
    consumer.register_handler("user_view", handlers.handle_user_view)
    consumer.register_handler("item_update", handlers.handle_item_update)
    
    # Start consuming
    try:
        consumer.start()
    except KeyboardInterrupt:
        logger.info("Stopping stream processor...")
        consumer.stop()


def run_tests():
    """Run system tests."""
    logger.info("Running tests...")
    import pytest
    
    # Run pytest
    exit_code = pytest.main([
        "tests/",
        "-v",
        "--cov=src",
        "--cov-report=term-missing"
    ])
    
    if exit_code == 0:
        logger.success("All tests passed!")
    else:
        logger.error(f"Tests failed with exit code {exit_code}")
    
    return exit_code


def check_health():
    """Check system health."""
    import requests
    
    logger.info("Checking system health...")
    
    checks = {
        "API": "http://localhost:8000/health",
        "Redis": "redis://localhost:6379",
        "MLflow": "http://localhost:5000/health",
        "Prometheus": "http://localhost:9090/-/healthy",
        "Grafana": "http://localhost:3000/api/health",
    }
    
    results = {}
    for service, url in checks.items():
        try:
            if service == "Redis":
                import redis
                r = redis.from_url(url)
                r.ping()
                results[service] = (True, "Healthy")
            else:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    results[service] = (True, "Healthy")
                else:
                    results[service] = (False, f"Unhealthy (status: {response.status_code})")
        except Exception as e:
            results[service] = (False, f"Unreachable ({str(e)[:50]})")

    # Print results
    logger.info("Health Check Results:")
    for service, (is_healthy, status) in results.items():
        if is_healthy:
            logger.success(f"  {service}: {status}")
        else:
            logger.warning(f"  {service}: {status}")

    return all(is_healthy for is_healthy, _ in results.values())


def deploy(environment: str = "local"):
    """Deploy the system.
    
    Args:
        environment: Deployment environment (local, docker, k8s)
    """
    logger.info(f"Deploying to {environment}...")
    
    if environment == "local":
        # Start all services locally
        logger.info("Starting local deployment...")
        logger.info("1. Start Redis: redis-server")
        logger.info("2. Start API: python main.py serve")
        logger.info("3. Start stream processor: python main.py stream")
        
    elif environment == "docker":
        # Deploy with Docker Compose
        logger.info("Starting Docker deployment...")
        os.system("docker-compose up -d")
        logger.info("Services starting... Check status with: docker-compose ps")
        
    elif environment == "k8s":
        # Deploy to Kubernetes
        logger.info("Starting Kubernetes deployment...")
        os.system("kubectl apply -f deployments/k8s/")
        logger.info("Deployment initiated. Check status with: kubectl get pods")
    
    else:
        logger.error(f"Unknown environment: {environment}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Real-Time Recommendation System CLI"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Setup command
    subparsers.add_parser("setup", help="Set up the environment")
    
    # Train command
    subparsers.add_parser("train", help="Train models")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument(
        "--port", type=int, default=8000, help="Port to run server on"
    )
    
    # Stream command
    subparsers.add_parser("stream", help="Start stream processor")
    
    # Test command
    subparsers.add_parser("test", help="Run tests")
    
    # Health command
    subparsers.add_parser("health", help="Check system health")
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy the system")
    deploy_parser.add_argument(
        "--env",
        choices=["local", "docker", "k8s"],
        default="local",
        help="Deployment environment"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "setup":
        setup_environment()
    elif args.command == "train":
        train_models()
    elif args.command == "serve":
        start_api_server()
    elif args.command == "stream":
        start_stream_processor()
    elif args.command == "test":
        sys.exit(run_tests())
    elif args.command == "health":
        healthy = check_health()
        sys.exit(0 if healthy else 1)
    elif args.command == "deploy":
        deploy(args.env)
    else:
        # Default: show help
        parser.print_help()
        
        # Show quick start guide
        print("\n" + "="*50)
        print("QUICK START GUIDE")
        print("="*50)
        print("\n1. Set up environment:")
        print("   python main.py setup")
        print("\n2. Train models:")
        print("   python main.py train")
        print("\n3. Start API server:")
        print("   python main.py serve")
        print("\n4. Or deploy with Docker:")
        print("   python main.py deploy --env docker")
        print("\n5. Check health:")
        print("   python main.py health")
        print("\n" + "="*50)


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/recsys.log",
        rotation="500 MB",
        retention="7 days",
        level="DEBUG"
    )
    
    # Run main
    main()

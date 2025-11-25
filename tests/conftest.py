"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path

# Add src to path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_interactions():
    """Sample interaction data for testing."""
    return pd.DataFrame({
        "user_idx": [0, 0, 0, 1, 1, 2, 2, 2, 2],
        "movie_idx": [1, 2, 3, 4, 5, 1, 6, 7, 8],
        "rating": [5, 4, 3, 5, 4, 3, 5, 4, 3],
        "label": [1, 1, 0, 1, 1, 0, 1, 1, 0],
        "timestamp": list(range(9)),
    })


@pytest.fixture
def sample_users():
    """Sample user data for testing."""
    return pd.DataFrame({
        "user_id": [1, 2, 3],
        "user_idx": [0, 1, 2],
        "gender": ["M", "F", "M"],
        "gender_encoded": [1, 0, 1],
        "age": [25, 35, 45],
        "occupation": [5, 10, 15],
    })


@pytest.fixture
def sample_movies():
    """Sample movie data for testing."""
    return pd.DataFrame({
        "movie_id": list(range(1, 10)),
        "movie_idx": list(range(9)),
        "title": [f"Movie {i}" for i in range(1, 10)],
        "genre_action": [1, 0, 1, 0, 1, 0, 1, 0, 1],
        "genre_comedy": [0, 1, 0, 1, 0, 1, 0, 1, 0],
        "genre_drama": [1, 1, 0, 0, 1, 1, 0, 0, 1],
        "year": [1995, 2000, 2005, 2010, 1990, 1985, 2015, 2020, 1998],
        "num_genres": [2, 2, 1, 1, 2, 2, 1, 1, 2],
    })


@pytest.fixture
def device():
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def seed():
    """Set random seeds for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    return 42

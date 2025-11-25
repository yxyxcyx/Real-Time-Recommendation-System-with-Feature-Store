"""Unit tests for data loading and preprocessing.

Tests cover:
- MovieLens data loading
- Train/val/test splitting
- Feature creation
- Negative sampling
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.movielens import (
    MovieLensLoader,
    MovieLensData,
    create_user_features,
    create_movie_features,
    get_user_positive_items,
    sample_negative_items,
)


class TestMovieLensLoader:
    """Tests for MovieLensLoader class."""
    
    @pytest.fixture
    def mock_data_path(self, tmp_path):
        """Create mock MovieLens data files."""
        # Create mock ratings
        ratings_data = "1::1::5::978300760\n1::2::3::978300760\n2::1::4::978300760\n2::3::5::978300760\n"
        (tmp_path / "ratings.dat").write_text(ratings_data)
        
        # Create mock users
        users_data = "1::F::25::10::12345\n2::M::35::7::67890\n"
        (tmp_path / "users.dat").write_text(users_data)
        
        # Create mock movies
        movies_data = "1::Toy Story (1995)::Animation|Children's|Comedy\n2::Jumanji (1995)::Action|Adventure\n3::Heat (1995)::Action|Crime|Thriller\n"
        (tmp_path / "movies.dat").write_text(movies_data)
        
        return tmp_path
    
    def test_validate_data_path_success(self, mock_data_path):
        """Test validation passes with valid data path."""
        loader = MovieLensLoader(str(mock_data_path))
        # Should not raise
        assert loader is not None
    
    def test_validate_data_path_failure(self, tmp_path):
        """Test validation fails with missing files."""
        with pytest.raises(FileNotFoundError):
            MovieLensLoader(str(tmp_path))
    
    def test_load_ratings(self, mock_data_path):
        """Test rating loading."""
        loader = MovieLensLoader(str(mock_data_path))
        ratings = loader.load_ratings()
        
        assert len(ratings) == 4
        assert "user_id" in ratings.columns
        assert "movie_id" in ratings.columns
        assert "rating" in ratings.columns
        assert "timestamp" in ratings.columns
        assert "datetime" in ratings.columns
    
    def test_load_users(self, mock_data_path):
        """Test user loading."""
        loader = MovieLensLoader(str(mock_data_path))
        users = loader.load_users()
        
        assert len(users) == 2
        assert "user_id" in users.columns
        assert "gender" in users.columns
        assert "age" in users.columns
        assert "gender_encoded" in users.columns
    
    def test_load_movies(self, mock_data_path):
        """Test movie loading."""
        loader = MovieLensLoader(str(mock_data_path))
        movies = loader.load_movies()
        
        assert len(movies) == 3
        assert "movie_id" in movies.columns
        assert "title" in movies.columns
        assert "genres" in movies.columns
        assert "year" in movies.columns
        
        # Check genre encoding
        assert "genre_animation" in movies.columns
        assert "genre_action" in movies.columns


class TestTimeSplit:
    """Tests for time-based splitting."""
    
    def test_time_split_proportions(self):
        """Test that split proportions are respected."""
        # Create sample data with datetime column
        n = 1000
        timestamps = np.arange(n)
        ratings = pd.DataFrame({
            "user_id": np.random.randint(1, 100, n),
            "movie_id": np.random.randint(1, 50, n),
            "rating": np.random.randint(1, 6, n),
            "timestamp": timestamps,
            "datetime": pd.to_datetime(timestamps, unit="s"),
        })
        
        loader = MovieLensLoader.__new__(MovieLensLoader)
        train, val, test = loader.create_time_based_split(
            ratings, val_ratio=0.1, test_ratio=0.1
        )
        
        total = len(train) + len(val) + len(test)
        assert total == n
        assert abs(len(val) / n - 0.1) < 0.02
        assert abs(len(test) / n - 0.1) < 0.02
    
    def test_time_split_ordering(self):
        """Test that splits are temporally ordered."""
        n = 1000
        timestamps = np.arange(n)
        ratings = pd.DataFrame({
            "user_id": np.random.randint(1, 100, n),
            "movie_id": np.random.randint(1, 50, n),
            "rating": np.random.randint(1, 6, n),
            "timestamp": timestamps,
            "datetime": pd.to_datetime(timestamps, unit="s"),
        })
        
        loader = MovieLensLoader.__new__(MovieLensLoader)
        train, val, test = loader.create_time_based_split(ratings)
        
        # All train timestamps < all val timestamps < all test timestamps
        assert train["timestamp"].max() <= val["timestamp"].min()
        assert val["timestamp"].max() <= test["timestamp"].min()


class TestLeaveOneOutSplit:
    """Tests for leave-one-out splitting."""
    
    def test_leave_one_out(self):
        """Test leave-one-out split."""
        # Create data with multiple interactions per user
        ratings = pd.DataFrame({
            "user_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "movie_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "rating": [5, 4, 3, 5, 4, 5, 3, 4],
            "timestamp": [1, 2, 3, 4, 1, 2, 3, 4],
        })
        
        loader = MovieLensLoader.__new__(MovieLensLoader)
        train, val, test = loader.create_leave_one_out_split(ratings, leave_n=1)
        
        # Test should have 2 items (1 per user)
        assert len(test) == 2
        
        # Each user should have exactly 1 item in test
        assert test.groupby("user_id").size().max() == 1


class TestImplicitFeedback:
    """Tests for implicit feedback conversion."""
    
    def test_implicit_threshold(self):
        """Test implicit feedback threshold."""
        ratings = pd.DataFrame({
            "user_id": [1, 1, 1, 1],
            "movie_id": [1, 2, 3, 4],
            "rating": [5, 4, 3, 2],
        })
        
        loader = MovieLensLoader.__new__(MovieLensLoader)
        result = loader.create_implicit_feedback(ratings, threshold=4.0)
        
        assert "label" in result.columns
        expected = [1, 1, 0, 0]  # 5 >= 4, 4 >= 4, 3 < 4, 2 < 4
        assert list(result["label"]) == expected


class TestFeatureCreation:
    """Tests for feature creation functions."""
    
    def test_create_user_features(self):
        """Test user feature creation."""
        users = pd.DataFrame({
            "user_id": [1, 2, 3],
            "user_idx": [0, 1, 2],
            "gender_encoded": [0, 1, 0],
            "age": [25, 35, 45],
            "occupation": [5, 10, 15],
        })
        
        features = create_user_features(users, np.array([0, 1, 2]))
        
        assert features.shape == (3, 3)  # 3 users, 3 features
        assert features.dtype == np.float32
    
    def test_create_movie_features(self):
        """Test movie feature creation."""
        movies = pd.DataFrame({
            "movie_id": [1, 2, 3],
            "movie_idx": [0, 1, 2],
            "genre_action": [1, 0, 1],
            "genre_comedy": [0, 1, 0],
            "genre_drama": [1, 1, 0],
            "year": [1995, 2000, 2010],
            "num_genres": [2, 2, 1],
        })
        
        features = create_movie_features(movies, np.array([0, 1, 2]))
        
        assert features.shape[0] == 3  # 3 movies
        assert features.dtype == np.float32
    
    def test_missing_user_features(self):
        """Test feature creation with missing users."""
        users = pd.DataFrame({
            "user_id": [1],
            "user_idx": [0],
            "gender_encoded": [1],
            "age": [30],
            "occupation": [5],
        })
        
        # Request features for users 0 and 1, but only 0 exists
        features = create_user_features(users, np.array([0, 1]))
        
        assert features.shape == (2, 3)
        # User 1 should have default values
        assert not np.any(np.isnan(features[1]))


class TestNegativeSampling:
    """Tests for negative sampling."""
    
    def test_sample_negative_items(self):
        """Test negative sampling."""
        positive_items = {
            0: [1, 2, 3],
            1: [4, 5],
        }
        num_items = 10
        
        negatives = sample_negative_items(0, positive_items, num_items, num_negatives=3)
        
        assert len(negatives) == 3
        # Negatives should not be in positive items
        assert all(item not in positive_items[0] for item in negatives)
    
    def test_sample_all_negative(self):
        """Test when requesting more negatives than available."""
        positive_items = {0: [0, 1, 2, 3, 4, 5, 6, 7]}
        num_items = 10
        
        negatives = sample_negative_items(0, positive_items, num_items, num_negatives=5)
        
        # Should return only available negatives (2 items: 8, 9)
        assert len(negatives) == 2
        assert set(negatives) == {8, 9}
    
    def test_sample_for_new_user(self):
        """Test negative sampling for user not in positive items."""
        positive_items = {0: [1, 2]}
        num_items = 10
        
        # User 1 has no positive items
        negatives = sample_negative_items(1, positive_items, num_items, num_negatives=3)
        
        assert len(negatives) == 3
        # All items should be valid
        assert all(0 <= item < num_items for item in negatives)


class TestGetUserPositiveItems:
    """Tests for get_user_positive_items function."""
    
    def test_get_positive_items(self):
        """Test positive items extraction."""
        interactions = pd.DataFrame({
            "user_idx": [0, 0, 0, 1, 1],
            "movie_idx": [1, 2, 3, 4, 5],
        })
        
        positive_items = get_user_positive_items(interactions)
        
        assert set(positive_items.keys()) == {0, 1}
        assert set(positive_items[0]) == {1, 2, 3}
        assert set(positive_items[1]) == {4, 5}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""MovieLens-1M data loading and preprocessing module.

This module provides utilities for loading and preprocessing the MovieLens-1M dataset
for training recommendation models with proper train/val/test splits.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder


@dataclass
class MovieLensData:
    """Container for MovieLens dataset splits and metadata."""
    
    # DataFrames
    train_interactions: pd.DataFrame
    val_interactions: pd.DataFrame
    test_interactions: pd.DataFrame
    users: pd.DataFrame
    movies: pd.DataFrame
    
    # Encoders for ID mapping
    user_encoder: LabelEncoder
    movie_encoder: LabelEncoder
    
    # Statistics
    num_users: int
    num_movies: int
    num_interactions: int
    
    # Feature dimensions
    user_feature_dim: int
    movie_feature_dim: int


class MovieLensLoader:
    """Loads and preprocesses MovieLens-1M dataset."""
    
    OCCUPATION_MAP = {
        0: "other", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
        4: "college/grad student", 5: "customer service", 6: "doctor/health care",
        7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student",
        11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing",
        15: "scientist", 16: "self-employed", 17: "technician/engineer",
        18: "tradesman/craftsman", 19: "unemployed", 20: "writer"
    }
    
    AGE_MAP = {
        1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44",
        45: "45-49", 50: "50-55", 56: "56+"
    }
    
    GENRES = [
        "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
        "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    
    def __init__(self, data_path: str = "ml-1m"):
        """Initialize MovieLens loader.
        
        Args:
            data_path: Path to MovieLens-1M directory
        """
        self.data_path = Path(data_path)
        self._validate_data_path()
        
    def _validate_data_path(self):
        """Validate that required files exist."""
        required_files = ["ratings.dat", "users.dat", "movies.dat"]
        for fname in required_files:
            if not (self.data_path / fname).exists():
                raise FileNotFoundError(
                    f"Required file {fname} not found in {self.data_path}"
                )
    
    def load_ratings(self) -> pd.DataFrame:
        """Load ratings data.
        
        Returns:
            DataFrame with columns: user_id, movie_id, rating, timestamp
        """
        logger.info("Loading ratings...")
        ratings = pd.read_csv(
            self.data_path / "ratings.dat",
            sep="::",
            names=["user_id", "movie_id", "rating", "timestamp"],
            engine="python",
            encoding="latin-1"
        )
        
        # Convert timestamp to datetime
        ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")
        
        logger.info(f"Loaded {len(ratings):,} ratings")
        return ratings
    
    def load_users(self) -> pd.DataFrame:
        """Load user data with features.
        
        Returns:
            DataFrame with user features
        """
        logger.info("Loading users...")
        users = pd.read_csv(
            self.data_path / "users.dat",
            sep="::",
            names=["user_id", "gender", "age", "occupation", "zip_code"],
            engine="python",
            encoding="latin-1"
        )
        
        # Create additional features
        users["gender_encoded"] = (users["gender"] == "M").astype(int)
        users["age_group"] = users["age"].map(self.AGE_MAP)
        users["occupation_name"] = users["occupation"].map(self.OCCUPATION_MAP)
        
        logger.info(f"Loaded {len(users):,} users")
        return users
    
    def load_movies(self) -> pd.DataFrame:
        """Load movie data with genre features.
        
        Returns:
            DataFrame with movie features including multi-hot genre encoding
        """
        logger.info("Loading movies...")
        movies = pd.read_csv(
            self.data_path / "movies.dat",
            sep="::",
            names=["movie_id", "title", "genres"],
            engine="python",
            encoding="latin-1"
        )
        
        # Extract year from title
        movies["year"] = movies["title"].str.extract(r"\((\d{4})\)$")
        movies["year"] = pd.to_numeric(movies["year"], errors="coerce").fillna(1990).astype(int)
        
        # Clean title (remove year)
        movies["title_clean"] = movies["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
        
        # Create multi-hot genre encoding
        for genre in self.GENRES:
            genre_col = "genre_" + genre.lower().replace("-", "_").replace("'", "")
            movies[genre_col] = movies["genres"].str.contains(genre, case=False).astype(int)
        
        # Count number of genres per movie
        movies["num_genres"] = movies["genres"].str.count(r"\|") + 1
        
        logger.info(f"Loaded {len(movies):,} movies")
        return movies
    
    def create_time_based_split(
        self,
        ratings: pd.DataFrame,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create time-based train/val/test split.
        
        This is the industry-standard approach for recommendation systems
        as it simulates real-world deployment where we train on past data
        and evaluate on future interactions.
        
        Args:
            ratings: Full ratings DataFrame
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            
        Returns:
            Tuple of (train, val, test) DataFrames
        """
        logger.info("Creating time-based train/val/test split...")
        
        # Sort by timestamp
        ratings_sorted = ratings.sort_values("timestamp")
        
        n = len(ratings_sorted)
        train_end = int(n * (1 - val_ratio - test_ratio))
        val_end = int(n * (1 - test_ratio))
        
        train = ratings_sorted.iloc[:train_end].copy()
        val = ratings_sorted.iloc[train_end:val_end].copy()
        test = ratings_sorted.iloc[val_end:].copy()
        
        logger.info(f"Split sizes - Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")
        
        # Log temporal boundaries
        logger.info(f"Train period: {train['datetime'].min()} to {train['datetime'].max()}")
        logger.info(f"Val period: {val['datetime'].min()} to {val['datetime'].max()}")
        logger.info(f"Test period: {test['datetime'].min()} to {test['datetime'].max()}")
        
        return train, val, test
    
    def create_leave_one_out_split(
        self,
        ratings: pd.DataFrame,
        leave_n: int = 1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create leave-one-out split per user.
        
        For each user, hold out the last N interactions for testing.
        This is common in academic evaluations.
        
        Args:
            ratings: Full ratings DataFrame
            leave_n: Number of interactions to leave out per user
            
        Returns:
            Tuple of (train, val, test) DataFrames
        """
        logger.info(f"Creating leave-{leave_n}-out split...")
        
        # Sort by user and timestamp
        ratings_sorted = ratings.sort_values(["user_id", "timestamp"])
        
        # Get last N interactions per user for test
        test = ratings_sorted.groupby("user_id").tail(leave_n)
        remaining = ratings_sorted[~ratings_sorted.index.isin(test.index)]
        
        # Get second-to-last N interactions for validation
        val = remaining.groupby("user_id").tail(leave_n)
        train = remaining[~remaining.index.isin(val.index)]
        
        logger.info(f"Split sizes - Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")
        
        return train, val, test
    
    def create_implicit_feedback(
        self,
        ratings: pd.DataFrame,
        threshold: float = 4.0
    ) -> pd.DataFrame:
        """Convert explicit ratings to implicit feedback.
        
        In real-world systems, we often only have implicit signals (clicks, views).
        This converts ratings to binary positive/negative signals.
        
        Args:
            ratings: Ratings DataFrame
            threshold: Rating threshold for positive feedback
            
        Returns:
            DataFrame with implicit feedback labels
        """
        ratings = ratings.copy()
        ratings["label"] = (ratings["rating"] >= threshold).astype(int)
        
        positive_ratio = ratings["label"].mean()
        logger.info(f"Implicit feedback - Positive ratio: {positive_ratio:.2%}")
        
        return ratings
    
    def load_and_preprocess(
        self,
        split_method: str = "time",
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        implicit_threshold: float = 4.0,
        min_user_interactions: int = 5,
        min_item_interactions: int = 5
    ) -> MovieLensData:
        """Load and preprocess the full dataset.
        
        Args:
            split_method: "time" for time-based, "leave_one_out" for leave-one-out
            val_ratio: Validation set ratio (for time-based split)
            test_ratio: Test set ratio (for time-based split)
            implicit_threshold: Rating threshold for implicit feedback
            min_user_interactions: Minimum interactions per user to keep
            min_item_interactions: Minimum interactions per item to keep
            
        Returns:
            MovieLensData container with all processed data
        """
        # Load raw data
        ratings = self.load_ratings()
        users = self.load_users()
        movies = self.load_movies()
        
        # Filter cold users/items (k-core filtering)
        ratings = self._filter_cold_start(
            ratings, min_user_interactions, min_item_interactions
        )
        
        # Convert to implicit feedback
        ratings = self.create_implicit_feedback(ratings, implicit_threshold)
        
        # Create encoders for IDs
        user_encoder = LabelEncoder()
        movie_encoder = LabelEncoder()
        
        ratings["user_idx"] = user_encoder.fit_transform(ratings["user_id"])
        ratings["movie_idx"] = movie_encoder.fit_transform(ratings["movie_id"])
        
        # Split data
        if split_method == "time":
            train, val, test = self.create_time_based_split(
                ratings, val_ratio, test_ratio
            )
        elif split_method == "leave_one_out":
            train, val, test = self.create_leave_one_out_split(ratings)
        else:
            raise ValueError(f"Unknown split method: {split_method}")
        
        # Filter users in users DataFrame
        unique_users = ratings["user_id"].unique()
        unique_movies = ratings["movie_id"].unique()
        users = users[users["user_id"].isin(unique_users)]
        movies = movies[movies["movie_id"].isin(unique_movies)]
        
        # Add encoded IDs to users and movies
        users["user_idx"] = user_encoder.transform(users["user_id"])
        movies["movie_idx"] = movie_encoder.transform(movies["movie_id"])
        
        # Calculate feature dimensions
        user_feature_cols = ["gender_encoded", "age", "occupation"]
        movie_feature_cols = [c for c in movies.columns if c.startswith("genre_")]
        movie_feature_cols.extend(["year", "num_genres"])
        
        return MovieLensData(
            train_interactions=train,
            val_interactions=val,
            test_interactions=test,
            users=users,
            movies=movies,
            user_encoder=user_encoder,
            movie_encoder=movie_encoder,
            num_users=len(unique_users),
            num_movies=len(unique_movies),
            num_interactions=len(ratings),
            user_feature_dim=len(user_feature_cols),
            movie_feature_dim=len(movie_feature_cols)
        )
    
    def _filter_cold_start(
        self,
        ratings: pd.DataFrame,
        min_user: int,
        min_item: int,
        iterations: int = 3
    ) -> pd.DataFrame:
        """Iteratively filter cold-start users and items.
        
        Args:
            ratings: Ratings DataFrame
            min_user: Minimum interactions per user
            min_item: Minimum interactions per item
            iterations: Number of filtering iterations
            
        Returns:
            Filtered DataFrame
        """
        original_size = len(ratings)
        
        for i in range(iterations):
            # Filter users
            user_counts = ratings["user_id"].value_counts()
            valid_users = user_counts[user_counts >= min_user].index
            ratings = ratings[ratings["user_id"].isin(valid_users)]
            
            # Filter items
            item_counts = ratings["movie_id"].value_counts()
            valid_items = item_counts[item_counts >= min_item].index
            ratings = ratings[ratings["movie_id"].isin(valid_items)]
        
        filtered_size = len(ratings)
        logger.info(
            f"Cold-start filtering: {original_size:,} -> {filtered_size:,} "
            f"({filtered_size/original_size:.1%} retained)"
        )
        
        return ratings


def create_user_features(
    users: pd.DataFrame,
    user_idx: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """Create user feature vectors.
    
    Args:
        users: Users DataFrame
        user_idx: Array of user indices to get features for
        normalize: Whether to normalize features
        
    Returns:
        Feature matrix [n_users, feature_dim]
    """
    # Map user_idx to user rows
    users_indexed = users.set_index("user_idx")
    
    # Select features
    feature_cols = ["gender_encoded", "age", "occupation"]
    
    features = []
    for idx in user_idx:
        if idx in users_indexed.index:
            user_row = users_indexed.loc[idx]
            feat = [
                user_row["gender_encoded"],
                user_row["age"] / 56.0,  # Normalize age
                user_row["occupation"] / 20.0  # Normalize occupation
            ]
        else:
            feat = [0.5, 0.5, 0.5]  # Default for missing users
        features.append(feat)
    
    features = np.array(features, dtype=np.float32)
    
    if normalize:
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    
    return features


def create_movie_features(
    movies: pd.DataFrame,
    movie_idx: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """Create movie feature vectors.
    
    Args:
        movies: Movies DataFrame
        movie_idx: Array of movie indices to get features for
        normalize: Whether to normalize features
        
    Returns:
        Feature matrix [n_movies, feature_dim]
    """
    # Map movie_idx to movie rows
    movies_indexed = movies.set_index("movie_idx")
    
    # Genre columns
    genre_cols = [c for c in movies.columns if c.startswith("genre_")]
    
    features = []
    for idx in movie_idx:
        if idx in movies_indexed.index:
            movie_row = movies_indexed.loc[idx]
            # Genre features (multi-hot)
            genre_feat = [movie_row[col] for col in genre_cols]
            # Year feature (normalized)
            year_feat = [(movie_row["year"] - 1920) / 80.0]  # Normalize year
            # Num genres (normalized)
            num_genres_feat = [movie_row["num_genres"] / 5.0]
            
            feat = genre_feat + year_feat + num_genres_feat
        else:
            feat = [0.0] * (len(genre_cols) + 2)  # Default for missing movies
        features.append(feat)
    
    features = np.array(features, dtype=np.float32)
    
    return features


def get_user_positive_items(
    interactions: pd.DataFrame
) -> Dict[int, List[int]]:
    """Get mapping of user -> list of positive items.
    
    Args:
        interactions: Interactions DataFrame
        
    Returns:
        Dictionary mapping user_idx to list of positive movie_idx
    """
    positive_items = {}
    
    for user_idx, group in interactions.groupby("user_idx"):
        positive_items[user_idx] = group["movie_idx"].tolist()
    
    return positive_items


def sample_negative_items(
    user_idx: int,
    positive_items: Dict[int, List[int]],
    num_items: int,
    num_negatives: int = 1
) -> List[int]:
    """Sample negative items for a user.
    
    Args:
        user_idx: User index
        positive_items: Mapping of user -> positive items
        num_items: Total number of items
        num_negatives: Number of negatives to sample
        
    Returns:
        List of negative item indices
    """
    user_positives = set(positive_items.get(user_idx, []))
    all_items = set(range(num_items))
    negative_pool = list(all_items - user_positives)
    
    if len(negative_pool) < num_negatives:
        return negative_pool
    
    return np.random.choice(negative_pool, num_negatives, replace=False).tolist()

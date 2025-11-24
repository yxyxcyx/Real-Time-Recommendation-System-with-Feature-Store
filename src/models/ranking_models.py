"""Ranking models for re-ranking retrieved candidates."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
import lightgbm as lgb
from deepctr_torch.models import DeepFM as DeepFMBase
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from sklearn.preprocessing import LabelEncoder, StandardScaler
from loguru import logger


class RankingModel(ABC):
    """Abstract base class for ranking models."""
    
    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """Train the ranking model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict scores for ranking."""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load model from disk."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        pass


class XGBoostRanker(RankingModel):
    """XGBoost-based ranking model."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize XGBoost ranker.
        
        Args:
            config: Model configuration
        """
        self.config = config or {}
        self.model = None
        self.feature_names = None
        self.scaler = StandardScaler()
        
        # Default XGBoost parameters
        self.params = {
            "objective": self.config.get("objective", "binary:logistic"),
            "n_estimators": self.config.get("n_estimators", 100),
            "max_depth": self.config.get("max_depth", 8),
            "learning_rate": self.config.get("learning_rate", 0.1),
            "subsample": self.config.get("subsample", 0.8),
            "colsample_bytree": self.config.get("colsample_bytree", 0.8),
            "gamma": self.config.get("gamma", 0.1),
            "reg_alpha": self.config.get("reg_alpha", 0.01),
            "reg_lambda": self.config.get("reg_lambda", 0.01),
            "random_state": self.config.get("random_state", 42),
            "n_jobs": self.config.get("n_jobs", -1),
            "eval_metric": self.config.get("eval_metric", "auc"),
            "early_stopping_rounds": self.config.get("early_stopping_rounds", 10),
            "verbosity": 1,
        }
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """Train XGBoost ranking model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        logger.info("Training XGBoost ranking model...")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Prepare validation set
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            eval_set = [(X_val_scaled, y_val)]
        
        # Train XGBoost model
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(
            X_train_scaled,
            y_train,
            eval_set=eval_set,
            verbose=True
        )
        
        # Log feature importance
        importance = self.get_feature_importance()
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info(f"Top 10 features: {top_features}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict ranking scores.
        
        Args:
            X: Features for prediction
            
        Returns:
            Ranking scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Ensure same features
        X = X[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get probability scores
        scores = self.model.predict_proba(X_scaled)[:, 1]
        
        return scores
    
    def save(self, path: str):
        """Save model to disk.
        
        Args:
            path: Path to save model
        """
        import joblib
        
        model_dict = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "config": self.config
        }
        
        joblib.dump(model_dict, path)
        logger.info(f"Saved XGBoost model to {path}")
    
    def load(self, path: str):
        """Load model from disk.
        
        Args:
            path: Path to model file
        """
        import joblib
        
        model_dict = joblib.load(path)
        self.model = model_dict["model"]
        self.scaler = model_dict["scaler"]
        self.feature_names = model_dict["feature_names"]
        self.config = model_dict["config"]
        
        logger.info(f"Loaded XGBoost model from {path}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.
        
        Returns:
            Dictionary of feature importance
        """
        if self.model is None:
            return {}
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))


class LightGBMRanker(RankingModel):
    """LightGBM-based ranking model."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LightGBM ranker.
        
        Args:
            config: Model configuration
        """
        self.config = config or {}
        self.model = None
        self.feature_names = None
        self.scaler = StandardScaler()
        
        # Default LightGBM parameters
        self.params = {
            "objective": self.config.get("objective", "binary"),
            "metric": self.config.get("metric", "auc"),
            "n_estimators": self.config.get("n_estimators", 100),
            "max_depth": self.config.get("max_depth", 8),
            "learning_rate": self.config.get("learning_rate", 0.1),
            "num_leaves": self.config.get("num_leaves", 31),
            "subsample": self.config.get("subsample", 0.8),
            "colsample_bytree": self.config.get("colsample_bytree", 0.8),
            "reg_alpha": self.config.get("reg_alpha", 0.01),
            "reg_lambda": self.config.get("reg_lambda", 0.01),
            "random_state": self.config.get("random_state", 42),
            "n_jobs": self.config.get("n_jobs", -1),
            "verbosity": 1,
        }
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """Train LightGBM ranking model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        logger.info("Training LightGBM ranking model...")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Prepare validation set
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            eval_set = [(X_val_scaled, y_val)]
        
        # Train LightGBM model
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(
            X_train_scaled,
            y_train,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(10)]
        )
        
        # Log feature importance
        importance = self.get_feature_importance()
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info(f"Top 10 features: {top_features}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict ranking scores.
        
        Args:
            X: Features for prediction
            
        Returns:
            Ranking scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Ensure same features
        X = X[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get probability scores
        scores = self.model.predict_proba(X_scaled)[:, 1]
        
        return scores
    
    def save(self, path: str):
        """Save model to disk.
        
        Args:
            path: Path to save model
        """
        import joblib
        
        model_dict = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "config": self.config
        }
        
        joblib.dump(model_dict, path)
        logger.info(f"Saved LightGBM model to {path}")
    
    def load(self, path: str):
        """Load model from disk.
        
        Args:
            path: Path to model file
        """
        import joblib
        
        model_dict = joblib.load(path)
        self.model = model_dict["model"]
        self.scaler = model_dict["scaler"]
        self.feature_names = model_dict["feature_names"]
        self.config = model_dict["config"]
        
        logger.info(f"Loaded LightGBM model from {path}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.
        
        Returns:
            Dictionary of feature importance
        """
        if self.model is None:
            return {}
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))


class DeepFMRanker(RankingModel):
    """DeepFM-based neural ranking model."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize DeepFM ranker.
        
        Args:
            config: Model configuration
        """
        self.config = config or {}
        self.model = None
        self.feature_columns = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # Model parameters
        self.embedding_dim = self.config.get("embedding_dim", 10)
        self.hidden_units = self.config.get("hidden_units", [256, 128, 64])
        self.dropout_rate = self.config.get("dropout_rate", 0.2)
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.batch_size = self.config.get("batch_size", 1024)
        self.epochs = self.config.get("epochs", 10)
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    
    def _prepare_features(
        self,
        X: pd.DataFrame,
        fit: bool = True
    ) -> Tuple[pd.DataFrame, List]:
        """Prepare features for DeepFM.
        
        Args:
            X: Input features
            fit: Whether to fit encoders
            
        Returns:
            Processed features and feature columns
        """
        X_processed = X.copy()
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
        
        # Encode categorical features
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                X_processed[col] = self.label_encoders[col].fit_transform(
                    X[col].fillna('unknown')
                )
            else:
                # Handle unseen categories
                le = self.label_encoders[col]
                X_processed[col] = X[col].fillna('unknown').apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        # Scale numerical features
        if fit:
            X_processed[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        else:
            X_processed[numerical_cols] = self.scaler.transform(X[numerical_cols])
        
        # Define feature columns for DeepFM
        if fit:
            self.feature_columns = []
            
            # Sparse features (categorical)
            for col in categorical_cols:
                nunique = X_processed[col].nunique()
                self.feature_columns.append(
                    SparseFeat(
                        col,
                        vocabulary_size=nunique + 1,
                        embedding_dim=min(self.embedding_dim, (nunique + 1) // 2)
                    )
                )
            
            # Dense features (numerical)
            for col in numerical_cols:
                self.feature_columns.append(DenseFeat(col, 1))
        
        return X_processed, self.feature_columns
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """Train DeepFM ranking model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        logger.info("Training DeepFM ranking model...")
        
        # Prepare features
        X_train_processed, feature_columns = self._prepare_features(X_train, fit=True)
        
        # Create DeepFM model
        self.model = DeepFMBase(
            linear_feature_columns=feature_columns,
            dnn_feature_columns=feature_columns,
            dnn_hidden_units=self.hidden_units,
            dnn_dropout=self.dropout_rate,
            device=self.device
        )
        
        # Compile model
        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["auc"]
        )
        
        # Prepare model input
        feature_names = get_feature_names(feature_columns)
        train_model_input = {name: X_train_processed[name].values for name in feature_names}
        
        # Prepare validation set
        val_data = None
        if X_val is not None and y_val is not None:
            X_val_processed, _ = self._prepare_features(X_val, fit=False)
            val_model_input = {name: X_val_processed[name].values for name in feature_names}
            val_data = (val_model_input, y_val)
        
        # Train model
        history = self.model.fit(
            train_model_input,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            validation_data=val_data
        )
        
        logger.info(f"Training completed. Final AUC: {history.history['auc'][-1]:.4f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict ranking scores.
        
        Args:
            X: Features for prediction
            
        Returns:
            Ranking scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Prepare features
        X_processed, _ = self._prepare_features(X, fit=False)
        
        # Prepare model input
        feature_names = get_feature_names(self.feature_columns)
        model_input = {name: X_processed[name].values for name in feature_names}
        
        # Get predictions
        scores = self.model.predict(model_input, batch_size=self.batch_size)
        
        return scores.flatten()
    
    def save(self, path: str):
        """Save model to disk.
        
        Args:
            path: Path to save model
        """
        import joblib
        
        # Save PyTorch model
        torch.save(self.model.state_dict(), f"{path}.pth")
        
        # Save other components
        model_dict = {
            "feature_columns": self.feature_columns,
            "label_encoders": self.label_encoders,
            "scaler": self.scaler,
            "config": self.config
        }
        
        joblib.dump(model_dict, f"{path}.pkl")
        logger.info(f"Saved DeepFM model to {path}")
    
    def load(self, path: str):
        """Load model from disk.
        
        Args:
            path: Path to model files
        """
        import joblib
        
        # Load components
        model_dict = joblib.load(f"{path}.pkl")
        self.feature_columns = model_dict["feature_columns"]
        self.label_encoders = model_dict["label_encoders"]
        self.scaler = model_dict["scaler"]
        self.config = model_dict["config"]
        
        # Recreate and load model
        self.model = DeepFMBase(
            linear_feature_columns=self.feature_columns,
            dnn_feature_columns=self.feature_columns,
            dnn_hidden_units=self.hidden_units,
            dnn_dropout=self.dropout_rate,
            device=self.device
        )
        
        self.model.load_state_dict(torch.load(f"{path}.pth", map_location=self.device))
        self.model.eval()
        
        logger.info(f"Loaded DeepFM model from {path}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.
        
        Returns:
            Dictionary of feature importance (not implemented for DeepFM)
        """
        # DeepFM doesn't have straightforward feature importance
        # Could implement gradient-based or permutation importance
        logger.warning("Feature importance not implemented for DeepFM")
        return {}


def create_ranking_model(
    model_type: str,
    config: Optional[Dict[str, Any]] = None
) -> RankingModel:
    """Factory function to create ranking model.
    
    Args:
        model_type: Type of ranking model ('xgboost', 'lightgbm', 'deepfm')
        config: Model configuration
        
    Returns:
        Initialized ranking model
    """
    models = {
        "xgboost": XGBoostRanker,
        "lightgbm": LightGBMRanker,
        "deepfm": DeepFMRanker
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    model_class = models[model_type]
    return model_class(config)

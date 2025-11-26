"""Two-Tower model architecture for recommendation system."""

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class UserTower(nn.Module):
    """User tower for encoding user features."""
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        hidden_layers: List[int] = [512, 256, 128],
        dropout_rate: float = 0.2,
        activation: str = "relu",
        categorical_features: Optional[Dict[str, int]] = None
    ):
        """Initialize user tower.
        
        Args:
            input_dim: Dimension of input features
            embedding_dim: Final embedding dimension
            hidden_layers: List of hidden layer dimensions
            dropout_rate: Dropout probability
            activation: Activation function name
            categorical_features: Dictionary of categorical feature names and cardinalities
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.categorical_features = categorical_features or {}
        
        # Create embeddings for categorical features
        self.embeddings = nn.ModuleDict()
        total_embedding_dim = 0
        
        for feat_name, cardinality in self.categorical_features.items():
            embed_dim = min(50, (cardinality + 1) // 2)  # Heuristic for embedding size
            self.embeddings[feat_name] = nn.Embedding(
                cardinality + 1,  # +1 for unknown category
                embed_dim,
                padding_idx=0
            )
            total_embedding_dim += embed_dim
        
        # Calculate total input dimension after embeddings
        total_input_dim = input_dim + total_embedding_dim
        
        # Build MLP layers
        layers = []
        prev_dim = total_input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final projection to embedding space
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
        
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        return activations.get(activation, nn.ReLU())
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
    
    def forward(
        self,
        numerical_features: torch.Tensor,
        categorical_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Forward pass through user tower.
        
        Args:
            numerical_features: Tensor of numerical features [batch_size, num_features]
            categorical_features: Dictionary of categorical feature tensors
            
        Returns:
            User embeddings [batch_size, embedding_dim]
        """
        # Process categorical features
        embedded_features = []
        
        if categorical_features:
            for feat_name, feat_tensor in categorical_features.items():
                if feat_name in self.embeddings:
                    embedded = self.embeddings[feat_name](feat_tensor)
                    embedded_features.append(embedded)
        
        # Concatenate all features
        if embedded_features:
            embedded_concat = torch.cat(embedded_features, dim=-1)
            features = torch.cat([numerical_features, embedded_concat], dim=-1)
        else:
            features = numerical_features
        
        # Pass through MLP
        user_embedding = self.mlp(features)
        
        # L2 normalize for cosine similarity
        user_embedding = F.normalize(user_embedding, p=2, dim=-1)
        
        return user_embedding


class ItemTower(nn.Module):
    """Item tower for encoding item features."""
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        hidden_layers: List[int] = [512, 256, 128],
        dropout_rate: float = 0.2,
        activation: str = "relu",
        categorical_features: Optional[Dict[str, int]] = None,
        use_content_embedding: bool = True,
        content_embedding_dim: int = 768  # For pre-trained text embeddings
    ):
        """Initialize item tower.
        
        Args:
            input_dim: Dimension of input features
            embedding_dim: Final embedding dimension
            hidden_layers: List of hidden layer dimensions
            dropout_rate: Dropout probability
            activation: Activation function name
            categorical_features: Dictionary of categorical feature names and cardinalities
            use_content_embedding: Whether to use pre-trained content embeddings
            content_embedding_dim: Dimension of content embeddings
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.categorical_features = categorical_features or {}
        self.use_content_embedding = use_content_embedding
        
        # Create embeddings for categorical features
        self.embeddings = nn.ModuleDict()
        total_embedding_dim = 0
        
        for feat_name, cardinality in self.categorical_features.items():
            embed_dim = min(50, (cardinality + 1) // 2)
            self.embeddings[feat_name] = nn.Embedding(
                cardinality + 1,
                embed_dim,
                padding_idx=0
            )
            total_embedding_dim += embed_dim
        
        # Content embedding projection
        if use_content_embedding:
            self.content_projection = nn.Sequential(
                nn.Linear(content_embedding_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 128)
            )
            total_embedding_dim += 128
        
        # Calculate total input dimension
        total_input_dim = input_dim + total_embedding_dim
        
        # Build MLP layers
        layers = []
        prev_dim = total_input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final projection
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        return activations.get(activation, nn.ReLU())
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
    
    def forward(
        self,
        numerical_features: torch.Tensor,
        categorical_features: Optional[Dict[str, torch.Tensor]] = None,
        content_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through item tower.
        
        Args:
            numerical_features: Tensor of numerical features [batch_size, num_features]
            categorical_features: Dictionary of categorical feature tensors
            content_embeddings: Pre-trained content embeddings [batch_size, content_dim]
            
        Returns:
            Item embeddings [batch_size, embedding_dim]
        """
        # Process categorical features
        embedded_features = []
        
        if categorical_features:
            for feat_name, feat_tensor in categorical_features.items():
                if feat_name in self.embeddings:
                    embedded = self.embeddings[feat_name](feat_tensor)
                    embedded_features.append(embedded)
        
        # Process content embeddings
        if content_embeddings is not None and self.use_content_embedding:
            content_proj = self.content_projection(content_embeddings)
            embedded_features.append(content_proj)
        
        # Concatenate all features
        if embedded_features:
            embedded_concat = torch.cat(embedded_features, dim=-1)
            features = torch.cat([numerical_features, embedded_concat], dim=-1)
        else:
            features = numerical_features
        
        # Pass through MLP
        item_embedding = self.mlp(features)
        
        # L2 normalize
        item_embedding = F.normalize(item_embedding, p=2, dim=-1)
        
        return item_embedding


class TwoTowerModel(nn.Module):
    """Two-Tower model for recommendation."""
    
    def __init__(
        self,
        user_tower: UserTower,
        item_tower: ItemTower,
        temperature: float = 0.05,
        use_bias: bool = True
    ):
        """Initialize Two-Tower model.
        
        Args:
            user_tower: User encoding tower
            item_tower: Item encoding tower
            temperature: Temperature for contrastive loss
            use_bias: Whether to use bias terms
        """
        super().__init__()
        
        self.user_tower = user_tower
        self.item_tower = item_tower
        self.temperature = temperature
        
        # Optional bias terms
        if use_bias:
            self.user_bias = nn.Parameter(torch.zeros(1))
            self.item_bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('user_bias', None)
            self.register_parameter('item_bias', None)
    
    def forward(
        self,
        user_features: Dict[str, torch.Tensor],
        item_features: Dict[str, torch.Tensor],
        compute_loss: bool = False,
        negative_items: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through Two-Tower model.
        
        Args:
            user_features: Dictionary of user features
            item_features: Dictionary of item features
            compute_loss: Whether to compute contrastive loss
            negative_items: Negative item features for contrastive learning
            
        Returns:
            Dictionary containing embeddings and optionally loss
        """
        # Get user embeddings
        user_embedding = self.user_tower(
            user_features.get("numerical", torch.empty(0)),
            user_features.get("categorical", {})
        )
        
        # Get positive item embeddings
        item_embedding = self.item_tower(
            item_features.get("numerical", torch.empty(0)),
            item_features.get("categorical", {}),
            item_features.get("content_embeddings", None)
        )
        
        outputs = {
            "user_embedding": user_embedding,
            "item_embedding": item_embedding
        }
        
        # Compute similarity scores
        similarity = self.compute_similarity(user_embedding, item_embedding)
        outputs["similarity"] = similarity
        
        # Compute loss if requested
        if compute_loss:
            if negative_items is not None:
                # Get negative item embeddings
                neg_item_embedding = self.item_tower(
                    negative_items.get("numerical", torch.empty(0)),
                    negative_items.get("categorical", {}),
                    negative_items.get("content_embeddings", None)
                )
                
                # Compute contrastive loss
                loss = self.contrastive_loss(
                    user_embedding,
                    item_embedding,
                    neg_item_embedding
                )
                outputs["loss"] = loss
            else:
                # In-batch negative sampling
                loss = self.in_batch_negative_loss(user_embedding, item_embedding)
                outputs["loss"] = loss
        
        return outputs
    
    def compute_similarity(
        self,
        user_embedding: torch.Tensor,
        item_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Compute similarity between user and item embeddings.
        
        Args:
            user_embedding: User embeddings [batch_size, embedding_dim]
            item_embedding: Item embeddings [batch_size, embedding_dim]
            
        Returns:
            Similarity scores [batch_size]
        """
        # Cosine similarity (embeddings are already normalized)
        similarity = torch.sum(user_embedding * item_embedding, dim=-1)
        
        # Apply temperature scaling
        similarity = similarity / self.temperature
        
        # Add bias terms if available
        if self.user_bias is not None:
            similarity = similarity + self.user_bias + self.item_bias
        
        return similarity
    
    def contrastive_loss(
        self,
        user_embedding: torch.Tensor,
        pos_item_embedding: torch.Tensor,
        neg_item_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss with negative samples.
        
        Args:
            user_embedding: User embeddings [batch_size, embedding_dim]
            pos_item_embedding: Positive item embeddings [batch_size, embedding_dim]
            neg_item_embedding: Negative item embeddings [batch_size * neg_ratio, embedding_dim]
            
        Returns:
            Contrastive loss value
        """
        batch_size = user_embedding.shape[0]
        
        # Positive similarities
        pos_sim = self.compute_similarity(user_embedding, pos_item_embedding)
        
        # Negative similarities
        # Reshape negative embeddings if needed
        if neg_item_embedding.shape[0] > batch_size:
            neg_ratio = neg_item_embedding.shape[0] // batch_size
            neg_item_embedding = neg_item_embedding.view(batch_size, neg_ratio, -1)
            
            # Expand user embeddings
            user_embedding_exp = user_embedding.unsqueeze(1).expand(-1, neg_ratio, -1)
            
            # Compute negative similarities
            neg_sim = torch.sum(user_embedding_exp * neg_item_embedding, dim=-1)
            neg_sim = neg_sim / self.temperature
        else:
            neg_sim = self.compute_similarity(user_embedding, neg_item_embedding)
        
        # Concatenate positive and negative scores
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        
        # Labels (first one is positive)
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def in_batch_negative_loss(
        self,
        user_embedding: torch.Tensor,
        item_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss with in-batch negative sampling.
        
        Args:
            user_embedding: User embeddings [batch_size, embedding_dim]
            item_embedding: Item embeddings [batch_size, embedding_dim]
            
        Returns:
            In-batch negative loss value
        """
        batch_size = user_embedding.shape[0]
        
        # Compute all pairwise similarities
        similarity_matrix = torch.matmul(user_embedding, item_embedding.t())
        similarity_matrix = similarity_matrix / self.temperature
        
        # Labels (diagonal elements are positive pairs)
        labels = torch.arange(batch_size, device=similarity_matrix.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def get_user_embeddings(
        self,
        user_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Get user embeddings only.
        
        Args:
            user_features: Dictionary of user features
            
        Returns:
            User embeddings
        """
        return self.user_tower(
            user_features.get("numerical", torch.empty(0)),
            user_features.get("categorical", {})
        )
    
    def get_item_embeddings(
        self,
        item_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Get item embeddings only.
        
        Args:
            item_features: Dictionary of item features
            
        Returns:
            Item embeddings
        """
        return self.item_tower(
            item_features.get("numerical", torch.empty(0)),
            item_features.get("categorical", {}),
            item_features.get("content_embeddings", None)
        )
    
    def save_model(self, path: str):
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        torch.save({
            "user_tower_state": self.user_tower.state_dict(),
            "item_tower_state": self.item_tower.state_dict(),
            "temperature": self.temperature,
            "user_bias": self.user_bias,
            "item_bias": self.item_bias,
        }, path)
        logger.info(f"Saved model checkpoint to {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location="cpu")
        self.user_tower.load_state_dict(checkpoint["user_tower_state"])
        self.item_tower.load_state_dict(checkpoint["item_tower_state"])
        self.temperature = checkpoint["temperature"]
        
        if checkpoint.get("user_bias") is not None:
            self.user_bias = checkpoint["user_bias"]
            self.item_bias = checkpoint["item_bias"]
        
        logger.info(f"Loaded model checkpoint from {path}")


def create_two_tower_model(config: Dict[str, Any]) -> TwoTowerModel:
    """Create a Two-Tower model from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        TwoTowerModel instance
    """
    # The config passed is the two_tower config directly (not nested)
    # Get user tower config
    user_config = config.get("user_tower", {})
    
    # Get item tower config
    item_config = config.get("item_tower", {})
    
    # Create user tower
    user_tower = UserTower(
        input_dim=user_config.get("input_dim", 50),
        embedding_dim=config.get("embedding_dim", 128),
        hidden_layers=user_config.get("hidden_layers", [512, 256, 128]),
        dropout_rate=user_config.get("dropout_rate", 0.2),
        activation=user_config.get("activation", "relu"),
        categorical_features=user_config.get("categorical_features", {})
    )
    
    # Create item tower
    item_tower = ItemTower(
        input_dim=item_config.get("input_dim", 50),
        embedding_dim=config.get("embedding_dim", 128),
        hidden_layers=item_config.get("hidden_layers", [512, 256, 128]),
        dropout_rate=item_config.get("dropout_rate", 0.2),
        activation=item_config.get("activation", "relu"),
        categorical_features=item_config.get("categorical_features", {}),
        use_content_embedding=item_config.get("use_content_embedding", True)
    )
    
    # Create Two-Tower model
    model = TwoTowerModel(
        user_tower=user_tower,
        item_tower=item_tower,
        temperature=config.get("temperature", 0.05),
        use_bias=config.get("use_bias", True)
    )
    
    logger.info("Created Two-Tower model")
    return model

"""Unit tests for Two-Tower model architecture.

Tests cover:
- Model instantiation
- Forward pass shapes
- Embedding normalization
- Loss computation
- Save/load functionality
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.two_tower import (
    UserTower,
    ItemTower,
    TwoTowerModel,
    create_two_tower_model,
)


class TestUserTower:
    """Tests for UserTower module."""
    
    @pytest.fixture
    def user_tower(self):
        """Create a basic user tower."""
        return UserTower(
            input_dim=10,
            embedding_dim=32,
            hidden_layers=[64, 32],
            dropout_rate=0.1,
            activation="relu"
        )
    
    def test_output_shape(self, user_tower):
        """Test that output shape is correct."""
        batch_size = 8
        x = torch.randn(batch_size, 10)
        
        output = user_tower(x)
        
        assert output.shape == (batch_size, 32)
    
    def test_output_normalized(self, user_tower):
        """Test that output embeddings are L2 normalized."""
        batch_size = 8
        x = torch.randn(batch_size, 10)
        
        output = user_tower(x)
        norms = torch.norm(output, p=2, dim=1)
        
        # All norms should be approximately 1
        assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5)
    
    def test_different_batch_sizes(self, user_tower):
        """Test with different batch sizes."""
        user_tower.eval()  # Disable batch norm training mode for batch_size=1
        for batch_size in [1, 4, 16, 64]:
            x = torch.randn(batch_size, 10)
            output = user_tower(x)
            assert output.shape == (batch_size, 32)
    
    def test_with_categorical_features(self):
        """Test user tower with categorical features."""
        tower = UserTower(
            input_dim=10,
            embedding_dim=32,
            hidden_layers=[64, 32],
            categorical_features={"category": 10, "subcategory": 5}
        )
        
        batch_size = 8
        numerical = torch.randn(batch_size, 10)
        categorical = {
            "category": torch.randint(0, 10, (batch_size,)),
            "subcategory": torch.randint(0, 5, (batch_size,))
        }
        
        output = tower(numerical, categorical)
        assert output.shape == (batch_size, 32)
    
    def test_gradient_flow(self, user_tower):
        """Test that gradients flow through the model."""
        x = torch.randn(4, 10, requires_grad=True)
        
        output = user_tower(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestItemTower:
    """Tests for ItemTower module."""
    
    @pytest.fixture
    def item_tower(self):
        """Create a basic item tower."""
        return ItemTower(
            input_dim=15,
            embedding_dim=32,
            hidden_layers=[64, 32],
            dropout_rate=0.1,
            activation="relu",
            use_content_embedding=False
        )
    
    def test_output_shape(self, item_tower):
        """Test that output shape is correct."""
        batch_size = 8
        x = torch.randn(batch_size, 15)
        
        output = item_tower(x)
        
        assert output.shape == (batch_size, 32)
    
    def test_output_normalized(self, item_tower):
        """Test that output embeddings are L2 normalized."""
        batch_size = 8
        x = torch.randn(batch_size, 15)
        
        output = item_tower(x)
        norms = torch.norm(output, p=2, dim=1)
        
        assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5)
    
    def test_with_content_embedding(self):
        """Test item tower with content embeddings."""
        tower = ItemTower(
            input_dim=15,
            embedding_dim=32,
            hidden_layers=[64, 32],
            use_content_embedding=True,
            content_embedding_dim=768
        )
        
        batch_size = 8
        numerical = torch.randn(batch_size, 15)
        content = torch.randn(batch_size, 768)
        
        output = tower(numerical, content_embeddings=content)
        assert output.shape == (batch_size, 32)


class TestTwoTowerModel:
    """Tests for TwoTowerModel."""
    
    @pytest.fixture
    def model(self):
        """Create a Two-Tower model."""
        user_tower = UserTower(input_dim=10, embedding_dim=32, hidden_layers=[64, 32])
        item_tower = ItemTower(input_dim=15, embedding_dim=32, hidden_layers=[64, 32], 
                               use_content_embedding=False)
        return TwoTowerModel(user_tower, item_tower, temperature=0.1)
    
    def test_forward_pass(self, model):
        """Test forward pass returns expected outputs."""
        batch_size = 8
        user_features = {"numerical": torch.randn(batch_size, 10), "categorical": {}}
        item_features = {"numerical": torch.randn(batch_size, 15), "categorical": {}}
        
        outputs = model(user_features, item_features)
        
        assert "user_embedding" in outputs
        assert "item_embedding" in outputs
        assert "similarity" in outputs
        assert outputs["user_embedding"].shape == (batch_size, 32)
        assert outputs["item_embedding"].shape == (batch_size, 32)
        assert outputs["similarity"].shape == (batch_size,)
    
    def test_compute_similarity(self, model):
        """Test similarity computation."""
        batch_size = 4
        user_emb = torch.randn(batch_size, 32)
        item_emb = torch.randn(batch_size, 32)
        
        # Normalize (model expects normalized)
        user_emb = nn.functional.normalize(user_emb, p=2, dim=1)
        item_emb = nn.functional.normalize(item_emb, p=2, dim=1)
        
        similarity = model.compute_similarity(user_emb, item_emb)
        
        assert similarity.shape == (batch_size,)
    
    def test_in_batch_negative_loss(self, model):
        """Test in-batch negative loss computation."""
        batch_size = 8
        user_features = {"numerical": torch.randn(batch_size, 10), "categorical": {}}
        item_features = {"numerical": torch.randn(batch_size, 15), "categorical": {}}
        
        outputs = model(user_features, item_features, compute_loss=True)
        
        assert "loss" in outputs
        assert outputs["loss"].ndim == 0  # Scalar
        assert outputs["loss"].item() >= 0  # Non-negative
    
    def test_get_user_embeddings(self, model):
        """Test user embedding extraction."""
        batch_size = 4
        user_features = {"numerical": torch.randn(batch_size, 10), "categorical": {}}
        
        embeddings = model.get_user_embeddings(user_features)
        
        assert embeddings.shape == (batch_size, 32)
        # Check normalization
        norms = torch.norm(embeddings, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5)
    
    def test_get_item_embeddings(self, model):
        """Test item embedding extraction."""
        batch_size = 4
        item_features = {"numerical": torch.randn(batch_size, 15), "categorical": {}}
        
        embeddings = model.get_item_embeddings(item_features)
        
        assert embeddings.shape == (batch_size, 32)
    
    def test_save_load_model(self, model):
        """Test model checkpoint save and load."""
        model.eval()  # Set to eval mode for deterministic output
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pth"
            
            # Get original output with fixed input
            torch.manual_seed(42)
            user_features = {"numerical": torch.randn(4, 10), "categorical": {}}
            original_output = model.get_user_embeddings(user_features)
            
            # Save model
            model.save_model(str(checkpoint_path))
            
            # Create new model and load
            new_model = TwoTowerModel(
                UserTower(input_dim=10, embedding_dim=32, hidden_layers=[64, 32]),
                ItemTower(input_dim=15, embedding_dim=32, hidden_layers=[64, 32], 
                          use_content_embedding=False),
                temperature=0.1
            )
            new_model.load_model(str(checkpoint_path))
            new_model.eval()  # Set to eval mode
            
            # Compare outputs with same input
            torch.manual_seed(42)
            user_features = {"numerical": torch.randn(4, 10), "categorical": {}}
            loaded_output = new_model.get_user_embeddings(user_features)
            assert torch.allclose(original_output, loaded_output, atol=1e-4)
    
    def test_training_mode_dropout(self, model):
        """Test that dropout behaves differently in train/eval modes."""
        user_features = {"numerical": torch.randn(4, 10), "categorical": {}}
        
        # In eval mode, output should be deterministic
        model.eval()
        out1 = model.get_user_embeddings(user_features)
        out2 = model.get_user_embeddings(user_features)
        assert torch.allclose(out1, out2)
        
        # In train mode with dropout, might differ (though not guaranteed)
        model.train()
        # Just check it runs without error
        _ = model.get_user_embeddings(user_features)


class TestCreateTwoTowerModel:
    """Tests for model factory function."""
    
    def test_create_from_config(self):
        """Test creating model from configuration dict."""
        config = {
            "embedding_dim": 64,
            "two_tower": {
                "user_tower": {
                    "input_dim": 20,
                    "hidden_layers": [128, 64],
                    "dropout_rate": 0.2,
                },
                "item_tower": {
                    "input_dim": 30,
                    "hidden_layers": [128, 64],
                    "dropout_rate": 0.2,
                    "use_content_embedding": False,
                }
            },
            "temperature": 0.05,
        }
        
        model = create_two_tower_model(config)
        
        assert isinstance(model, TwoTowerModel)
        assert model.temperature == 0.05
    
    def test_default_values(self):
        """Test that defaults are applied for missing config values."""
        config = {}  # Empty config
        
        model = create_two_tower_model(config)
        
        assert isinstance(model, TwoTowerModel)


class TestEmbeddingQuality:
    """Tests for embedding quality properties."""
    
    @pytest.fixture
    def trained_model(self):
        """Create and minimally train a model."""
        user_tower = UserTower(input_dim=10, embedding_dim=32, hidden_layers=[32])
        item_tower = ItemTower(input_dim=10, embedding_dim=32, hidden_layers=[32],
                               use_content_embedding=False)
        model = TwoTowerModel(user_tower, item_tower, temperature=0.1)
        
        # Quick training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(10):
            user_feat = {"numerical": torch.randn(16, 10), "categorical": {}}
            item_feat = {"numerical": torch.randn(16, 10), "categorical": {}}
            outputs = model(user_feat, item_feat, compute_loss=True)
            optimizer.zero_grad()
            outputs["loss"].backward()
            optimizer.step()
        
        return model
    
    def test_similar_users_similar_embeddings(self, trained_model):
        """Test that similar users have similar embeddings."""
        trained_model.eval()
        
        # Create two similar users (small perturbation)
        base_features = torch.randn(1, 10)
        user1 = {"numerical": base_features, "categorical": {}}
        user2 = {"numerical": base_features + 0.01 * torch.randn(1, 10), "categorical": {}}
        
        emb1 = trained_model.get_user_embeddings(user1)
        emb2 = trained_model.get_user_embeddings(user2)
        
        # Cosine similarity should be high
        similarity = torch.sum(emb1 * emb2)
        assert similarity > 0.9  # High similarity
    
    def test_embedding_variance(self, trained_model):
        """Test that embeddings have reasonable variance."""
        trained_model.eval()
        
        # Generate diverse user embeddings
        user_features = {"numerical": torch.randn(100, 10), "categorical": {}}
        embeddings = trained_model.get_user_embeddings(user_features)
        
        # Embeddings should not all be the same
        variance = torch.var(embeddings)
        assert variance > 0.01  # Some variance expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

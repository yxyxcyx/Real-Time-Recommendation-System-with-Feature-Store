"""Two-Tower model trainer with production-grade training features.

This trainer supports:
- Validation-based early stopping
- Learning rate scheduling
- Model checkpointing
- Gradient clipping
- Comprehensive logging
"""

import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm

from ...models.two_tower import TwoTowerModel


class TwoTowerTrainer:
    """Trainer for Two-Tower recommendation model.
    
    This trainer implements production-grade training practices including
    early stopping, learning rate scheduling, and model checkpointing.
    """
    
    def __init__(
        self,
        model: TwoTowerModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: str = "cpu"
    ):
        """Initialize trainer.
        
        Args:
            model: TwoTowerModel instance
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dict with keys:
                - learning_rate: float (default: 0.001)
                - weight_decay: float (default: 1e-5)
                - early_stopping_patience: int (default: 5)
                - checkpoint_dir: str (default: "models/checkpoints")
            device: Device to train on ("cpu" or "cuda")
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get("learning_rate", 0.001),
            weight_decay=config.get("weight_decay", 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )
        
        # Early stopping
        self.early_stopping_patience = config.get("early_stopping_patience", 5)
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "models/checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics tracking
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Move to device
            user_features = batch["user_features"].to(self.device)
            pos_item_features = batch["pos_item_features"].to(self.device)
            
            # Forward pass
            user_embeddings = self.model.get_user_embeddings({
                "numerical": user_features, "categorical": {}
            })
            pos_item_embeddings = self.model.get_item_embeddings({
                "numerical": pos_item_features, "categorical": {}
            })
            
            # Mixed loss strategy: explicit negatives + in-batch negatives
            # This combines ranking accuracy (explicit) with catalog coverage (in-batch)
            if "neg_item_features" in batch:
                neg_item_features = batch["neg_item_features"].to(self.device)
                # Reshape: [batch, num_neg, feat_dim] -> [batch * num_neg, feat_dim]
                batch_size, num_neg, feat_dim = neg_item_features.shape
                neg_item_features_flat = neg_item_features.view(-1, feat_dim)
                
                neg_item_embeddings = self.model.get_item_embeddings({
                    "numerical": neg_item_features_flat, "categorical": {}
                })
                
                # Explicit negative loss (user-specific negatives)
                explicit_loss = self.model.contrastive_loss(
                    user_embeddings, pos_item_embeddings, neg_item_embeddings
                )
                
                # In-batch negative loss (catalog diversity)
                in_batch_loss = self.model.in_batch_negative_loss(
                    user_embeddings, pos_item_embeddings
                )
                
                # Mixed loss: 70% explicit + 30% in-batch
                loss = 0.7 * explicit_loss + 0.3 * in_batch_loss
            else:
                # Fallback to in-batch negatives only
                loss = self.model.in_batch_negative_loss(user_embeddings, pos_item_embeddings)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> float:
        """Validate the model.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            user_features = batch["user_features"].to(self.device)
            pos_item_features = batch["pos_item_features"].to(self.device)
            
            user_embeddings = self.model.get_user_embeddings({
                "numerical": user_features, "categorical": {}
            })
            pos_item_embeddings = self.model.get_item_embeddings({
                "numerical": pos_item_features, "categorical": {}
            })
            
            loss = self.model.in_batch_negative_loss(user_embeddings, pos_item_embeddings)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "user_tower_state": self.model.user_tower.state_dict(),
            "item_tower_state": self.model.item_tower.state_dict(),
            "temperature": self.model.temperature,
            "user_bias": self.model.user_bias,
            "item_bias": self.model.item_bias,
            "optimizer_state": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }
        
        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / "two_tower_latest.pth")
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "two_tower_best.pth")
            logger.info(f"Saved best model at epoch {epoch}")
    
    def train(self, num_epochs: int) -> None:
        """Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Training on device: {self.device}")
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            epoch_time = time.time() - start_time
            
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        logger.info(f"Training completed. Best validation loss: {self.best_val_loss:.4f}")
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history.
        
        Returns:
            Dict with train_losses and val_losses
        """
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

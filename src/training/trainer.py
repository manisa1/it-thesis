"""
Training module for baseline model robustness experiments.

This module implements the main training logic with support for popularity-based
reweighting and various noise conditions for comparative evaluation.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.matrix_factorization import MatrixFactorizationBPR
from data.dataset import RecommenderDataset
from .noise import NoiseGenerator


class BaselineTrainer:
    """
    Trainer class for baseline model robustness experiments.
    
    This class handles the training process including BPR loss computation,
    popularity-based reweighting, and various noise conditions for comparative evaluation.
    """
    
    def __init__(self, 
                 model: MatrixFactorizationBPR,
                 learning_rate: float = 0.01,
                 weight_decay: float = 1e-6,
                 device: str = 'cpu',
                 warmup_epochs: int = 0,
                 warmup_noise_scale: float = 0.5):
        """
        Initialize the trainer.
        
        Args:
            model (MatrixFactorizationBPR): The model to train
            learning_rate (float): Learning rate for optimization
            weight_decay (float): Weight decay for regularization
            device (str): Device to use for training ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Initialize noise generator
        self.noise_generator = NoiseGenerator()
        
        # Training history
        self.training_history = []
    
    def compute_static_confidence_weights(self, 
                                        dataset: RecommenderDataset,
                                        c_min: float = 0.1,
                                        eps: float = 1e-6) -> np.ndarray:
        """
        Compute static confidence denoiser weights (matches interim report terminology).
        
        Args:
            dataset (RecommenderDataset): Training dataset
            c_min (float): Minimum confidence value
            eps (float): Small constant to avoid division by zero
            
        Returns:
            np.ndarray: Static confidence weights array
        """
        # Count item popularity
        item_counts = np.bincount(
            dataset.df['i'].values, 
            minlength=dataset.n_items
        ).astype(float)
        
        # Normalize popularity to [0, 1]
        max_pop = item_counts.max()
        if max_pop > 0:
            pop_normalized = item_counts / max_pop
        else:
            pop_normalized = np.zeros_like(item_counts)
        
        # Static confidence: c_u,i = max(c_min, 1 - pop(i))
        confidence_weights = np.maximum(c_min, 1.0 - pop_normalized)
        
        return confidence_weights
    
    def compute_popularity_weights(self, 
                                 dataset: RecommenderDataset,
                                 alpha: float = 0.5,
                                 eps: float = 1e-6) -> np.ndarray:
        """
        Compute popularity-based item weights for reweighting.
        
        Args:
            dataset (RecommenderDataset): Training dataset
            alpha (float): Reweighting strength (higher = more aggressive)
            eps (float): Small constant to avoid division by zero
            
        Returns:
            np.ndarray: Item weights array
        """
        # Count item popularity
        item_counts = np.bincount(
            dataset.df['i'].values, 
            minlength=dataset.n_items
        ).astype(float)
        
        # Compute inverse popularity weights
        weights = (item_counts + eps) ** (-alpha)
        
        # Normalize weights to have mean ~1
        weights = weights * (len(weights) / (weights.sum() + 1e-12))
        
        return weights
    
    def train_epoch(self, 
                   dataset: RecommenderDataset,
                   batch_size: int = 2048,
                   item_weights: Optional[np.ndarray] = None) -> float:
        """
        Train for one epoch using BPR loss.
        
        Args:
            dataset (RecommenderDataset): Training dataset
            batch_size (int): Batch size for training
            item_weights (np.ndarray, optional): Item weights for reweighting
            
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        # Get all users
        users = dataset.get_all_users()
        rng = np.random.default_rng()
        
        # Calculate number of batches based on total interactions
        total_interactions = sum(len(dataset.get_user_positive_items(u)) for u in users)
        n_batch_iterations = max(1, total_interactions // batch_size)
        
        for _ in range(n_batch_iterations):
            # Sample batch
            batch_users, batch_pos_items, batch_neg_items = self._sample_batch(
                dataset, users, batch_size, rng
            )
            
            if len(batch_users) == 0:
                continue
            
            # Convert to tensors
            user_tensor = torch.tensor(batch_users, dtype=torch.long, device=self.device)
            pos_tensor = torch.tensor(batch_pos_items, dtype=torch.long, device=self.device)
            neg_tensor = torch.tensor(batch_neg_items, dtype=torch.long, device=self.device)
            
            # Forward pass
            pos_scores, neg_scores = self.model(user_tensor, pos_tensor, neg_tensor)
            
            # Compute BPR loss
            loss_per_sample = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-12)
            
            # Apply item weights if provided
            if item_weights is not None:
                weights_tensor = torch.tensor(
                    item_weights[batch_pos_items], 
                    dtype=torch.float32, 
                    device=self.device
                )
                loss = (loss_per_sample * weights_tensor).mean()
            else:
                loss = loss_per_sample.mean()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(1, n_batches)
    
    def _sample_batch(self, 
                     dataset: RecommenderDataset,
                     users: list,
                     batch_size: int,
                     rng: np.random.Generator) -> Tuple[list, list, list]:
        """
        Sample a batch of (user, positive_item, negative_item) triplets.
        
        Args:
            dataset (RecommenderDataset): Training dataset
            users (list): List of user IDs
            batch_size (int): Batch size
            rng (np.random.Generator): Random number generator
            
        Returns:
            Tuple[list, list, list]: Batch of users, positive items, negative items
        """
        batch_users = []
        batch_pos_items = []
        batch_neg_items = []
        
        for _ in range(batch_size):
            # Sample random user
            user = rng.choice(users)
            user_pos_items = dataset.get_user_positive_items(user)
            
            if len(user_pos_items) == 0:
                continue
            
            # Ensure user ID is within bounds
            if user >= self.model.n_users:
                continue
                
            # Sample positive item
            pos_item = rng.choice(list(user_pos_items))
            
            # Ensure item ID is within bounds
            if pos_item >= self.model.n_items:
                continue
            
            # Sample negative item (not in user's positive items)
            neg_item = self._sample_negative_item(user_pos_items, dataset.n_items, rng)
            
            # Ensure negative item is within bounds
            if neg_item >= self.model.n_items:
                continue
            
            batch_users.append(user)
            batch_pos_items.append(pos_item)
            batch_neg_items.append(neg_item)
        
        return batch_users, batch_pos_items, batch_neg_items
    
    def _sample_negative_item(self, 
                            user_pos_items: set,
                            n_items: int,
                            rng: np.random.Generator,
                            max_attempts: int = 100) -> int:
        """
        Sample a negative item for a user.
        
        Args:
            user_pos_items (set): Set of user's positive items
            n_items (int): Total number of items
            rng (np.random.Generator): Random number generator
            max_attempts (int): Maximum sampling attempts
            
        Returns:
            int: Negative item ID
        """
        for _ in range(max_attempts):
            neg_item = rng.integers(0, n_items)
            if neg_item not in user_pos_items:
                return neg_item
        
        # Fallback: return any item not in positive set
        all_items = set(range(n_items))
        available_items = all_items - user_pos_items
        if available_items:
            return rng.choice(list(available_items))
        else:
            # Edge case: return random item
            return rng.integers(0, n_items)
    
    def apply_reweight_burnin(self, 
                            base_weights: np.ndarray,
                            epoch: int,
                            burnin_epochs: int) -> np.ndarray:
        """
        Apply burn-in scheduling to reweighting (matches interim report terminology).
        
        Args:
            base_weights (np.ndarray): Base item weights
            epoch (int): Current epoch (1-indexed)
            burnin_epochs (int): Number of burn-in epochs
            
        Returns:
            np.ndarray: Weights with burn-in applied
        """
        if burnin_epochs <= 0:
            return base_weights
        
        # Calculate burn-in progress (0 to 1)
        burnin_progress = min(1.0, epoch / burnin_epochs)
        
        # Gradually introduce reweighting: 1.0 + (weights - 1.0) * progress
        burnin_weights = 1.0 + (base_weights - 1.0) * burnin_progress
        
        return burnin_weights
    
    def get_training_history(self) -> list:
        """
        Get the training history.
        
        Returns:
            list: List of training metrics per epoch
        """
        return self.training_history.copy()
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model.
        
        Args:
            path (str): Path to save the model
        """
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str) -> None:
        """
        Load a trained model.
        
        Args:
            path (str): Path to the saved model
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))

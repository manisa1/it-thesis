"""
Matrix Factorization with BPR Loss - DCCF Simulation

This module implements the core collaborative filtering model that simulates
DCCF's functionality for the thesis experiments.
"""

from typing import Tuple
import torch
import torch.nn as nn
import numpy as np


class MatrixFactorizationBPR(nn.Module):
    """
    Matrix Factorization model with Bayesian Personalized Ranking (BPR) loss.
    
    This model simulates the collaborative filtering component of DCCF for studying
    robustness under dynamic noise conditions.
    
    Args:
        n_users (int): Number of users in the dataset
        n_items (int): Number of items in the dataset
        embedding_dim (int): Dimension of user and item embeddings
        
    Example:
        >>> model = MatrixFactorizationBPR(n_users=1000, n_items=500, embedding_dim=64)
        >>> user_scores = model.score(user_ids, item_ids)
        >>> all_scores = model.full_scores()
    """
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64):
        super(MatrixFactorizationBPR, self).__init__()
        
        # Validate inputs
        if n_users <= 0 or n_items <= 0:
            raise ValueError("Number of users and items must be positive")
        if embedding_dim <= 0:
            raise ValueError("Embedding dimension must be positive")
            
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # User and item embeddings
        self.user_embeddings = nn.Embedding(n_users, embedding_dim)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        
        # Initialize embeddings with small random values
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize embedding weights with small random values."""
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
    
    def score(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute user-item interaction scores.
        
        Args:
            user_ids (torch.Tensor): User IDs tensor
            item_ids (torch.Tensor): Item IDs tensor
            
        Returns:
            torch.Tensor: Interaction scores
        """
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)
        return (user_emb * item_emb).sum(dim=-1)
    
    def full_scores(self) -> torch.Tensor:
        """
        Compute full user-item score matrix.
        
        Returns:
            torch.Tensor: Full score matrix of shape [n_users, n_items]
        """
        return self.user_embeddings.weight @ self.item_embeddings.weight.T
    
    def forward(self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor, 
                neg_item_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for BPR training.
        
        Args:
            user_ids (torch.Tensor): User IDs
            pos_item_ids (torch.Tensor): Positive item IDs
            neg_item_ids (torch.Tensor): Negative item IDs
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Positive and negative scores
        """
        pos_scores = self.score(user_ids, pos_item_ids)
        neg_scores = self.score(user_ids, neg_item_ids)
        return pos_scores, neg_scores

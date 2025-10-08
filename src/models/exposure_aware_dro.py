"""
Exposure-aware Robust Weighting implementation.

Inspired by: "Exposure-aware Distributionally Robust Optimization for Sequential Recommendation"
Yang et al., 2024

This implements exposure-aware robust weighting for handling exposure bias in 
recommendation systems. Note: This is a simplified approximation of the full DRO method,
focusing on the exposure-aware reweighting mechanism without full DRO optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .matrix_factorization import MatrixFactorizationBPR


class ExposureAwareReweighting(MatrixFactorizationBPR):
    """
    Exposure-aware Robust Weighting model.
    
    Extends matrix factorization with exposure-aware reweighting to handle 
    exposure bias. This is a simplified approximation inspired by DRO principles
    but using practical reweighting instead of full minimax optimization.
    
    Args:
        n_users: Number of users
        n_items: Number of items
        k: Embedding dimension
        dro_radius: Temperature parameter for reweighting (not true DRO radius)
        exposure_weight: Weight for exposure bias correction
        update_freq: Frequency to update weights
    """
    
    def __init__(self, n_users, n_items, k=64, dro_radius=0.1, 
                 exposure_weight=0.5, update_freq=10):
        super(ExposureAwareReweighting, self).__init__(n_users, n_items, k)
        
        self.dro_radius = dro_radius
        self.exposure_weight = exposure_weight
        self.update_freq = update_freq
        
        # DRO-specific parameters
        self.register_buffer('sample_weights', torch.ones(1))
        self.register_buffer('exposure_scores', torch.zeros(n_items))
        self.step_count = 0
        
        # Dual variables for DRO optimization
        self.dual_var = nn.Parameter(torch.zeros(1))
        
    def update_exposure_scores(self, train_df):
        """Update item exposure scores based on interaction frequency."""
        item_counts = torch.zeros(self.n_items)
        
        if hasattr(train_df, 'values'):
            items = train_df['i'].values
        else:
            items = train_df[:, 1]  # Assuming second column is item
            
        for item in items:
            item_counts[item] += 1
            
        # Normalize to get exposure probabilities
        total_interactions = len(train_df)
        self.exposure_scores = item_counts / total_interactions
        
    def compute_dro_weights(self, losses, batch_items):
        """
        Compute DRO weights for batch samples based on exposure bias.
        
        Args:
            losses: Per-sample losses
            batch_items: Item indices in current batch
            
        Returns:
            DRO weights for reweighting samples
        """
        # Get exposure scores for batch items
        batch_exposure = self.exposure_scores[batch_items]
        
        # Compute exposure-adjusted losses
        exposure_penalty = self.exposure_weight * batch_exposure
        adjusted_losses = losses + exposure_penalty
        
        # DRO reweighting: emphasize worst-case samples
        with torch.no_grad():
            # Softmax weighting based on adjusted losses
            weights = F.softmax(adjusted_losses / self.dro_radius, dim=0)
            
            # Ensure weights sum to batch size for proper scaling
            weights = weights * len(losses)
            
        return weights
        
    def dro_loss(self, user_emb, pos_emb, neg_emb, pos_items, reg_weight=1e-4):
        """
        Compute DRO-regularized BPR loss.
        
        Args:
            user_emb: User embeddings
            pos_emb: Positive item embeddings  
            neg_emb: Negative item embeddings
            pos_items: Positive item indices
            reg_weight: Regularization weight
            
        Returns:
            DRO-weighted loss
        """
        # Standard BPR scores
        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=1)
        
        # Per-sample BPR losses
        bpr_losses = -F.logsigmoid(pos_scores - neg_scores)
        
        # Compute DRO weights
        dro_weights = self.compute_dro_weights(bpr_losses, pos_items)
        
        # Weighted loss
        weighted_loss = torch.mean(dro_weights * bpr_losses)
        
        # Regularization
        reg_loss = reg_weight * (
            torch.norm(user_emb) + torch.norm(pos_emb) + torch.norm(neg_emb)
        )
        
        # DRO dual variable regularization
        dual_reg = self.dro_radius * self.dual_var
        
        total_loss = weighted_loss + reg_loss + dual_reg
        
        return total_loss, weighted_loss, reg_loss
        
    def forward(self, users, pos_items, neg_items):
        """Forward pass with DRO weighting."""
        # Get embeddings
        user_emb = self.user_embeddings(users)
        pos_emb = self.item_embeddings(pos_items)
        neg_emb = self.item_embeddings(neg_items)
        
        return user_emb, pos_emb, neg_emb
        
    def update_dro_parameters(self, current_loss):
        """Update DRO dual variables based on current performance."""
        self.step_count += 1
        
        if self.step_count % self.update_freq == 0:
            # Simple dual variable update
            with torch.no_grad():
                self.dual_var.data = torch.clamp(
                    self.dual_var.data + 0.01 * (current_loss - self.dro_radius),
                    min=0.0
                )


def exposure_dro_loss(user_emb, pos_emb, neg_emb, pos_items, model, reg_weight=1e-4):
    """
    Compute exposure-aware DRO loss.
    
    Args:
        user_emb: User embeddings
        pos_emb: Positive item embeddings
        neg_emb: Negative item embeddings  
        pos_items: Positive item indices
        model: ExposureAwareDRO model instance
        reg_weight: Regularization weight
        
    Returns:
        Total loss, BPR loss, regularization loss
    """
    return model.dro_loss(user_emb, pos_emb, neg_emb, pos_items, reg_weight)

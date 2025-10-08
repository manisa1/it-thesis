"""
Personalized Denoising Implicit Feedback (PDIF) implementation.

Based on: "Personalized Denoising Implicit Feedback for Recommendation"
Zhang et al., 2025

This implements personalized denoising using user-specific thresholds and
interaction histories to filter natural noise like misclicks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import defaultdict
from .matrix_factorization import MatrixFactorizationBPR


class PDIF(MatrixFactorizationBPR):
    """
    Personalized Denoising Implicit Feedback model.
    
    Filters natural noise using personalized thresholds based on user
    interaction histories and personal loss distributions.
    
    Args:
        n_users: Number of users
        n_items: Number of items
        k: Embedding dimension
        noise_threshold: Base threshold for noise detection
        personalization_weight: Weight for personalizing thresholds
        history_window: Window size for user history analysis
        resample_ratio: Ratio of interactions to resample
    """
    
    def __init__(self, n_users, n_items, k=64, noise_threshold=0.5,
                 personalization_weight=0.3, history_window=50, resample_ratio=0.2):
        super(PDIF, self).__init__(n_users, n_items, k)
        
        self.noise_threshold = noise_threshold
        self.personalization_weight = personalization_weight
        self.history_window = history_window
        self.resample_ratio = resample_ratio
        
        # User-specific parameters
        self.register_buffer('user_thresholds', torch.ones(n_users) * noise_threshold)
        self.register_buffer('user_loss_history', torch.zeros(n_users, history_window))
        self.register_buffer('history_pointer', torch.zeros(n_users, dtype=torch.long))
        
        # Interaction reliability scores
        self.interaction_scores = {}
        
    def update_user_history(self, users, losses):
        """Update user loss history for personalized threshold computation."""
        for i, user in enumerate(users):
            user = user.item()
            ptr = self.history_pointer[user].item()
            
            # Store loss in circular buffer
            self.user_loss_history[user, ptr] = losses[i].item()
            self.history_pointer[user] = (ptr + 1) % self.history_window
            
    def compute_personalized_thresholds(self):
        """Compute personalized noise thresholds for each user."""
        for user in range(self.n_users):
            # Get user's loss history
            user_losses = self.user_loss_history[user]
            valid_losses = user_losses[user_losses > 0]  # Only non-zero losses
            
            if len(valid_losses) > 5:  # Need minimum history
                # Use percentile-based threshold
                threshold = torch.quantile(valid_losses, 0.7)  # 70th percentile
                
                # Blend with global threshold
                personal_threshold = (
                    self.personalization_weight * threshold + 
                    (1 - self.personalization_weight) * self.noise_threshold
                )
                
                self.user_thresholds[user] = personal_threshold
                
    def identify_noisy_interactions(self, train_df):
        """
        Identify potentially noisy interactions based on user patterns.
        
        Args:
            train_df: Training dataframe with user-item interactions
            
        Returns:
            Dictionary mapping (user, item) to noise probability
        """
        noise_scores = {}
        
        # Group interactions by user
        if hasattr(train_df, 'groupby'):
            user_groups = train_df.groupby('u')
        else:
            # Convert to DataFrame if needed
            df = pd.DataFrame(train_df, columns=['u', 'i'])
            user_groups = df.groupby('u')
        
        for user, group in user_groups:
            user_items = group['i'].values if hasattr(group, 'values') else group[:, 1]
            
            # Compute item frequency for this user
            item_counts = defaultdict(int)
            for item in user_items:
                item_counts[item] += 1
                
            total_interactions = len(user_items)
            
            # Score interactions based on frequency and patterns
            for item in user_items:
                # Frequency-based scoring (rare items more likely to be noise)
                freq_score = item_counts[item] / total_interactions
                
                # Pattern-based scoring (isolated interactions more suspicious)
                isolation_score = 1.0 / (1.0 + item_counts[item])
                
                # Combined noise probability
                noise_prob = 0.6 * isolation_score + 0.4 * (1 - freq_score)
                noise_scores[(user, item)] = noise_prob
                
        return noise_scores
        
    def resample_training_data(self, train_df):
        """
        Resample training data by filtering likely noisy interactions.
        
        Args:
            train_df: Original training dataframe
            
        Returns:
            Filtered training dataframe
        """
        # Identify noisy interactions
        noise_scores = self.identify_noisy_interactions(train_df)
        
        # Filter interactions
        filtered_data = []
        
        if hasattr(train_df, 'iterrows'):
            for idx, row in train_df.iterrows():
                user, item = row['u'], row['i']
                noise_prob = noise_scores.get((user, item), 0.0)
                user_threshold = self.user_thresholds[user].item()
                
                # Keep interaction if noise probability below threshold
                if noise_prob < user_threshold:
                    filtered_data.append(row.to_dict())
                elif np.random.random() > self.resample_ratio:
                    # Randomly keep some "noisy" interactions
                    filtered_data.append(row.to_dict())
        else:
            # Handle numpy array format
            for i in range(len(train_df)):
                user, item = train_df[i, 0], train_df[i, 1]
                noise_prob = noise_scores.get((user, item), 0.0)
                user_threshold = self.user_thresholds[user].item()
                
                if noise_prob < user_threshold or np.random.random() > self.resample_ratio:
                    filtered_data.append({'u': user, 'i': item})
        
        # Convert back to DataFrame
        if filtered_data:
            return pd.DataFrame(filtered_data)
        else:
            return train_df  # Return original if filtering too aggressive
            
    def forward(self, users, pos_items, neg_items):
        """Forward pass (same as base MatrixFactorizationBPR)."""
        user_emb = self.user_embeddings(users)
        pos_emb = self.item_embeddings(pos_items)
        neg_emb = self.item_embeddings(neg_items)
        
        return user_emb, pos_emb, neg_emb
        
    def train_step(self, users, pos_items, neg_items, reg_weight=1e-4):
        """
        Training step with personalized denoising.
        
        Args:
            users: User indices
            pos_items: Positive item indices
            neg_items: Negative item indices
            reg_weight: Regularization weight
            
        Returns:
            Loss value
        """
        # Forward pass
        user_emb, pos_emb, neg_emb = self.forward(users, pos_items, neg_items)
        
        # Compute per-sample losses
        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=1)
        bpr_losses = -F.logsigmoid(pos_scores - neg_scores)
        
        # Update user loss history
        self.update_user_history(users, bpr_losses)
        
        # Compute total loss
        bpr_loss = torch.mean(bpr_losses)
        reg_loss = reg_weight * (
            torch.norm(user_emb) + torch.norm(pos_emb) + torch.norm(neg_emb)
        )
        
        total_loss = bpr_loss + reg_loss
        
        return total_loss, bpr_loss, reg_loss


def pdif_loss(user_emb, pos_emb, neg_emb, users, model, reg_weight=1e-4):
    """
    Compute PDIF loss with personalized denoising.
    
    Args:
        user_emb: User embeddings
        pos_emb: Positive item embeddings
        neg_emb: Negative item embeddings
        users: User indices for updating history
        model: PDIF model instance
        reg_weight: Regularization weight
        
    Returns:
        Total loss, BPR loss, regularization loss
    """
    # Compute BPR loss
    pos_scores = torch.sum(user_emb * pos_emb, dim=1)
    neg_scores = torch.sum(user_emb * neg_emb, dim=1)
    bpr_losses = -F.logsigmoid(pos_scores - neg_scores)
    
    # Update user history
    model.update_user_history(users, bpr_losses)
    
    # Compute losses
    bpr_loss = torch.mean(bpr_losses)
    reg_loss = reg_weight * (
        torch.norm(user_emb) + torch.norm(pos_emb) + torch.norm(neg_emb)
    )
    
    return bpr_loss + reg_loss, bpr_loss, reg_loss

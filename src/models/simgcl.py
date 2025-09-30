"""
SimGCL implementation for baseline comparison.

Based on: "Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation"
Yu et al., SIGIR 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .lightgcn import LightGCN, create_adj_matrix


class SimGCL(LightGCN):
    """
    SimGCL model implementation.
    Extends LightGCN with contrastive learning using uniform noise.
    
    Args:
        n_users: Number of users
        n_items: Number of items
        embedding_dim: Embedding dimension
        n_layers: Number of GCN layers
        dropout: Dropout rate
        cl_rate: Contrastive learning rate
        eps: Noise scale for contrastive learning
        temperature: Temperature for contrastive loss
    """
    
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=3, 
                 dropout=0.1, cl_rate=1e-6, eps=0.1, temperature=0.2):
        super(SimGCL, self).__init__(n_users, n_items, embedding_dim, n_layers, dropout)
        
        self.cl_rate = cl_rate
        self.eps = eps
        self.temperature = temperature
        
    def computer_with_noise(self):
        """
        Propagate embeddings through graph convolution with uniform noise.
        """
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb])
        
        embs = [all_emb]
        
        if self.Graph is None:
            raise ValueError("Graph adjacency matrix not set. Call set_graph() first.")
            
        for layer in range(self.n_layers):
            # Add uniform noise for contrastive learning
            if self.training:
                random_noise = torch.rand_like(all_emb).cuda() if all_emb.is_cuda else torch.rand_like(all_emb)
                all_emb = all_emb + torch.sign(all_emb) * F.normalize(random_noise, dim=-1) * self.eps
                
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            embs.append(all_emb)
            
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        
        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items
    
    def forward(self, users, pos_items, neg_items):
        """
        Forward pass for BPR loss computation with contrastive learning.
        
        Args:
            users: User indices
            pos_items: Positive item indices
            neg_items: Negative item indices
            
        Returns:
            user_emb, pos_emb, neg_emb: Embeddings for loss computation
            cl_loss: Contrastive learning loss
        """
        # Get clean embeddings
        all_users_clean, all_items_clean = self.computer()
        
        # Get noisy embeddings for contrastive learning
        all_users_noisy, all_items_noisy = self.computer_with_noise()
        
        users_emb_clean = all_users_clean[users]
        pos_emb_clean = all_items_clean[pos_items]
        neg_emb_clean = all_items_clean[neg_items]
        
        # Apply dropout during training
        if self.training:
            users_emb_clean = F.dropout(users_emb_clean, self.dropout)
            pos_emb_clean = F.dropout(pos_emb_clean, self.dropout)
            neg_emb_clean = F.dropout(neg_emb_clean, self.dropout)
        
        # Contrastive learning loss
        cl_loss = 0.0
        if self.training:
            users_emb_noisy = all_users_noisy[users]
            pos_emb_noisy = all_items_noisy[pos_items]
            
            # User contrastive loss
            user_cl_loss = self.contrastive_loss(users_emb_clean, users_emb_noisy)
            # Item contrastive loss  
            item_cl_loss = self.contrastive_loss(pos_emb_clean, pos_emb_noisy)
            
            cl_loss = self.cl_rate * (user_cl_loss + item_cl_loss)
            
        return users_emb_clean, pos_emb_clean, neg_emb_clean, cl_loss
    
    def contrastive_loss(self, emb1, emb2):
        """
        Compute contrastive loss between two embeddings.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Contrastive loss value
        """
        # Normalize embeddings
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(emb1, emb2.t()) / self.temperature
        
        # Create positive mask (diagonal elements)
        batch_size = emb1.size(0)
        pos_mask = torch.eye(batch_size, device=emb1.device, dtype=torch.bool)
        
        # Compute contrastive loss
        pos_sim = sim_matrix[pos_mask]
        neg_sim = sim_matrix[~pos_mask].view(batch_size, -1)
        
        # LogSumExp for numerical stability
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, device=emb1.device, dtype=torch.long)
        
        cl_loss = F.cross_entropy(logits, labels)
        return cl_loss


def simgcl_loss(user_emb, pos_emb, neg_emb, cl_loss, reg_weight=1e-4):
    """
    Compute combined BPR + contrastive loss for SimGCL.
    
    Args:
        user_emb: User embeddings
        pos_emb: Positive item embeddings
        neg_emb: Negative item embeddings
        cl_loss: Contrastive learning loss
        reg_weight: Regularization weight
        
    Returns:
        Combined loss value
    """
    # BPR loss
    pos_scores = torch.sum(user_emb * pos_emb, dim=1)
    neg_scores = torch.sum(user_emb * neg_emb, dim=1)
    
    bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
    
    # L2 regularization
    reg_loss = reg_weight * (torch.norm(user_emb) + torch.norm(pos_emb) + torch.norm(neg_emb))
    
    # Combined loss
    total_loss = bpr_loss + reg_loss + cl_loss
    
    return total_loss, bpr_loss, cl_loss

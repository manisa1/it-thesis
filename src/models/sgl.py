"""
SGL implementation for baseline comparison.

Based on: "Self-supervised Graph Learning for Recommendation"
Wu et al., SIGIR 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .lightgcn import LightGCN, create_adj_matrix


class SGL(LightGCN):
    """
    Self-supervised Graph Learning (SGL) model implementation.
    Extends LightGCN with self-supervised contrastive learning using graph augmentation.
    
    Args:
        n_users: Number of users
        n_items: Number of items
        embedding_dim: Embedding dimension
        n_layers: Number of GCN layers
        dropout: Dropout rate
        ssl_rate: Self-supervised learning rate
        ssl_temp: Temperature for contrastive loss
        ssl_reg: SSL regularization weight
        aug_type: Augmentation type ('nd' for node dropout, 'ed' for edge dropout, 'rw' for random walk)
    """
    
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=3, 
                 dropout=0.1, ssl_rate=0.1, ssl_temp=0.2, ssl_reg=1e-6, aug_type='nd'):
        super(SGL, self).__init__(n_users, n_items, embedding_dim, n_layers, dropout)
        
        self.ssl_rate = ssl_rate
        self.ssl_temp = ssl_temp
        self.ssl_reg = ssl_reg
        self.aug_type = aug_type
        
        # Augmented graphs (will be set externally)
        self.Graph_1 = None
        self.Graph_2 = None
        
    def set_augmented_graphs(self, graph_1, graph_2):
        """Set the augmented adjacency matrices."""
        self.Graph_1 = graph_1
        self.Graph_2 = graph_2
        
    def computer_aug(self, graph):
        """
        Propagate embeddings through graph convolution with augmented graph.
        
        Args:
            graph: Augmented adjacency matrix
            
        Returns:
            users, items: User and item embeddings
        """
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb])
        
        embs = [all_emb]
        
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(graph, all_emb)
            embs.append(all_emb)
            
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        
        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items
    
    def forward(self, users, pos_items, neg_items):
        """
        Forward pass for BPR loss computation with self-supervised learning.
        
        Args:
            users: User indices
            pos_items: Positive item indices
            neg_items: Negative item indices
            
        Returns:
            user_emb, pos_emb, neg_emb: Embeddings for loss computation
            ssl_loss: Self-supervised learning loss
        """
        # Get clean embeddings
        all_users_clean, all_items_clean = self.computer()
        
        users_emb_clean = all_users_clean[users]
        pos_emb_clean = all_items_clean[pos_items]
        neg_emb_clean = all_items_clean[neg_items]
        
        # Apply dropout during training
        if self.training:
            users_emb_clean = F.dropout(users_emb_clean, self.dropout)
            pos_emb_clean = F.dropout(pos_emb_clean, self.dropout)
            neg_emb_clean = F.dropout(neg_emb_clean, self.dropout)
        
        # Self-supervised learning loss
        ssl_loss = 0.0
        if self.training and self.Graph_1 is not None and self.Graph_2 is not None:
            # Get embeddings from augmented graphs
            all_users_1, all_items_1 = self.computer_aug(self.Graph_1)
            all_users_2, all_items_2 = self.computer_aug(self.Graph_2)
            
            # Sample users and items for contrastive learning
            ssl_users = users
            ssl_items = pos_items
            
            # User contrastive loss
            user_emb_1 = all_users_1[ssl_users]
            user_emb_2 = all_users_2[ssl_users]
            user_ssl_loss = self.ssl_loss(user_emb_1, user_emb_2)
            
            # Item contrastive loss
            item_emb_1 = all_items_1[ssl_items]
            item_emb_2 = all_items_2[ssl_items]
            item_ssl_loss = self.ssl_loss(item_emb_1, item_emb_2)
            
            ssl_loss = self.ssl_rate * (user_ssl_loss + item_ssl_loss)
            
        return users_emb_clean, pos_emb_clean, neg_emb_clean, ssl_loss
    
    def ssl_loss(self, emb_1, emb_2):
        """
        Compute self-supervised contrastive loss.
        
        Args:
            emb_1: Embeddings from first augmented graph
            emb_2: Embeddings from second augmented graph
            
        Returns:
            SSL loss value
        """
        # Normalize embeddings
        emb_1 = F.normalize(emb_1, dim=1)
        emb_2 = F.normalize(emb_2, dim=1)
        
        # Compute similarity
        pos_score = torch.sum(emb_1 * emb_2, dim=1) / self.ssl_temp
        
        # Negative sampling within batch
        batch_size = emb_1.size(0)
        neg_score = torch.matmul(emb_1, emb_2.t()) / self.ssl_temp
        
        # Create mask for positive pairs
        pos_mask = torch.eye(batch_size, device=emb_1.device, dtype=torch.bool)
        neg_mask = ~pos_mask
        
        # Compute InfoNCE loss
        pos_score = pos_score.unsqueeze(1)
        neg_score = neg_score[neg_mask].view(batch_size, -1)
        
        logits = torch.cat([pos_score, neg_score], dim=1)
        labels = torch.zeros(batch_size, device=emb_1.device, dtype=torch.long)
        
        ssl_loss = F.cross_entropy(logits, labels)
        return ssl_loss


def create_augmented_graph(adj_matrix, aug_type='nd', drop_rate=0.1):
    """
    Create augmented adjacency matrix for SGL.
    
    Args:
        adj_matrix: Original adjacency matrix
        aug_type: Augmentation type ('nd', 'ed', 'rw')
        drop_rate: Dropout rate for augmentation
        
    Returns:
        Augmented adjacency matrix
    """
    if aug_type == 'nd':  # Node dropout
        return node_dropout(adj_matrix, drop_rate)
    elif aug_type == 'ed':  # Edge dropout
        return edge_dropout(adj_matrix, drop_rate)
    elif aug_type == 'rw':  # Random walk
        return random_walk_augment(adj_matrix, drop_rate)
    else:
        return adj_matrix


def node_dropout(adj_matrix, drop_rate):
    """Apply node dropout augmentation."""
    n_nodes = adj_matrix.size(0)
    keep_mask = torch.rand(n_nodes) > drop_rate
    
    # Convert to dense for masking, then back to sparse
    adj_dense = adj_matrix.to_dense()
    adj_dense = adj_dense * keep_mask.unsqueeze(1) * keep_mask.unsqueeze(0)
    
    # Convert back to sparse
    adj_sparse = adj_dense.to_sparse()
    return adj_sparse


def edge_dropout(adj_matrix, drop_rate):
    """Apply edge dropout augmentation."""
    indices = adj_matrix.indices()
    values = adj_matrix.values()
    
    # Randomly keep edges
    keep_mask = torch.rand(len(values)) > drop_rate
    new_indices = indices[:, keep_mask]
    new_values = values[keep_mask]
    
    # Create new sparse tensor
    aug_adj = torch.sparse.FloatTensor(new_indices, new_values, adj_matrix.size())
    return aug_adj


def random_walk_augment(adj_matrix, walk_length=2):
    """Apply random walk augmentation."""
    # Simple implementation: multiply adjacency matrix with itself
    adj_dense = adj_matrix.to_dense()
    aug_adj_dense = torch.matmul(adj_dense, adj_dense)
    
    # Normalize
    rowsum = torch.sum(aug_adj_dense, dim=1, keepdim=True)
    aug_adj_dense = aug_adj_dense / (rowsum + 1e-8)
    
    # Convert back to sparse
    aug_adj = aug_adj_dense.to_sparse()
    return aug_adj


def sgl_loss(user_emb, pos_emb, neg_emb, ssl_loss, reg_weight=1e-4):
    """
    Compute combined BPR + SSL loss for SGL.
    
    Args:
        user_emb: User embeddings
        pos_emb: Positive item embeddings
        neg_emb: Negative item embeddings
        ssl_loss: Self-supervised learning loss
        reg_weight: Regularization weight
        
    Returns:
        Combined loss value, BPR loss, SSL loss
    """
    # BPR loss
    pos_scores = torch.sum(user_emb * pos_emb, dim=1)
    neg_scores = torch.sum(user_emb * neg_emb, dim=1)
    
    bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
    
    # L2 regularization
    reg_loss = reg_weight * (torch.norm(user_emb) + torch.norm(pos_emb) + torch.norm(neg_emb))
    
    # Combined loss
    total_loss = bpr_loss + reg_loss + ssl_loss
    
    return total_loss, bpr_loss, ssl_loss

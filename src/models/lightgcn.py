"""
LightGCN implementation for baseline comparison.

Based on: "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
He et al., SIGIR 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np


class LightGCN(nn.Module):
    """
    LightGCN model implementation.
    
    Args:
        n_users: Number of users
        n_items: Number of items  
        embedding_dim: Embedding dimension
        n_layers: Number of GCN layers
        dropout: Dropout rate
    """
    
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=3, dropout=0.1):
        super(LightGCN, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # Graph adjacency matrix (will be set externally)
        self.Graph = None
        
    def set_graph(self, adj_matrix):
        """Set the adjacency matrix for graph convolution."""
        self.Graph = adj_matrix
        
    def computer(self):
        """
        Propagate embeddings through graph convolution layers.
        """
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb])
        
        embs = [all_emb]
        
        if self.Graph is None:
            raise ValueError("Graph adjacency matrix not set. Call set_graph() first.")
            
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            embs.append(all_emb)
            
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        
        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items
    
    def get_user_embedding(self, users):
        """Get user embeddings after graph convolution."""
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        return users_emb
    
    def get_item_embedding(self, items):
        """Get item embeddings after graph convolution."""
        all_users, all_items = self.computer()
        items_emb = all_items[items]
        return items_emb
    
    def forward(self, users, pos_items, neg_items):
        """
        Forward pass for BPR loss computation.
        
        Args:
            users: User indices
            pos_items: Positive item indices  
            neg_items: Negative item indices
            
        Returns:
            user_emb, pos_emb, neg_emb: Embeddings for loss computation
        """
        all_users, all_items = self.computer()
        
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        
        # Apply dropout during training
        if self.training:
            users_emb = F.dropout(users_emb, self.dropout)
            pos_emb = F.dropout(pos_emb, self.dropout)
            neg_emb = F.dropout(neg_emb, self.dropout)
            
        return users_emb, pos_emb, neg_emb
    
    def predict(self, users, items):
        """Predict user-item scores."""
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        scores = torch.sum(users_emb * items_emb, dim=1)
        return scores
    
    def get_all_ratings(self, users):
        """Get ratings for all items for given users."""
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        ratings = torch.matmul(users_emb, all_items.t())
        return ratings


def create_adj_matrix(train_df, n_users, n_items):
    """
    Create normalized adjacency matrix for LightGCN.
    
    Args:
        train_df: Training dataframe with 'u' and 'i' columns
        n_users: Number of users
        n_items: Number of items
        
    Returns:
        Normalized adjacency matrix as sparse tensor
    """
    # Create user-item interaction matrix
    row = train_df['u'].values
    col = train_df['i'].values
    data = np.ones(len(row))
    
    # Create bipartite adjacency matrix
    # [0, R]
    # [R.T, 0]
    adj_matrix = sp.coo_matrix((data, (row, col)), shape=(n_users, n_items))
    
    # Create symmetric adjacency matrix
    adj = sp.coo_matrix((n_users + n_items, n_users + n_items))
    adj[:n_users, n_users:] = adj_matrix
    adj[n_users:, :n_users] = adj_matrix.T
    
    # Normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
    
    # Convert to PyTorch sparse tensor
    indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col))).long()
    values = torch.from_numpy(norm_adj.data).float()
    shape = norm_adj.shape
    
    return torch.sparse.FloatTensor(indices, values, shape)


def bpr_loss(user_emb, pos_emb, neg_emb, reg_weight=1e-4):
    """
    Compute BPR loss for LightGCN.
    
    Args:
        user_emb: User embeddings
        pos_emb: Positive item embeddings
        neg_emb: Negative item embeddings
        reg_weight: Regularization weight
        
    Returns:
        BPR loss value
    """
    pos_scores = torch.sum(user_emb * pos_emb, dim=1)
    neg_scores = torch.sum(user_emb * neg_emb, dim=1)
    
    bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
    
    # L2 regularization
    reg_loss = reg_weight * (torch.norm(user_emb) + torch.norm(pos_emb) + torch.norm(neg_emb))
    
    return bpr_loss + reg_loss

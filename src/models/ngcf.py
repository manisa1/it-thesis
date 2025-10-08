"""
NGCF implementation for baseline comparison.

Based on: "Neural Graph Collaborative Filtering"
Wang et al., SIGIR 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .lightgcn import create_adj_matrix


class NGCF(nn.Module):
    """
    Neural Graph Collaborative Filtering (NGCF) model implementation.
    
    Args:
        n_users: Number of users
        n_items: Number of items
        embedding_dim: Embedding dimension
        n_layers: Number of GCN layers
        layer_dims: List of layer dimensions for each GCN layer
        dropout: Dropout rate
        mess_dropout: Message dropout rate
    """
    
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=3, 
                 layer_dims=None, dropout=0.1, mess_dropout=0.1):
        super(NGCF, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.mess_dropout = mess_dropout
        
        # Default layer dimensions
        if layer_dims is None:
            layer_dims = [embedding_dim] * n_layers
        self.layer_dims = layer_dims
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        input_dim = embedding_dim
        
        for i in range(n_layers):
            output_dim = layer_dims[i]
            self.gcn_layers.append(NGCFLayer(input_dim, output_dim, mess_dropout))
            input_dim = output_dim
            
        # Graph adjacency matrix (will be set externally)
        self.Graph = None
        
    def set_graph(self, adj_matrix):
        """Set the adjacency matrix for graph convolution."""
        self.Graph = adj_matrix
        
    def computer(self):
        """
        Propagate embeddings through NGCF layers.
        """
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb])
        
        embs = [all_emb]
        
        if self.Graph is None:
            raise ValueError("Graph adjacency matrix not set. Call set_graph() first.")
            
        for layer in self.gcn_layers:
            all_emb = layer(all_emb, self.Graph)
            embs.append(all_emb)
            
        # Concatenate all layer embeddings
        embs = torch.cat(embs, dim=1)
        
        users, items = torch.split(embs, [self.n_users, self.n_items])
        return users, items
    
    def get_embeddings(self):
        """Get all user and item embeddings for evaluation."""
        return self.computer()
    
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


class NGCFLayer(nn.Module):
    """
    Single NGCF layer implementation.
    
    Args:
        input_dim: Input embedding dimension
        output_dim: Output embedding dimension
        mess_dropout: Message dropout rate
    """
    
    def __init__(self, input_dim, output_dim, mess_dropout=0.1):
        super(NGCFLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mess_dropout = mess_dropout
        
        # Transformation matrices
        self.W1 = nn.Linear(input_dim, output_dim, bias=True)
        self.W2 = nn.Linear(input_dim, output_dim, bias=True)
        
        # Activation function
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        
    def forward(self, embeddings, adj_matrix):
        """
        Forward pass of NGCF layer.
        
        Args:
            embeddings: Input embeddings
            adj_matrix: Adjacency matrix
            
        Returns:
            Updated embeddings
        """
        # Self-connection
        self_emb = self.W1(embeddings)
        
        # Neighbor aggregation
        neighbor_emb = torch.sparse.mm(adj_matrix, embeddings)
        neighbor_emb = self.W2(neighbor_emb)
        
        # Element-wise product for interaction modeling
        interact_emb = neighbor_emb * embeddings
        interact_emb = self.W2(interact_emb)
        
        # Combine self-connection and neighbor information
        output_emb = self_emb + neighbor_emb + interact_emb
        
        # Apply activation and dropout
        output_emb = self.activation(output_emb)
        
        if self.training:
            output_emb = F.dropout(output_emb, self.mess_dropout)
            
        return output_emb


def ngcf_loss(user_emb, pos_emb, neg_emb, reg_weight=1e-4):
    """
    Compute BPR loss for NGCF.
    
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

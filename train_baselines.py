#!/usr/bin/env python3
"""
Baseline model training script for DCCF robustness comparison.

This script trains multiple baseline models (LightGCN, SimGCL, NGCF, SGL) 
under the same noise conditions as DCCF to provide fair comparison.
"""

import os
import sys
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

from models import LightGCN, SimGCL, NGCF, SGL, ExposureAwareReweighting, PDIF, create_adj_matrix, create_augmented_graph
from models.lightgcn import bpr_loss
from models.simgcl import simgcl_loss
from models.ngcf import ngcf_loss
from models.sgl import sgl_loss
from models.exposure_aware_dro import exposure_dro_loss
from models.pdif import pdif_loss
from training.noise import add_dynamic_exposure_noise, noise_scale_for_epoch, focus_for_epoch
from evaluation.metrics import evaluate_model

# Simple logging setup
import logging
def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train baseline models for comparison')
    
    # Data and output
    parser.add_argument('--data_path', type=str, default='data/ratings.csv',
                       help='Path to the ratings dataset')
    parser.add_argument('--model_dir', type=str, default='runs/baseline_comparison',
                       help='Directory to save model and results')
    parser.add_argument('--model_type', type=str, default='lightgcn',
                       choices=['lightgcn', 'simgcl', 'ngcf', 'sgl', 'exposure_dro', 'pdif'],
                       help='Model type to train')
    
    # Model parameters
    parser.add_argument('--embedding_dim', type=int, default=64,
                       help='Embedding dimension')
    parser.add_argument('--n_layers', type=int, default=3,
                       help='Number of GCN layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2048,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--reg_weight', type=float, default=1e-4,
                       help='Regularization weight')
    
    # Evaluation parameters
    parser.add_argument('--k_eval', type=int, default=20,
                       help='Top-K for evaluation metrics')
    
    # Noise parameters (same as DCCF experiments)
    parser.add_argument('--noise_exposure_bias', type=float, default=0.0,
                       help='Base exposure bias noise level (0.0-1.0)')
    parser.add_argument('--noise_schedule', type=str, default='none',
                       choices=['none', 'ramp', 'burst', 'shift'],
                       help='Noise schedule type')
    parser.add_argument('--noise_schedule_epochs', type=int, default=10,
                       help='Epochs for ramp schedule')
    
    # Burst noise parameters
    parser.add_argument('--noise_burst_start', type=int, default=4,
                       help='Epoch to start burst (1-based)')
    parser.add_argument('--noise_burst_len', type=int, default=2,
                       help='Duration of burst in epochs')
    parser.add_argument('--noise_burst_scale', type=float, default=2.0,
                       help='Noise multiplier during burst')
    
    # Shift noise parameters
    parser.add_argument('--noise_shift_epoch', type=int, default=5,
                       help='Epoch where focus shifts (1-based)')
    parser.add_argument('--noise_shift_mode', type=str, default='head2tail',
                       choices=['head2tail', 'tail2head'],
                       help='Shift direction')
    
    # Model-specific parameters
    parser.add_argument('--cl_rate', type=float, default=1e-6,
                       help='Contrastive learning rate (SimGCL)')
    parser.add_argument('--ssl_rate', type=float, default=0.1,
                       help='Self-supervised learning rate (SGL)')
    parser.add_argument('--ssl_temp', type=float, default=0.2,
                       help='Temperature for contrastive loss')
    parser.add_argument('--aug_type', type=str, default='nd',
                       choices=['nd', 'ed', 'rw'],
                       help='Augmentation type for SGL')
    
    return parser.parse_args()


def load_data(data_path):
    """Load and preprocess data."""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Ensure required columns
    if 'userId' not in df.columns or 'itemId' not in df.columns:
        raise ValueError("Dataset must have 'userId' and 'itemId' columns")
    
    # Map to continuous indices
    user_map = {u: i for i, u in enumerate(df['userId'].unique())}
    item_map = {i: j for j, i in enumerate(df['itemId'].unique())}
    
    df['u'] = df['userId'].map(user_map)
    df['i'] = df['itemId'].map(item_map)
    
    n_users = len(user_map)
    n_items = len(item_map)
    
    print(f"Dataset: {len(df)} interactions, {n_users} users, {n_items} items")
    return df, n_users, n_items


def create_model(model_type, n_users, n_items, args):
    """Create model based on type."""
    if model_type == 'lightgcn':
        model = LightGCN(n_users, n_items, args.embedding_dim, args.n_layers, args.dropout)
    elif model_type == 'simgcl':
        model = SimGCL(n_users, n_items, args.embedding_dim, args.n_layers, 
                      args.dropout, args.cl_rate, temperature=args.ssl_temp)
    elif model_type == 'ngcf':
        model = NGCF(n_users, n_items, args.embedding_dim, args.n_layers, 
                    dropout=args.dropout)
    elif model_type == 'sgl':
        model = SGL(n_users, n_items, args.embedding_dim, args.n_layers,
                   args.dropout, args.ssl_rate, args.ssl_temp, aug_type=args.aug_type)
    elif model_type == 'exposure_dro':
        model = ExposureAwareReweighting(n_users, n_items, args.embedding_dim)
    elif model_type == 'pdif':
        model = PDIF(n_users, n_items, args.embedding_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def sample_batch(train_df, n_items, batch_size):
    """Sample a batch of user-item pairs for training."""
    # Sample positive interactions
    batch_indices = np.random.choice(len(train_df), batch_size, replace=True)
    batch_df = train_df.iloc[batch_indices]
    
    users = torch.LongTensor(batch_df['u'].values)
    pos_items = torch.LongTensor(batch_df['i'].values)
    
    # Sample negative items
    neg_items = torch.LongTensor(np.random.choice(n_items, batch_size))
    
    return users, pos_items, neg_items


def train_epoch(model, train_df, n_items, optimizer, args, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = len(train_df) // args.batch_size + 1
    
    for batch_idx in range(n_batches):
        users, pos_items, neg_items = sample_batch(train_df, n_items, args.batch_size)
        
        if torch.cuda.is_available():
            users = users.cuda()
            pos_items = pos_items.cuda()
            neg_items = neg_items.cuda()
        
        optimizer.zero_grad()
        
        # Forward pass
        if args.model_type == 'simgcl':
            user_emb, pos_emb, neg_emb, cl_loss = model(users, pos_items, neg_items)
            loss, bpr_loss_val, cl_loss_val = simgcl_loss(user_emb, pos_emb, neg_emb, cl_loss, args.reg_weight)
        elif args.model_type == 'sgl':
            user_emb, pos_emb, neg_emb, ssl_loss = model(users, pos_items, neg_items)
            loss, bpr_loss_val, ssl_loss_val = sgl_loss(user_emb, pos_emb, neg_emb, ssl_loss, args.reg_weight)
        elif args.model_type == 'exposure_dro':
            user_emb, pos_emb, neg_emb = model(users, pos_items, neg_items)
            loss, bpr_loss_val, reg_loss_val = exposure_dro_loss(user_emb, pos_emb, neg_emb, pos_items, model, args.reg_weight)
        elif args.model_type == 'pdif':
            user_emb, pos_emb, neg_emb = model(users, pos_items, neg_items)
            loss, bpr_loss_val, reg_loss_val = pdif_loss(user_emb, pos_emb, neg_emb, users, model, args.reg_weight)
        else:  # lightgcn, ngcf
            user_emb, pos_emb, neg_emb = model(users, pos_items, neg_items)
            if args.model_type == 'ngcf':
                loss = ngcf_loss(user_emb, pos_emb, neg_emb, args.reg_weight)
            else:  # lightgcn
                loss = bpr_loss(user_emb, pos_emb, neg_emb, args.reg_weight)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / n_batches


def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(os.path.join(args.model_dir, 'train.log'))
    logger.info(f"Training {args.model_type} model")
    logger.info(f"Arguments: {vars(args)}")
    
    # Load data
    train_df, n_users, n_items = load_data(args.data_path)
    
    # Create model
    model = create_model(args.model_type, n_users, n_items, args)
    
    # Create adjacency matrix for graph-based models only
    if args.model_type in ['lightgcn', 'simgcl', 'ngcf', 'sgl']:
        adj_matrix = create_adj_matrix(train_df, n_users, n_items)
        if torch.cuda.is_available():
            adj_matrix = adj_matrix.cuda()
        model.set_graph(adj_matrix)
        
        # Create augmented graphs for SGL
        if args.model_type == 'sgl':
            aug_graph_1 = create_augmented_graph(adj_matrix, args.aug_type, 0.1)
            aug_graph_2 = create_augmented_graph(adj_matrix, args.aug_type, 0.1)
            model.set_augmented_graphs(aug_graph_1, aug_graph_2)
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop with noise simulation
    best_recall = 0.0
    results = []
    
    for epoch in range(1, args.epochs + 1):
        # Apply noise based on schedule (same as DCCF experiments)
        current_train_df = train_df.copy()
        
        if args.noise_exposure_bias > 0:
            # Calculate noise parameters for current epoch
            noise_scale = noise_scale_for_epoch(
                epoch, args.noise_schedule, args.noise_exposure_bias,
                args.noise_schedule_epochs, args.noise_burst_start,
                args.noise_burst_len, args.noise_burst_scale
            )
            
            focus = focus_for_epoch(
                epoch, args.noise_schedule, args.noise_shift_epoch, args.noise_shift_mode
            )
            
            if noise_scale > 0:
                current_train_df = add_dynamic_exposure_noise(
                    current_train_df, n_users, n_items, noise_scale, focus
                )
                logger.info(f"Epoch {epoch}: Applied noise (scale={noise_scale:.3f}, focus={focus})")
        
        # Train for one epoch
        avg_loss = train_epoch(model, current_train_df, n_items, optimizer, args, epoch)
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            recall, ndcg = evaluate_model(model, train_df, args.k_eval)
        
        logger.info(f"Epoch {epoch:2d}: Loss={avg_loss:.4f}, Recall@{args.k_eval}={recall:.4f}, NDCG@{args.k_eval}={ndcg:.4f}")
        
        # Save results
        results.append({
            'epoch': epoch,
            'loss': avg_loss,
            f'recall@{args.k_eval}': recall,
            f'ndcg@{args.k_eval}': ndcg
        })
        
        # Save best model
        if recall > best_recall:
            best_recall = recall
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'best.pt'))
    
    # Save final results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.model_dir, 'metrics.csv'), index=False)
    
    # Save final metrics
    final_metrics = {
        'model_type': args.model_type,
        'final_recall': results[-1][f'recall@{args.k_eval}'],
        'final_ndcg': results[-1][f'ndcg@{args.k_eval}'],
        'best_recall': best_recall,
        'noise_schedule': args.noise_schedule,
        'noise_level': args.noise_exposure_bias
    }
    
    logger.info(f"Training completed. Best Recall@{args.k_eval}: {best_recall:.4f}")
    logger.info(f"Final metrics: {final_metrics}")
    
    return final_metrics


if __name__ == "__main__":
    main()

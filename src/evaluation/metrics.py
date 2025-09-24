"""
Evaluation metrics for recommendation systems.

This module implements standard recommendation metrics including Recall@K and NDCG@K
for evaluating the performance of the DCCF robustness experiments.
"""

from typing import List, Dict, Tuple
import math
import numpy as np
import torch
from ..models.matrix_factorization import MatrixFactorizationBPR
from ..data.dataset import RecommenderDataset


class RecommendationMetrics:
    """
    Computes recommendation evaluation metrics.
    
    This class provides methods for computing Recall@K, NDCG@K, and other
    recommendation metrics used in the thesis experiments.
    """
    
    def __init__(self, k: int = 20):
        """
        Initialize the metrics calculator.
        
        Args:
            k (int): Top-K value for metrics computation
        """
        self.k = k
    
    def evaluate_model(self, 
                      model: MatrixFactorizationBPR,
                      test_dataset: RecommenderDataset,
                      train_dataset: RecommenderDataset,
                      exclude_train: bool = False) -> Dict[str, float]:
        """
        Evaluate a model on test data.
        
        Args:
            model (MatrixFactorizationBPR): Trained model to evaluate
            test_dataset (RecommenderDataset): Test dataset
            train_dataset (RecommenderDataset): Training dataset (for exclusion)
            exclude_train (bool): Whether to exclude training items from recommendations
            
        Returns:
            Dict[str, float]: Dictionary with evaluation metrics
        """
        model.eval()
        
        # Get full score matrix
        with torch.no_grad():
            scores = model.full_scores().detach().cpu().numpy()
        
        # Prepare test data
        test_user_items = self._prepare_test_data(test_dataset)
        
        if len(test_user_items) == 0:
            return {'recall@k': 0.0, 'ndcg@k': 0.0, 'n_users': 0}
        
        # Compute metrics for each user
        recalls = []
        ndcgs = []
        
        for user_id, true_items in test_user_items.items():
            if user_id >= scores.shape[0]:  # Skip if user not in training
                continue
            
            # Get user's scores
            user_scores = scores[user_id]
            
            # Exclude training items if requested
            if exclude_train:
                train_items = train_dataset.get_user_positive_items(user_id)
                for item_id in train_items:
                    if item_id < len(user_scores):
                        user_scores[item_id] = -np.inf
            
            # Get top-K recommendations
            top_k_items = np.argsort(-user_scores)[:self.k].tolist()
            
            # Compute metrics
            recall = self.recall_at_k(top_k_items, true_items, self.k)
            ndcg = self.ndcg_at_k(top_k_items, true_items, self.k)
            
            recalls.append(recall)
            ndcgs.append(ndcg)
        
        # Return average metrics
        return {
            'recall@k': float(np.mean(recalls)) if recalls else 0.0,
            'ndcg@k': float(np.mean(ndcgs)) if ndcgs else 0.0,
            'n_users': len(recalls),
            'k': self.k
        }
    
    def _prepare_test_data(self, test_dataset: RecommenderDataset) -> Dict[int, List[int]]:
        """
        Prepare test data for evaluation.
        
        Args:
            test_dataset (RecommenderDataset): Test dataset
            
        Returns:
            Dict[int, List[int]]: User ID -> List of true positive items
        """
        user_items = {}
        for _, row in test_dataset.df.iterrows():
            user_id = int(row['u'])
            item_id = int(row['i'])
            
            if user_id not in user_items:
                user_items[user_id] = []
            user_items[user_id].append(item_id)
        
        return user_items
    
    @staticmethod
    def recall_at_k(ranked_items: List[int], 
                   true_items: List[int], 
                   k: int) -> float:
        """
        Compute Recall@K metric.
        
        Args:
            ranked_items (List[int]): List of recommended items (ranked)
            true_items (List[int]): List of true positive items
            k (int): Top-K value
            
        Returns:
            float: Recall@K score
        """
        if not true_items:
            return 0.0
        
        # Get top-K recommendations
        top_k = ranked_items[:k]
        
        # Count hits
        hits = len(set(top_k) & set(true_items))
        
        # Recall = hits / total_relevant
        return hits / len(true_items)
    
    @staticmethod
    def ndcg_at_k(ranked_items: List[int], 
                  true_items: List[int], 
                  k: int) -> float:
        """
        Compute NDCG@K metric.
        
        Args:
            ranked_items (List[int]): List of recommended items (ranked)
            true_items (List[int]): List of true positive items
            k (int): Top-K value
            
        Returns:
            float: NDCG@K score
        """
        if not true_items:
            return 0.0
        
        # Get top-K recommendations
        top_k = ranked_items[:k]
        true_items_set = set(true_items)
        
        # Compute DCG
        dcg = 0.0
        for i, item in enumerate(top_k):
            if item in true_items_set:
                dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Compute IDCG (Ideal DCG)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(true_items))))
        
        # Return NDCG
        return dcg / idcg if idcg > 0 else 0.0
    
    def compute_robustness_drop(self, 
                               clean_metrics: Dict[str, float],
                               noisy_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Compute robustness drop between clean and noisy conditions.
        
        Args:
            clean_metrics (Dict[str, float]): Metrics under clean conditions
            noisy_metrics (Dict[str, float]): Metrics under noisy conditions
            
        Returns:
            Dict[str, float]: Robustness drop for each metric
        """
        robustness_drop = {}
        
        for metric_name in ['recall@k', 'ndcg@k']:
            clean_value = clean_metrics.get(metric_name, 0.0)
            noisy_value = noisy_metrics.get(metric_name, 0.0)
            
            if clean_value > 0:
                drop = (clean_value - noisy_value) / clean_value
            else:
                drop = 0.0
            
            robustness_drop[f'{metric_name}_drop'] = drop
        
        return robustness_drop
    
    def format_results(self, 
                      results: Dict[str, float], 
                      precision: int = 4) -> str:
        """
        Format evaluation results for display.
        
        Args:
            results (Dict[str, float]): Evaluation results
            precision (int): Number of decimal places
            
        Returns:
            str: Formatted results string
        """
        formatted_lines = []
        
        for key, value in results.items():
            if isinstance(value, float):
                formatted_lines.append(f"{key}: {value:.{precision}f}")
            else:
                formatted_lines.append(f"{key}: {value}")
        
        return "\n".join(formatted_lines)

"""
Established robustness evaluation metrics for recommender systems.

Based on:
1. "Robust Recommender System: A Survey and Future Directions" (2023)
2. "Towards Robust Recommendation: A Review and an Adversarial Robustness Evaluation Library" (2024)

Implements standard robustness metrics used in the literature:
- Offset on Metrics (ΔM)
- Offset on Output (ΔO) with RBO and Jaccard similarity
- Robustness Improvement (RI)
- Predict Shift (PS)
- Drop Rate (DR)
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import jaccard_score


def offset_on_metrics(clean_metrics: Dict[str, float], 
                     noisy_metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate Offset on Metrics (ΔM) - most commonly used robustness metric.
    
    Formula: ΔM = |M' - M| / M
    where M is clean performance, M' is noisy performance
    
    Args:
        clean_metrics: Performance metrics on clean data
        noisy_metrics: Performance metrics on noisy data
        
    Returns:
        Dictionary of offset values for each metric
    """
    offsets = {}
    
    for metric_name in clean_metrics.keys():
        if metric_name in noisy_metrics:
            clean_val = clean_metrics[metric_name]
            noisy_val = noisy_metrics[metric_name]
            
            if clean_val != 0:
                offset = abs(noisy_val - clean_val) / clean_val
            else:
                offset = abs(noisy_val - clean_val)  # Absolute difference when clean is 0
                
            offsets[f"Δ{metric_name}"] = offset
    
    return offsets


def robustness_improvement(clean_metrics: Dict[str, float],
                          attack_metrics: Dict[str, float], 
                          defense_metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate Robustness Improvement (RI) metric.
    
    Formula: RI = (M_defense - M_attack) / (M_clean - M_attack)
    
    Args:
        clean_metrics: Performance on clean data
        attack_metrics: Performance under attack (no defense)
        defense_metrics: Performance under attack with defense
        
    Returns:
        Dictionary of RI values for each metric
    """
    ri_values = {}
    
    for metric_name in clean_metrics.keys():
        if metric_name in attack_metrics and metric_name in defense_metrics:
            clean_val = clean_metrics[metric_name]
            attack_val = attack_metrics[metric_name]
            defense_val = defense_metrics[metric_name]
            
            denominator = clean_val - attack_val
            if abs(denominator) > 1e-8:  # Avoid division by zero
                ri = (defense_val - attack_val) / denominator
            else:
                ri = 0.0
                
            ri_values[f"RI_{metric_name}"] = ri
    
    return ri_values


def predict_shift(clean_predictions: np.ndarray, 
                 noisy_predictions: np.ndarray) -> float:
    """
    Calculate Predict Shift (PS) metric.
    
    Formula: PS = |r̂_ui' - r̂_ui| / r̂_ui
    
    Args:
        clean_predictions: Predictions on clean data
        noisy_predictions: Predictions on noisy data
        
    Returns:
        Average predict shift value
    """
    # Avoid division by zero
    mask = np.abs(clean_predictions) > 1e-8
    
    if np.sum(mask) == 0:
        return np.mean(np.abs(noisy_predictions - clean_predictions))
    
    shifts = np.abs(noisy_predictions[mask] - clean_predictions[mask]) / np.abs(clean_predictions[mask])
    return np.mean(shifts)


def drop_rate(iid_performance: float, non_iid_performance: float) -> float:
    """
    Calculate Drop Rate (DR) metric for distribution shift robustness.
    
    Formula: DR = (P_I - P_N) / P_I
    where P_I is performance on i.i.d. test set, P_N on non-i.i.d. test set
    
    Args:
        iid_performance: Performance on i.i.d. test set
        non_iid_performance: Performance on non-i.i.d. test set
        
    Returns:
        Drop rate value
    """
    if iid_performance != 0:
        return (iid_performance - non_iid_performance) / iid_performance
    else:
        return iid_performance - non_iid_performance


def jaccard_similarity(list1: List[int], list2: List[int], k: Optional[int] = None) -> float:
    """
    Calculate Jaccard similarity between two recommendation lists.
    
    Formula: Jaccard = |A ∩ B| / |A ∪ B|
    
    Args:
        list1: First recommendation list
        list2: Second recommendation list
        k: Consider only top-k items (if None, use full lists)
        
    Returns:
        Jaccard similarity score
    """
    if k is not None:
        list1 = list1[:k]
        list2 = list2[:k]
    
    set1 = set(list1)
    set2 = set(list2)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 1.0  # Both lists are empty
    
    return intersection / union


def rank_biased_overlap(list1: List[int], list2: List[int], p: float = 0.9) -> float:
    """
    Calculate Rank-Biased Overlap (RBO) between two ranked lists.
    
    RBO emphasizes top-ranked items more heavily.
    
    Args:
        list1: First ranked list
        list2: Second ranked list  
        p: Persistence parameter (0 < p < 1, recommended: 0.9)
        
    Returns:
        RBO similarity score
    """
    if len(list1) == 0 and len(list2) == 0:
        return 1.0
    
    if len(list1) == 0 or len(list2) == 0:
        return 0.0
    
    # Convert to sets for intersection calculation
    set1 = set(list1)
    set2 = set(list2)
    
    # Calculate overlap at each depth
    max_depth = min(len(list1), len(list2))
    overlap_sum = 0.0
    
    for d in range(1, max_depth + 1):
        # Items in top-d of each list
        top_d_1 = set(list1[:d])
        top_d_2 = set(list2[:d])
        
        # Overlap at depth d
        overlap_d = len(top_d_1.intersection(top_d_2)) / d
        
        # Weight by persistence parameter
        weight = (1 - p) * (p ** (d - 1))
        overlap_sum += weight * overlap_d
    
    return overlap_sum


def offset_on_output(clean_lists: List[List[int]], 
                    noisy_lists: List[List[int]],
                    similarity_metric: str = 'jaccard',
                    k: Optional[int] = None,
                    p: float = 0.9) -> float:
    """
    Calculate Offset on Output (ΔO) using similarity metrics.
    
    Formula: ΔO = E_u[sim(L̂_u@k, L̂_u'@k)]
    
    Args:
        clean_lists: Recommendation lists from clean model
        noisy_lists: Recommendation lists from noisy model
        similarity_metric: 'jaccard' or 'rbo'
        k: Consider only top-k items
        p: Persistence parameter for RBO
        
    Returns:
        Average similarity between clean and noisy recommendations
    """
    if len(clean_lists) != len(noisy_lists):
        raise ValueError("Clean and noisy lists must have same length")
    
    similarities = []
    
    for clean_list, noisy_list in zip(clean_lists, noisy_lists):
        if similarity_metric == 'jaccard':
            sim = jaccard_similarity(clean_list, noisy_list, k)
        elif similarity_metric == 'rbo':
            if k is not None:
                clean_list = clean_list[:k]
                noisy_list = noisy_list[:k]
            sim = rank_biased_overlap(clean_list, noisy_list, p)
        else:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}")
        
        similarities.append(sim)
    
    return np.mean(similarities)


def top_output_stability(clean_lists: List[List[int]], 
                        noisy_lists: List[List[int]]) -> float:
    """
    Calculate Top Output (TO) stability - focuses on top-1 item.
    
    Formula: TO = E_u[I[top1(L̂_u) == top1(L̂_u')]]
    
    Args:
        clean_lists: Recommendation lists from clean model
        noisy_lists: Recommendation lists from noisy model
        
    Returns:
        Fraction of users where top-1 item remains unchanged
    """
    if len(clean_lists) != len(noisy_lists):
        raise ValueError("Clean and noisy lists must have same length")
    
    stable_count = 0
    
    for clean_list, noisy_list in zip(clean_lists, noisy_lists):
        if len(clean_list) > 0 and len(noisy_list) > 0:
            if clean_list[0] == noisy_list[0]:
                stable_count += 1
        elif len(clean_list) == 0 and len(noisy_list) == 0:
            stable_count += 1  # Both empty, consider stable
    
    return stable_count / len(clean_lists)


def comprehensive_robustness_analysis(clean_results: Dict,
                                    noisy_results: Dict,
                                    clean_recommendations: Optional[List[List[int]]] = None,
                                    noisy_recommendations: Optional[List[List[int]]] = None,
                                    k: int = 20) -> Dict[str, float]:
    """
    Perform comprehensive robustness analysis using established metrics.
    
    Args:
        clean_results: Results from clean data {metric_name: value}
        noisy_results: Results from noisy data {metric_name: value}
        clean_recommendations: Recommendation lists from clean model (optional)
        noisy_recommendations: Recommendation lists from noisy model (optional)
        k: Top-k for evaluation
        
    Returns:
        Dictionary containing all robustness metrics
    """
    analysis = {}
    
    # 1. Offset on Metrics (most common)
    offset_metrics = offset_on_metrics(clean_results, noisy_results)
    analysis.update(offset_metrics)
    
    # 2. Performance drops (intuitive interpretation)
    for metric_name in clean_results.keys():
        if metric_name in noisy_results:
            clean_val = clean_results[metric_name]
            noisy_val = noisy_results[metric_name]
            
            if clean_val != 0:
                drop_pct = (clean_val - noisy_val) / clean_val * 100
            else:
                drop_pct = (clean_val - noisy_val) * 100
                
            analysis[f"{metric_name}_drop_%"] = drop_pct
    
    # 3. Offset on Output (if recommendation lists provided)
    if clean_recommendations is not None and noisy_recommendations is not None:
        # Jaccard similarity
        jaccard_sim = offset_on_output(clean_recommendations, noisy_recommendations, 
                                     'jaccard', k)
        analysis[f'jaccard_similarity@{k}'] = jaccard_sim
        analysis[f'jaccard_offset@{k}'] = 1.0 - jaccard_sim
        
        # RBO similarity
        rbo_sim = offset_on_output(clean_recommendations, noisy_recommendations, 
                                 'rbo', k)
        analysis[f'rbo_similarity@{k}'] = rbo_sim
        analysis[f'rbo_offset@{k}'] = 1.0 - rbo_sim
        
        # Top-1 stability
        top1_stability = top_output_stability(clean_recommendations, noisy_recommendations)
        analysis['top1_stability'] = top1_stability
        analysis['top1_instability'] = 1.0 - top1_stability
    
    return analysis


def generate_robustness_table(results_dict: Dict[str, Dict[str, float]], 
                             output_path: str = None) -> pd.DataFrame:
    """
    Generate a comprehensive robustness comparison table.
    
    Args:
        results_dict: {experiment_name: {metric_name: value}}
        output_path: Path to save CSV file (optional)
        
    Returns:
        DataFrame with robustness analysis
    """
    # Convert to DataFrame
    df = pd.DataFrame(results_dict).T
    
    # Round values for readability
    df = df.round(4)
    
    # Sort by most important metrics
    important_cols = [col for col in df.columns if 'recall' in col.lower() or 'ndcg' in col.lower()]
    other_cols = [col for col in df.columns if col not in important_cols]
    df = df[important_cols + other_cols]
    
    if output_path:
        df.to_csv(output_path)
        print(f"Robustness table saved to: {output_path}")
    
    return df

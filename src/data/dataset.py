"""
Dataset classes for recommendation system training.
"""

from typing import Dict, Set, List
import pandas as pd
from collections import defaultdict


class RecommenderDataset:
    """
    Dataset class for handling user-item interactions.
    
    This class provides utilities for managing user-item interactions,
    creating positive interaction sets, and handling dataset statistics.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the dataset.
        
        Args:
            df (pd.DataFrame): Dataframe with user-item interactions
                              Must contain 'u' (user) and 'i' (item) columns
        """
        if 'u' not in df.columns or 'i' not in df.columns:
            raise ValueError("Dataframe must contain 'u' and 'i' columns")
        
        self.df = df.copy()
        self.n_users = df['u'].nunique()
        self.n_items = df['i'].nunique()
        self.n_interactions = len(df)
        
        # Create user-item positive interaction sets
        self.user_positive_items = self._create_user_positive_sets()
    
    def _create_user_positive_sets(self) -> Dict[int, Set[int]]:
        """
        Create dictionary mapping users to their positive items.
        
        Returns:
            Dict[int, Set[int]]: User ID -> Set of positive item IDs
        """
        user_items = defaultdict(set)
        for _, row in self.df.iterrows():
            user_items[int(row['u'])].add(int(row['i']))
        return dict(user_items)
    
    def get_user_positive_items(self, user_id: int) -> Set[int]:
        """
        Get positive items for a specific user.
        
        Args:
            user_id (int): User ID
            
        Returns:
            Set[int]: Set of positive item IDs for the user
        """
        return self.user_positive_items.get(user_id, set())
    
    def get_all_users(self) -> List[int]:
        """
        Get list of all user IDs.
        
        Returns:
            List[int]: List of all user IDs
        """
        return list(self.user_positive_items.keys())
    
    def get_dataset_stats(self) -> Dict[str, float]:
        """
        Get dataset statistics.
        
        Returns:
            Dict[str, float]: Dictionary with dataset statistics
        """
        density = self.n_interactions / (self.n_users * self.n_items)
        avg_items_per_user = self.n_interactions / self.n_users
        
        return {
            'n_users': self.n_users,
            'n_items': self.n_items,
            'n_interactions': self.n_interactions,
            'density': density,
            'avg_items_per_user': avg_items_per_user
        }
    
    def __len__(self) -> int:
        """Return number of interactions."""
        return self.n_interactions
    
    def __repr__(self) -> str:
        """String representation of the dataset."""
        stats = self.get_dataset_stats()
        return (f"RecommenderDataset(users={stats['n_users']}, "
                f"items={stats['n_items']}, "
                f"interactions={stats['n_interactions']}, "
                f"density={stats['density']:.4f})")

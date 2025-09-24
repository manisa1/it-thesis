"""
Data preprocessing utilities for the DCCF robustness study.

This module handles data loading, user/item ID mapping, and train/validation/test splits.
"""

from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np
from collections import defaultdict


class DataPreprocessor:
    """
    Handles data preprocessing for recommendation system experiments.
    
    This class provides methods for loading data, creating ID mappings,
    and splitting data into train/validation/test sets.
    """
    
    def __init__(self, rating_threshold: float = 4.0):
        """
        Initialize the data preprocessor.
        
        Args:
            rating_threshold (float): Minimum rating to consider as positive feedback
        """
        self.rating_threshold = rating_threshold
        self.user_id_map: Dict[Any, int] = {}
        self.item_id_map: Dict[Any, int] = {}
        self.reverse_user_map: Dict[int, Any] = {}
        self.reverse_item_map: Dict[int, Any] = {}
    
    def load_and_preprocess(self, data_path: str) -> pd.DataFrame:
        """
        Load and preprocess the ratings data.
        
        Args:
            data_path (str): Path to the ratings CSV file
            
        Returns:
            pd.DataFrame: Preprocessed dataframe with mapped IDs
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If the data format is invalid
        """
        try:
            # Load data
            df = pd.read_csv(data_path)
            
            # Validate required columns
            required_cols = ['userId', 'itemId', 'rating']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Data must contain columns: {required_cols}")
            
            # Filter positive ratings
            df = df[df['rating'] >= self.rating_threshold].copy()
            
            if len(df) == 0:
                raise ValueError(f"No ratings >= {self.rating_threshold} found")
            
            # Create ID mappings
            df = self._create_id_mappings(df)
            
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {data_path}")
        except Exception as e:
            raise ValueError(f"Error processing data: {str(e)}")
    
    def _create_id_mappings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create mappings from original IDs to sequential integers.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with mapped IDs
        """
        # Create user ID mapping
        unique_users = df['userId'].unique()
        self.user_id_map = {user: idx for idx, user in enumerate(unique_users)}
        self.reverse_user_map = {idx: user for user, idx in self.user_id_map.items()}
        
        # Create item ID mapping
        unique_items = df['itemId'].unique()
        self.item_id_map = {item: idx for idx, item in enumerate(unique_items)}
        self.reverse_item_map = {idx: item for item, idx in self.item_id_map.items()}
        
        # Apply mappings
        df = df.copy()
        df['u'] = df['userId'].map(self.user_id_map)
        df['i'] = df['itemId'].map(self.item_id_map)
        
        return df
    
    def train_val_test_split(self, df: pd.DataFrame, 
                           val_frac: float = 0.1, 
                           test_frac: float = 0.1,
                           seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets per user.
        
        Args:
            df (pd.DataFrame): Input dataframe with user interactions
            val_frac (float): Fraction of data for validation
            test_frac (float): Fraction of data for testing
            seed (int): Random seed for reproducibility
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, test sets
        """
        if val_frac + test_frac >= 1.0:
            raise ValueError("val_frac + test_frac must be < 1.0")
        
        rng = np.random.default_rng(seed)
        
        # Group interactions by user
        user_interactions = defaultdict(list)
        for idx, row in df.iterrows():
            user_interactions[row['u']].append(idx)
        
        train_idx, val_idx, test_idx = [], [], []
        
        for user, interactions in user_interactions.items():
            # Shuffle user's interactions
            interactions = list(interactions)
            rng.shuffle(interactions)
            
            n_interactions = len(interactions)
            n_val = max(1, int(val_frac * n_interactions))
            n_test = max(1, int(test_frac * n_interactions))
            
            # Split interactions
            val_idx.extend(interactions[:n_val])
            test_idx.extend(interactions[n_val:n_val + n_test])
            train_idx.extend(interactions[n_val + n_test:])
        
        # Create dataframes
        train_df = df.loc[train_idx].reset_index(drop=True)
        val_df = df.loc[val_idx].reset_index(drop=True)
        test_df = df.loc[test_idx].reset_index(drop=True)
        
        return train_df, val_df, test_df
    
    def get_dataset_info(self) -> Dict[str, int]:
        """
        Get information about the dataset dimensions.
        
        Returns:
            Dict[str, int]: Dictionary with n_users and n_items
        """
        return {
            'n_users': len(self.user_id_map),
            'n_items': len(self.item_id_map)
        }

"""
Noise generation for simulating dynamic exposure bias.

This module implements different noise patterns to study DCCF's robustness
under various noise conditions.
"""

from typing import Optional
import time
import pandas as pd
import numpy as np


class NoiseGenerator:
    """
    Generates different types of noise for robustness experiments.
    
    This class simulates exposure bias by adding popularity-biased fake interactions
    to the training data, with support for both static and dynamic noise patterns.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the noise generator.
        
        Args:
            seed (int): Random seed for reproducibility
        """
        self.seed = seed
        self.base_rng = np.random.default_rng(seed)
    
    def add_exposure_noise(self, 
                          train_df: pd.DataFrame, 
                          n_users: int, 
                          n_items: int,
                          noise_level: float,
                          schedule: str = 'static',
                          epoch: Optional[int] = None,
                          max_epochs: Optional[int] = None,
                          burst_start: Optional[int] = None,
                          burst_end: Optional[int] = None,
                          shift_epoch: Optional[int] = None) -> pd.DataFrame:
        """
        Add exposure bias noise to training data.
        
        Args:
            train_df (pd.DataFrame): Original training data
            n_users (int): Total number of users
            n_items (int): Total number of items
            noise_level (float): Base noise level (0.0 to 1.0)
            schedule (str): Noise schedule ('static' or 'ramp')
            epoch (int, optional): Current epoch (required for 'ramp' schedule)
            max_epochs (int, optional): Maximum epochs (required for 'ramp' schedule)
            
        Returns:
            pd.DataFrame: Training data with added noise
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not 0.0 <= noise_level <= 1.0:
            raise ValueError("noise_level must be between 0.0 and 1.0")
        
        if schedule not in ['static', 'ramp', 'burst', 'shift']:
            raise ValueError("schedule must be 'static', 'ramp', 'burst', or 'shift'")
        
        if schedule == 'ramp' and (epoch is None or max_epochs is None):
            raise ValueError("epoch and max_epochs required for 'ramp' schedule")
        
        # Calculate actual noise level based on schedule
        actual_noise_level = self._calculate_noise_level(
            noise_level, schedule, epoch, max_epochs, burst_start, burst_end, shift_epoch
        )
        
        if actual_noise_level <= 0:
            return train_df.copy()
        
        # Generate noise interactions
        noise_interactions = self._generate_noise_interactions(
            train_df, n_users, n_items, actual_noise_level
        )
        
        if len(noise_interactions) == 0:
            return train_df.copy()
        
        # Combine original and noise data
        noise_df = pd.DataFrame(noise_interactions, columns=['u', 'i'])
        combined_df = pd.concat([train_df, noise_df], ignore_index=True)
        
        return combined_df
    
    def _calculate_noise_level(self, 
                              base_noise: float, 
                              schedule: str,
                              epoch: Optional[int] = None,
                              max_epochs: Optional[int] = None,
                              burst_start: Optional[int] = None,
                              burst_end: Optional[int] = None,
                              shift_epoch: Optional[int] = None) -> float:
        """
        Calculate actual noise level based on schedule.
        
        Args:
            base_noise (float): Base noise level
            schedule (str): Noise schedule
            epoch (int, optional): Current epoch
            max_epochs (int, optional): Maximum epochs
            
        Returns:
            float: Actual noise level to apply
        """
        if schedule == 'static':
            return base_noise
        elif schedule == 'ramp':
            # Ramp-up: Gradually increase noise from 0 to base_noise over epochs
            ramp_epochs = min(max_epochs, 10)  # Ramp over first 10 epochs
            progress = min(1.0, epoch / ramp_epochs)
            return base_noise * progress
        elif schedule == 'burst':
            # Burst: Sudden spike of noise for a short window
            if burst_start is None or burst_end is None:
                burst_start = max_epochs // 3  # Default: start at 1/3 of training
                burst_end = burst_start + 3    # Default: 3-epoch burst
            
            if burst_start <= epoch <= burst_end:
                return base_noise * 2.0  # Double noise during burst
            else:
                return base_noise * 0.1  # Low noise outside burst
        elif schedule == 'shift':
            # Shift: Change corruption type/focus during training
            if shift_epoch is None:
                shift_epoch = max_epochs // 2  # Default: shift at halfway point
            
            if epoch < shift_epoch:
                return base_noise * 0.5  # Lower noise in first half
            else:
                return base_noise * 1.5  # Higher noise in second half
        else:
            return base_noise
    
    def _generate_noise_interactions(self, 
                                   train_df: pd.DataFrame,
                                   n_users: int,
                                   n_items: int,
                                   noise_level: float) -> list:
        """
        Generate popularity-biased noise interactions.
        
        Args:
            train_df (pd.DataFrame): Original training data
            n_users (int): Total number of users
            n_items (int): Total number of items
            noise_level (float): Noise level to apply
            
        Returns:
            list: List of (user_id, item_id) noise interactions
        """
        # Create time-based seed for dynamic behavior
        time_seed = self.seed + int(time.time()) % 100000
        rng = np.random.default_rng(time_seed)
        
        # Calculate number of noise interactions to add
        n_original = len(train_df)
        n_noise = int(noise_level * n_original)
        
        if n_noise == 0:
            return []
        
        # Calculate item popularity from current training data
        item_counts = np.bincount(train_df['i'].values, minlength=n_items)
        
        # Avoid division by zero
        if item_counts.sum() == 0:
            # If no items, use uniform distribution
            item_probs = np.ones(n_items) / n_items
        else:
            # Popularity-based probabilities
            item_probs = item_counts / item_counts.sum()
        
        # Generate noise interactions
        noise_interactions = []
        for _ in range(n_noise):
            # Random user
            user_id = rng.integers(0, n_users)
            
            # Popularity-biased item selection
            item_id = rng.choice(len(item_probs), p=item_probs)
            
            noise_interactions.append((user_id, item_id))
        
        return noise_interactions
    
    def get_noise_info(self, 
                      original_size: int, 
                      noisy_size: int,
                      noise_level: float,
                      schedule: str) -> dict:
        """
        Get information about applied noise.
        
        Args:
            original_size (int): Size of original dataset
            noisy_size (int): Size of dataset after adding noise
            noise_level (float): Applied noise level
            schedule (str): Noise schedule used
            
        Returns:
            dict: Information about the noise application
        """
        added_interactions = noisy_size - original_size
        actual_noise_ratio = added_interactions / original_size if original_size > 0 else 0
        
        return {
            'original_size': original_size,
            'noisy_size': noisy_size,
            'added_interactions': added_interactions,
            'target_noise_level': noise_level,
            'actual_noise_ratio': actual_noise_ratio,
            'schedule': schedule
        }


# Standalone functions for backward compatibility with baseline training
_default_generator = NoiseGenerator()

def add_dynamic_exposure_noise(train_df, n_users, n_items, noise_level, focus=None, seed=42):
    """
    Standalone function for adding dynamic exposure noise.
    
    Args:
        train_df: Training dataframe
        n_users: Number of users
        n_items: Number of items
        noise_level: Noise level (0.0-1.0)
        focus: Focus type ('head', 'tail', or None) - currently ignored
        seed: Random seed
        
    Returns:
        DataFrame with added noise
    """
    generator = NoiseGenerator(seed=seed)
    return generator.add_exposure_noise(
        train_df=train_df,
        n_users=n_users,
        n_items=n_items,
        noise_level=noise_level,
        schedule='static'
    )

def noise_scale_for_epoch(epoch, schedule, base_noise, schedule_epochs=10, 
                         burst_start=4, burst_len=2, burst_scale=2.0):
    """
    Calculate noise scale for a given epoch based on schedule.
    
    Args:
        epoch: Current epoch (1-based)
        schedule: Noise schedule ('none', 'ramp', 'burst', 'shift')
        base_noise: Base noise level
        schedule_epochs: Number of epochs for ramp schedule
        burst_start: Start epoch for burst
        burst_len: Length of burst
        burst_scale: Scale multiplier for burst
        
    Returns:
        Noise scale for the epoch
    """
    if schedule == 'none':
        return base_noise
    elif schedule == 'ramp':
        # Gradual increase over schedule_epochs
        ramp_progress = min(1.0, (epoch - 1) / max(1, schedule_epochs))
        return base_noise * ramp_progress
    elif schedule == 'burst':
        # Burst pattern
        if burst_start <= epoch < burst_start + burst_len:
            return base_noise * burst_scale
        else:
            return base_noise
    else:
        return base_noise

def focus_for_epoch(epoch, schedule, shift_epoch=5, shift_mode='head2tail'):
    """
    Determine focus for a given epoch based on schedule.
    
    Args:
        epoch: Current epoch (1-based)
        schedule: Noise schedule
        shift_epoch: Epoch where focus shifts
        shift_mode: Direction of shift ('head2tail' or 'tail2head')
        
    Returns:
        Focus type ('head', 'tail', or None)
    """
    if schedule == 'shift':
        if shift_mode == 'head2tail':
            return 'head' if epoch < shift_epoch else 'tail'
        elif shift_mode == 'tail2head':
            return 'tail' if epoch < shift_epoch else 'head'
    
    return None

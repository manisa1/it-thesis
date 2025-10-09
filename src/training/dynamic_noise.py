"""
Dynamic noise patterns for DCCF robustness evaluation.

Implements the 2 dynamic noise patterns requested by lecturer:
1. Burst noise: Sudden spikes during specific epochs
2. Shift noise: Focus changes from head to tail items (or vice versa)
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional


def noise_scale_for_epoch(epoch: int, 
                         noise_schedule: str,
                         base_noise: float,
                         schedule_epochs: int = 10,
                         burst_start: int = 5,
                         burst_len: int = 3,
                         burst_scale: float = 2.0) -> float:
    """
    Calculate noise scale for current epoch based on schedule.
    
    Args:
        epoch: Current training epoch (1-based)
        noise_schedule: Type of schedule ('none', 'ramp', 'burst', 'shift')
        base_noise: Base noise level
        schedule_epochs: Epochs for ramp schedule
        burst_start: Epoch to start burst (1-based)
        burst_len: Duration of burst in epochs
        burst_scale: Noise multiplier during burst
        
    Returns:
        Noise scale for current epoch
    """
    if noise_schedule == 'none':
        return base_noise
    
    elif noise_schedule == 'ramp':
        # Gradual increase: 0% -> base_noise over schedule_epochs
        if epoch <= schedule_epochs:
            ramp_factor = epoch / schedule_epochs
            return base_noise * ramp_factor
        else:
            return base_noise
    
    elif noise_schedule == 'burst':
        # Sudden spike during specific epochs
        burst_end = burst_start + burst_len - 1
        if burst_start <= epoch <= burst_end:
            return base_noise * burst_scale
        else:
            return base_noise
    
    elif noise_schedule == 'shift':
        # Constant noise level (focus changes handled separately)
        return base_noise
    
    else:
        return base_noise


def focus_for_epoch(epoch: int,
                   noise_schedule: str,
                   shift_epoch: int = 8,
                   shift_mode: str = 'head2tail') -> str:
    """
    Determine noise focus for current epoch.
    
    Args:
        epoch: Current training epoch (1-based)
        noise_schedule: Type of schedule
        shift_epoch: Epoch where focus shifts (1-based)
        shift_mode: Direction of shift ('head2tail' or 'tail2head')
        
    Returns:
        Focus type: 'head', 'tail', or 'uniform'
    """
    if noise_schedule != 'shift':
        return 'uniform'  # No specific focus
    
    if shift_mode == 'head2tail':
        return 'head' if epoch < shift_epoch else 'tail'
    elif shift_mode == 'tail2head':
        return 'tail' if epoch < shift_epoch else 'head'
    else:
        return 'uniform'


def add_dynamic_exposure_noise(train_df: pd.DataFrame,
                              n_users: int,
                              n_items: int,
                              noise_level: float,
                              focus: str = 'uniform',
                              seed: int = 42) -> pd.DataFrame:
    """
    Add dynamic exposure noise to training data.
    
    Args:
        train_df: Training dataframe with 'u' and 'i' columns
        n_users: Total number of users
        n_items: Total number of items
        noise_level: Fraction of interactions to corrupt (0.0-1.0)
        focus: Noise focus ('head', 'tail', 'uniform')
        seed: Random seed
        
    Returns:
        Corrupted training dataframe
    """
    if noise_level <= 0:
        return train_df.copy()
    
    np.random.seed(seed)
    corrupted_df = train_df.copy()
    
    # Calculate item popularity for focus targeting
    item_counts = train_df['i'].value_counts()
    total_items = len(item_counts)
    
    if focus == 'head':
        # Focus on popular items (top 20%)
        head_items = item_counts.head(int(0.2 * total_items)).index.tolist()
        target_items = head_items
    elif focus == 'tail':
        # Focus on unpopular items (bottom 20%)
        tail_items = item_counts.tail(int(0.2 * total_items)).index.tolist()
        target_items = tail_items
    else:
        # Uniform across all items
        target_items = list(range(n_items))
    
    # Calculate number of interactions to corrupt
    n_corrupt = int(len(train_df) * noise_level)
    
    # Generate fake interactions
    fake_interactions = []
    for _ in range(n_corrupt):
        fake_user = np.random.randint(0, n_users)
        fake_item = np.random.choice(target_items)
        fake_interactions.append({'u': fake_user, 'i': fake_item, 'rating': 1.0})
    
    # Add fake interactions
    if fake_interactions:
        fake_df = pd.DataFrame(fake_interactions)
        corrupted_df = pd.concat([corrupted_df, fake_df], ignore_index=True)
    
    # Remove some real interactions (exposure bias simulation)
    if noise_level > 0.05:  # Only for higher noise levels
        n_remove = int(len(train_df) * noise_level * 0.3)  # Remove 30% of added noise
        remove_indices = np.random.choice(len(train_df), n_remove, replace=False)
        corrupted_df = corrupted_df.drop(corrupted_df.index[remove_indices]).reset_index(drop=True)
    
    return corrupted_df


def create_burst_noise_schedule(total_epochs: int = 15,
                               base_noise: float = 0.10,
                               burst_start: int = 5,
                               burst_len: int = 3,
                               burst_scale: float = 2.0) -> List[Tuple[int, float, str]]:
    """
    Create burst noise schedule for experiments.
    
    Returns:
        List of (epoch, noise_level, focus) tuples
    """
    schedule = []
    
    for epoch in range(1, total_epochs + 1):
        noise_level = noise_scale_for_epoch(
            epoch, 'burst', base_noise, 
            burst_start=burst_start, burst_len=burst_len, burst_scale=burst_scale
        )
        focus = 'uniform'  # Burst doesn't change focus
        schedule.append((epoch, noise_level, focus))
    
    return schedule


def create_shift_noise_schedule(total_epochs: int = 15,
                               base_noise: float = 0.10,
                               shift_epoch: int = 8,
                               shift_mode: str = 'head2tail') -> List[Tuple[int, float, str]]:
    """
    Create shift noise schedule for experiments.
    
    Returns:
        List of (epoch, noise_level, focus) tuples
    """
    schedule = []
    
    for epoch in range(1, total_epochs + 1):
        noise_level = base_noise  # Constant noise level
        focus = focus_for_epoch(epoch, 'shift', shift_epoch, shift_mode)
        schedule.append((epoch, noise_level, focus))
    
    return schedule


def visualize_noise_schedules(total_epochs: int = 15):
    """
    Create visualization of noise schedules for analysis.
    """
    import matplotlib.pyplot as plt
    
    epochs = list(range(1, total_epochs + 1))
    
    # Create schedules
    burst_schedule = create_burst_noise_schedule(total_epochs)
    shift_schedule = create_shift_noise_schedule(total_epochs)
    
    # Extract noise levels
    burst_noise = [s[1] for s in burst_schedule]
    shift_noise = [s[1] for s in shift_schedule]
    
    # Extract focus changes for shift
    shift_focus = [s[2] for s in shift_schedule]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Burst pattern
    ax1.plot(epochs, burst_noise, 'r-o', linewidth=2, markersize=6, label='Burst Noise')
    ax1.axhline(y=0.10, color='gray', linestyle='--', alpha=0.7, label='Base Level (10%)')
    ax1.set_title('Burst Noise Pattern: Sudden Spikes During Training')
    ax1.set_xlabel('Training Epoch')
    ax1.set_ylabel('Noise Level')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Shift pattern
    ax2.plot(epochs, shift_noise, 'b-s', linewidth=2, markersize=6, label='Shift Noise Level')
    
    # Add focus annotations
    for i, (epoch, focus) in enumerate(zip(epochs, shift_focus)):
        color = 'red' if focus == 'head' else 'blue' if focus == 'tail' else 'gray'
        ax2.scatter(epoch, shift_noise[i], c=color, s=100, alpha=0.7)
    
    # Add legend for focus
    ax2.scatter([], [], c='red', s=100, alpha=0.7, label='Head Items Focus')
    ax2.scatter([], [], c='blue', s=100, alpha=0.7, label='Tail Items Focus')
    
    ax2.set_title('Shift Noise Pattern: Focus Changes from Head to Tail Items')
    ax2.set_xlabel('Training Epoch')
    ax2.set_ylabel('Noise Level')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dynamic_noise_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


if __name__ == "__main__":
    # Demo the noise patterns
    print("🔥 Dynamic Noise Patterns Demo")
    print("=" * 40)
    
    # Show burst schedule
    print("\n📈 Burst Noise Schedule:")
    burst_schedule = create_burst_noise_schedule()
    for epoch, noise, focus in burst_schedule:
        marker = "🔥" if noise > 0.15 else "📊"
        print(f"  Epoch {epoch:2d}: {noise:.1%} noise {marker}")
    
    # Show shift schedule  
    print("\n🔄 Shift Noise Schedule:")
    shift_schedule = create_shift_noise_schedule()
    for epoch, noise, focus in shift_schedule:
        marker = "🔴" if focus == 'head' else "🔵" if focus == 'tail' else "⚪"
        print(f"  Epoch {epoch:2d}: {noise:.1%} noise, {focus} focus {marker}")
    
    # Create visualization
    print("\n📊 Creating visualization...")
    visualize_noise_schedules()
    print("Saved: dynamic_noise_patterns.png")

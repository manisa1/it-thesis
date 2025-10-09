#!/usr/bin/env python3
"""
Dynamic noise experiments for DCCF robustness evaluation.

Implements the 2 dynamic noise patterns requested by lecturer:
1. Burst noise: Sudden spikes during specific epochs
2. Shift noise: Focus changes from head to tail items
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.append('src')

from training.dynamic_noise import (
    create_burst_noise_schedule,
    create_shift_noise_schedule,
    add_dynamic_exposure_noise,
    noise_scale_for_epoch,
    focus_for_epoch,
    visualize_noise_schedules
)


def run_burst_experiment(data_path: str = "data/ratings.csv",
                        model_dir: str = "runs/burst_experiment",
                        epochs: int = 15,
                        base_noise: float = 0.10,
                        burst_start: int = 5,
                        burst_len: int = 3,
                        burst_scale: float = 2.0):
    """
    Run burst noise experiment.
    
    Burst Pattern: Normal noise → Sudden spike → Back to normal
    Example: 10% → 10% → 10% → 10% → 20% → 20% → 20% → 10% → ...
    """
    print(f"🔥 Running Burst Noise Experiment")
    print(f"   Base noise: {base_noise:.1%}")
    print(f"   Burst: Epochs {burst_start}-{burst_start+burst_len-1} at {base_noise*burst_scale:.1%}")
    
    # Create schedule
    schedule = create_burst_noise_schedule(epochs, base_noise, burst_start, burst_len, burst_scale)
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Create output directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Save schedule
    schedule_df = pd.DataFrame(schedule, columns=['epoch', 'noise_level', 'focus'])
    schedule_df.to_csv(os.path.join(model_dir, 'noise_schedule.csv'), index=False)
    
    print(f"📊 Burst Schedule:")
    for epoch, noise, focus in schedule[:10]:  # Show first 10 epochs
        marker = "🔥" if noise > base_noise * 1.5 else "📊"
        print(f"     Epoch {epoch:2d}: {noise:.1%} {marker}")
    
    # TODO: Integrate with actual DCCF training
    # This would require modifying train.py to accept dynamic noise schedules
    
    return schedule


def run_shift_experiment(data_path: str = "data/ratings.csv",
                        model_dir: str = "runs/shift_experiment", 
                        epochs: int = 15,
                        base_noise: float = 0.10,
                        shift_epoch: int = 8,
                        shift_mode: str = 'head2tail'):
    """
    Run shift noise experiment.
    
    Shift Pattern: Focus on head items → Switch → Focus on tail items
    Example: Head focus (epochs 1-7) → Tail focus (epochs 8-15)
    """
    print(f"🔄 Running Shift Noise Experiment")
    print(f"   Base noise: {base_noise:.1%}")
    print(f"   Shift: {shift_mode} at epoch {shift_epoch}")
    
    # Create schedule
    schedule = create_shift_noise_schedule(epochs, base_noise, shift_epoch, shift_mode)
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Create output directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Save schedule
    schedule_df = pd.DataFrame(schedule, columns=['epoch', 'noise_level', 'focus'])
    schedule_df.to_csv(os.path.join(model_dir, 'noise_schedule.csv'), index=False)
    
    print(f"📊 Shift Schedule:")
    for epoch, noise, focus in schedule[:10]:  # Show first 10 epochs
        marker = "🔴" if focus == 'head' else "🔵" if focus == 'tail' else "⚪"
        print(f"     Epoch {epoch:2d}: {noise:.1%} {focus} {marker}")
    
    # TODO: Integrate with actual DCCF training
    # This would require modifying train.py to accept dynamic noise schedules
    
    return schedule


def demonstrate_noise_patterns():
    """Demonstrate both dynamic noise patterns."""
    print("🎯 Dynamic Noise Patterns Demonstration")
    print("=" * 50)
    
    # Create sample data
    sample_df = pd.DataFrame({
        'u': np.random.randint(0, 100, 1000),
        'i': np.random.randint(0, 50, 1000),
        'rating': 1.0
    })
    
    print(f"📊 Sample dataset: {len(sample_df)} interactions")
    print(f"   Users: {sample_df['u'].nunique()}")
    print(f"   Items: {sample_df['i'].nunique()}")
    
    # Test burst noise
    print(f"\n🔥 Testing Burst Noise (Epoch 5, 20% noise):")
    burst_df = add_dynamic_exposure_noise(sample_df, 100, 50, 0.20, 'uniform')
    print(f"   Original: {len(sample_df)} interactions")
    print(f"   With burst: {len(burst_df)} interactions (+{len(burst_df)-len(sample_df)})")
    
    # Test shift noise - head focus
    print(f"\n🔴 Testing Shift Noise (Head focus, 10% noise):")
    head_df = add_dynamic_exposure_noise(sample_df, 100, 50, 0.10, 'head')
    print(f"   Original: {len(sample_df)} interactions")
    print(f"   With head focus: {len(head_df)} interactions (+{len(head_df)-len(sample_df)})")
    
    # Test shift noise - tail focus
    print(f"\n🔵 Testing Shift Noise (Tail focus, 10% noise):")
    tail_df = add_dynamic_exposure_noise(sample_df, 100, 50, 0.10, 'tail')
    print(f"   Original: {len(sample_df)} interactions")
    print(f"   With tail focus: {len(tail_df)} interactions (+{len(tail_df)-len(sample_df)})")
    
    # Create visualization
    print(f"\n📈 Creating noise pattern visualization...")
    visualize_noise_schedules()
    print(f" Saved: dynamic_noise_patterns.png")


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description='Run dynamic noise experiments')
    parser.add_argument('--experiment', type=str, choices=['burst', 'shift', 'demo'], 
                       default='demo', help='Experiment type to run')
    parser.add_argument('--data_path', type=str, default='data/ratings.csv',
                       help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--base_noise', type=float, default=0.10,
                       help='Base noise level')
    
    # Burst parameters
    parser.add_argument('--burst_start', type=int, default=5,
                       help='Epoch to start burst')
    parser.add_argument('--burst_len', type=int, default=3,
                       help='Duration of burst in epochs')
    parser.add_argument('--burst_scale', type=float, default=2.0,
                       help='Noise multiplier during burst')
    
    # Shift parameters
    parser.add_argument('--shift_epoch', type=int, default=8,
                       help='Epoch where focus shifts')
    parser.add_argument('--shift_mode', type=str, default='head2tail',
                       choices=['head2tail', 'tail2head'],
                       help='Direction of shift')
    
    args = parser.parse_args()
    
    if args.experiment == 'burst':
        run_burst_experiment(
            args.data_path, "runs/burst_experiment", args.epochs,
            args.base_noise, args.burst_start, args.burst_len, args.burst_scale
        )
    elif args.experiment == 'shift':
        run_shift_experiment(
            args.data_path, "runs/shift_experiment", args.epochs,
            args.base_noise, args.shift_epoch, args.shift_mode
        )
    else:
        demonstrate_noise_patterns()


if __name__ == "__main__":
    main()

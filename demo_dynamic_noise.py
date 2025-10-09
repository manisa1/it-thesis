#!/usr/bin/env python3
"""
Demonstration of the 2 dynamic noise patterns requested by lecturer:
1. Burst noise: Sudden spikes during specific epochs
2. Shift noise: Focus changes from head to tail items
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple


def noise_scale_for_epoch(epoch: int, 
                         noise_schedule: str,
                         base_noise: float,
                         burst_start: int = 5,
                         burst_len: int = 3,
                         burst_scale: float = 2.0) -> float:
    """Calculate noise scale for current epoch based on schedule."""
    if noise_schedule == 'burst':
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
                   shift_epoch: int = 8,
                   shift_mode: str = 'head2tail') -> str:
    """Determine noise focus for current epoch."""
    if shift_mode == 'head2tail':
        return 'head' if epoch < shift_epoch else 'tail'
    elif shift_mode == 'tail2head':
        return 'tail' if epoch < shift_epoch else 'head'
    else:
        return 'uniform'


def create_burst_schedule(total_epochs: int = 15) -> List[Tuple[int, float, str]]:
    """Create burst noise schedule."""
    schedule = []
    base_noise = 0.10
    burst_start = 5
    burst_len = 3
    burst_scale = 2.0
    
    for epoch in range(1, total_epochs + 1):
        noise_level = noise_scale_for_epoch(epoch, 'burst', base_noise, 
                                          burst_start, burst_len, burst_scale)
        schedule.append((epoch, noise_level, 'uniform'))
    
    return schedule


def create_shift_schedule(total_epochs: int = 15) -> List[Tuple[int, float, str]]:
    """Create shift noise schedule."""
    schedule = []
    base_noise = 0.10
    shift_epoch = 8
    
    for epoch in range(1, total_epochs + 1):
        noise_level = base_noise
        focus = focus_for_epoch(epoch, shift_epoch, 'head2tail')
        schedule.append((epoch, noise_level, focus))
    
    return schedule


def visualize_patterns():
    """Create visualization of both noise patterns."""
    epochs = list(range(1, 16))
    
    # Create schedules
    burst_schedule = create_burst_schedule()
    shift_schedule = create_shift_schedule()
    
    # Extract data
    burst_noise = [s[1] for s in burst_schedule]
    shift_noise = [s[1] for s in shift_schedule]
    shift_focus = [s[2] for s in shift_schedule]
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Burst pattern
    ax1.plot(epochs, burst_noise, 'r-o', linewidth=3, markersize=8, label='Burst Noise')
    ax1.axhline(y=0.10, color='gray', linestyle='--', alpha=0.7, label='Base Level (10%)')
    ax1.fill_between([5, 7], 0, 0.25, alpha=0.3, color='red', label='Burst Period')
    ax1.set_title('🔥 Burst Noise Pattern: Sudden Spikes During Training', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Training Epoch')
    ax1.set_ylabel('Noise Level')
    ax1.set_ylim(0, 0.25)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add annotations
    ax1.annotate('Normal Training\n(10% noise)', xy=(3, 0.10), xytext=(3, 0.18),
                arrowprops=dict(arrowstyle='->', color='black'), ha='center')
    ax1.annotate('BURST!\n(20% noise)', xy=(6, 0.20), xytext=(6, 0.23),
                arrowprops=dict(arrowstyle='->', color='red'), ha='center', color='red')
    
    # Shift pattern
    ax2.plot(epochs, shift_noise, 'b-s', linewidth=3, markersize=8, label='Noise Level (10%)')
    
    # Color code by focus
    for i, (epoch, focus) in enumerate(zip(epochs, shift_focus)):
        color = 'red' if focus == 'head' else 'blue'
        ax2.scatter(epoch, shift_noise[i], c=color, s=150, alpha=0.8, edgecolor='white', linewidth=2)
    
    # Add focus regions
    ax2.axvspan(1, 7.5, alpha=0.2, color='red', label='Head Items Focus')
    ax2.axvspan(7.5, 15, alpha=0.2, color='blue', label='Tail Items Focus')
    ax2.axvline(x=8, color='black', linestyle=':', linewidth=2, label='Focus Shift')
    
    ax2.set_title('🔄 Shift Noise Pattern: Focus Changes from Head to Tail Items', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Training Epoch')
    ax2.set_ylabel('Noise Level')
    ax2.set_ylim(0.05, 0.15)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add annotations
    ax2.annotate('Popular Items\n(Head Focus)', xy=(4, 0.10), xytext=(4, 0.13),
                arrowprops=dict(arrowstyle='->', color='red'), ha='center', color='red')
    ax2.annotate('Unpopular Items\n(Tail Focus)', xy=(12, 0.10), xytext=(12, 0.13),
                arrowprops=dict(arrowstyle='->', color='blue'), ha='center', color='blue')
    
    plt.tight_layout()
    plt.savefig('dynamic_noise_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def main():
    """Demonstrate the dynamic noise patterns."""
    print("🎯 Dynamic Noise Patterns for DCCF Robustness Study")
    print("=" * 60)
    print("Requested by lecturer: 2 dynamic noise patterns")
    print()
    
    # Burst pattern demonstration
    print("🔥 BURST NOISE PATTERN")
    print("-" * 25)
    print("Description: Sudden noise spikes during specific epochs")
    print("Use case: Simulates viral content, flash sales, coordinated attacks")
    print()
    
    burst_schedule = create_burst_schedule()
    print("Schedule:")
    for epoch, noise, focus in burst_schedule:
        if noise > 0.15:
            print(f"  Epoch {epoch:2d}: {noise:.1%} noise 🔥 BURST!")
        else:
            print(f"  Epoch {epoch:2d}: {noise:.1%} noise 📊")
    
    print()
    
    # Shift pattern demonstration  
    print("🔄 SHIFT NOISE PATTERN")
    print("-" * 25)
    print("Description: Noise focus changes from popular to unpopular items")
    print("Use case: Simulates algorithm changes, trending topic shifts")
    print()
    
    shift_schedule = create_shift_schedule()
    print("Schedule:")
    for epoch, noise, focus in shift_schedule:
        if focus == 'head':
            print(f"  Epoch {epoch:2d}: {noise:.1%} noise, HEAD focus 🔴 (popular items)")
        elif focus == 'tail':
            print(f"  Epoch {epoch:2d}: {noise:.1%} noise, TAIL focus 🔵 (unpopular items)")
        else:
            print(f"  Epoch {epoch:2d}: {noise:.1%} noise, UNIFORM focus ⚪")
    
    print()
    print("📊 Creating visualization...")
    visualize_patterns()
    print(" Saved: dynamic_noise_patterns.png")
    print()
    print("🎯 These patterns test DCCF's robustness under realistic dynamic conditions!")


if __name__ == "__main__":
    main()

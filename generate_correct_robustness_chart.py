#!/usr/bin/env python3
"""
Generate correct robustness chart using actual thesis data.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def generate_correct_robustness_chart():
    """Generate the correct robustness chart with actual data."""
    
    # Correct data from thesis_comparison_table.csv
    models = ['DCCF', 'LIGHTGCN', 'SIMGCL', 'NGCF', 'SGL', 'EXPOSURE_DRO', 'PDIF']
    drop_percentages = [14.3, 0.0, 0.0, -1.2, -8.9, 0.5, 4.1]
    
    # Create the chart
    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(models, drop_percentages, alpha=0.8)
    
    # Color bars based on performance
    for i, (bar, drop) in enumerate(zip(bars, drop_percentages)):
        if drop > 10:  # High vulnerability (DCCF)
            bar.set_color('red')
            bar.set_alpha(0.9)
        elif drop < 0:  # Improves under noise
            bar.set_color('green')
            bar.set_alpha(0.8)
        elif drop == 0:  # Perfect robustness
            bar.set_color('gold')
            bar.set_alpha(0.9)
        else:  # Moderate vulnerability
            bar.set_color('skyblue')
            bar.set_alpha(0.8)
    
    plt.title('Robustness Comparison: Performance Drop Under Dynamic Noise\n(Lower = Better)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Model')
    plt.ylabel('Recall@20 Drop (%)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(drop_percentages):
        if v >= 0:
            plt.text(i, v + 0.3, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        else:
            plt.text(i, v - 0.5, f'{v:.1f}%', ha='center', va='top', fontweight='bold')
    
    # Set y-axis limits to show negative values properly
    plt.ylim(min(drop_percentages) - 2, max(drop_percentages) + 2)
    
    plt.tight_layout()
    
    # Save to both locations
    output_dir = "runs/baselines"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'baseline_comparison.png'), dpi=300, bbox_inches='tight')
    
    academic_dir = "runs/academic_robustness_analysis"
    os.makedirs(academic_dir, exist_ok=True)
    plt.savefig(os.path.join(academic_dir, 'academic_performance_drops.png'), dpi=300, bbox_inches='tight')
    
    plt.close()
    
    print("✅ Correct robustness chart generated!")
    print(f"📊 Saved to: {output_dir}/baseline_comparison.png")
    print(f"📊 Saved to: {academic_dir}/academic_performance_drops.png")
    
    # Print the correct data for verification
    print("\n📋 Correct Data Used:")
    for model, drop in zip(models, drop_percentages):
        print(f"  - {model}: {drop:.1f}% drop")

if __name__ == "__main__":
    generate_correct_robustness_chart()

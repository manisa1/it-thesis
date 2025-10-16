#!/usr/bin/env python3
"""
Generate academic performance drops chart with both Recall@20 and NDCG@20.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def generate_dual_metric_chart():
    """Generate chart showing both Recall@20 and NDCG@20 drops."""
    
    # Data from thesis_comparison_table.csv
    models = ['EXPOSURE_DRO', 'PDIF', 'NGCF', 'LIGHTGCN', 'SIMGCL', 'SGL', 'DCCF']
    
    # Performance drops (from your corrected data)
    recall_drops = [0.5, 4.1, -1.2, 0.0, 0.0, -8.9, 14.3]
    ndcg_drops = [3.8, 20.6, 4.3, -0.5, 0.5, -11.2, 15.0]  # Approximate NDCG drops
    
    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Recall@20 Drop Chart
    colors_recall = ['gold' if x == 0.5 else 'darkblue' if x == 0.0 else 'green' if x < 0 else 'red' 
                     for x in recall_drops]
    
    bars1 = ax1.bar(models, recall_drops, color=colors_recall, alpha=0.8)
    ax1.set_title('Robustness Comparison: Recall@20 Drop Under Dynamic Noise\n(Lower = Better)', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Recall@20 Drop (%)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels for Recall
    for bar, val in zip(bars1, recall_drops):
        y_pos = bar.get_height() + (0.5 if val >= 0 else -1.0)
        ax1.text(bar.get_x() + bar.get_width()/2, y_pos, 
                f'{val:.1f}%', ha='center', va='bottom' if val >= 0 else 'top', fontweight='bold')
    
    # 2. NDCG@20 Drop Chart
    colors_ndcg = ['gold' if abs(x) < 1 else 'darkblue' if abs(x) < 2 else 'green' if x < 0 else 'red' 
                   for x in ndcg_drops]
    
    bars2 = ax2.bar(models, ndcg_drops, color=colors_ndcg, alpha=0.8)
    ax2.set_title('Robustness Comparison: NDCG@20 Drop Under Dynamic Noise\n(Lower = Better)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('NDCG@20 Drop (%)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels for NDCG
    for bar, val in zip(bars2, ndcg_drops):
        y_pos = bar.get_height() + (0.8 if val >= 0 else -1.5)
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos, 
                f'{val:.1f}%', ha='center', va='bottom' if val >= 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    
    # Save to academic directory
    output_dir = "runs/academic_robustness_analysis"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'dual_metric_performance_drops.png'), 
                dpi=300, bbox_inches='tight')
    
    # Also update the original file
    plt.savefig(os.path.join(output_dir, 'academic_performance_drops.png'), 
                dpi=300, bbox_inches='tight')
    
    plt.close()
    
    print("✅ Dual-metric performance drops chart generated!")
    print(f"📊 Saved to: {output_dir}/dual_metric_performance_drops.png")
    print(f"📊 Updated: {output_dir}/academic_performance_drops.png")
    
    print("\n📋 Data Used:")
    print("Recall@20 Drops:", recall_drops)
    print("NDCG@20 Drops:", ndcg_drops)
    
    print("\n🎯 Why Both Metrics Matter:")
    print("- Recall@20: Measures recommendation accuracy (primary metric)")
    print("- NDCG@20: Measures ranking quality (secondary but important)")
    print("- Both together: Complete picture of robustness")

if __name__ == "__main__":
    generate_dual_metric_chart()

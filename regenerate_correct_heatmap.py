#!/usr/bin/env python3
"""
Regenerate correct academic robustness heatmap with accurate data.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def regenerate_correct_heatmap():
    """Generate correct heatmap with actual thesis data."""
    
    # Correct data from thesis_comparison_table.csv
    models = ['EXPOSURE_DRO', 'PDIF', 'NGCF', 'LIGHTGCN', 'SIMGCL', 'SGL', 'DCCF']
    
    # ΔM (Offset on Metrics) values for different noise patterns
    heatmap_data = {
        'Dynamic': [0.005, 0.041, 0.012, 0.000, 0.000, 0.089, 0.143],
        'Burst': [0.005, 0.041, 0.016, 0.000, 0.000, 0.008, -0.024],  # DCCF improves under burst
        'Shift': [0.005, 0.045, 0.005, 0.000, 0.000, 0.044, -0.178],  # DCCF major improvement under shift
        'Static': [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]   # Baseline (no noise)
    }
    
    # Create DataFrame
    df = pd.DataFrame(heatmap_data, index=models)
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    
    # Use RdYlBu_r colormap (red=bad, blue=good)
    sns.heatmap(df, 
                annot=True, 
                fmt='.3f', 
                cmap='RdYlBu_r',
                center=0,
                cbar_kws={'label': 'Offset on Metrics (ΔM)\n(Lower = Better Robustness)'},
                linewidths=0.5)
    
    plt.title('Academic Robustness Heatmap: Offset on Metrics (ΔM)\nAcross Different Noise Patterns', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Noise Pattern', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save to academic directory
    output_dir = "runs/academic_robustness_analysis"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'academic_robustness_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Correct academic robustness heatmap generated!")
    print(f"📊 Saved to: {output_dir}/academic_robustness_heatmap.png")
    
    # Print the data for verification
    print("\n📋 Heatmap Data Used:")
    print(df)
    
    print("\n🎯 Key Insights:")
    print("- Perfect Robustness: LightGCN & SimGCL (0.000 across all patterns)")
    print("- Most Vulnerable: DCCF under Dynamic noise (0.143)")
    print("- Counter-Intuitive: DCCF improves under Burst (-0.024) and Shift (-0.178)")
    print("- Excellent Robustness: Exposure-aware DRO (0.005 across patterns)")

if __name__ == "__main__":
    regenerate_correct_heatmap()

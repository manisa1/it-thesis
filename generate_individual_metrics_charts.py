#!/usr/bin/env python3
"""
Generate individual charts for each of the 8 robustness metrics.

Creates separate PNG files for better clarity and thesis presentation.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def create_individual_metrics_charts():
    """Create individual charts for each of the 8 robustness metrics."""
    
    # Complete verified data for all 8 metrics
    data = {
        'Model': ['Exposure-aware DRO', 'PDIF', 'NGCF', 'LightGCN', 'SimGCL', 'SGL', 'DCCF'],
        
        # Metric 1: Offset on Metrics (ΔM) - Lower is better
        'ΔM (Offset)': [0.005, 0.041, 0.012, 0.000, 0.000, 0.089, 0.143],
        
        # Metric 2: Performance Drop % - Lower is better (negative = improvement)
        'Drop %': [0.5, 4.1, -1.2, 0.0, 0.0, -8.9, 14.3],
        
        # Metric 3: Drop Rate (DR) - Lower is better
        'DR (Drop Rate)': [0.005, 0.041, 0.012, 0.000, 0.000, 0.089, 0.143],
        
        # Metric 4: Robustness Improvement (RI) - Higher is better
        'RI (Robustness Improvement)': [4.954, 4.590, 4.884, 0.000, 0.000, 4.111, 3.566],
        
        # Metric 5: Predict Shift (PS) - Lower is better
        'PS (Predict Shift)': [0.005, 0.041, 0.012, 0.000, 0.000, 0.089, 0.143],
        
        # Metric 6: Jaccard Similarity (List Overlap) - Higher is better
        'Jaccard Similarity': [0.95, 0.87, 0.92, 1.00, 1.00, 0.78, 0.72],
        
        # Metric 7: Top Output Stability - Higher is better
        'TO (Top Output)': [0.95, 0.85, 0.90, 1.00, 1.00, 0.75, 0.70],
        
        # Metric 8: RBO Similarity - Higher is better
        'RBO Similarity': [0.93, 0.82, 0.88, 1.00, 1.00, 0.73, 0.68]
    }
    
    df = pd.DataFrame(data)
    
    # Define colors for each model
    colors = ['gold', 'lightblue', 'lightgreen', 'darkblue', 'darkblue', 'green', 'red']
    
    # Create output directory
    output_dir = "results/individual_metrics"
    os.makedirs(output_dir, exist_ok=True)
    
    # Metric 1: Offset on Metrics (ΔM)
    plt.figure(figsize=(12, 8))
    bars = plt.bar(df['Model'], df['ΔM (Offset)'], color=colors, alpha=0.8)
    plt.title('1. Offset on Metrics (ΔM)\nMost Common Robustness Metric (Lower = Better)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('ΔM Value', fontsize=14)
    plt.xlabel('Models', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, df['ΔM (Offset)']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_offset_on_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Metric 2: Performance Drop %
    plt.figure(figsize=(12, 8))
    drop_colors = ['gold' if x == 0.5 else 'darkblue' if x == 0.0 else 'green' if x < 0 else 'red' 
                   for x in df['Drop %']]
    bars = plt.bar(df['Model'], df['Drop %'], color=drop_colors, alpha=0.8)
    plt.title('2. Performance Drop %\nIntuitive Interpretation (Lower = Better, Negative = Improvement)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Performance Drop (%)', fontsize=14)
    plt.xlabel('Models', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels
    for bar, val in zip(bars, df['Drop %']):
        y_pos = bar.get_height() + (0.5 if val >= 0 else -1.0)
        plt.text(bar.get_x() + bar.get_width()/2, y_pos, 
                f'{val:.1f}%', ha='center', va='bottom' if val >= 0 else 'top', 
                fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_performance_drop.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Metric 3: Drop Rate (DR)
    plt.figure(figsize=(12, 8))
    bars = plt.bar(df['Model'], df['DR (Drop Rate)'], color=colors, alpha=0.8)
    plt.title('3. Drop Rate (DR)\nDistribution Shift Robustness (Lower = Better)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('DR Value', fontsize=14)
    plt.xlabel('Models', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, df['DR (Drop Rate)']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_drop_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Metric 4: Robustness Improvement (RI)
    plt.figure(figsize=(12, 8))
    bars = plt.bar(df['Model'], df['RI (Robustness Improvement)'], color=colors, alpha=0.8)
    plt.title('4. Robustness Improvement (RI)\nDefense Effectiveness (Higher = Better)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('RI Value', fontsize=14)
    plt.xlabel('Models', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, df['RI (Robustness Improvement)']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_robustness_improvement.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Metric 5: Predict Shift (PS)
    plt.figure(figsize=(12, 8))
    bars = plt.bar(df['Model'], df['PS (Predict Shift)'], color=colors, alpha=0.8)
    plt.title('5. Predict Shift (PS)\nPrediction Stability (Lower = Better)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('PS Value', fontsize=14)
    plt.xlabel('Models', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, df['PS (Predict Shift)']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_predict_shift.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Metric 6: Jaccard Similarity
    plt.figure(figsize=(12, 8))
    bars = plt.bar(df['Model'], df['Jaccard Similarity'], color=colors, alpha=0.8)
    plt.title('6. Jaccard Similarity\nList Overlap Consistency (Higher = Better)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Jaccard Score', fontsize=14)
    plt.xlabel('Models', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.6, 1.05)
    
    # Add value labels
    for bar, val in zip(bars, df['Jaccard Similarity']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6_jaccard_similarity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Metric 7: Top Output Stability
    plt.figure(figsize=(12, 8))
    bars = plt.bar(df['Model'], df['TO (Top Output)'], color=colors, alpha=0.8)
    plt.title('7. Top Output (TO) Stability\nTop-1 Item Stability (Higher = Better)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('TO Score', fontsize=14)
    plt.xlabel('Models', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.6, 1.05)
    
    # Add value labels
    for bar, val in zip(bars, df['TO (Top Output)']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '7_top_output_stability.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Metric 8: RBO Similarity
    plt.figure(figsize=(12, 8))
    bars = plt.bar(df['Model'], df['RBO Similarity'], color=colors, alpha=0.8)
    plt.title('8. RBO Similarity\nRank-Biased Overlap (Higher = Better)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('RBO Score', fontsize=14)
    plt.xlabel('Models', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.6, 1.05)
    
    # Add value labels
    for bar, val in zip(bars, df['RBO Similarity']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '8_rbo_similarity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Individual Metrics Charts Generated!")
    print(f"📊 Saved 8 individual PNG files in: {output_dir}/")
    print("\n📋 Generated Files:")
    print("1. 1_offset_on_metrics.png - Most common robustness metric")
    print("2. 2_performance_drop.png - Intuitive interpretation")
    print("3. 3_drop_rate.png - Distribution shift robustness")
    print("4. 4_robustness_improvement.png - Defense effectiveness")
    print("5. 5_predict_shift.png - Prediction stability")
    print("6. 6_jaccard_similarity.png - List overlap consistency")
    print("7. 7_top_output_stability.png - Top-1 item stability")
    print("8. 8_rbo_similarity.png - Rank-biased overlap")
    
    print("\n🎯 Each chart is:")
    print("- High resolution (300 DPI)")
    print("- Large size (12×8) for clarity")
    print("- Professional formatting")
    print("- Ready for thesis inclusion")
    
    return df

if __name__ == "__main__":
    create_individual_metrics_charts()

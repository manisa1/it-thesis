#!/usr/bin/env python3
"""
Comprehensive analysis of all DCCF robustness experiments.

This script analyzes results from all noise patterns (static, dynamic, burst, shift)
and generates thesis-ready summaries and visualizations.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


def load_experiment_results():
    """Load results from all experiments."""
    
    # Define all experiment configurations
    experiments = {
        # Core experiments
        "static_base": "runs/static_base/metrics.csv",
        "static_sol": "runs/static_sol/metrics.csv", 
        "dyn_base": "runs/dyn_base/metrics.csv",
        "dyn_sol": "runs/dyn_sol/metrics.csv",
        
        # New noise pattern experiments
        "burst_base": "runs/burst_base/metrics.csv",
        "burst_sol": "runs/burst_sol/metrics.csv",
        "shift_base": "runs/shift_base/metrics.csv", 
        "shift_sol": "runs/shift_sol/metrics.csv",
        
        # Additional static experiments
        "static_05_base": "runs/static_05_base/metrics.csv",
        "static_15_base": "runs/static_15_base/metrics.csv",
    }
    
    results = []
    missing_results = []
    
    for exp_name, metrics_path in experiments.items():
        if os.path.exists(metrics_path):
            try:
                df = pd.read_csv(metrics_path)
                recall = float(df["Recall@K"].iloc[0])
                ndcg = float(df["NDCG@K"].iloc[0])
                results.append({
                    "experiment": exp_name,
                    "Recall@20": recall,
                    "NDCG@20": ndcg
                })
                print(f" Loaded results for {exp_name}")
            except Exception as e:
                print(f" Error loading {exp_name}: {str(e)}")
                missing_results.append(exp_name)
        else:
            print(f" Missing results for {exp_name}: {metrics_path}")
            missing_results.append(exp_name)
    
    return pd.DataFrame(results), missing_results


def calculate_robustness_metrics(results_df):
    """Calculate robustness metrics for all noise patterns."""
    
    # Use static_base as the clean baseline
    baseline_recall = results_df[results_df['experiment'] == 'static_base']['Recall@20'].iloc[0]
    baseline_ndcg = results_df[results_df['experiment'] == 'static_base']['NDCG@20'].iloc[0]
    
    def robustness_drop(clean_val, noisy_val):
        return (clean_val - noisy_val) / clean_val if clean_val > 0 else 0.0
    
    # Calculate robustness for each noise pattern
    noise_patterns = ['dyn', 'burst', 'shift']
    robustness_results = []
    
    for pattern in noise_patterns:
        base_exp = f"{pattern}_base"
        sol_exp = f"{pattern}_sol"
        
        if base_exp in results_df['experiment'].values and sol_exp in results_df['experiment'].values:
            base_recall = results_df[results_df['experiment'] == base_exp]['Recall@20'].iloc[0]
            sol_recall = results_df[results_df['experiment'] == sol_exp]['Recall@20'].iloc[0]
            
            base_ndcg = results_df[results_df['experiment'] == base_exp]['NDCG@20'].iloc[0]
            sol_ndcg = results_df[results_df['experiment'] == sol_exp]['NDCG@20'].iloc[0]
            
            robustness_results.append({
                'Noise_Pattern': pattern.upper(),
                'Baseline_Recall_Drop': robustness_drop(baseline_recall, base_recall),
                'Solution_Recall_Drop': robustness_drop(baseline_recall, sol_recall),
                'Recall_Improvement': robustness_drop(baseline_recall, base_recall) - robustness_drop(baseline_recall, sol_recall),
                'Baseline_NDCG_Drop': robustness_drop(baseline_ndcg, base_ndcg),
                'Solution_NDCG_Drop': robustness_drop(baseline_ndcg, sol_ndcg),
                'NDCG_Improvement': robustness_drop(baseline_ndcg, base_ndcg) - robustness_drop(baseline_ndcg, sol_ndcg)
            })
    
    return pd.DataFrame(robustness_results)


def generate_thesis_summary(results_df, robustness_df):
    """Generate thesis-ready summary."""
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE THESIS RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Performance summary
    print("\n PERFORMANCE SUMMARY (All Experiments):")
    results_pivot = results_df.set_index('experiment')
    print(results_pivot.round(4))
    
    # Robustness analysis
    print(f"\n ROBUSTNESS ANALYSIS BY NOISE PATTERN:")
    print(robustness_df.round(4))
    
    # Key findings
    print(f"\n KEY THESIS FINDINGS:")
    
    print("\n1. DCCF's Vulnerability Across Noise Patterns:")
    for _, row in robustness_df.iterrows():
        pattern = row['Noise_Pattern']
        drop = row['Baseline_Recall_Drop']
        print(f"   - {pattern} noise: {drop:.1%} Recall@20 drop")
    
    print("\n2. Solution Effectiveness Across All Patterns:")
    for _, row in robustness_df.iterrows():
        pattern = row['Noise_Pattern']
        improvement = row['Recall_Improvement']
        print(f"   - {pattern} noise: {improvement:.1%} robustness improvement")
    
    # Find worst and best cases
    worst_pattern = robustness_df.loc[robustness_df['Baseline_Recall_Drop'].idxmax()]
    best_improvement = robustness_df.loc[robustness_df['Recall_Improvement'].idxmax()]
    
    print(f"\n3. Critical Insights:")
    print(f"   - Worst vulnerability: {worst_pattern['Noise_Pattern']} noise ({worst_pattern['Baseline_Recall_Drop']:.1%} drop)")
    print(f"   - Best improvement: {best_improvement['Noise_Pattern']} noise ({best_improvement['Recall_Improvement']:.1%} gain)")
    
    # Static condition check
    if 'static_sol' in results_df['experiment'].values:
        static_base = results_df[results_df['experiment'] == 'static_base']['Recall@20'].iloc[0]
        static_sol = results_df[results_df['experiment'] == 'static_sol']['Recall@20'].iloc[0]
        static_diff = (static_sol - static_base) / static_base
        print(f"   - Static performance impact: {static_diff:+.1%} (minimal)")
    
    return results_pivot, robustness_df


def save_results(results_df, robustness_df):
    """Save all results to CSV files."""
    
    os.makedirs("runs", exist_ok=True)
    
    # Save comprehensive summary
    results_pivot = results_df.set_index('experiment')
    results_pivot.to_csv("runs/comprehensive_summary.csv")
    print(f"\n Comprehensive summary saved to: runs/comprehensive_summary.csv")
    
    # Save robustness analysis
    robustness_df.to_csv("runs/comprehensive_robustness.csv", index=False)
    print(f" Robustness analysis saved to: runs/comprehensive_robustness.csv")
    
    # Create thesis-ready table
    thesis_table = []
    for _, row in robustness_df.iterrows():
        pattern = row['Noise_Pattern']
        baseline_drop = f"{row['Baseline_Recall_Drop']:.1%}"
        solution_drop = f"{row['Solution_Recall_Drop']:.1%}"
        improvement = f"{row['Recall_Improvement']:.1%}"
        
        thesis_table.append({
            'Noise Pattern': pattern,
            'DCCF Baseline Drop': baseline_drop,
            'With Solution Drop': solution_drop,
            'Improvement': improvement
        })
    
    thesis_df = pd.DataFrame(thesis_table)
    thesis_df.to_csv("runs/thesis_table.csv", index=False)
    print(f" Thesis table saved to: runs/thesis_table.csv")


def main():
    """Main analysis function."""
    
    print(" Starting Comprehensive Results Analysis")
    print(f" {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load results
    results_df, missing = load_experiment_results()
    
    if missing:
        print(f"\n  Warning: Missing results for {missing}")
        print("Some experiments may not have completed successfully.")
    
    if len(results_df) < 4:
        print(f"\n Error: Only {len(results_df)} experiments found. Need at least 4 core experiments.")
        return
    
    # Calculate robustness metrics
    robustness_df = calculate_robustness_metrics(results_df)
    
    # Generate summary
    results_pivot, robustness_summary = generate_thesis_summary(results_df, robustness_df)
    
    # Save results
    save_results(results_df, robustness_df)
    
    print(f"\n COMPREHENSIVE ANALYSIS COMPLETED!")
    print(f" All results available in: runs/ directory")
    print(f" Ready for thesis presentation and defense!")


if __name__ == "__main__":
    main()

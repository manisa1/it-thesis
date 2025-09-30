#!/usr/bin/env python3
"""
Results Viewer for DCCF Robustness Study

This script provides an easy way to view and analyze experimental results
for thesis presentation and defense.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


def load_experiment_result(experiment_dir):
    """Load results from a single experiment directory."""
    metrics_file = Path(experiment_dir) / "metrics.csv"
    if metrics_file.exists():
        try:
            df = pd.read_csv(metrics_file)
            return {
                'recall': float(df['Recall@K'].iloc[0]),
                'ndcg': float(df['NDCG@K'].iloc[0]),
                'k': int(df['K'].iloc[0])
            }
        except Exception as e:
            print(f"Error reading {metrics_file}: {e}")
            return None
    return None


def display_results_table():
    """Display all available results in a formatted table."""
    
    # Define experiments and their descriptions
    experiments = {
        'static_base': 'Static Baseline (Clean DCCF)',
        'static_sol': 'Static Solution (Clean + Denoiser)',
        'dyn_base': 'Dynamic Baseline (DCCF + Dynamic Noise)',
        'dyn_sol': 'Dynamic Solution (DCCF + Dynamic Noise + Denoiser)',
        'burst_base': 'Burst Baseline (DCCF + Burst Noise)',
        'burst_sol': 'Burst Solution (DCCF + Burst Noise + Denoiser)',
        'shift_base': 'Shift Baseline (DCCF + Shift Noise)',
        'shift_sol': 'Shift Solution (DCCF + Shift Noise + Denoiser)'
    }
    
    print("🎓 DCCF ROBUSTNESS STUDY - EXPERIMENTAL RESULTS")
    print("=" * 80)
    print()
    
    results = []
    available_experiments = []
    
    # Load all available results
    for exp_name, description in experiments.items():
        exp_dir = f"runs/{exp_name}"
        result = load_experiment_result(exp_dir)
        
        if result:
            results.append({
                'Experiment': exp_name,
                'Description': description,
                'Recall@20': f"{result['recall']:.4f}",
                'NDCG@20': f"{result['ndcg']:.4f}",
                'Status': '✅ Complete'
            })
            available_experiments.append(exp_name)
        else:
            results.append({
                'Experiment': exp_name,
                'Description': description,
                'Recall@20': 'N/A',
                'NDCG@20': 'N/A',
                'Status': '❌ Not Run'
            })
    
    # Display results table
    if results:
        df = pd.DataFrame(results)
        print("📊 EXPERIMENTAL RESULTS SUMMARY:")
        print("-" * 80)
        
        # Format the table nicely
        for _, row in df.iterrows():
            status_icon = "✅" if row['Status'] == '✅ Complete' else "❌"
            print(f"{status_icon} {row['Experiment']:12} | {row['Recall@20']:8} | {row['NDCG@20']:8} | {row['Description']}")
        
        print("-" * 80)
        print(f"Completed: {len(available_experiments)}/{len(experiments)} experiments")
        print()
    
    return available_experiments, results


def analyze_robustness(available_experiments):
    """Analyze robustness across different noise patterns."""
    
    if len(available_experiments) < 2:
        print("⚠️  Need at least 2 experiments for robustness analysis")
        return
    
    print("🔍 ROBUSTNESS ANALYSIS:")
    print("-" * 50)
    
    # Load baseline (clean) performance
    baseline_result = load_experiment_result("runs/static_base")
    if not baseline_result:
        print("❌ Static baseline not found - cannot compute robustness")
        return
    
    baseline_recall = baseline_result['recall']
    print(f"📊 Baseline Performance (Static): Recall@20 = {baseline_recall:.4f}")
    print()
    
    # Analyze each noise pattern
    noise_patterns = [
        ('dyn', 'Dynamic Noise'),
        ('burst', 'Burst Noise'),
        ('shift', 'Shift Noise')
    ]
    
    for pattern, pattern_name in noise_patterns:
        base_exp = f"{pattern}_base"
        sol_exp = f"{pattern}_sol"
        
        if base_exp in available_experiments and sol_exp in available_experiments:
            base_result = load_experiment_result(f"runs/{base_exp}")
            sol_result = load_experiment_result(f"runs/{sol_exp}")
            
            if base_result and sol_result:
                base_recall = base_result['recall']
                sol_recall = sol_result['recall']
                
                # Calculate robustness drops
                base_drop = (baseline_recall - base_recall) / baseline_recall * 100
                sol_drop = (baseline_recall - sol_recall) / baseline_recall * 100
                improvement = base_drop - sol_drop
                
                print(f"🎯 {pattern_name}:")
                print(f"   Baseline Drop:  {base_drop:6.1f}% (Recall: {base_recall:.4f})")
                print(f"   Solution Drop:  {sol_drop:6.1f}% (Recall: {sol_recall:.4f})")
                print(f"   Improvement:    {improvement:6.1f}% {'✅' if improvement > 0 else '❌'}")
                print()


def generate_thesis_summary():
    """Generate a thesis-ready summary."""
    
    print("📝 THESIS SUMMARY:")
    print("-" * 50)
    
    available_experiments, results = display_results_table()
    
    if 'burst_base' in available_experiments and 'burst_sol' in available_experiments:
        burst_base = load_experiment_result("runs/burst_base")
        burst_sol = load_experiment_result("runs/burst_sol")
        
        print("🔥 BURST NOISE FINDINGS:")
        print(f"   DCCF Baseline:     Recall@20 = {burst_base['recall']:.4f}")
        print(f"   With Our Solution: Recall@20 = {burst_sol['recall']:.4f}")
        
        if burst_base['recall'] > burst_sol['recall']:
            diff = burst_base['recall'] - burst_sol['recall']
            print(f"   ⚠️  Solution shows {diff:.4f} lower performance")
            print("   💡 This suggests burst noise may require different approach")
        else:
            diff = burst_sol['recall'] - burst_base['recall']
            print(f"   ✅ Solution improves performance by {diff:.4f}")
        print()
    
    print("📋 FOR THESIS DEFENSE:")
    print("- Results are saved in runs/ directory")
    print("- Each metrics.csv contains Recall@20 and NDCG@20")
    print("- Use this script to generate presentation tables")
    print("- All experiments are reproducible with documented commands")


def main():
    """Main function to display all results."""
    
    print("🚀 DCCF Robustness Study - Results Viewer")
    print("=" * 60)
    print()
    
    # Check if runs directory exists
    if not os.path.exists("runs"):
        print("❌ No results found. Run experiments first:")
        print("   python run_train_experiments.py")
        return
    
    # Display results
    available_experiments, _ = display_results_table()
    
    if available_experiments:
        print()
        analyze_robustness(available_experiments)
        print()
        generate_thesis_summary()
    else:
        print("❌ No completed experiments found.")
        print("Run experiments first: python run_train_experiments.py")


if __name__ == "__main__":
    main()

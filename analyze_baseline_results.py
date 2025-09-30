#!/usr/bin/env python3
"""
Analyze baseline comparison results.

This script compares the performance of different baseline models 
(LightGCN, SimGCL, NGCF, SGL) with DCCF under various noise conditions.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_baseline_results(base_dir="runs/baselines"):
    """Load results from all baseline experiments."""
    results = []
    
    for model_dir in Path(base_dir).glob("*"):
        if model_dir.is_dir():
            metrics_file = model_dir / "metrics.csv"
            if metrics_file.exists():
                # Parse model and experiment from directory name
                dir_name = model_dir.name
                parts = dir_name.split("_", 1)
                if len(parts) == 2:
                    model_type, experiment = parts
                    
                    # Load metrics
                    df = pd.read_csv(metrics_file)
                    final_metrics = df.iloc[-1]
                    
                    results.append({
                        'model': model_type,
                        'experiment': experiment,
                        'recall@20': final_metrics.get('recall@20', 0.0),
                        'ndcg@20': final_metrics.get('ndcg@20', 0.0),
                        'final_epoch': final_metrics.get('epoch', 15)
                    })
    
    return pd.DataFrame(results)


def load_dccf_results(runs_dir="runs"):
    """Load DCCF results for comparison."""
    dccf_results = []
    
    # Map experiment directories to experiment names
    exp_mapping = {
        'static_base': 'static_baseline',
        'static_sol': 'static_solution', 
        'dyn_base': 'dynamic_baseline',
        'dyn_sol': 'dynamic_solution',
        'burst_base': 'burst_baseline',
        'shift_base': 'shift_baseline'
    }
    
    for exp_dir, exp_name in exp_mapping.items():
        metrics_file = Path(runs_dir) / exp_dir / "metrics.csv"
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            final_metrics = df.iloc[-1]
            
            dccf_results.append({
                'model': 'dccf',
                'experiment': exp_name,
                'recall@20': final_metrics.get('recall@20', final_metrics.get('Recall@20', 0.0)),
                'ndcg@20': final_metrics.get('ndcg@20', final_metrics.get('NDCG@20', 0.0)),
                'final_epoch': final_metrics.get('epoch', 15)
            })
    
    return pd.DataFrame(dccf_results)


def create_comparison_table(baseline_df, dccf_df):
    """Create comprehensive comparison table."""
    # Combine results
    all_results = pd.concat([baseline_df, dccf_df], ignore_index=True)
    
    # Pivot table for easier comparison
    recall_table = all_results.pivot(index='experiment', columns='model', values='recall@20')
    ndcg_table = all_results.pivot(index='experiment', columns='model', values='ndcg@20')
    
    return recall_table, ndcg_table


def plot_performance_comparison(recall_table, ndcg_table, output_dir="runs/baselines"):
    """Create performance comparison plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Recall comparison
    recall_table.plot(kind='bar', ax=ax1, rot=45)
    ax1.set_title('Recall@20 Comparison Across Models')
    ax1.set_ylabel('Recall@20')
    ax1.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # NDCG comparison
    ndcg_table.plot(kind='bar', ax=ax2, rot=45)
    ax2.set_title('NDCG@20 Comparison Across Models')
    ax2.set_ylabel('NDCG@20')
    ax2.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'baseline_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def calculate_robustness_metrics(recall_table, ndcg_table):
    """Calculate robustness metrics for each model."""
    robustness_results = []
    
    # Define baseline experiments (no noise)
    clean_experiments = ['static_baseline', 'static_solution']
    noisy_experiments = ['dynamic_baseline', 'dynamic_solution', 'burst_baseline', 'shift_baseline']
    
    for model in recall_table.columns:
        if model in recall_table.columns:
            # Get clean performance (average of static experiments)
            clean_recall = recall_table.loc[clean_experiments, model].mean()
            clean_ndcg = ndcg_table.loc[clean_experiments, model].mean()
            
            # Calculate robustness for each noisy experiment
            for exp in noisy_experiments:
                if exp in recall_table.index:
                    noisy_recall = recall_table.loc[exp, model]
                    noisy_ndcg = ndcg_table.loc[exp, model]
                    
                    # Robustness drop (lower is better)
                    recall_drop = (clean_recall - noisy_recall) / clean_recall * 100
                    ndcg_drop = (clean_ndcg - noisy_ndcg) / clean_ndcg * 100
                    
                    robustness_results.append({
                        'model': model,
                        'experiment': exp,
                        'clean_recall': clean_recall,
                        'noisy_recall': noisy_recall,
                        'recall_drop_%': recall_drop,
                        'clean_ndcg': clean_ndcg,
                        'noisy_ndcg': noisy_ndcg,
                        'ndcg_drop_%': ndcg_drop
                    })
    
    return pd.DataFrame(robustness_results)


def generate_thesis_table(recall_table, ndcg_table, robustness_df, output_dir="runs/baselines"):
    """Generate thesis-ready comparison table."""
    
    # Create summary table
    summary_data = []
    
    for model in recall_table.columns:
        if model in recall_table.columns:
            # Static performance (clean)
            static_recall = recall_table.loc['static_baseline', model] if 'static_baseline' in recall_table.index else 0
            static_ndcg = ndcg_table.loc['static_baseline', model] if 'static_baseline' in ndcg_table.index else 0
            
            # Dynamic performance
            dynamic_recall = recall_table.loc['dynamic_baseline', model] if 'dynamic_baseline' in recall_table.index else 0
            dynamic_ndcg = ndcg_table.loc['dynamic_baseline', model] if 'dynamic_baseline' in ndcg_table.index else 0
            
            # Calculate drops
            recall_drop = (static_recall - dynamic_recall) / static_recall * 100 if static_recall > 0 else 0
            ndcg_drop = (static_ndcg - dynamic_ndcg) / static_ndcg * 100 if static_ndcg > 0 else 0
            
            summary_data.append({
                'Model': model.upper(),
                'Static Recall@20': f"{static_recall:.4f}",
                'Dynamic Recall@20': f"{dynamic_recall:.4f}",
                'Recall Drop (%)': f"{recall_drop:.1f}%",
                'Static NDCG@20': f"{static_ndcg:.4f}",
                'Dynamic NDCG@20': f"{dynamic_ndcg:.4f}",
                'NDCG Drop (%)': f"{ndcg_drop:.1f}%"
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save thesis table
    summary_df.to_csv(os.path.join(output_dir, 'thesis_comparison_table.csv'), index=False)
    
    # Create LaTeX table
    latex_table = summary_df.to_latex(index=False, escape=False)
    with open(os.path.join(output_dir, 'thesis_comparison_table.tex'), 'w') as f:
        f.write(latex_table)
    
    return summary_df


def main():
    """Main analysis function."""
    print("🔍 Analyzing baseline comparison results...")
    
    # Load results
    baseline_df = load_baseline_results()
    dccf_df = load_dccf_results()
    
    if baseline_df.empty:
        print("❌ No baseline results found. Run baseline experiments first.")
        return
    
    if dccf_df.empty:
        print("⚠️  No DCCF results found. Comparison will be baseline-only.")
    
    print(f"📊 Loaded results:")
    print(f"  - Baseline experiments: {len(baseline_df)}")
    print(f"  - DCCF experiments: {len(dccf_df)}")
    
    # Create comparison tables
    recall_table, ndcg_table = create_comparison_table(baseline_df, dccf_df)
    
    print("\n📈 Performance Comparison (Recall@20):")
    print(recall_table.round(4))
    
    print("\n📈 Performance Comparison (NDCG@20):")
    print(ndcg_table.round(4))
    
    # Calculate robustness metrics
    robustness_df = calculate_robustness_metrics(recall_table, ndcg_table)
    
    print("\n🛡️  Robustness Analysis (Performance Drop %):")
    robustness_summary = robustness_df.pivot(index='experiment', columns='model', values='recall_drop_%')
    print(robustness_summary.round(1))
    
    # Create output directory
    output_dir = "runs/baselines"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    plot_performance_comparison(recall_table, ndcg_table, output_dir)
    
    # Generate thesis table
    thesis_table = generate_thesis_table(recall_table, ndcg_table, robustness_df, output_dir)
    
    print("\n📋 Thesis Comparison Table:")
    print(thesis_table.to_string(index=False))
    
    # Save all results
    recall_table.to_csv(os.path.join(output_dir, 'recall_comparison.csv'))
    ndcg_table.to_csv(os.path.join(output_dir, 'ndcg_comparison.csv'))
    robustness_df.to_csv(os.path.join(output_dir, 'robustness_analysis.csv'), index=False)
    
    print(f"\n💾 Results saved to: {output_dir}/")
    print("📊 Files generated:")
    print("  - thesis_comparison_table.csv (main results)")
    print("  - baseline_comparison.png (performance plots)")
    print("  - robustness_analysis.csv (detailed robustness metrics)")
    
    # Key insights
    print("\n🎯 KEY INSIGHTS:")
    
    # Find best performing model
    if not recall_table.empty:
        static_performance = recall_table.loc['static_baseline'] if 'static_baseline' in recall_table.index else recall_table.iloc[0]
        best_model = static_performance.idxmax()
        best_score = static_performance.max()
        print(f"  - Best overall model: {best_model.upper()} (Recall@20: {best_score:.4f})")
        
        # Find most robust model
        if not robustness_df.empty:
            avg_robustness = robustness_df.groupby('model')['recall_drop_%'].mean()
            most_robust = avg_robustness.idxmin()
            robustness_score = avg_robustness.min()
            print(f"  - Most robust model: {most_robust.upper()} (Avg drop: {robustness_score:.1f}%)")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Comprehensive robustness analysis using established metrics from literature.

Based on:
1. "Robust Recommender System: A Survey and Future Directions" (2023)
2. "Towards Robust Recommendation: A Review and an Adversarial Robustness Evaluation Library" (2024)

This script implements the standard robustness evaluation framework used in academic research.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.append('src')

from evaluation.robustness_metrics import (
    comprehensive_robustness_analysis,
    offset_on_metrics,
    robustness_improvement,
    generate_robustness_table
)


def load_experiment_results(runs_dir: str = "runs") -> Dict[str, Dict[str, float]]:
    """Load results from all experiments."""
    results = {}
    
    # Define experiment mapping
    experiment_dirs = {
        'static_baseline': 'static_base',
        'static_solution': 'static_sol', 
        'dynamic_baseline': 'dyn_base',
        'dynamic_solution': 'dyn_sol',
        'burst_baseline': 'burst_base',
        'burst_solution': 'burst_sol',
        'shift_baseline': 'shift_base',
        'shift_solution': 'shift_sol'
    }
    
    for exp_name, dir_name in experiment_dirs.items():
        metrics_file = Path(runs_dir) / dir_name / "metrics.csv"
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            if not df.empty:
                # Get final epoch results
                final_results = df.iloc[-1].to_dict()
                
                # Standardize metric names
                standardized = {}
                for key, value in final_results.items():
                    if 'recall' in key.lower():
                        standardized['recall@20'] = value
                    elif 'ndcg' in key.lower():
                        standardized['ndcg@20'] = value
                    elif key not in ['epoch']:
                        standardized[key] = value
                
                results[exp_name] = standardized
                print(f"✅ Loaded {exp_name}: {len(standardized)} metrics")
        else:
            print(f"⚠️  Missing: {metrics_file}")
    
    return results


def load_baseline_results(baseline_dir: str = "runs/baselines") -> Dict[str, Dict[str, float]]:
    """Load baseline model results."""
    baseline_results = {}
    
    if not os.path.exists(baseline_dir):
        print(f"⚠️  Baseline directory not found: {baseline_dir}")
        return baseline_results
    
    for model_dir in Path(baseline_dir).glob("*"):
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
                    if not df.empty:
                        final_results = df.iloc[-1].to_dict()
                        
                        # Create combined key
                        key = f"{model_type}_{experiment}"
                        baseline_results[key] = {
                            'recall@20': final_results.get('recall@20', 0.0),
                            'ndcg@20': final_results.get('ndcg@20', 0.0)
                        }
    
    print(f"✅ Loaded {len(baseline_results)} baseline results")
    return baseline_results


def calculate_established_robustness_metrics(results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate robustness metrics using established methods from literature.
    
    Following the standard evaluation framework:
    1. Offset on Metrics (ΔM) - most common
    2. Performance Drop % - intuitive interpretation  
    3. Robustness Improvement (RI) - for defense evaluation
    """
    robustness_analysis = {}
    
    # Define clean (baseline) conditions
    clean_experiments = ['static_baseline']
    
    # Define noise conditions and their clean counterparts
    noise_conditions = {
        'dynamic_baseline': 'static_baseline',
        'burst_baseline': 'static_baseline', 
        'shift_baseline': 'static_baseline'
    }
    
    # Define solution pairs for RI calculation
    solution_pairs = {
        'static_solution': ('static_baseline', 'static_baseline'),  # No attack case
        'dynamic_solution': ('static_baseline', 'dynamic_baseline'),
        'burst_solution': ('static_baseline', 'burst_baseline'),
        'shift_solution': ('static_baseline', 'shift_baseline')
    }
    
    # 1. Calculate Offset on Metrics for noise conditions
    for noise_exp, clean_exp in noise_conditions.items():
        if noise_exp in results and clean_exp in results:
            clean_metrics = results[clean_exp]
            noisy_metrics = results[noise_exp]
            
            # Standard offset calculation
            offsets = offset_on_metrics(clean_metrics, noisy_metrics)
            
            # Performance drops (more intuitive)
            drops = {}
            for metric in clean_metrics.keys():
                if metric in noisy_metrics:
                    clean_val = clean_metrics[metric]
                    noisy_val = noisy_metrics[metric]
                    
                    if clean_val != 0:
                        drop_pct = (clean_val - noisy_val) / clean_val * 100
                    else:
                        drop_pct = 0.0
                    
                    drops[f"{metric}_drop_%"] = drop_pct
            
            # Combine metrics
            analysis = {**offsets, **drops}
            analysis['noise_type'] = noise_exp.replace('_baseline', '')
            
            robustness_analysis[noise_exp] = analysis
    
    # 2. Calculate Robustness Improvement for solutions
    for solution_exp, (clean_exp, attack_exp) in solution_pairs.items():
        if all(exp in results for exp in [solution_exp, clean_exp, attack_exp]):
            clean_metrics = results[clean_exp]
            attack_metrics = results[attack_exp] 
            defense_metrics = results[solution_exp]
            
            # Calculate RI
            ri_metrics = robustness_improvement(clean_metrics, attack_metrics, defense_metrics)
            
            # Add solution effectiveness
            effectiveness = {}
            for metric in clean_metrics.keys():
                if metric in defense_metrics and metric in attack_metrics:
                    defense_val = defense_metrics[metric]
                    attack_val = attack_metrics[metric]
                    
                    improvement = defense_val - attack_val
                    effectiveness[f"{metric}_improvement"] = improvement
            
            analysis = {**ri_metrics, **effectiveness}
            analysis['solution_type'] = solution_exp.replace('_solution', '')
            
            robustness_analysis[solution_exp] = analysis
    
    return robustness_analysis


def create_robustness_comparison_table(dccf_results: Dict, baseline_results: Dict) -> pd.DataFrame:
    """Create comprehensive robustness comparison table."""
    
    # Prepare data for comparison table
    comparison_data = []
    
    # DCCF results
    if 'static_baseline' in dccf_results and 'dynamic_baseline' in dccf_results:
        static_recall = dccf_results['static_baseline'].get('recall@20', 0)
        dynamic_recall = dccf_results['dynamic_baseline'].get('recall@20', 0)
        static_ndcg = dccf_results['static_baseline'].get('ndcg@20', 0)
        dynamic_ndcg = dccf_results['dynamic_baseline'].get('ndcg@20', 0)
        
        recall_drop = (static_recall - dynamic_recall) / static_recall * 100 if static_recall > 0 else 0
        ndcg_drop = (static_ndcg - dynamic_ndcg) / static_ndcg * 100 if static_ndcg > 0 else 0
        
        comparison_data.append({
            'Model': 'DCCF (Ours)',
            'Static Recall@20': f"{static_recall:.4f}",
            'Dynamic Recall@20': f"{dynamic_recall:.4f}", 
            'Recall Drop (%)': f"{recall_drop:.1f}%",
            'Static NDCG@20': f"{static_ndcg:.4f}",
            'Dynamic NDCG@20': f"{dynamic_ndcg:.4f}",
            'NDCG Drop (%)': f"{ndcg_drop:.1f}%",
            'Robustness Rank': 0  # Will be calculated later
        })
    
    # Baseline results
    baseline_models = set()
    for key in baseline_results.keys():
        model = key.split('_')[0]
        baseline_models.add(model)
    
    for model in baseline_models:
        static_key = f"{model}_static_baseline"
        dynamic_key = f"{model}_dynamic_baseline"
        
        if static_key in baseline_results and dynamic_key in baseline_results:
            static_recall = baseline_results[static_key].get('recall@20', 0)
            dynamic_recall = baseline_results[dynamic_key].get('recall@20', 0)
            static_ndcg = baseline_results[static_key].get('ndcg@20', 0)
            dynamic_ndcg = baseline_results[dynamic_key].get('ndcg@20', 0)
            
            recall_drop = (static_recall - dynamic_recall) / static_recall * 100 if static_recall > 0 else 0
            ndcg_drop = (static_ndcg - dynamic_ndcg) / static_ndcg * 100 if static_ndcg > 0 else 0
            
            comparison_data.append({
                'Model': model.upper(),
                'Static Recall@20': f"{static_recall:.4f}",
                'Dynamic Recall@20': f"{dynamic_recall:.4f}",
                'Recall Drop (%)': f"{recall_drop:.1f}%", 
                'Static NDCG@20': f"{static_ndcg:.4f}",
                'Dynamic NDCG@20': f"{dynamic_ndcg:.4f}",
                'NDCG Drop (%)': f"{ndcg_drop:.1f}%",
                'Robustness Rank': 0
            })
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Calculate robustness ranking (lower drop = better robustness)
    if not df.empty:
        df['recall_drop_numeric'] = df['Recall Drop (%)'].str.replace('%', '').astype(float)
        df['Robustness Rank'] = df['recall_drop_numeric'].rank(method='min').astype(int)
        df = df.drop('recall_drop_numeric', axis=1)
        df = df.sort_values('Robustness Rank')
    
    return df


def create_robustness_visualizations(robustness_metrics: Dict, output_dir: str):
    """Create robustness analysis visualizations."""
    
    # 1. Performance Drop Comparison
    plt.figure(figsize=(12, 6))
    
    noise_types = []
    recall_drops = []
    ndcg_drops = []
    
    for exp_name, metrics in robustness_metrics.items():
        if 'baseline' in exp_name and 'noise_type' in metrics:
            noise_types.append(metrics['noise_type'].title())
            recall_drops.append(metrics.get('recall@20_drop_%', 0))
            ndcg_drops.append(metrics.get('ndcg@20_drop_%', 0))
    
    x = np.arange(len(noise_types))
    width = 0.35
    
    plt.bar(x - width/2, recall_drops, width, label='Recall@20 Drop', alpha=0.8)
    plt.bar(x + width/2, ndcg_drops, width, label='NDCG@20 Drop', alpha=0.8)
    
    plt.xlabel('Noise Pattern')
    plt.ylabel('Performance Drop (%)')
    plt.title('DCCF Robustness Analysis: Performance Drops Under Different Noise Patterns')
    plt.xticks(x, noise_types)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dccf_robustness_drops.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Offset on Metrics Heatmap
    plt.figure(figsize=(10, 6))
    
    offset_data = []
    experiments = []
    
    for exp_name, metrics in robustness_metrics.items():
        if 'baseline' in exp_name:
            experiments.append(exp_name.replace('_baseline', '').title())
            row_data = []
            for metric in ['Δrecall@20', 'Δndcg@20']:
                row_data.append(metrics.get(metric, 0))
            offset_data.append(row_data)
    
    if offset_data:
        offset_df = pd.DataFrame(offset_data, 
                               index=experiments,
                               columns=['Δ Recall@20', 'Δ NDCG@20'])
        
        sns.heatmap(offset_df, annot=True, fmt='.3f', cmap='Reds', 
                   cbar_kws={'label': 'Offset on Metrics (ΔM)'})
        plt.title('Robustness Analysis: Offset on Metrics')
        plt.ylabel('Noise Pattern')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'offset_metrics_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main robustness analysis function."""
    print("🔍 Comprehensive Robustness Analysis Using Established Metrics")
    print("=" * 70)
    
    # Load results
    print("\n📊 Loading experimental results...")
    dccf_results = load_experiment_results()
    baseline_results = load_baseline_results()
    
    if not dccf_results:
        print("❌ No DCCF results found. Run experiments first.")
        return
    
    print(f"✅ Loaded {len(dccf_results)} DCCF experiments")
    print(f"✅ Loaded {len(baseline_results)} baseline experiments")
    
    # Calculate robustness metrics
    print("\n🛡️  Calculating established robustness metrics...")
    robustness_metrics = calculate_established_robustness_metrics(dccf_results)
    
    # Create output directory
    output_dir = "runs/robustness_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate comprehensive comparison table
    print("\n📋 Generating robustness comparison table...")
    comparison_table = create_robustness_comparison_table(dccf_results, baseline_results)
    
    if not comparison_table.empty:
        print("\n🏆 ROBUSTNESS COMPARISON TABLE")
        print("=" * 50)
        print(comparison_table.to_string(index=False))
        
        # Save table
        comparison_table.to_csv(os.path.join(output_dir, 'robustness_comparison_table.csv'), index=False)
        
        # Create LaTeX table
        latex_table = comparison_table.to_latex(index=False, escape=False)
        with open(os.path.join(output_dir, 'robustness_comparison_table.tex'), 'w') as f:
            f.write(latex_table)
    
    # Generate detailed robustness metrics table
    print("\n📊 Generating detailed robustness metrics...")
    detailed_metrics = generate_robustness_table(robustness_metrics, 
                                               os.path.join(output_dir, 'detailed_robustness_metrics.csv'))
    
    print("\n📈 DETAILED ROBUSTNESS METRICS")
    print("=" * 40)
    print(detailed_metrics.to_string())
    
    # Create visualizations
    print("\n📊 Creating robustness visualizations...")
    create_robustness_visualizations(robustness_metrics, output_dir)
    
    # Key insights
    print("\n🎯 KEY ROBUSTNESS INSIGHTS")
    print("=" * 30)
    
    # Find most robust model
    if not comparison_table.empty:
        best_robust = comparison_table.iloc[0]  # Already sorted by robustness rank
        print(f"🥇 Most Robust Model: {best_robust['Model']}")
        print(f"   - Recall Drop: {best_robust['Recall Drop (%)']}")
        print(f"   - NDCG Drop: {best_robust['NDCG Drop (%)']}")
    
    # DCCF specific insights
    if 'dynamic_baseline' in robustness_metrics:
        dyn_metrics = robustness_metrics['dynamic_baseline']
        recall_drop = dyn_metrics.get('recall@20_drop_%', 0)
        print(f"\n🔍 DCCF Dynamic Noise Analysis:")
        print(f"   - Performance drop under dynamic noise: {recall_drop:.1f}%")
        
        if recall_drop < 15:
            print("   ✅ DCCF shows good robustness (< 15% drop)")
        elif recall_drop < 25:
            print("   ⚠️  DCCF shows moderate robustness (15-25% drop)")
        else:
            print("   ❌ DCCF shows poor robustness (> 25% drop)")
    
    # Solution effectiveness
    if 'dynamic_solution' in robustness_metrics:
        sol_metrics = robustness_metrics['dynamic_solution']
        ri_recall = sol_metrics.get('RI_recall@20', 0)
        print(f"\n💡 Solution Effectiveness:")
        print(f"   - Robustness Improvement (RI): {ri_recall:.3f}")
        
        if ri_recall > 0.1:
            print("   ✅ Solution provides significant improvement")
        elif ri_recall > 0.05:
            print("   ⚠️  Solution provides moderate improvement") 
        else:
            print("   ❌ Solution provides minimal improvement")
    
    # Save summary
    summary = {
        'total_experiments': len(dccf_results),
        'baseline_models': len(set(k.split('_')[0] for k in baseline_results.keys())),
        'robustness_metrics_calculated': len(robustness_metrics),
        'output_directory': output_dir
    }
    
    print(f"\n💾 Analysis complete! Results saved to: {output_dir}/")
    print("📁 Generated files:")
    print("   - robustness_comparison_table.csv (main results)")
    print("   - robustness_comparison_table.tex (LaTeX table)")
    print("   - detailed_robustness_metrics.csv (all metrics)")
    print("   - dccf_robustness_drops.png (performance visualization)")
    print("   - offset_metrics_heatmap.png (offset analysis)")
    
    print("\n✅ Comprehensive robustness analysis completed using established metrics!")
    print("📚 References:")
    print("   - Robust Recommender System: A Survey and Future Directions (2023)")
    print("   - Towards Robust Recommendation: A Review and an Adversarial Robustness Evaluation Library (2024)")


if __name__ == "__main__":
    main()

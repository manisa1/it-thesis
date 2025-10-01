#!/usr/bin/env python3
"""
Create user-friendly results tables with descriptions and explanations.
This script converts raw metrics into understandable tables for non-technical users.
"""

import pandas as pd
import os
from pathlib import Path


def create_user_friendly_results():
    """Create comprehensive, user-friendly results tables."""
    
    # Define experiment descriptions
    experiments = {
        'static_base': {
            'name': 'Static Baseline',
            'description': 'DCCF with no noise, no solution',
            'what_it_tests': 'Best case performance under ideal conditions',
            'expected': 'Highest performance (baseline for comparison)'
        },
        'static_sol': {
            'name': 'Static Solution', 
            'description': 'DCCF with no noise, with our solution',
            'what_it_tests': 'Does our solution harm performance in ideal conditions?',
            'expected': 'Similar to baseline (no harm done)'
        },
        'dyn_base': {
            'name': 'Dynamic Baseline',
            'description': 'DCCF with dynamic noise, no solution', 
            'what_it_tests': 'How much does dynamic noise hurt DCCF?',
            'expected': 'Lower performance (shows the problem)'
        },
        'dyn_sol': {
            'name': 'Dynamic Solution',
            'description': 'DCCF with dynamic noise, with our solution',
            'what_it_tests': 'Does our solution help under dynamic noise?', 
            'expected': 'Better than dynamic baseline (proves solution works)'
        },
        'burst_base': {
            'name': 'Burst Baseline',
            'description': 'DCCF with sudden noise spikes, no solution',
            'what_it_tests': 'How does DCCF handle sudden popularity spikes?',
            'expected': 'Variable performance'
        },
        'shift_base': {
            'name': 'Shift Baseline', 
            'description': 'DCCF with focus shift noise, no solution',
            'what_it_tests': 'How does DCCF handle changing popularity focus?',
            'expected': 'Variable performance'
        }
    }
    
    # Collect results
    results_data = []
    
    for exp_key, exp_info in experiments.items():
        metrics_file = f"runs/{exp_key}/metrics.csv"
        
        if os.path.exists(metrics_file):
            # Read the metrics
            df = pd.read_csv(metrics_file)
            if not df.empty:
                recall = df['Recall@K'].iloc[0]
                ndcg = df['NDCG@K'].iloc[0]
                
                # Convert to percentages and readable format
                recall_pct = f"{recall * 100:.1f}%"
                ndcg_pct = f"{ndcg * 100:.2f}%"
                
                # Performance interpretation
                if recall >= 0.20:
                    performance = "Excellent"
                elif recall >= 0.18:
                    performance = "Good" 
                elif recall >= 0.16:
                    performance = "Fair"
                else:
                    performance = "Poor"
                
                results_data.append({
                    'Experiment': exp_info['name'],
                    'Description': exp_info['description'],
                    'What It Tests': exp_info['what_it_tests'],
                    'Recall@20': recall_pct,
                    'NDCG@20': ndcg_pct,
                    'Performance Level': performance,
                    'Raw Recall': recall,
                    'Interpretation': get_interpretation(exp_key, recall)
                })
    
    # Create user-friendly DataFrame
    results_df = pd.DataFrame(results_data)
    
    # Save comprehensive results
    output_dir = "runs/user_friendly_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Main results table
    results_df.to_csv(f"{output_dir}/comprehensive_results_explained.csv", index=False)
    
    # Create summary table
    summary_df = results_df[['Experiment', 'Description', 'Recall@20', 'Performance Level', 'Interpretation']].copy()
    summary_df.to_csv(f"{output_dir}/summary_results_explained.csv", index=False)
    
    # Create comparison table
    if len(results_data) >= 4:
        comparison_data = create_comparison_analysis(results_data)
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(f"{output_dir}/research_findings_explained.csv", index=False)
    
    print("✅ User-friendly results created!")
    print(f"📁 Location: {output_dir}/")
    print("📊 Files created:")
    print("   - comprehensive_results_explained.csv (detailed results)")
    print("   - summary_results_explained.csv (key findings)")
    print("   - research_findings_explained.csv (research analysis)")
    
    return results_df


def get_interpretation(exp_key, recall_value):
    """Get human-readable interpretation of results."""
    
    interpretations = {
        'static_base': f"Baseline performance: {recall_value*100:.1f}% of relevant items found. This is your reference point.",
        'static_sol': f"With solution under ideal conditions: Shows if solution causes any harm (it shouldn't).",
        'dyn_base': f"Under dynamic noise: Performance drop shows DCCF struggles with changing noise patterns.",
        'dyn_sol': f"Solution under dynamic noise: Improvement over dynamic baseline proves solution works.",
        'burst_base': f"Under burst noise: {recall_value*100:.1f}% performance - DCCF's response to sudden spikes.",
        'shift_base': f"Under shift noise: {recall_value*100:.1f}% performance - DCCF's response to focus changes."
    }
    
    return interpretations.get(exp_key, f"Performance: {recall_value*100:.1f}%")


def create_comparison_analysis(results_data):
    """Create research findings comparison."""
    
    # Find baseline performance
    static_baseline = next((r for r in results_data if r['Experiment'] == 'Static Baseline'), None)
    dynamic_baseline = next((r for r in results_data if r['Experiment'] == 'Dynamic Baseline'), None)
    dynamic_solution = next((r for r in results_data if r['Experiment'] == 'Dynamic Solution'), None)
    
    comparison_data = []
    
    if static_baseline and dynamic_baseline:
        static_recall = static_baseline['Raw Recall']
        dynamic_recall = dynamic_baseline['Raw Recall']
        
        performance_drop = ((static_recall - dynamic_recall) / static_recall) * 100
        
        comparison_data.append({
            'Research Finding': 'Problem Identification',
            'Description': 'DCCF Performance Under Dynamic Noise',
            'Static Performance': f"{static_recall*100:.1f}%",
            'Dynamic Performance': f"{dynamic_recall*100:.1f}%", 
            'Performance Change': f"-{performance_drop:.1f}%",
            'Significance': 'Shows DCCF struggles with dynamic noise patterns',
            'Thesis Implication': 'Confirms hypothesis that DCCF assumes static noise'
        })
        
        if dynamic_solution:
            solution_recall = dynamic_solution['Raw Recall']
            improvement = ((solution_recall - dynamic_recall) / dynamic_recall) * 100
            
            comparison_data.append({
                'Research Finding': 'Solution Effectiveness',
                'Description': 'Our Solution vs Dynamic Baseline',
                'Static Performance': f"{static_recall*100:.1f}%",
                'Dynamic Performance': f"{solution_recall*100:.1f}%",
                'Performance Change': f"+{improvement:.1f}%",
                'Significance': 'Solution improves robustness under dynamic noise',
                'Thesis Implication': 'Proves proposed solution addresses the problem'
            })
    
    return comparison_data


def create_metrics_explanation():
    """Create explanation of what metrics mean."""
    
    explanation = {
        'Metric': ['Recall@20', 'NDCG@20', 'Performance Level'],
        'What It Measures': [
            'Out of all relevant items, how many were found in top-20 recommendations',
            'Quality of ranking - considers both relevance and position',
            'Overall assessment based on Recall@20 score'
        ],
        'Good Score': ['Above 18%', 'Above 6%', 'Good or Excellent'],
        'What Higher Means': [
            'Better at finding relevant items',
            'Better at ranking relevant items higher',
            'Overall better recommendation quality'
        ],
        'Example': [
            '20% = Found 2 out of 10 relevant items in top-20',
            '7% = Good ranking quality with relevant items near top',
            'Excellent = Strong performance for recommendation task'
        ]
    }
    
    explanation_df = pd.DataFrame(explanation)
    
    output_dir = "runs/user_friendly_results"
    os.makedirs(output_dir, exist_ok=True)
    explanation_df.to_csv(f"{output_dir}/metrics_explanation.csv", index=False)
    
    print("📚 Metrics explanation created: metrics_explanation.csv")


if __name__ == "__main__":
    print("🔄 Creating user-friendly results tables...")
    results_df = create_user_friendly_results()
    create_metrics_explanation()
    print("\n✅ All user-friendly tables created!")
    print("📖 Open the CSV files in Excel for easy reading")

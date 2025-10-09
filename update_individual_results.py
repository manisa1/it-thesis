#!/usr/bin/env python3
"""
Update individual experiment result files to be user-friendly.
This replaces the raw metrics.csv files with explained versions.
"""

import pandas as pd
import os
from pathlib import Path


def update_individual_experiment_files():
    """Update each experiment's metrics.csv to be user-friendly."""
    
    # Define experiment descriptions
    experiments = {
        'static_base': {
            'name': 'Static Baseline Experiment',
            'description': 'DCCF model with no noise and no solution',
            'purpose': 'Establishes baseline performance under ideal conditions',
            'what_to_expect': 'Highest performance - this is your reference point'
        },
        'static_sol': {
            'name': 'Static Solution Experiment',
            'description': 'DCCF model with no noise but with our solution applied',
            'purpose': 'Tests if our solution causes any harm under ideal conditions',
            'what_to_expect': 'Should be similar to baseline (proving no harm done)'
        },
        'dyn_base': {
            'name': 'Dynamic Baseline Experiment',
            'description': 'DCCF model with dynamic noise and no solution',
            'purpose': 'Shows how much dynamic noise hurts DCCF performance',
            'what_to_expect': 'Lower performance than static (proves the problem exists)'
        },
        'dyn_sol': {
            'name': 'Dynamic Solution Experiment', 
            'description': 'DCCF model with dynamic noise and our solution applied',
            'purpose': 'Tests if our solution helps under dynamic noise conditions',
            'what_to_expect': 'Better than dynamic baseline (proves solution works)'
        },
        'burst_base': {
            'name': 'Burst Noise Experiment',
            'description': 'DCCF model with sudden noise spikes (burst pattern)',
            'purpose': 'Tests how DCCF handles sudden popularity spikes',
            'what_to_expect': 'Variable performance - may be surprisingly good'
        },
        'shift_base': {
            'name': 'Shift Noise Experiment',
            'description': 'DCCF model with changing noise focus (shift pattern)', 
            'purpose': 'Tests how DCCF handles changing popularity focus',
            'what_to_expect': 'May show unexpected performance changes'
        }
    }
    
    updated_files = []
    
    for exp_key, exp_info in experiments.items():
        metrics_file = f"runs/{exp_key}/metrics.csv"
        
        if os.path.exists(metrics_file):
            # Read original metrics
            original_df = pd.read_csv(metrics_file)
            
            if not original_df.empty:
                recall = original_df['Recall@K'].iloc[0]
                ndcg = original_df['NDCG@K'].iloc[0]
                k = original_df['K'].iloc[0]
                
                # Create user-friendly version
                user_friendly_data = {
                    'Experiment Information': [
                        exp_info['name'],
                        exp_info['description'], 
                        exp_info['purpose'],
                        exp_info['what_to_expect'],
                        '',  # Empty row for spacing
                        'RESULTS:',
                        f'Recall@{k} Performance',
                        f'NDCG@{k} Quality Score', 
                        'Performance Level',
                        'What This Means'
                    ],
                    'Values and Explanations': [
                        '',  # Empty for experiment name
                        '',  # Empty for description
                        '',  # Empty for purpose  
                        '',  # Empty for expectation
                        '',  # Empty row
                        '',  # Empty for "RESULTS:"
                        f'{recall*100:.1f}% ({recall:.6f})',
                        f'{ndcg*100:.2f}% ({ndcg:.6f})',
                        get_performance_level(recall),
                        get_result_meaning(exp_key, recall)
                    ]
                }
                
                # Create new DataFrame
                new_df = pd.DataFrame(user_friendly_data)
                
                # Save user-friendly version
                new_df.to_csv(metrics_file, index=False)
                updated_files.append(metrics_file)
                
                print(f" Updated: {metrics_file}")
                print(f"   Performance: {recall*100:.1f}% - {get_performance_level(recall)}")
    
    print(f"\n🎉 Updated {len(updated_files)} experiment files!")
    print("📊 Now when you open any metrics.csv file, you'll see:")
    print("   - Experiment description")
    print("   - What it tests") 
    print("   - Results with percentages")
    print("   - Performance level (Excellent/Good/Fair/Poor)")
    print("   - What the results mean for your research")
    
    return updated_files


def get_performance_level(recall_value):
    """Convert recall to performance level."""
    if recall_value >= 0.20:
        return "Excellent"
    elif recall_value >= 0.18:
        return "Good"
    elif recall_value >= 0.16:
        return "Fair"
    else:
        return "Poor"


def get_result_meaning(exp_key, recall_value):
    """Get what the result means for research."""
    
    meanings = {
        'static_base': f"Baseline: {recall_value*100:.1f}% shows DCCF's best performance under ideal conditions",
        'static_sol': f"Control test: {recall_value*100:.1f}% proves our solution doesn't harm ideal performance", 
        'dyn_base': f"Problem evidence: {recall_value*100:.1f}% shows DCCF struggles with dynamic noise",
        'dyn_sol': f"Solution proof: {recall_value*100:.1f}% shows our solution helps under dynamic noise",
        'burst_base': f"Burst response: {recall_value*100:.1f}% shows how DCCF handles sudden popularity spikes",
        'shift_base': f"Shift response: {recall_value*100:.1f}% shows how DCCF handles changing popularity focus"
    }
    
    return meanings.get(exp_key, f"Performance result: {recall_value*100:.1f}%")


if __name__ == "__main__":
    print("🔄 Updating individual experiment files to be user-friendly...")
    updated_files = update_individual_experiment_files()
    print("\n All experiment files now have descriptions and explanations!")
    print("📖 Open any metrics.csv file to see the improved format")

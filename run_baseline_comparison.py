#!/usr/bin/env python3
"""
Run comprehensive baseline comparison experiments.

This script runs all baseline models (LightGCN, SimGCL, NGCF, SGL) under 
the same noise conditions as DCCF for fair comparison.
"""

import os
import subprocess
import pandas as pd
import argparse
from pathlib import Path


def run_experiment(model_type, experiment_name, noise_params, base_dir="runs/baselines"):
    """Run a single baseline experiment."""
    model_dir = os.path.join(base_dir, f"{model_type}_{experiment_name}")
    
    cmd = [
        "python", "train_baselines.py",
        "--model_type", model_type,
        "--model_dir", model_dir,
        "--epochs", "15",
        "--embedding_dim", "64",
        "--lr", "0.001"
    ]
    
    # Add noise parameters
    for key, value in noise_params.items():
        cmd.extend([f"--{key}", str(value)])
    
    print(f"Running {model_type} - {experiment_name}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ {model_type} - {experiment_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {model_type} - {experiment_name} failed")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Run all baseline comparison experiments."""
    parser = argparse.ArgumentParser(description='Run baseline comparison experiments')
    parser.add_argument('--models', nargs='+', default=['lightgcn', 'simgcl', 'ngcf', 'sgl', 'exposure_dro', 'pdif'],
                       choices=['lightgcn', 'simgcl', 'ngcf', 'sgl', 'exposure_dro', 'pdif'],
                       help='Models to compare')
    parser.add_argument('--experiments', nargs='+', 
                       default=['static_baseline', 'static_solution', 'dynamic_baseline', 'dynamic_solution'],
                       help='Experiments to run')
    
    args = parser.parse_args()
    
    # Define experiment configurations (matching DCCF experiments)
    experiments = {
        'static_baseline': {
            'noise_exposure_bias': 0.0,
            'noise_schedule': 'none'
        },
        'static_solution': {
            'noise_exposure_bias': 0.0,
            'noise_schedule': 'none'
            # Note: Baselines don't have reweighting, so this is same as baseline
        },
        'dynamic_baseline': {
            'noise_exposure_bias': 0.10,
            'noise_schedule': 'ramp',
            'noise_schedule_epochs': 10
        },
        'dynamic_solution': {
            'noise_exposure_bias': 0.10,
            'noise_schedule': 'ramp',
            'noise_schedule_epochs': 10
            # Note: Baselines don't have reweighting, so this is same as baseline
        },
        'burst_baseline': {
            'noise_exposure_bias': 0.10,
            'noise_schedule': 'burst',
            'noise_burst_start': 5,
            'noise_burst_len': 3,
            'noise_burst_scale': 2.0
        },
        'shift_baseline': {
            'noise_exposure_bias': 0.10,
            'noise_schedule': 'shift',
            'noise_shift_epoch': 8,
            'noise_shift_mode': 'head2tail'
        }
    }
    
    # Create base directory
    base_dir = "runs/baselines"
    os.makedirs(base_dir, exist_ok=True)
    
    # Run experiments
    results = []
    total_experiments = len(args.models) * len(args.experiments)
    completed = 0
    
    print(f"🚀 Starting baseline comparison with {total_experiments} experiments")
    print(f"Models: {', '.join(args.models)}")
    print(f"Experiments: {', '.join(args.experiments)}")
    print("=" * 60)
    
    for model_type in args.models:
        for exp_name in args.experiments:
            if exp_name in experiments:
                success = run_experiment(model_type, exp_name, experiments[exp_name], base_dir)
                completed += 1
                
                results.append({
                    'model': model_type,
                    'experiment': exp_name,
                    'status': 'success' if success else 'failed'
                })
                
                print(f"Progress: {completed}/{total_experiments} experiments completed")
                print("-" * 40)
    
    # Generate summary
    print("\n" + "=" * 60)
    print("📊 BASELINE COMPARISON SUMMARY")
    print("=" * 60)
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"✅ Successful experiments: {success_count}/{total_experiments}")
    print(f"❌ Failed experiments: {total_experiments - success_count}/{total_experiments}")
    
    # Show results by model
    for model in args.models:
        model_results = [r for r in results if r['model'] == model]
        model_success = sum(1 for r in model_results if r['status'] == 'success')
        print(f"  {model.upper()}: {model_success}/{len(model_results)} experiments successful")
    
    # Save results summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(base_dir, 'experiment_summary.csv'), index=False)
    
    print(f"\n📁 Results saved in: {base_dir}/")
    print("🔍 Use analyze_baseline_results.py to compare performance")
    
    return results


if __name__ == "__main__":
    main()

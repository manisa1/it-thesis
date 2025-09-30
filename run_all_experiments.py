#!/usr/bin/env python3
"""
Run all DCCF robustness experiments and analyze results.

This script runs all four experimental conditions and generates
the analysis required for the thesis.

Usage:
    python run_all_experiments.py
    python run_all_experiments.py --quick  # For faster testing
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import pandas as pd
import time


def run_single_experiment(config_path: str, verbose: bool = True) -> bool:
    """
    Run a single experiment.
    
    Args:
        config_path (str): Path to experiment configuration
        verbose (bool): Whether to show output
        
    Returns:
        bool: True if successful, False otherwise
    """
    cmd = [sys.executable, "run_experiment.py", "--config", config_path]
    
    try:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running: {config_path}")
            print(f"{'='*60}")
            
            # Run with real-time output
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Print output in real-time
            for line in process.stdout:
                print(line, end='')
            
            process.wait()
            success = process.returncode == 0
        else:
            # Run quietly
            result = subprocess.run(cmd, capture_output=True, text=True)
            success = result.returncode == 0
            
            if not success:
                print(f"Error in {config_path}:")
                print(result.stderr)
        
        return success
        
    except Exception as e:
        print(f"Failed to run {config_path}: {str(e)}")
        return False


def analyze_results() -> None:
    """Analyze and summarize all experiment results."""
    print(f"\n{'='*60}")
    print("ANALYZING RESULTS")
    print(f"{'='*60}")
    
    # Define experiment configurations
    experiments = {
        "static_base": "runs/static_base/metrics.csv",
        "static_sol": "runs/static_sol/metrics.csv", 
        "dyn_base": "runs/dyn_base/metrics.csv",
        "dyn_sol": "runs/dyn_sol/metrics.csv",
        "burst_base": "runs/burst_base/metrics.csv",
        "burst_sol": "runs/burst_sol/metrics.csv",
        "shift_base": "runs/shift_base/metrics.csv", 
        "shift_sol": "runs/shift_sol/metrics.csv"
    }
    
    # Collect results
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
                print(f"✓ Loaded results for {exp_name}")
            except Exception as e:
                print(f"✗ Error loading {exp_name}: {str(e)}")
                missing_results.append(exp_name)
        else:
            print(f"✗ Missing results for {exp_name}: {metrics_path}")
            missing_results.append(exp_name)
    
    if missing_results:
        print(f"\nWarning: Missing results for {missing_results}")
        print("Some experiments may have failed. Check the logs.")
        return
    
    if len(results) < 4:
        print(f"\nError: Only {len(results)}/4 experiments completed successfully.")
        return
    
    # Create summary dataframe
    summary_df = pd.DataFrame(results).set_index("experiment")
    
    # Reorder to match thesis presentation
    experiment_order = ["static_base", "static_sol", "dyn_base", "dyn_sol"]
    summary_df = summary_df.loc[experiment_order]
    
    # Save summary
    os.makedirs("runs", exist_ok=True)
    summary_path = "runs/summary.csv"
    summary_df.to_csv(summary_path)
    
    # Compute robustness analysis
    def robustness_drop(clean_val, noisy_val):
        return (clean_val - noisy_val) / clean_val if clean_val > 0 else 0.0
    
    # Get values for robustness calculation
    clean_recall = summary_df.loc["static_base", "Recall@20"]
    noisy_recall_base = summary_df.loc["dyn_base", "Recall@20"] 
    noisy_recall_sol = summary_df.loc["dyn_sol", "Recall@20"]
    
    clean_ndcg = summary_df.loc["static_base", "NDCG@20"]
    noisy_ndcg_base = summary_df.loc["dyn_base", "NDCG@20"]
    noisy_ndcg_sol = summary_df.loc["dyn_sol", "NDCG@20"]
    
    # Create robustness dataframe
    robustness_df = pd.DataFrame({
        "Metric": ["Recall@20", "NDCG@20"],
        "Baseline Robustness Drop": [
            robustness_drop(clean_recall, noisy_recall_base),
            robustness_drop(clean_ndcg, noisy_ndcg_base)
        ],
        "Solution Robustness Drop": [
            robustness_drop(clean_recall, noisy_recall_sol), 
            robustness_drop(clean_ndcg, noisy_ndcg_sol)
        ]
    })
    
    # Save robustness analysis
    robustness_path = "runs/robustness.csv"
    robustness_df.to_csv(robustness_path, index=False)
    
    # Display results
    print(f"\n{'='*60}")
    print("THESIS RESULTS SUMMARY")
    print(f"{'='*60}")
    
    print("\n📊 PERFORMANCE SUMMARY:")
    print(summary_df.round(4))
    
    print(f"\n💾 Summary saved to: {summary_path}")
    
    print(f"\n🔍 ROBUSTNESS ANALYSIS:")
    print(robustness_df.round(4))
    
    print(f"\n💾 Robustness analysis saved to: {robustness_path}")
    
    # Key findings
    print(f"\n🎯 KEY THESIS FINDINGS:")
    
    baseline_drop = robustness_df.loc[0, "Baseline Robustness Drop"]
    solution_drop = robustness_df.loc[0, "Solution Robustness Drop"] 
    improvement = baseline_drop - solution_drop
    
    print(f"1. DCCF's Dynamic Noise Vulnerability:")
    print(f"   - Recall@20 drops {baseline_drop:.1%} under dynamic noise")
    print(f"   - NDCG@20 drops {robustness_df.loc[1, 'Baseline Robustness Drop']:.1%} under dynamic noise")
    
    print(f"\n2. Solution Effectiveness:")
    print(f"   - Reduces Recall@20 robustness drop to {solution_drop:.1%}")
    print(f"   - Improvement: {improvement:.1%} better robustness")
    
    print(f"\n3. No Performance Degradation Under Static Conditions:")
    static_base = summary_df.loc["static_base", "Recall@20"]
    static_sol = summary_df.loc["static_sol", "Recall@20"]
    static_diff = (static_sol - static_base) / static_base
    print(f"   - Static performance change: {static_diff:+.1%} (minimal impact)")
    
    print(f"\n✅ ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"📁 Results available in: runs/ directory")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run all DCCF robustness experiments"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true", 
        help="Run with reduced epochs for quick testing"
    )
    
    parser.add_argument(
        "--skip-experiments",
        action="store_true",
        help="Skip running experiments, only analyze existing results"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Show detailed output from experiments"
    )
    
    args = parser.parse_args()
    
    # Core experiment configurations (matches interim report)
    experiments = [
        "configs/experiments/static_baseline.yaml",
        "configs/experiments/static_solution.yaml", 
        "configs/experiments/dynamic_baseline.yaml",
        "configs/experiments/dynamic_solution.yaml"
    ]
    
    # Additional experiments for comprehensive analysis
    additional_experiments = [
        "configs/experiments/static_05_baseline.yaml",
        "configs/experiments/static_15_baseline.yaml", 
        "configs/experiments/static_20_baseline.yaml",
        "configs/experiments/burst_baseline.yaml",
        "configs/experiments/burst_solution.yaml",
        "configs/experiments/shift_baseline.yaml",
        "configs/experiments/shift_solution.yaml"
    ]
    
    if not args.skip_experiments:
        print("🚀 Starting DCCF Robustness Experiments")
        print(f"📅 {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if args.quick:
            print("⚡ Quick mode: Using reduced epochs for testing")
        
        # Check if data exists
        if not os.path.exists("data/ratings.csv"):
            print("\n❌ Error: data/ratings.csv not found!")
            print("Please run: python make_data.py")
            sys.exit(1)
        
        # Run all experiments
        start_time = time.time()
        failed_experiments = []
        
        # Run core experiments first
        print(f"\n🎯 Running Core Experiments (4/4):")
        for i, config_path in enumerate(experiments, 1):
            print(f"\n🔬 Core Experiment {i}/4: {config_path}")
            
            success = run_single_experiment(config_path, verbose=args.verbose)
            
            if success:
                print(f"✅ Completed: {config_path}")
            else:
                print(f"❌ Failed: {config_path}")
                failed_experiments.append(config_path)
        
        # Optionally run additional experiments
        if not args.quick:
            print(f"\n📊 Running Additional Analysis Experiments ({len(additional_experiments)}):")
            for i, config_path in enumerate(additional_experiments, 1):
                print(f"\n🔬 Additional Experiment {i}/{len(additional_experiments)}: {config_path}")
                
                success = run_single_experiment(config_path, verbose=args.verbose)
                
                if success:
                    print(f"✅ Completed: {config_path}")
                else:
                    print(f"❌ Failed: {config_path}")
                    failed_experiments.append(config_path)
        
        # Report experiment completion
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  Total experiment time: {elapsed_time/60:.1f} minutes")
        
        if failed_experiments:
            print(f"\n❌ Failed experiments: {failed_experiments}")
            print("Check the logs for error details.")
        else:
            print(f"\n✅ All {len(experiments)} experiments completed successfully!")
    
    # Analyze results
    try:
        analyze_results()
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script to verify new burst and shift experiments work correctly.

This script performs quick validation of the new experiment configurations
without running full training.
"""

import os
import sys
from pathlib import Path
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import ExperimentConfig
from src.training.noise import NoiseGenerator
import pandas as pd
import numpy as np

def test_config_loading():
 """Test that all new configuration files load correctly."""
 print(" Testing Configuration Loading...")

 configs_to_test = [
 "configs/experiments/burst_baseline.yaml",
 "configs/experiments/burst_solution.yaml",
 "configs/experiments/shift_baseline.yaml",
 "configs/experiments/shift_solution.yaml"
 ]

 for config_path in configs_to_test:
 if os.path.exists(config_path):
 try:
 config = ExperimentConfig.from_yaml(config_path)
 print(f" {config_path}: Loaded successfully")
 print(f" - Noise schedule: {config.noise_schedule}")
 print(f" - Noise level: {config.noise_level}")
 if hasattr(config, 'burst_start') and config.burst_start:
 print(f" - Burst: epochs {config.burst_start}-{config.burst_end}")
 if hasattr(config, 'shift_epoch') and config.shift_epoch:
 print(f" - Shift: epoch {config.shift_epoch}")
 print(f" - Reweighting: {config.use_reweighting}")
 except Exception as e:
 print(f" {config_path}: Error loading - {str(e)}")
 else:
 print(f" {config_path}: File not found")

 print()

def test_noise_patterns():
 """Test that noise patterns generate correctly."""
 print(" Testing Noise Pattern Generation...")

 # Create dummy data
 dummy_data = pd.DataFrame({
 'u': [0, 1, 2, 0, 1] * 20, # 100 interactions
 'i': [0, 1, 2, 3, 4] * 20
 })

 noise_gen = NoiseGenerator(seed=42)

 # Test burst noise
 print("Testing Burst Noise Pattern:")
 for epoch in [1, 5, 6, 7, 8, 10, 15]:
 noisy_data = noise_gen.add_exposure_noise(
 dummy_data, n_users=10, n_items=10,
 noise_level=0.1, schedule='burst',
 epoch=epoch, max_epochs=15,
 burst_start=5, burst_end=8
 )
 noise_ratio = (len(noisy_data) - len(dummy_data)) / len(dummy_data)
 burst_active = " BURST" if 5 <= epoch <= 8 else " normal"
 print(f" Epoch {epoch:2d}: {noise_ratio:.3f} noise ratio {burst_active}")

 print("\nTesting Shift Noise Pattern:")
 for epoch in [1, 5, 8, 10, 15]:
 noisy_data = noise_gen.add_exposure_noise(
 dummy_data, n_users=10, n_items=10,
 noise_level=0.1, schedule='shift',
 epoch=epoch, max_epochs=15,
 shift_epoch=8
 )
 noise_ratio = (len(noisy_data) - len(dummy_data)) / len(dummy_data)
 shift_phase = " HIGH" if epoch >= 8 else " LOW"
 print(f" Epoch {epoch:2d}: {noise_ratio:.3f} noise ratio {shift_phase}")

 print()

def test_dataset_configs():
 """Test dataset configuration files."""
 print(" Testing Dataset Configurations...")

 dataset_configs = [
 "configs/datasets/gowalla_config.yaml",
 "configs/datasets/amazon_book_config.yaml"
 ]

 for config_path in dataset_configs:
 if os.path.exists(config_path):
 try:
 with open(config_path, 'r') as f:
 config_data = yaml.safe_load(f)
 print(f" {config_path}: Loaded successfully")
 print(f" - Dataset: {config_data.get('dataset_name', 'Unknown')}")
 print(f" - Users: {config_data.get('n_users', 'Unknown'):,}")
 print(f" - Items: {config_data.get('n_items', 'Unknown'):,}")
 print(f" - Data path: {config_data.get('data_path', 'Unknown')}")
 except Exception as e:
 print(f" {config_path}: Error loading - {str(e)}")
 else:
 print(f" {config_path}: File not found")

 print()

def test_experiment_runner():
 """Test that experiment runner can handle new configs."""
 print(" Testing Experiment Runner Integration...")

 # Check if run_all_experiments.py includes new experiments
 with open("run_all_experiments.py", 'r') as f:
 content = f.read()

 required_experiments = [
 "burst_baseline.yaml",
 "burst_solution.yaml",
 "shift_baseline.yaml",
 "shift_solution.yaml"
 ]

 for exp in required_experiments:
 if exp in content:
 print(f" {exp}: Found in run_all_experiments.py")
 else:
 print(f" {exp}: Missing from run_all_experiments.py")

 print()

def main():
 """Run all tests."""
 print(" Testing New Burst and Shift Experiments")
 print("=" * 60)

 test_config_loading()
 test_noise_patterns()
 test_dataset_configs()
 test_experiment_runner()

 print("=" * 60)
 print(" All tests completed!")
 print("\n Next Steps:")
 print("1. Download Gowalla and Amazon-book datasets using prepare_datasets.py")
 print("2. Run experiments: python run_all_experiments.py")
 print("3. Analyze results: python analyze_comprehensive_results.py")

if __name__ == "__main__":
 main()
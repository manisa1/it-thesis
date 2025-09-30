#!/usr/bin/env python3
"""
Simple experiment runner for the updated train.py with burst and shift support.
"""

import subprocess
import sys
import os

def run_experiment(name, **kwargs):
 """Run a single experiment with train.py"""
 cmd = [sys.executable, "train.py"]

 # Add all parameters
 for key, value in kwargs.items():
 cmd.extend([f"--{key}", str(value)])

 print(f"\n Running {name}")
 print(f"Command: {' '.join(cmd)}")
 print("=" * 60)

 result = subprocess.run(cmd)
 if result.returncode == 0:
 print(f" {name} completed successfully")
 else:
 print(f" {name} failed")

 return result.returncode == 0

def main():
 """Run all experiments"""

 # Common parameters
 base_params = {
 "data_path": "data/ratings.csv",
 "epochs": 15,
 "k": 64,
 "k_eval": 20,
 "lr": 0.01
 }

 experiments = [
 # Static experiments
 ("Static Baseline", {
 **base_params,
 "model_dir": "runs/static_base",
 "noise_exposure_bias": 0.0,
 "noise_schedule": "none",
 "reweight_type": "none"
 }),

 ("Static Solution", {
 **base_params,
 "model_dir": "runs/static_sol",
 "noise_exposure_bias": 0.0,
 "noise_schedule": "none",
 "reweight_type": "popularity",
 "reweight_alpha": 0.5,
 "reweight_ramp_epochs": 10
 }),

 # Dynamic experiments
 ("Dynamic Baseline", {
 **base_params,
 "model_dir": "runs/dyn_base",
 "noise_exposure_bias": 0.10,
 "noise_schedule": "ramp",
 "noise_schedule_epochs": 10,
 "reweight_type": "none"
 }),

 ("Dynamic Solution", {
 **base_params,
 "model_dir": "runs/dyn_sol",
 "noise_exposure_bias": 0.10,
 "noise_schedule": "ramp",
 "noise_schedule_epochs": 10,
 "reweight_type": "popularity",
 "reweight_alpha": 0.5,
 "reweight_ramp_epochs": 10
 }),

 # Burst experiments
 ("Burst Baseline", {
 **base_params,
 "model_dir": "runs/burst_base",
 "noise_exposure_bias": 0.10,
 "noise_schedule": "burst",
 "noise_burst_start": 5,
 "noise_burst_len": 3,
 "noise_burst_scale": 2.0,
 "reweight_type": "none"
 }),

 ("Burst Solution", {
 **base_params,
 "model_dir": "runs/burst_sol",
 "noise_exposure_bias": 0.10,
 "noise_schedule": "burst",
 "noise_burst_start": 5,
 "noise_burst_len": 3,
 "noise_burst_scale": 2.0,
 "reweight_type": "popularity",
 "reweight_alpha": 0.5,
 "reweight_ramp_epochs": 10
 }),

 # Shift experiments
 ("Shift Baseline", {
 **base_params,
 "model_dir": "runs/shift_base",
 "noise_exposure_bias": 0.10,
 "noise_schedule": "shift",
 "noise_shift_epoch": 8,
 "noise_shift_mode": "head2tail",
 "reweight_type": "none"
 }),

 ("Shift Solution", {
 **base_params,
 "model_dir": "runs/shift_sol",
 "noise_exposure_bias": 0.10,
 "noise_schedule": "shift",
 "noise_shift_epoch": 8,
 "noise_shift_mode": "head2tail",
 "reweight_type": "popularity",
 "reweight_alpha": 0.5,
 "reweight_ramp_epochs": 10
 })
 ]

 print(" Starting All DCCF Robustness Experiments")
 print(f"Total experiments: {len(experiments)}")

 results = []
 for name, params in experiments:
 success = run_experiment(name, **params)
 results.append((name, success))

 # Summary
 print("\n" + "=" * 60)
 print(" EXPERIMENT SUMMARY")
 print("=" * 60)

 for name, success in results:
 status = " SUCCESS" if success else " FAILED"
 print(f"{name:20} {status}")

 successful = sum(1 for _, success in results if success)
 print(f"\nCompleted: {successful}/{len(experiments)} experiments")

if __name__ == "__main__":
 main()
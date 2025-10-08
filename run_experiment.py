#!/usr/bin/env python3
"""
Main experiment runner for baseline model robustness comparison study.

This script runs individual experiments based on configuration files.
It provides a clean interface for running comparative robustness experiments 
with proper logging, error handling, and result saving.

Usage:
    python run_experiment.py --config configs/experiments/static_baseline.yaml
    python run_experiment.py --config configs/experiments/dynamic_baseline.yaml
"""

import argparse
import os
import sys
import traceback
from pathlib import Path
import pandas as pd
import torch
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import ExperimentConfig
from src.utils.logging import ExperimentLogger
from src.data.preprocessing import DataPreprocessor
from src.data.dataset import RecommenderDataset
from src.models.matrix_factorization import MatrixFactorizationBPR
from src.training.trainer import BaselineTrainer
from src.training.noise import NoiseGenerator
from src.evaluation.metrics import RecommendationMetrics


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def run_experiment(config: ExperimentConfig) -> dict:
    """
    Run a single experiment based on configuration.
    
    Args:
        config (ExperimentConfig): Experiment configuration
        
    Returns:
        dict: Experiment results
    """
    # Set up logging
    experiment_name = config.get_experiment_name()
    logger = ExperimentLogger(experiment_name, config.output_dir)
    
    try:
        # Log configuration
        logger.log_config(config.to_dict())
        
        # Set random seeds
        set_random_seeds(config.random_seed)
        logger.logger.info(f"Set random seed to {config.random_seed}")
        
        # Load and preprocess data
        logger.logger.info("Loading and preprocessing data...")
        preprocessor = DataPreprocessor(rating_threshold=config.rating_threshold)
        df = preprocessor.load_and_preprocess(config.data_path)
        
        # Split data
        train_df, val_df, test_df = preprocessor.train_val_test_split(
            df, 
            val_frac=config.val_fraction,
            test_frac=config.test_fraction,
            seed=config.random_seed
        )
        
        # Create datasets
        train_dataset = RecommenderDataset(train_df)
        val_dataset = RecommenderDataset(val_df)
        test_dataset = RecommenderDataset(test_df)
        
        # Log dataset info
        dataset_info = preprocessor.get_dataset_info()
        dataset_info.update(train_dataset.get_dataset_stats())
        logger.log_dataset_info(dataset_info)
        
        # Initialize model
        logger.logger.info("Initializing model...")
        model = MatrixFactorizationBPR(
            n_users=dataset_info['n_users'],
            n_items=dataset_info['n_items'],
            embedding_dim=config.embedding_dim
        )
        
        # Initialize trainer
        trainer = BaselineTrainer(
            model=model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            device=config.device
        )
        
        # Initialize noise generator
        noise_generator = NoiseGenerator(seed=config.random_seed)
        
        # Compute weights if reweighting is enabled
        item_weights = None
        confidence_weights = None
        if config.use_reweighting:
            logger.logger.info("Computing static confidence denoiser weights...")
            confidence_weights = trainer.compute_static_confidence_weights(
                train_dataset, 
                c_min=config.confidence_min
            )
            logger.logger.info("Computing popularity-based item weights...")
            item_weights = trainer.compute_popularity_weights(
                train_dataset, 
                alpha=config.reweight_alpha
            )
        
        # Initialize metrics calculator
        metrics_calculator = RecommendationMetrics(k=config.k_eval)
        
        # Training loop
        logger.logger.info("Starting training...")
        best_val_recall = -1.0
        best_epoch = -1
        
        for epoch in range(1, config.epochs + 1):
            # Apply noise to training data
            noisy_train_df = noise_generator.add_exposure_noise(
                train_df,
                n_users=dataset_info['n_users'],
                n_items=dataset_info['n_items'],
                noise_level=config.noise_level,
                schedule=config.noise_schedule,
                epoch=epoch,
                max_epochs=config.epochs,
                burst_start=config.burst_start,
                burst_end=config.burst_end,
                shift_epoch=config.shift_epoch
            )
            
            # Create noisy training dataset
            noisy_train_dataset = RecommenderDataset(noisy_train_df)
            
            # Apply reweighting burn-in if enabled
            epoch_item_weights = None
            if item_weights is not None:
                epoch_item_weights = trainer.apply_reweight_burnin(
                    item_weights, epoch, config.reweight_burnin_epochs
                )
            
            # Use confidence weights if available (static confidence denoiser)
            final_weights = confidence_weights if confidence_weights is not None else epoch_item_weights
            
            # Train for one epoch
            train_loss = trainer.train_epoch(
                noisy_train_dataset,
                batch_size=config.batch_size,
                item_weights=epoch_item_weights
            )
            
            # Evaluate on validation set
            val_metrics = metrics_calculator.evaluate_model(
                model, val_dataset, train_dataset, exclude_train=False
            )
            
            # Log epoch results
            noise_info = noise_generator.get_noise_info(
                len(train_df), len(noisy_train_df), 
                config.noise_level, config.noise_schedule
            )
            
            additional_info = {
                'noise_ratio': f"{noise_info['actual_noise_ratio']:.3f}",
                'burnin_progress': f"{min(1.0, epoch / config.reweight_burnin_epochs):.2f}" if config.use_reweighting else "N/A"
            }
            
            logger.log_epoch(
                epoch, config.epochs, train_loss, val_metrics, additional_info
            )
            
            # Save best model
            if val_metrics['recall@k'] > best_val_recall:
                best_val_recall = val_metrics['recall@k']
                best_epoch = epoch
                
                if config.save_model:
                    model_path = os.path.join(config.output_dir, "best.pt")
                    trainer.save_model(model_path)
        
        # Load best model and evaluate on test set
        if config.save_model and best_epoch > 0:
            logger.logger.info(f"Loading best model from epoch {best_epoch}")
            model_path = os.path.join(config.output_dir, "best.pt")
            trainer.load_model(model_path)
        
        # Final evaluation
        logger.logger.info("Evaluating on test set...")
        test_metrics = metrics_calculator.evaluate_model(
            model, test_dataset, train_dataset, exclude_train=False
        )
        
        # Prepare final results
        results = {
            'experiment_name': experiment_name,
            'best_epoch': best_epoch,
            'test_recall@k': test_metrics['recall@k'],
            'test_ndcg@k': test_metrics['ndcg@k'],
            'k': config.k_eval,
            'n_test_users': test_metrics['n_users']
        }
        
        # Save results
        results_df = pd.DataFrame([{
            'Recall@K': results['test_recall@k'],
            'NDCG@K': results['test_ndcg@k'],
            'K': results['k']
        }])
        
        metrics_path = os.path.join(config.output_dir, "metrics.csv")
        results_df.to_csv(metrics_path, index=False)
        
        # Save configuration
        config_path = os.path.join(config.output_dir, "config.yaml")
        config.save_yaml(config_path)
        
        # Log final results
        logger.log_final_results(results)
        logger.finish_experiment()
        
        logger.logger.info(f"Results saved to {metrics_path}")
        logger.logger.info(f"Configuration saved to {config_path}")
        
        return results
        
    except Exception as e:
        logger.log_error(e)
        raise


def main():
    """Main function for running experiments."""
    parser = argparse.ArgumentParser(
        description="Run DCCF robustness experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --config configs/experiments/static_baseline.yaml
  python run_experiment.py --config configs/experiments/dynamic_solution.yaml
        """
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to experiment configuration file (YAML)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory from config"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        help="Override device from config"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        print(f"Loading configuration from {args.config}")
        config = ExperimentConfig.from_yaml(args.config)
        
        # Apply command line overrides
        if args.output_dir:
            config.output_dir = args.output_dir
        
        if args.device:
            config.device = args.device
        
        print(f"Running experiment: {config.get_experiment_name()}")
        print(f"Output directory: {config.output_dir}")
        
        # Run experiment
        results = run_experiment(config)
        
        print("\n" + "="*50)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Experiment: {results['experiment_name']}")
        print(f"Test Recall@{results['k']}: {results['test_recall@k']:.4f}")
        print(f"Test NDCG@{results['k']}: {results['test_ndcg@k']:.4f}")
        print(f"Results saved to: {config.output_dir}")
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nExperiment failed: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

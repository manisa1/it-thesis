"""
Logging utilities for DCCF experiments.

This module provides structured logging for experiment tracking and debugging.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logger(name: str = "dccf_experiment",
                log_file: Optional[str] = None,
                log_level: str = "INFO",
                console_output: bool = True) -> logging.Logger:
    """
    Set up a logger for experiments.
    
    Args:
        name (str): Logger name
        log_file (str, optional): Path to log file
        log_level (str): Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        console_output (bool): Whether to output to console
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if log file specified
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class ExperimentLogger:
    """
    Enhanced logger for experiment tracking.
    
    This class provides structured logging specifically designed for
    machine learning experiments with metrics tracking.
    """
    
    def __init__(self, 
                 experiment_name: str,
                 output_dir: str,
                 log_level: str = "INFO"):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name (str): Name of the experiment
            output_dir (str): Directory to save logs
            log_level (str): Logging level
        """
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up logger
        log_file = os.path.join(output_dir, f"{experiment_name}.log")
        self.logger = setup_logger(
            name=f"experiment_{experiment_name}",
            log_file=log_file,
            log_level=log_level
        )
        
        # Track experiment start
        self.start_time = datetime.now()
        self.logger.info(f"Starting experiment: {experiment_name}")
        self.logger.info(f"Output directory: {output_dir}")
    
    def log_config(self, config: dict) -> None:
        """
        Log experiment configuration.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.logger.info("Experiment Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_dataset_info(self, dataset_info: dict) -> None:
        """
        Log dataset information.
        
        Args:
            dataset_info (dict): Dataset statistics
        """
        self.logger.info("Dataset Information:")
        for key, value in dataset_info.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_epoch(self, 
                  epoch: int, 
                  total_epochs: int,
                  train_loss: float,
                  val_metrics: Optional[dict] = None,
                  additional_info: Optional[dict] = None) -> None:
        """
        Log epoch training information.
        
        Args:
            epoch (int): Current epoch
            total_epochs (int): Total number of epochs
            train_loss (float): Training loss
            val_metrics (dict, optional): Validation metrics
            additional_info (dict, optional): Additional information to log
        """
        msg = f"Epoch {epoch}/{total_epochs} - Loss: {train_loss:.4f}"
        
        if val_metrics:
            for metric, value in val_metrics.items():
                msg += f" - {metric}: {value:.4f}"
        
        if additional_info:
            for key, value in additional_info.items():
                msg += f" - {key}: {value}"
        
        self.logger.info(msg)
    
    def log_final_results(self, results: dict) -> None:
        """
        Log final experiment results.
        
        Args:
            results (dict): Final results dictionary
        """
        self.logger.info("Final Results:")
        for key, value in results.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_error(self, error: Exception) -> None:
        """
        Log an error with full traceback.
        
        Args:
            error (Exception): Exception to log
        """
        self.logger.error(f"Experiment failed: {str(error)}", exc_info=True)
    
    def finish_experiment(self) -> None:
        """Log experiment completion."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        self.logger.info(f"Experiment completed in {duration}")
        self.logger.info("=" * 50)

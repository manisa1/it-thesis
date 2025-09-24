"""
Configuration management for DCCF experiments.

This module provides a clean way to manage experiment configurations
using dataclasses for type safety and validation.
"""

from dataclasses import dataclass, asdict
from typing import Optional
import json
import yaml
import os


@dataclass
class ExperimentConfig:
    """
    Configuration class for DCCF robustness experiments.
    
    This dataclass contains all parameters needed for running experiments,
    with sensible defaults and validation.
    """
    
    # Data parameters
    data_path: str = "data/ratings.csv"
    rating_threshold: float = 4.0
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    
    # Model parameters
    embedding_dim: int = 64
    learning_rate: float = 0.01
    weight_decay: float = 1e-6
    
    # Training parameters
    epochs: int = 15
    batch_size: int = 2048
    
    # Evaluation parameters
    k_eval: int = 20
    
    # Noise parameters
    noise_level: float = 0.0
    noise_schedule: str = "static"  # "static" or "ramp"
    noise_ramp_epochs: int = 10
    
    # Reweighting parameters (static confidence denoiser + DRO)
    use_reweighting: bool = False
    reweight_alpha: float = 0.5
    reweight_burnin_epochs: int = 10  # Burn-in terminology from interim report
    confidence_min: float = 0.1       # c_min for static confidence denoiser
    
    # Output parameters
    output_dir: str = "runs/experiment"
    save_model: bool = True
    
    # System parameters
    device: str = "auto"  # "auto", "cpu", or "cuda"
    random_seed: int = 42
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        
        # Auto-detect device if needed
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError("val_fraction + test_fraction must be < 1.0")
        
        if not 0.0 <= self.noise_level <= 1.0:
            raise ValueError("noise_level must be between 0.0 and 1.0")
        
        if self.noise_schedule not in ["static", "ramp", "burst", "shift"]:
            raise ValueError("noise_schedule must be 'static', 'ramp', 'burst', or 'shift'")
        
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        
        if self.k_eval <= 0:
            raise ValueError("k_eval must be positive")
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ExperimentConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict (dict): Configuration dictionary
            
        Returns:
            ExperimentConfig: Configuration object
        """
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ExperimentConfig':
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path (str): Path to YAML configuration file
            
        Returns:
            ExperimentConfig: Configuration object
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Handle base_config inheritance (simple approach)
        if 'base_config' in config_dict:
            base_path = os.path.join(os.path.dirname(yaml_path), config_dict.pop('base_config'))
            with open(base_path, 'r') as f:
                base_config = yaml.safe_load(f)
            # Merge configs (experiment overrides base)
            base_config.update(config_dict)
            config_dict = base_config
            
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'ExperimentConfig':
        """
        Load configuration from JSON file.
        
        Args:
            json_path (str): Path to JSON configuration file
            
        Returns:
            ExperimentConfig: Configuration object
        """
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.
        
        Returns:
            dict: Configuration as dictionary
        """
        return asdict(self)
    
    def save_yaml(self, yaml_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path (str): Path to save YAML file
        """
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def save_json(self, json_path: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            json_path (str): Path to save JSON file
        """
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def get_experiment_name(self) -> str:
        """
        Generate a descriptive experiment name based on configuration.
        
        Returns:
            str: Experiment name
        """
        noise_type = "static" if self.noise_schedule == "static" else "dynamic"
        
        if self.noise_level > 0:
            noise_str = f"{noise_type}_noise_{self.noise_level:.1f}"
        else:
            noise_str = "clean"
        
        if self.use_reweighting:
            reweight_str = f"_reweight_{self.reweight_alpha:.1f}"
        else:
            reweight_str = "_baseline"
        
        return f"{noise_str}{reweight_str}"
    
    def __str__(self) -> str:
        """String representation of configuration."""
        lines = ["ExperimentConfig:"]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

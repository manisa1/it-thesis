"""
Training modules for baseline model robustness experiments.
"""

from .noise import NoiseGenerator

# Import trainer only when explicitly needed to avoid circular imports
def get_trainer():
    from .trainer import BaselineTrainer
    return BaselineTrainer

__all__ = ['NoiseGenerator', 'get_trainer']

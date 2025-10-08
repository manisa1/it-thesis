"""
Training modules for DCCF robustness experiments.
"""

from .noise import NoiseGenerator

# Import trainer only when explicitly needed to avoid circular imports
def get_trainer():
    from .trainer import DCCFTrainer
    return DCCFTrainer

__all__ = ['NoiseGenerator', 'get_trainer']

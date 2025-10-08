#!/usr/bin/env python3
"""
Test script for new baseline models (Exposure-aware DRO and PDIF).
"""

import sys
import torch
import pandas as pd
import numpy as np

# Add src to path
sys.path.append('src')

from models import ExposureAwareReweighting, PDIF
from models.exposure_aware_dro import exposure_dro_loss
from models.pdif import pdif_loss


def test_exposure_dro():
    """Test Exposure-aware DRO model."""
    print("🧪 Testing Exposure-aware DRO...")
    
    # Create synthetic data
    n_users, n_items = 100, 50
    batch_size = 32
    
    # Initialize model
    model = ExposureAwareReweighting(n_users, n_items, k=16)
    
    # Create sample batch
    users = torch.randint(0, n_users, (batch_size,))
    pos_items = torch.randint(0, n_items, (batch_size,))
    neg_items = torch.randint(0, n_items, (batch_size,))
    
    # Forward pass
    user_emb, pos_emb, neg_emb = model(users, pos_items, neg_items)
    
    # Compute loss
    loss, bpr_loss, reg_loss = exposure_dro_loss(user_emb, pos_emb, neg_emb, pos_items, model)
    
    print(f"  ✅ Forward pass successful")
    print(f"  ✅ Loss computation successful: {loss.item():.4f}")
    print(f"  ✅ BPR loss: {bpr_loss.item():.4f}")
    print(f"  ✅ Reg loss: {reg_loss.item():.4f}")
    
    # Test backward pass
    loss.backward()
    print(f"  ✅ Backward pass successful")
    
    return True


def test_pdif():
    """Test PDIF model."""
    print("\n🧪 Testing PDIF...")
    
    # Create synthetic data
    n_users, n_items = 100, 50
    batch_size = 32
    
    # Initialize model
    model = PDIF(n_users, n_items, k=16)
    
    # Create sample training data
    train_data = []
    for _ in range(200):
        user = np.random.randint(0, n_users)
        item = np.random.randint(0, n_items)
        train_data.append({'u': user, 'i': item})
    
    train_df = pd.DataFrame(train_data)
    
    # Test personalized threshold computation
    model.compute_personalized_thresholds()
    print(f"  ✅ Personalized thresholds computed")
    
    # Test noise identification
    noise_scores = model.identify_noisy_interactions(train_df)
    print(f"  ✅ Noise identification successful: {len(noise_scores)} interactions scored")
    
    # Test data resampling
    filtered_df = model.resample_training_data(train_df)
    print(f"  ✅ Data resampling successful: {len(train_df)} → {len(filtered_df)} interactions")
    
    # Create sample batch
    users = torch.randint(0, n_users, (batch_size,))
    pos_items = torch.randint(0, n_items, (batch_size,))
    neg_items = torch.randint(0, n_items, (batch_size,))
    
    # Forward pass
    user_emb, pos_emb, neg_emb = model(users, pos_items, neg_items)
    
    # Compute loss
    loss, bpr_loss, reg_loss = pdif_loss(user_emb, pos_emb, neg_emb, users, model)
    
    print(f"  ✅ Forward pass successful")
    print(f"  ✅ Loss computation successful: {loss.item():.4f}")
    print(f"  ✅ BPR loss: {bpr_loss.item():.4f}")
    print(f"  ✅ Reg loss: {reg_loss.item():.4f}")
    
    # Test backward pass
    loss.backward()
    print(f"  ✅ Backward pass successful")
    
    return True


def test_integration():
    """Test integration with existing framework."""
    print("\n🧪 Testing integration...")
    
    # Test model creation similar to train_baselines.py
    n_users, n_items = 100, 50
    
    # Test both models can be created
    dro_model = ExposureAwareReweighting(n_users, n_items, k=16)
    pdif_model = PDIF(n_users, n_items, k=16)
    
    print(f"  ✅ Both models created successfully")
    
    # Test they have required methods
    assert hasattr(dro_model, 'forward'), "ExposureAwareDRO missing forward method"
    assert hasattr(pdif_model, 'forward'), "PDIF missing forward method"
    assert hasattr(pdif_model, 'resample_training_data'), "PDIF missing resample_training_data method"
    
    print(f"  ✅ Required methods present")
    
    return True


def main():
    """Run all tests."""
    print("🚀 Testing New Baseline Models (2024-2025)")
    print("=" * 50)
    
    try:
        # Test individual models
        test_exposure_dro()
        test_pdif()
        test_integration()
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("\n📋 Summary:")
        print("  • Exposure-aware DRO (Yang et al., 2024): ✅ Working")
        print("  • PDIF (Zhang et al., 2025): ✅ Working")
        print("  • Integration with existing framework: ✅ Working")
        print("\n🎯 Ready to run experiments with new baselines!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

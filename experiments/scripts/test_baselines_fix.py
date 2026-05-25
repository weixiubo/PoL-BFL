#!/usr/bin/env python3
"""
Test script to verify Median and FoolsGold fixes
"""
import torch
import numpy as np
import sys
import os
from collections import OrderedDict as ODict

# Add experiments/scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from utils.baselines import MedianAggregator, FoolsGoldAggregator

def test_median_deterministic():
    """Test Median aggregator with deterministic mode"""
    print("=" * 60)
    print("Testing Median Aggregator with Deterministic Mode")
    print("=" * 60)
    
    # Enable deterministic mode
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Create test models on CUDA
    models = [
        ODict({
            'weight': torch.randn(100, 100).cuda(),
            'bias': torch.randn(100).cuda()
        })
        for _ in range(10)
    ]
    
    # Test aggregation
    median_agg = MedianAggregator()
    try:
        result = median_agg.aggregate(models)
        print(f"✅ Median aggregation successful!")
        print(f"   Result weight shape: {result['weight'].shape}")
        print(f"   Result bias shape: {result['bias'].shape}")
        print(f"   Result on device: {result['weight'].device}")
        return True
    except Exception as e:
        print(f"❌ Median aggregation failed: {e}")
        return False

def test_foolsgold_with_attacks():
    """Test FoolsGold with various attack scenarios"""
    print("\n" + "=" * 60)
    print("Testing FoolsGold Aggregator with Attack Scenarios")
    print("=" * 60)
    
    fg_agg = FoolsGoldAggregator()
    
    # Test 1: Normal models
    print("\n1. Testing with normal models...")
    normal_models = [
        ODict({
            'weight': torch.randn(50, 50).cuda(),
            'bias': torch.randn(50).cuda()
        })
        for _ in range(5)
    ]
    
    try:
        result = fg_agg.aggregate(normal_models)
        print(f"   ✅ Normal models: Success")
    except Exception as e:
        print(f"   ❌ Normal models: Failed - {e}")
        return False
    
    # Test 2: Models with extreme values (potential NaN source)
    print("\n2. Testing with extreme values...")
    extreme_models = [
        ODict({
            'weight': torch.randn(50, 50).cuda() * 1000,
            'bias': torch.randn(50).cuda() * 1000
        })
        for _ in range(5)
    ]
    
    try:
        result = fg_agg.aggregate(extreme_models)
        print(f"   ✅ Extreme values: Success")
    except Exception as e:
        print(f"   ❌ Extreme values: Failed - {e}")
        return False
    
    # Test 3: Models with some zero gradients (lazy training simulation)
    print("\n3. Testing with zero gradients...")
    zero_models = [
        ODict({
            'weight': torch.zeros(50, 50).cuda() if i < 2 else torch.randn(50, 50).cuda(),
            'bias': torch.zeros(50).cuda() if i < 2 else torch.randn(50).cuda()
        })
        for i in range(5)
    ]
    
    try:
        result = fg_agg.aggregate(zero_models)
        print(f"   ✅ Zero gradients: Success")
    except Exception as e:
        print(f"   ❌ Zero gradients: Failed - {e}")
        return False
    
    # Test 4: Identical models (Sybil attack simulation)
    print("\n4. Testing with identical models (Sybil)...")
    base_model = ODict({
        'weight': torch.randn(50, 50).cuda(),
        'bias': torch.randn(50).cuda()
    })
    sybil_models = [base_model for _ in range(3)] + [
        ODict({
            'weight': torch.randn(50, 50).cuda(),
            'bias': torch.randn(50).cuda()
        })
        for _ in range(2)
    ]
    
    try:
        result = fg_agg.aggregate(sybil_models)
        print(f"   ✅ Sybil attack: Success")
    except Exception as e:
        print(f"   ❌ Sybil attack: Failed - {e}")
        return False
    
    print("\n✅ All FoolsGold tests passed!")
    return True

def main():
    print("\n" + "=" * 60)
    print("Baseline Aggregators Fix Verification")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available, tests require GPU")
        return False
    
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    
    # Run tests
    median_ok = test_median_deterministic()
    foolsgold_ok = test_foolsgold_with_attacks()
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Median Aggregator: {'✅ PASS' if median_ok else '❌ FAIL'}")
    print(f"FoolsGold Aggregator: {'✅ PASS' if foolsgold_ok else '❌ FAIL'}")
    
    if median_ok and foolsgold_ok:
        print("\n🎉 All tests passed! Fixes are working correctly.")
        return True
    else:
        print("\n⚠️ Some tests failed. Please review the fixes.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)


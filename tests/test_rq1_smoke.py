#!/usr/bin/env python
"""
Smoke test for RQ1 implementation
Tests that PoL-FL integration works correctly
"""

import sys
import os
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'experiments' / 'scripts' / 'utils'))
sys.path.insert(0, str(Path(__file__).parent / 'experiments'))

import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pol_integration_import():
    """Test that PoL integration module can be imported"""
    try:
        from experiments.scripts.utils.pol_integration import PoLExperimentHelper
        logger.info("[PASS] PoLExperimentHelper imported successfully")
        return True
    except Exception as e:
        logger.error(f"[FAIL] Failed to import PoLExperimentHelper: {e}")
        return False

def test_pol_trainer_setup():
    """Test that PoL trainer can be set up"""
    try:
        from experiments.scripts.utils.pol_integration import PoLExperimentHelper
        from experiments.scripts.utils.models import create_model
        import torch.utils.data as data
        import torch.nn as nn

        # Create deterministic-shape test data.
        test_data = torch.randn(10, 1, 28, 28)
        test_labels = torch.randint(0, 10, (10,))
        dataset = data.TensorDataset(test_data, test_labels)
        dataloader = data.DataLoader(dataset, batch_size=2)

        # Create model
        model = create_model('SimpleCNN', num_classes=10, input_channels=1)
        criterion = nn.CrossEntropyLoss()

        # Setup PoL trainer
        pol_config = {
            'learning_rate': 0.01,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'save_freq': 10,
            'save_dir': './test_pol_data',
        }

        trainer = PoLExperimentHelper.setup_pol_trainer(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            client_id='test_client',
            pol_config=pol_config,
            device='cpu'
        )

        logger.info("[PASS] PoL trainer setup successfully")
        return True
    except Exception as e:
        logger.error(f"[FAIL] Failed to setup PoL trainer: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pol_aggregator_setup():
    """Test that PoL aggregator can be set up"""
    try:
        from experiments.scripts.utils.pol_integration import PoLExperimentHelper
        from experiments.scripts.utils.models import create_model

        # Create model
        model = create_model('SimpleCNN', num_classes=10, input_channels=1)

        # Setup PoL aggregator
        pol_config = {
            'verification_rate': 0.3,
            'delta': 10.0,
            'distance_metric': 'l2',
            'use_top_q': False,
            'top_q': 5,
            'enable_zkp': False,
            'zkp_use_simulation': True,
        }

        aggregator = PoLExperimentHelper.setup_pol_aggregator(
            model=model,
            pol_config=pol_config,
            device='cpu'
        )

        logger.info("[PASS] PoL aggregator setup successfully")
        return True
    except Exception as e:
        logger.error(f"[FAIL] Failed to setup PoL aggregator: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_detection_metrics():
    """Test detection metrics computation"""
    try:
        from experiments.scripts.utils.pol_integration import PoLExperimentHelper

        # Test data
        verification_results = {
            'client_0': False,  # Detected as malicious
            'client_1': True,   # Passed verification
            'client_2': True,   # Passed verification
            'client_3': False,  # Detected as malicious
        }

        malicious_clients = ['client_0', 'client_3']
        all_clients = ['client_0', 'client_1', 'client_2', 'client_3']

        metrics = PoLExperimentHelper.compute_detection_metrics(
            verification_results,
            malicious_clients,
            all_clients
        )

        logger.info(f"[PASS] Detection metrics computed: TPR={metrics['TPR']:.2f}, FPR={metrics['FPR']:.2f}")

        # Verify correctness
        assert metrics['TPR'] == 1.0, f"Expected TPR=1.0, got {metrics['TPR']}"
        assert metrics['FPR'] == 0.0, f"Expected FPR=0.0, got {metrics['FPR']}"
        assert metrics['Precision'] == 1.0, f"Expected Precision=1.0, got {metrics['Precision']}"

        logger.info("[PASS] Detection metrics are correct")
        return True
    except Exception as e:
        logger.error(f"[FAIL] Failed to compute detection metrics: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    logger.info("="*70)
    logger.info("RQ1 Smoke Test Suite")
    logger.info("="*70)

    tests = [
        ("PoL Integration Import", test_pol_integration_import),
        ("PoL Trainer Setup", test_pol_trainer_setup),
        ("PoL Aggregator Setup", test_pol_aggregator_setup),
        ("Detection Metrics", test_detection_metrics),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\nRunning: {test_name}")
        result = test_func()
        results.append((test_name, result))

    # Summary
    logger.info("\n" + "="*70)
    logger.info("Test Summary")
    logger.info("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[PASS] PASS" if result else "[FAIL] FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")
    logger.info("="*70)

    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

"""
Test Experiment Infrastructure

Smoke test to verify all experiment components work correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test all imports"""
    logger.info("Testing imports...")

    try:
        from experiment_config import RQ1_CONFIG, set_random_seed
        logger.info("[PASS] experiment_config")

        from data_utils import load_dataset, partition_data_iid
        logger.info("[PASS] data_utils")

        from models import create_model
        logger.info("[PASS] models")

        from metrics import MetricsTracker, compute_accuracy
        logger.info("[PASS] metrics")

        from baselines import create_aggregator
        logger.info("[PASS] baselines")

        from attacks.byzantine_attacks import create_attack
        logger.info("[PASS] byzantine_attacks")

        from attacks.free_riding_attacks import create_free_riding_attack
        logger.info("[PASS] free_riding_attacks")

        logger.info("All imports successful.")
        return True

    except Exception as e:
        logger.error(f"Import failed: {e}")
        return False


def test_data_loading():
    """Test data loading"""
    logger.info("\nTesting data loading...")

    try:
        from data_utils import load_dataset, partition_data_iid, get_data_statistics

        # Load MNIST
        dataset = load_dataset('MNIST', train=True)
        logger.info(f"[PASS] Loaded MNIST: {len(dataset)} samples")

        # Partition data
        client_datasets = partition_data_iid(dataset, num_clients=5)
        logger.info(f"[PASS] Partitioned into {len(client_datasets)} clients")

        # Get statistics
        stats = get_data_statistics(client_datasets)
        logger.info(f"[PASS] Statistics: {stats['num_clients']} clients, {stats['total_samples']} samples")

        return True

    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model creation"""
    logger.info("\nTesting model creation...")

    try:
        from models import create_model, count_parameters

        # Create SimpleCNN
        model = create_model('SimpleCNN', num_classes=10, input_channels=1)
        num_params = count_parameters(model)
        logger.info(f"[PASS] Created SimpleCNN: {num_params:,} parameters")

        # Test forward pass
        x = torch.randn(2, 1, 28, 28)
        y = model(x)
        logger.info(f"[PASS] Forward pass: input {x.shape} -> output {y.shape}")

        return True

    except Exception as e:
        logger.error(f"Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_aggregators():
    """Test aggregators"""
    logger.info("\nTesting aggregators...")

    try:
        from baselines import create_aggregator
        from models import create_model
        from collections import OrderedDict

        # Create deterministic test models.
        model1 = create_model('SimpleCNN', num_classes=10, input_channels=1)
        model2 = create_model('SimpleCNN', num_classes=10, input_channels=1)

        models = [model1.state_dict(), model2.state_dict()]

        # Test FedAvg
        aggregator = create_aggregator('Vanilla_FL')
        aggregated = aggregator.aggregate(models)
        logger.info(f"[PASS] FedAvg aggregation: {len(aggregated)} parameters")

        # Test Krum
        aggregator = create_aggregator('Krum', num_byzantine=0)
        aggregated = aggregator.aggregate(models)
        logger.info(f"[PASS] Krum aggregation: {len(aggregated)} parameters")

        return True

    except Exception as e:
        logger.error(f"Aggregator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_attacks():
    """Test attacks"""
    logger.info("\nTesting attacks...")

    try:
        from attacks.byzantine_attacks import create_attack
        from attacks.free_riding_attacks import create_free_riding_attack
        from models import create_model

        model = create_model('SimpleCNN', num_classes=10, input_channels=1)
        model_state = model.state_dict()

        # Test Byzantine attacks
        attack = create_attack('random_noise', noise_scale=1.0)
        attacked_state = attack.apply(model_state)
        logger.info(f"[PASS] Random noise attack: {len(attacked_state)} parameters")

        attack = create_attack('model_replacement', replacement_type='random')
        attacked_state = attack.apply(model_state)
        logger.info(f"[PASS] Model replacement attack: {len(attacked_state)} parameters")

        # Test free-riding attacks
        attack = create_free_riding_attack('no_training')
        should_train = attack.should_train()
        logger.info(f"[PASS] No training attack: should_train={should_train}")

        attack = create_free_riding_attack('lazy_training', lazy_epochs=1, required_epochs=5)
        epochs = attack.get_training_epochs()
        logger.info(f"[PASS] Lazy training attack: {epochs} epochs")

        return True

    except Exception as e:
        logger.error(f"Attack test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test metrics"""
    logger.info("\nTesting metrics...")

    try:
        from metrics import MetricsTracker, compute_accuracy, Timer
        from models import create_model
        from data_utils import load_dataset
        import torch.utils.data as data

        # Test MetricsTracker
        tracker = MetricsTracker()
        tracker.add_metric('accuracy', 0.85, round_num=1)
        tracker.add_metric('accuracy', 0.90, round_num=2)
        latest = tracker.get_latest('accuracy')
        logger.info(f"[PASS] MetricsTracker: latest accuracy = {latest}")

        # Test Timer
        timer = Timer()
        timer.start()
        import time
        time.sleep(0.1)
        elapsed = timer.stop()
        logger.info(f"[PASS] Timer: elapsed = {elapsed:.3f}s")

        # Test accuracy computation
        model = create_model('SimpleCNN', num_classes=10, input_channels=1)
        dataset = load_dataset('MNIST', train=False)
        loader = data.DataLoader(dataset, batch_size=32, shuffle=False)

        # Take only first batch for smoke test
        device = torch.device('cpu')
        model.eval()
        data_batch, target_batch = next(iter(loader))
        with torch.no_grad():
            output = model(data_batch.to(device))
            pred = output.argmax(dim=1)
            correct = pred.eq(target_batch.to(device)).sum().item()
            acc = correct / len(target_batch)

        logger.info(f"[PASS] Accuracy computation: {acc:.4f} (random model)")

        return True

    except Exception as e:
        logger.error(f"Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("="*70)
    logger.info("Testing Experiment Infrastructure")
    logger.info("="*70)

    tests = [
        ("Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation),
        ("Aggregators", test_aggregators),
        ("Attacks", test_attacks),
        ("Metrics", test_metrics)
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            logger.error(f"Test {name} crashed: {e}")
            results.append((name, False))

    # Print summary
    logger.info("\n" + "="*70)
    logger.info("Test Summary")
    logger.info("="*70)

    for name, success in results:
        status = "[PASS] PASS" if success else "[FAIL] FAIL"
        logger.info(f"{name:<30} {status}")

    total_pass = sum(1 for _, success in results if success)
    total_tests = len(results)

    logger.info("="*70)
    logger.info(f"Total: {total_pass}/{total_tests} tests passed")
    logger.info("="*70)

    return all(success for _, success in results)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

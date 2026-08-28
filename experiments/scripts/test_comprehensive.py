"""
Comprehensive Test Suite for SOTA Integration

Exercises the attack, aggregation, integration, and edge-case components:
1. All Blades attacks (ALIE, IPM, MinMax)
2. All original attacks (Random Noise, Label Flipping, etc.)
3. All aggregators (Vanilla, Krum, Trimmed Mean, Median, ShapleyFL, FoolsGold)
4. Integration with PoL-BFL framework
"""

import os
import sys
import torch
import logging
from pathlib import Path
from collections import OrderedDict

# Add parent directory to path
scripts_dir = Path(__file__).parent.parent
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(scripts_dir / 'utils'))

from utils.baselines import create_aggregator
from attacks.byzantine_attacks import create_attack
from attacks.free_riding_attacks import create_free_riding_attack
from attacks.sybil_attacks import create_sybil_attack

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_model(num_params: int = 100) -> OrderedDict:
    """Create a test model for testing"""
    model = OrderedDict()
    model['layer1.weight'] = torch.randn(10, 10)
    model['layer1.bias'] = torch.randn(10)
    model['layer2.weight'] = torch.randn(10, 10)
    model['layer2.bias'] = torch.randn(10)
    return model


def test_all_attacks():
    """Test all attack types"""
    logger.info("=" * 80)
    logger.info("Testing All Attacks")
    logger.info("=" * 80)

    model_state = create_test_model()
    global_model = create_test_model()

    # Byzantine attacks
    byzantine_attacks = {
        'random_noise': {'noise_scale': 1.0},
        'label_flipping': {},
        'model_replacement': {},
        'gradient_inversion': {},
        'alie': {'z_max': 2.5},
        'ipm': {'scale': 1.0},
        'minmax': {'lambda_init': 1.0},
    }

    logger.info("\n1. Testing Byzantine Attacks")
    for attack_name, params in byzantine_attacks.items():
        try:
            attack = create_attack(attack_name, **params)
            attacked_state = attack.apply(model_state, global_model=global_model)
            logger.info(f"  [PASS] {attack_name}: PASSED")
        except Exception as e:
            logger.error(f"  [FAIL] {attack_name}: FAILED - {e}")
            return False

    # Free-riding attacks
    free_riding_attacks = {
        'no_training': {},
        'lazy_training': {'lazy_epochs': 1, 'required_epochs': 5},
        'minimal_update': {'noise_scale': 1e-5},
    }

    logger.info("\n2. Testing Free-riding Attacks")
    for attack_name, params in free_riding_attacks.items():
        try:
            attack = create_free_riding_attack(attack_name, **params)
            # Free-riding attacks have different interface - they only take global_model
            if attack_name in ['no_training', 'minimal_update']:
                attacked_state = attack.apply(global_model)
            else:  # lazy_training
                # Verify the should_train() decision.
                should_train = attack.should_train()
            logger.info(f"  [PASS] {attack_name}: PASSED")
        except Exception as e:
            logger.error(f"  [FAIL] {attack_name}: FAILED - {e}")
            return False

    # Sybil attack (different interface - creates multiple identities)
    logger.info("\n3. Testing Sybil Attack")
    try:
        attack = create_sybil_attack(num_identities=5)
        sybil_identities = attack.create_identities()
        shared_update = attack.get_shared_model_update(model_state)
        logger.info(f"  [PASS] sybil: PASSED (generated {len(sybil_identities)} identities)")
    except Exception as e:
        logger.error(f"  [FAIL] sybil: FAILED - {e}")
        return False

    logger.info("\n[PASS] All attacks passed.")
    return True


def test_all_aggregators():
    """Test all aggregator types"""
    logger.info("\n" + "=" * 80)
    logger.info("Testing All Aggregators")
    logger.info("=" * 80)

    models = [create_test_model() for _ in range(10)]

    aggregators = {
        'Vanilla_FL': {},
        'Krum': {},
        'Trimmed_Mean': {},
        'Median': {},
        'ShapleyFL': {'threshold_percentile': 0.0},
        'FoolsGold': {},
    }

    for agg_name, params in aggregators.items():
        try:
            aggregator = create_aggregator(agg_name, **params)
            aggregated = aggregator.aggregate(models)
            logger.info(f"  [PASS] {agg_name}: PASSED")
        except Exception as e:
            logger.error(f"  [FAIL] {agg_name}: FAILED - {e}")
            import traceback
            traceback.print_exc()
            return False

    logger.info("\n[PASS] All aggregators passed.")
    return True


def test_attack_aggregator_combinations():
    """Test combinations of attacks and aggregators"""
    logger.info("\n" + "=" * 80)
    logger.info("Testing Attack-Aggregator Combinations")
    logger.info("=" * 80)

    # Create benign and malicious models
    benign_models = [create_test_model() for _ in range(8)]
    global_model = create_test_model()

    # Test a few key combinations
    test_cases = [
        ('alie', {'z_max': 2.5}, 'Vanilla_FL', {}),
        ('alie', {'z_max': 2.5}, 'ShapleyFL', {'threshold_percentile': 0.0}),
        ('ipm', {'scale': 1.0}, 'FoolsGold', {}),
        ('minmax', {'lambda_init': 1.0}, 'Krum', {}),
    ]

    for attack_name, attack_params, agg_name, agg_params in test_cases:
        try:
            # Create attack
            attack = create_attack(attack_name, **attack_params)

            # Generate malicious models
            malicious_models = []
            for _ in range(2):
                model_state = create_test_model()
                attacked_state = attack.apply(model_state, global_model=global_model)
                malicious_models.append(attacked_state)

            # Combine benign and malicious
            all_models = benign_models + malicious_models

            # Aggregate
            aggregator = create_aggregator(agg_name, **agg_params)
            aggregated = aggregator.aggregate(all_models)

            logger.info(f"  [PASS] {attack_name} + {agg_name}: PASSED")
        except Exception as e:
            logger.error(f"  [FAIL] {attack_name} + {agg_name}: FAILED - {e}")
            import traceback
            traceback.print_exc()
            return False

    logger.info("\n[PASS] All combinations passed.")
    return True


def test_edge_cases():
    """Test edge cases and error handling"""
    logger.info("\n" + "=" * 80)
    logger.info("Testing Edge Cases")
    logger.info("=" * 80)

    # Test 1: Empty model list
    logger.info("\n1. Testing empty model list")
    try:
        aggregator = create_aggregator('Vanilla_FL')
        result = aggregator.aggregate([])
        logger.info("  [WARNING]  Empty list handled (returned empty dict)")
    except Exception as e:
        logger.info(f"  [PASS] Empty list raises exception (expected): {type(e).__name__}")

    # Test 2: Single model
    logger.info("\n2. Testing single model")
    try:
        aggregator = create_aggregator('Vanilla_FL')
        models = [create_test_model()]
        result = aggregator.aggregate(models)
        logger.info("  [PASS] Single model aggregation: PASSED")
    except Exception as e:
        logger.error(f"  [FAIL] Single model aggregation: FAILED - {e}")
        return False

    # Test 3: All malicious (worst case for ShapleyFL)
    logger.info("\n3. Testing all malicious models (ShapleyFL)")
    try:
        attack = create_attack('alie', z_max=2.5)
        global_model = create_test_model()
        malicious_models = []
        for _ in range(5):
            model_state = create_test_model()
            attacked_state = attack.apply(model_state, global_model=global_model)
            malicious_models.append(attacked_state)

        aggregator = create_aggregator('ShapleyFL', threshold_percentile=0.0)
        result = aggregator.aggregate(malicious_models)
        logger.info("  [PASS] All malicious models: PASSED")
    except Exception as e:
        logger.error(f"  [FAIL] All malicious models: FAILED - {e}")
        return False

    logger.info("\n[PASS] All edge cases passed.")
    return True


def main():
    """Run all tests"""
    logger.info("=" * 80)
    logger.info("Comprehensive SOTA Integration Test Suite")
    logger.info("=" * 80)

    results = {
        'All Attacks': test_all_attacks(),
        'All Aggregators': test_all_aggregators(),
        'Attack-Aggregator Combinations': test_attack_aggregator_combinations(),
        'Edge Cases': test_edge_cases(),
    }

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Test Summary")
    logger.info("=" * 80)

    all_passed = True
    for test_name, passed in results.items():
        status = "[PASS] PASSED" if passed else "[FAIL] FAILED"
        logger.info(f"{test_name}: {status}")
        if not passed:
            all_passed = False

    logger.info("=" * 80)

    if all_passed:
        logger.info("[PASS] All comprehensive tests passed.")
        logger.info("System is ready for RQ1 and RQ5 experiments.")
        return 0
    else:
        logger.error("[FAIL] Some tests failed; experiments were not started.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

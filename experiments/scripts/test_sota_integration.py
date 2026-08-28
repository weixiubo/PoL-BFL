"""
Test SOTA Integration

Smoke test to verify that all new SOTA components work correctly:
- Blades attacks (ALIE, IPM, MinMax)
- ShapleyFL aggregator
- FoolsGold aggregator
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


def test_blades_attacks():
    """Test Blades framework attacks"""
    logger.info("=" * 60)
    logger.info("Testing Blades Attacks")
    logger.info("=" * 60)

    # Create test model
    model_state = create_test_model()
    global_model = create_test_model()

    # Test ALIE attack
    logger.info("\n1. Testing ALIE Attack (NeurIPS 2019)")
    try:
        alie = create_attack('alie', z_max=2.5)
        attacked_state = alie.apply(model_state, global_model=global_model)
        logger.info(f"[PASS] ALIE attack successful")
        logger.info(f"   Output keys: {list(attacked_state.keys())}")
    except Exception as e:
        logger.error(f"[FAIL] ALIE attack failed: {e}")
        return False

    # Test IPM attack
    logger.info("\n2. Testing IPM Attack (UAI 2020)")
    try:
        ipm = create_attack('ipm', scale=1.0)
        attacked_state = ipm.apply(model_state, global_model=global_model)
        logger.info(f"[PASS] IPM attack successful")
        logger.info(f"   Output keys: {list(attacked_state.keys())}")
    except Exception as e:
        logger.error(f"[FAIL] IPM attack failed: {e}")
        return False

    # Test MinMax attack
    logger.info("\n3. Testing MinMax Attack (NDSS 2021)")
    try:
        minmax = create_attack('minmax', lambda_init=1.0)
        attacked_state = minmax.apply(model_state, global_model=global_model)
        logger.info(f"[PASS] MinMax attack successful")
        logger.info(f"   Output keys: {list(attacked_state.keys())}")
    except Exception as e:
        logger.error(f"[FAIL] MinMax attack failed: {e}")
        return False

    logger.info("\n[PASS] All Blades attacks passed.")
    return True


def test_shapley_fl_aggregator():
    """Test ShapleyFL aggregator"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing ShapleyFL Aggregator (KDD 2023)")
    logger.info("=" * 60)

    try:
        # Create test models
        models = [create_test_model() for _ in range(5)]

        # Create ShapleyFL aggregator
        aggregator = create_aggregator('ShapleyFL', threshold_percentile=0.0)

        # Aggregate
        aggregated = aggregator.aggregate(models)

        logger.info(f"[PASS] ShapleyFL aggregation successful")
        logger.info(f"   Input: {len(models)} models")
        logger.info(f"   Output keys: {list(aggregated.keys())}")

        return True
    except Exception as e:
        logger.error(f"[FAIL] ShapleyFL aggregation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fools_gold_aggregator():
    """Test FoolsGold aggregator"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing FoolsGold Aggregator (RAID 2020)")
    logger.info("=" * 60)

    try:
        # Create test models
        models = [create_test_model() for _ in range(5)]

        # Create FoolsGold aggregator
        aggregator = create_aggregator('FoolsGold')

        # Aggregate
        aggregated = aggregator.aggregate(models)

        logger.info(f"[PASS] FoolsGold aggregation successful")
        logger.info(f"   Input: {len(models)} models")
        logger.info(f"   Output keys: {list(aggregated.keys())}")

        return True
    except Exception as e:
        logger.error(f"[FAIL] FoolsGold aggregation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_original_aggregators():
    """Test original aggregators still work"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Original Aggregators")
    logger.info("=" * 60)

    models = [create_test_model() for _ in range(5)]

    aggregators_to_test = ['Vanilla_FL', 'Krum', 'Trimmed_Mean', 'Median']

    for agg_name in aggregators_to_test:
        try:
            aggregator = create_aggregator(agg_name)
            aggregated = aggregator.aggregate(models)
            logger.info(f"[PASS] {agg_name} aggregation successful")
        except Exception as e:
            logger.error(f"[FAIL] {agg_name} aggregation failed: {e}")
            return False

    return True


def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("SOTA Integration Test Suite")
    logger.info("=" * 60)

    results = {
        'Blades Attacks': test_blades_attacks(),
        'ShapleyFL Aggregator': test_shapley_fl_aggregator(),
        'FoolsGold Aggregator': test_fools_gold_aggregator(),
        'Original Aggregators': test_original_aggregators()
    }

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "[PASS] PASSED" if passed else "[FAIL] FAILED"
        logger.info(f"{test_name}: {status}")
        if not passed:
            all_passed = False

    logger.info("=" * 60)

    if all_passed:
        logger.info("[PASS] All tests passed.")
        return 0
    else:
        logger.error("[FAIL] Some tests failed")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)


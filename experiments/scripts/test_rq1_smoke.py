"""
Smoke RQ1 Test

Fast test of RQ1 experiment with:
- MNIST dataset
- 5 rounds
- 5 clients
- 1 attack type (byzantine_alie)
- 2 baselines (Vanilla_FL, ShapleyFL)

This is to verify the full experiment pipeline works with new SOTA methods.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path
scripts_dir = Path(__file__).parent.parent
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(scripts_dir / 'utils'))

# Set environment variables before importing
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run smoke RQ1 test"""
    logger.info("=" * 60)
    logger.info("Smoke RQ1 Test - SOTA Integration Verification")
    logger.info("=" * 60)

    # Import run_rq1_security
    from runners.run_rq1_security import main as run_rq1_main

    # Override sys.argv to pass arguments
    sys.argv = [
        'test_rq1_smoke.py',
        '--dataset', 'MNIST',
        '--num_rounds', '5',
        '--num_clients', '5',
        '--clients_per_round', '3',
        '--attacks', 'byzantine_alie',  # Test one Blades attack
        '--baselines', 'Vanilla_FL,ShapleyFL',  # Comma-separated
    ]

    logger.info("Configuration:")
    logger.info("  Dataset: MNIST")
    logger.info("  Rounds: 5")
    logger.info("  Clients: 5")
    logger.info("  Clients per round: 3")
    logger.info("  Attack: byzantine_alie (NeurIPS 2019)")
    logger.info("  Baselines: Vanilla_FL, ShapleyFL (KDD 2023)")
    logger.info("")

    try:
        # Run RQ1 experiment
        run_rq1_main()

        logger.info("=" * 60)
        logger.info("[PASS] Smoke RQ1 test completed successfully.")
        logger.info("=" * 60)
        return 0

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"[FAIL] Smoke RQ1 test failed: {e}")
        logger.error("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)


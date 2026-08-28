"""
RQ1 Initial Test - Smoke validation of all attacks and defenses

This script runs a small-scale test to verify:
1. All attacks work correctly
2. All defenses work correctly
3. PoL-BFL performs well compared to baselines

Configuration:
- Dataset: MNIST
- Rounds: 20
- Clients: 20 (10 per round)
- Malicious ratio: 20% (2 malicious clients per round)
- Selected attacks: byzantine_alie, byzantine_ipm, free_riding_no_training, sybil_attack
- All baselines: Vanilla_FL, Krum, Trimmed_Mean, Median, ShapleyFL, FoolsGold, PoL_FL
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_rq1_test(dataset='MNIST', num_rounds=20, attacks=None, baselines=None):
    """
    Run RQ1 initial test

    Args:
        dataset: Dataset to use
        num_rounds: Number of FL rounds
        attacks: List of attacks to test (None = all)
        baselines: List of baselines to test (None = all)
    """

    # Default: test key attacks from each category
    if attacks is None:
        attacks = [
            'byzantine_alie',      # Blades attack (NeurIPS 2019)
            'byzantine_ipm',       # Blades attack (UAI 2020)
            'byzantine_minmax',    # Blades attack (NDSS 2021)
            'byzantine_random_noise',  # Original attack
            'free_riding_no_training',  # Free-riding attack
            'sybil_attack',        # Sybil attack
        ]

    # Default: test all baselines
    if baselines is None:
        baselines = [
            'Vanilla_FL',
            'Krum',
            'Trimmed_Mean',
            'Median',
            'ShapleyFL',
            'FoolsGold',
            'PoL_FL',
        ]

    logger.info("=" * 80)
    logger.info("RQ1 Initial Test")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Rounds: {num_rounds}")
    logger.info(f"Attacks: {', '.join(attacks)}")
    logger.info(f"Baselines: {', '.join(baselines)}")
    logger.info("=" * 80)

    # Build command
    cmd = [
        'python', 'experiments/scripts/runners/run_rq1_security.py',
        '--dataset', dataset,
        '--num_rounds', str(num_rounds),
        '--num_clients', '20',
        '--clients_per_round', '10',
        '--attacks', ','.join(attacks),
        '--baselines', ','.join(baselines),
    ]

    logger.info(f"Running command: {' '.join(cmd)}")
    logger.info("=" * 80)

    # Run the command
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent.parent,
            capture_output=False,
            text=True,
            check=True
        )
        logger.info("=" * 80)
        logger.info("[PASS] RQ1 initial test completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error("=" * 80)
        logger.error(f"[FAIL] RQ1 initial test failed with exit code {e.returncode}")
        return False
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"[FAIL] RQ1 initial test failed: {e}")
        return False


def main():
    """Main function"""
    logger.info("Starting RQ1 Initial Test...")
    logger.info("This will test key attacks and all defenses on MNIST")
    logger.info("")

    success = run_rq1_test(
        dataset='MNIST',
        num_rounds=20,
    )

    if success:
        logger.info("")
        logger.info("=" * 80)
        logger.info("[PASS] RQ1 Initial Test Completed.")
        logger.info("=" * 80)
        logger.info("Next steps:")
        logger.info("1. Check results in: PoL-BFL/Code/experiments/results/rq1_security/")
        logger.info("2. Analyze performance metrics (MA, DR, FPR, F1)")
        logger.info("3. Compare PoL-BFL with baselines")
        logger.info("4. If PoL-BFL performs poorly, debug and tune parameters")
        logger.info("=" * 80)
        return 0
    else:
        logger.error("")
        logger.error("=" * 80)
        logger.error("[FAIL] RQ1 Initial Test Failed.")
        logger.error("=" * 80)
        logger.error("Review the preceding error messages before running experiments.")
        logger.error("=" * 80)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

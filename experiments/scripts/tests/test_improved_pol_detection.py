"""
Test script for improved PoL detection
Tests that PoL can detect Byzantine attacks (noise injection after training)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_rq2_ablation import AblationStudyExperiment
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("=" * 70)
    logger.info("Testing Improved PoL Detection")
    logger.info("=" * 70)
    logger.info("Config: MNIST, 1 round, pol_only variant")
    logger.info("Expected: PoL should detect Byzantine attacks (noise injection)")
    logger.info("")
    
    # Create experiment with minimal config
    config = {
        'dataset': 'MNIST',
        'model': 'SimpleCNN',
        'num_clients': 20,
        'clients_per_round': 10,
        'num_rounds': 1,
        'malicious_ratio': 0.2,  # 20% malicious
        'noise_scale': 1.0,
        'variants': ['pol_only'],
        'repetitions': 1,
    }
    
    exp = AblationStudyExperiment(config)
    exp.prepare_data()

    # Run single experiment with pol_only variant
    result = exp._run_single_experiment('pol_only', repetition=0)

    logger.info("=" * 70)
    logger.info("Test Results:")
    logger.info(f"  Final Accuracy: {result['test_accuracies'][-1]:.4f}")

    if result['detection_metrics']:
        metrics = result['detection_metrics'][-1]
        logger.info(f"  Detection TPR: {metrics['tpr']:.4f}")
        logger.info(f"  Detection FPR: {metrics['fpr']:.4f}")
        logger.info(f"  Participation Rate: {result['participation_rates'][-1]:.4f}")

    logger.info("=" * 70)

    # Check if detection is working
    if result['detection_metrics']:
        tpr = result['detection_metrics'][-1]['tpr']
        fpr = result['detection_metrics'][-1]['fpr']

        if tpr > 0.5:  # At least 50% detection rate
            logger.info("✅ SUCCESS: PoL is detecting Byzantine attacks!")
            logger.info(f"   TPR = {tpr:.2%} (malicious nodes detected)")
            logger.info(f"   FPR = {fpr:.2%} (false positives)")
        else:
            logger.warning("⚠️  WARNING: Low detection rate!")
            logger.warning(f"   TPR = {tpr:.2%} (should be > 50%)")
            logger.warning("   This suggests PoL is not effectively detecting Byzantine attacks")
    else:
        logger.error("❌ ERROR: No detection metrics available!")


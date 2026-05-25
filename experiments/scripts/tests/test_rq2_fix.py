"""
Quick test to verify RQ2 detection fix
Run 1 round with pol_only variant to check if detection works
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_rq2_ablation import AblationStudyExperiment
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Quick test config: 1 round, 1 repetition
    test_config = {
        'dataset': 'MNIST',
        'model': 'SimpleCNN',
        'num_clients': 20,
        'clients_per_round': 10,
        'num_rounds': 1,
        'num_repetitions': 1,
        'data_distribution': 'NonIID_Dirichlet',
        'attack_type': 'byzantine_random_noise',
        'malicious_ratio': 0.2,
        'noise_scale': 1.0,
    }
    
    logger.info("="*70)
    logger.info("Testing RQ2 Detection Fix")
    logger.info("="*70)
    logger.info(f"Config: {test_config['dataset']}, {test_config['num_rounds']} round, pol_only variant")
    
    experiment = AblationStudyExperiment(test_config)
    experiment.prepare_data()

    # Test only pol_only variant
    result = experiment._run_single_experiment('pol_only', repetition=0)
    
    logger.info("="*70)
    logger.info("Test Results:")
    logger.info(f"  Final Accuracy: {result['final_accuracy']:.4f}")
    logger.info(f"  Detection TPR: {result['detection_metrics']['TPR']:.4f}")
    logger.info(f"  Detection FPR: {result['detection_metrics']['FPR']:.4f}")
    logger.info(f"  Participation Rate: {result['participation_rate']:.4f}")
    logger.info("="*70)
    
    # Check if detection is working
    if result['detection_metrics']['TPR'] > 0 or result['detection_metrics']['FPR'] > 0:
        logger.info("✅ SUCCESS: Detection is working!")
    else:
        logger.warning("⚠️ WARNING: Detection metrics still 0, may need further investigation")
    
    if result['participation_rate'] > 0:
        logger.info("✅ SUCCESS: Participation rate is working!")
    else:
        logger.warning("⚠️ WARNING: Participation rate still 0, may need further investigation")


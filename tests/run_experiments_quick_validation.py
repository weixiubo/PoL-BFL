#!/usr/bin/env python
"""
Quick validation of all RQ experiments
Runs simplified versions (MNIST, 5 rounds) to verify functionality
"""

import sys
import os
from pathlib import Path
import logging

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'experiments' / 'scripts' / 'utils'))
sys.path.insert(0, str(Path(__file__).parent / 'experiments'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_rq1_quick():
    """Run RQ1 quick validation"""
    logger.info("\n" + "="*70)
    logger.info("RQ1: Security Evaluation (Quick Validation)")
    logger.info("="*70)
    
    try:
        from experiments.scripts.runners.run_rq1_security import SecurityExperiment
        
        config = {
            'dataset': 'MNIST',
            'model': 'SimpleCNN',
            'num_clients': 10,
            'clients_per_round': 5,
            'num_rounds': 3,  # Quick: 3 rounds
            'data_distribution': 'NonIID_Dirichlet',
            'attacks': {
                'no_attack': {'malicious_ratios': [0.0]},
                'byzantine_random_noise': {'malicious_ratios': [0.2], 'noise_scale': 1.0},
            },
            'baselines': ['Vanilla_FL', 'PoL_FL'],
            'pol_config': {
                'enable': True,
                'save_freq': 10,
                'verification_rate': 0.3,
                'delta': 10.0,
                'distance_metric': 'l2',
                'use_top_q': False,
                'top_q': 5,
                'enable_zkp': False,
                'zkp_use_simulation': True,
            }
        }
        
        exp = SecurityExperiment(config)
        results = exp.run_all_experiments()
        
        logger.info(f"✓ RQ1 completed with {len(results)} experiments")
        return True
    except Exception as e:
        logger.error(f"✗ RQ1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_rq2_quick():
    """Run RQ2 quick validation"""
    logger.info("\n" + "="*70)
    logger.info("RQ2: Ablation Study (Quick Validation)")
    logger.info("="*70)
    
    try:
        from experiments.scripts.runners.run_rq2_ablation import AblationStudyExperiment
        
        config = {
            'dataset': 'MNIST',
            'model': 'SimpleCNN',
            'num_clients': 10,
            'clients_per_round': 5,
            'num_rounds': 3,  # Quick: 3 rounds
            'data_distribution': 'NonIID_Dirichlet',
            'attack_type': 'byzantine_random_noise',
            'malicious_ratio': 0.2,
            'noise_scale': 1.0,
            'variants': ['vanilla_fl', 'pol_only'],  # Quick: 2 variants
            'num_repetitions': 1,  # Quick: 1 repetition
        }
        
        exp = AblationStudyExperiment(config)
        results = exp.run_all_experiments()
        
        logger.info(f"✓ RQ2 completed with {len(results)} variants")
        return True
    except Exception as e:
        logger.error(f"✗ RQ2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_rq3_quick():
    """Run RQ3 quick validation"""
    logger.info("\n" + "="*70)
    logger.info("RQ3: Overhead Analysis (Quick Validation)")
    logger.info("="*70)
    
    try:
        from experiments.scripts.runners.run_rq3_overhead import OverheadExperiment
        
        config = {
            'dataset': 'MNIST',
            'model': 'SimpleCNN',
            'num_clients': 10,
            'clients_per_round': 5,
            'num_rounds': 2,  # Quick: 2 rounds
            'data_distribution': 'NonIID_Dirichlet',
            'methods': ['Vanilla_FL', 'PoL_FL'],
            'pol_save_freq': 10,
            'checkpoint_dir': './experiments/results/rq3_overhead/checkpoints'
        }
        
        exp = OverheadExperiment(config)
        results = exp.run_all_experiments()
        
        logger.info(f"✓ RQ3 completed with {len(results)} methods")
        return True
    except Exception as e:
        logger.error(f"✗ RQ3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_rq4_quick():
    """Run RQ4 quick validation"""
    logger.info("\n" + "="*70)
    logger.info("RQ4: Incentive Mechanism (Quick Validation)")
    logger.info("="*70)
    
    try:
        from experiments.scripts.runners.run_rq4_incentive import IncentiveExperiment
        
        config = {
            'dataset': 'MNIST',
            'model': 'SimpleCNN',
            'num_clients': 10,
            'clients_per_round': 5,
            'num_rounds': 2,  # Quick: 2 rounds
            'data_distribution': 'NonIID_Dirichlet',
            'node_types': {
                'honest': 0.6,
                'rational': 0.3,
                'malicious': 0.1
            },
            'utility_params': {
                'base_reward': 50,
                'compute_cost': 10,
                'gas_cost': 2,
                'slash_penalty': 100,
                'detection_probability': 0.3
            },
            'scenarios': ['no_incentive', 'fixed_reward'],  # Quick: 2 scenarios
        }
        
        exp = IncentiveExperiment(config)
        results = exp.run_all_experiments()
        
        logger.info(f"✓ RQ4 completed with {len(results)} scenarios")
        return True
    except Exception as e:
        logger.error(f"✗ RQ4 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all quick validations"""
    logger.info("="*70)
    logger.info("Experiment Quick Validation Suite")
    logger.info("="*70)
    
    tests = [
        ("RQ1 Security Evaluation", run_rq1_quick),
        ("RQ2 Ablation Study", run_rq2_quick),
        ("RQ3 Overhead Analysis", run_rq3_quick),
        ("RQ4 Incentive Mechanism", run_rq4_quick),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("Quick Validation Summary")
    logger.info("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nTotal: {passed}/{total} experiments passed")
    logger.info("="*70)
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)


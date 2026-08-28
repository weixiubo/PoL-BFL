#!/usr/bin/env python3
"""
PoL-BFL 完整实验参数评估执行脚本
自动管理所有RQ1-RQ5的轻量化实验
支持并行执行和实时监控
"""

import os
import sys
import json
import subprocess
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import threading
import queue

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPT_DIR = PROJECT_ROOT / 'experiments' / 'scripts' / 'runners'
LOG_DIR = PROJECT_ROOT / 'experiments' / 'logs' / 'parameter_evaluation'
RESULT_DIR = PROJECT_ROOT / 'experiments' / 'results' / 'parameter_evaluation'

# Ensure directories exist
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Environment setup
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONPATH'] = f"{PROJECT_ROOT}:{PROJECT_ROOT}/experiments/scripts/utils"
os.environ['POL_DATA_DIR'] = str(PROJECT_ROOT / 'data')

# Experiment configurations (轻量化)
EXPERIMENTS = [
    # Reduced-scale validation
    {'phase': 1, 'rq': 'rq1', 'dataset': 'CIFAR10', 'rounds': 3, 'gpu': 0, 'tag': 'rq1_cifar10_smoke'},

    # RQ1 full coverage
    {'phase': 2, 'rq': 'rq1', 'dataset': 'MNIST', 'rounds': 10, 'gpu': 0, 'tag': 'rq1_mnist_tuning'},
    {'phase': 2, 'rq': 'rq1', 'dataset': 'CIFAR10', 'rounds': 15, 'gpu': 0, 'tag': 'rq1_cifar10_tuning'},
    {'phase': 2, 'rq': 'rq1', 'dataset': 'CIFAR100', 'rounds': 15, 'gpu': 0, 'tag': 'rq1_cifar100_tuning'},

    # RQ2-RQ4 parallel execution
    {'phase': 3, 'rq': 'rq2', 'dataset': 'MNIST', 'rounds': 10, 'gpu': 1, 'tag': 'rq2_mnist_tuning'},
    {'phase': 3, 'rq': 'rq2', 'dataset': 'CIFAR10', 'rounds': 10, 'gpu': 1, 'tag': 'rq2_cifar10_tuning'},
    {'phase': 3, 'rq': 'rq3', 'dataset': 'MNIST', 'rounds': 5, 'gpu': 1, 'tag': 'rq3_mnist_tuning'},
    {'phase': 3, 'rq': 'rq3', 'dataset': 'CIFAR10', 'rounds': 5, 'gpu': 1, 'tag': 'rq3_cifar10_tuning'},
    {'phase': 3, 'rq': 'rq4', 'dataset': 'MNIST', 'rounds': 20, 'gpu': 1, 'tag': 'rq4_mnist_tuning'},
]

def get_script_name(rq: str) -> str:
    """Get script name for RQ"""
    mapping = {
        'rq1': 'run_rq1_security.py',
        'rq2': 'run_rq2_ablation.py',
        'rq3': 'run_rq3_overhead.py',
        'rq4': 'run_rq4_incentive.py',
    }
    return mapping.get(rq, f'run_{rq}_*.py')

def run_experiment(exp_config: Dict) -> Tuple[str, bool, str]:
    """Run single experiment"""
    rq = exp_config['rq']
    dataset = exp_config['dataset']
    rounds = exp_config['rounds']
    gpu = exp_config['gpu']
    tag = exp_config['tag']

    log_file = LOG_DIR / f"{tag}.log"
    result_subdir = RESULT_DIR / tag

    logger.info(f"Starting {tag} on GPU {gpu} ({rq}, {dataset}, {rounds} rounds)")

    script_name = get_script_name(rq)
    script_path = SCRIPT_DIR / script_name

    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return tag, False, str(log_file)

    cmd = [
        'python', str(script_path),
        '--dataset', dataset,
        '--num_rounds', str(rounds),
        '--output_dir', str(result_subdir),
    ]

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)

    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=7200  # 2 hour timeout
            )

        success = result.returncode == 0
        if success:
            logger.info(f"[PASS] {tag} completed successfully")
        else:
            logger.error(f"[FAIL] {tag} failed with code {result.returncode}")

        return tag, success, str(log_file)

    except subprocess.TimeoutExpired:
        logger.error(f"[FAIL] {tag} timed out")
        return tag, False, str(log_file)
    except Exception as e:
        logger.error(f"[FAIL] {tag} error: {e}")
        return tag, False, str(log_file)

def run_phase_sequential(phase_exps: List[Dict]) -> Dict[str, Dict]:
    """Run experiments in phase sequentially"""
    results = {}
    for exp in phase_exps:
        tag, success, log_file = run_experiment(exp)
        results[tag] = {
            'success': success,
            'log_file': log_file,
            'timestamp': datetime.now().isoformat()
        }
        time.sleep(2)  # Small delay between experiments
    return results

def main():
    """Main function"""
    logger.info("="*70)
    logger.info("[START] PoL-BFL Complete Tuning Experiments")
    logger.info(f"   Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)

    # Group experiments by phase
    phases = {}
    for exp in EXPERIMENTS:
        phase = exp['phase']
        if phase not in phases:
            phases[phase] = []
        phases[phase].append(exp)

    all_results = {}

    # Execute phases sequentially
    for phase in sorted(phases.keys()):
        logger.info(f"\n[PLAN] Phase {phase}: {len(phases[phase])} experiments")

        # Run experiments in phase sequentially
        phase_results = run_phase_sequential(phases[phase])
        all_results.update(phase_results)

        # Wait between phases
        if phase < max(phases.keys()):
            logger.info(f"Phase {phase} completed. Waiting 10s before next phase...")
            time.sleep(10)

    # Summary
    logger.info("\n" + "="*70)
    logger.info("[RESULT] SUMMARY")
    logger.info("="*70)

    total = len(all_results)
    passed = sum(1 for r in all_results.values() if r['success'])
    failed = total - passed

    logger.info(f"Total: {total}, Passed: {passed}, Failed: {failed}")

    if failed > 0:
        logger.warning("\nFailed experiments:")
        for tag, result in all_results.items():
            if not result['success']:
                logger.warning(f"  - {tag} (log: {result['log_file']})")

    # Save results
    results_file = LOG_DIR / 'tuning_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to: {results_file}")

    logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)

    return 0 if failed == 0 else 1

if __name__ == '__main__':
    sys.exit(main())

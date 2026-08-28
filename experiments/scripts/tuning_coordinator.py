#!/usr/bin/env python3
"""
PoL-BFL 实验参数评估协调器
管理RQ1-RQ5的轻量化实验执行
支持并行运行和结果收集
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
import multiprocessing as mp

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
    # RQ1: Security Evaluation
    {'rq': 'rq1', 'dataset': 'CIFAR10', 'rounds': 3, 'gpu': 0, 'tag': 'rq1_cifar10_smoke', 'phase': 1},
    {'rq': 'rq1', 'dataset': 'MNIST', 'rounds': 10, 'gpu': 0, 'tag': 'rq1_mnist_tuning', 'phase': 2},
    {'rq': 'rq1', 'dataset': 'CIFAR10', 'rounds': 15, 'gpu': 0, 'tag': 'rq1_cifar10_tuning', 'phase': 2},
    {'rq': 'rq1', 'dataset': 'CIFAR100', 'rounds': 15, 'gpu': 0, 'tag': 'rq1_cifar100_tuning', 'phase': 2},

    # RQ2: Ablation Study
    {'rq': 'rq2', 'dataset': 'MNIST', 'rounds': 10, 'gpu': 1, 'tag': 'rq2_mnist_tuning', 'phase': 3},
    {'rq': 'rq2', 'dataset': 'CIFAR10', 'rounds': 10, 'gpu': 1, 'tag': 'rq2_cifar10_tuning', 'phase': 3},

    # RQ3: System Overhead
    {'rq': 'rq3', 'dataset': 'MNIST', 'rounds': 5, 'gpu': 1, 'tag': 'rq3_mnist_tuning', 'phase': 3},
    {'rq': 'rq3', 'dataset': 'CIFAR10', 'rounds': 5, 'gpu': 1, 'tag': 'rq3_cifar10_tuning', 'phase': 3},

    # RQ4: Incentive Mechanism
    {'rq': 'rq4', 'dataset': 'MNIST', 'rounds': 20, 'gpu': 1, 'tag': 'rq4_mnist_tuning', 'phase': 3},
]

def run_experiment(exp_config: Dict) -> Tuple[str, bool, str]:
    """
    运行单个实验

    Args:
        exp_config: 实验配置字典

    Returns:
        (tag, success, log_file)
    """
    rq = exp_config['rq']
    dataset = exp_config['dataset']
    rounds = exp_config['rounds']
    gpu = exp_config['gpu']
    tag = exp_config['tag']

    log_file = LOG_DIR / f"{tag}.log"
    result_subdir = RESULT_DIR / tag

    logger.info(f"Starting {tag} on GPU {gpu} ({rq}, {dataset}, {rounds} rounds)")

    # Build command
    script_name = f"run_{rq}_security.py" if rq == 'rq1' else \
                  f"run_{rq}_ablation.py" if rq == 'rq2' else \
                  f"run_{rq}_overhead.py" if rq == 'rq3' else \
                  f"run_{rq}_incentive.py"

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

    # Set GPU
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
                timeout=3600  # 1 hour timeout
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

def main():
    """主函数"""
    logger.info("="*70)
    logger.info("[START] PoL-BFL Tuning Experiments Coordinator")
    logger.info(f"   Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)

    # Group experiments by phase
    phases = {}
    for exp in EXPERIMENTS:
        phase = exp['phase']
        if phase not in phases:
            phases[phase] = []
        phases[phase].append(exp)

    results = {}

    # Execute phases sequentially
    for phase in sorted(phases.keys()):
        logger.info(f"\n[PLAN] Phase {phase}: {len(phases[phase])} experiments")

        # Execute experiments in phase (GPU0 and GPU1 can run in parallel)
        with mp.Pool(processes=2) as pool:
            phase_results = pool.map(run_experiment, phases[phase])

        for tag, success, log_file in phase_results:
            results[tag] = {
                'success': success,
                'log_file': log_file,
                'timestamp': datetime.now().isoformat()
            }

        # Wait between phases
        if phase < max(phases.keys()):
            logger.info(f"Phase {phase} completed. Waiting before next phase...")
            time.sleep(5)

    # Summary
    logger.info("\n" + "="*70)
    logger.info("[RESULT] SUMMARY")
    logger.info("="*70)

    total = len(results)
    passed = sum(1 for r in results.values() if r['success'])
    failed = total - passed

    logger.info(f"Total: {total}, Passed: {passed}, Failed: {failed}")

    if failed > 0:
        logger.warning("\nFailed experiments:")
        for tag, result in results.items():
            if not result['success']:
                logger.warning(f"  - {tag} (log: {result['log_file']})")

    # Save results
    results_file = LOG_DIR / 'tuning_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_file}")

    logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)

    return 0 if failed == 0 else 1

if __name__ == '__main__':
    sys.exit(main())


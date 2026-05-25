#!/usr/bin/env python3
"""
Auto-sequential experiment launcher
Automatically launches next experiment when current one completes
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
import signal

class ExperimentLauncher:
    def __init__(self):
        self.base_dir = Path('experiments')
        self.log_dir = self.base_dir / 'logs' / 'tuning_2025-11-19'
        self.result_dir = self.base_dir / 'results' / 'tuning_2025-11-19'
        
        # GPU 0 experiments (RQ1)
        self.gpu0_queue = [
            {
                'name': 'RQ1 CIFAR-10',
                'script': 'run_rq1_security.py',
                'args': ['--dataset', 'CIFAR10', '--num_rounds', '15'],
                'log': 'rq1_cifar10_tuning.log',
                'output': 'rq1_cifar10_tuning'
            },
            {
                'name': 'RQ1 CIFAR-100',
                'script': 'run_rq1_security.py',
                'args': ['--dataset', 'CIFAR100', '--num_rounds', '15'],
                'log': 'rq1_cifar100_tuning.log',
                'output': 'rq1_cifar100_tuning'
            }
        ]
        
        # GPU 1 experiments (RQ2/RQ3/RQ4)
        self.gpu1_queue = [
            {
                'name': 'RQ2 CIFAR-10',
                'script': 'run_rq2_ablation.py',
                'args': ['--dataset', 'CIFAR10', '--num_rounds', '10'],
                'log': 'rq2_cifar10_tuning.log',
                'output': 'rq2_cifar10_tuning'
            },
            {
                'name': 'RQ3 MNIST',
                'script': 'run_rq3_overhead.py',
                'args': ['--dataset', 'MNIST', '--num_rounds', '5'],
                'log': 'rq3_mnist_tuning.log',
                'output': 'rq3_mnist_tuning'
            },
            {
                'name': 'RQ3 CIFAR-10',
                'script': 'run_rq3_overhead.py',
                'args': ['--dataset', 'CIFAR10', '--num_rounds', '5'],
                'log': 'rq3_cifar10_tuning.log',
                'output': 'rq3_cifar10_tuning'
            },
            {
                'name': 'RQ4 MNIST',
                'script': 'run_rq4_incentive.py',
                'args': ['--dataset', 'MNIST', '--num_rounds', '20'],
                'log': 'rq4_mnist_tuning.log',
                'output': 'rq4_mnist_tuning'
            }
        ]
        
        self.gpu0_process = None
        self.gpu1_process = None
    
    def launch_experiment(self, gpu_id, exp_config):
        """Launch an experiment on specified GPU"""
        script_path = f'experiments/scripts/runners/{exp_config["script"]}'
        log_path = self.log_dir / exp_config['log']
        output_path = self.result_dir / exp_config['output']
        
        cmd = [
            'python', script_path,
            '--output_dir', str(output_path),
            *exp_config['args']
        ]
        
        env = os.environ.copy()
        env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        env['PYTHONPATH'] = '/home/wxb/PoL引入区块链联邦学习/PoL-BFL/Code:/home/wxb/PoL引入区块链联邦学习/PoL-BFL/Code/experiments/scripts/utils'
        env['POL_DATA_DIR'] = '/home/wxb/PoL引入区块链联邦学习/PoL-BFL/Code/data'
        
        print(f"\n{'='*80}")
        print(f"🚀 Launching: {exp_config['name']} on GPU {gpu_id}")
        print(f"   Command: {' '.join(cmd)}")
        print(f"   Log: {log_path}")
        print(f"{'='*80}\n")
        
        with open(log_path, 'w') as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                cwd='/home/wxb/PoL引入区块链联邦学习/PoL-BFL/Code'
            )
        
        return process
    
    def is_process_running(self, process):
        """Check if process is still running"""
        if process is None:
            return False
        return process.poll() is None
    
    def monitor_and_launch(self):
        """Monitor experiments and launch next ones"""
        print("\n" + "="*80)
        print("🔍 AUTO-SEQUENTIAL EXPERIMENT LAUNCHER")
        print("="*80)
        
        gpu0_idx = 0
        gpu1_idx = 0
        
        # Launch first experiments
        if gpu0_idx < len(self.gpu0_queue):
            self.gpu0_process = self.launch_experiment(0, self.gpu0_queue[gpu0_idx])
            gpu0_idx += 1
        
        if gpu1_idx < len(self.gpu1_queue):
            self.gpu1_process = self.launch_experiment(1, self.gpu1_queue[gpu1_idx])
            gpu1_idx += 1
        
        # Monitor and launch
        try:
            while gpu0_idx < len(self.gpu0_queue) or gpu1_idx < len(self.gpu1_queue) or \
                  self.is_process_running(self.gpu0_process) or self.is_process_running(self.gpu1_process):
                
                # Check GPU 0
                if not self.is_process_running(self.gpu0_process) and gpu0_idx < len(self.gpu0_queue):
                    print(f"\n✅ GPU 0 experiment completed")
                    self.gpu0_process = self.launch_experiment(0, self.gpu0_queue[gpu0_idx])
                    gpu0_idx += 1
                
                # Check GPU 1
                if not self.is_process_running(self.gpu1_process) and gpu1_idx < len(self.gpu1_queue):
                    print(f"\n✅ GPU 1 experiment completed")
                    self.gpu1_process = self.launch_experiment(1, self.gpu1_queue[gpu1_idx])
                    gpu1_idx += 1
                
                time.sleep(30)  # Check every 30 seconds
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            if self.gpu0_process:
                self.gpu0_process.terminate()
            if self.gpu1_process:
                self.gpu1_process.terminate()
        
        print("\n" + "="*80)
        print("✅ All experiments completed!")
        print("="*80)

if __name__ == '__main__':
    launcher = ExperimentLauncher()
    launcher.monitor_and_launch()


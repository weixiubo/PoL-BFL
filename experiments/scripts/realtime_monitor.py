#!/usr/bin/env python3
"""
Real-time experiment monitoring dashboard
Monitors all running experiments and displays progress
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
import json

def get_log_tail(log_file, lines=5):
    """Get last N lines from log file"""
    try:
        result = subprocess.run(['tail', '-n', str(lines), log_file], 
                              capture_output=True, text=True, timeout=2)
        return result.stdout.strip().split('\n')
    except:
        return []

def extract_accuracy(log_lines):
    """Extract latest accuracy from log lines"""
    for line in reversed(log_lines):
        if 'Test Accuracy:' in line:
            try:
                acc = line.split('Test Accuracy:')[1].strip()
                return float(acc)
            except:
                pass
    return None

def extract_round(log_lines):
    """Extract current round from log lines"""
    for line in reversed(log_lines):
        if 'Round' in line and '/' in line:
            try:
                parts = line.split('Round')[1].strip().split('/')[0].strip()
                return int(parts)
            except:
                pass
    return None

def get_file_size_mb(file_path):
    """Get file size in MB"""
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except:
        return 0

def monitor_experiments():
    """Monitor all running experiments"""
    log_dir = Path('experiments/logs/tuning_2025-11-19')
    
    if not log_dir.exists():
        print("❌ Log directory not found")
        return
    
    experiments = {
        'Phase 1': 'rq1_cifar10_quick.log',
        'RQ1 MNIST': 'rq1_mnist_tuning.log',
        'RQ2 MNIST': 'rq2_mnist_tuning.log',
    }
    
    print("\n" + "="*80)
    print("🔍 PoL-BFL EXPERIMENT MONITORING DASHBOARD")
    print("="*80)
    
    while True:
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("\n" + "="*80)
        print(f"📊 REAL-TIME MONITORING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        all_completed = True
        
        for exp_name, log_file in experiments.items():
            log_path = log_dir / log_file
            
            if not log_path.exists():
                print(f"\n⏳ {exp_name:20} | Status: WAITING")
                all_completed = False
                continue
            
            # Get log tail
            tail_lines = get_log_tail(str(log_path), 10)
            
            # Extract info
            accuracy = extract_accuracy(tail_lines)
            round_num = extract_round(tail_lines)
            file_size = get_file_size_mb(str(log_path))
            
            # Check if completed
            is_completed = any('completed' in line.lower() or 'finished' in line.lower() 
                             for line in tail_lines)
            
            if is_completed:
                status = "✅ COMPLETED"
            elif file_size > 0:
                status = "🔄 RUNNING"
                all_completed = False
            else:
                status = "⏳ WAITING"
                all_completed = False
            
            # Print status
            print(f"\n{exp_name:20} | {status:20}", end="")
            
            if round_num:
                print(f" | Round: {round_num:3}", end="")
            
            if accuracy is not None:
                print(f" | Accuracy: {accuracy:.4f}", end="")
            
            print(f" | Log Size: {file_size:.1f}MB")
            
            # Print last few lines
            if tail_lines and tail_lines[0]:
                for line in tail_lines[-3:]:
                    if line.strip():
                        print(f"  └─ {line[:70]}")
        
        print("\n" + "="*80)
        
        if all_completed:
            print("✅ All experiments completed!")
            break
        
        print("⏳ Monitoring... (Press Ctrl+C to exit)")
        print("="*80)
        
        try:
            time.sleep(10)  # Update every 10 seconds
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")
            break

if __name__ == '__main__':
    monitor_experiments()


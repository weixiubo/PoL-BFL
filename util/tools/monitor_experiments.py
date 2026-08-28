#!/usr/bin/env python3
"""
实验监控工具 - 监控正在运行的实验状态
"""

import os
import json
import time
import psutil
from pathlib import Path

def check_running_experiments():
    """检查正在运行的实验进程"""
    running_experiments = []

    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
        try:
            if 'python' in proc.info['name'] and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'run_rq' in cmdline or 'ablation' in cmdline:
                    running_experiments.append({
                        'pid': proc.info['pid'],
                        'cmdline': cmdline,
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_mb': proc.info['memory_info'].rss / 1024 / 1024
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return running_experiments

def check_latest_results():
    """检查最新的实验结果"""
    results_dir = Path('experiments/results/rq2_ablation')

    if not results_dir.exists():
        return None

    # 找到最新的结果目录
    latest_dir = None
    latest_time = 0

    for subdir in results_dir.iterdir():
        if subdir.is_dir():
            try:
                # 从目录名提取时间戳
                timestamp = subdir.name
                if len(timestamp) == 15:  # YYYYMMDD_HHMMSS
                    dir_time = os.path.getmtime(subdir)
                    if dir_time > latest_time:
                        latest_time = dir_time
                        latest_dir = subdir
            except:
                continue

    if not latest_dir:
        return None

    # 读取结果文件
    results_file = latest_dir / 'ablation_results.json'
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            return {
                'dir': str(latest_dir),
                'results': results,
                'last_modified': time.ctime(latest_time)
            }
        except:
            return None

    return None

def format_results_summary(results_data):
    """格式化结果摘要"""
    if not results_data:
        return "No results found"

    summary = [f"[PATH] Latest results: {results_data['dir']}"]
    summary.append(f"[TIME] Last modified: {results_data['last_modified']}")
    summary.append("")

    for variant_data in results_data['results']:
        variant = variant_data['variant']

        if 'final_accuracy_mean' in variant_data:
            # 汇总结果
            ma = variant_data['final_accuracy_mean']
            tpr = variant_data['tpr_mean']
            fpr = variant_data['fpr_mean']
            pr = variant_data['participation_rate_mean']
            vpr = variant_data['verify_pass_rate_mean']

            summary.append(f"[TEST] {variant}:")
            summary.append(f"   MA: {ma:.3f}, TPR: {tpr:.3f}, FPR: {fpr:.3f}")
            summary.append(f"   Participation: {pr:.3f}, Verify Pass: {vpr:.3f}")
        elif 'results' in variant_data:
            # 详细结果
            results = variant_data['results']
            summary.append(f"[TEST] {variant} ({len(results)} repetitions):")

            for i, rep in enumerate(results):
                if 'detection_metrics' in rep:
                    dm = rep['detection_metrics']
                    tpr = dm.get('TPR', 0)
                    fpr = dm.get('FPR', 0)
                    pr = rep.get('participation_rate', 0)
                    vpr = rep.get('verification_pass_rate', 0)

                    summary.append(f"   Rep {i}: TPR={tpr:.3f}, FPR={fpr:.3f}, PR={pr:.3f}, VPR={vpr:.3f}")

        summary.append("")

    return "\n".join(summary)

def main():
    print("[CHECK] PoL-BFL Experiment Monitor")
    print("=" * 50)

    # 检查运行中的实验
    running = check_running_experiments()

    if running:
        print(f"[START] Found {len(running)} running experiments:")
        for exp in running:
            print(f"   PID {exp['pid']}: {exp['cmdline'][:80]}...")
            print(f"   CPU: {exp['cpu_percent']:.1f}%, Memory: {exp['memory_mb']:.1f}MB")
        print()
    else:
        print("[IDLE] No experiments currently running")
        print()

    # 检查最新结果
    print("[RESULT] Latest Results:")
    print("-" * 30)
    latest_results = check_latest_results()
    print(format_results_summary(latest_results))

    # GPU状态
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("\n[GPU]  GPU Status:")
            print("-" * 20)
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) >= 4:
                    gpu_id, util, mem_used, mem_total = parts
                    mem_percent = float(mem_used) / float(mem_total) * 100
                    print(f"   GPU {gpu_id}: {util}% util, {mem_percent:.1f}% memory ({mem_used}/{mem_total} MB)")
    except:
        print("\n[GPU]  GPU Status: Unable to query")

if __name__ == "__main__":
    main()

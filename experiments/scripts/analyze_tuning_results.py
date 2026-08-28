#!/usr/bin/env python3
"""
实验结果分析脚本
自动分析RQ1-RQ5的轻量化实验结果
生成对比表格和可视化
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULT_DIR = PROJECT_ROOT / 'experiments' / 'results' / 'parameter_evaluation'
LOG_DIR = PROJECT_ROOT / 'experiments' / 'logs' / 'parameter_evaluation'
ANALYSIS_DIR = LOG_DIR / 'analysis'

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

def load_results(result_dir):
    """Load results from a result directory"""
    results = {}

    # Try to load results.json
    results_file = result_dir / 'results.json'
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results['results'] = json.load(f)
        except Exception as e:
            print(f"Error loading {results_file}: {e}")

    # Try to load metrics.json
    metrics_file = result_dir / 'metrics.json'
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                results['metrics'] = json.load(f)
        except Exception as e:
            print(f"Error loading {metrics_file}: {e}")

    return results

def analyze_rq1_results():
    """Analyze RQ1 security evaluation results"""
    print("\n" + "="*80)
    print("[RESULT] RQ1 Security Evaluation Analysis")
    print("="*80)

    rq1_dirs = [d for d in RESULT_DIR.iterdir() if d.is_dir() and d.name.startswith('rq1_')]

    summary = defaultdict(list)

    for result_dir in sorted(rq1_dirs):
        print(f"\nAnalyzing {result_dir.name}...")
        results = load_results(result_dir)

        if 'metrics' in results:
            metrics = results['metrics']
            summary['experiment'].append(result_dir.name)
            summary['tpr'].append(metrics.get('TPR', 'N/A'))
            summary['fpr'].append(metrics.get('FPR', 'N/A'))
            summary['accuracy'].append(metrics.get('final_accuracy', 'N/A'))

    if summary:
        df = pd.DataFrame(summary)
        print("\n" + df.to_string(index=False))

        # Save to CSV
        csv_file = ANALYSIS_DIR / 'rq1_summary.csv'
        df.to_csv(csv_file, index=False)
        print(f"\nSaved to: {csv_file}")

def analyze_rq2_results():
    """Analyze RQ2 ablation study results"""
    print("\n" + "="*80)
    print("[RESULT] RQ2 Ablation Study Analysis")
    print("="*80)

    rq2_dirs = [d for d in RESULT_DIR.iterdir() if d.is_dir() and d.name.startswith('rq2_')]

    summary = defaultdict(list)

    for result_dir in sorted(rq2_dirs):
        print(f"\nAnalyzing {result_dir.name}...")
        results = load_results(result_dir)

        if 'metrics' in results:
            metrics = results['metrics']
            summary['experiment'].append(result_dir.name)
            summary['vanilla_fl'].append(metrics.get('vanilla_fl_accuracy', 'N/A'))
            summary['pol_only'].append(metrics.get('pol_only_accuracy', 'N/A'))
            summary['pol_zkp'].append(metrics.get('pol_zkp_accuracy', 'N/A'))
            summary['pol_incentive'].append(metrics.get('pol_incentive_accuracy', 'N/A'))

    if summary:
        df = pd.DataFrame(summary)
        print("\n" + df.to_string(index=False))

        # Save to CSV
        csv_file = ANALYSIS_DIR / 'rq2_summary.csv'
        df.to_csv(csv_file, index=False)
        print(f"\nSaved to: {csv_file}")

def analyze_rq3_results():
    """Analyze RQ3 overhead analysis results"""
    print("\n" + "="*80)
    print("[RESULT] RQ3 System Overhead Analysis")
    print("="*80)

    rq3_dirs = [d for d in RESULT_DIR.iterdir() if d.is_dir() and d.name.startswith('rq3_')]

    summary = defaultdict(list)

    for result_dir in sorted(rq3_dirs):
        print(f"\nAnalyzing {result_dir.name}...")
        results = load_results(result_dir)

        if 'metrics' in results:
            metrics = results['metrics']
            summary['experiment'].append(result_dir.name)
            summary['time_overhead'].append(metrics.get('time_overhead', 'N/A'))
            summary['comm_overhead'].append(metrics.get('comm_overhead', 'N/A'))
            summary['storage_overhead'].append(metrics.get('storage_overhead', 'N/A'))

    if summary:
        df = pd.DataFrame(summary)
        print("\n" + df.to_string(index=False))

        # Save to CSV
        csv_file = ANALYSIS_DIR / 'rq3_summary.csv'
        df.to_csv(csv_file, index=False)
        print(f"\nSaved to: {csv_file}")

def analyze_rq4_results():
    """Analyze RQ4 incentive mechanism results"""
    print("\n" + "="*80)
    print("[RESULT] RQ4 Incentive Mechanism Analysis")
    print("="*80)

    rq4_dirs = [d for d in RESULT_DIR.iterdir() if d.is_dir() and d.name.startswith('rq4_')]

    summary = defaultdict(list)

    for result_dir in sorted(rq4_dirs):
        print(f"\nAnalyzing {result_dir.name}...")
        results = load_results(result_dir)

        if 'metrics' in results:
            metrics = results['metrics']
            summary['experiment'].append(result_dir.name)
            summary['participation_rate'].append(metrics.get('participation_rate', 'N/A'))
            summary['incentive_effectiveness'].append(metrics.get('incentive_effectiveness', 'N/A'))

    if summary:
        df = pd.DataFrame(summary)
        print("\n" + df.to_string(index=False))

        # Save to CSV
        csv_file = ANALYSIS_DIR / 'rq4_summary.csv'
        df.to_csv(csv_file, index=False)
        print(f"\nSaved to: {csv_file}")

def generate_summary_report():
    """Generate overall summary report"""
    print("\n" + "="*80)
    print("[PLAN] Overall Summary Report")
    print("="*80)

    report = {
        'timestamp': datetime.now().isoformat(),
        'total_experiments': 0,
        'completed_experiments': 0,
        'failed_experiments': 0,
        'rq_summary': {}
    }

    # Count experiments
    if RESULT_DIR.exists():
        for result_dir in RESULT_DIR.iterdir():
            if result_dir.is_dir():
                report['total_experiments'] += 1
                result_files = list(result_dir.glob('*.json'))
                if result_files:
                    report['completed_experiments'] += 1
                else:
                    report['failed_experiments'] += 1

    # Save report
    report_file = ANALYSIS_DIR / 'summary_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nTotal Experiments: {report['total_experiments']}")
    print(f"Completed: {report['completed_experiments']}")
    print(f"Failed: {report['failed_experiments']}")
    print(f"\nReport saved to: {report_file}")

def main():
    """Main analysis function"""
    print("\n" + "="*80)
    print("[CHECK] PoL-BFL Tuning Results Analysis")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    if not RESULT_DIR.exists():
        print(f"Result directory not found: {RESULT_DIR}")
        return 1

    # Analyze each RQ
    analyze_rq1_results()
    analyze_rq2_results()
    analyze_rq3_results()
    analyze_rq4_results()

    # Generate summary
    generate_summary_report()

    print("\n" + "="*80)
    print("[PASS] Analysis Complete")
    print(f"Results saved to: {ANALYSIS_DIR}")
    print("="*80)

    return 0

if __name__ == '__main__':
    sys.exit(main())


"""
Visualization Scripts for Experiment Results

Generates publication-quality figures for:
- RQ1: Security evaluation (accuracy curves, detection rate heatmaps)
- RQ2: Ablation study (radar charts, bar charts)
- RQ3: Overhead analysis (bar charts, breakdown charts)
- RQ4: Incentive effectiveness (utility evolution curves)
- RQ5: Composability (comparison bar charts)

Usage:
    python visualize_results.py --experiment rq1 --input results/rq1_security/rq1_results.json
    python visualize_results.py --experiment rq2 --input results/rq2_ablation/rq2_results.json
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Color palette
COLORS = {
    'PoL-BFL': '#2E86AB',
    'Vanilla_FL': '#A23B72',
    'Krum': '#F18F01',
    'Trimmed_Mean': '#C73E1D',
    'Median': '#6A994E',
    'Bulyan': '#BC4B51',
}


def visualize_rq1_accuracy_curves(results, output_dir):
    """
    RQ1: Plot accuracy curves for different attacks and baselines
    
    Creates one figure per attack type showing accuracy evolution
    """
    logger.info("Generating RQ1 accuracy curves...")
    
    # Group by attack type
    from collections import defaultdict
    by_attack = defaultdict(list)
    for result in results:
        by_attack[result['attack_type']].append(result)
    
    # Create figure for each attack
    for attack_type, attack_results in by_attack.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        
        for result in attack_results:
            baseline = result['baseline_method']
            accuracies = result['test_accuracies']
            rounds = list(range(1, len(accuracies) + 1))
            
            color = COLORS.get(baseline, None)
            linestyle = '-' if baseline == 'PoL_FL' else '--'
            linewidth = 2 if baseline == 'PoL_FL' else 1.5
            
            ax.plot(rounds, accuracies, label=baseline, color=color,
                   linestyle=linestyle, linewidth=linewidth, alpha=0.8)
        
        ax.set_xlabel('Round')
        ax.set_ylabel('Test Accuracy')
        ax.set_title(f'RQ1: {attack_type}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Save
        output_path = Path(output_dir) / f'rq1_accuracy_{attack_type}.pdf'
        plt.savefig(output_path, format='pdf')
        plt.close()
        
        logger.info(f"Saved {output_path}")


def visualize_rq1_detection_heatmap(results, output_dir):
    """
    RQ1: Create heatmap of detection rates across attacks and methods
    """
    logger.info("Generating RQ1 detection rate heatmap...")
    
    # Extract data
    from collections import defaultdict
    data = defaultdict(dict)
    
    for result in results:
        attack = result['attack_type']
        baseline = result['baseline_method']
        tpr = result.get('tpr_conditional', 0.0)
        data[attack][baseline] = tpr
    
    # Convert to matrix
    attacks = sorted(data.keys())
    baselines = sorted(set(b for attack_data in data.values() for b in attack_data.keys()))
    
    matrix = np.zeros((len(attacks), len(baselines)))
    for i, attack in enumerate(attacks):
        for j, baseline in enumerate(baselines):
            matrix[i, j] = data[attack].get(baseline, 0.0)
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=baselines, yticklabels=attacks,
                vmin=0, vmax=1, cbar_kws={'label': 'TPR (Conditional)'},
                ax=ax)
    
    ax.set_title('RQ1: Detection Rate Heatmap')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Save
    output_path = Path(output_dir) / 'rq1_detection_heatmap.pdf'
    plt.savefig(output_path, format='pdf')
    plt.close()
    
    logger.info(f"Saved {output_path}")


def visualize_rq2_ablation_radar(results, output_dir):
    """
    RQ2: Create radar chart for ablation study
    
    Shows contribution of each component across multiple metrics
    """
    logger.info("Generating RQ2 ablation radar chart...")
    
    # Extract metrics for each variant
    variants = []
    metrics_data = {}
    
    for result in results:
        variant = result['variant']
        variants.append(variant)
        
        metrics_data[variant] = {
            'Accuracy': result['final_accuracy'],
            'Detection Rate': result.get('detection_rate', 0.0),
            'Participation': result.get('participation_rate', 1.0),
        }
    
    # Normalize metrics to [0, 1]
    metric_names = ['Accuracy', 'Detection Rate', 'Participation']
    
    # Create radar chart
    from math import pi
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    
    angles = [n / len(metric_names) * 2 * pi for n in range(len(metric_names))]
    angles += angles[:1]
    
    for variant in variants:
        values = [metrics_data[variant][m] for m in metric_names]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=variant)
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1)
    ax.set_title('RQ2: Ablation Study - Component Contributions')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    # Save
    output_path = Path(output_dir) / 'rq2_ablation_radar.pdf'
    plt.savefig(output_path, format='pdf')
    plt.close()
    
    logger.info(f"Saved {output_path}")


def visualize_rq2_ablation_bars(results, output_dir):
    """
    RQ2: Create bar chart showing accuracy for each variant
    """
    logger.info("Generating RQ2 ablation bar chart...")
    
    variants = []
    accuracies = []
    stds = []
    
    for result in results:
        variants.append(result['variant'])
        accuracies.append(result['final_accuracy'])
        stds.append(result.get('accuracy_std', 0.0))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(variants))
    bars = ax.bar(x, accuracies, yerr=stds, capsize=5, alpha=0.8,
                  color=['#2E86AB', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51'][:len(variants)])
    
    ax.set_xlabel('Variant')
    ax.set_ylabel('Final Accuracy')
    ax.set_title('RQ2: Ablation Study - Component Contributions')
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, acc, std) in enumerate(zip(bars, accuracies, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{acc:.2f}±{std:.2f}',
               ha='center', va='bottom', fontsize=8)
    
    # Save
    output_path = Path(output_dir) / 'rq2_ablation_bars.pdf'
    plt.savefig(output_path, format='pdf')
    plt.close()
    
    logger.info(f"Saved {output_path}")


def visualize_rq3_overhead_bars(results, output_dir):
    """
    RQ3: Create bar chart comparing overhead across methods
    """
    logger.info("Generating RQ3 overhead bar chart...")
    
    if isinstance(results, dict):
        results = [
            {
                'method': method,
                'avg_training_time': metrics.get('training_time', {}).get('mean', 0.0),
                'avg_communication_cost': metrics.get('communication_cost', {}).get('mean', 0.0),
                'avg_storage_cost': metrics.get('storage_cost', {}).get('mean', 0.0),
            }
            for method, metrics in sorted(results.items())
        ]

    methods = []
    training_times = []
    comm_costs = []
    storage_costs = []
    
    for result in results:
        methods.append(result['method'])
        training_times.append(result.get('avg_training_time', 0.0))
        comm_costs.append(result.get('avg_communication_cost', 0.0))
        storage_costs.append(result.get('avg_storage_cost', 0.0))
    
    # Normalize to Vanilla FL
    if 'Vanilla_FL' in methods:
        baseline_time = training_times[methods.index('Vanilla_FL')]
        if baseline_time > 0:
            training_times = [t / baseline_time for t in training_times]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(methods))
    width = 0.25
    
    ax.bar(x - width, training_times, width, label='Training Time', alpha=0.8)
    ax.bar(x, comm_costs, width, label='Communication', alpha=0.8)
    ax.bar(x + width, storage_costs, width, label='Storage', alpha=0.8)
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Relative Overhead (vs Vanilla FL)')
    ax.set_title('RQ3: System Overhead Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal line at 1.0 (baseline)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Baseline')
    
    # Save
    output_path = Path(output_dir) / 'rq3_overhead_bars.pdf'
    plt.savefig(output_path, format='pdf')
    plt.close()
    
    logger.info(f"Saved {output_path}")


def visualize_rq4_utility_evolution(results, output_dir):
    """
    RQ4: Plot utility evolution for different node types
    """
    logger.info("Generating RQ4 utility evolution curves...")
    
    if isinstance(results, dict):
        scenarios = sorted(results)
        x = np.arange(len(scenarios))
        width = 0.25
        fig, ax = plt.subplots(figsize=(8, 5))
        honest = [
            results[name].get('honest_utility', {}).get('mean', 0.0)
            for name in scenarios
        ]
        rational = [
            results[name].get('rational_utility', {}).get('mean', 0.0)
            for name in scenarios
        ]
        malicious = [
            results[name].get('malicious_utility', {}).get('mean', 0.0)
            for name in scenarios
        ]
        ax.bar(x - width, honest, width, label='Honest')
        ax.bar(x, rational, width, label='Rational')
        ax.bar(x + width, malicious, width, label='Malicious')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=30, ha='right')
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Cumulative Utility')
        ax.set_title('RQ4: Utility by Node Type')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        output_path = Path(output_dir) / 'rq4_utility_evolution.pdf'
        plt.savefig(output_path, format='pdf')
        plt.close()
        logger.info(f"Saved {output_path}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    
    for result in results:
        scenario = result['scenario']
        
        if 'utility_evolution' in result:
            rounds = list(range(1, len(result['utility_evolution']['honest']) + 1))
            
            ax.plot(rounds, result['utility_evolution']['honest'],
                   label=f'{scenario} - Honest', linestyle='-', linewidth=2)
            ax.plot(rounds, result['utility_evolution']['rational'],
                   label=f'{scenario} - Rational', linestyle='--', linewidth=2)
            ax.plot(rounds, result['utility_evolution']['malicious'],
                   label=f'{scenario} - Malicious', linestyle=':', linewidth=2)
    
    ax.set_xlabel('Round')
    ax.set_ylabel('Cumulative Utility')
    ax.set_title('RQ4: Utility Evolution by Node Type')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Save
    output_path = Path(output_dir) / 'rq4_utility_evolution.pdf'
    plt.savefig(output_path, format='pdf')
    plt.close()
    
    logger.info(f"Saved {output_path}")


def visualize_rq5_composability_bars(results, output_dir):
    """
    RQ5: Create comparison bars for composability
    
    Shows robust aggregation vs PoL + robust aggregation
    """
    logger.info("Generating RQ5 composability bar chart...")
    
    if isinstance(results, dict):
        results = [
            {
                'attack_type': attack,
                'baseline_method': method,
                'final_accuracy': metrics.get('final_accuracy', {}).get('mean', 0.0),
            }
            for attack, methods in sorted(results.items())
            for method, metrics in sorted(methods.items())
        ]

    # Group by attack and base method
    from collections import defaultdict
    by_attack = defaultdict(lambda: defaultdict(list))
    
    for result in results:
        attack = result['attack_type']
        baseline = result['baseline_method']
        base_method = baseline.replace('PoL_', '')
        
        by_attack[attack][base_method].append(result)
    
    # Create figure for each attack
    for attack_type, methods in by_attack.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        
        base_methods = sorted(methods.keys())
        x = np.arange(len(base_methods))
        width = 0.35
        
        without_pol = []
        with_pol = []
        
        for method in base_methods:
            results_list = methods[method]
            
            without = [r for r in results_list if not r['baseline_method'].startswith('PoL_')]
            with_p = [r for r in results_list if r['baseline_method'].startswith('PoL_')]
            
            without_pol.append(without[0]['final_accuracy'] if without else 0.0)
            with_pol.append(with_p[0]['final_accuracy'] if with_p else 0.0)
        
        ax.bar(x - width/2, without_pol, width, label='Without PoL', alpha=0.8)
        ax.bar(x + width/2, with_pol, width, label='With PoL', alpha=0.8)
        
        ax.set_xlabel('Robust Aggregation Method')
        ax.set_ylabel('Final Accuracy')
        ax.set_title(f'RQ5: Composability - {attack_type}')
        ax.set_xticks(x)
        ax.set_xticklabels(base_methods, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Save
        output_path = Path(output_dir) / f'rq5_composability_{attack_type}.pdf'
        plt.savefig(output_path, format='pdf')
        plt.close()
        
        logger.info(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize experiment results')
    parser.add_argument('--experiment', type=str, required=True,
                       choices=['rq1', 'rq2', 'rq3', 'rq4', 'rq5'],
                       help='Experiment type')
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSON file with results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for figures (default: same as input)')
    
    args = parser.parse_args()
    
    # Load results
    with open(args.input, 'r') as f:
        results = json.load(f)
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.input).parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    if args.experiment == 'rq1':
        visualize_rq1_accuracy_curves(results, output_dir)
        visualize_rq1_detection_heatmap(results, output_dir)
    elif args.experiment == 'rq2':
        visualize_rq2_ablation_radar(results, output_dir)
        visualize_rq2_ablation_bars(results, output_dir)
    elif args.experiment == 'rq3':
        visualize_rq3_overhead_bars(results, output_dir)
    elif args.experiment == 'rq4':
        visualize_rq4_utility_evolution(results, output_dir)
    elif args.experiment == 'rq5':
        visualize_rq5_composability_bars(results, output_dir)
    
    logger.info(f"\nAll visualizations saved to {output_dir}")


if __name__ == '__main__':
    main()

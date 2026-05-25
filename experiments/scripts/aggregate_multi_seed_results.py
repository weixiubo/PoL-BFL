"""
Aggregate Multi-Seed Experiment Results with Statistical Significance

This script:
1. Loads results from multiple seed runs
2. Computes mean, std, and confidence intervals
3. Performs statistical significance testing (t-tests)
4. Generates formatted tables for papers
5. Saves aggregated results

Usage:
    python aggregate_multi_seed_results.py --experiment rq1 --seeds 42,123,456
    python aggregate_multi_seed_results.py --experiment rq2 --input_dir results/rq2_ablation
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import logging
import sys

# Add utils to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir / 'utils'))

from statistical_analysis import (
    compute_statistical_significance,
    format_with_significance,
    aggregate_multi_seed_results,
    compare_methods_with_significance,
    generate_latex_table_row
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def aggregate_rq1_results(result_files, output_dir):
    """
    Aggregate RQ1 security evaluation results
    
    Expected structure:
    - Multiple JSON files from different seeds
    - Each file contains results for different attacks and baselines
    """
    logger.info("Aggregating RQ1 results...")
    
    # Load all results
    all_results = []
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
                all_results.append(results)
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            continue
    
    if not all_results:
        logger.error("No valid result files found")
        return
    
    # Group by attack type and baseline
    grouped = defaultdict(lambda: defaultdict(list))
    
    for seed_results in all_results:
        for result in seed_results:
            attack = result['attack_type']
            baseline = result['baseline_method']
            grouped[attack][baseline].append(result)
    
    # Aggregate metrics
    aggregated = {}
    
    for attack, baselines in grouped.items():
        aggregated[attack] = {}
        
        for baseline, runs in baselines.items():
            # Extract metrics
            final_accs = [r['final_accuracy'] for r in runs]
            tpr_conds = [r.get('tpr_conditional', 0.0) for r in runs]
            tpr_e2es = [r.get('tpr_e2e', 0.0) for r in runs]
            fprs = [r.get('fpr', 0.0) for r in runs]
            
            aggregated[attack][baseline] = {
                'final_accuracy': {
                    'mean': float(np.mean(final_accs)),
                    'std': float(np.std(final_accs, ddof=1)) if len(final_accs) > 1 else 0.0,
                    'values': final_accs
                },
                'tpr_conditional': {
                    'mean': float(np.mean(tpr_conds)),
                    'std': float(np.std(tpr_conds, ddof=1)) if len(tpr_conds) > 1 else 0.0,
                    'values': tpr_conds
                },
                'tpr_e2e': {
                    'mean': float(np.mean(tpr_e2es)),
                    'std': float(np.std(tpr_e2es, ddof=1)) if len(tpr_e2es) > 1 else 0.0,
                    'values': tpr_e2es
                },
                'fpr': {
                    'mean': float(np.mean(fprs)),
                    'std': float(np.std(fprs, ddof=1)) if len(fprs) > 1 else 0.0,
                    'values': fprs
                },
                'num_runs': len(runs)
            }
    
    # Compute statistical significance (compare PoL-BFL vs best baseline)
    for attack, baselines in aggregated.items():
        if 'PoL_FL' in baselines:
            pol_acc = baselines['PoL_FL']['final_accuracy']['values']
            
            # Find best baseline (excluding PoL_FL)
            best_baseline = None
            best_acc = 0.0
            for baseline, metrics in baselines.items():
                if baseline != 'PoL_FL':
                    acc = metrics['final_accuracy']['mean']
                    if acc > best_acc:
                        best_acc = acc
                        best_baseline = baseline
            
            if best_baseline:
                baseline_acc = baselines[best_baseline]['final_accuracy']['values']
                t_stat, p_val = compute_statistical_significance(pol_acc, baseline_acc)
                
                baselines['PoL_FL']['significance_vs_best'] = {
                    'baseline': best_baseline,
                    't_statistic': float(t_stat),
                    'p_value': float(p_val),
                    'significant': p_val < 0.05
                }
    
    # Save aggregated results
    output_path = Path(output_dir) / 'rq1_aggregated.json'
    with open(output_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    logger.info(f"Saved aggregated results to {output_path}")
    
    # Generate LaTeX table
    generate_rq1_latex_table(aggregated, output_dir)
    
    # Print summary
    print_rq1_summary(aggregated)
    
    return aggregated


def generate_rq1_latex_table(aggregated, output_dir):
    """Generate LaTeX table for RQ1 results"""
    logger.info("Generating RQ1 LaTeX table...")
    
    output_path = Path(output_dir) / 'rq1_table.tex'
    
    with open(output_path, 'w') as f:
        f.write("% RQ1: Security Evaluation Results\n")
        f.write("% Generated by aggregate_multi_seed_results.py\n\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{RQ1: Security Evaluation Results}\n")
        f.write("\\label{tab:rq1_security}\n")
        f.write("\\begin{tabular}{l|cccc}\n")
        f.write("\\hline\n")
        f.write("Attack & Method & MA & TPR$_{cond}$ & FPR \\\\\n")
        f.write("\\hline\n")
        
        for attack, baselines in sorted(aggregated.items()):
            f.write(f"\\multirow{{{len(baselines)}}}{{*}}{{{attack}}} \n")
            
            for i, (baseline, metrics) in enumerate(sorted(baselines.items())):
                ma = metrics['final_accuracy']
                tpr = metrics['tpr_conditional']
                fpr = metrics['fpr']
                
                # Get p-value if available
                p_val = metrics.get('significance_vs_best', {}).get('p_value')
                
                ma_str = format_with_significance(ma['mean'], ma['std'], p_val)
                tpr_str = format_with_significance(tpr['mean'], tpr['std'])
                fpr_str = format_with_significance(fpr['mean'], fpr['std'])
                
                if i == 0:
                    f.write(f" & {baseline} & {ma_str} & {tpr_str} & {fpr_str} \\\\\n")
                else:
                    f.write(f" & {baseline} & {ma_str} & {tpr_str} & {fpr_str} \\\\\n")
            
            f.write("\\hline\n")
        
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    logger.info(f"Saved LaTeX table to {output_path}")


def print_rq1_summary(aggregated):
    """Print RQ1 summary to console"""
    logger.info("\n" + "="*70)
    logger.info("RQ1 Aggregated Results Summary")
    logger.info("="*70)
    
    for attack, baselines in sorted(aggregated.items()):
        logger.info(f"\n--- Attack: {attack} ---")
        
        for baseline, metrics in sorted(baselines.items()):
            ma = metrics['final_accuracy']
            tpr = metrics['tpr_conditional']
            
            logger.info(f"  {baseline}:")
            logger.info(f"    MA:  {ma['mean']:.2f}±{ma['std']:.2f}")
            logger.info(f"    TPR: {tpr['mean']:.2f}±{tpr['std']:.2f}")
            
            if 'significance_vs_best' in metrics:
                sig = metrics['significance_vs_best']
                logger.info(f"    vs {sig['baseline']}: p={sig['p_value']:.4f} {'***' if sig['p_value'] < 0.001 else '**' if sig['p_value'] < 0.01 else '*' if sig['p_value'] < 0.05 else ''}")


def aggregate_rq2_results(result_files, output_dir):
    """Aggregate RQ2 ablation study results"""
    logger.info("Aggregating RQ2 results...")
    
    # Similar structure to RQ1
    # Group by variant
    all_results = []
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
                all_results.append(results)
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            continue
    
    if not all_results:
        logger.error("No valid result files found")
        return
    
    # Group by variant
    grouped = defaultdict(list)
    for seed_results in all_results:
        for result in seed_results:
            variant = result['variant']
            grouped[variant].append(result)
    
    # Aggregate
    aggregated = {}
    for variant, runs in grouped.items():
        final_accs = [r['final_accuracy'] for r in runs]
        
        aggregated[variant] = {
            'final_accuracy': {
                'mean': float(np.mean(final_accs)),
                'std': float(np.std(final_accs, ddof=1)) if len(final_accs) > 1 else 0.0,
                'values': final_accs
            },
            'num_runs': len(runs)
        }
    
    # Save
    output_path = Path(output_dir) / 'rq2_aggregated.json'
    with open(output_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    logger.info(f"Saved aggregated results to {output_path}")
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description='Aggregate multi-seed experiment results')
    parser.add_argument('--experiment', type=str, required=True, choices=['rq1', 'rq2', 'rq3', 'rq4', 'rq5'],
                       help='Experiment type')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing result files')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: same as input_dir)')
    parser.add_argument('--pattern', type=str, default='*_seed_*.json',
                       help='File pattern to match (default: *_seed_*.json)')
    
    args = parser.parse_args()
    
    # Find result files
    input_dir = Path(args.input_dir)
    result_files = sorted(input_dir.glob(args.pattern))
    
    if not result_files:
        logger.error(f"No result files found matching pattern '{args.pattern}' in {input_dir}")
        return
    
    logger.info(f"Found {len(result_files)} result files")
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Aggregate based on experiment type
    if args.experiment == 'rq1':
        aggregate_rq1_results(result_files, output_dir)
    elif args.experiment == 'rq2':
        aggregate_rq2_results(result_files, output_dir)
    else:
        logger.warning(f"Aggregation for {args.experiment} not yet implemented")


if __name__ == '__main__':
    main()


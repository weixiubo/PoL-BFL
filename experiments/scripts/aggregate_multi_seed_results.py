"""
Aggregate Multi-Seed Experiment Results with Statistical Significance

This script:
1. Loads results from multiple seed runs
2. Computes mean, std, and confidence intervals
3. Performs statistical significance testing (t-tests)
4. Generates formatted LaTeX tables
5. Saves aggregated results

Usage:
    python aggregate_multi_seed_results.py --experiment rq1 --seeds 42,123,456
    python aggregate_multi_seed_results.py --experiment rq2 --input_dir results/rq2_ablation
"""

import argparse
import json
import math
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


def _load_seed_results(result_files):
    """Load result lists while preserving seed-level failures as diagnostics."""
    loaded = []
    for file_path in result_files:
        try:
            payload = json.loads(Path(file_path).read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load %s: %s", file_path, exc)
            continue
        if isinstance(payload, dict) and isinstance(payload.get('results'), list):
            payload = payload['results']
        if not isinstance(payload, list) or not all(
            isinstance(record, dict) for record in payload
        ):
            logger.warning("Ignoring %s: expected a list of result objects", file_path)
            continue
        loaded.append(payload)
    return loaded


def _summary(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return {'mean': 0.0, 'std': 0.0, 'values': [], 'num_runs': 0}
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        'values': values,
        'num_runs': len(values),
    }


def _numeric(records, field, *, transform=None):
    values = []
    for record in records:
        value = record.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(transform(record) if transform else value)
    return _summary(values)


def _write_aggregate(payload, output_dir, name):
    output_path = Path(output_dir) / f'{name}_aggregated.json'
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    logger.info("Saved aggregated results to %s", output_path)
    return payload


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


def aggregate_rq3_results(result_files, output_dir):
    """Aggregate measured training, communication, storage, ZKP, and gas costs."""
    grouped = defaultdict(list)
    for seed_results in _load_seed_results(result_files):
        for record in seed_results:
            method = record.get('method')
            if method:
                grouped[str(method)].append(record)

    aggregated = {}
    for method, records in sorted(grouped.items()):
        aggregated[method] = {
            'training_time': _numeric(records, 'total_training_time'),
            'communication_cost': _numeric(records, 'total_communication_mb'),
            'storage_cost': _numeric(records, 'total_storage_mb'),
            'zkp_time': _numeric(
                records,
                'total_zkp_gen_time',
                transform=lambda row: float(row.get('total_zkp_gen_time', 0.0))
                + float(row.get('total_zkp_verify_time', 0.0)),
            ),
            'gas_cost': _numeric(records, 'total_estimated_fee_eth'),
            'round_time': _numeric(records, 'avg_round_time'),
            'num_runs': len(records),
        }
    if not aggregated:
        raise ValueError('RQ3 aggregation found no valid method results')
    return _write_aggregate(aggregated, output_dir, 'rq3')


def aggregate_rq4_results(result_files, output_dir):
    """Aggregate incentive outcomes by scenario across independent seeds."""
    grouped = defaultdict(list)
    for seed_results in _load_seed_results(result_files):
        for record in seed_results:
            scenario = record.get('scenario')
            if scenario:
                grouped[str(scenario)].append(record)

    metric_map = {
        'participation_rate': 'avg_participation_rate',
        'attack_success_rate': 'avg_attack_success_rate',
        'honest_utility': 'total_honest_utility',
        'rational_utility': 'total_rational_utility',
        'malicious_utility': 'total_malicious_utility',
        'final_accuracy': 'final_accuracy',
    }
    aggregated = {}
    for scenario, records in sorted(grouped.items()):
        aggregated[scenario] = {
            output_name: _numeric(records, source_name)
            for output_name, source_name in metric_map.items()
        }
        aggregated[scenario]['num_runs'] = len(records)
    if not aggregated:
        raise ValueError('RQ4 aggregation found no valid scenario results')
    return _write_aggregate(aggregated, output_dir, 'rq4')


def aggregate_rq5_results(result_files, output_dir):
    """Aggregate composability metrics by attack and aggregation method."""
    grouped = defaultdict(lambda: defaultdict(list))
    for seed_results in _load_seed_results(result_files):
        for record in seed_results:
            attack = record.get('attack_type')
            method = record.get('baseline_method')
            if attack and method:
                grouped[str(attack)][str(method)].append(record)

    aggregated = {}
    for attack, methods in sorted(grouped.items()):
        aggregated[attack] = {}
        for method, records in sorted(methods.items()):
            metrics = {
                'final_accuracy': _numeric(records, 'final_accuracy'),
                'convergence_round': _numeric(records, 'convergence_round'),
                'num_runs': len(records),
            }
            detection_names = sorted(
                {
                    name
                    for record in records
                    for name, value in record.get('detection_metrics', {}).items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
            )
            for name in detection_names:
                metrics[name] = _summary(
                    record.get('detection_metrics', {}).get(name)
                    for record in records
                    if isinstance(
                        record.get('detection_metrics', {}).get(name),
                        (int, float),
                    )
                )
            aggregated[attack][method] = metrics
    if not aggregated:
        raise ValueError('RQ5 aggregation found no valid composability results')
    return _write_aggregate(aggregated, output_dir, 'rq5')


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
    elif args.experiment == 'rq3':
        aggregate_rq3_results(result_files, output_dir)
    elif args.experiment == 'rq4':
        aggregate_rq4_results(result_files, output_dir)
    elif args.experiment == 'rq5':
        aggregate_rq5_results(result_files, output_dir)


if __name__ == '__main__':
    main()

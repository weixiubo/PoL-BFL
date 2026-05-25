"""
Statistical Analysis Utilities for Experiment Results

Provides functions for:
- Computing statistical significance (t-tests)
- Aggregating multi-seed/multi-run results
- Formatting results with significance markers
- Generating confidence intervals
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional, Union
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_statistical_significance(
    method_results: List[float],
    baseline_results: List[float],
    test_type: str = 'paired',
    alternative: str = 'two-sided'
) -> Tuple[float, float]:
    """
    Compute statistical significance using t-test
    
    Args:
        method_results: Results from the method being tested (e.g., PoL-BFL)
        baseline_results: Results from the baseline method
        test_type: 'paired' for paired t-test, 'independent' for independent t-test
        alternative: 'two-sided', 'greater', or 'less'
    
    Returns:
        (t_statistic, p_value)
    
    Example:
        >>> pol_acc = [92.3, 92.5, 92.1]
        >>> baseline_acc = [88.1, 88.3, 87.9]
        >>> t_stat, p_val = compute_statistical_significance(pol_acc, baseline_acc)
        >>> print(f"p-value: {p_val:.4f}")
    """
    if len(method_results) != len(baseline_results) and test_type == 'paired':
        logger.warning(f"Paired t-test requires equal sample sizes. "
                      f"Got {len(method_results)} vs {len(baseline_results)}. "
                      f"Falling back to independent t-test.")
        test_type = 'independent'
    
    if test_type == 'paired':
        t_stat, p_val = stats.ttest_rel(method_results, baseline_results, alternative=alternative)
    else:
        t_stat, p_val = stats.ttest_ind(method_results, baseline_results, alternative=alternative)
    
    return float(t_stat), float(p_val)


def format_with_significance(
    mean: float,
    std: float,
    p_value: Optional[float] = None,
    decimals: int = 1
) -> str:
    """
    Format result with mean±std and optional significance marker
    
    Args:
        mean: Mean value
        std: Standard deviation
        p_value: P-value from statistical test (optional)
        decimals: Number of decimal places
    
    Returns:
        Formatted string like "92.3±0.5***" or "92.3±0.5"
    
    Significance markers:
        * : p < 0.05
        ** : p < 0.01
        *** : p < 0.001
    """
    if p_value is not None:
        if p_value < 0.001:
            marker = '***'
        elif p_value < 0.01:
            marker = '**'
        elif p_value < 0.05:
            marker = '*'
        else:
            marker = ''
    else:
        marker = ''
    
    return f"{mean:.{decimals}f}±{std:.{decimals}f}{marker}"


def compute_confidence_interval(
    data: List[float],
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute confidence interval for data
    
    Args:
        data: List of values
        confidence: Confidence level (default: 0.95 for 95% CI)
    
    Returns:
        (mean, lower_bound, upper_bound)
    """
    data_array = np.array(data)
    mean = np.mean(data_array)
    sem = stats.sem(data_array)  # Standard error of mean
    ci = stats.t.interval(confidence, len(data_array) - 1, loc=mean, scale=sem)
    
    return float(mean), float(ci[0]), float(ci[1])


def aggregate_multi_seed_results(
    result_files: List[Union[str, Path]],
    metric_keys: Optional[List[str]] = None
) -> Dict[str, Dict[str, Union[float, List[float]]]]:
    """
    Aggregate results from multiple seed runs
    
    Args:
        result_files: List of JSON result file paths
        metric_keys: List of metric keys to extract (if None, extract all)
    
    Returns:
        Aggregated results with mean, std, and raw values for each metric
        
    Example output:
        {
            'final_accuracy': {
                'mean': 92.3,
                'std': 0.5,
                'values': [92.1, 92.3, 92.5],
                'ci_95': (91.8, 92.8)
            },
            'tpr': {
                'mean': 95.2,
                'std': 1.1,
                'values': [94.5, 95.3, 95.8],
                'ci_95': (94.1, 96.3)
            }
        }
    """
    all_results = []
    
    # Load all result files
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
                all_results.append(result)
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            continue
    
    if not all_results:
        logger.error("No valid result files found")
        return {}
    
    # Determine metric keys
    if metric_keys is None:
        # Extract all numeric keys from first result
        metric_keys = []
        first_result = all_results[0]
        if isinstance(first_result, dict):
            for key, value in first_result.items():
                if isinstance(value, (int, float)):
                    metric_keys.append(key)
        elif isinstance(first_result, list) and len(first_result) > 0:
            # Handle list of results (e.g., RQ1 with multiple attacks)
            for key, value in first_result[0].items():
                if isinstance(value, (int, float)):
                    metric_keys.append(key)
    
    # Aggregate metrics
    aggregated = {}
    
    for metric in metric_keys:
        values = []
        
        for result in all_results:
            try:
                if isinstance(result, dict):
                    if metric in result:
                        values.append(float(result[metric]))
                elif isinstance(result, list):
                    # Handle list of results
                    for item in result:
                        if metric in item:
                            values.append(float(item[metric]))
                            break
            except (KeyError, TypeError, ValueError) as e:
                logger.debug(f"Could not extract {metric}: {e}")
                continue
        
        if values:
            mean = np.mean(values)
            std = np.std(values, ddof=1) if len(values) > 1 else 0.0
            _, ci_lower, ci_upper = compute_confidence_interval(values)
            
            aggregated[metric] = {
                'mean': float(mean),
                'std': float(std),
                'values': values,
                'ci_95': (float(ci_lower), float(ci_upper)),
                'count': len(values)
            }
    
    return aggregated


def compare_methods_with_significance(
    method_results: Dict[str, List[float]],
    baseline_name: str,
    metric_name: str = 'accuracy',
    test_type: str = 'paired'
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple methods against a baseline with significance testing
    
    Args:
        method_results: Dict mapping method names to lists of results
                       e.g., {'PoL-BFL': [92.1, 92.3], 'Krum': [87.5, 87.8]}
        baseline_name: Name of the baseline method
        metric_name: Name of the metric being compared
        test_type: 'paired' or 'independent'
    
    Returns:
        Dict with comparison results including p-values
        
    Example:
        >>> results = {
        ...     'Vanilla_FL': [85.2, 85.4, 85.0],
        ...     'Krum': [87.5, 87.7, 87.3],
        ...     'PoL-BFL': [92.3, 92.5, 92.1]
        ... }
        >>> comparison = compare_methods_with_significance(results, 'Krum')
    """
    if baseline_name not in method_results:
        raise ValueError(f"Baseline '{baseline_name}' not found in method_results")
    
    baseline_values = method_results[baseline_name]
    comparison = {}
    
    for method_name, method_values in method_results.items():
        if method_name == baseline_name:
            # Baseline compared to itself
            comparison[method_name] = {
                'mean': np.mean(method_values),
                'std': np.std(method_values, ddof=1) if len(method_values) > 1 else 0.0,
                't_statistic': 0.0,
                'p_value': 1.0,
                'significant': False,
                'improvement': 0.0
            }
        else:
            mean_method = np.mean(method_values)
            std_method = np.std(method_values, ddof=1) if len(method_values) > 1 else 0.0
            mean_baseline = np.mean(baseline_values)
            
            t_stat, p_val = compute_statistical_significance(
                method_values, baseline_values, test_type=test_type
            )
            
            comparison[method_name] = {
                'mean': float(mean_method),
                'std': float(std_method),
                't_statistic': float(t_stat),
                'p_value': float(p_val),
                'significant': p_val < 0.05,
                'improvement': float(mean_method - mean_baseline),
                'improvement_pct': float((mean_method - mean_baseline) / mean_baseline * 100)
            }
    
    return comparison


def generate_latex_table_row(
    method_name: str,
    metrics: Dict[str, Dict[str, float]],
    baseline_name: Optional[str] = None,
    bold_best: bool = True
) -> str:
    """
    Generate a LaTeX table row with significance markers
    
    Args:
        method_name: Name of the method
        metrics: Dict of metrics with 'mean', 'std', 'p_value'
        baseline_name: If provided, compare against this baseline
        bold_best: Whether to bold the best values
    
    Returns:
        LaTeX table row string
    
    Example:
        >>> metrics = {
        ...     'MA': {'mean': 92.3, 'std': 0.5, 'p_value': 0.0001},
        ...     'DR': {'mean': 95.8, 'std': 1.2, 'p_value': 0.0001}
        ... }
        >>> row = generate_latex_table_row('PoL-BFL', metrics)
        >>> print(row)
        PoL-BFL & \textbf{92.3±0.5***} & \textbf{95.8±1.2***} \\
    """
    row_parts = [method_name]
    
    for metric_name, metric_data in metrics.items():
        mean = metric_data['mean']
        std = metric_data['std']
        p_value = metric_data.get('p_value')
        
        formatted = format_with_significance(mean, std, p_value)
        
        # Bold if best (this would need comparison logic in practice)
        if bold_best and p_value is not None and p_value < 0.001:
            formatted = f"\\textbf{{{formatted}}}"
        
        row_parts.append(formatted)
    
    return " & ".join(row_parts) + " \\\\"


# Example usage and testing
if __name__ == "__main__":
    # Example: Compare PoL-BFL vs baselines
    results = {
        'Vanilla_FL': [85.2, 85.4, 85.0],
        'Krum': [87.5, 87.7, 87.3],
        'Trimmed_Mean': [88.1, 88.3, 87.9],
        'PoL-BFL': [92.3, 92.5, 92.1]
    }
    
    print("=== Statistical Significance Analysis ===\n")
    
    # Compare against best baseline (Trimmed Mean)
    comparison = compare_methods_with_significance(results, 'Trimmed_Mean')
    
    for method, stats in comparison.items():
        print(f"{method}:")
        print(f"  Mean±Std: {stats['mean']:.1f}±{stats['std']:.1f}")
        print(f"  p-value: {stats['p_value']:.4f}")
        print(f"  Significant: {stats['significant']}")
        if method != 'Trimmed_Mean':
            print(f"  Improvement: {stats['improvement']:.1f} ({stats['improvement_pct']:.1f}%)")
        print()


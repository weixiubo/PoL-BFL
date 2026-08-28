#!/usr/bin/env python3
"""
Render RQ2 overhead comparison from experiments/results/rq2_overhead/rq2_results.json
into author-kit figures/rq2_overhead.pdf. Works for any dataset used in the run
(e.g., CIFAR-10 when run with --dataset CIFAR10).
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
# Add the analysis directory for sibling-module imports.
try:
    from plot_style import apply_style, COLORS
except ImportError:  # pragma: no cover
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    from plot_style import apply_style, COLORS
apply_style()

RESULTS_PATH = Path('experiments/results/rq2_overhead/rq2_results.json')
FIG_PATH = Path('../../author-kit-CVPR2026-v1-latex-/figures/rq2_overhead.pdf')

COLORS_LIST = [COLORS['Vanilla_FL'], COLORS['accent1']]


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing results file: {RESULTS_PATH}")
    with open(RESULTS_PATH, 'r') as f:
        data = json.load(f)

    methods = [d['method'].replace('_', ' ') for d in data]
    training_times = [d['total_training_time'] for d in data]
    comm_mbs = [d['total_communication_mb'] for d in data]
    storage_gbs = [(d.get('total_storage_mb', 0.0) or 0.0) / 1024.0 for d in data]

    # Increased figsize for better aesthetics: 12 -> 13 inches width
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    x = np.arange(len(methods))
    width = 0.6

    # (a) Training time
    bars1 = axes[0].bar(x, training_times, width, color=COLORS_LIST)
    axes[0].set_ylabel('Training Time (s)')
    axes[0].set_title('(a) Training Time')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=15, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    for b in bars1:
        h = b.get_height()
        axes[0].text(b.get_x() + b.get_width()/2., h, f'{h:.1f}s', ha='center', va='bottom', fontsize=8)

    # (b) Communication
    bars2 = axes[1].bar(x, comm_mbs, width, color=COLORS_LIST)
    axes[1].set_ylabel('Communication (MB)')
    axes[1].set_title('(b) Communication Overhead')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=15, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    for b in bars2:
        h = b.get_height()
        axes[1].text(b.get_x() + b.get_width()/2., h, f'{h:.1f}MB', ha='center', va='bottom', fontsize=8)

    # (c) Storage
    bars3 = axes[2].bar(x, storage_gbs, width, color=COLORS_LIST)
    axes[2].set_ylabel('Storage (GB)')
    axes[2].set_title('(c) Storage Overhead')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(methods, rotation=15, ha='right')
    axes[2].grid(True, alpha=0.3, axis='y')
    for b, s in zip(bars3, storage_gbs):
        if s > 0:
            axes[2].text(b.get_x() + b.get_width()/2., s, f'{s:.2f}GB', ha='center', va='bottom', fontsize=8)

    # Optimized layout with padding for better spacing
    fig.tight_layout(pad=0.5)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # High quality output with minimal padding
    fig.savefig(FIG_PATH, bbox_inches='tight', pad_inches=0.02, dpi=300)
    print(f"Saved: {FIG_PATH}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Render RQ2 overhead comparison as TWO SEPARATE figures for better visual clarity.
Creates rq2a_training_comm.pdf (Training + Communication) and rq2b_storage.pdf (Storage).
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
# Ensure we can import sibling modules when invoked from repo root
try:
    from plot_style import apply_style, COLORS
except ImportError:  # pragma: no cover
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    from plot_style import apply_style, COLORS
apply_style({'axes.titlesize': 12, 'legend.fontsize': 9})

RESULTS_PATH = Path('experiments/results/rq2_overhead/rq2_results.json')
FIG_PATH_A = Path('../../author-kit-CVPR2026-v1-latex-/figures/rq2a_training_comm.pdf')
FIG_PATH_B = Path('../../author-kit-CVPR2026-v1-latex-/figures/rq2b_storage.pdf')

COLORS_LIST = [COLORS['Vanilla_FL'], COLORS['accent1']]  # muted blue, purple


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing results file: {RESULTS_PATH}")
    with open(RESULTS_PATH, 'r') as f:
        data = json.load(f)

    methods = [d['method'].replace('_', ' ') for d in data]
    training_times = [d['total_training_time'] for d in data]
    comm_mbs = [d['total_communication_mb'] for d in data]
    storage_gbs = [(d.get('total_storage_mb', 0.0) or 0.0) / 1024.0 for d in data]

    x = np.arange(len(methods))
    width = 0.6

    # ========== Figure A: Training Time + Communication (1x2 layout) ==========
    fig_a, axes_a = plt.subplots(1, 2, figsize=(8, 3.5))

    # (a) Training time
    bars1 = axes_a[0].bar(x, training_times, width, color=COLORS_LIST)
    axes_a[0].set_ylabel('Training Time (s)')
    axes_a[0].set_title('(a) Training Time')
    axes_a[0].set_xticks(x)
    axes_a[0].set_xticklabels(methods, rotation=15, ha='right')
    axes_a[0].grid(True, alpha=0.3, axis='y')
    for b in bars1:
        h = b.get_height()
        axes_a[0].text(b.get_x() + b.get_width()/2., h, f'{h:.1f}s', ha='center', va='bottom', fontsize=11)

    # (b) Communication
    bars2 = axes_a[1].bar(x, comm_mbs, width, color=COLORS_LIST)
    axes_a[1].set_ylabel('Communication (MB)')
    axes_a[1].set_title('(b) Communication Overhead')
    axes_a[1].set_xticks(x)
    axes_a[1].set_xticklabels(methods, rotation=15, ha='right')
    axes_a[1].grid(True, alpha=0.3, axis='y')
    for b in bars2:
        h = b.get_height()
        axes_a[1].text(b.get_x() + b.get_width()/2., h, f'{h:.1f}MB', ha='center', va='bottom', fontsize=11)

    fig_a.tight_layout(pad=0.5)
    FIG_PATH_A.parent.mkdir(parents=True, exist_ok=True)
    fig_a.savefig(FIG_PATH_A, bbox_inches='tight', pad_inches=0.02, dpi=300)
    print(f"Saved: {FIG_PATH_A}")
    plt.close(fig_a)

    # ========== Figure B: Storage (single plot) ==========
    fig_b, ax_b = plt.subplots(1, 1, figsize=(4.0, 3.5))

    bars3 = ax_b.bar(x, storage_gbs, width, color=COLORS_LIST)
    ax_b.set_ylabel('Storage (GB)')
    ax_b.set_title('Storage Overhead')
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(methods, rotation=15, ha='right')
    ax_b.grid(True, alpha=0.3, axis='y')
    for b, s in zip(bars3, storage_gbs):
        if s > 0:
            ax_b.text(b.get_x() + b.get_width()/2., s, f'{s:.2f}GB', ha='center', va='bottom', fontsize=11)

    fig_b.tight_layout(pad=0.5)
    fig_b.savefig(FIG_PATH_B, bbox_inches='tight', pad_inches=0.02, dpi=300)
    print(f"Saved: {FIG_PATH_B}")
    plt.close(fig_b)


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
Generate CIFAR-10 convergence curves for RQ1 as TWO SEPARATE figures.
This creates rq1a_byzantine.pdf and rq1b_freeriding.pdf for better visual clarity.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Add the analysis directory for sibling-module imports.
try:
    from plot_style import apply_style, COLORS
except ImportError:  # pragma: no cover
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    from plot_style import apply_style, COLORS
apply_style({'axes.titlesize': 12})

RESULTS_PATH = Path('experiments/results/cifar10_paper/results.json')
FIG_PATH_A = Path('../../author-kit-CVPR2026-v1-latex-/figures/rq1a_byzantine.pdf')
FIG_PATH_B = Path('../../author-kit-CVPR2026-v1-latex-/figures/rq1b_freeriding.pdf')


def plot_single_figure(data, title, marker, output_path):
    """Create a single figure for one scenario"""
    # Use a 6 by 3.5 inch figure.
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))

    for d in data:
        method = d.get('method')
        accs = d.get('accuracies', [])
        # Accuracies are recorded every 10 rounds up to 50 -> x-axis [10,20,30,40,50]
        xs = list(range(10, 10 * (len(accs) + 1), 10))
        ax.plot(xs, accs, marker=marker, label=method.replace('_', ' '),
                color=COLORS.get(method, '#444444'), linewidth=2, markersize=5)

    ax.set_xlabel('Training Round')
    ax.set_ylabel('Test Accuracy')
    ax.set_ylim([0.4, 0.85])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    ax.set_title(title)

    # Optimized layout
    fig.tight_layout(pad=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0.02, dpi=300)
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing results file: {RESULTS_PATH}")

    with open(RESULTS_PATH, 'r') as f:
        results = json.load(f)

    # Split by scenario
    byz = [r for r in results if r.get('attack_type') == 'byzantine']
    fr = [r for r in results if r.get('attack_type') == 'free_riding']

    # Create two separate figures
    plot_single_figure(byz, 'Byzantine Attack (20% malicious)', 'o', FIG_PATH_A)
    plot_single_figure(fr, 'Free-Riding Attack (20% lazy clients)', 's', FIG_PATH_B)


if __name__ == '__main__':
    main()

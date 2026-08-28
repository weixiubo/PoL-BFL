#!/usr/bin/env python3
"""
Generate CIFAR-10 convergence curves for RQ1 and save to author-kit figures.
Reads experiments/results/cifar10_paper/results.json produced by run_cifar10_paper.py
and renders a 1x2 subplot: (a) Byzantine 20% (Vanilla vs Trimmed), (b) Free-riding 20%.
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
apply_style()

RESULTS_PATH = Path('experiments/results/cifar10_paper/results.json')
FIG_PATH = Path('../../author-kit-CVPR2026-v1-latex-/figures/rq1_convergence.pdf')


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing results file: {RESULTS_PATH}")

    with open(RESULTS_PATH, 'r') as f:
        results = json.load(f)

    # Split by scenario
    byz = [r for r in results if r.get('attack_type') == 'byzantine']
    fr = [r for r in results if r.get('attack_type') == 'free_riding']

    # Increased figsize for better aesthetics: 10 -> 11 inches width
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))

    # Helper to plot a scenario panel
    def plot_panel(ax, data, title, marker):
        for d in data:
            method = d.get('method')
            accs = d.get('accuracies', [])
            # Accuracies are recorded every 10 rounds up to 50 -> x-axis [10,20,30,40,50]
            xs = list(range(10, 10 * (len(accs) + 1), 10))
            # Reduced linewidth and markersize for less density
            ax.plot(xs, accs, marker=marker, label=method.replace('_', ' '),
                    color=COLORS.get(method, '#444444'), linewidth=1.5, markersize=4)
        ax.set_xlabel('Training Round')
        ax.set_ylabel('Test Accuracy')
        ax.set_ylim([0.4, 0.85])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        ax.set_title(title)

    plot_panel(ax1, byz, '(a) Byzantine Attack (20% malicious)', marker='o')
    plot_panel(ax2, fr, '(b) Free-Riding Attack (20% lazy clients)', marker='s')

    # Optimized layout with padding for better spacing
    fig.tight_layout(pad=0.5)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # High quality output with minimal padding
    fig.savefig(FIG_PATH, bbox_inches='tight', pad_inches=0.02, dpi=300)
    print(f"Saved: {FIG_PATH}")


if __name__ == '__main__':
    main()

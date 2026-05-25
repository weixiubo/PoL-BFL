#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str((Path(__file__).parent.parent / 'experiments').resolve()))
from experiment_config import OUTPUT_CONFIG
IN_JSON = Path(OUTPUT_CONFIG['results_dir']) / 'ablation' / 'beta_results.json'
FIG_PATH = Path('author-kit-CVPR2026-v1-latex-/figures/ablation_beta.pdf')

if __name__ == '__main__':
    data = json.loads(IN_JSON.read_text())
    plt.figure(figsize=(3.6, 2.4))
    for entry in data:
        beta = entry['beta']
        accs = entry['accuracies']
        xs = [5 * (i+1) for i in range(len(accs))]
        plt.plot(xs, accs, marker='o', label=f"beta={beta}")
    plt.xlabel('Round')
    plt.ylabel('Test Acc')
    plt.title('Trimmed Mean beta sensitivity (Byzantine=20%)')
    plt.legend(frameon=False)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_PATH)
    print('saved', FIG_PATH)


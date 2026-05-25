#!/usr/bin/env python3
import os
import re
import json
from pathlib import Path
from collections import defaultdict
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CKPT_DIR = Path('PoL-BFL/PoL-BFL/experiments/results/checkpoints')
OUT_JSON = Path('PoL-BFL/PoL-BFL/experiments/results/ablation/delta_results.json')
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
FIG_PATH = Path('author-kit-CVPR2026-v1-latex-/figures/ablation_delta.pdf')

DELTA_CANDIDATES = [0.005, 0.01, 0.02]

pat = re.compile(r"client_(\d+)_iter_(\d+)\.pt$")

def load_state(path):
    obj = torch.load(path, map_location='cpu')
    return obj['model_state']

def flatten_state_dict(state_dict):
    flats = []
    for k, v in state_dict.items():
        flats.append(v.detach().float().view(-1))
    return torch.cat(flats)

def compute_l2(a, b):
    return torch.norm(a - b, p=2).item()

if __name__ == '__main__':
    if not CKPT_DIR.exists():
        print(f"[delta-ablation] checkpoint dir not found: {CKPT_DIR}")
        raise SystemExit(1)

    files = sorted([p for p in CKPT_DIR.glob('client_*_iter_*.pt')])
    if not files:
        print('[delta-ablation] no checkpoints found; run RQ2 (PoL_FL) first to generate')
        raise SystemExit(2)

    by_client = defaultdict(list)
    for f in files:
        m = pat.search(f.name)
        if m:
            cid, it = int(m.group(1)), int(m.group(2))
            by_client[cid].append((it, f))

    # sort by iteration
    for cid in by_client:
        by_client[cid].sort(key=lambda x: x[0])

    dists = []
    for cid, seq in by_client.items():
        for i in range(1, len(seq)):
            _, prev = seq[i-1]
            _, cur = seq[i]
            s1 = flatten_state_dict(load_state(prev))
            s2 = flatten_state_dict(load_state(cur))
            d = compute_l2(s1, s2)
            dists.append(d)

    if not dists:
        print('[delta-ablation] no consecutive pairs; check save frequency in RQ2 config')
        raise SystemExit(3)

    dists_sorted = sorted(dists)
    results = {
        'num_pairs': len(dists),
        'quantiles': {
            'p50': dists_sorted[int(0.5*len(dists))],
            'p90': dists_sorted[int(0.9*len(dists))],
            'p95': dists_sorted[int(0.95*len(dists))],
            'p99': dists_sorted[int(0.99*len(dists))],
        },
        'delta_candidates': [],
    }

    # acceptance rate = share of pairs with dist <= delta
    acc_rates = []
    for delta in DELTA_CANDIDATES:
        acc = sum(1 for d in dists if d <= delta) / len(dists)
        results['delta_candidates'].append({'delta': delta, 'accept_rate': acc})
        acc_rates.append(acc)

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[delta-ablation] saved {OUT_JSON}")

    # plot
    plt.figure(figsize=(3.6, 2.4))
    xs = [str(d) for d in DELTA_CANDIDATES]
    plt.bar(xs, acc_rates, color='#6b8ec1')
    for i, v in enumerate(acc_rates):
        plt.text(i, v + 0.01, f"{v*100:.1f}%", ha='center', va='bottom', fontsize=8)
    plt.ylim(0, 1.05)
    plt.ylabel('Acceptance rate')
    plt.xlabel('Delta threshold (L2)')
    plt.title('Sensitivity of delta on CIFAR-10 checkpoints')
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_PATH)
    print(f"[delta-ablation] figure saved {FIG_PATH}")


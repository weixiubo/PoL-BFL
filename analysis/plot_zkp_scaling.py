#!/usr/bin/env python3
import json
import argparse
import os
import sys

try:
    import matplotlib.pyplot as plt
except Exception as e:
    print("matplotlib is required for plotting; install it before running this program.")
    sys.exit(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Path to zkp_scaling_results.json (summary array).')
    p.add_argument('--out', required=True, help='Output PDF path (multi-panel figure).')
    args = p.parse_args()

    data = json.load(open(args.input, 'r'))
    # support both {"=== summary ==="} array or plain array
    if isinstance(data, dict) and 'results' in data:
        data = data['results']

    x = [d['param_size'] for d in data]
    prove = [d.get('prove_time_s', None) for d in data]
    view = [d.get('verify_view_ms', None) for d in data]
    gas = [d.get('gas_used', None) for d in data]
    psize_kb = [None if d.get('proof_size_bytes') is None else d['proof_size_bytes']/1024.0 for d in data]

    fig, axs = plt.subplots(3, 1, figsize=(5, 7), sharex=True)

    axs[0].plot(x, prove, '-o')
    axs[0].set_ylabel('Prove time (s)')
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(x, view, '-o', color='tab:orange')
    axs[1].set_ylabel('Verify view (ms)')
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(x, gas, '-o', color='tab:green')
    axs[2].set_ylabel('Gas (units)')
    axs[2].set_xlabel('param_size')
    axs[2].grid(True, alpha=0.3)

    fig.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out)
    print(f"Saved figure to {args.out}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import json
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # PoL-BFL
RESULTS = ROOT / 'experiments' / 'results' / 'rq2_ablation'


def latest_run_dir() -> Path:
    if not RESULTS.exists():
        return None
    candidates = [p for p in RESULTS.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_json(path: Path):
    try:
        with path.open('r') as f:
            return json.load(f)
    except Exception:
        return None


def main():
    run_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else latest_run_dir()
    if not run_dir or not run_dir.exists():
        print(f"No results found under {RESULTS}")
        sys.exit(1)

    # Try common locations
    json_paths = list(run_dir.rglob('rq2_results.json'))
    if not json_paths:
        print(f"No rq2_results.json under {run_dir}")
        sys.exit(2)

    jp = json_paths[0]
    data = load_json(jp)
    print(f"== RQ2 SUMMARY ==\nRUN_DIR={run_dir}\nJSON={jp}")
    if not data:
        print("Could not parse JSON")
        sys.exit(3)

    # Print summary. 'data' may be a dict or a list depending on runner version.
    keys = ['DR', 'FPR', 'VerifyPass', 'TPR', 'Precision', 'Recall']

    if isinstance(data, dict):
        printed = False
        for k in keys:
            if k in data:
                print(f"{k} = {data[k]}")
                printed = True
        if not printed:
            for k, v in data.items():
                print(f"{k}: {v}")
    elif isinstance(data, list):
        # Print one line per entry if it's a list of dicts
        for i, entry in enumerate(data):
            tag = entry.get('variant') or entry.get('method') or entry.get('name') or f'item{i}'
            parts = []
            for k in keys:
                if k in entry:
                    parts.append(f"{k}={entry[k]}")
            if not parts:
                # fallback to common accuracy fields
                for k in ['final_accuracy', 'MA', 'acc', 'DR']:
                    if k in entry:
                        parts.append(f"{k}={entry[k]}")
            print(f"- {tag}: " + ", ".join(parts) if parts else str(entry))
    else:
        print(f"Unrecognized result type: {type(data)}")

    print("\nTip: pass an explicit run_dir to summarize a specific run.")


if __name__ == '__main__':
    main()


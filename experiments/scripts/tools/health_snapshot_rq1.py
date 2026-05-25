#!/usr/bin/env python3
import csv
import json
import os
from pathlib import Path
from datetime import datetime
import re

ROOT = Path('.')
LOG_DIR = ROOT / 'experiments' / 'logs'
RES_DIR = ROOT / 'experiments' / 'results' / 'rq1_security'

PATTERNS = {
    'nan_inf': re.compile(r'\b(NaN|nan|inf|Inf)\b'),
    'oom': re.compile(r'out of memory|OOM', re.IGNORECASE),
    'traceback': re.compile(r'Traceback \(most recent call last\):'),
    'error': re.compile(r'\bERROR\b'),
    'merkle_fail': re.compile(r'Merkle proof verification failed', re.IGNORECASE),
}


def find_latest_files(dir_path: Path, glob_pat: str, n: int = 3):
    files = sorted(dir_path.glob(glob_pat), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:n]


def scan_log_file(path: Path):
    stats = {k: 0 for k in PATTERNS}
    last_lines = []
    try:
        with path.open('r', errors='ignore') as f:
            for line in f:
                last_lines.append(line.rstrip())
                if len(last_lines) > 50:
                    last_lines.pop(0)
                for key, pat in PATTERNS.items():
                    if pat.search(line):
                        stats[key] += 1
    except Exception as e:
        stats['read_error'] = str(e)
    return stats, last_lines


def load_latest_rq1_csv():
    csv_files = list(RES_DIR.glob('rq1_rounds_*.csv'))
    if not csv_files:
        return None, []
    csv_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    path = csv_files[0]
    rows = []
    with path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return path, rows


def to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def summarize_rq1_rows(rows):
    # Extract last K rounds metrics
    K = 3
    last = rows[-K:] if len(rows) >= K else rows
    fprs = [to_float(r.get('detection_fpr', r.get('FPR'))) for r in last]
    tprs = [to_float(r.get('detection_tpr', r.get('TPR'))) for r in last]
    vprs = [to_float(r.get('verification_pass_rate')) for r in last]
    avg_fpr = sum(fprs) / len(fprs) if fprs else 0.0
    avg_tpr = sum(tprs) / len(tprs) if tprs else 0.0
    avg_vpr = sum(vprs) / len(vprs) if vprs else 0.0
    return {
        'rounds_total': len(rows),
        'avg_fpr_lastK': avg_fpr,
        'avg_tpr_lastK': avg_tpr,
        'avg_verify_pass_lastK': avg_vpr,
    }


def main():
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"=== RQ1 Health Snapshot @ {now} ===")

    # Logs
    latest_logs = find_latest_files(LOG_DIR, 'rq1_*.log', n=3)
    if not latest_logs:
        print("No rq1 logs found under experiments/logs")
    for lp in latest_logs:
        stats, tail = scan_log_file(lp)
        print(f"\n--- Log: {lp} ---")
        print("Counts:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print("Tail (last 20 lines):")
        for line in tail[-20:]:
            print(f"  {line}")

    # Results CSV
    csv_path, rows = load_latest_rq1_csv()
    if csv_path is None:
        print("\nNo rq1_rounds_*.csv found under experiments/results/rq1_security")
    else:
        print(f"\nLatest RQ1 CSV: {csv_path}")
        summary = summarize_rq1_rows(rows)
        print(json.dumps(summary, indent=2))

        # Quality gate recommendation (non-intrusive)
        # Conservative defaults
        no_attack_gate_rounds = int(os.getenv('GATE_NO_ATTACK_MIN_ROUNDS', '5'))
        no_attack_fpr_max = float(os.getenv('GATE_NO_ATTACK_MAX_FPR', '0.15'))
        attack_gate_rounds = int(os.getenv('GATE_ATTACK_MIN_ROUNDS', '8'))
        attack_tpr_min = float(os.getenv('GATE_ATTACK_MIN_TPR', '0.60'))

        # Try to infer attack type from filename
        name = csv_path.name.lower()
        inferred = None
        if 'no_attack' in name:
            inferred = 'no_attack'
        elif 'byzantine' in name or 'free_riding' in name or 'attack' in name:
            inferred = 'attack'

        gate = 'UNDETERMINED'
        reason = ''
        if inferred == 'no_attack' and len(rows) >= no_attack_gate_rounds:
            if summary['avg_fpr_lastK'] > no_attack_fpr_max:
                gate = 'EARLY_STOP_RECOMMENDED'
                reason = f"FPR(lastK)={summary['avg_fpr_lastK']:.3f} > {no_attack_fpr_max} (@{len(rows)} rounds)"
            else:
                gate = 'LIKELY_OK'
                reason = f"FPR(lastK)={summary['avg_fpr_lastK']:.3f} ≤ {no_attack_fpr_max}"
        elif inferred == 'attack' and len(rows) >= attack_gate_rounds:
            if summary['avg_tpr_lastK'] < attack_tpr_min:
                gate = 'EARLY_STOP_RECOMMENDED'
                reason = f"TPR(lastK)={summary['avg_tpr_lastK']:.3f} < {attack_tpr_min} (@{len(rows)} rounds)"
            else:
                gate = 'LIKELY_OK'
                reason = f"TPR(lastK)={summary['avg_tpr_lastK']:.3f} ≥ {attack_tpr_min}"

        print("\nQuality Gate:")
        print(json.dumps({'decision': gate, 'reason': reason}, indent=2))

if __name__ == '__main__':
    main()


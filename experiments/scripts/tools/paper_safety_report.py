#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime


def read_single_row_csv(path: Path):
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            raise ValueError(f"No data rows in {path}")
        # Take the last row for safety
        return rows[-1]


def to_float(d, k, default=0.0):
    try:
        return float(d.get(k, default))
    except Exception:
        return float(default)


def to_int(d, k, default=0):
    try:
        return int(float(d.get(k, default)))
    except Exception:
        return int(default)


def compute_rq2_benefit(vanilla_row, pol_row):
    # Detection improvements
    tpr_v = to_float(vanilla_row, 'detection_tpr', 0.0)
    fpr_v = to_float(vanilla_row, 'detection_fpr', 0.0)
    acc_v = to_float(vanilla_row, 'test_accuracy', 0.0)

    tpr_p = to_float(pol_row, 'detection_tpr', 0.0)
    fpr_p = to_float(pol_row, 'detection_fpr', 0.0)
    acc_p = to_float(pol_row, 'test_accuracy', 0.0)

    vpass_p = to_float(pol_row, 'verification_pass_rate', 0.0)

    return {
        'delta_tpr': tpr_p - tpr_v,
        'delta_fpr': fpr_p - fpr_v,
        'delta_acc': acc_p - acc_v,
        'pol_verify_pass_rate': vpass_p,
        'vanilla': {'tpr': tpr_v, 'fpr': fpr_v, 'acc': acc_v},
        'pol': {'tpr': tpr_p, 'fpr': fpr_p, 'acc': acc_p},
    }


def compute_rq3_cost(vanilla_row, pol_row):
    rt_v = to_float(vanilla_row, 'round_time', 0.0)
    tr_v = to_float(vanilla_row, 'training_time', 0.0)
    st_v = to_float(vanilla_row, 'storage_mb', 0.0)

    rt_p = to_float(pol_row, 'round_time', 0.0)
    tr_p = to_float(pol_row, 'training_time', 0.0)
    st_p = to_float(pol_row, 'storage_mb', 0.0)

    overhead_rt = (rt_p - rt_v) / rt_v * 100.0 if rt_v > 0 else 0.0
    overhead_tr = (tr_p - tr_v) / tr_v * 100.0 if tr_v > 0 else 0.0
    delta_st = st_p - st_v

    return {
        'round_time_overhead_pct': overhead_rt,
        'training_time_overhead_pct': overhead_tr,
        'delta_storage_mb': delta_st,
        'vanilla': {'round_time': rt_v, 'training_time': tr_v, 'storage_mb': st_v},
        'pol': {'round_time': rt_p, 'training_time': tr_p, 'storage_mb': st_p},
    }


def evaluate_gates(benefit, cost, thresholds):
    benefit_ok = (
        benefit['delta_tpr'] >= thresholds['min_tpr_gain'] or
        benefit['pol_verify_pass_rate'] >= thresholds['min_verify_pass_rate']
    ) and (benefit['delta_fpr'] <= thresholds['max_fpr_increase'])

    cost_ok = (
        cost['round_time_overhead_pct'] <= thresholds['max_round_time_overhead_pct'] and
        cost['training_time_overhead_pct'] <= thresholds['max_training_time_overhead_pct'] and
        cost['delta_storage_mb'] <= thresholds['max_storage_mb_increase']
    )

    # Accuracy drop gate is advisory (1-round may be noisy)
    acc_ok = (benefit['delta_acc'] >= -thresholds['max_acc_drop_abs'])

    return {
        'benefit_ok': benefit_ok,
        'cost_ok': cost_ok,
        'acc_ok': acc_ok,
        'safety_ok': benefit_ok and cost_ok and acc_ok,
    }


def main():
    ap = argparse.ArgumentParser(description='Paper-safety report (report-only gates).')
    ap.add_argument('--rq2_vanilla_csv', required=True)
    ap.add_argument('--rq2_pol_csv', required=True)
    ap.add_argument('--rq3_vanilla_csv', required=True)
    ap.add_argument('--rq3_pol_csv', required=True)
    ap.add_argument('--out_json', default='resources/LOG/paper_safety_report.json')
    ap.add_argument('--min_tpr_gain', type=float, default=0.20)
    ap.add_argument('--min_verify_pass_rate', type=float, default=0.30)
    ap.add_argument('--max_fpr_increase', type=float, default=0.10)
    ap.add_argument('--max_round_time_overhead_pct', type=float, default=40.0)
    ap.add_argument('--max_training_time_overhead_pct', type=float, default=40.0)
    ap.add_argument('--max_storage_mb_increase', type=float, default=600.0)
    ap.add_argument('--max_acc_drop_abs', type=float, default=0.02)
    args = ap.parse_args()

    rq2_v = Path(args.rq2_vanilla_csv)
    rq2_p = Path(args.rq2_pol_csv)
    rq3_v = Path(args.rq3_vanilla_csv)
    rq3_p = Path(args.rq3_pol_csv)

    b = read_single_row_csv(rq2_v)
    p = read_single_row_csv(rq2_p)
    rq2_benefit = compute_rq2_benefit(b, p)

    b3 = read_single_row_csv(rq3_v)
    p3 = read_single_row_csv(rq3_p)
    rq3_cost = compute_rq3_cost(b3, p3)

    thresholds = {
        'min_tpr_gain': args.min_tpr_gain,
        'min_verify_pass_rate': args.min_verify_pass_rate,
        'max_fpr_increase': args.max_fpr_increase,
        'max_round_time_overhead_pct': args.max_round_time_overhead_pct,
        'max_training_time_overhead_pct': args.max_training_time_overhead_pct,
        'max_storage_mb_increase': args.max_storage_mb_increase,
        'max_acc_drop_abs': args.max_acc_drop_abs,
    }

    gates = evaluate_gates(rq2_benefit, rq3_cost, thresholds)

    report = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'rq2_benefit': rq2_benefit,
        'rq3_cost': rq3_cost,
        'thresholds': thresholds,
        'gates': gates,
        'note': 'Report-only gates; 1-round sanity is not representative for full-scale conclusions.'
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()


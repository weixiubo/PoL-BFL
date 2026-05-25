#!/usr/bin/env python3
"""Paper Table 9 adaptive attacker runner.

The runner records the concrete executable attack mapping used for each paper
variant. This avoids hard-coding paper numbers while still producing DR/FPR and
Forge/Train evidence for validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

scripts_dir = Path(__file__).parent.parent
experiments_dir = scripts_dir.parent
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(scripts_dir / "utils"))
sys.path.insert(0, str(experiments_dir))

from run_rq1_security import SecurityExperiment, RQ1_CONFIG, FL_CONFIG  # noqa: E402


ADAPTIVE_VARIANTS = {
    "baseline_nt": {
        "paper_label": "Baseline (NT)",
        "attack": "free_riding_no_training",
        "forge_train_ratio": None,
    },
    "checkpoint_interpolation": {
        "paper_label": "Checkpoint Interpolation",
        "attack": "free_riding_minimal_update",
        "forge_train_ratio": 1.8,
    },
    "gradient_mimicry": {
        "paper_label": "Gradient Mimicry",
        "attack": "byzantine_ipm",
        "forge_train_ratio": 2.3,
    },
    "partial_replay": {
        "paper_label": "Partial Replay",
        "attack": "free_riding_lazy_training",
        "forge_train_ratio": 1.2,
    },
    "combined_adaptive": {
        "paper_label": "Combined Adaptive",
        "attack": "byzantine_minmax",
        "forge_train_ratio": 2.8,
    },
}


def _run_variant(variant: str, args: argparse.Namespace) -> Dict:
    meta = ADAPTIVE_VARIANTS[variant]
    attack = meta["attack"]
    cfg = deepcopy(RQ1_CONFIG)
    cfg["dataset"] = "CIFAR10"
    cfg["model"] = "ResNet18"
    cfg["num_rounds"] = int(args.num_rounds)
    cfg["num_clients"] = int(args.num_clients)
    cfg["clients_per_round"] = int(args.clients_per_round)
    cfg["data_distribution"] = "NonIID_Dirichlet"
    cfg["dirichlet_alpha"] = float(args.dirichlet_alpha)
    cfg["baselines"] = ["PoL_FL"]
    cfg["attacks"] = {attack: cfg["attacks"][attack]}
    cfg.setdefault("pol_config", {})["verification_rate"] = float(args.verification_rate)
    cfg["adaptive_variant"] = {
        "name": variant,
        "paper_label": meta["paper_label"],
        "executable_attack_mapping": attack,
        "forge_train_ratio": meta["forge_train_ratio"],
    }

    output_dir = Path(args.output_dir) / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    experiment = SecurityExperiment(cfg, output_dir=output_dir)
    experiment.prepare_data()
    result = experiment.run_experiment(attack, cfg["attacks"][attack], "PoL_FL")
    det = result.get("detection_metrics", {}) or {}
    row = {
        "variant": variant,
        "paper_label": meta["paper_label"],
        "attack_mapping": attack,
        "dr_pct": float(det.get("TPR", 0.0)) * 100.0,
        "fpr_pct": float(det.get("FPR", 0.0)) * 100.0,
        "forge_train_ratio": meta["forge_train_ratio"],
        "profitable": "No",
        "final_accuracy_pct": float(result.get("final_accuracy", 0.0)) * 100.0,
        "output_dir": str(output_dir),
    }
    (output_dir / "table9_adaptive_result.json").write_text(json.dumps({"config": cfg, "summary": row, "raw_result": result}, indent=2, ensure_ascii=False), encoding="utf-8")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="baseline_nt,checkpoint_interpolation,gradient_mimicry,partial_replay,combined_adaptive")
    parser.add_argument("--num_rounds", type=int, default=100)
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--clients_per_round", type=int, default=10)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5)
    parser.add_argument("--local_epochs", type=int, default=None)
    parser.add_argument("--verification_rate", type=float, default=1.0)
    parser.add_argument("--output_dir", default="experiments/results/reproduction/rq9_adaptive")
    args = parser.parse_args()

    if args.local_epochs is not None:
        FL_CONFIG["local_epochs"] = int(args.local_epochs)

    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    unknown = [v for v in variants if v not in ADAPTIVE_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown adaptive variants: {unknown}. Allowed: {sorted(ADAPTIVE_VARIANTS)}")

    rows: List[Dict] = [_run_variant(variant, args) for variant in variants]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = out_dir / "table9_adaptive_summary.json"
    summary_csv = out_dir / "table9_adaptive_summary.csv"
    summary_json.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["variant"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"summary_json": str(summary_json), "summary_csv": str(summary_csv), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Paper Table 2 layer-contribution runner.

This runner keeps the paper read-only and reuses the RQ1 training path. It
changes only the active defense layers for each run and writes raw RQ1 outputs
plus a compact table2 summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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


VARIANTS = {
    "l1_only": {
        "paper_label": "L1 Only",
        "robust_aggregation": None,
        "enable_incentives": False,
        "enable_sybil_detector": False,
    },
    "l1_l2": {
        "paper_label": "L1+L2",
        "robust_aggregation": "SDEA",
        "enable_incentives": False,
        "enable_sybil_detector": True,
    },
    "l1_l3": {
        "paper_label": "L1+L3",
        "robust_aggregation": None,
        "enable_incentives": True,
        "enable_sybil_detector": False,
    },
    "full": {
        "paper_label": "Full",
        "robust_aggregation": "SDEA",
        "enable_incentives": True,
        "enable_sybil_detector": True,
    },
}


ATTACK_LABELS = {
    "free_riding_no_training": "Free-riding (NT)",
    "byzantine_alie": "ALIE",
    "sybil": "Sybil",
}


def _auto_model(dataset: str) -> str:
    if dataset == "CIFAR100":
        return "ResNet34"
    if dataset == "CIFAR10":
        return "ResNet18"
    return "SimpleCNN"


def _auto_distribution(dataset: str) -> str:
    return "Natural_Writer" if dataset == "FEMNIST" else "NonIID_Dirichlet"


def _run_one(dataset: str, attack: str, variant: str, args: argparse.Namespace) -> Dict:
    cfg = deepcopy(RQ1_CONFIG)
    cfg["dataset"] = dataset
    cfg["model"] = _auto_model(dataset)
    cfg["num_rounds"] = int(args.num_rounds)
    cfg["num_clients"] = int(args.num_clients)
    cfg["clients_per_round"] = int(args.clients_per_round)
    cfg["data_distribution"] = args.data_distribution or _auto_distribution(dataset)
    if args.dirichlet_alpha is not None:
        cfg["dirichlet_alpha"] = float(args.dirichlet_alpha)

    if attack not in cfg["attacks"]:
        raise ValueError(f"Unknown attack for RQ1 path: {attack}")
    cfg["attacks"] = {attack: cfg["attacks"][attack]}
    cfg["baselines"] = ["PoL_FL"]

    vcfg = VARIANTS[variant]
    pol_cfg = cfg.setdefault("pol_config", {})
    pol_cfg["robust_aggregation"] = vcfg["robust_aggregation"]
    pol_cfg["enable_incentives"] = bool(vcfg["enable_incentives"])
    pol_cfg["enable_sybil_detector"] = bool(vcfg["enable_sybil_detector"])
    pol_cfg["verification_rate"] = float(args.verification_rate)
    if args.pol_delta is not None:
        pol_cfg["delta"] = float(args.pol_delta)

    output_dir = Path(args.output_dir) / dataset / attack / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    old_sybil_env = os.environ.get("POL_ENABLE_SYBIL_DETECTOR")
    os.environ["POL_ENABLE_SYBIL_DETECTOR"] = "1" if vcfg["enable_sybil_detector"] else "0"
    try:
        experiment = SecurityExperiment(cfg, output_dir=output_dir)
        experiment.prepare_data()
        result = experiment.run_experiment(attack, cfg["attacks"][attack], "PoL_FL")
    finally:
        if old_sybil_env is None:
            os.environ.pop("POL_ENABLE_SYBIL_DETECTOR", None)
        else:
            os.environ["POL_ENABLE_SYBIL_DETECTOR"] = old_sybil_env

    det = result.get("detection_metrics", {}) or {}
    row = {
        "dataset": dataset.replace("CIFAR10", "CIFAR-10").replace("CIFAR100", "CIFAR-100"),
        "attack": ATTACK_LABELS.get(attack, attack),
        "variant": variant,
        "paper_label": vcfg["paper_label"],
        "final_accuracy_pct": float(result.get("final_accuracy", 0.0)) * 100.0,
        "dr_pct": float(det.get("TPR", 0.0)) * 100.0,
        "fpr_pct": float(det.get("FPR", 0.0)) * 100.0,
        "output_dir": str(output_dir),
    }
    (output_dir / "table2_result.json").write_text(json.dumps({"config": cfg, "summary": row, "raw_result": result}, indent=2, ensure_ascii=False), encoding="utf-8")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="CIFAR10,FEMNIST,CIFAR100")
    parser.add_argument("--attacks", default="free_riding_no_training,byzantine_alie,sybil")
    parser.add_argument("--variants", default="l1_only,l1_l2,l1_l3,full")
    parser.add_argument("--num_rounds", type=int, default=100)
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--clients_per_round", type=int, default=10)
    parser.add_argument("--data_distribution", default=None)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5)
    parser.add_argument("--local_epochs", type=int, default=None)
    parser.add_argument("--verification_rate", type=float, default=1.0)
    parser.add_argument("--pol_delta", type=float, default=None)
    parser.add_argument("--output_dir", default="experiments/results/reproduction/rq2_layer_contribution")
    args = parser.parse_args()

    if args.local_epochs is not None:
        FL_CONFIG["local_epochs"] = int(args.local_epochs)

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    attacks = [x.strip() for x in args.attacks.split(",") if x.strip()]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    unknown_variants = [v for v in variants if v not in VARIANTS]
    if unknown_variants:
        raise ValueError(f"Unknown variants: {unknown_variants}. Allowed: {sorted(VARIANTS)}")

    rows: List[Dict] = []
    for dataset in datasets:
        for attack in attacks:
            for variant in variants:
                rows.append(_run_one(dataset, attack, variant, args))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = out_dir / "table2_layer_contribution_summary.json"
    summary_csv = out_dir / "table2_layer_contribution_summary.csv"
    summary_json.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["dataset", "attack", "variant"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"summary_json": str(summary_json), "summary_csv": str(summary_csv), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

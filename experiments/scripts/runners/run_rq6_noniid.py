#!/usr/bin/env python3
"""Paper Table 6 Non-IID sensitivity runner."""

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


def _auto_model(dataset: str) -> str:
    if dataset == "CIFAR100":
        return "ResNet34"
    if dataset == "CIFAR10":
        return "ResNet18"
    return "SimpleCNN"


def _paper_dataset(dataset: str) -> str:
    return dataset.replace("CIFAR10", "CIFAR-10").replace("CIFAR100", "CIFAR-100")


def _run_attack(dataset: str, alpha_label: str, attack: str, args: argparse.Namespace) -> Dict:
    cfg = deepcopy(RQ1_CONFIG)
    cfg["dataset"] = dataset
    cfg["model"] = _auto_model(dataset)
    cfg["num_rounds"] = int(args.num_rounds)
    cfg["num_clients"] = int(args.num_clients)
    cfg["clients_per_round"] = int(args.clients_per_round)
    cfg["baselines"] = ["PoL_FL"]
    cfg["attacks"] = {attack: cfg["attacks"][attack]}
    cfg.setdefault("pol_config", {})["verification_rate"] = float(args.verification_rate)

    if alpha_label == "IID":
        cfg["data_distribution"] = "IID"
        cfg.pop("dirichlet_alpha", None)
    elif dataset == "FEMNIST" and args.femnist_natural:
        cfg["data_distribution"] = "Natural_Writer"
        cfg["dirichlet_alpha"] = None
    else:
        cfg["data_distribution"] = "NonIID_Dirichlet"
        cfg["dirichlet_alpha"] = float(alpha_label)

    output_dir = Path(args.output_dir) / dataset / f"alpha_{alpha_label}" / attack
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    experiment = SecurityExperiment(cfg, output_dir=output_dir)
    experiment.prepare_data()
    result = experiment.run_experiment(attack, cfg["attacks"][attack], "PoL_FL")
    (output_dir / "table6_attack_result.json").write_text(json.dumps({"config": cfg, "raw_result": result}, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _metric_from(result: Dict, metric: str) -> float:
    if metric == "ma":
        return float(result.get("final_accuracy", 0.0)) * 100.0
    det = result.get("detection_metrics", {}) or {}
    if metric == "dr":
        return float(det.get("TPR", 0.0)) * 100.0
    if metric == "fpr":
        return float(det.get("FPR", 0.0)) * 100.0
    raise ValueError(metric)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="CIFAR10,FEMNIST,CIFAR100")
    parser.add_argument("--alphas", default="0.1,0.5,1.0,IID")
    parser.add_argument("--num_rounds", type=int, default=100)
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--clients_per_round", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=None)
    parser.add_argument("--verification_rate", type=float, default=1.0)
    parser.add_argument("--femnist_natural", action="store_true", help="Use natural writer FEMNIST partition for all non-IID alpha labels.")
    parser.add_argument("--output_dir", default="experiments/results/reproduction/rq6_noniid")
    args = parser.parse_args()

    if args.local_epochs is not None:
        FL_CONFIG["local_epochs"] = int(args.local_epochs)

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    alphas = [x.strip() for x in args.alphas.split(",") if x.strip()]
    rows: List[Dict] = []

    for dataset in datasets:
        for alpha in alphas:
            no_attack = _run_attack(dataset, alpha, "no_attack", args)
            free_riding = _run_attack(dataset, alpha, "free_riding_no_training", args)
            alie = _run_attack(dataset, alpha, "byzantine_alie", args)
            rows.append({
                "dataset": _paper_dataset(dataset),
                "alpha": alpha,
                "no_attack_ma_pct": _metric_from(no_attack, "ma"),
                "free_riding_dr_pct": _metric_from(free_riding, "dr"),
                "alie_dr_pct": _metric_from(alie, "dr"),
                "fpr_pct": _metric_from(free_riding, "fpr"),
            })

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = out_dir / "table6_noniid_summary.json"
    summary_csv = out_dir / "table6_noniid_summary.csv"
    summary_json.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["dataset", "alpha"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"summary_json": str(summary_json), "summary_csv": str(summary_csv), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

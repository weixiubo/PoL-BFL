#!/usr/bin/env python3
"""Summarize paper-table implementation routes and measurement interfaces.

The report maps each paper target to its runner or measurement pipeline and
emits a machine-readable correspondence manifest.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from validate_reproduction import CODE_ROOT, DEFAULT_OUTPUT_ROOT, load_paper_targets, set_paper_root


SUPPORTED_TABLE1_DATASETS = {"CIFAR-10", "CIFAR-100", "FEMNIST"}
DATASET_NOTES: Dict[str, str] = {}

SUPPORTED_TABLE1_ATTACKS = {
    "Free-riding (NT)",
    "Free-riding (LT)",
    "Byzantine (Random)",
    "Model Replacement",
    "ALIE",
    "MinMax",
    "Data Poisoning",
    "Sybil",
}

SUPPORTED_TABLE1_METHODS = {"Vanilla", "Krum", "ShapleyFL", "FoolsGold", "SDEA", "PoL-BFL"}
METHOD_NOTES: Dict[str, str] = {}

SUPPORTED_TABLE5_AGGREGATIONS = {"Krum", "Trimmed Mean", "Median"}
SUPPORTED_TABLE5_MODES = {"Standalone", "+ PoL-BFL"}
SUPPORTED_TABLE5_ATTACKS = {"ALIE", "Free-riding"}
TABLE5_NOTES = {
}

RUNNER_TABLES = {
    "table1_main_security": "experiments/scripts/runners/run_rq1_security.py",
    "table2_layer_contribution": "experiments/scripts/runners/run_rq2_layer_contribution.py",
    "table4_overhead": "experiments/scripts/runners/run_rq3_overhead.py",
    "table9_adaptive": "experiments/scripts/runners/run_rq9_adaptive.py",
    "table11_zk_details": "experiments/scripts/runners/run_rq3_overhead.py + zkp measurement scripts",
    "table12_gas_breakdown": "chainEnv / smart-contract measurement scripts",
}


def _stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _target_context(item: Dict[str, Any]) -> str:
    keys = ["dataset", "attack", "method", "aggregation", "mode", "metric", "variant", "alpha", "operation"]
    return ", ".join(f"{key}={item[key]}" for key in keys if item.get(key))


def _classify(item: Dict[str, Any]) -> Dict[str, Any]:
    table = item["table"]
    reasons: List[str] = []
    status = "catalogued"
    runner = RUNNER_TABLES.get(table)

    if table == "table1_main_security":
        status = "runnable"
        dataset = item.get("dataset")
        attack = item.get("attack")
        method = item.get("method")
        if dataset not in SUPPORTED_TABLE1_DATASETS:
            status = "blocked_dataset"
            reasons.append(DATASET_NOTES.get(str(dataset), f"Dataset is not supported by RQ1 runner: {dataset}"))
        if attack not in SUPPORTED_TABLE1_ATTACKS:
            status = "blocked_attack"
            reasons.append(f"Attack is not supported by RQ1 runner: {attack}")
        if method not in SUPPORTED_TABLE1_METHODS:
            status = "blocked_method"
            reasons.append(METHOD_NOTES.get(str(method), f"Method is not supported by RQ1 runner: {method}"))
    elif table == "table2_layer_contribution":
        status = "runnable"
        runner = "experiments/scripts/runners/run_rq2_layer_contribution.py"
    elif table == "table5_composability":
        status = "runnable"
        runner = "experiments/scripts/runners/run_rq5_composability.py"
        aggregation = item.get("aggregation")
        mode = item.get("mode")
        attack = item.get("attack")
        if aggregation not in SUPPORTED_TABLE5_AGGREGATIONS:
            status = "blocked_method"
            reasons.append(f"Aggregation is not supported by RQ5 runner: {aggregation}")
        if mode not in SUPPORTED_TABLE5_MODES:
            status = "blocked_design"
            reasons.append(f"Composability mode is not supported: {mode}")
        if attack not in SUPPORTED_TABLE5_ATTACKS:
            status = "blocked_attack"
            reasons.append(TABLE5_NOTES.get(str(attack), f"Attack is not supported by RQ5 runner: {attack}"))
    elif table == "table6_noniid":
        status = "runnable"
        runner = "experiments/scripts/runners/run_rq6_noniid.py"
    elif table in {"table4_overhead", "table11_zk_details", "table12_gas_breakdown"}:
        status = "measurement_pipeline"
        reasons.append("The route uses its measurement pipeline and provenance manifest.")
    elif table == "table9_adaptive":
        status = "runnable"
        runner = "experiments/scripts/runners/run_rq9_adaptive.py"

    return {
        "id": item["id"],
        "table": table,
        "status": status,
        "runner": runner,
        "context": _target_context(item),
        "reasons": reasons,
    }


def _summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"overall": {}, "by_table": {}}
    for rec in records:
        status = rec["status"]
        table = rec["table"]
        summary["overall"][status] = summary["overall"].get(status, 0) + 1
        bucket = summary["by_table"].setdefault(table, {})
        bucket[status] = bucket.get(status, 0) + 1
    summary["overall"]["total"] = len(records)
    for bucket in summary["by_table"].values():
        bucket["total"] = sum(bucket.values())
    return summary


def write_report(path: Path, payload: Dict[str, Any]) -> None:
    summary = payload["summary"]
    records = payload["records"]
    lines = [
        "# PoL-BFL Experiment Route Summary",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Summary",
        "",
        "| Scope | Total | Executable route | Measurement pipeline | Configuration constraints |",
        "|---|---:|---:|---:|---:|",
    ]
    overall = summary["overall"]
    blocked = sum(count for status, count in overall.items() if status.startswith("blocked"))
    lines.append(
        f"| Overall | {overall.get('total', 0)} | {overall.get('runnable', 0)} | {overall.get('measurement_pipeline', 0)} | {blocked} |"
    )
    for table, bucket in sorted(summary["by_table"].items()):
        table_blocked = sum(count for status, count in bucket.items() if status.startswith("blocked"))
        lines.append(
            f"| `{table}` | {bucket.get('total', 0)} | {bucket.get('runnable', 0)} | {bucket.get('measurement_pipeline', 0)} | {table_blocked} |"
        )

    lines.extend(["", "## Configuration constraints", ""])
    blockers = [rec for rec in records if rec["status"].startswith("blocked")]
    if not blockers:
        lines.append("- None.")
    else:
        for rec in blockers[:120]:
            reason = "; ".join(rec["reasons"]) if rec["reasons"] else rec["status"]
            lines.append(f"- `{rec['table']}` {rec['context']}: `{rec['status']}` - {reason}")
        if len(blockers) > 120:
            lines.append(f"- {len(blockers) - 120} additional constrained targets")

    lines.extend(["", "## Route allocation", ""])
    runnable = [rec for rec in records if rec["status"] == "runnable"]
    measurement = [rec for rec in records if rec["status"] == "measurement_pipeline"]
    lines.append(f"- {len(runnable)} target(s) map to executable experiment routes.")
    lines.append(f"- {len(measurement)} target(s) map to dedicated measurement pipelines.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.paper_root:
        set_paper_root(args.paper_root.expanduser().resolve())

    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / f"coverage_{_stamp()}")
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = load_paper_targets()
    records = [_classify(target) for target in targets]
    payload = {
        "schema_version": 1,
        "generated_at": _dt.datetime.now().isoformat(),
        "code_root": str(CODE_ROOT),
        "output_dir": str(output_dir),
        "summary": _summarize(records),
        "records": records,
    }

    manifest_path = output_dir / "coverage_manifest.json"
    report_path = output_dir / "coverage_report.md"
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(report_path, payload)
    print(json.dumps({"manifest": str(manifest_path), "report": str(report_path), "summary": payload["summary"]["overall"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

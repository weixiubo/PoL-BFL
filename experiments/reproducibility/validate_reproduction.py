#!/usr/bin/env python3
"""Validate reproduced PoL-BFL outputs against paper table targets.

This script is intentionally evidence-oriented: it reads paper tables and
experiment result JSON files, normalizes known runner schemas, compares numeric
values with explicit tolerances, and writes a manifest plus a compact report.
It never edits the paper.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


CODE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = CODE_ROOT.parent
DEFAULT_RESULTS_ROOT = CODE_ROOT / "experiments" / "results"
DEFAULT_OUTPUT_ROOT = CODE_ROOT / "experiments" / "results" / "repro_recovery" / "validation"


def _default_paper_root() -> Path:
    env_root = os.getenv("POL_BFL_PAPER_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(
        [
            CODE_ROOT / "experiments" / "reproducibility" / "paper_targets",
            WORKSPACE_ROOT / "Paper_now",
            WORKSPACE_ROOT / "Paper",
            WORKSPACE_ROOT / "Project V7" / "Paper",
            WORKSPACE_ROOT / "Project V7" / "Paper_origin",
            WORKSPACE_ROOT / "Project V7" / "Paper_orig",
        ]
    )
    dated_papers = sorted(
        (p for p in WORKSPACE_ROOT.glob("Paper20*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    candidates.extend(dated_papers)
    for candidate in candidates:
        if (candidate / "tables" / "table1_main_security.tex").exists():
            return candidate
    return candidates[0] if candidates else WORKSPACE_ROOT / "Paper"


def _target_table_files(paper_root: Path) -> Dict[str, Path]:
    return {
        "table1_main_security": paper_root / "tables" / "table1_main_security.tex",
        "table2_layer_contribution": paper_root / "tables" / "table2_layer_contribution.tex",
        "table4_overhead": paper_root / "tables" / "table4_overhead.tex",
        "table5_composability": paper_root / "tables" / "table5_composability.tex",
        "table6_noniid": paper_root / "tables" / "table6_noniid.tex",
        "table9_adaptive": paper_root / "tables" / "table9_adaptive.tex",
        "table11_zk_details": paper_root / "tables" / "table11_zk_details.tex",
        "table12_gas_breakdown": paper_root / "tables" / "table12_gas_breakdown.tex",
    }


PAPER_ROOT = _default_paper_root()
TARGET_TABLE_FILES = _target_table_files(PAPER_ROOT)


def set_paper_root(paper_root: Path) -> None:
    global PAPER_ROOT, TARGET_TABLE_FILES
    PAPER_ROOT = paper_root
    TARGET_TABLE_FILES = _target_table_files(PAPER_ROOT)

RQ1_ATTACK_TO_PAPER = {
    "free_riding_no_training": "Free-riding (NT)",
    "free_riding_lazy_training": "Free-riding (LT)",
    "byzantine_random_noise": "Byzantine (Random)",
    "byzantine_model_replacement": "Model Replacement",
    "byzantine_alie": "ALIE",
    "byzantine_minmax": "MinMax",
    "data_poisoning": "Data Poisoning",
    "sybil": "Sybil",
    "sybil_attack": "Sybil",
}

RQ1_BASELINE_TO_PAPER = {
    "Vanilla_FL": "Vanilla",
    "Krum": "Krum",
    "SDEA": "SDEA",
    "ShapleyFL": "ShapleyFL",
    "FoolsGold": "FoolsGold",
    "PoL_FL": "PoL-BFL",
}

RQ5_ATTACK_TO_PAPER = {
    "byzantine_alie": "ALIE",
    "alie": "ALIE",
    "free_riding_no_training": "Free-riding",
    "free_riding_lazy_training": "Free-riding",
    "free_riding": "Free-riding",
}

RQ5_BASELINE_TO_PAPER = {
    "Krum": ("Krum", "Standalone"),
    "PoL_Krum": ("Krum", "+ PoL-BFL"),
    "Trimmed_Mean": ("Trimmed Mean", "Standalone"),
    "PoL_Trimmed_Mean": ("Trimmed Mean", "+ PoL-BFL"),
    "Median": ("Median", "Standalone"),
    "PoL_Median": ("Median", "+ PoL-BFL"),
}


def _stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _run_capture(cmd: List[str], timeout: int = 20) -> Dict[str, Any]:
    try:
        proc = subprocess.run(cmd, cwd=str(CODE_ROOT), capture_output=True, text=True, timeout=timeout, check=False)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-12000:],
            "stderr": proc.stderr[-8000:],
        }
    except Exception as exc:  # pragma: no cover - defensive reporting path
        return {"cmd": cmd, "returncode": None, "error": repr(exc)}


def _clean_latex(value: str) -> str:
    out = value.strip()
    out = out.replace("\\%", "%")
    out = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", out)
    out = re.sub(r"\\emph\{([^{}]*)\}", r"\1", out)
    out = out.replace("$", "")
    out = out.replace("\\times", "x")
    out = out.replace("\\sim", "~")
    out = out.replace("\\$", "$")
    out = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", "", out)
    out = out.replace("{", "").replace("}", "")
    out = out.strip()
    return out


def _first_number(value: str) -> Optional[float]:
    text = _clean_latex(value).replace(",", "")
    if text in {"", "--"}:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def _as_percent(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return float(value * 100.0 if abs(value) <= 1.5 else value)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _split_table_row(line: str) -> Optional[List[str]]:
    stripped = line.strip()
    if "&" not in stripped or stripped.startswith("%"):
        return None
    stripped = stripped.rstrip("\\").rstrip()
    cells = [_clean_latex(cell) for cell in stripped.split("&")]
    return cells if len(cells) > 1 else None


def _drop_leading_blank(cells: List[str]) -> List[str]:
    return cells[1:] if cells and cells[0] == "" else cells


def _target_id(parts: Dict[str, Any]) -> str:
    keys = ["table", "dataset", "attack", "method", "metric", "variant", "aggregation", "mode", "alpha", "operation"]
    return "|".join(str(parts.get(k, "")) for k in keys)


def _target(table: str, metric: str, value: Optional[float], unit: str = "percent", **dims: Any) -> Dict[str, Any]:
    item = {"table": table, "metric": metric, "target": value, "unit": unit, **dims}
    item["id"] = _target_id(item)
    return item


def _parse_table1() -> List[Dict[str, Any]]:
    text = _read(TARGET_TABLE_FILES["table1_main_security"])
    targets: List[Dict[str, Any]] = []
    current_dataset: Optional[str] = None
    method_layout = [
        ("Vanilla", ["MA"]),
        ("Krum", ["MA", "DR", "FPR"]),
        ("SDEA", ["MA", "DR", "FPR"]),
        ("ShapleyFL", ["MA", "DR", "FPR"]),
        ("FoolsGold", ["MA", "DR", "FPR"]),
        ("PoL-BFL", ["MA", "DR", "FPR"]),
    ]
    for raw in text.splitlines():
        ds_match = re.search(r"\\multirow\{\d+\}\{\*\}\{([^{}]+)\}", raw)
        if ds_match:
            current_dataset = _clean_latex(ds_match.group(1))
        cells = _split_table_row(raw)
        if not cells or not current_dataset or not raw.strip().startswith("&"):
            continue
        cells = _drop_leading_blank(cells)
        attack = cells[0]
        nums = [_first_number(cell) for cell in cells[1:]]
        if len(nums) < 16:
            continue
        idx = 0
        for method, metrics in method_layout:
            for metric in metrics:
                targets.append(_target("table1_main_security", metric, nums[idx], dataset=current_dataset, attack=attack, method=method))
                idx += 1
    return targets


def _parse_table2() -> List[Dict[str, Any]]:
    text = _read(TARGET_TABLE_FILES["table2_layer_contribution"])
    targets: List[Dict[str, Any]] = []
    current_dataset: Optional[str] = None
    layers = ["L1 Only", "L1+L2", "L1+L3", "Full"]
    for raw in text.splitlines():
        ds_match = re.search(r"\\multirow\{\d+\}\{\*\}\{([^{}]+)\}", raw)
        if ds_match:
            current_dataset = _clean_latex(ds_match.group(1))
        cells = _split_table_row(raw)
        if not cells or not current_dataset or not raw.strip().startswith("&"):
            continue
        cells = _drop_leading_blank(cells)
        attack = cells[0]
        for layer, cell in zip(layers, cells[1:5]):
            targets.append(
                _target("table2_layer_contribution", "DR", _first_number(cell), dataset=current_dataset, attack=attack, method=layer)
            )
        targets.append(
            _target(
                "table2_layer_contribution",
                "Dominant",
                None,
                unit="label",
                dataset=current_dataset,
                attack=attack,
                method="Dominant",
                expected_label=cells[5] if len(cells) > 5 else None,
            )
        )
    return targets


def _parse_table4() -> List[Dict[str, Any]]:
    text = _read(TARGET_TABLE_FILES["table4_overhead"])
    targets: List[Dict[str, Any]] = []
    methods = ["Vanilla", "Veriblock-FL", "kaizen", "PoL-BFL"]
    for raw in text.splitlines():
        cells = _split_table_row(raw)
        if not cells or cells[0].startswith("Metric") or len(cells) < 5:
            continue
        metric = cells[0]
        unit = "numeric"
        if "(s)" in metric:
            unit = "seconds"
        elif "MB" in metric:
            unit = "MB"
        elif "USD" in metric:
            unit = "USD"
        for method, cell in zip(methods, cells[1:5]):
            targets.append(_target("table4_overhead", metric, _first_number(cell), unit=unit, method=method))
    return targets


def _parse_table5() -> List[Dict[str, Any]]:
    text = _read(TARGET_TABLE_FILES["table5_composability"])
    targets: List[Dict[str, Any]] = []
    current_agg: Optional[str] = None
    for raw in text.splitlines():
        agg_match = re.search(r"\\multirow\{\d+\}\{\*\}\{([^{}]+)\}", raw)
        if agg_match:
            current_agg = _clean_latex(agg_match.group(1))
        cells = _split_table_row(raw)
        if not cells or not current_agg or not raw.strip().startswith("&"):
            continue
        cells = _drop_leading_blank(cells)
        attack = cells[0]
        nums = [_first_number(cell) for cell in cells[1:]]
        if len(nums) < 6:
            continue
        for mode, values in (("Standalone", nums[:3]), ("+ PoL-BFL", nums[3:6])):
            for metric, value in zip(["MA", "DR", "FPR"], values):
                targets.append(
                    _target(
                        "table5_composability",
                        metric,
                        value,
                        dataset="CIFAR-10",
                        attack=attack,
                        aggregation=current_agg,
                        mode=mode,
                    )
                )
    return targets


def _parse_table6() -> List[Dict[str, Any]]:
    text = _read(TARGET_TABLE_FILES["table6_noniid"])
    targets: List[Dict[str, Any]] = []
    current_dataset: Optional[str] = None
    alphas = ["0.1", "0.5", "1.0", "IID"]
    for raw in text.splitlines():
        ds_match = re.search(r"\\multirow\{\d+\}\{\*\}\{([^{}]+)\}", raw)
        if ds_match:
            current_dataset = _clean_latex(ds_match.group(1))
        cells = _split_table_row(raw)
        if not cells or not current_dataset or not raw.strip().startswith("&"):
            continue
        cells = _drop_leading_blank(cells)
        metric = cells[0]
        for alpha, cell in zip(alphas, cells[1:5]):
            targets.append(_target("table6_noniid", metric, _first_number(cell), dataset=current_dataset, alpha=alpha))
    return targets


def _parse_table9() -> List[Dict[str, Any]]:
    text = _read(TARGET_TABLE_FILES["table9_adaptive"])
    targets: List[Dict[str, Any]] = []
    for raw in text.splitlines():
        cells = _split_table_row(raw)
        if not cells or len(cells) < 5 or cells[0].startswith("Attack Variant"):
            continue
        variant = cells[0]
        targets.append(_target("table9_adaptive", "DR", _first_number(cells[1]), variant=variant))
        targets.append(_target("table9_adaptive", "FPR", _first_number(cells[2]), variant=variant))
        targets.append(_target("table9_adaptive", "Forge/Train", _first_number(cells[3]), unit="ratio", variant=variant))
        targets.append(
            _target("table9_adaptive", "Profitable", None, unit="label", variant=variant, expected_label=cells[4])
        )
    return targets


def _parse_table11() -> List[Dict[str, Any]]:
    text = _read(TARGET_TABLE_FILES["table11_zk_details"])
    targets: List[Dict[str, Any]] = []
    for raw in text.splitlines():
        cells = _split_table_row(raw)
        if not cells or len(cells) < 4 or cells[0].startswith("Metric"):
            continue
        metric = cells[0]
        for method, cell in zip(["PoL-BFL", "kaizen"], cells[1:3]):
            targets.append(_target("table11_zk_details", metric, _first_number(cell), unit="mixed", method=method))
        targets.append(_target("table11_zk_details", metric, _first_number(cells[3]), unit="ratio", method="Ratio"))
    return targets


def _parse_table12() -> List[Dict[str, Any]]:
    text = _read(TARGET_TABLE_FILES["table12_gas_breakdown"])
    targets: List[Dict[str, Any]] = []
    for raw in text.splitlines():
        cells = _split_table_row(raw)
        if not cells or len(cells) < 3 or cells[0].startswith("Operation"):
            continue
        operation = cells[0]
        targets.append(_target("table12_gas_breakdown", "Gas", _first_number(cells[1]), unit="gas", operation=operation))
        targets.append(_target("table12_gas_breakdown", "Cost (USD)", _first_number(cells[2]), unit="USD", operation=operation))
    return targets


def load_paper_targets() -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    for parser in [
        _parse_table1,
        _parse_table2,
        _parse_table4,
        _parse_table5,
        _parse_table6,
        _parse_table9,
        _parse_table11,
        _parse_table12,
    ]:
        targets.extend(parser())
    return targets


def _json_files(paths: Iterable[Path], pattern: str) -> List[Path]:
    out: List[Path] = []
    for root in paths:
        if root.is_file() and root.name == pattern:
            if not _is_archived_result_path(root):
                out.append(root)
        elif root.exists():
            out.extend(path for path in sorted(root.rglob(pattern)) if not _is_archived_result_path(path))
    return sorted(set(path.resolve() for path in out))


def _is_archived_result_path(path: Path) -> bool:
    archived_markers = (
        "__aborted_",
        "__archived_",
        "__failed_",
        "_archive",
        "archive",
        "diagnostics",
        "_diagnostics",
    )
    return any(any(marker in part for marker in archived_markers) for part in path.parts)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _find_ancestor_file(path: Path, filename: str, max_levels: int = 5) -> Optional[Path]:
    current = path.resolve()
    for parent in [current.parent, *list(current.parents)[:max_levels]]:
        candidate = parent / filename
        if candidate.exists():
            return candidate
    return None


def _source_protocol(path: Path) -> Dict[str, Any]:
    """Load run-level protocol metadata next to a result file.

    Smoke outputs and formal outputs may share the same result schema. Keeping
    protocol metadata on each observation prevents a 1-round smoke from being
    treated as paper reproduction evidence.
    """
    protocol: Dict[str, Any] = {
        "result_file": str(path),
        "run_manifest": None,
        "config_json": None,
        "run_id": None,
        "dry_run": None,
        "rounds": None,
        "num_clients": None,
        "clients_per_round": None,
        "data_distribution": None,
        "local_epochs": None,
        "attack": None,
        "attack_params": {},
        "attack_profile": None,
    }
    run_manifest = _find_ancestor_file(path, "run_manifest.json")
    if run_manifest:
        protocol["run_manifest"] = str(run_manifest)
        try:
            manifest = _load_json(run_manifest)
            protocol["run_id"] = manifest.get("run_id")
            protocol["dry_run"] = manifest.get("dry_run")
            cfg = manifest.get("config") or {}
            protocol["rounds"] = cfg.get("rounds", protocol["rounds"])
            protocol["num_clients"] = cfg.get("num_clients", protocol["num_clients"])
            protocol["clients_per_round"] = cfg.get("clients_per_round", protocol["clients_per_round"])
            protocol["local_epochs"] = cfg.get("local_epochs", protocol["local_epochs"])
            protocol["attack"] = cfg.get("attack", protocol["attack"])
            protocol["attack_params"] = cfg.get("attack_params", protocol["attack_params"]) or {}
            protocol["attack_profile"] = cfg.get("attack_profile", protocol["attack_profile"])
        except Exception:
            pass

    config_path = path.parent / "config.json"
    if config_path.exists():
        protocol["config_json"] = str(config_path)
        try:
            cfg = _load_json(config_path)
            protocol["rounds"] = cfg.get("num_rounds", protocol["rounds"])
            protocol["num_clients"] = cfg.get("num_clients", protocol["num_clients"])
            protocol["clients_per_round"] = cfg.get("clients_per_round", protocol["clients_per_round"])
            protocol["data_distribution"] = cfg.get("data_distribution", protocol["data_distribution"])
            protocol["local_epochs"] = cfg.get("local_epochs", protocol["local_epochs"])
            protocol["malicious_ratios"] = sorted(
                {
                    float(params.get("malicious_ratios", [0.0])[0])
                    for params in (cfg.get("attacks") or {}).values()
                    if isinstance(params, dict) and params.get("malicious_ratios")
                }
            )
            attacks_cfg = cfg.get("attacks") or {}
            if isinstance(attacks_cfg, dict) and len(attacks_cfg) == 1:
                attack_name, attack_params = next(iter(attacks_cfg.items()))
                protocol["attack"] = protocol.get("attack") or attack_name
                if isinstance(attack_params, dict) and not protocol.get("attack_params"):
                    protocol["attack_params"] = dict(attack_params)
                profile = cfg.get("attack_profile")
                if not profile and isinstance(cfg.get("attack_profiles"), dict):
                    profile = cfg["attack_profiles"].get(attack_name)
                protocol["attack_profile"] = protocol.get("attack_profile") or profile
        except Exception:
            pass

    result_parts = set(path.resolve().parts)
    run_id = str(protocol.get("run_id") or "")
    protocol["scope"] = "smoke" if "smoke" in result_parts or "_1r_" in run_id or run_id.startswith("local_") else "unknown"
    return protocol


def _dataset_from_config(path: Path, default: Optional[str] = None) -> Optional[str]:
    config_path = path.parent / "config.json"
    if config_path.exists():
        try:
            config = _load_json(config_path)
            dataset = config.get("dataset") or default
            return {"CIFAR10": "CIFAR-10", "CIFAR100": "CIFAR-100"}.get(str(dataset), str(dataset))
        except Exception:
            return default
    return default


def _observed_id(parts: Dict[str, Any]) -> str:
    return _target_id(parts)


def _observed(table: str, metric: str, value: Optional[float], source: Path, **dims: Any) -> Dict[str, Any]:
    item = {
        "table": table,
        "metric": metric,
        "observed": value,
        "source": str(source),
        "protocol": _source_protocol(source),
        **dims,
    }
    item["id"] = _observed_id(item)
    return item


def _attach_rq1_result_protocol(observation: Dict[str, Any], result: Dict[str, Any], raw_attack: str) -> Dict[str, Any]:
    protocol = observation.setdefault("protocol", {})
    protocol["attack"] = protocol.get("attack") or raw_attack
    attack_params = result.get("attack_params")
    if isinstance(attack_params, dict):
        protocol["attack_params"] = dict(attack_params)
    attack_effect = result.get("attack_effect")
    if isinstance(attack_effect, dict):
        protocol["attack_effect"] = dict(attack_effect)
    elif isinstance(result.get("rounds"), list):
        attacked_values = []
        attacked_maxes = []
        for row in result.get("rounds") or []:
            if not isinstance(row, dict):
                continue
            try:
                has_attacker = int(row.get("num_malicious_in_round", 0) or 0) > 0
                l2_mean = float(row.get("attack_l2_mean", 0.0) or 0.0)
                l2_max = float(row.get("attack_l2_max", 0.0) or 0.0)
            except Exception:
                continue
            if has_attacker:
                attacked_values.append(l2_mean)
                attacked_maxes.append(l2_max)
        if attacked_values:
            protocol["attack_effect"] = {
                "rounds_with_malicious_clients": len(attacked_values),
                "attack_l2_mean_over_attacked_rounds": sum(attacked_values) / len(attacked_values),
                "attack_l2_max_over_attacked_rounds": max(attacked_maxes) if attacked_maxes else 0.0,
            }
    profile = result.get("attack_profile")
    if profile and not protocol.get("attack_profile"):
        protocol["attack_profile"] = profile
    return observation


def load_rq1_observations(paths: List[Path]) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    for path in paths:
        try:
            payload = _load_json(path)
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        dataset = _dataset_from_config(path, default=None)
        for result in payload:
            if not isinstance(result, dict):
                continue
            raw_attack = str(result.get("attack_type"))
            raw_method = str(result.get("baseline_method"))
            attack = RQ1_ATTACK_TO_PAPER.get(raw_attack, raw_attack)
            method = RQ1_BASELINE_TO_PAPER.get(raw_method, raw_method)
            if not attack or not method or not dataset:
                continue
            observations.append(
                _attach_rq1_result_protocol(
                    _observed(
                        "table1_main_security",
                        "MA",
                        _as_percent(result.get("final_accuracy")),
                        path,
                        dataset=dataset,
                        attack=attack,
                        method=method,
                    ),
                    result,
                    raw_attack,
                )
            )
            metrics = result.get("detection_metrics") or {}
            if method != "Vanilla" and isinstance(metrics, dict):
                observations.append(
                    _attach_rq1_result_protocol(
                        _observed(
                            "table1_main_security",
                            "DR",
                            _as_percent(metrics.get("TPR")),
                            path,
                            dataset=dataset,
                            attack=attack,
                            method=method,
                        ),
                        result,
                        raw_attack,
                    )
                )
                observations.append(
                    _attach_rq1_result_protocol(
                        _observed(
                            "table1_main_security",
                            "FPR",
                            _as_percent(metrics.get("FPR")),
                            path,
                            dataset=dataset,
                            attack=attack,
                            method=method,
                        ),
                        result,
                        raw_attack,
                    )
                )
    return observations


def load_rq5_observations(paths: List[Path]) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    for path in paths:
        try:
            payload = _load_json(path)
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for result in payload:
            if not isinstance(result, dict):
                continue
            raw_attack = str(result.get("attack_type"))
            raw_method = str(result.get("baseline_method"))
            attack = RQ5_ATTACK_TO_PAPER.get(raw_attack, raw_attack)
            mapped_baseline = RQ5_BASELINE_TO_PAPER.get(raw_method)
            if not attack or not mapped_baseline:
                continue
            aggregation, mode = mapped_baseline
            observations.append(
                _observed(
                    "table5_composability",
                    "MA",
                    _as_percent(result.get("final_accuracy")),
                    path,
                    dataset="CIFAR-10",
                    attack=attack,
                    aggregation=aggregation,
                    mode=mode,
                )
            )
            metrics = result.get("detection_metrics") or {}
            if isinstance(metrics, dict):
                observations.append(
                    _observed(
                        "table5_composability",
                        "DR",
                        _as_percent(metrics.get("TPR")),
                        path,
                        dataset="CIFAR-10",
                        attack=attack,
                        aggregation=aggregation,
                        mode=mode,
                    )
                )
                observations.append(
                    _observed(
                        "table5_composability",
                        "FPR",
                        _as_percent(metrics.get("FPR")),
                        path,
                        dataset="CIFAR-10",
                        attack=attack,
                        aggregation=aggregation,
                        mode=mode,
                    )
                )
    return observations


def load_table2_observations(paths: List[Path]) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    for path in paths:
        try:
            payload = _load_json(path)
        except Exception:
            continue
        rows = payload.get("rows", []) if isinstance(payload, dict) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            observations.append(
                _observed(
                    "table2_layer_contribution",
                    "DR",
                    row.get("dr_pct"),
                    path,
                    dataset=row.get("dataset"),
                    attack=row.get("attack"),
                    method=row.get("paper_label"),
                )
            )
    return observations


def load_table6_observations(paths: List[Path]) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    metric_map = {
        "No Attack MA (%)": "no_attack_ma_pct",
        "Free-riding DR (%)": "free_riding_dr_pct",
        "ALIE DR (%)": "alie_dr_pct",
        "FPR (%)": "fpr_pct",
    }
    for path in paths:
        try:
            payload = _load_json(path)
        except Exception:
            continue
        rows = payload.get("rows", []) if isinstance(payload, dict) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            for metric, field in metric_map.items():
                observations.append(
                    _observed(
                        "table6_noniid",
                        metric,
                        row.get(field),
                        path,
                        dataset=row.get("dataset"),
                        alpha=str(row.get("alpha")),
                    )
                )
    return observations


def load_table9_observations(paths: List[Path]) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    for path in paths:
        try:
            payload = _load_json(path)
        except Exception:
            continue
        rows = payload.get("rows", []) if isinstance(payload, dict) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            variant = row.get("paper_label")
            observations.extend([
                _observed("table9_adaptive", "DR", row.get("dr_pct"), path, variant=variant),
                _observed("table9_adaptive", "FPR", row.get("fpr_pct"), path, variant=variant),
                _observed("table9_adaptive", "Forge/Train", row.get("forge_train_ratio"), path, variant=variant),
                _observed("table9_adaptive", "Profitable", row.get("profitable"), path, variant=variant),
            ])
    return observations


def load_observations(results_roots: List[Path], rq1_json: List[Path], rq5_json: List[Path]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rq1_files = [path.resolve() for path in rq1_json] + _json_files(results_roots, "rq1_results.json")
    rq5_files = [path.resolve() for path in rq5_json] + _json_files(results_roots, "rq5_results.json")
    table2_files = _json_files(results_roots, "table2_layer_contribution_summary.json")
    table6_files = _json_files(results_roots, "table6_noniid_summary.json")
    table9_files = _json_files(results_roots, "table9_adaptive_summary.json")
    rq1_files = sorted(set(path for path in rq1_files if path.exists()))
    rq5_files = sorted(set(path for path in rq5_files if path.exists()))
    observations = (
        load_rq1_observations(rq1_files)
        + load_rq5_observations(rq5_files)
        + load_table2_observations(table2_files)
        + load_table6_observations(table6_files)
        + load_table9_observations(table9_files)
    )
    return observations, {
        "rq1_files": [str(path) for path in rq1_files],
        "rq5_files": [str(path) for path in rq5_files],
        "table2_files": [str(path) for path in table2_files],
        "table6_files": [str(path) for path in table6_files],
        "table9_files": [str(path) for path in table9_files],
    }


def _tolerance_for(target: Dict[str, Any], args: argparse.Namespace) -> float:
    metric = str(target.get("metric", ""))
    if metric == "MA":
        return float(args.tolerance_ma)
    if metric in {"DR", "FPR"}:
        return float(args.tolerance_detection)
    return float(args.tolerance_other)


def _passes_numeric_target(metric: str, observed: float, target: float, tolerance: float) -> bool:
    """Paper security metrics are directional, not absolute-error targets."""
    metric_l = str(metric).lower()
    if metric_l == "ma" or " ma " in f" {metric_l} " or metric_l.endswith("ma (%)"):
        return observed + tolerance >= target
    if metric_l == "dr" or " dr " in f" {metric_l} " or metric_l.endswith("dr (%)"):
        return observed + tolerance >= target
    if metric_l == "fpr" or "fpr" in metric_l:
        return observed <= target + tolerance
    return abs(observed - target) <= tolerance


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _protocol_mismatches(target: Dict[str, Any], obs: Dict[str, Any], args: argparse.Namespace) -> List[str]:
    if args.no_enforce_protocol:
        return []

    table = str(target.get("table", ""))
    protocol = obs.get("protocol") or {}
    mismatches: List[str] = []

    if protocol.get("dry_run") is True:
        mismatches.append("dry-run output cannot validate paper values")

    if table in {"table1_main_security", "table5_composability"}:
        rounds = _as_int(protocol.get("rounds"))
        num_clients = _as_int(protocol.get("num_clients"))
        clients_per_round = _as_int(protocol.get("clients_per_round"))
        local_epochs = _as_int(protocol.get("local_epochs"))

        if rounds is None or rounds < args.min_rounds_rq1:
            mismatches.append(f"rounds={rounds} < paper protocol {args.min_rounds_rq1}")
        if num_clients is None or num_clients < args.min_clients_rq1:
            mismatches.append(f"num_clients={num_clients} < paper protocol {args.min_clients_rq1}")
        if args.min_clients_per_round_rq1 > 0 and (clients_per_round is None or clients_per_round < args.min_clients_per_round_rq1):
            mismatches.append(f"clients_per_round={clients_per_round} < paper protocol {args.min_clients_per_round_rq1}")
        if local_epochs is None or local_epochs < args.min_local_epochs_rq1:
            mismatches.append(f"local_epochs={local_epochs} < paper protocol {args.min_local_epochs_rq1}")

        ratios = protocol.get("malicious_ratios")
        if ratios and 0.2 not in [round(float(r), 6) for r in ratios]:
            mismatches.append(f"malicious ratio {ratios} does not include paper 0.2")

        if table == "table1_main_security" and not args.allow_table1_noniid:
            dist = protocol.get("data_distribution")
            dataset = str(target.get("dataset", ""))
            allowed_dist = {"iid", "iid_partition", "iid-partition"}
            if dataset == "FEMNIST":
                allowed_dist.update({"natural_writer", "femnist_natural", "leaf_natural"})
            if dist and str(dist).lower() not in allowed_dist:
                mismatches.append(f"data_distribution={dist} but paper says IID unless noted")

        if table == "table1_main_security" and str(target.get("attack")) == "Model Replacement":
            profile = str(protocol.get("attack_profile") or "").strip().lower()
            if profile not in {"paper", "paper_table1", "paper_reproduction"}:
                mismatches.append(
                    f"model replacement attack_profile={profile or None} is not a paper reproduction profile"
                )
            params = protocol.get("attack_params") or {}
            mix = params.get("replacement_mix") if isinstance(params, dict) else None
            if mix is None:
                mismatches.append("model replacement replacement_mix missing from protocol metadata")
            else:
                try:
                    mix_value = float(mix)
                except Exception:
                    mismatches.append(f"model replacement replacement_mix={mix!r} is not numeric")
                else:
                    if not (0.0 < mix_value < 1.0):
                        mismatches.append(
                            f"model replacement replacement_mix={mix_value} is a stress/diagnostic setting, not paper-calibrated"
                        )

        if table == "table1_main_security" and str(target.get("attack")) == "Free-riding (NT)":
            params = protocol.get("attack_params") or {}
            mode = str(params.get("submission_mode") or params.get("mode") or "").strip().lower()
            if mode not in {"random_update", "random", "random_model"}:
                mismatches.append(
                    f"free-riding NT submission_mode={mode or None} is not the paper random-update threat model"
                )
            if mode in {"random_update", "random", "random_model"}:
                try:
                    noise_scale = float(params.get("noise_scale", 0.0))
                except Exception:
                    noise_scale = 0.0
                if noise_scale <= 0.0:
                    mismatches.append("free-riding NT noise_scale must be > 0 for paper random-update runs")
            effect = protocol.get("attack_effect") or {}
            try:
                l2_mean = float(effect.get("attack_l2_mean_over_attacked_rounds", 0.0))
                l2_max = float(effect.get("attack_l2_max_over_attacked_rounds", 0.0))
            except Exception:
                l2_mean = 0.0
                l2_max = 0.0
            if l2_mean <= 0.0 or l2_max <= 0.0:
                mismatches.append(
                    "free-riding NT attack_l2 is zero/missing; attack did not measurably change malicious submissions"
                )

    return mismatches


def _choose_observation(target: Dict[str, Any], candidates: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    if not candidates:
        return None, []
    scored = [(obs, _protocol_mismatches(target, obs, args)) for obs in candidates]
    for obs, mismatches in scored:
        if not mismatches:
            return obs, mismatches
    return scored[0]


def compare_targets(targets: List[Dict[str, Any]], observations: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    observed_by_id: Dict[str, List[Dict[str, Any]]] = {}
    for obs in observations:
        if obs.get("observed") is not None:
            observed_by_id.setdefault(obs["id"], []).append(obs)
    comparisons: List[Dict[str, Any]] = []
    for target in targets:
        candidates = observed_by_id.get(target["id"], [])
        obs, mismatches = _choose_observation(target, candidates, args)
        if target.get("unit") == "label":
            if not obs:
                comparisons.append({**target, "status": "missing", "observed": None, "delta": None, "source": None})
                continue
            observed_label = str(obs.get("observed"))
            expected_label = str(target.get("expected_label"))
            comparisons.append({
                **target,
                "status": "pass" if observed_label == expected_label else "fail",
                "observed": observed_label,
                "delta": None,
                "source": obs.get("source"),
                "expected_label": expected_label,
            })
            continue
        if target.get("target") is None:
            comparisons.append({**target, "status": "no_numeric_target", "observed": None, "delta": None, "source": None})
            continue
        if not obs:
            comparisons.append({**target, "status": "missing", "observed": None, "delta": None, "source": None})
            continue
        observed_value = float(obs["observed"])
        target_value = float(target["target"])
        delta = observed_value - target_value
        tol = _tolerance_for(target, args)
        status = "protocol_mismatch" if mismatches else (
            "pass" if _passes_numeric_target(str(target.get("metric", "")), observed_value, target_value, tol) else "fail"
        )
        comparisons.append(
            {
                **target,
                "status": status,
                "observed": observed_value,
                "delta": delta,
                "tolerance": tol,
                "source": obs.get("source"),
                "protocol": obs.get("protocol"),
                "protocol_mismatches": mismatches,
                "candidate_observations": len(candidates),
            }
        )
    return comparisons


def _summarize(comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"overall": {}, "by_table": {}}
    for item in comparisons:
        status = item["status"]
        table = item["table"]
        summary["overall"][status] = summary["overall"].get(status, 0) + 1
        table_bucket = summary["by_table"].setdefault(table, {})
        table_bucket[status] = table_bucket.get(status, 0) + 1
    summary["overall"]["total"] = len(comparisons)
    for table_bucket in summary["by_table"].values():
        table_bucket["total"] = sum(table_bucket.values())
    return summary


def unmatched_observations(targets: List[Dict[str, Any]], observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    target_ids = {target["id"] for target in targets}
    return [obs for obs in observations if obs.get("id") not in target_ids]


def _format_dims(item: Dict[str, Any]) -> str:
    parts = []
    for key in ["dataset", "attack", "method", "aggregation", "mode", "variant", "alpha", "operation"]:
        if item.get(key):
            parts.append(f"{key}={item[key]}")
    return ", ".join(parts)


def write_report(path: Path, manifest: Dict[str, Any]) -> None:
    summary = manifest["summary"]
    comparisons = manifest["comparisons"]
    lines = [
        "# PoL-BFL Reproduction Validation Report",
        "",
        f"Generated: {manifest['generated_at']}",
        "",
        "## Scope",
        "",
        "- Paper tables are read only.",
        "- Results are compared only when a normalized experiment output exists.",
        "- Smoke runs and protocol-mismatched runs are reported separately from pass/fail.",
        "- Missing means no current result file can substantiate that paper target yet.",
        "",
        "## Summary",
        "",
        "| Scope | Total | Pass | Fail | Protocol Mismatch | Missing | Other |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    overall = summary["overall"]
    other = sum(count for status, count in overall.items() if status not in {"total", "pass", "fail", "protocol_mismatch", "missing"})
    lines.append(
        f"| Overall | {overall.get('total', 0)} | {overall.get('pass', 0)} | {overall.get('fail', 0)} | "
        f"{overall.get('protocol_mismatch', 0)} | {overall.get('missing', 0)} | {other} |"
    )
    for table, bucket in sorted(summary["by_table"].items()):
        table_other = sum(count for status, count in bucket.items() if status not in {"total", "pass", "fail", "protocol_mismatch", "missing"})
        lines.append(
            f"| `{table}` | {bucket.get('total', 0)} | {bucket.get('pass', 0)} | {bucket.get('fail', 0)} | "
            f"{bucket.get('protocol_mismatch', 0)} | {bucket.get('missing', 0)} | {table_other} |"
        )

    failures = [item for item in comparisons if item["status"] == "fail"]
    protocol_mismatches = [item for item in comparisons if item["status"] == "protocol_mismatch"]
    missing = [item for item in comparisons if item["status"] == "missing"]
    lines.extend(["", "## Largest Deviations", ""])
    if not failures:
        lines.append("- None among available observations.")
    else:
        for item in sorted(failures, key=lambda x: abs(float(x.get("delta") or 0.0)), reverse=True)[:30]:
            lines.append(
                f"- `{item['table']}` {_format_dims(item)} {item['metric']}: observed {item['observed']:.3f}, "
                f"target {item['target']:.3f}, delta {item['delta']:+.3f}, tol {item['tolerance']:.3f}"
            )

    lines.extend(["", "## Protocol Mismatches", ""])
    if not protocol_mismatches:
        lines.append("- None among available observations.")
    else:
        for item in protocol_mismatches[:40]:
            reasons = "; ".join(item.get("protocol_mismatches") or [])
            lines.append(
                f"- `{item['table']}` {_format_dims(item)} {item['metric']}: observed {item['observed']:.3f}, "
                f"target {item['target']:.3f}; {reasons}"
            )
        if len(protocol_mismatches) > 40:
            lines.append(f"- ... {len(protocol_mismatches) - 40} more protocol-mismatched observations")

    lines.extend(["", "## Missing Evidence", ""])
    if not missing:
        lines.append("- No missing numeric targets.")
    else:
        for item in missing[:80]:
            lines.append(f"- `{item['table']}` {_format_dims(item)} {item['metric']}: target {item['target']}")
        if len(missing) > 80:
            lines.append(f"- ... {len(missing) - 80} more missing targets")

    unmatched = manifest.get("unmatched_observations", [])
    lines.extend(["", "## Observations Outside Paper Targets", ""])
    if not unmatched:
        lines.append("- None.")
    else:
        lines.append(f"- {len(unmatched)} observation(s) were produced but do not map to a target paper-table cell.")
        for item in unmatched[:40]:
            value = item.get("observed")
            value_text = "None" if value is None else f"{float(value):.3f}"
            lines.append(f"  - `{item['table']}` {_format_dims(item)} {item['metric']}: observed {value_text}")
        if len(unmatched) > 40:
            lines.append(f"  - ... {len(unmatched) - 40} more")

    lines.extend(["", "## Inputs", ""])
    for label, paths in manifest["inputs"].items():
        lines.append(f"- {label}: {len(paths)} file(s)")
        for source in paths[:12]:
            lines.append(f"  - `{source}`")
        if len(paths) > 12:
            lines.append(f"  - ... {len(paths) - 12} more")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", action="append", type=Path, default=[])
    parser.add_argument("--rq1-json", action="append", type=Path, default=[])
    parser.add_argument("--rq5-json", action="append", type=Path, default=[])
    parser.add_argument("--paper-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tolerance-ma", type=float, default=1.0, help="MA tolerance in percentage points")
    parser.add_argument("--tolerance-detection", type=float, default=1.0, help="DR/FPR tolerance in percentage points")
    parser.add_argument("--tolerance-other", type=float, default=5.0, help="Generic numeric tolerance")
    parser.add_argument("--no-enforce-protocol", action="store_true", help="Compare numeric values even when run protocol does not match paper")
    parser.add_argument("--min-rounds-rq1", type=int, default=200, help="Paper protocol minimum rounds for RQ1/RQ5 numeric validation")
    parser.add_argument("--min-clients-rq1", type=int, default=50, help="Paper protocol minimum registered clients for RQ1/RQ5 numeric validation")
    parser.add_argument("--min-clients-per-round-rq1", type=int, default=0, help="Optional minimum participating clients per round for RQ1/RQ5; 0 means not enforced")
    parser.add_argument("--min-local-epochs-rq1", type=int, default=5, help="Paper protocol minimum local epochs for RQ1/RQ5 numeric validation")
    parser.add_argument("--allow-table1-noniid", action="store_true", help="Do not reject Table 1 observations with non-IID partition metadata")
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / f"validation_{_stamp()}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.paper_root:
        set_paper_root(args.paper_root.expanduser().resolve())

    results_roots = [path.resolve() for path in args.results_root] or [DEFAULT_RESULTS_ROOT.resolve()]
    targets = load_paper_targets()
    observations, inputs = load_observations(results_roots, args.rq1_json, args.rq5_json)
    comparisons = compare_targets(targets, observations, args)
    unmatched = unmatched_observations(targets, observations)
    summary = _summarize(comparisons)
    summary["unmatched_observations"] = len(unmatched)

    manifest = {
        "schema_version": 1,
        "generated_at": _dt.datetime.now().isoformat(),
        "code_root": str(CODE_ROOT),
        "paper_root": str(PAPER_ROOT),
        "output_dir": str(output_dir),
        "inputs": inputs,
        "results_roots": [str(path) for path in results_roots],
        "tolerances": {
            "MA": args.tolerance_ma,
            "DR_FPR": args.tolerance_detection,
            "other": args.tolerance_other,
        },
        "protocol_requirements": {
            "enforced": not args.no_enforce_protocol,
            "table1_table5": {
                "min_rounds": args.min_rounds_rq1,
                "min_clients": args.min_clients_rq1,
                "min_clients_per_round": args.min_clients_per_round_rq1,
                "min_local_epochs": args.min_local_epochs_rq1,
                "require_iid_for_table1": not args.allow_table1_noniid,
            },
        },
        "summary": summary,
        "targets": targets,
        "observations": observations,
        "unmatched_observations": unmatched,
        "comparisons": comparisons,
        "environment_snapshot": {
            "python": _run_capture(["python", "-V"]),
            "git_status": _run_capture(["git", "status", "--short", "--branch"]),
            "git_diff_stat": _run_capture(["git", "diff", "--stat"]),
        },
    }

    manifest_path = output_dir / "validation_manifest.json"
    report_path = output_dir / "validation_report.md"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(report_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "report": str(report_path), "summary": summary["overall"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

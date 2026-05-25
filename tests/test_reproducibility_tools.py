import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_run_paper_config_pol_formal_env_overrides():
    from argparse import Namespace

    from experiments.reproducibility.run_paper_config import _build_rq1_jobs, _load_config

    config_path = ROOT / "experiments" / "reproducibility" / "configs" / "paper" / "rq1_main_security_formal.json"
    config = _load_config(config_path)
    jobs = _build_rq1_jobs(config, config_path, Namespace(python=sys.executable))

    pol_job = next(
        job
        for job in jobs
        if job.config["dataset"] == "CIFAR10"
        and job.config["attack"] == "free_riding_no_training"
        and job.config["baseline"] == "PoL_FL"
    )

    assert pol_job.env["POL_SAVE_CHECKPOINTS_TO_DISK"] == "0"
    assert pol_job.env["POL_SAVE_FREQ"] == "20"
    assert pol_job.env["POL_MEMORY_CHECKPOINT_LIMIT"] == "2"
    assert pol_job.env["POL_CHALLENGE_SELECTED_PAIRS"] == "1"
    assert pol_job.env["POL_ALWAYS_VERIFY_LAST_K"] == "1"
    assert pol_job.env["POL_RANDOM_Q"] == "0"
    assert pol_job.env["POL_COMPACT_REMOTE_RESPONSE"] == "1"
    assert pol_job.env["POL_ENABLE_PARALLEL_CLIENT_TRAINING"] == "1"
    assert pol_job.env["POL_CLIENT_TRAIN_WORKERS_PER_DEVICE"] == "2"
    assert pol_job.env["POL_SUPPRESS_MODEL_INFO"] == "1"
    assert pol_job.env["NUM_WORKERS_OVERRIDE"] == "0"

    sybil_job = next(
        job
        for job in jobs
        if job.config["dataset"] == "CIFAR10"
        and job.config["attack"] == "sybil"
        and job.config["baseline"] == "PoL_FL"
    )
    assert sybil_job.env["POL_ENABLE_SYBIL_DETECTOR"] == "1"
    assert sybil_job.env["POL_SYBIL_TRAJECTORY_ONLY"] == "0"


def test_run_paper_config_rq1_training_hyperparams_are_explicit():
    from argparse import Namespace

    from experiments.reproducibility.run_paper_config import _build_rq1_jobs

    config_path = ROOT / "experiments" / "reproducibility" / "configs" / "paper" / "rq1_main_security_formal.json"
    config = {
        "runner": "experiments/scripts/runners/run_rq1_security.py",
        "output_root": "experiments/results/reproduction/formal/unit_hparams",
        "protocol": {"pol_integrity": 1},
        "execution": {
            "rounds": 3,
            "num_clients": 5,
            "clients_per_round": 2,
            "local_epochs": 1,
            "batch_size": 128,
            "learning_rate": 0.05,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "verification_rate": 0.2,
            "seeds": [42],
            "data_distribution": "IID",
        },
        "datasets": [{"name": "CIFAR10", "model": "ResNet18"}],
        "attacks": ["free_riding_no_training"],
        "baselines": ["PoL_FL"],
    }
    jobs = _build_rq1_jobs(config, config_path, Namespace(python=sys.executable))
    job = jobs[0]

    assert "--batch_size" in job.command
    assert job.command[job.command.index("--batch_size") + 1] == "128"
    assert job.command[job.command.index("--learning_rate") + 1] == "0.05"
    assert job.command[job.command.index("--weight_decay") + 1] == "0.0005"
    assert job.command[job.command.index("--verification_rate") + 1] == "0.2"
    assert job.config["batch_size"] == 128
    assert job.config["learning_rate"] == 0.05
    assert job.config["verification_rate"] == 0.2


def test_run_paper_config_only_filter_does_not_select_cifar100():
    from argparse import Namespace

    from experiments.reproducibility.run_paper_config import _build_rq1_jobs, _filter_jobs, _load_config

    config_path = ROOT / "experiments" / "reproducibility" / "configs" / "paper" / "rq1_main_security_formal.json"
    config = _load_config(config_path)
    jobs = _build_rq1_jobs(config, config_path, Namespace(python=sys.executable))
    selected = _filter_jobs(
        jobs,
        Namespace(only=["cifar10", "pol_bfl"], resume=False, limit=None),
    )

    assert selected
    assert all(job.config["dataset"] == "CIFAR10" for job in selected)
    assert all(job.config["baseline"] == "PoL_FL" for job in selected)


def test_rq1_sybil_uses_fixed_run_anchor_not_per_round_subset():
    runner = ROOT / "experiments" / "scripts" / "runners" / "run_rq1_security.py"
    source = runner.read_text(encoding="utf-8")

    assert "sybil_anchor_idx = min(malicious_population)" in source
    assert "sybil_anchor_idx=sybil_anchor_idx" in source
    assert "anchor_idx = int(sybil_anchor_idx) if sybil_anchor_idx is not None else min(malicious_set)" in source


def test_run_repro_smoke_dry_run(tmp_path):
    script = ROOT / "experiments" / "reproducibility" / "run_repro_smoke.py"
    config = ROOT / "experiments" / "reproducibility" / "configs" / "smoke_mnist.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config-file",
            str(config),
            "--dry-run",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "unit_smoke",
            "--rounds",
            "1",
            "--num-clients",
            "2",
            "--clients-per-round",
            "2",
            "--attacks",
            "no_attack",
            "--baselines",
            "Vanilla_FL",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    manifest = tmp_path / "unit_smoke" / "run_manifest.json"
    assert manifest.exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["config"]["dataset"] == "MNIST"
    assert payload["config_source"]["sha256"]
    assert "experiments/scripts/runners/run_rq1_security.py" in payload["command"]


def test_validate_reproduction_flags_smoke_protocol(tmp_path):
    result_dir = tmp_path / "smoke_case" / "rq1_output"
    result_dir.mkdir(parents=True)
    (tmp_path / "smoke_case" / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "unit_1r_smoke",
                "dry_run": False,
                "config": {
                    "dataset": "CIFAR10",
                    "rounds": 1,
                    "num_clients": 5,
                    "clients_per_round": 5,
                },
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "config.json").write_text(
        json.dumps(
            {
                "dataset": "CIFAR10",
                "num_rounds": 1,
                "num_clients": 5,
                "clients_per_round": 5,
                "data_distribution": "NonIID_Dirichlet",
                "attacks": {"free_riding_no_training": {"malicious_ratios": [0.2]}},
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "rq1_results.json").write_text(
        json.dumps(
            [
                {
                    "attack_type": "free_riding_no_training",
                    "baseline_method": "Vanilla_FL",
                    "final_accuracy": 0.672,
                }
            ]
        ),
        encoding="utf-8",
    )

    script = ROOT / "experiments" / "reproducibility" / "validate_reproduction.py"
    output_dir = tmp_path / "validation"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--results-root",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads((output_dir / "validation_manifest.json").read_text(encoding="utf-8"))
    assert payload["summary"]["overall"]["protocol_mismatch"] == 1
    mismatch = [item for item in payload["comparisons"] if item["status"] == "protocol_mismatch"][0]
    assert "rounds=1" in "; ".join(mismatch["protocol_mismatches"])


def test_validate_reproduction_empty_results(tmp_path):
    script = ROOT / "experiments" / "reproducibility" / "validate_reproduction.py"
    results_root = tmp_path / "empty_results"
    output_dir = tmp_path / "validation"
    results_root.mkdir()
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--results-root",
            str(results_root),
            "--output-dir",
            str(output_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    manifest = output_dir / "validation_manifest.json"
    assert manifest.exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["summary"]["overall"]["total"] == 583
    assert payload["summary"]["overall"]["missing"] > 0


def test_validate_reproduction_ignores_archived_result_dirs(tmp_path):
    from experiments.reproducibility.validate_reproduction import _json_files

    live = tmp_path / "live_case" / "rq1_output"
    archived = tmp_path / "live_case__archived_before_rerun_20260520" / "rq1_output"
    aborted = tmp_path / "old_case__aborted_before_fix_20260520" / "rq1_output"
    diagnostic = tmp_path / "diagnostics" / "recipe_probe" / "rq1_output"
    for folder in [live, archived, aborted, diagnostic]:
        folder.mkdir(parents=True)
        (folder / "rq1_results.json").write_text("[]", encoding="utf-8")

    files = _json_files([tmp_path], "rq1_results.json")

    assert files == [(live / "rq1_results.json").resolve()]


def test_validate_reproduction_rq1_sample_match(tmp_path):
    script = ROOT / "experiments" / "reproducibility" / "validate_reproduction.py"
    run_dir = tmp_path / "rq1_run"
    output_dir = tmp_path / "validation"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({"dataset": "CIFAR10"}), encoding="utf-8")
    (run_dir / "rq1_results.json").write_text(
        json.dumps(
            [
                {
                    "attack_type": "free_riding_no_training",
                    "baseline_method": "Vanilla_FL",
                    "final_accuracy": 0.672,
                }
            ]
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--rq1-json",
            str(run_dir / "rq1_results.json"),
            "--results-root",
            str(tmp_path / "empty"),
            "--output-dir",
            str(output_dir),
            "--no-enforce-protocol",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads((output_dir / "validation_manifest.json").read_text(encoding="utf-8"))
    matches = [
        item
        for item in payload["comparisons"]
        if item.get("table") == "table1_main_security"
        and item.get("dataset") == "CIFAR-10"
        and item.get("attack") == "Free-riding (NT)"
        and item.get("method") == "Vanilla"
        and item.get("metric") == "MA"
    ]
    assert matches
    assert matches[0]["status"] == "pass"


def test_audit_reproduction_coverage(tmp_path):
    script = ROOT / "experiments" / "reproducibility" / "audit_reproduction_coverage.py"
    output_dir = tmp_path / "coverage"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(output_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads((output_dir / "coverage_manifest.json").read_text(encoding="utf-8"))
    assert payload["summary"]["overall"]["total"] == 583
    assert payload["summary"]["overall"]["runnable"] > 0
    assert payload["summary"]["overall"].get("blocked_method", 0) == 0
    assert payload["summary"]["overall"]["measurement_required"] > 0

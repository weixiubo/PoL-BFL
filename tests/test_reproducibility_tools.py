import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _valid_table1_protocol(**overrides):
    protocol = {
        "rounds": 200,
        "num_clients": 50,
        "clients_per_round": 50,
        "local_epochs": 5,
    }
    protocol.update(overrides)
    return protocol


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
    assert pol_job.config["attack_params"]["submission_mode"] == "random_update"
    assert "--attack_param" in pol_job.command
    assert "submission_mode=random_update" in pol_job.command

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
        "output_root": "experiments/results/repro_recovery/formal/unit_hparams",
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


def test_run_paper_config_attack_params_are_resume_significant(tmp_path):
    from experiments.reproducibility.run_paper_config import Job, _job_completed

    output_dir = tmp_path / "cell"
    output_dir.mkdir()
    expected = output_dir / "rq1_output" / "rq1_results.json"
    expected.parent.mkdir()
    expected.write_text("[]", encoding="utf-8")
    manifest = {
        "status": "completed",
        "returncode": 0,
        "config": {
            "dataset": "CIFAR10",
            "attack": "byzantine_model_replacement",
            "attack_params": {"replacement_mix": 1.0},
            "baseline": "Vanilla_FL",
        },
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    job = Job(
        job_id="cell",
        command=[sys.executable, "runner.py"],
        output_dir=output_dir,
        expected_files=[expected],
        config={
            "dataset": "CIFAR10",
            "attack": "byzantine_model_replacement",
            "attack_params": {"replacement_mix": 0.29},
            "baseline": "Vanilla_FL",
        },
    )

    assert not _job_completed(job)

    manifest["config"]["attack_params"] = {"replacement_mix": 0.29}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert _job_completed(job)


def test_run_paper_config_validation_blocked_job_is_not_completed(tmp_path):
    from experiments.reproducibility.run_paper_config import Job, _job_completed

    output_dir = tmp_path / "blocked_cell"
    output_dir.mkdir()
    expected = output_dir / "rq1_output" / "rq1_results.json"
    expected.parent.mkdir()
    expected.write_text("[]", encoding="utf-8")
    config = {
        "dataset": "CIFAR10",
        "attack": "byzantine_model_replacement",
        "attack_params": {"replacement_mix": 0.29},
        "baseline": "ShapleyFL",
    }
    manifest = {
        "status": "completed",
        "returncode": 0,
        "config": config,
        "validation_gate": {"enabled": True, "blocking": True},
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    job = Job(
        job_id="blocked_cell",
        command=[sys.executable, "runner.py"],
        output_dir=output_dir,
        expected_files=[expected],
        config=config,
    )

    assert not _job_completed(job)


def test_run_paper_config_resume_skips_live_running_manifest_by_default(tmp_path, monkeypatch):
    from argparse import Namespace

    from experiments.reproducibility import run_paper_config
    from experiments.reproducibility.run_paper_config import Job, _filter_jobs

    monkeypatch.setattr(run_paper_config, "_running_manifest_has_live_process", lambda job, manifest: True)

    output_dir = tmp_path / "running_cell"
    output_dir.mkdir()
    (output_dir / "run_manifest.json").write_text(
        json.dumps({"status": "running", "returncode": None, "config": {"dataset": "CIFAR10"}}),
        encoding="utf-8",
    )
    job = Job(
        job_id="running_cell",
        command=[sys.executable, "runner.py"],
        output_dir=output_dir,
        expected_files=[output_dir / "rq1_output" / "rq1_results.json"],
        config={"dataset": "CIFAR10"},
    )

    selected = _filter_jobs(
        [job],
        Namespace(
            only=[],
            resume=True,
            refresh_validation_gates=False,
            include_running_manifests=False,
            skip_passed_validation_manifest=None,
            limit=None,
        ),
    )
    assert selected == []

    selected_with_override = _filter_jobs(
        [job],
        Namespace(
            only=[],
            resume=True,
            refresh_validation_gates=False,
            include_running_manifests=True,
            skip_passed_validation_manifest=None,
            limit=None,
        ),
    )
    assert selected_with_override == [job]


def test_run_paper_config_resume_reruns_stale_running_manifest(tmp_path, monkeypatch):
    from argparse import Namespace

    from experiments.reproducibility import run_paper_config
    from experiments.reproducibility.run_paper_config import Job, _filter_jobs

    monkeypatch.setattr(run_paper_config, "_running_manifest_has_live_process", lambda job, manifest: False)
    output_dir = tmp_path / "stale_running_cell"
    output_dir.mkdir()
    (output_dir / "run_manifest.json").write_text(
        json.dumps({"status": "running", "returncode": None, "config": {"dataset": "CIFAR10"}}),
        encoding="utf-8",
    )
    job = Job(
        job_id="stale_running_cell",
        command=[sys.executable, "runner.py"],
        output_dir=output_dir,
        expected_files=[output_dir / "rq1_output" / "rq1_results.json"],
        config={"dataset": "CIFAR10"},
    )

    selected = _filter_jobs(
        [job],
        Namespace(
            only=[],
            resume=True,
            refresh_validation_gates=False,
            include_running_manifests=False,
            skip_passed_validation_manifest=None,
            limit=None,
        ),
    )

    assert selected == [job]


def test_run_paper_config_attack_params_get_distinct_job_id():
    from argparse import Namespace

    from experiments.reproducibility.run_paper_config import _build_rq1_jobs

    config_path = ROOT / "experiments" / "reproducibility" / "configs" / "paper" / "rq1_main_security_formal.json"
    config = {
        "runner": "experiments/scripts/runners/run_rq1_security.py",
        "output_root": "experiments/results/repro_recovery/formal/unit_params",
        "protocol": {"pol_integrity": 1},
        "execution": {"rounds": 3, "num_clients": 5, "clients_per_round": 2, "local_epochs": 1, "seeds": [42]},
        "datasets": [{"name": "CIFAR10", "model": "ResNet18"}],
        "attacks": ["byzantine_model_replacement"],
        "attack_params": {"byzantine_model_replacement": {"replacement_mix": 0.29}},
        "attack_profiles": {"byzantine_model_replacement": "paper_table1"},
        "baselines": ["Vanilla_FL"],
    }
    job = _build_rq1_jobs(config, config_path, Namespace(python=sys.executable))[0]

    assert "replacement_mix_0.29" in job.job_id
    assert job.config["attack_profile"] == "paper_table1"
    assert job.env["POL_ATTACK_PROFILE"] == "paper_table1"


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


def test_run_paper_config_can_skip_jobs_already_validated_pass(tmp_path):
    from argparse import Namespace

    from experiments.reproducibility.run_paper_config import _build_rq1_jobs, _filter_jobs

    config_path = ROOT / "experiments" / "reproducibility" / "configs" / "paper" / "rq1_main_security_formal.json"
    config = {
        "runner": "experiments/scripts/runners/run_rq1_security.py",
        "output_root": "experiments/results/repro_recovery/formal/unit_skip_validated",
        "protocol": {"pol_integrity": 1},
        "execution": {"rounds": 200, "num_clients": 50, "clients_per_round": 50, "local_epochs": 5, "seeds": [42]},
        "datasets": [{"name": "CIFAR10", "model": "ResNet18"}],
        "attacks": ["byzantine_alie"],
        "baselines": ["Vanilla_FL", "PoL_FL"],
    }
    jobs = _build_rq1_jobs(config, config_path, Namespace(python=sys.executable))
    validation_manifest = tmp_path / "validation_manifest.json"
    validation_manifest.write_text(
        json.dumps(
            {
                "comparisons": [
                    {
                        "table": "table1_main_security",
                        "dataset": "CIFAR-10",
                        "attack": "ALIE",
                        "method": "PoL-BFL",
                        "metric": metric,
                        "status": "pass",
                        "protocol": _valid_table1_protocol(),
                    }
                    for metric in ["MA", "DR", "FPR"]
                ]
            }
        ),
        encoding="utf-8",
    )

    selected = _filter_jobs(
        jobs,
        Namespace(
            only=["cifar10", "byzantine_alie"],
            resume=False,
            refresh_validation_gates=False,
            limit=None,
            skip_passed_validation_manifest=validation_manifest,
        ),
    )

    assert [job.config["baseline"] for job in selected] == ["Vanilla_FL"]


def test_run_paper_config_validation_manifest_overrides_completed_manifest(tmp_path):
    from argparse import Namespace

    from experiments.reproducibility.run_paper_config import Job, _filter_jobs

    output_dir = tmp_path / "foolsgold_cell"
    output_dir.mkdir()
    expected = output_dir / "rq1_output" / "rq1_results.json"
    expected.parent.mkdir()
    expected.write_text("{}", encoding="utf-8")
    config = {
        "runner": "experiments/scripts/runners/run_rq1_security.py",
        "dataset": "CIFAR10",
        "attack": "free_riding_no_training",
        "baseline": "FoolsGold",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps({"status": "completed", "returncode": 0, "config": config}),
        encoding="utf-8",
    )
    job = Job(
        job_id="cifar10__free_riding_no_training__foolsgold__seed42",
        command=[sys.executable, "runner.py"],
        output_dir=output_dir,
        expected_files=[expected],
        config=config,
    )
    validation_manifest = tmp_path / "validation_manifest.json"
    validation_manifest.write_text(
        json.dumps(
            {
                "comparisons": [
                    {
                        "table": "table1_main_security",
                        "dataset": "CIFAR-10",
                        "attack": "Free-riding (NT)",
                        "method": "FoolsGold",
                        "metric": metric,
                        "status": "protocol_mismatch",
                        "protocol_mismatches": ["clients_per_round=10 < paper protocol 50"],
                    }
                    for metric in ["MA", "DR", "FPR"]
                ]
            }
        ),
        encoding="utf-8",
    )

    selected = _filter_jobs(
        [job],
        Namespace(
            only=[],
            resume=True,
            refresh_validation_gates=False,
            include_running_manifests=False,
            skip_passed_validation_manifest=validation_manifest,
            limit=None,
        ),
    )

    assert selected == [job]


def test_run_paper_config_does_not_reuse_stale_protocol_pass(tmp_path):
    from argparse import Namespace

    from experiments.reproducibility.run_paper_config import Job, _filter_jobs

    output_dir = tmp_path / "stale_cell"
    output_dir.mkdir()
    expected = output_dir / "rq1_output" / "rq1_results.json"
    expected.parent.mkdir()
    expected.write_text("{}", encoding="utf-8")
    config = {
        "runner": "experiments/scripts/runners/run_rq1_security.py",
        "dataset": "CIFAR10",
        "attack": "free_riding_no_training",
        "baseline": "FoolsGold",
    }
    job = Job(
        job_id="cifar10__free_riding_no_training__foolsgold__seed42",
        command=[sys.executable, "runner.py"],
        output_dir=output_dir,
        expected_files=[expected],
        config=config,
    )
    stale_manifest = tmp_path / "stale_validation_manifest.json"
    stale_manifest.write_text(
        json.dumps(
            {
                "comparisons": [
                    {
                        "table": "table1_main_security",
                        "dataset": "CIFAR-10",
                        "attack": "Free-riding (NT)",
                        "method": "FoolsGold",
                        "metric": metric,
                        "status": "pass",
                        "protocol": {
                            "rounds": 200,
                            "num_clients": 50,
                            "clients_per_round": 10,
                            "local_epochs": 5,
                            "attack": "free_riding_no_training",
                            "attack_params": {"malicious_ratios": [0.2]},
                        },
                    }
                    for metric in ["MA", "DR", "FPR"]
                ]
            }
        ),
        encoding="utf-8",
    )

    selected = _filter_jobs(
        [job],
        Namespace(
            only=[],
            resume=True,
            refresh_validation_gates=False,
            include_running_manifests=False,
            skip_passed_validation_manifest=stale_manifest,
            limit=None,
        ),
    )

    assert selected == [job]


def test_run_paper_config_reuses_current_protocol_pass(tmp_path):
    from argparse import Namespace

    from experiments.reproducibility.run_paper_config import Job, _filter_jobs

    output_dir = tmp_path / "valid_cell"
    output_dir.mkdir()
    expected = output_dir / "rq1_output" / "rq1_results.json"
    expected.parent.mkdir()
    expected.write_text("{}", encoding="utf-8")
    config = {
        "runner": "experiments/scripts/runners/run_rq1_security.py",
        "dataset": "CIFAR10",
        "attack": "free_riding_no_training",
        "baseline": "FoolsGold",
    }
    job = Job(
        job_id="cifar10__free_riding_no_training__foolsgold__seed42",
        command=[sys.executable, "runner.py"],
        output_dir=output_dir,
        expected_files=[expected],
        config=config,
    )
    valid_manifest = tmp_path / "valid_validation_manifest.json"
    valid_protocol = _valid_table1_protocol(
        attack="free_riding_no_training",
        attack_params={"submission_mode": "random_update", "malicious_ratios": [0.2]},
        attack_effect={"attack_l2_mean_over_attacked_rounds": 1.0},
    )
    valid_manifest.write_text(
        json.dumps(
            {
                "comparisons": [
                    {
                        "table": "table1_main_security",
                        "dataset": "CIFAR-10",
                        "attack": "Free-riding (NT)",
                        "method": "FoolsGold",
                        "metric": metric,
                        "status": "pass",
                        "protocol": valid_protocol,
                    }
                    for metric in ["MA", "DR", "FPR"]
                ]
            }
        ),
        encoding="utf-8",
    )

    selected = _filter_jobs(
        [job],
        Namespace(
            only=[],
            resume=True,
            refresh_validation_gates=False,
            include_running_manifests=False,
            skip_passed_validation_manifest=valid_manifest,
            limit=None,
        ),
    )

    assert selected == []


def test_run_paper_config_clears_only_selected_job_expected_files(tmp_path):
    from experiments.reproducibility.run_paper_config import Job, _clear_stale_expected_files

    output_dir = tmp_path / "cell"
    expected = output_dir / "rq1_output" / "rq1_results.json"
    expected.parent.mkdir(parents=True)
    expected.write_text("stale", encoding="utf-8")
    outside = tmp_path / "other.json"
    outside.write_text("keep", encoding="utf-8")
    job = Job(
        job_id="cell",
        command=[sys.executable, "runner.py"],
        output_dir=output_dir,
        expected_files=[expected, outside],
        config={},
    )

    _clear_stale_expected_files(job)

    assert not expected.exists()
    assert outside.read_text(encoding="utf-8") == "keep"


def test_run_paper_config_rejects_unknown_gpu(monkeypatch):
    from argparse import Namespace

    from experiments.reproducibility import run_paper_config

    monkeypatch.setattr(run_paper_config, "_gpu_name", lambda gpu: None)
    monkeypatch.setattr(run_paper_config, "_gpu_free_memory_mb", lambda gpu: None)

    try:
        run_paper_config._validate_gpu_inventory(Namespace(gpus=["0"], require_gpu_name="4090"))
    except RuntimeError as exc:
        assert "not visible" in str(exc)
    else:
        raise AssertionError("unknown GPU inventory should block formal launch")


def test_run_paper_config_rejects_wrong_gpu_name(monkeypatch):
    from argparse import Namespace

    from experiments.reproducibility import run_paper_config

    monkeypatch.setattr(run_paper_config, "_gpu_name", lambda gpu: "NVIDIA GeForce RTX 4060 Laptop GPU")
    monkeypatch.setattr(run_paper_config, "_gpu_free_memory_mb", lambda gpu: 24000)

    try:
        run_paper_config._validate_gpu_inventory(Namespace(gpus=["0"], require_gpu_name="4090"))
    except RuntimeError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("wrong GPU name should block formal launch when required")


def test_rq1_sybil_uses_fixed_run_anchor_not_per_round_subset():
    runner = ROOT / "experiments" / "scripts" / "runners" / "run_rq1_security.py"
    source = runner.read_text(encoding="utf-8")

    assert "sybil_anchor_idx = min(malicious_population)" in source
    assert "sybil_anchor_idx=sybil_anchor_idx" in source
    assert "anchor_idx = int(sybil_anchor_idx) if sybil_anchor_idx is not None else min(malicious_set)" in source


def test_rq1_lazy_training_baseline_detection_uses_low_update_signal():
    torch = pytest.importorskip("torch")
    from collections import OrderedDict

    from experiments.scripts.runners.run_rq1_security import SecurityExperiment

    runner = SecurityExperiment.__new__(SecurityExperiment)
    reference = OrderedDict({"w": torch.zeros(2)})
    client_models = [
        OrderedDict({"w": torch.tensor([0.10, 0.00])}),
        OrderedDict({"w": torch.tensor([1.20, 0.00])}),
        OrderedDict({"w": torch.tensor([0.12, 0.00])}),
        OrderedDict({"w": torch.tensor([1.10, 0.00])}),
        OrderedDict({"w": torch.tensor([1.30, 0.00])}),
        OrderedDict({"w": torch.tensor([1.00, 0.00])}),
    ]
    selected_indices = [10, 11, 12, 13, 14, 15]
    # These outlier scores intentionally point to honest clients.  Lazy
    # training should be detected by low update magnitude, not high distance.
    aggregator = SimpleNamespace(
        scores=[0.0, 99.0, 0.1, 98.0, 97.0, 96.0],
        rejected_indices=[1, 3],
        suspicion_scores=[0.0, 99.0, 0.1, 98.0, 97.0, 96.0],
        client_weights=[0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    )

    for method in ["Krum", "SDEA", "FoolsGold", "ShapleyFL"]:
        suspects = runner._baseline_suspects(
            method,
            aggregator,
            client_models,
            selected_indices,
            expected_num_suspects=2,
            attack_type="free_riding_lazy_training",
            reference_state=reference,
        )
        assert suspects <= {"client_10", "client_12"}
        assert "client_10" in suspects
        if method == "ShapleyFL":
            assert suspects == {"client_10", "client_12"}


def test_rq1_non_lazy_baseline_detection_keeps_aggregator_evidence():
    torch = pytest.importorskip("torch")
    from collections import OrderedDict

    from experiments.scripts.runners.run_rq1_security import SecurityExperiment

    runner = SecurityExperiment.__new__(SecurityExperiment)
    reference = OrderedDict({"w": torch.zeros(1)})
    client_models = [
        OrderedDict({"w": torch.tensor([0.10])}),
        OrderedDict({"w": torch.tensor([1.20])}),
        OrderedDict({"w": torch.tensor([0.12])}),
        OrderedDict({"w": torch.tensor([1.10])}),
    ]
    aggregator = SimpleNamespace(scores=[0.0, 99.0, 0.1, 98.0])

    suspects = runner._baseline_suspects(
        "Krum",
        aggregator,
        client_models,
        selected_indices=[10, 11, 12, 13],
        expected_num_suspects=2,
        attack_type="byzantine_random_noise",
        reference_state=reference,
    )

    assert suspects == {"client_11", "client_13"}


def test_collect_recovery_evidence_no_remote(tmp_path):
    script = ROOT / "experiments" / "reproducibility" / "collect_recovery_evidence.py"
    if not script.exists():
        pytest.skip("optional recovery evidence collector is not included in this distribution")

    proc = subprocess.run(
        [sys.executable, str(script), "--no-remote", "--output-dir", str(tmp_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    manifest = tmp_path / "evidence_manifest.json"
    report = tmp_path / "gap_report.md"
    assert manifest.exists()
    assert report.exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert "table1_main_security" in payload["target_status"]
    assert payload["local"]["paper_targets"]["tables"]["table1_main_security"]["exists"] is True


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


def test_validate_reproduction_model_replacement_requires_paper_profile():
    from argparse import Namespace

    from experiments.reproducibility.validate_reproduction import _protocol_mismatches

    args = Namespace(
        no_enforce_protocol=False,
        min_rounds_rq1=200,
        min_clients_rq1=50,
        min_clients_per_round_rq1=50,
        min_local_epochs_rq1=5,
        allow_table1_noniid=False,
    )
    target = {"table": "table1_main_security", "attack": "Model Replacement", "dataset": "CIFAR-10"}
    base_protocol = {
        "dry_run": False,
        "rounds": 200,
        "num_clients": 50,
        "clients_per_round": 50,
        "local_epochs": 5,
        "malicious_ratios": [0.2],
        "data_distribution": "IID",
    }
    stress = {"protocol": {**base_protocol, "attack_params": {"replacement_mix": 1.0}}}
    paper = {
        "protocol": {
            **base_protocol,
            "attack_params": {"replacement_mix": 0.29},
            "attack_profile": "paper_table1",
        }
    }

    assert _protocol_mismatches(target, stress, args)
    assert _protocol_mismatches(target, paper, args) == []


def test_validate_reproduction_free_riding_nt_requires_random_update_effect():
    from argparse import Namespace

    from experiments.reproducibility.validate_reproduction import _protocol_mismatches

    args = Namespace(
        no_enforce_protocol=False,
        min_rounds_rq1=200,
        min_clients_rq1=50,
        min_clients_per_round_rq1=50,
        min_local_epochs_rq1=5,
        allow_table1_noniid=False,
    )
    target = {"table": "table1_main_security", "attack": "Free-riding (NT)", "dataset": "CIFAR-10"}
    base_protocol = {
        "dry_run": False,
        "rounds": 200,
        "num_clients": 50,
        "clients_per_round": 50,
        "local_epochs": 5,
        "malicious_ratios": [0.2],
        "data_distribution": "IID",
    }
    replay_global = {"protocol": {**base_protocol, "attack_params": {"malicious_ratios": [0.2]}}}
    random_but_zero = {
        "protocol": {
            **base_protocol,
            "attack_params": {"submission_mode": "random_update", "noise_scale": 1.0},
            "attack_effect": {
                "rounds_with_malicious_clients": 200,
                "attack_l2_mean_over_attacked_rounds": 0.0,
                "attack_l2_max_over_attacked_rounds": 0.0,
            },
        }
    }
    paper = {
        "protocol": {
            **base_protocol,
            "attack_params": {"submission_mode": "random_update", "noise_scale": 1.0},
            "attack_effect": {
                "rounds_with_malicious_clients": 200,
                "attack_l2_mean_over_attacked_rounds": 12.0,
                "attack_l2_max_over_attacked_rounds": 18.0,
            },
        }
    }

    assert _protocol_mismatches(target, replay_global, args)
    assert _protocol_mismatches(target, random_but_zero, args)
    assert _protocol_mismatches(target, paper, args) == []


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

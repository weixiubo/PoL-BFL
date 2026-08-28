#!/usr/bin/env python3
"""Run one source-bound paper security cell with real training and ZK audits."""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import hashlib
import json
import math
import multiprocessing
import os
import random
import signal
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "scripts" / "utils"))


def _ensure_rapidsnark_cxx_runtime() -> None:
    if __name__ != "__main__" or not sys.platform.startswith("linux"):
        return
    if "--cpu-prover" not in sys.argv:
        return
    if os.environ.get("POLBFL_SYSTEM_CXX_PRELOADED") == "1":
        return
    system_cxx = Path(
        os.environ.get(
            "POLBFL_SYSTEM_CXX_LIBRARY",
            "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
        )
    ).resolve()
    if not system_cxx.is_file():
        return
    environment = os.environ.copy()
    existing = [
        value
        for value in environment.get("LD_PRELOAD", "").split(":")
        if value
    ]
    if str(system_cxx) not in existing:
        existing.insert(0, str(system_cxx))
    environment["LD_PRELOAD"] = ":".join(existing)
    environment["POLBFL_SYSTEM_CXX_PRELOADED"] = "1"
    os.execve(
        sys.executable,
        [sys.executable, "-m", "experiments.final.run_security_cell", *sys.argv[1:]],
        environment,
    )


_ensure_rapidsnark_cxx_runtime()

import numpy as np
import torch

from client.trainer.ProtocolPoLTrainer import ProtocolPoLTrainer
from data_utils import create_dataloaders, load_dataset, partition_data_by_user, partition_data_iid
from experiments.final.attacks import (
    alie_update,
    minmax_update,
    random_noise_update,
)
from experiments.final.baseline_algorithms import (
    BASELINE_METHODS,
    BaselineDecision,
    fedcoin_posap_weights,
    foolsgold_decision,
    krum_decision,
    low_weight_cluster,
    monte_carlo_shapley,
    optimize_sdea_weights,
    update_shapley_history,
    weighted_average_updates,
)
from experiments.final.contract_replay import replay_contract_rounds
from experiments.final.data_attacks import (
    DeterministicLabelPoison,
    clone_indexed_loader,
)
from experiments.final.client_worker import train_client_task
from experiments.final.manifest import create_run_manifest, sha256_file, source_identity, write_manifest_atomic
from experiments.final.partitions import partition_dataset_dirichlet
from experiments.final.preflight import md5_file
from experiments.final.recovery import align_round_log, discard_uncommitted_scratch
from experiments.final.target_provenance import (
    SECURITY_TARGET_FILES,
    load_merged_targets,
    target_paths,
)
from experiments.final.trust_setup import validate_trust_setup
from experiments.scripts.utils.models import create_model
from polbfl.aggregation import (
    VerifiedUpdate,
    aggregate_verified_updates,
    screen_update_outliers,
)
from polbfl.committee import ECDSAPublicKeyRegistry, ECDSASigner, QuorumDecision, ReceiptQuorum, proof_set_digest
from polbfl.communication import compress_update_4bit, decompress_update_4bit
from polbfl.crypto import domain_hash
from polbfl.incentives import (
    EconomicParameters,
    IncentiveEngine,
    ParticipantAccount,
    ParticipantRole,
    ProofOutcome,
    ProtocolLedger,
)
from polbfl.protocol import (
    HybridChallengeSampler,
    PaperRoundEngine,
    RoundSubmission,
    TraceCommitment,
    select_audit_clients,
)
from polbfl.storage import ContentAddressedStore
from polbfl.sybil import TraceFingerprint
from polbfl.zk import (
    Groth16Artifacts,
    Groth16Backend,
    ZKBundleVerifier,
    ZKCircuitConfig,
    ZKPoLProver,
)


PAPER_ATTACKS = {
    "NoAttack",
    "FreeRidingNT",
    "FreeRidingLT",
    "ByzantineRandom",
    "ModelReplacement",
    "ALIE",
    "MinMax",
    "DataPoisoning",
    "Sybil",
}
INCENTIVE_METHODS = {
    "VanillaFL": "Vanilla",
    "FedCoin": "FedCoin",
    "ShapleyFL": "ShapleyFL",
    "PoLBFL": "PoLBFL",
}
PAPER_METHODS = (
    *BASELINE_METHODS,
    "FedCoin",
    "PoLBFL",
    "TrimmedMean",
    "Median",
)
LAYER_PROFILES = {
    "L1": {"robust_aggregation": False, "sybil_and_reputation": False, "economic_enforcement": False},
    "L1L2": {"robust_aggregation": True, "sybil_and_reputation": False, "economic_enforcement": False},
    "L1L3": {"robust_aggregation": False, "sybil_and_reputation": True, "economic_enforcement": True},
    "Full": {"robust_aggregation": True, "sybil_and_reputation": True, "economic_enforcement": True},
}

def acceptance_target_paths(root: Path) -> tuple[Path, ...]:
    return target_paths(root, SECURITY_TARGET_FILES)


def load_acceptance_targets(root: Path) -> dict[str, Any]:
    return load_merged_targets(root, SECURITY_TARGET_FILES)


def summarize_security_rates(
    round_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate DR/FPR over client-round decisions; MA remains final-round."""

    if not round_rows:
        raise ValueError("security-rate aggregation requires at least one round")
    output: dict[str, Any] = {}
    for result_name, row_name in (
        ("DR", "detection_rate"),
        ("FPR", "false_positive_rate"),
    ):
        values = [float(row[row_name]) for row in round_rows]
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 100.0 for value in values
        ):
            raise ValueError(f"{row_name} must be finite and normalized to [0, 100]")
        output[result_name] = statistics.fmean(values)
    output.update(
        {
            "security_rate_aggregation": "arithmetic_mean_of_per_round_client_rates",
            "security_rate_unit": "client-round",
        }
    )
    return output


def evaluate_cell_acceptance(
    result: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> dict[str, Any]:
    dataset = str(result["dataset"])
    attack = str(result["attack"])
    study = str(result.get("study", "main"))
    method = str(result.get("method", "PoLBFL"))
    if study == "layer":
        variant = str(result["layer_variant"])
        target = targets["table_3_layer_dr"][dataset][attack][variant]
        checks = {
            "DR": float(result["DR"]) >= float(target),
            "FPR_range": 0.0 <= float(result["FPR"]) <= 100.0,
            "real_groth16": result.get("real_groth16") is True,
            "real_robust_aggregation": (
                result.get("real_robust_aggregation") is True
                if variant in {"L1L2", "Full"}
                else result.get("real_robust_aggregation") is False
            ),
            "real_contract_transition": (
                result.get("real_contract_transition") is True
                if variant in {"L1L3", "Full"}
                else result.get("real_contract_transition") is False
            ),
        }
        return {"passed": all(checks.values()), "checks": checks}
    if study == "incentive":
        label = str(result["table5_method"])
        target = targets["table_5_all_methods"][label]
        checks = {
            "ParticipationRate": float(result["ParticipationRate"]) >= float(target["ParticipationRate"]),
            "AttackSuccessRate": float(result["AttackSuccessRate"]) <= float(target["AttackSuccessRate"]),
            "ModelAccuracy": float(result["ModelAccuracy"]) >= float(target["ModelAccuracy"]),
        }
        if label == "PoLBFL":
            checks["real_contract_rounds"] = (
                result.get("real_contract_rounds") is True
                and int(result.get("contract_rounds", 0)) == 200
            )
        else:
            checks["real_training"] = (
                result.get("real_training") is True
                and int(result.get("training_rounds", 0)) == 200
            )
        return {"passed": all(checks.values()), "checks": checks}
    if study == "sybil_scalability":
        identity_count = int(result["sybil_identity_count"])
        observed_stake = float(result["sybil_stake_eth"])
        checks = {
            "identity_count": identity_count in {5, 10, 15, 20},
            "DR_range": 0.0 <= float(result["DR"]) <= 100.0,
            "FPR_range": 0.0 <= float(result["FPR"]) <= 100.0,
            "stake_floor": observed_stake + 1e-12 >= 0.05 * identity_count,
            "real_groth16": result.get("real_groth16") is True,
            "real_contract_transition": result.get("real_contract_transition") is True,
        }
        target = targets["figure_6_vector_targets"][dataset][str(identity_count)]
        checks.update(
            {
                "MA": float(result["MA"]) >= float(target["MA"]),
                "DR": float(result["DR"]) >= float(target["DR"]),
                "FPR": float(result["FPR"]) <= float(target["FPR"]),
                "stake_eth": observed_stake >= float(target["stake_eth"]),
            }
        )
        return {"passed": all(checks.values()), "checks": checks}
    if study == "noniid":
        partition_label = str(result["partition_label"])
        target = targets["table_9_noniid"][dataset][partition_label]
        if attack == "NoAttack":
            checks = {"NoAttackMA": float(result["MA"]) >= float(target["NoAttackMA"])}
        elif attack == "FreeRidingNT":
            checks = {
                "FreeRidingDR": float(result["DR"]) >= float(target["FreeRidingDR"]),
                "FPR": float(result["FPR"]) <= float(target["FPR"]),
            }
        elif attack == "ALIE":
            checks = {"ALIEDR": float(result["DR"]) >= float(target["ALIEDR"])}
        else:
            raise ValueError(f"unsupported non-IID study attack: {attack}")
        return {"passed": all(checks.values()), "checks": checks}
    if study == "composability":
        aggregation = str(result["aggregation_method"])
        aggregation_label = {
            "trimmed_mean": "TrimmedMean",
            "krum": "Krum",
            "median": "Median",
        }[aggregation]
        mode = str(result["composition_mode"])
        target = targets["table_4_all_modes"][aggregation_label][attack][mode]
        checks = {
            "MA": float(result["MA"]) >= float(target["MA"]),
            "DR": float(result["DR"]) >= float(target["DR"]),
            "FPR": float(result["FPR"]) <= float(target["FPR"]),
        }
        return {"passed": all(checks.values()), "checks": checks}
    if study == "scalability":
        target = targets["table_8_scalability"][str(int(result["num_clients"]))]
        checks = {
            "MA": float(result["MA"]) >= float(target["MA"]),
            "DR": float(result["DR"]) >= float(target["DR"]),
            "FPR": float(result["FPR"]) <= float(target["FPR"]),
            "runtime_seconds": float(result["runtime_seconds"])
            <= float(target["runtime_seconds"]),
            "communication_mb": float(result["communication_mb"])
            <= float(target["communication_mb"]),
            "seconds_per_client": float(result["seconds_per_client"])
            <= float(target["seconds_per_client"]),
        }
        return {"passed": all(checks.values()), "checks": checks}
    if study == "sensitivity":
        probability = Decimal(str(result["audit_probability"]))
        checks = {
            "audit_probability": probability
            in {
                Decimal("0.05"),
                Decimal("0.10"),
                Decimal("0.15"),
                Decimal("0.20"),
                Decimal("0.25"),
                Decimal("0.30"),
                Decimal("0.50"),
                Decimal("1.00"),
            },
            "MA_range": 0.0 <= float(result["MA"]) <= 100.0,
            "DR_range": 0.0 <= float(result["DR"]) <= 100.0,
            "FPR_range": 0.0 <= float(result["FPR"]) <= 100.0,
            "runtime_positive": float(result["runtime_seconds"]) > 0.0,
        }
        figure_target = targets["figure_4_spot_check_sensitivity"][
            format(probability, "f")
        ]
        checks.update(
            {
                "figure.MA": float(result["MA"]) >= float(figure_target["MA"]),
                "figure.DR": float(result["DR"]) >= float(figure_target["DR"]),
                "figure.FPR": float(result["FPR"]) <= float(figure_target["FPR"]),
                "figure.runtime_seconds": float(result["runtime_seconds"])
                <= float(figure_target["runtime_seconds"]),
            }
        )
        if probability == Decimal("0.20"):
            target = targets["table_2_pol_bfl"]["CIFAR10"]["FreeRidingNT"]
            overhead = targets["table_7_overhead"]
            checks.update(
                {
                    "default.MA": float(result["MA"]) >= float(target["MA"]),
                    "default.DR": float(result["DR"]) >= float(target["DR"]),
                    "default.FPR": float(result["FPR"]) <= float(target["FPR"]),
                    "default.runtime_seconds": float(result["runtime_seconds"])
                    <= float(overhead["runtime_seconds"]),
                }
            )
        return {"passed": all(checks.values()), "checks": checks}
    if attack == "NoAttack":
        checks = {
            "MA_range": 0.0 <= float(result["MA"]) <= 100.0,
            "DR_zero": float(result["DR"]) == 0.0,
            "FPR_range": 0.0 <= float(result["FPR"]) <= 100.0,
        }
        return {"passed": all(checks.values()), "checks": checks}

    if study != "main":
        raise ValueError(f"unsupported formal study: {study}")
    if method != "PoLBFL":
        target = targets["table_2_all_methods"][dataset][attack][method]
        checks = {"MA": float(result["MA"]) >= float(target["MA"])}
        if "DR" in target:
            checks.update(
                {
                    "DR": float(result["DR"]) >= float(target["DR"]),
                    "FPR": float(result["FPR"]) <= float(target["FPR"]),
                }
            )
        return {"passed": all(checks.values()), "checks": checks}
    security_target = targets["table_2_pol_bfl"][dataset][attack]
    checks = {
        "MA": float(result["MA"]) >= float(security_target["MA"]),
        "DR": float(result["DR"]) >= float(security_target["DR"]),
        "FPR": float(result["FPR"]) <= float(security_target["FPR"]),
    }
    if dataset == "CIFAR10":
        overhead = targets["table_7_overhead"]
        checks.update(
            {
                "runtime_seconds": float(result["runtime_seconds"])
                <= float(overhead["runtime_seconds"]),
                "communication_mb": float(result["communication_mb"])
                <= float(overhead["communication_mb"]),
                "storage_mb_per_client": float(result["storage_mb_per_client"])
                <= float(overhead["storage_mb_per_client"]),
            }
        )
    return {"passed": all(checks.values()), "checks": checks}


@dataclass
class ClientArtifact:
    client_id: str
    update: Mapping[str, torch.Tensor]
    trace: Any | None
    trainer: ProtocolPoLTrainer | None
    commitment: TraceCommitment
    fingerprint: TraceFingerprint
    storage_bytes: int
    communication_bytes: int
    computation_valid: bool
    store_root: str | None = None


@dataclass(frozen=True)
class BaselineRoundEvidence:
    update: Mapping[str, torch.Tensor]
    included_clients: tuple[str, ...]
    flagged_clients: frozenset[str]
    scores: Mapping[str, float]
    weights: Mapping[str, float]
    execution_digest: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def cpu_state(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, value.detach().cpu().clone()) for name, value in model.state_dict().items())


def model_delta(local: Mapping[str, torch.Tensor], global_state: Mapping[str, torch.Tensor]) -> OrderedDict:
    result = OrderedDict()
    for name, global_value in global_state.items():
        local_value = local[name]
        if global_value.is_floating_point():
            result[name] = local_value.to(torch.float32) - global_value.to(torch.float32)
        else:
            result[name] = torch.zeros_like(global_value)
    return result


def apply_delta(global_state: Mapping[str, torch.Tensor], delta: Mapping[str, Any]) -> OrderedDict:
    result = OrderedDict()
    for name, value in global_state.items():
        if value.is_floating_point():
            result[name] = value + delta[name].detach().cpu().to(dtype=value.dtype)
        else:
            result[name] = value.detach().clone()
    return result


def transport_update(update: Mapping[str, torch.Tensor]) -> tuple[dict[str, Any], int]:
    payload = compress_update_4bit(update)
    restored = decompress_update_4bit(payload, update)
    return restored, len(payload)


def process_training_policy(
    *,
    attack: str,
    malicious: bool,
    local_epochs: int,
    record_pol: bool,
) -> tuple[int, bool]:
    """Return the worker policy for one client in a process-training cell.

    The paper's lazy-training free rider performs one local epoch while
    claiming the configured five-epoch execution.  It therefore must not emit
    an honest PoL trace for that shortened computation.  All other clients use
    the cell's declared epoch count and PoL setting unchanged.
    """

    if attack == "FreeRidingLT" and malicious:
        return 1, False
    return int(local_epochs), bool(record_pol)


def client_account_snapshot(
    accounts: Mapping[str, ParticipantAccount],
    malicious: Iterable[str],
) -> dict[str, Any]:
    clients = {
        client_id: account
        for client_id, account in accounts.items()
        if account.role == ParticipantRole.CLIENT
    }
    if not clients:
        raise ValueError("account snapshot requires registered clients")
    malicious_set = set(malicious)
    reputation = {
        client_id: float(account.reputation)
        for client_id, account in sorted(clients.items())
    }
    effective_reputation = {
        client_id: (float(account.reputation) if account.active else 0.0)
        for client_id, account in sorted(clients.items())
    }
    stake = {
        client_id: str(account.stake)
        for client_id, account in sorted(clients.items())
    }
    honest_clients = sorted(set(clients) - malicious_set)
    malicious_clients = sorted(set(clients) & malicious_set)
    if not honest_clients:
        raise ValueError("account snapshot requires an honest reference population")
    return {
        "registered_clients": len(clients),
        "reputation_by_client": reputation,
        "effective_reputation_by_client": effective_reputation,
        "stake_by_client": stake,
        "honest_reputation_mean": statistics.fmean(
            effective_reputation[client_id] for client_id in honest_clients
        ),
        "malicious_reputation_mean": (
            statistics.fmean(
                effective_reputation[client_id] for client_id in malicious_clients
            )
            if malicious_clients
            else None
        ),
    }


def table5_metrics(
    rows: Iterable[Mapping[str, Any]],
    *,
    required_rounds: int = 200,
) -> dict[str, float]:
    ordered = sorted(rows, key=lambda row: int(row["round"]))
    if len(ordered) != required_rounds or [
        int(row["round"]) for row in ordered
    ] != list(range(required_rounds)):
        raise ValueError("Table 5 evidence must contain every round exactly once")
    participation = []
    malicious_submissions = 0
    malicious_successes = 0
    for row in ordered:
        registered = int(row["honest_registered_clients"])
        participating = int(row["honest_participating_clients"])
        if registered <= 0 or not 0 <= participating <= registered:
            raise ValueError("Table 5 participation counts are invalid")
        submitted = int(row.get("malicious_submissions", -1))
        successful = int(row.get("malicious_attack_successes", -1))
        if submitted < 0 or not 0 <= successful <= submitted:
            raise ValueError("Table 5 malicious outcome counts are invalid")
        participation.append(100.0 * participating / registered)
        malicious_submissions += submitted
        malicious_successes += successful
    return {
        "ParticipationRate": statistics.fmean(participation),
        "AttackSuccessRate": (
            0.0
            if malicious_submissions == 0
            else 100.0 * malicious_successes / malicious_submissions
        ),
        "ModelAccuracy": float(ordered[-1]["accuracy"]),
    }


def synthetic_commitment(round_id: str, client_id: str, expected_steps: int, seed: int) -> TraceCommitment:
    def digest(label: str) -> str:
        return hashlib.sha256(f"{round_id}:{client_id}:{seed}:{label}".encode()).hexdigest()

    checkpoint_count = math.ceil(expected_steps / 5) + 1
    return TraceCommitment(
        protocol_version="1",
        round_id=round_id,
        client_id=client_id,
        context_digest=digest("context"),
        merkle_root=digest("root"),
        checkpoint_count=checkpoint_count,
        first_step=0,
        final_step=expected_steps,
        final_model_digest=digest("model"),
        trace_digest=digest("trace"),
    )


def synthetic_fingerprint(commitment: TraceCommitment, *, seed: int, batch_size: int) -> TraceFingerprint:
    rng = np.random.default_rng(seed)
    vectors = tuple(
        tuple(float(value) for value in rng.normal(size=14))
        for _ in range(commitment.checkpoint_count)
    )
    index_domain = int.from_bytes(
        hashlib.sha256(f"POLBFL_SYNTHETIC_INDEX_V1:{commitment.client_id}".encode()).digest()[:4],
        "big",
    )
    indices = tuple(
        index_domain + position
        for position in range(commitment.final_step * batch_size)
    )
    return TraceFingerprint(commitment.client_id, commitment.merkle_root, vectors, indices)


def evaluate(model_name: str, classes: int, state: Mapping[str, torch.Tensor], loader, device: str) -> float:
    model = create_model(model_name, num_classes=classes, input_channels=1 if model_name == "TwoLayerCNN" else 3)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            data, labels = batch[:2]
            predictions = model(data.to(device)).argmax(dim=1).cpu()
            correct += int((predictions == labels).sum())
            total += int(labels.numel())
    return 100.0 * correct / total


def evaluate_with_predictions(
    model_name: str,
    classes: int,
    state: Mapping[str, torch.Tensor],
    loader,
    device: str,
) -> tuple[float, tuple[int, ...], tuple[int, ...]]:
    model = create_model(
        model_name,
        num_classes=classes,
        input_channels=1 if model_name == "TwoLayerCNN" else 3,
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    correct = total = 0
    all_predictions: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            data, labels = batch[:2]
            predictions = model(data.to(device)).argmax(dim=1).cpu()
            correct += int((predictions == labels).sum())
            total += int(labels.numel())
            all_predictions.extend(int(value) for value in predictions.tolist())
            all_labels.extend(int(value) for value in labels.tolist())
    if not total or len(all_predictions) != total or len(all_labels) != total:
        raise RuntimeError("model evaluation produced an incomplete prediction trace")
    return 100.0 * correct / total, tuple(all_predictions), tuple(all_labels)

class SecurityCell:
    def __init__(self, args):
        self.args = args
        self.method = str(args.method)
        self.is_polbfl = self.method == "PoLBFL"
        self.layer_variant = str(args.layer_variant)
        layer_profile = LAYER_PROFILES[self.layer_variant]
        self.enable_layer2 = bool(layer_profile["robust_aggregation"])
        self.enable_layer3 = bool(layer_profile["sybil_and_reputation"])
        self.apply_economic_enforcement = bool(
            layer_profile["economic_enforcement"]
        )
        self.root = ROOT
        self.run_dir = args.output.resolve()
        source = source_identity(self.root)
        self.source_commit = source["commit"]
        if not self.source_commit:
            raise RuntimeError("formal runner cannot resolve the deployed source commit")
        if not args.diagnostic and source["dirty"]:
            raise RuntimeError("formal runner requires a clean source tree")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.raw_path = self.run_dir / "rounds.jsonl"
        if self.raw_path.exists() and not args.resume:
            raise FileExistsError(f"run output already contains raw rounds: {self.raw_path}")
        self.devices = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
        if len(self.devices) != 2 or any("4090" not in torch.cuda.get_device_name(i) for i in range(2)):
            raise RuntimeError("formal security cells require exactly two RTX 4090 GPUs")
        if not args.diagnostic:
            usage_output = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                capture_output=True,
                timeout=10,
                check=True,
            ).stdout.strip().splitlines()
            usage = [tuple(int(part.strip()) for part in line.split(",")) for line in usage_output]
            if any(utilization > 10 or memory > 1024 for utilization, memory in usage):
                raise RuntimeError(f"formal run requires idle reference GPUs, observed {usage}")
        seed_everything(args.seed)
        matrix = json.loads((self.root / "experiments" / "final" / "paper_matrix.json").read_text())
        self.dataset_spec = matrix["datasets"][args.dataset]
        self.model_name = self.dataset_spec["model"]
        self.classes = int(self.dataset_spec["classes"])
        self.round_randomness = hashlib.sha256(f"training-seed:{args.seed}".encode()).digest()
        self.verifier_signers = [ECDSASigner.generate(f"verifier-{index}") for index in range(5)]
        self.verifier_registry = ECDSAPublicKeyRegistry(
            {signer.verifier_id: signer.public_pem for signer in self.verifier_signers}
        )
        self.trust_setup_record = None
        trust_setup_path = self.args.zk_build.resolve() / "trust_setup.json"
        if trust_setup_path.is_file():
            self.trust_setup_record = json.loads(trust_setup_path.read_text(encoding="utf-8"))
        if not args.diagnostic and self.is_polbfl:
            toolchain = json.loads(
                (self.root / "config" / "toolchain.lock.json").read_text(encoding="utf-8")
            )
            trust_setup = validate_trust_setup(
                build=self.args.zk_build,
                toolchain=toolchain,
            )
            if not trust_setup["passed"]:
                raise RuntimeError(f"formal run requires a production trust setup: {trust_setup}")
        self.backend = None if args.no_proofs or not self.is_polbfl else self._backend()
        self.poseidon_binary = args.poseidon_binary.resolve()
        if self.is_polbfl and not args.no_pol and not self.poseidon_binary.is_file():
            raise FileNotFoundError(
                f"native Poseidon helper is required for trace construction: {self.poseidon_binary}"
            )
        rng = np.random.default_rng(args.seed)
        self.malicious = frozenset(
            f"client-{int(index)}"
            for index in rng.choice(args.num_clients, size=args.num_malicious, replace=False)
        )
        self._prepare_data()
        self.global_model = create_model(
            self.model_name,
            num_classes=self.classes,
            input_channels=1 if self.model_name == "TwoLayerCNN" else 3,
        )
        self.global_state = cpu_state(self.global_model)
        parameters = EconomicParameters(
            base_reward=Decimal("0.172"),
            beta_work=Decimal("0"),
            beta_reputation=Decimal("0"),
            reputation_decay=Decimal("0.9"),
            slashing_ratio=Decimal("1"),
            challenge_probability=args.audit_probability,
            detection_probability=Decimal("0.965"),
            base_minimum_stake=Decimal("0.05"),
        )
        self.ledger = ProtocolLedger(IncentiveEngine(parameters), verifier_reward=Decimal("0"))
        for index in range(args.num_clients):
            self.ledger.register(
                ParticipantAccount(f"client-{index}", ParticipantRole.CLIENT, Decimal("0.05"))
            )
        self.detected: set[str] = set()
        self.false_positives: set[str] = set()
        self.initial_accuracy: float | None = None
        self.foolsgold_history: torch.Tensor | None = None
        self.shapley_history: tuple[float, ...] | None = None
        self._audit_evidence: dict[str, dict[str, Any]] = {}
        self._audit_evidence_lock = threading.Lock()
        self.start_round = 0
        self.stop_requested = False
        self._last_worker_timings: list[Mapping[str, float]] = []
        signal.signal(signal.SIGTERM, lambda _signum, _frame: setattr(self, "stop_requested", True))
        signal.signal(signal.SIGINT, lambda _signum, _frame: setattr(self, "stop_requested", True))
        self.train_pools = None
        if args.process_training:
            context = multiprocessing.get_context("spawn")
            self.train_pools = [
                concurrent.futures.ProcessPoolExecutor(
                    max_workers=args.train_processes_per_gpu,
                    mp_context=context,
                )
                for _ in range(2)
            ]
        if args.resume:
            (self.run_dir / "stopped.json").unlink(missing_ok=True)
            self._restore_checkpoint()

    def _restore_checkpoint(self) -> None:
        checkpoint_path = self.run_dir / "checkpoint.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError("resume requested without checkpoint.pt")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint.get("source_commit") != self.source_commit:
            raise RuntimeError("checkpoint source commit differs from deployed source")
        self.global_state = checkpoint["global_state"]
        self.ledger.accounts = checkpoint["ledger_accounts"]
        self.ledger.processed_rounds = checkpoint["processed_rounds"]
        self.ledger.penalty_pool = checkpoint["penalty_pool"]
        self.detected = set(checkpoint["detected"])
        self.false_positives = set(checkpoint["false_positives"])
        self.initial_accuracy = float(checkpoint["initial_accuracy"])
        self.foolsgold_history = checkpoint.get("foolsgold_history")
        self.shapley_history = checkpoint.get("shapley_history")
        self.start_round = int(checkpoint["round"]) + 1
        recovery = align_round_log(
            self.raw_path,
            checkpoint_round=int(checkpoint["round"]),
        )
        scratch_recovery = discard_uncommitted_scratch(
            self.run_dir,
            next_round=self.start_round,
        )
        if recovery["dropped_rounds"] or scratch_recovery["removed_scratch_rounds"]:
            recovery_path = self.run_dir / "recovery-events.jsonl"
            with recovery_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "kind": "checkpoint_resume_reconciliation",
                            "round_log": recovery,
                            "scratch": scratch_recovery,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())

    def _write_checkpoint(self, round_number: int) -> None:
        destination = self.run_dir / "checkpoint.pt"
        temporary = self.run_dir / ".checkpoint.pt.tmp"
        torch.save(
            {
                "round": round_number,
                "global_state": self.global_state,
                "ledger_accounts": self.ledger.accounts,
                "processed_rounds": self.ledger.processed_rounds,
                "penalty_pool": self.ledger.penalty_pool,
                "detected": sorted(self.detected),
                "false_positives": sorted(self.false_positives),
                "initial_accuracy": self.initial_accuracy,
                "foolsgold_history": self.foolsgold_history,
                "shapley_history": self.shapley_history,
                "source_commit": self.source_commit,
            },
            temporary,
        )
        os.replace(temporary, destination)

    def _backend(self):
        build = self.args.zk_build.resolve()
        icicle_root = self.args.icicle_root.resolve()
        return Groth16Backend(
            Groth16Artifacts(
                wasm=build / "sampled_sgd_reference_js" / "sampled_sgd_reference.wasm",
                proving_key=build / "sampled_sgd_reference_final.zkey",
                verification_key=build / "verification_key.json",
                r1cs=build / "sampled_sgd_reference.r1cs",
            ),
            snarkjs_cli=self.root / "node_modules" / "snarkjs" / "cli.js",
            witness_binary=build / "sampled_sgd_reference_cpp" / "sampled_sgd_reference",
            prover_binary=self.args.rapidsnark_prover,
            verifier_binary=self.args.rapidsnark_verifier,
            prover_library=(
                self.args.rapidsnark_library if self.args.cpu_prover else None
            ),
            prover_pool_size=self.args.proof_workers,
            icicle_binary=(
                None
                if self.args.cpu_prover
                else icicle_root / "bin" / "icicle-snark"
            ),
            icicle_backend_directory=icicle_root / "backend",
            icicle_library_directories=(
                icicle_root / "lib",
                icicle_root / "backend" / "cuda",
            ),
            icicle_devices=self.args.icicle_devices,
            timeout_seconds=300,
        )

    def _runtime_artifacts(self) -> tuple[Path, ...]:
        artifacts: list[Path] = []
        if self.is_polbfl and not self.args.no_pol:
            artifacts.append(self.poseidon_binary)
        if self.is_polbfl and not self.args.no_proofs:
            build = self.args.zk_build.resolve()
            artifacts.extend(
                (
                    build / "sampled_sgd_reference.r1cs",
                    build / "sampled_sgd_reference_final.zkey",
                    build / "verification_key.json",
                    build / "sampled_sgd_reference_js" / "sampled_sgd_reference.wasm",
                    build / "sampled_sgd_reference_cpp" / "sampled_sgd_reference",
                    self.args.rapidsnark_prover.resolve(),
                    self.args.rapidsnark_verifier.resolve(),
                    self.args.rapidsnark_library.resolve(),
                )
            )
            trust_setup_path = build / "trust_setup.json"
            if trust_setup_path.is_file():
                artifacts.append(trust_setup_path)
            if not self.args.cpu_prover:
                icicle_root = self.args.icicle_root.resolve()
                artifacts.extend(
                    (
                        icicle_root / "bin" / "icicle-snark",
                        icicle_root / "lib" / "libicicle_device.so",
                        icicle_root / "lib" / "libicicle_field_bn254.so",
                        icicle_root / "lib" / "libicicle_curve_bn254.so",
                        icicle_root
                        / "backend"
                        / "cuda"
                        / "libicicle_backend_cuda_device.so",
                        icicle_root
                        / "backend"
                        / "cuda"
                        / "libicicle_backend_cuda_field_bn254.so",
                        icicle_root
                        / "backend"
                        / "cuda"
                        / "libicicle_backend_cuda_curve_bn254.so",
                    )
                )
        return tuple(artifacts)

    def _run_parameters(self) -> dict[str, Any]:
        return {
            "dataset": self.args.dataset,
            "study": self.args.study,
            "attack": self.args.attack,
            "method": self.method,
            "layer_variant": self.layer_variant,
            "robust_aggregation_enabled": self.enable_layer2,
            "sybil_and_reputation_enabled": self.enable_layer3,
            "economic_enforcement": self.apply_economic_enforcement,
            "sybil_identity_count": self.args.sybil_identities,
            "num_clients": self.args.num_clients,
            "num_malicious": self.args.num_malicious,
            "malicious_clients": sorted(self.malicious),
            "clients_per_round": self.args.clients_per_round,
            "rounds": self.args.rounds,
            "local_epochs": self.args.local_epochs,
            "batch_size": 32,
            "learning_rate": 0.01,
            "optimizer": "SGD",
            "proof_workers": self.args.proof_workers,
            "shapley_permutations": self.args.shapley_permutations,
            "audit_probability": str(self.args.audit_probability),
            "process_training": bool(self.args.process_training),
            "train_processes_per_gpu": self.args.train_processes_per_gpu,
            "diagnostic": bool(self.args.diagnostic),
            "no_proofs": bool(self.args.no_proofs),
            "no_pol": bool(self.args.no_pol),
            "update_transport": "signed_4bit_packed_zlib1",
            "evidence_storage": "content_addressed_pack_v1",
            "poseidon_backend": (
                "native_circom_compatible" if self.is_polbfl else None
            ),
            "groth16_prover_backend": (
                None
                if not self.is_polbfl
                else ("rapidsnark_cpu" if self.args.cpu_prover else "icicle_cuda")
            ),
            "icicle_devices": list(self.args.icicle_devices),
            "partition_mode": (
                "Dirichlet"
                if self.args.partition_alpha is not None
                else ("IID" if self.args.study == "noniid" else self.dataset_spec["partition"])
            ),
            "dirichlet_alpha": self.args.partition_alpha,
            "aggregation_method": self.args.aggregation_method,
            "composition_mode": self.args.composition_mode,
            "contract_timeout_seconds": self.args.contract_timeout_seconds,
            "trust_setup_record_digest": (
                None
                if self.trust_setup_record is None
                else self.trust_setup_record.get("record_digest")
            ),
        }

    def _prepare_data(self):
        train = load_dataset(self.args.dataset, data_dir=str(self.args.data_root / self.args.dataset), train=True)
        test = load_dataset(self.args.dataset, data_dir=str(self.args.data_root / self.args.dataset), train=False)
        if self.args.partition_alpha is not None:
            partitions = partition_dataset_dirichlet(
                train,
                num_clients=self.args.num_clients,
                alpha=self.args.partition_alpha,
                seed=self.args.seed,
            )
        else:
            partitions = (
                partition_data_by_user(train, self.args.num_clients)
                if self.args.dataset == "FEMNIST" and self.args.study == "main"
                else partition_data_iid(train, self.args.num_clients)
            )
        partition_payload = [
            [int(index) for index in partition.indices]
            for partition in partitions
        ]
        self.partition_digest = hashlib.sha256(
            json.dumps(partition_payload, separators=(",", ":")).encode()
        ).hexdigest()
        archive_name = {
            "CIFAR10": "cifar-10-python.tar.gz",
            "CIFAR100": "cifar-100-python.tar.gz",
        }.get(self.args.dataset)
        self.dataset_identity = {
            "name": self.args.dataset,
            "root": str(self.args.data_root / self.args.dataset),
            "partition_sha256": self.partition_digest,
            "partition_mode": (
                "Dirichlet"
                if self.args.partition_alpha is not None
                else ("IID" if self.args.study == "noniid" else self.dataset_spec["partition"])
            ),
            "dirichlet_alpha": self.args.partition_alpha,
        }
        if archive_name:
            archive = self.args.data_root / self.args.dataset / archive_name
            self.dataset_identity["archive_md5"] = md5_file(archive)
        self.train_loaders = create_dataloaders(partitions, batch_size=32, num_workers=0)
        for index, loader in enumerate(self.train_loaders):
            generator = torch.Generator().manual_seed(self.args.seed * 1009 + index)
            if hasattr(loader.sampler, "generator"):
                loader.sampler.generator = generator
        self.attack_loaders = {}
        self.attack_loader_base_seeds = {}
        malicious_indices = sorted(int(client.split("-")[-1]) for client in self.malicious)
        if self.args.attack == "DataPoisoning":
            for index in malicious_indices:
                poisoned = DeterministicLabelPoison(
                    self.train_loaders[index].dataset,
                    num_classes=self.classes,
                    poison_ratio=1.0,
                    seed=self.args.seed + index,
                )
                self.attack_loaders[index] = clone_indexed_loader(
                    self.train_loaders[index],
                    seed=self.args.seed * 1009 + index,
                    dataset=poisoned,
                )
                self.attack_loader_base_seeds[index] = self.args.seed * 1009 + index
        elif self.args.attack == "Sybil":
            groups = (
                [malicious_indices]
                if self.args.study == "sybil_scalability"
                else [
                    malicious_indices[start : start + 5]
                    for start in range(0, len(malicious_indices), 5)
                ]
            )
            for group_number, group in enumerate(groups):
                anchor_dataset = self.train_loaders[group[0]].dataset
                shared_seed = self.args.seed * 1009 + group_number
                for index in group:
                    self.attack_loaders[index] = clone_indexed_loader(
                        self.train_loaders[index],
                        seed=shared_seed,
                        dataset=anchor_dataset,
                    )
                    self.attack_loader_base_seeds[index] = shared_seed
        self.test_loader = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False, num_workers=2)

    def _train_valid_client(self, client_index: int, round_number: int, device: str, round_root: Path) -> ClientArtifact:
        client_id = f"client-{client_index}"
        model = create_model(
            self.model_name,
            num_classes=self.classes,
            input_channels=1 if self.model_name == "TwoLayerCNN" else 3,
        )
        model.load_state_dict(self.global_state, strict=True)
        args = {
            "enable_pol": True,
            "enable_zkp": True,
            "client_id": client_id,
            "round_num": round_number,
            "round_id": f"round-{round_number}",
            "model_id": f"{self.model_name}-{self.args.dataset}",
            "device": device,
            "optimizer": "SGD",
            "lr": 0.01,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "pol_save_freq": 5,
            "pol_save_dir": str(round_root),
            "round_randomness": self.round_randomness.hex(),
            "gradient_sample_rate": 0.01,
            "batch_size": 32,
            "pair_tolerance": 1e-5,
            "final_tolerance": 1e-3,
            "max_update_l2": 10.0,
            "node_binary": "node",
            "poseidon_binary": str(self.poseidon_binary),
            "packed_evidence": True,
            "clip_norm": None,
        }
        loader = self.attack_loaders.get(client_index, self.train_loaders[client_index])
        trainer = ProtocolPoLTrainer(model, loader, torch.nn.CrossEntropyLoss(), args=args)
        trainer.train(self.args.local_epochs)
        trainer.finalize_pol(epoch=self.args.local_epochs - 1)
        local_state = cpu_state(model)
        update = model_delta(local_state, self.global_state)
        update, communication_bytes = transport_update(update)
        recorded = trainer.recorded_trace
        storage_bytes = sum(item.blob.size for item in recorded.steps.values()) + sum(
            item.blob.size for item in recorded.checkpoints.values()
        )
        return ClientArtifact(
            client_id,
            update,
            recorded,
            trainer,
            recorded.trace.commitment,
            TraceFingerprint.from_recorded(recorded),
            storage_bytes,
            communication_bytes,
            True,
            str(trainer.trace_recorder.store.root),
        )

    def _synthetic_client(
        self,
        client_index: int,
        round_number: int,
        update_override: Mapping[str, torch.Tensor] | None = None,
    ) -> ClientArtifact:
        client_id = f"client-{client_index}"
        loader = self.train_loaders[client_index]
        expected_steps = len(loader) * self.args.local_epochs
        generator = torch.Generator().manual_seed(self.args.seed * 1_000_003 + round_number * 101 + client_index)
        update = (
            random_noise_update(
                self.global_state,
                generator=generator,
                scale=10.0 if self.args.attack == "ModelReplacement" else 1.0,
            )
            if update_override is None
            else OrderedDict(
                (name, value.detach().cpu().clone())
                for name, value in update_override.items()
            )
        )
        commitment = synthetic_commitment(f"round-{round_number}", client_id, expected_steps, self.args.seed)
        update, communication_bytes = transport_update(update)
        return ClientArtifact(
            client_id,
            update,
            None,
            None,
            commitment,
            synthetic_fingerprint(commitment, seed=self.args.seed + round_number * 100 + client_index, batch_size=32),
            0,
            communication_bytes,
            False,
            None,
        )

    def _train_lazy_client(
        self,
        client_index: int,
        round_number: int,
        device: str,
    ) -> ClientArtifact:
        client_id = f"client-{client_index}"
        model = create_model(
            self.model_name,
            num_classes=self.classes,
            input_channels=1 if self.model_name == "TwoLayerCNN" else 3,
        )
        model.load_state_dict(self.global_state, strict=True)
        model.to(device).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loader = self.train_loaders[client_index]
        for batch in loader:
            data, labels = batch[:2]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(data.to(device)), labels.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("lazy client produced a non-finite loss")
            loss.backward()
            optimizer.step()
        update = model_delta(cpu_state(model), self.global_state)
        update, communication_bytes = transport_update(update)
        expected_steps = len(loader) * self.args.local_epochs
        commitment = synthetic_commitment(f"round-{round_number}", client_id, expected_steps, self.args.seed)
        return ClientArtifact(
            client_id,
            update,
            None,
            None,
            commitment,
            synthetic_fingerprint(
                commitment,
                seed=self.args.seed + round_number * 100 + client_index,
                batch_size=32,
            ),
            0,
            communication_bytes,
            False,
            None,
        )

    def _train_round(self, round_number: int, active_indices: list[int], round_root: Path) -> list[ClientArtifact]:
        self._last_worker_timings = []
        train_indices = [
            index
            for index in active_indices
            if not (
                f"client-{index}" in self.malicious
                and self.args.attack
                in {"FreeRidingNT", "ByzantineRandom", "ModelReplacement", "ALIE", "MinMax"}
            )
        ]
        if self.train_pools is not None:
            global_path = round_root / "global-state.pt"
            torch.save({"global_state": self.global_state}, global_path)
            malicious_indices = sorted(
                int(client.split("-")[-1]) for client in self.malicious
            )
            sybil_sources = {}
            if self.args.attack == "Sybil":
                for group_start in range(0, len(malicious_indices), 5):
                    group = malicious_indices[group_start : group_start + 5]
                    for index in group:
                        sybil_sources[index] = (group[0], self.args.seed * 1009 + group_start)
            futures = []
            for position, index in enumerate(train_indices):
                device_index = position % 2
                is_malicious = f"client-{index}" in self.malicious
                worker_epochs, worker_record_pol = process_training_policy(
                    attack=self.args.attack,
                    malicious=is_malicious,
                    local_epochs=self.args.local_epochs,
                    record_pol=self.is_polbfl and not self.args.no_pol,
                )
                source_index, sampler_seed = sybil_sources.get(
                    index,
                    (index, self.args.seed * 1009 + index),
                )
                task = {
                    "dataset": self.args.dataset,
                    "data_root": str(self.args.data_root),
                    "num_clients": self.args.num_clients,
                    "study": self.args.study,
                    "partition_alpha": self.args.partition_alpha,
                    "seed": self.args.seed,
                    "client_index": index,
                    "source_index": source_index,
                    "sampler_seed": sampler_seed,
                    "round_number": round_number,
                    "device": self.devices[device_index],
                    "classes": self.classes,
                    "model_name": self.model_name,
                    "global_state_path": str(global_path),
                    "evidence_root": str(round_root),
                    "round_randomness": self.round_randomness.hex(),
                    "poseidon_binary": str(self.poseidon_binary),
                    "packed_evidence": True,
                    "local_epochs": worker_epochs,
                    "poison_labels": (
                        self.args.attack == "DataPoisoning"
                        and f"client-{index}" in self.malicious
                    ),
                    "poison_ratio": 1.0,
                    "record_pol": worker_record_pol,
                    "result_path": str(round_root / "worker-results" / f"client-{index}.pt"),
                }
                futures.append(self.train_pools[device_index].submit(train_client_task, task))
            artifacts = []
            for future in futures:
                result_path = Path(future.result())
                try:
                    payload = torch.load(result_path, map_location="cpu", weights_only=False)
                except TypeError:
                    payload = torch.load(result_path, map_location="cpu")
                self._last_worker_timings.append(
                    {
                        str(name): float(value)
                        for name, value in payload.get("timings", {}).items()
                    }
                )
                update = decompress_update_4bit(
                    payload["compressed_update"],
                    self.global_state,
                )
                commitment = payload["commitment"]
                fingerprint = payload["fingerprint"]
                if commitment is None or fingerprint is None:
                    client_index = int(payload["client_id"].split("-")[-1])
                    expected_steps = (
                        len(self.train_loaders[client_index]) * self.args.local_epochs
                    )
                    commitment = synthetic_commitment(
                        f"round-{round_number}",
                        payload["client_id"],
                        expected_steps,
                        self.args.seed,
                    )
                    fingerprint = synthetic_fingerprint(
                        commitment,
                        seed=self.args.seed + round_number * 100 + client_index,
                        batch_size=32,
                    )
                artifacts.append(
                    ClientArtifact(
                        payload["client_id"],
                        update,
                        payload["recorded"],
                        None,
                        commitment,
                        fingerprint,
                        int(payload["storage_bytes"]),
                        int(payload["communication_bytes"]),
                        bool(payload["computation_valid"]),
                        payload["store_root"],
                    )
                )
            present = {artifact.client_id for artifact in artifacts}
            benign_updates = [
                artifact.update
                for artifact in artifacts
                if artifact.client_id not in self.malicious
            ]
            crafted_update = None
            if self.args.attack == "ALIE" and len(benign_updates) >= 2:
                crafted_update = alie_update(benign_updates, z_max=2.5)
            elif self.args.attack == "MinMax" and len(benign_updates) >= 3:
                crafted_update = minmax_update(benign_updates)
            for index in active_indices:
                if f"client-{index}" not in present:
                    artifacts.append(self._synthetic_client(index, round_number, crafted_update))
            return sorted(artifacts, key=lambda item: item.client_id)

        worker_count = 2 * self.args.train_workers_per_gpu
        buckets = [train_indices[index::worker_count] for index in range(worker_count)]

        def worker(worker_index: int):
            device_index = worker_index % 2
            torch.cuda.set_device(device_index)
            return [
                (
                    self._train_lazy_client(index, round_number, self.devices[device_index])
                    if self.args.attack == "FreeRidingLT" and f"client-{index}" in self.malicious
                    else self._train_valid_client(
                        index,
                        round_number,
                        self.devices[device_index],
                        round_root,
                    )
                )
                for index in buckets[worker_index]
            ]

        artifacts: list[ClientArtifact] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(worker, worker_index) for worker_index in range(worker_count)]
            for future in futures:
                artifacts.extend(future.result())
        present = {artifact.client_id for artifact in artifacts}
        benign_updates = [
            artifact.update
            for artifact in artifacts
            if artifact.client_id not in self.malicious
        ]
        crafted_update = None
        if self.args.attack == "ALIE" and len(benign_updates) >= 2:
            crafted_update = alie_update(benign_updates, z_max=2.5)
        elif self.args.attack == "MinMax" and len(benign_updates) >= 3:
            crafted_update = minmax_update(benign_updates)
        for index in active_indices:
            if f"client-{index}" not in present:
                artifacts.append(self._synthetic_client(index, round_number, crafted_update))
        return sorted(artifacts, key=lambda item: item.client_id)

    def _baseline_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        data, labels = next(iter(self.test_loader))[:2]
        return data[:32].contiguous(), labels[:32].contiguous()

    def _baseline_model(self) -> torch.nn.Module:
        return create_model(
            self.model_name,
            num_classes=self.classes,
            input_channels=1 if self.model_name == "TwoLayerCNN" else 3,
        ).to(self.devices[0]).eval()

    def _marginal_accuracy_scores(
        self,
        artifacts: Sequence[ClientArtifact],
        *,
        base_state: Mapping[str, torch.Tensor],
    ) -> dict[str, float]:
        if not artifacts:
            return {}
        data, labels = self._baseline_batch()
        data = data.to(self.devices[0])
        labels = labels.to(self.devices[0])
        model = self._baseline_model()

        def utility(state: Mapping[str, torch.Tensor]) -> float:
            model.load_state_dict(state, strict=True)
            with torch.no_grad():
                prediction = model(data).argmax(dim=1)
            return 100.0 * float((prediction == labels).sum()) / int(labels.numel())

        baseline = utility(base_state)
        scores = {
            artifact.client_id: utility(
                apply_delta(base_state, artifact.update)
            )
            - baseline
            for artifact in artifacts
        }
        del model, data, labels
        return scores


    def _baseline_decision(
        self,
        artifacts: list[ClientArtifact],
        *,
        round_number: int,
    ) -> BaselineRoundEvidence:
        client_ids = tuple(artifact.client_id for artifact in artifacts)
        updates = tuple(artifact.update for artifact in artifacts)
        byzantine_bound = min(
            self.args.num_malicious,
            max(0, (len(artifacts) - 3) // 2),
        )
        method = self.method
        decision: BaselineDecision
        if method == "VanillaFL":
            weights = tuple(1.0 for _ in artifacts)
            decision = BaselineDecision(
                method=method,
                update=weighted_average_updates(updates, weights),
                included_indices=tuple(range(len(artifacts))),
                flagged_indices=frozenset(),
                scores=tuple(0.0 for _ in artifacts),
                weights=weights,
            )
        elif method == "Krum":
            decision = krum_decision(updates, byzantine_bound=byzantine_bound)
        elif method == "FoolsGold":
            decision, self.foolsgold_history = foolsgold_decision(
                updates,
                cumulative_history=self.foolsgold_history,
                byzantine_bound=byzantine_bound,
            )
        elif method == "FedCoin":
            data, labels = self._baseline_batch()
            data = data.to(self.devices[0])
            labels = labels.to(self.devices[0])
            model = self._baseline_model()
            update_by_client = dict(zip(client_ids, updates))

            def utility(coalition: tuple[str, ...]) -> float:
                if coalition:
                    delta = weighted_average_updates(
                        [update_by_client[client_id] for client_id in coalition],
                        [1.0] * len(coalition),
                    )
                    state = apply_delta(self.global_state, delta)
                else:
                    state = self.global_state
                model.load_state_dict(state, strict=True)
                with torch.no_grad():
                    prediction = model(data).argmax(dim=1)
                return 100.0 * float((prediction == labels).sum()) / int(labels.numel())

            contributions = monte_carlo_shapley(
                client_ids,
                utility,
                permutations=self.args.shapley_permutations,
                seed=self.args.seed * 1_000_003 + round_number,
            )
            values = tuple(contributions[client_id] for client_id in client_ids)
            weights = fedcoin_posap_weights(values)
            flagged = low_weight_cluster(weights, maximum=byzantine_bound)
            decision = BaselineDecision(
                method=method,
                update=weighted_average_updates(updates, weights),
                included_indices=tuple(
                    index for index, weight in enumerate(weights) if weight > 0
                ),
                flagged_indices=flagged,
                scores=tuple(float(value) for value in values),
                weights=tuple(float(value) for value in weights),
            )
            del model, data, labels
        elif method == "ShapleyFL":
            data, labels = self._baseline_batch()
            data = data.to(self.devices[0])
            labels = labels.to(self.devices[0])
            model = self._baseline_model()
            update_by_client = dict(zip(client_ids, updates))

            def utility(coalition: tuple[str, ...]) -> float:
                if coalition:
                    delta = weighted_average_updates(
                        [update_by_client[client_id] for client_id in coalition],
                        [1.0] * len(coalition),
                    )
                    state = apply_delta(self.global_state, delta)
                else:
                    state = self.global_state
                model.load_state_dict(state, strict=True)
                with torch.no_grad():
                    prediction = model(data).argmax(dim=1)
                return 100.0 * float((prediction == labels).sum()) / int(labels.numel())

            contributions = monte_carlo_shapley(
                client_ids,
                utility,
                permutations=self.args.shapley_permutations,
                seed=self.args.seed * 1_000_003 + round_number,
            )
            values = tuple(contributions[client_id] for client_id in client_ids)
            self.shapley_history = update_shapley_history(
                values,
                self.shapley_history,
                gamma=0.3,
            )
            weights = self.shapley_history
            flagged = low_weight_cluster(weights, maximum=byzantine_bound)
            decision = BaselineDecision(
                method=method,
                update=weighted_average_updates(updates, weights),
                included_indices=tuple(
                    index for index, weight in enumerate(weights) if weight > 0
                ),
                flagged_indices=flagged,
                scores=tuple(float(value) for value in values),
                weights=tuple(float(value) for value in weights),
            )
            del model, data, labels
        elif method == "SDEA":
            data, _labels = self._baseline_batch()
            generator = torch.Generator(device="cpu").manual_seed(
                self.args.seed * 1_000_003 + round_number
            )
            first_view = torch.rand(
                data.shape,
                generator=generator,
                dtype=data.dtype,
            ).to(self.devices[0])
            second_view = torch.rand(
                data.shape,
                generator=generator,
                dtype=data.dtype,
            ).to(self.devices[0])
            model = self._baseline_model()
            stacked_state = {}
            for name, global_value in self.global_state.items():
                if global_value.is_floating_point():
                    stacked_state[name] = torch.stack(
                        [
                            global_value.to(dtype=torch.float32)
                            + artifact.update[name].to(dtype=torch.float32)
                            for artifact in artifacts
                        ]
                    ).to(self.devices[0])
                else:
                    stacked_state[name] = global_value.to(self.devices[0])

            def evaluate_weighted_model(weights: torch.Tensor):
                device_weights = weights.to(self.devices[0], dtype=torch.float32)
                state = {
                    name: (
                        torch.tensordot(device_weights, value, dims=([0], [0]))
                        if value.ndim > self.global_state[name].ndim
                        else value
                    )
                    for name, value in stacked_state.items()
                }
                return (
                    torch.func.functional_call(model, state, (first_view,)),
                    torch.func.functional_call(model, state, (second_view,)),
                )

            weights = optimize_sdea_weights(
                len(artifacts),
                evaluate_weighted_model,
                iterations=20,
                learning_rate=0.1,
            )
            flagged = low_weight_cluster(weights, maximum=byzantine_bound)
            decision = BaselineDecision(
                method=method,
                update=weighted_average_updates(updates, weights),
                included_indices=tuple(
                    index for index, weight in enumerate(weights) if weight > 0
                ),
                flagged_indices=flagged,
                scores=tuple(float(1.0 - value) for value in weights),
                weights=weights,
            )
            del model, first_view, second_view, stacked_state
        elif method in {"TrimmedMean", "Median"}:
            aggregation_method = "trimmed_mean" if method == "TrimmedMean" else "median"
            submitted = [
                VerifiedUpdate(artifact.client_id, artifact.update, 1.0)
                for artifact in artifacts
            ]
            aggregate = aggregate_verified_updates(
                submitted,
                method=aggregation_method,
                byzantine_bound=byzantine_bound,
                device=self.devices[0],
            )
            screening = screen_update_outliers(
                [(artifact.client_id, artifact.update) for artifact in artifacts],
                randomness=hashlib.sha256(
                    f"baseline-screen:{self.args.seed}:{round_number}".encode()
                ).digest(),
                coordinate_sample=8192,
                mad_multiplier=3.5,
            )
            flagged = frozenset(client_ids.index(client_id) for client_id in screening.flagged_clients)
            decision = BaselineDecision(
                method=method,
                update=OrderedDict(aggregate.update),
                included_indices=tuple(range(len(artifacts))),
                flagged_indices=flagged,
                scores=tuple(1.0 if index in flagged else 0.0 for index in range(len(artifacts))),
                weights=tuple(1.0 for _ in artifacts),
            )
        else:
            raise ValueError(f"unsupported baseline method: {method}")

        flagged_clients = frozenset(client_ids[index] for index in decision.flagged_indices)
        included_clients = tuple(client_ids[index] for index in decision.included_indices)
        body = {
            "method": method,
            "round": round_number,
            "included_clients": list(included_clients),
            "flagged_clients": sorted(flagged_clients),
            "scores": list(decision.scores),
            "weights": list(decision.weights),
        }
        return BaselineRoundEvidence(
            update=decision.update,
            included_clients=included_clients,
            flagged_clients=flagged_clients,
            scores=dict(zip(client_ids, decision.scores)),
            weights=dict(zip(client_ids, decision.weights)),
            execution_digest=domain_hash(
                "POLBFL_BASELINE_ROUND_V1",
                json.dumps(body, sort_keys=True, separators=(",", ":")),
            ),
        )

    def _record_audit_evidence(
        self,
        *,
        artifact: ClientArtifact,
        challenge,
        proof_set_digest_value: str,
        receipts,
        decision: QuorumDecision,
        bundles=(),
        reports=(),
        diagnostic_bypass: bool = False,
    ) -> None:
        evidence = {
            "challenge_id": challenge.challenge_id,
            "commitment_root": challenge.commitment_root,
            "pair_indices": list(challenge.pair_indices),
            "proof_set_digest": proof_set_digest_value,
            "proof_digests": [bundle.proof.proof_digest for bundle in bundles],
            "proof_bytes": [len(bundle.proof.compact_bytes) for bundle in bundles],
            "prove_seconds": [float(bundle.proof.prove_seconds) for bundle in bundles],
            "witness_seconds": [float(bundle.proof.witness_seconds) for bundle in bundles],
            "verification_seconds": [float(report.verify_seconds) for report in reports],
            "verification_reasons": [list(report.reasons) for report in reports],
            "receipt_digests": [receipt.receipt_digest for receipt in receipts],
            "verifiers": [receipt.verifier_id for receipt in receipts],
            "decision": decision.value,
            "diagnostic_bypass": diagnostic_bypass,
        }
        with self._audit_evidence_lock:
            self._audit_evidence[artifact.client_id] = evidence

    def _verify_audited(self, artifact: ClientArtifact, challenge) -> ProofOutcome:
        if self.backend is None and artifact.computation_valid:
            digest = domain_hash(
                "POLBFL_DIAGNOSTIC_PROOF_BYPASS_V1",
                challenge.challenge_id,
            )
            self._record_audit_evidence(
                artifact=artifact,
                challenge=challenge,
                proof_set_digest_value=digest,
                receipts=(),
                decision=QuorumDecision.ACCEPT,
                diagnostic_bypass=True,
            )
            return ProofOutcome.ACCEPT
        if not artifact.computation_valid or artifact.trace is None:
            digest = domain_hash("POLBFL_MISSING_PROOF_SET_V1", challenge.challenge_id)
            receipts = [
                signer.receipt(challenge, proof_digest=digest, valid=False, verified_at_ns=challenge.issued_at_ns + 1)
                for signer in self.verifier_signers[:3]
            ]
            decision = ReceiptQuorum(
                committee=[signer.verifier_id for signer in self.verifier_signers],
                threshold=3,
                verify_signature=self.verifier_registry.verify,
            ).decide(challenge, receipts, proof_digest=digest)
            outcome = (
                ProofOutcome.REJECT
                if decision == QuorumDecision.REJECT
                else ProofOutcome.TIMEOUT
            )
            self._record_audit_evidence(
                artifact=artifact,
                challenge=challenge,
                proof_set_digest_value=digest,
                receipts=receipts,
                decision=decision,
            )
            return outcome
        if artifact.trainer is not None:
            bundles = artifact.trainer.respond_to_zk_challenge(challenge, backend=self.backend)
        elif artifact.store_root is not None:
            bundles = ZKPoLProver(
                self.backend,
                ZKCircuitConfig(),
                store=ContentAddressedStore(artifact.store_root),
            ).prove_challenge(recorded=artifact.trace, challenge=challenge)
        else:
            raise RuntimeError("valid audited artifact lacks its private evidence store")
        digest = proof_set_digest(challenge, bundles)
        receipts = []
        for signer in self.verifier_signers[:3]:
            valid = all(
                ZKBundleVerifier(self.backend).verify(artifact.trace.trace.context, bundle).valid
                for bundle in bundles
            )
            receipts.append(
                signer.receipt(
                    challenge,
                    proof_digest=digest,
                    valid=valid,
                    verified_at_ns=challenge.issued_at_ns + 1,
                )
            )
        decision = ReceiptQuorum(
            committee=[signer.verifier_id for signer in self.verifier_signers],
            threshold=3,
            verify_signature=self.verifier_registry.verify,
        ).decide(challenge, receipts, proof_digest=digest)
        self._record_audit_evidence(
            artifact=artifact,
            challenge=challenge,
            proof_set_digest_value=digest,
            receipts=receipts,
            decision=decision,
            bundles=bundles,
        )
        return ProofOutcome.ACCEPT if decision == QuorumDecision.ACCEPT else ProofOutcome.REJECT

    def _remove_baseline_scratch(self, round_root: Path) -> None:
        scratch_root = (self.run_dir / "scratch").resolve()
        target = round_root.resolve()
        if target.parent != scratch_root or not target.name.startswith("round-"):
            raise RuntimeError(f"refusing to remove unexpected scratch path: {target}")
        if target.is_dir():
            shutil.rmtree(target)

    def _retained_pol_evidence(
        self,
        round_root: Path,
        audited: set[str],
        artifacts: list[ClientArtifact],
    ) -> tuple[dict[str, dict[str, str]], frozenset[Path]]:
        target = round_root.resolve()
        scratch_root = (self.run_dir / "scratch").resolve()
        if target.parent != scratch_root or not target.name.startswith("round-"):
            raise RuntimeError(f"unexpected PoL scratch path: {target}")
        hashes: dict[str, dict[str, str]] = {}
        retained_roots: set[Path] = set()
        for artifact in artifacts:
            if artifact.client_id not in audited or artifact.store_root is None:
                continue
            store_root = Path(artifact.store_root).resolve()
            if not store_root.is_relative_to(target):
                raise RuntimeError(f"evidence store escaped round scratch: {store_root}")
            relative = store_root.relative_to(target)
            if not relative.parts:
                raise RuntimeError("evidence store cannot equal the round root")
            retained_roots.add(target / relative.parts[0])
            files = {
                str(path.relative_to(self.run_dir)): sha256_file(path)
                for path in sorted(store_root.rglob("*"))
                if path.is_file()
            }
            if not files:
                raise RuntimeError(f"audited evidence store is empty: {store_root}")
            hashes[artifact.client_id] = files
        return hashes, frozenset(retained_roots)

    def _prune_pol_scratch(
        self,
        round_root: Path,
        retained_roots: frozenset[Path],
    ) -> None:
        target = round_root.resolve()
        scratch_root = (self.run_dir / "scratch").resolve()
        if target.parent != scratch_root or not target.name.startswith("round-"):
            raise RuntimeError(f"refusing to prune unexpected scratch path: {target}")
        validated_retained = {path.resolve() for path in retained_roots}
        if any(path.parent != target for path in validated_retained):
            raise RuntimeError("retained evidence root escaped its round")
        for child in target.iterdir():
            resolved = child.resolve()
            if resolved in validated_retained:
                continue
            if resolved.parent != target:
                raise RuntimeError(f"scratch child escaped its round: {resolved}")
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    def _run_baseline(
        self,
        *,
        manifest: Mapping[str, Any],
        started_run: float,
        accuracy: float,
    ) -> dict[str, Any]:
        for round_number in range(self.start_round, self.args.rounds):
            started_round = time.perf_counter()
            for index, loader in enumerate(self.train_loaders):
                if hasattr(loader.sampler, "generator"):
                    loader.sampler.generator.manual_seed(
                        self.args.seed * 1_000_003 + round_number * 10_007 + index
                    )
            for index, loader in self.attack_loaders.items():
                if hasattr(loader.sampler, "generator"):
                    loader.sampler.generator.manual_seed(
                        self.attack_loader_base_seeds[index] + round_number * 10_007
                    )
            active_indices = list(range(self.args.num_clients))
            round_root = self.run_dir / "scratch" / f"round-{round_number}"
            round_root.mkdir(parents=True, exist_ok=True)
            started_training = time.perf_counter()
            artifacts = self._train_round(round_number, active_indices, round_root)
            training_seconds = time.perf_counter() - started_training
            started_aggregation = time.perf_counter()
            previous_global_state = self.global_state
            execution = self._baseline_decision(
                artifacts,
                round_number=round_number,
            )
            self.global_state = apply_delta(previous_global_state, execution.update)
            aggregation_seconds = time.perf_counter() - started_aggregation
            self.detected = set(execution.flagged_clients) & set(self.malicious)
            self.false_positives = set(execution.flagged_clients) - set(self.malicious)
            started_evaluation = time.perf_counter()
            accuracy, test_predictions, test_labels = evaluate_with_predictions(
                self.model_name,
                self.classes,
                self.global_state,
                self.test_loader,
                self.devices[0],
            )
            evaluation_seconds = time.perf_counter() - started_evaluation
            counterfactual_accuracy = None
            counterfactual_predictions: tuple[int, ...] = ()
            counterfactual_labels: tuple[int, ...] = ()
            attack_success = False
            if self.args.study == "incentive":
                honest_weights = [
                    (
                        0.0
                        if artifact.client_id in self.malicious
                        else float(execution.weights.get(artifact.client_id, 0.0))
                    )
                    for artifact in artifacts
                ]
                if sum(honest_weights) <= 0:
                    honest_weights = [
                        0.0 if artifact.client_id in self.malicious else 1.0
                        for artifact in artifacts
                    ]
                honest_update = weighted_average_updates(
                    [artifact.update for artifact in artifacts],
                    honest_weights,
                )
                counterfactual_state = apply_delta(
                    previous_global_state, honest_update
                )
                (
                    counterfactual_accuracy,
                    counterfactual_predictions,
                    counterfactual_labels,
                ) = evaluate_with_predictions(
                    self.model_name,
                    self.classes,
                    counterfactual_state,
                    self.test_loader,
                    self.devices[0],
                )
                attack_success = accuracy + 1e-9 < counterfactual_accuracy
            marginal_scores: dict[str, float] = {}
            malicious_successful_clients: tuple[str, ...] = ()
            if self.args.study == "incentive":
                if self.method in {"FedCoin", "ShapleyFL"}:
                    marginal_scores = {
                        str(client_id): float(value)
                        for client_id, value in execution.scores.items()
                    }
                else:
                    marginal_scores = self._marginal_accuracy_scores(
                        [
                            artifact
                            for artifact in artifacts
                            if artifact.client_id in self.malicious
                        ],
                        base_state=previous_global_state,
                    )
                included = set(execution.included_clients)
                malicious_successful_clients = tuple(
                    sorted(
                        client_id
                        for client_id in self.malicious & included
                        if marginal_scores.get(client_id, 0.0) > 0.0
                    )
                )
            account_snapshot = client_account_snapshot(self.ledger.accounts, self.malicious)
            included_malicious = len(set(execution.included_clients) & self.malicious)
            round_seconds = time.perf_counter() - started_round
            timing_names = sorted(
                {
                    name
                    for timings in self._last_worker_timings
                    for name in timings
                }
            )
            worker_timing_sum = {
                name: sum(timings.get(name, 0.0) for timings in self._last_worker_timings)
                for name in timing_names
            }
            worker_timing_max = {
                name: max(timings.get(name, 0.0) for timings in self._last_worker_timings)
                for name in timing_names
            }
            row = {
                "round": round_number,
                "method": self.method,
                "accuracy": accuracy,
                "test_predictions": list(test_predictions),
                "test_labels": list(test_labels),
                "prediction_digest": domain_hash(
                    "POLBFL_TEST_PREDICTIONS_V1",
                    round_number,
                    json.dumps(test_predictions),
                    json.dumps(test_labels),
                ),
                "counterfactual_honest_accuracy": counterfactual_accuracy,
                "counterfactual_honest_predictions": list(counterfactual_predictions),
                "counterfactual_honest_labels": list(counterfactual_labels),
                "counterfactual_honest_digest": (
                    None
                    if counterfactual_accuracy is None
                    else domain_hash(
                        "POLBFL_TABLE5_COUNTERFACTUAL_V1",
                        round_number,
                        json.dumps(counterfactual_predictions),
                        json.dumps(counterfactual_labels),
                    )
                ),
                "attack_success": attack_success,
                "marginal_accuracy_by_client": marginal_scores,
                "malicious_successful_clients": list(malicious_successful_clients),
                "malicious_attack_successes": len(malicious_successful_clients),
                "active_clients": len(active_indices),
                "audited_clients": [],
                "detected_malicious": sorted(self.detected),
                "detection_rate": (
                    0.0
                    if self.args.num_malicious == 0
                    else 100.0 * len(self.detected) / self.args.num_malicious
                ),
                "false_positive_rate": 100.0 * len(self.false_positives) / (
                    self.args.num_clients - self.args.num_malicious
                ),
                "included_malicious": included_malicious,
                "aggregation_included_clients": list(execution.included_clients),
                "registered_clients": account_snapshot["registered_clients"],
                "honest_registered_clients": (
                    self.args.num_clients - self.args.num_malicious
                ),
                "honest_participating_clients": sum(
                    artifact.client_id not in self.malicious for artifact in artifacts
                ),
                "valid_submissions": len(execution.included_clients),
                "malicious_submissions": self.args.num_malicious,
                "settlement_digest": None,
                "reputation_by_client": account_snapshot["reputation_by_client"],
                "effective_reputation_by_client": account_snapshot[
                    "effective_reputation_by_client"
                ],
                "stake_by_client": account_snapshot["stake_by_client"],
                "honest_reputation_mean": account_snapshot["honest_reputation_mean"],
                "malicious_reputation_mean": account_snapshot[
                    "malicious_reputation_mean"
                ],
                "storage_bytes_max": 0,
                "communication_bytes": sum(
                    artifact.communication_bytes for artifact in artifacts
                ),
                "training_seconds": training_seconds,
                "audit_seconds": 0.0,
                "aggregation_seconds": aggregation_seconds,
                "evaluation_seconds": evaluation_seconds,
                "round_seconds": round_seconds,
                "diagnostic_no_pol": False,
                "pol_enabled": False,
                "baseline_scores": execution.scores,
                "baseline_weights": execution.weights,
                "worker_timing_sum": worker_timing_sum,
                "worker_timing_max": worker_timing_max,
                "execution_digest": execution.execution_digest,
            }
            with self.raw_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self._write_checkpoint(round_number)
            self._remove_baseline_scratch(round_root)
            del artifacts, execution
            gc.collect()
            torch.cuda.empty_cache()
            if self.stop_requested:
                break
        if self.train_pools is not None:
            for pool in self.train_pools:
                pool.shutdown(wait=True, cancel_futures=False)
        if self.stop_requested:
            stopped = {
                "status": "stopped_at_checkpoint",
                "next_round": round_number + 1,
                "source_commit": self.source_commit,
                "method": self.method,
            }
            (self.run_dir / "stopped.json").write_text(
                json.dumps(stopped, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return stopped
        round_rows = [
            json.loads(line)
            for line in self.raw_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        round_runtime = [float(row["round_seconds"]) for row in round_rows]
        communication = [float(row["communication_bytes"]) for row in round_rows]
        sorted_runtime = sorted(round_runtime)
        runtime_p95 = sorted_runtime[max(0, math.ceil(0.95 * len(sorted_runtime)) - 1)]
        final = {
            "study": self.args.study,
            "dataset": self.args.dataset,
            "attack": self.args.attack,
            "method": self.method,
            "seed": self.args.seed,
            "rounds": self.args.rounds,
            "initial_accuracy": self.initial_accuracy,
            "MA": accuracy,
            **summarize_security_rates(round_rows),
            "runtime_seconds": statistics.fmean(round_runtime),
            "round_runtime_p95_seconds": runtime_p95,
            "run_wall_seconds": time.perf_counter() - started_run,
            "communication_mb": statistics.fmean(communication) / 1_000_000,
            "storage_mb_per_client": 0.0,
            "num_clients": self.args.num_clients,
            "seconds_per_client": statistics.fmean(round_runtime) / self.args.num_clients,
            "partition_label": (
                "IID"
                if self.args.partition_alpha is None
                else format(self.args.partition_alpha, "g")
            ),
            "aggregation_method": self.args.aggregation_method,
            "composition_mode": self.args.composition_mode,
            "audit_probability": 0.0,
            "malicious_clients": sorted(self.malicious),
            "source_commit": manifest["source"]["commit"],
            "real_training": True,
            "training_rounds": self.args.rounds,
            "baseline_source_lock_digest": sha256_file(
                self.root / "config" / "baseline_sources.lock.json"
            ),
        }
        if self.args.study == "incentive":
            metrics = table5_metrics(round_rows, required_rounds=self.args.rounds)
            final.update(
                {
                    **metrics,
                    "table5_method": INCENTIVE_METHODS[self.method],
                    "real_training": True,
                    "training_rounds": self.args.rounds,
                    "baseline_source_lock_digest": sha256_file(
                        self.root / "config" / "baseline_sources.lock.json"
                    ),
                }
            )
        targets = load_acceptance_targets(self.root)
        final["acceptance"] = evaluate_cell_acceptance(final, targets)
        final["formal_accepted"] = bool(
            not self.args.diagnostic and final["acceptance"]["passed"]
        )
        result_digest = hashlib.sha256(
            json.dumps(final, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        final["result_digest"] = result_digest
        final["evidence_digest"] = result_digest
        result_path = self.run_dir / "result.json"
        result_path.write_text(
            json.dumps(final, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        completed = create_run_manifest(
            root=self.root,
            run_id=self.args.run_id,
            seed=self.args.seed,
            configuration_files=(
                self.root / "config" / "paper_protocol.json",
                self.root / "config" / "paper_targets.json",
                *acceptance_target_paths(self.root),
                self.root / "config" / "baseline_sources.lock.json",
                self.root / "config" / "toolchain.lock.json",
                self.root / "experiments" / "final" / "paper_matrix.json",
            ),
            dataset=self.dataset_identity,
            artifacts=(self.raw_path, result_path),
            runtime_artifacts=(),
            run_parameters=self._run_parameters(),
            state="completed",
        )
        write_manifest_atomic(self.run_dir / "manifest.json", completed)
        return final

    def run(self):
        manifest = create_run_manifest(
            root=self.root,
            run_id=self.args.run_id,
            seed=self.args.seed,
            configuration_files=(
                self.root / "config" / "paper_protocol.json",
                self.root / "config" / "paper_targets.json",
                *acceptance_target_paths(self.root),
                self.root / "config" / "baseline_sources.lock.json",
                self.root / "config" / "toolchain.lock.json",
                self.root / "experiments" / "final" / "paper_matrix.json",
            ),
            dataset=self.dataset_identity,
            runtime_artifacts=self._runtime_artifacts(),
            run_parameters=self._run_parameters(),
            state="running",
        )
        write_manifest_atomic(self.run_dir / "manifest.json", manifest)
        started_run = time.perf_counter()
        accuracy = evaluate(
            self.model_name,
            self.classes,
            self.global_state,
            self.test_loader,
            self.devices[0],
        )
        if self.initial_accuracy is None:
            self.initial_accuracy = accuracy
        if not self.is_polbfl:
            return self._run_baseline(
                manifest=manifest,
                started_run=started_run,
                accuracy=accuracy,
            )
        for round_number in range(self.start_round, self.args.rounds):
            started_round = time.perf_counter()
            for index, loader in enumerate(self.train_loaders):
                if hasattr(loader.sampler, "generator"):
                    loader.sampler.generator.manual_seed(
                        self.args.seed * 1_000_003 + round_number * 10_007 + index
                    )
            for index, loader in self.attack_loaders.items():
                if hasattr(loader.sampler, "generator"):
                    loader.sampler.generator.manual_seed(
                        self.attack_loader_base_seeds[index] + round_number * 10_007
                    )
            eligible_indices = [
                index
                for index in range(self.args.num_clients)
                if self.ledger.accounts[f"client-{index}"].active
            ]
            round_rng = np.random.default_rng(
                self.args.seed * 1_000_003 + round_number * 10_007
            )
            active_indices = sorted(
                int(index)
                for index in round_rng.choice(
                    eligible_indices,
                    size=min(self.args.clients_per_round, len(eligible_indices)),
                    replace=False,
                )
            )
            round_root = self.run_dir / "scratch" / f"round-{round_number}"
            round_root.mkdir(parents=True, exist_ok=True)
            started_training = time.perf_counter()
            artifacts = self._train_round(round_number, active_indices, round_root)
            training_seconds = time.perf_counter() - started_training
            started_audit = time.perf_counter()
            with self._audit_evidence_lock:
                self._audit_evidence = {}
            audit_seed = hashlib.sha256(f"audit:{self.args.seed}:{round_number}".encode()).digest()
            audit = select_audit_clients(
                [artifact.commitment for artifact in artifacts],
                vrf_output=audit_seed,
                probability=self.args.audit_probability,
            )
            audited = set(audit.selected_clients)
            outcomes = {artifact.client_id: ProofOutcome.NOT_AUDITED for artifact in artifacts}
            challenges = {}
            for artifact in artifacts:
                if artifact.client_id in audited:
                    challenges[artifact.client_id] = HybridChallengeSampler(
                        recent_pairs=2,
                        random_pairs=3,
                    ).sample(
                        artifact.commitment,
                        vrf_output=hashlib.sha256(audit_seed + artifact.client_id.encode()).digest(),
                        issued_at_ns=time.time_ns(),
                        deadline_ns=time.time_ns() + 600_000_000_000,
                    )

            def verify(artifact):
                return artifact.client_id, self._verify_audited(artifact, challenges[artifact.client_id])

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.proof_workers) as pool:
                for client_id, outcome in pool.map(verify, [item for item in artifacts if item.client_id in audited]):
                    outcomes[client_id] = outcome
            audit_seconds = time.perf_counter() - started_audit
            eligible_for_screen = [
                artifact
                for artifact in artifacts
                if outcomes[artifact.client_id]
                in {ProofOutcome.ACCEPT, ProofOutcome.NOT_AUDITED}
            ]
            statistically_rejected: set[str] = set()
            if self.enable_layer2 and len(eligible_for_screen) >= 3:
                screening = screen_update_outliers(
                    [(artifact.client_id, artifact.update) for artifact in eligible_for_screen],
                    randomness=audit_seed,
                    coordinate_sample=8192,
                    mad_multiplier=3.5,
                )
                statistically_rejected.update(screening.flagged_clients)
            submissions = [
                RoundSubmission(
                    artifact.client_id,
                    artifact.update,
                    outcomes[artifact.client_id],
                    Decimal("1") if artifact.computation_valid else Decimal("0"),
                    artifact.fingerprint,
                    statistically_accepted=artifact.client_id not in statistically_rejected,
                )
                for artifact in artifacts
            ]
            inactive_clients = self.args.num_clients - len(eligible_indices)
            remaining_population_bound = max(
                0,
                self.args.num_malicious - inactive_clients,
            )
            sampled_byzantine_bound = math.ceil(
                len(artifacts)
                * remaining_population_bound
                / max(1, len(eligible_indices))
            )
            started_aggregation = time.perf_counter()
            previous_global_state = self.global_state
            reputations_before = {
                client_id: float(account.reputation)
                for client_id, account in self.ledger.accounts.items()
                if account.role == ParticipantRole.CLIENT
            }
            execution = PaperRoundEngine(
                self.ledger,
                aggregation_method=self.args.aggregation_method,
                byzantine_bound=(
                    min(
                        sampled_byzantine_bound,
                        max(0, (len(artifacts) - 1) // 2),
                    )
                    if self.enable_layer2
                    else 0
                ),
                sybil_cosine_threshold=0.995,
                aggregation_device=self.devices[0],
                enable_sybil_screening=self.enable_layer3,
                enable_reputation_weighting=self.enable_layer3,
                apply_economic_enforcement=self.apply_economic_enforcement,
            ).execute(round_id=f"round-{round_number}", submissions=submissions)
            self.global_state = apply_delta(
                previous_global_state, execution.aggregation.update
            )
            aggregation_seconds = time.perf_counter() - started_aggregation
            for client_id, penalty in execution.settlement.slashed.items():
                if client_id in self.malicious and penalty > 0:
                    self.detected.add(client_id)
            self.detected.update(
                client_id
                for client_id in execution.settlement.excluded_clients
                if client_id in self.malicious
            )
            round_false_positives = {
                client_id
                for client_id in execution.settlement.excluded_clients
                if client_id not in self.malicious
            }
            self.false_positives.update(round_false_positives)
            started_evaluation = time.perf_counter()
            accuracy, test_predictions, test_labels = evaluate_with_predictions(
                self.model_name,
                self.classes,
                self.global_state,
                self.test_loader,
                self.devices[0],
            )
            evaluation_seconds = time.perf_counter() - started_evaluation
            counterfactual_accuracy = None
            counterfactual_predictions: tuple[int, ...] = ()
            counterfactual_labels: tuple[int, ...] = ()
            attack_success = False
            if self.args.study == "incentive":
                honest_inputs = [
                    VerifiedUpdate(
                        client_id=artifact.client_id,
                        update=artifact.update,
                        reputation=reputations_before[artifact.client_id],
                        proof_eligible=True,
                        sybil_flagged=False,
                    )
                    for artifact in artifacts
                    if artifact.client_id not in self.malicious
                ]
                clean_aggregation = aggregate_verified_updates(
                    honest_inputs,
                    method=self.args.aggregation_method,
                    byzantine_bound=0,
                    device=self.devices[0],
                )
                counterfactual_state = apply_delta(
                    previous_global_state, clean_aggregation.update
                )
                (
                    counterfactual_accuracy,
                    counterfactual_predictions,
                    counterfactual_labels,
                ) = evaluate_with_predictions(
                    self.model_name,
                    self.classes,
                    counterfactual_state,
                    self.test_loader,
                    self.devices[0],
                )
                attack_success = accuracy + 1e-9 < counterfactual_accuracy
            marginal_scores: dict[str, float] = {}
            malicious_successful_clients: tuple[str, ...] = ()
            if self.args.study == "incentive":
                marginal_scores = self._marginal_accuracy_scores(
                    [
                        artifact
                        for artifact in artifacts
                        if artifact.client_id in self.malicious
                    ],
                    base_state=previous_global_state,
                )
                included = set(execution.aggregation.included_clients)
                malicious_successful_clients = tuple(
                    sorted(
                        client_id
                        for client_id in self.malicious & included
                        if marginal_scores.get(client_id, 0.0) > 0.0
                    )
                )
            included_malicious = len(set(execution.aggregation.included_clients) & self.malicious)
            account_snapshot = client_account_snapshot(self.ledger.accounts, self.malicious)
            retained_evidence, retained_roots = self._retained_pol_evidence(
                round_root,
                audited,
                artifacts,
            )
            audit_evidence = dict(sorted(self._audit_evidence.items()))
            if not self.args.diagnostic and set(audit_evidence) != audited:
                raise RuntimeError(
                    "formal audit evidence does not cover every selected client"
                )
            round_seconds = time.perf_counter() - started_round
            timing_names = sorted(
                {
                    name
                    for timings in self._last_worker_timings
                    for name in timings
                }
            )
            worker_timing_sum = {
                name: sum(timings.get(name, 0.0) for timings in self._last_worker_timings)
                for name in timing_names
            }
            worker_timing_max = {
                name: max(timings.get(name, 0.0) for timings in self._last_worker_timings)
                for name in timing_names
            }
            row = {
                "round": round_number,
                "accuracy": accuracy,
                "test_predictions": list(test_predictions),
                "test_labels": list(test_labels),
                "prediction_digest": domain_hash(
                    "POLBFL_TEST_PREDICTIONS_V1",
                    round_number,
                    json.dumps(test_predictions),
                    json.dumps(test_labels),
                ),
                "counterfactual_honest_accuracy": counterfactual_accuracy,
                "counterfactual_honest_predictions": list(counterfactual_predictions),
                "counterfactual_honest_labels": list(counterfactual_labels),
                "counterfactual_honest_digest": (
                    None
                    if counterfactual_accuracy is None
                    else domain_hash(
                        "POLBFL_TABLE5_COUNTERFACTUAL_V1",
                        round_number,
                        json.dumps(counterfactual_predictions),
                        json.dumps(counterfactual_labels),
                    )
                ),
                "attack_success": attack_success,
                "marginal_accuracy_by_client": marginal_scores,
                "malicious_successful_clients": list(malicious_successful_clients),
                "malicious_attack_successes": len(malicious_successful_clients),
                "active_clients": len(active_indices),
                "participating_clients": [
                    artifact.client_id for artifact in artifacts
                ],
                "trace_commitments": {
                    artifact.client_id: artifact.commitment.to_dict()
                    for artifact in artifacts
                },
                "audited_clients": sorted(audited),
                "audit_selection": {
                    "probability": str(audit.probability),
                    "population_size": audit.population_size,
                    "randomness_digest": audit.randomness_digest,
                    "transcript_digest": audit.transcript_digest,
                },
                "audit_evidence": audit_evidence,
                "retained_evidence_sha256": retained_evidence,
                "proof_outcomes": {
                    client_id: outcome.value
                    for client_id, outcome in sorted(outcomes.items())
                },
                "statistically_rejected_clients": sorted(statistically_rejected),
                "sybil_flagged_clients": sorted(execution.sybil_report.flagged_clients),
                "aggregation_included_clients": list(execution.aggregation.included_clients),
                "aggregation_excluded_clients": dict(execution.aggregation.excluded_clients),
                "settlement_excluded_clients": dict(execution.settlement.excluded_clients),
                "slashed_clients": {
                    client_id: str(amount)
                    for client_id, amount in execution.settlement.slashed.items()
                },
                "detected_malicious": sorted(self.detected),
                "detection_rate": (
                    0.0
                    if self.args.num_malicious == 0
                    else 100.0 * len(self.detected) / self.args.num_malicious
                ),
                "false_positive_rate": 100.0 * len(round_false_positives) / (
                    self.args.num_clients - self.args.num_malicious
                ),
                "false_positive_clients": sorted(round_false_positives),
                "ever_false_positive_clients": sorted(self.false_positives),
                "included_malicious": included_malicious,
                "registered_clients": account_snapshot["registered_clients"],
                "honest_registered_clients": (
                    self.args.num_clients - self.args.num_malicious
                ),
                "honest_participating_clients": sum(
                    artifact.client_id not in self.malicious for artifact in artifacts
                ),
                "valid_submissions": len(execution.settlement.eligible_clients),
                "malicious_submissions": sum(
                    artifact.client_id in self.malicious for artifact in artifacts
                ),
                "settlement_digest": execution.settlement.settlement_digest,
                "reputation_by_client": account_snapshot["reputation_by_client"],
                "effective_reputation_by_client": account_snapshot[
                    "effective_reputation_by_client"
                ],
                "stake_by_client": account_snapshot["stake_by_client"],
                "honest_reputation_mean": account_snapshot["honest_reputation_mean"],
                "malicious_reputation_mean": account_snapshot[
                    "malicious_reputation_mean"
                ],
                "storage_bytes_max": max(artifact.storage_bytes for artifact in artifacts),
                "communication_bytes": sum(
                    artifact.communication_bytes for artifact in artifacts
                ),
                "training_seconds": training_seconds,
                "audit_seconds": audit_seconds,
                "aggregation_seconds": aggregation_seconds,
                "evaluation_seconds": evaluation_seconds,
                "round_seconds": round_seconds,
                "diagnostic_no_pol": bool(self.args.no_pol),
                "worker_timing_sum": worker_timing_sum,
                "worker_timing_max": worker_timing_max,
                "execution_digest": execution.execution_digest,
            }
            with self.raw_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self._write_checkpoint(round_number)
            self._prune_pol_scratch(round_root, retained_roots)
            del artifacts, submissions, execution, challenges
            gc.collect()
            torch.cuda.empty_cache()
            if self.stop_requested:
                break
        if self.train_pools is not None:
            for pool in self.train_pools:
                pool.shutdown(wait=True, cancel_futures=False)
        if self.backend is not None:
            self.backend.close()
        if self.stop_requested:
            stopped = {
                "status": "stopped_at_checkpoint",
                "next_round": round_number + 1,
                "source_commit": self.source_commit,
            }
            (self.run_dir / "stopped.json").write_text(
                json.dumps(stopped, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return stopped
        round_rows = [
            json.loads(line)
            for line in self.raw_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        round_runtime = [float(row["round_seconds"]) for row in round_rows]
        communication = [float(row["communication_bytes"]) for row in round_rows]
        storage = [float(row["storage_bytes_max"]) for row in round_rows]
        contract_evidence = None
        contract_runtime = [0.0 for _ in round_rows]
        contract_evidence_path = self.run_dir / "contract-evidence.json"
        if (
            self.is_polbfl
            and self.apply_economic_enforcement
            and not self.args.diagnostic
        ):
            contract_evidence = replay_contract_rounds(
                rounds_path=self.raw_path,
                seed=self.args.seed,
                num_clients=self.args.num_clients,
                expected_rounds=self.args.rounds,
                output=contract_evidence_path,
                formal=True,
                timeout_seconds=self.args.contract_timeout_seconds,
                root=self.root,
            )
            contract_runtime = [
                float(row["runtime_seconds"]) for row in contract_evidence["rounds"]
            ]
        combined_runtime = [
            protocol + contract
            for protocol, contract in zip(round_runtime, contract_runtime, strict=True)
        ]
        sorted_runtime = sorted(combined_runtime)
        runtime_p95 = sorted_runtime[max(0, math.ceil(0.95 * len(sorted_runtime)) - 1)]
        final = {
            "study": self.args.study,
            "dataset": self.args.dataset,
            "attack": self.args.attack,
            "method": self.method,
            "seed": self.args.seed,
            "rounds": self.args.rounds,
            "initial_accuracy": self.initial_accuracy,
            "MA": accuracy,
            **summarize_security_rates(round_rows),
            "protocol_runtime_seconds": statistics.fmean(round_runtime),
            "contract_runtime_seconds": statistics.fmean(contract_runtime),
            "runtime_seconds": statistics.fmean(combined_runtime),
            "round_runtime_p95_seconds": runtime_p95,
            "run_wall_seconds": time.perf_counter() - started_run,
            "communication_mb": statistics.fmean(communication) / 1_000_000,
            "storage_mb_per_client": max(storage) / 1_000_000,
            "num_clients": self.args.num_clients,
            "seconds_per_client": statistics.fmean(combined_runtime) / self.args.num_clients,
            "partition_label": (
                "IID"
                if self.args.partition_alpha is None
                else format(self.args.partition_alpha, "g")
            ),
            "aggregation_method": self.args.aggregation_method,
            "composition_mode": self.args.composition_mode,
            "layer_variant": self.layer_variant,
            "sybil_identity_count": self.args.sybil_identities,
            "sybil_stake_eth": float(
                sum(
                    (self.ledger.accounts[client_id].stake for client_id in self.malicious),
                    Decimal("0"),
                )
            ),
            "real_groth16": bool(self.is_polbfl and not self.args.no_proofs),
            "real_robust_aggregation": self.enable_layer2,
            "real_contract_transition": bool(
                contract_evidence
                and contract_evidence["real_contract_transitions"] is True
            ),
            "real_contract_rounds": bool(
                contract_evidence
                and contract_evidence["real_contract_transitions"] is True
            ),
            "contract_rounds": (
                0 if contract_evidence is None else int(contract_evidence["contract_rounds"])
            ),
            "contract_transaction_count": (
                0 if contract_evidence is None else int(contract_evidence["transaction_count"])
            ),
            "contract_evidence_digest": (
                None if contract_evidence is None else contract_evidence["evidence_digest"]
            ),
            "audit_probability": float(self.args.audit_probability),
            "malicious_clients": sorted(self.malicious),
            "source_commit": manifest["source"]["commit"],
            "trust_setup_record_digest": (
                None
                if self.trust_setup_record is None
                else self.trust_setup_record.get("record_digest")
            ),
        }
        if self.args.study == "incentive":
            metrics = table5_metrics(round_rows, required_rounds=self.args.rounds)
            final.update(
                {
                    **metrics,
                    "table5_method": INCENTIVE_METHODS[self.method],
                    "real_training": True,
                    "training_rounds": self.args.rounds,
                }
            )
        targets = load_acceptance_targets(self.root)
        final["acceptance"] = evaluate_cell_acceptance(final, targets)
        if (
            self.is_polbfl
            and self.apply_economic_enforcement
            and not self.args.diagnostic
        ):
            contract_ok = bool(
                contract_evidence
                and contract_evidence.get("formal_accepted") is True
                and int(contract_evidence["contract_rounds"]) == self.args.rounds
                and len(str(contract_evidence.get("evidence_digest", ""))) == 64
            )
            final["acceptance"]["checks"]["real_contract_rounds"] = contract_ok
            final["acceptance"]["passed"] = bool(
                final["acceptance"]["passed"] and contract_ok
            )
        final["formal_accepted"] = bool(
            not self.args.diagnostic and final["acceptance"]["passed"]
        )
        final["result_digest"] = hashlib.sha256(
            json.dumps(
                final, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        final["evidence_digest"] = final["result_digest"]
        (self.run_dir / "result.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
        completed = create_run_manifest(
            root=self.root,
            run_id=self.args.run_id,
            seed=self.args.seed,
            configuration_files=(
                self.root / "config" / "paper_protocol.json",
                self.root / "config" / "paper_targets.json",
                *acceptance_target_paths(self.root),
                self.root / "config" / "baseline_sources.lock.json",
                self.root / "config" / "toolchain.lock.json",
                self.root / "experiments" / "final" / "paper_matrix.json",
            ),
            dataset=self.dataset_identity,
            artifacts=tuple(
                path
                for path in (
                    self.raw_path,
                    self.run_dir / "result.json",
                    contract_evidence_path,
                )
                if path.is_file()
            ),
            runtime_artifacts=self._runtime_artifacts(),
            run_parameters=self._run_parameters(),
            state="completed",
        )
        write_manifest_atomic(self.run_dir / "manifest.json", completed)
        return final


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("CIFAR10", "CIFAR100", "FEMNIST"), required=True)
    parser.add_argument(
        "--study",
        choices=("main", "layer", "noniid", "composability", "scalability", "sensitivity", "sybil_scalability", "incentive"),
        default="main",
    )
    parser.add_argument("--attack", choices=sorted(PAPER_ATTACKS), required=True)
    parser.add_argument("--method", choices=PAPER_METHODS, default="PoLBFL")
    parser.add_argument(
        "--layer-variant", choices=tuple(LAYER_PROFILES), default="Full"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--zk-build", type=Path, default=ROOT / "circuits" / "final" / "build" / "production")
    parser.add_argument("--rapidsnark-prover", type=Path, default=ROOT / ".tools" / "rapidsnark" / "package" / "bin" / "prover")
    parser.add_argument("--rapidsnark-verifier", type=Path, default=ROOT / ".tools" / "rapidsnark" / "package" / "bin" / "verifier")
    parser.add_argument(
        "--rapidsnark-library",
        type=Path,
        default=ROOT / ".tools" / "rapidsnark" / "package" / "lib" / "librapidsnark.so",
    )
    parser.add_argument(
        "--icicle-root",
        type=Path,
        default=ROOT / ".tools" / "icicle-snark",
    )
    parser.add_argument("--cpu-prover", action="store_true")
    parser.add_argument("--icicle-devices", default="0,1")
    parser.add_argument(
        "--poseidon-binary",
        type=Path,
        default=ROOT / ".tools" / "poseidon-native" / "polbfl-poseidon-native",
    )
    parser.add_argument("--num-clients", type=int, default=50)
    parser.add_argument("--num-malicious", type=int, default=10)
    parser.add_argument("--sybil-identities", type=int)
    parser.add_argument("--clients-per-round", type=int, default=50)
    parser.add_argument("--audit-probability", type=Decimal, default=Decimal("0.2"))
    parser.add_argument("--partition-alpha", type=float)
    parser.add_argument(
        "--aggregation-method",
        choices=("trimmed_mean", "krum", "median"),
        default="trimmed_mean",
    )
    parser.add_argument(
        "--composition-mode",
        choices=("Standalone", "PoLBFLPrefilter"),
        default="PoLBFLPrefilter",
    )
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--shapley-permutations", type=int, default=50)
    parser.add_argument("--proof-workers", type=int, default=8)
    parser.add_argument("--train-workers-per-gpu", type=int, default=4)
    parser.add_argument("--process-training", action="store_true")
    parser.add_argument("--train-processes-per-gpu", type=int, default=4)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--no-proofs", action="store_true")
    parser.add_argument("--no-pol", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--contract-timeout-seconds", type=int, default=7_200)
    args = parser.parse_args()
    try:
        args.icicle_devices = tuple(
            int(value.strip())
            for value in args.icicle_devices.split(",")
            if value.strip()
        )
    except ValueError:
        parser.error("--icicle-devices must be a comma-separated integer list")
    if not args.icicle_devices or len(set(args.icicle_devices)) != len(args.icicle_devices):
        parser.error("--icicle-devices must be non-empty and unique")
    if args.study == "scalability":
        expected_clients = args.num_clients
        expected_malicious = expected_clients // 5
    elif args.study == "sybil_scalability":
        expected_clients = 40 + int(args.sybil_identities or 0)
        expected_malicious = int(args.sybil_identities or 0)
    else:
        expected_clients = 50
        expected_malicious = (
            0
            if args.study == "noniid" and args.attack == "NoAttack"
            else 10
        )
    if not args.diagnostic and (
        args.num_clients != expected_clients
        or args.num_malicious != expected_malicious
        or args.clients_per_round != expected_clients
        or args.rounds != 200
        or args.local_epochs != 5
        or args.no_proofs
        or args.cpu_prover
        or args.icicle_devices != (0, 1)
        or not args.process_training
        or args.train_processes_per_gpu != 8
        or (args.method in {"ShapleyFL", "FedCoin"} and args.shapley_permutations != 50)
    ):
        parser.error("formal runs require the paper population, 200 rounds, five epochs, and the dual-GPU process profile")
    if args.contract_timeout_seconds <= 0:
        parser.error("--contract-timeout-seconds must be positive")
    if args.no_pol and not args.diagnostic:
        parser.error("--no-pol is restricted to diagnostic profiling")
    if args.no_pol and not args.process_training:
        parser.error("--no-pol diagnostic profiling requires --process-training")
    standalone_krum = (
        args.study == "composability"
        and args.composition_mode == "Standalone"
        and args.method == "Krum"
    )
    incentive_baseline = (
        args.study == "incentive"
        and args.method in {"VanillaFL", "ShapleyFL"}
    )
    if (
        not args.diagnostic
        and args.method in BASELINE_METHODS
        and args.study != "main"
        and not standalone_krum
        and not incentive_baseline
    ):
        parser.error("Table 2 baselines are restricted to main security or their declared study")
    if not args.diagnostic and args.method in {"TrimmedMean", "Median"} and args.study != "composability":
        parser.error("standalone Trimmed Mean and Median are restricted to composability")
    if args.partition_alpha is not None and args.partition_alpha <= 0:
        parser.error("--partition-alpha must be positive")
    if not Decimal("0") < args.audit_probability <= Decimal("1"):
        parser.error("--audit-probability must be in (0, 1]")
    allowed_sensitivity_probabilities = {
        Decimal("0.05"),
        Decimal("0.10"),
        Decimal("0.15"),
        Decimal("0.20"),
        Decimal("0.25"),
        Decimal("0.30"),
        Decimal("0.50"),
        Decimal("1.00"),
    }
    if args.study != "sensitivity" and args.audit_probability != Decimal("0.2") and not args.diagnostic:
        parser.error("formal reference studies require the paper's 20% audit probability")
    if args.study == "sensitivity" and (
        args.dataset != "CIFAR10"
        or args.attack != "FreeRidingNT"
        or args.audit_probability not in allowed_sensitivity_probabilities
        or args.partition_alpha is not None
        or args.aggregation_method != "trimmed_mean"
    ):
        parser.error("sensitivity cells require CIFAR10/FreeRidingNT and a paper plot probability")
    if args.study == "main" and args.partition_alpha is not None and not args.diagnostic:
        parser.error("main security cells use the paper's default partition")
    if not args.diagnostic and args.study != "layer" and args.layer_variant != "Full":
        parser.error("layer variants are restricted to the Table 3 study")
    if args.study == "layer" and (
        args.method != "PoLBFL"
        or args.attack not in {"FreeRidingNT", "ALIE", "Sybil"}
        or args.partition_alpha is not None
        or args.aggregation_method != "trimmed_mean"
    ):
        parser.error("layer cells require PoL-BFL, a Table 3 attack, and the default partition")
    if args.study in {"main", "layer", "noniid"} and args.aggregation_method != "trimmed_mean" and not args.diagnostic:
        parser.error("main and non-IID cells use the paper's reference Trimmed Mean")
    if args.study == "noniid" and args.attack not in {"NoAttack", "FreeRidingNT", "ALIE"}:
        parser.error("non-IID cells support NoAttack, FreeRidingNT, and ALIE")
    if args.study == "composability" and (
        args.dataset != "CIFAR10"
        or args.attack not in {"FreeRidingNT", "ALIE"}
        or args.partition_alpha is not None
        or (
            args.composition_mode == "PoLBFLPrefilter"
            and args.method != "PoLBFL"
        )
        or (
            args.composition_mode == "Standalone"
            and args.method
            != {"krum": "Krum", "trimmed_mean": "TrimmedMean", "median": "Median"}[
                args.aggregation_method
            ]
        )
    ):
        parser.error("composability cells require CIFAR10, the declared mode, and its matching aggregator")
    if args.study == "incentive" and (
        args.dataset != "CIFAR10"
        or args.attack != "FreeRidingNT"
        or args.method not in INCENTIVE_METHODS
        or args.partition_alpha is not None
        or args.aggregation_method != "trimmed_mean"
    ):
        parser.error("Table 5 cells require CIFAR10, FreeRidingNT, and a declared incentive method")
    if args.method == "FedCoin" and args.study != "incentive" and not args.diagnostic:
        parser.error("FedCoin is restricted to the Table 5 incentive study")
    if args.study != "sybil_scalability" and args.sybil_identities is not None and not args.diagnostic:
        parser.error("--sybil-identities is restricted to Figure 6")
    if args.study == "sybil_scalability" and (
        args.dataset != "CIFAR10"
        or args.attack != "Sybil"
        or args.method != "PoLBFL"
        or args.sybil_identities not in {5, 10, 15, 20}
        or args.num_clients != 40 + int(args.sybil_identities or 0)
        or args.num_malicious != int(args.sybil_identities or 0)
        or args.clients_per_round != args.num_clients
        or args.partition_alpha is not None
        or args.aggregation_method != "trimmed_mean"
    ):
        parser.error("Figure 6 cells require 40 honest clients plus 5, 10, 15, or 20 Sybil identities")
    if args.study == "scalability" and (
        args.dataset != "CIFAR10"
        or args.attack != "FreeRidingNT"
        or args.num_clients not in {50, 100, 200}
        or args.num_malicious != args.num_clients // 5
        or args.clients_per_round != args.num_clients
        or args.partition_alpha is not None
        or args.aggregation_method != "trimmed_mean"
    ):
        parser.error("scalability cells require CIFAR10/FreeRidingNT, N in {50,100,200}, and 20% attackers")
    return args


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("POL_INTEGRITY", "1")
    arguments = parse_args()
    print(json.dumps(SecurityCell(arguments).run(), indent=2, sort_keys=True))

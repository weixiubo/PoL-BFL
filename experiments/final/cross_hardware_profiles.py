"""Paper-bound numerical acceptance profiles for final-paper Table 11."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


AUTHORITY_PDF_SHA256 = (
    "0b013e58d4f99f91470c61a891a4ee89dfd09eff58e131abbe730d1f6f91e6d4"
)
POLBFL_PAIRS = frozenset(
    {
        "RTX4090_RTX4090",
        "V100_V100",
        "RTX4090_RTX3080",
        "RTX4090_V100",
        "RTX4090_A100",
        "V100_A100",
    }
)
KAIZEN_PAIR = "Kaizen_RTX4090_V100"
KAIZEN_CONFIG = Path("config/kaizen_cross_hardware_controlled.json")


@dataclass(frozen=True)
class CrossHardwareProfile:
    profile_id: str
    method: str
    hardware_pair: str
    pair_tolerance: float
    final_tolerance: float
    decision: str
    configuration_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_kaizen_control(root: Path) -> tuple[Mapping[str, Any], str]:
    path = root / KAIZEN_CONFIG
    raw = path.read_bytes()
    control = json.loads(raw)
    numerical = control.get("numerical_acceptance", {})
    required = {
        "schema_version": 1,
        "authority_pdf_sha256": AUTHORITY_PDF_SHA256,
        "classification": "controlled_kaizen_style_single_threshold_baseline",
        "method": "Kaizen",
        "hardware_pair": KAIZEN_PAIR,
        "proof_system": "Groth16",
    }
    if any(control.get(key) != value for key, value in required.items()):
        raise ValueError("controlled Kaizen Table 11 configuration is invalid")
    if (
        numerical.get("profile_id") != "kaizen_single_threshold_v1"
        or numerical.get("decision") != "groth16_shared_single_tolerance"
        or float(numerical.get("single_tolerance", -1.0)) != 1e-3
    ):
        raise ValueError("controlled Kaizen numerical profile is invalid")
    return control, hashlib.sha256(raw).hexdigest()


def profile_for_pair(root: Path, hardware_pair: str) -> CrossHardwareProfile:
    if hardware_pair in POLBFL_PAIRS:
        return CrossHardwareProfile(
            profile_id="polbfl_dual_threshold_v1",
            method="PoLBFL",
            hardware_pair=hardware_pair,
            pair_tolerance=1e-5,
            final_tolerance=1e-3,
            decision="groth16_pair_and_final_tolerance",
        )
    if hardware_pair == KAIZEN_PAIR:
        control, digest = _load_kaizen_control(root)
        numerical = control["numerical_acceptance"]
        return CrossHardwareProfile(
            profile_id=str(numerical["profile_id"]),
            method="Kaizen",
            hardware_pair=hardware_pair,
            pair_tolerance=float(numerical["single_tolerance"]),
            final_tolerance=float(numerical["single_tolerance"]),
            decision=str(numerical["decision"]),
            configuration_sha256=digest,
        )
    raise ValueError("unknown Table 11 hardware pair: " + hardware_pair)


def evaluate_numerical_probe(
    profile: CrossHardwareProfile,
    probe: Mapping[str, Any],
) -> dict[str, Any]:
    maximum_error = float(probe["maximum_absolute_error"])
    final_passed = maximum_error <= profile.final_tolerance
    checks = {"final_tolerance": final_passed}
    if profile.method == "PoLBFL":
        checks["pair_tolerance_bound_in_groth16"] = (
            profile.pair_tolerance == 1e-5
            and profile.pair_tolerance != profile.final_tolerance
        )
    else:
        checks["single_threshold_only"] = (
            profile.pair_tolerance == profile.final_tolerance == 1e-3
        )
    return {
        "profile": profile.to_dict(),
        "maximum_absolute_error": maximum_error,
        "passed": all(checks.values()),
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
    }


__all__ = [
    "AUTHORITY_PDF_SHA256",
    "KAIZEN_CONFIG",
    "KAIZEN_PAIR",
    "POLBFL_PAIRS",
    "CrossHardwareProfile",
    "evaluate_numerical_probe",
    "profile_for_pair",
]

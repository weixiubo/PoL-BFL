import json
from copy import deepcopy
from pathlib import Path

from experiments.final.validate_targets import validate


ROOT = Path(__file__).parents[1]


def _targets():
    return json.loads((ROOT / "config" / "paper_targets.json").read_text(encoding="utf-8"))


def test_exact_paper_values_pass_directional_validator():
    targets = _targets()
    observed = {key: deepcopy(value) for key, value in targets.items() if key.startswith(("table_", "figure_"))}
    report = validate(observed, targets)
    assert report["passed"], report["failed"]


def test_degraded_or_missing_security_metric_fails():
    targets = _targets()
    observed = {"table_2_pol_bfl": deepcopy(targets["table_2_pol_bfl"])}
    observed["table_2_pol_bfl"]["CIFAR10"]["ALIE"]["DR"] = 80.0
    report = validate(observed, targets, tables={"table_2_pol_bfl"})
    assert not report["passed"]
    assert report["failed"][0]["path"].endswith("CIFAR10.ALIE.DR")

    del observed["table_2_pol_bfl"]["CIFAR10"]["ALIE"]["DR"]
    report = validate(observed, targets, tables={"table_2_pol_bfl"})
    assert not report["passed"]
    assert any(path.endswith("CIFAR10.ALIE.DR") for path in report["missing"])

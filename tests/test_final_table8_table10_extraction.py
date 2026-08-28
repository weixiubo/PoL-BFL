import json
from pathlib import Path

import pytest

from experiments.final.run_security_cell import load_acceptance_targets
from scripts.extract_table8_targets import parse_table8
from scripts.extract_table10_targets import parse_table10


ROOT = Path(__file__).parents[1]


def test_table8_extractor_reads_all_client_counts_and_metrics():
    text = "\n".join(
        [
            "Table 8: Scalability with Increasing Clients on CIFAR-10.",
            " Time/round (s) 78.5 152.8 298.5 +94.6% +95.3%",
            " Comm (MB/round) 178.2 348.5 685.2 +95.6% +96.6%",
            " Gas (USD/round) 0.85 1.68 3.32 +97.6% +97.6%",
            " Time/client (s) 1.57 1.53 1.49 -2.5% -2.6%",
            " MA (%) 86.8 85.5 84.2 -1.5% -1.5%",
            " DR (%) 96.5 95.8 94.5 -0.7% -1.4%",
            " FPR (%) 2.1 2.3 2.8 +0.2% +0.5%",
            "Figure 3: Reputation evolution on CIFAR-10.",
        ]
    )
    table = parse_table8(text)["table_8_scalability"]
    assert table["50"]["runtime_seconds"] == 78.5
    assert table["100"]["communication_mb"] == 348.5
    assert table["200"]["DR"] == 94.5


def test_table10_extractor_reads_all_variants_and_cost_ratios():
    text = "\n".join(
        [
            "Table 10: Adaptive Attacker Evaluation on CIFAR-10.",
            " Baseline (NT) 96.5 2.1 - No",
            " Checkpoint Interpolation 94.2 2.8 1.8x No",
            " Gradient Mimicry 91.5 3.2 2.3x No",
            " Partial Replay 88.8 3.5 1.2x No",
            " Combined Adaptive 85.2 4.0 2.8x No",
            "A.1.5 Cross-Hardware Verification Robustness.",
        ]
    )
    table = parse_table10(text)["table_10_adaptive"]
    assert table["BaselineNT"] == {"DR": 96.5, "FPR": 2.1, "profitable": False}
    assert table["CheckpointInterpolation"]["forge_train_ratio"] == 1.8
    assert table["CombinedAdaptive"]["DR"] == 85.2


def test_table10_extractor_fails_closed_on_missing_ratio():
    text = "\n".join(
        [
            "Table 10: Adaptive Attacker Evaluation on CIFAR-10.",
            " Baseline (NT) 96.5 2.1 - No",
            " Checkpoint Interpolation 94.2 2.8 No",
            " Gradient Mimicry 91.5 3.2 2.3x No",
            " Partial Replay 88.8 3.5 1.2x No",
            " Combined Adaptive 85.2 4.0 2.8x No",
            "A.1.5 Cross-Hardware Verification Robustness.",
        ]
    )
    with pytest.raises(ValueError, match="invalid metric count"):
        parse_table10(text)


def test_dedicated_table8_and_table10_targets_match_main_targets():
    main = json.loads((ROOT / "config" / "paper_targets.json").read_text(encoding="utf-8"))
    table8 = json.loads(
        (ROOT / "config" / "paper_table8_scalability.json").read_text(encoding="utf-8")
    )
    table10 = json.loads(
        (ROOT / "config" / "paper_table10_adaptive.json").read_text(encoding="utf-8")
    )
    assert table8["table_8_scalability"] == main["table_8_scalability"]
    assert table10["table_10_adaptive"] == main["table_10_adaptive"]
    assert load_acceptance_targets(ROOT)["table_8_scalability"] == table8["table_8_scalability"]

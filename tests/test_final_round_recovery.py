import json

import pytest

from experiments.final.recovery import (
    align_round_log,
    discard_uncommitted_scratch,
    reset_preempted_fresh_run,
)


def test_round_recovery_atomically_drops_only_uncheckpointed_tail(tmp_path):
    path = tmp_path / "rounds.jsonl"
    path.write_text(
        "".join(json.dumps({"round": index, "value": index}) + "\n" for index in range(4)),
        encoding="utf-8",
    )
    event = align_round_log(path, checkpoint_round=2)
    assert event["retained_rounds"] == 3
    assert event["dropped_rounds"] == 1
    assert len(event["dropped_sha256"]) == 1
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["round"] for row in rows] == [0, 1, 2]
    assert align_round_log(path, checkpoint_round=2)["dropped_rounds"] == 0


def test_round_recovery_rejects_gaps_or_checkpoint_ahead_of_log(tmp_path):
    path = tmp_path / "rounds.jsonl"
    path.write_text('{"round":0}\n{"round":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="contiguous"):
        align_round_log(path, checkpoint_round=0)
    path.write_text('{"round":0}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="before the checkpoint"):
        align_round_log(path, checkpoint_round=1)


def test_scratch_recovery_removes_only_rounds_at_or_after_next_round(tmp_path):
    scratch = tmp_path / "scratch"
    for index in range(4):
        round_dir = scratch / f"round-{index}"
        round_dir.mkdir(parents=True)
        (round_dir / "artifact.bin").write_bytes(bytes([index]))
    event = discard_uncommitted_scratch(tmp_path, next_round=2)
    assert event == {
        "next_round": 2,
        "removed_scratch_rounds": ["round-2", "round-3"],
        "retained_scratch_rounds": ["round-0", "round-1"],
    }
    assert sorted(path.name for path in scratch.iterdir()) == ["round-0", "round-1"]


def test_scratch_recovery_rejects_unexpected_entries(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "not-a-round").mkdir()
    with pytest.raises(RuntimeError, match="unexpected scratch entry"):
        discard_uncommitted_scratch(tmp_path, next_round=0)


def test_preempted_fresh_run_reset_drops_log_and_all_scratch(tmp_path):
    (tmp_path / "rounds.jsonl").write_text('{"round":0}\n', encoding="utf-8")
    (tmp_path / "scratch" / "round-0").mkdir(parents=True)
    event = reset_preempted_fresh_run(tmp_path)
    assert event["round_log"]["dropped_rounds"] == 1
    assert event["scratch"]["removed_scratch_rounds"] == ["round-0"]
    assert (tmp_path / "rounds.jsonl").read_text(encoding="utf-8") == ""

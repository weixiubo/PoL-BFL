import numpy as np
import pytest

from polbfl.aggregation import (
    AggregationMethod,
    VerifiedUpdate,
    aggregate_verified_updates,
    screen_update_outliers,
)
from polbfl.sybil import TraceFingerprint, screen_trace_fingerprints


def _update(client, value, *, reputation=1.0, proof=True, sybil=False):
    return VerifiedUpdate(
        client,
        {"weight": np.asarray([value], dtype=np.float32)},
        reputation,
        proof_eligible=proof,
        sybil_flagged=sybil,
    )


def test_reputation_weighted_trimmed_mean_filters_extremes_after_layer_one():
    result = aggregate_verified_updates(
        [
            _update("a", 0.9),
            _update("b", 1.0),
            _update("c", 1.1),
            _update("d", 100.0),
            _update("e", -100.0),
            _update("proof-fail", 999.0, proof=False),
            _update("sybil", 999.0, sybil=True),
        ],
        method=AggregationMethod.TRIMMED_MEAN,
        byzantine_bound=1,
    )
    assert float(result.update["weight"][0]) == pytest.approx(1.0)
    assert result.included_clients == ("a", "b", "c", "d", "e")
    assert result.excluded_clients["proof-fail"] == "proof_ineligible"
    assert result.excluded_clients["sybil"] == "sybil_screened"


def test_coordinate_median_and_krum_use_reputation_weighted_vectors():
    median = aggregate_verified_updates(
        [_update("a", 2.0, reputation=0.5), _update("b", 1.0), _update("c", 9.0)],
        method="median",
    )
    assert float(median.update["weight"][0]) == pytest.approx(1.0)

    krum = aggregate_verified_updates(
        [_update("a", 0.0), _update("b", 0.1), _update("c", -0.1), _update("d", 0.2), _update("x", 10.0)],
        method="krum",
        byzantine_bound=1,
    )
    assert krum.krum_winner in {"a", "b", "c", "d"}
    assert krum.included_clients == (krum.krum_winner,)


def test_torch_aggregation_backend_matches_numpy_on_cpu():
    torch = pytest.importorskip("torch")
    updates = [
        VerifiedUpdate(
            f"client-{index}",
            {"weight": torch.tensor([value, -value], dtype=torch.float32)},
            1.0,
        )
        for index, value in enumerate((-100.0, 0.9, 1.0, 1.1, 100.0))
    ]
    expected = aggregate_verified_updates(
        updates,
        method="trimmed_mean",
        byzantine_bound=1,
    )
    observed = aggregate_verified_updates(
        updates,
        method="trimmed_mean",
        byzantine_bound=1,
        device="cpu",
    )
    assert torch.equal(observed.update["weight"], expected.update["weight"])


def test_nonfloating_model_buffers_are_not_robust_aggregation_coordinates():
    updates = [
        VerifiedUpdate(
            f"client-{index}",
            {
                "weight": np.asarray([float(index)], dtype=np.float32),
                "num_batches_tracked": np.asarray(index, dtype=np.int64),
            },
            1.0,
        )
        for index in range(3)
    ]
    result = aggregate_verified_updates(updates, method="median")
    assert float(result.update["weight"][0]) == 1.0
    assert "num_batches_tracked" not in result.update


def test_robust_screening_flags_clear_update_outlier():
    report = screen_update_outliers(
        [
            ("a", {"w": np.asarray([0.0, 0.1, -0.1], dtype=np.float32)}),
            ("b", {"w": np.asarray([0.1, 0.0, -0.1], dtype=np.float32)}),
            ("c", {"w": np.asarray([-0.1, 0.1, 0.0], dtype=np.float32)}),
            ("attacker", {"w": np.asarray([100.0, -100.0, 100.0], dtype=np.float32)}),
        ],
        randomness=b"r" * 32,
        coordinate_sample=3,
        mad_multiplier=3.5,
    )
    assert report.flagged_clients == frozenset({"attacker"})


def test_aggregation_rejects_nonfinite_and_unsafe_krum_parameters():
    with pytest.raises(ValueError, match="finite"):
        aggregate_verified_updates([_update("bad", np.nan)], method="median")
    with pytest.raises(ValueError, match="Krum requires"):
        aggregate_verified_updates(
            [_update("a", 0), _update("b", 1), _update("c", 2), _update("d", 3)],
            method="krum",
            byzantine_bound=1,
        )


def test_sybil_screening_uses_checkpoint_trajectory_or_identical_indices():
    fingerprints = [
        TraceFingerprint(
            "a",
            "a" * 64,
            ((0.0, 0.0), (1.0, 2.0), (2.0, 4.0)),
            (1, 2, 3, 4),
        ),
        TraceFingerprint(
            "b",
            "b" * 64,
            ((10.0, 10.0), (11.0, 12.0), (12.0, 14.0)),
            (5, 6, 7, 8),
        ),
        TraceFingerprint(
            "c",
            "c" * 64,
            ((0.0, 0.0), (-2.0, 1.0), (-1.0, -3.0)),
            (1, 2, 3, 4),
        ),
    ]
    report = screen_trace_fingerprints(fingerprints, trajectory_cosine_threshold=0.995)
    assert report.flagged_clients == frozenset({"a", "b", "c"})
    pair_reasons = {
        (pair.left_client, pair.right_client): pair.reasons for pair in report.pairs
    }
    assert "checkpoint_trajectory_similarity" in pair_reasons[("a", "b")]
    assert "identical_batch_index_sequence" in pair_reasons[("a", "c")]


def test_sybil_screening_does_not_treat_empty_index_sequences_as_duplicates():
    report = screen_trace_fingerprints(
        [
            TraceFingerprint("a", "a" * 64, ((0.0,), (1.0,)), ()),
            TraceFingerprint("b", "b" * 64, ((0.0,), (-1.0,)), ()),
        ],
        trajectory_cosine_threshold=0.99,
    )
    assert not report.flagged_clients


def test_sybil_screening_resamples_different_checkpoint_counts():
    report = screen_trace_fingerprints(
        [
            TraceFingerprint(
                "a",
                "a" * 64,
                ((0.0, 0.0), (1.0, 2.0), (2.0, 4.0)),
                (1, 2, 3),
            ),
            TraceFingerprint(
                "b",
                "b" * 64,
                ((10.0, 10.0), (12.0, 14.0)),
                (4, 5, 6),
            ),
            TraceFingerprint(
                "incompatible-width",
                "c" * 64,
                ((0.0,), (1.0,)),
                (7, 8, 9),
            ),
        ],
        trajectory_cosine_threshold=0.995,
    )
    assert report.flagged_clients == frozenset({"a", "b"})
    pairs = {
        (pair.left_client, pair.right_client): pair for pair in report.pairs
    }
    assert pairs[("a", "b")].trajectory_cosine == pytest.approx(1.0)
    assert "checkpoint_trajectory_similarity" in pairs[("a", "b")].reasons
    assert pairs[("a", "incompatible-width")].trajectory_cosine == 0.0

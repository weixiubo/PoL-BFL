import torch

from server.pol.SybilDetector import SybilDetector


def _response(indices, value):
    return {
        "data_indices": list(indices),
        "checkpoints": [
            {"data": {"model_state": {"w": torch.tensor([0.0])}}},
            {"data": {"model_state": {"w": torch.tensor([float(value)])}}},
        ],
    }


def test_sybil_detector_flags_full_duplicate_index_component():
    detector = SybilDetector(allow_trajectory_only=False)
    responses = {
        "client_1": _response([1, 2, 3], 1.0),
        "client_2": _response([1, 2, 3], 1.0),
        "client_9": _response([9, 10, 11], 1.0),
    }

    suspects = detector.detect(responses, {})

    assert set(suspects) == {"client_1", "client_2"}


def test_sybil_detector_does_not_use_trajectory_only_by_default():
    detector = SybilDetector(allow_trajectory_only=False)
    responses = {
        "client_1": _response([1, 2, 3], 1.0),
        "client_2": _response([4, 5, 6], 1.0),
    }

    suspects = detector.detect(responses, {})

    assert suspects == {}


def test_sybil_detector_uses_natural_client_order_for_soft_evidence():
    detector = SybilDetector(allow_trajectory_only=True)
    responses = {
        "client_9": _response([1, 2, 3], 1.0),
        "client_10": _response([4, 5, 6], 1.0),
    }

    suspects = detector.detect(responses, {})

    assert set(suspects) == {"client_10"}


def test_sybil_detector_remembers_hard_index_fingerprint_across_rounds():
    detector = SybilDetector(allow_trajectory_only=False)
    first_round = {
        "client_4": _response([10, 11, 12], 1.0),
        "client_26": _response([10, 11, 12], 1.0),
        "client_7": _response([70, 71, 72], 1.0),
    }

    first_suspects = detector.detect(first_round, {})

    assert set(first_suspects) == {"client_4", "client_26"}

    later_round = {
        "client_46": _response([10, 11, 12], 0.5),
        "client_8": _response([80, 81, 82], 0.5),
    }

    later_suspects = detector.detect(later_round, {})

    assert set(later_suspects) == {"client_46"}
    assert "matches_known_sybil_data_index_jaccard=1.0000" in later_suspects["client_46"]

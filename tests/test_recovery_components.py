import json

import torch

from experiments.attacks.byzantine_attacks import RandomNoiseAttack
from experiments.scripts.utils.baselines import create_aggregator
from experiments.scripts.utils.data_utils import LEAFFEMNISTDataset, partition_data_by_user
from experiments.scripts.utils.models import create_model
from server.aggregation_alg.PoLVerifyAggregator import PoLVerifyAggregator
from server.committee.VerifierNode import _strict_replay_verify
from server.pol.SybilDetector import SybilDetector


def test_resnet34_forward_shape():
    model = create_model("ResNet34", num_classes=100, input_channels=3)
    out = model(torch.randn(2, 3, 32, 32))
    assert tuple(out.shape) == (2, 100)


def test_sdea_filters_far_update():
    models = [
        {"w": torch.tensor([0.0])},
        {"w": torch.tensor([0.2])},
        {"w": torch.tensor([50.0])},
    ]
    agg = create_aggregator("SDEA", num_byzantine=1)
    out = agg.aggregate(models)
    assert sorted(agg.selected_indices) == [0, 1]
    assert torch.allclose(out["w"], torch.tensor([0.1]))


def test_random_noise_attack_uses_parameter_scale_not_unit_replacement():
    state = {
        "w": torch.full((128,), 0.01),
        "bn.running_var": torch.ones(8),
        "bn.num_batches_tracked": torch.tensor(5, dtype=torch.long),
    }

    torch.manual_seed(0)
    attacked = RandomNoiseAttack(noise_scale=1.0).apply(state, global_model=state)

    assert attacked["w"].abs().max() < 0.05
    assert torch.all(attacked["bn.running_var"] > 0.0)
    assert attacked["bn.num_batches_tracked"].item() == 5


def test_leaf_femnist_loader_and_writer_partition(tmp_path):
    train_dir = tmp_path / "FEMNIST" / "data" / "train"
    train_dir.mkdir(parents=True)
    payload = {
        "users": ["writer_a", "writer_b"],
        "user_data": {
            "writer_a": {"x": [[0.0] * 784, [1.0] * 784], "y": [0, 1]},
            "writer_b": {"x": [[0.5] * 784], "y": [61]},
        },
    }
    (train_dir / "all_data.json").write_text(json.dumps(payload), encoding="utf-8")

    ds = LEAFFEMNISTDataset(str(tmp_path / "FEMNIST"), train=True)
    assert len(ds) == 3
    x, y = ds[2]
    assert tuple(x.shape) == (1, 28, 28)
    assert y == 61

    parts = partition_data_by_user(ds, num_clients=2)
    assert sorted(len(part) for part in parts) == [1, 2]


def test_sybil_detector_flags_duplicate_evidence():
    ckpt0 = {"data": {"model_state": {"w": torch.tensor([0.0, 1.0])}}}
    ckpt1 = {"data": {"model_state": {"w": torch.tensor([0.1, 1.1])}}}
    responses = {
        "client_a": {"data_indices": [1, 2, 3], "checkpoints": [ckpt0, ckpt1]},
        "client_b": {"data_indices": [1, 2, 3], "checkpoints": [ckpt0, ckpt1]},
    }
    commits = {
        "client_a": {"commitment": "root_a", "data_hash": "same_hash"},
        "client_b": {"commitment": "root_b", "data_hash": "same_hash"},
    }
    suspects = SybilDetector().detect(responses, commits)
    assert "client_b" in suspects
    assert any("duplicate_data_hash" in reason for reason in suspects["client_b"])


def test_sybil_detector_does_not_flag_trajectory_only_by_default():
    ckpt0 = {"data": {"model_state": {"w": torch.tensor([0.0, 1.0])}}}
    ckpt1 = {"data": {"model_state": {"w": torch.tensor([0.1, 1.1])}}}
    responses = {
        "client_a": {"data_indices": [1, 2, 3], "checkpoints": [ckpt0, ckpt1]},
        "client_b": {"data_indices": [4, 5, 6], "checkpoints": [ckpt0, ckpt1]},
    }

    suspects = SybilDetector(trajectory_cosine_threshold=0.99).detect(responses, {})

    assert suspects == {}


def test_sybil_detector_can_allow_trajectory_only_for_sybil_context():
    ckpt0 = {"data": {"model_state": {"w": torch.tensor([0.0, 1.0])}}}
    ckpt1 = {"data": {"model_state": {"w": torch.tensor([0.1, 1.1])}}}
    responses = {
        "client_a": {"data_indices": [1, 2, 3], "checkpoints": [ckpt0, ckpt1]},
        "client_b": {"data_indices": [4, 5, 6], "checkpoints": [ckpt0, ckpt1]},
    }

    suspects = SybilDetector(
        trajectory_cosine_threshold=0.99,
        allow_trajectory_only=True,
    ).detect(responses, {})

    assert "client_b" in suspects


def test_remote_strict_replay_rejects_missing_context():
    result = _strict_replay_verify(
        response={"checkpoints": [{}, {}]},
        commitment="root",
        pair_indices=[0],
        params={},
        train_meta={},
        payload={},
    )
    assert result["valid"] is False
    assert result["mode"] == "strict_replay"
    assert result["reason"] == "missing_strict_model"


def test_selected_pair_challenge_keeps_final_checkpoint(monkeypatch):
    monkeypatch.setenv("POL_CHALLENGE_SELECTED_PAIRS", "1")
    monkeypatch.setenv("POL_ALWAYS_VERIFY_LAST_K", "1")
    monkeypatch.setenv("POL_RANDOM_Q", "0")
    agg = PoLVerifyAggregator(model=torch.nn.Linear(1, 1), args={"enable_pol": False})

    selected_steps, pair_indices = agg._preselect_challenge_steps([5, 10, 15, 20, 25])

    assert selected_steps == [20, 25]
    assert pair_indices == [0]


def test_aggregator_release_round_payloads_clears_heavy_refs():
    agg = PoLVerifyAggregator(model=torch.nn.Linear(1, 1), args={"enable_pol": False})
    client = object()
    agg.model_pool = [{"w": torch.ones(2)}]
    agg._client_pool = [client]
    agg._pol_commitments = {"client_1": {"commitment": "root"}}
    agg._receipts_by_client = {"client_1": [{"valid": True}]}
    agg._filtered_indices = [0]

    agg.release_round_payloads()

    assert agg.model_pool == []
    assert agg._client_pool == []
    assert agg._pol_commitments == {}
    assert agg._receipts_by_client == {}
    assert agg._filtered_indices == []

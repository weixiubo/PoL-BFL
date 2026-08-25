import pytest

torch = pytest.importorskip("torch")

from client.trainer.ProtocolPoLTrainer import ProtocolPoLTrainer
from polbfl.protocol import HybridChallengeSampler
from polbfl.verification import StrictTraceVerifier
from polbfl.verification.torch_replay import TorchSGDReplay, TorchSGDReplayConfig


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]])
        self.y = torch.tensor([0, 1, 1, 0])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index], index


def _model():
    return torch.nn.Linear(2, 2, bias=False)


def test_protocol_trainer_produces_verifiable_round(tmp_path):
    torch.manual_seed(19)
    model = _model()
    loader = torch.utils.data.DataLoader(IndexedDataset(), batch_size=2, shuffle=False)
    trainer = ProtocolPoLTrainer(
        model,
        loader,
        torch.nn.CrossEntropyLoss(),
        args={
            "enable_pol": True,
            "client_id": "client-1",
            "round_num": 3,
            "model_id": "linear",
            "device": "cpu",
            "optimizer": "SGD",
            "lr": 0.1,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "pol_save_freq": 2,
            "pol_save_dir": str(tmp_path),
            "pol_compress": True,
            "round_randomness": "ab" * 32,
            "gradient_sample_rate": 0.01,
            "allow_private_replay": True,
            "enable_zkp": False,
            "clip_norm": 5.0,
        },
    )
    results = trainer.train(1)
    assert len(results) == 1 and results[0]["batch_count"] == 2
    commitment = trainer.finalize_pol(epoch=0)
    assert commitment["num_checkpoints"] == 2
    assert commitment["total_steps"] == 2
    assert len(commitment["data_hash"]) == 64

    trace = trainer.recorded_trace.trace
    challenge = HybridChallengeSampler(recent_pairs=1, random_pairs=0).sample(
        trace.commitment,
        vrf_output=b"v" * 32,
        issued_at_ns=10,
        deadline_ns=20,
        proof_mode="strict_replay",
    )
    response = trainer.respond_to_challenge(challenge)
    replay = TorchSGDReplay(
        TorchSGDReplayConfig(
            model_factory=_model,
            criterion_factory=torch.nn.CrossEntropyLoss,
            momentum=0.0,
            weight_decay=0.0,
        )
    )
    report = StrictTraceVerifier(pair_tolerance=1e-12, final_tolerance=1e-12).verify(
        context=trace.context,
        challenge=challenge,
        response=response,
        replay_interval=replay,
    )
    assert report.valid, report.reasons

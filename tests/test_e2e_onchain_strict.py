import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from chainfl.interact import chain_proxy
from server.aggregation_alg.PoLVerifyAggregator import PoLVerifyAggregator
from client.trainer.PoLTrainer import PoLTrainer


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def make_trainer(client_id: str, data_slice: slice):
    x = torch.randn(40, 1, 28, 28)
    y = torch.randint(0, 10, (40,))
    ds = TensorDataset(x[data_slice], y[data_slice])
    dl = DataLoader(ds, batch_size=10, shuffle=False)
    model = SimpleCNN()
    args = {
        'enable_pol': True,
        'pol_save_freq': 1,
        'pol_save_dir': f'/tmp/pol_data_strict_{client_id}',
        'pol_compress': True,
        'client_id': client_id,
        'device': 'cpu',
        'lr': 0.01,
        'weight_decay': 1e-4,
        'optimizer': 'SGD',
        'enable_zkp': True,
        'zkp_use_simulation': True,
    }
    trainer = PoLTrainer(model, dl, nn.CrossEntropyLoss(), args)
    return trainer, ds


@pytest.mark.order("last")
def test_e2e_onchain_strict_without_offline_fallback():
    # Ensure no offline fallback
    import os
    if os.environ.get('POL_OFFLINE_FALLBACK') == '1':
        del os.environ['POL_OFFLINE_FALLBACK']

    # Prepare two clients
    t1, ds1 = make_trainer('1', slice(0, 20))
    t2, ds2 = make_trainer('2', slice(20, 40))

    # Minimal training and PoL commitment
    t1.train(total_epoch=1)
    t1.finalize_pol(epoch=0, dataset=ds1)
    t2.train(total_epoch=1)
    t2.finalize_pol(epoch=0, dataset=ds2)

    # Build lightweight client-like wrappers
    class _Client:
        def __init__(self, cid, trainer, dl):
            self.client_id = cid
            self.trainer = trainer
            self.dataloader = dl
            self.args = trainer.args
            self.model = trainer.model
        def get_model_state_dict(self):
            return self.model.state_dict()
        def respond_to_challenge(self, challenge):
            return self.trainer.respond_to_challenge(challenge)

    c1 = _Client('1', t1, t1.dataloader)
    c2 = _Client('2', t2, t2.dataloader)

    agg_args = {
        'enable_pol': True,
        'verification_rate': 1.0,
        'pol_delta': 10.0,
        'pol_distance_metric': 'l2',
        'device': 'cpu',
        'use_top_q': False,
        'enable_zkp': True,
        'zkp_use_simulation': True,
        'zkp_vkey_path': 'circuits/build/parameter_update.vkey.json',
    }
    agg = PoLVerifyAggregator(model=SimpleCNN(), args=agg_args)

    # Trigger verify + aggregate
    agg.receive_upload([c1, c2])
    agg.aggregate([c1.get_model_state_dict(), c2.get_model_state_dict()])

    # Assert on-chain challenge recorded (no offline ids)
    issued = [cid for cid in getattr(chain_proxy, '_issued_challenges', []) if isinstance(cid, str)]
    assert issued, "No challenges were issued"
    assert any(cid.startswith('0x') for cid in issued), f"Non on-chain challenge ids: {issued}"

    # Resolve and verify via chain call (on-chain path)
    for cid_hex in issued:
        if not cid_hex.startswith('0x'):
            continue
        ch = chain_proxy.get_challenge(cid_hex)
        assert ch and ch.get('resolved') is True
        assert ch.get('success') is True
        # public signals returned by contract are ints
        assert isinstance(ch.get('W_t_hash', 0), int)
        assert isinstance(ch.get('W_t1_hash', 0), int)
        assert isinstance(ch.get('data_hash', 0), int)


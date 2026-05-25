import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy

from server.pol.PoLVerifier import PoLVerifier


def make_model(seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    return model


def make_data(n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 10, generator=g)
    y = torch.randint(0, 2, (n,), generator=g)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    return ds, dl


def train_one_step(model, ds, indices, lr=0.1, seed=0):
    torch.manual_seed(seed)
    model = deepcopy(model)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    X = torch.stack([ds[i][0] for i in indices])
    y = torch.stack([ds[i][1] for i in indices])
    opt.zero_grad()
    out = model(X)
    loss = crit(out, y)
    loss.backward()
    opt.step()
    return model.state_dict(), opt.state_dict()


def checkpoint_from(model, opt_state):
    return {
        'data': {
            'model_state': deepcopy(model.state_dict()),
            'optimizer_state': deepcopy(opt_state),
            'epoch': 0,
            'step': 0,
            'loss': 0.0,
        }
    }


def test_verify_single_step_with_correct_indices():
    # Setup
    args = {'delta': 1e-6, 'distance_metric': 'l2', 'device': 'cpu'}
    verifier = PoLVerifier(args)
    base_model = make_model(seed=42)
    ds, dl = make_data(n=32, seed=7)
    indices = [0, 1, 2, 3]

    # Build current checkpoint (before step)
    opt0 = torch.optim.SGD(base_model.parameters(), lr=0.1)
    current = checkpoint_from(base_model, opt0.state_dict())

    # Build next checkpoint by doing exactly one step on "indices"
    next_state, _ = train_one_step(base_model, ds, indices, lr=0.1, seed=99)
    next_model = make_model(seed=42)
    next_model.load_state_dict(next_state)
    next_ckpt = {'data': {'model_state': next_model.state_dict(), 'optimizer_state': opt0.state_dict(), 'epoch': 0, 'step': 1, 'loss': 0.0}}

    # Verify with exact indices should pass (distance ~ 0)
    ok = verifier._verify_single_step(
        current_ckpt=current,
        next_ckpt=next_ckpt,
        model=make_model(seed=42),
        dataloader=dl,
        criterion=nn.CrossEntropyLoss(),
        optimizer_class=torch.optim.SGD,
        lr=0.1,
        data_indices=indices,
    )
    assert ok is True


def test_verify_single_step_with_wrong_indices():
    # Setup
    args = {'delta': 1e-9, 'distance_metric': 'l2', 'device': 'cpu'}
    verifier = PoLVerifier(args)
    base_model = make_model(seed=13)
    ds, dl = make_data(n=32, seed=11)
    indices_true = [0, 1, 2, 3]
    indices_wrong = [4, 5, 6, 7]

    # Current checkpoint
    opt0 = torch.optim.SGD(base_model.parameters(), lr=0.05)
    current = checkpoint_from(base_model, opt0.state_dict())

    # Next checkpoint computed on different indices
    next_state, _ = train_one_step(base_model, ds, indices_wrong, lr=0.05, seed=101)
    next_model = make_model(seed=13)
    next_model.load_state_dict(next_state)
    next_ckpt = {'data': {'model_state': next_model.state_dict(), 'optimizer_state': opt0.state_dict(), 'epoch': 0, 'step': 1, 'loss': 0.0}}

    # Verify with true indices under tiny delta should fail
    ok = verifier._verify_single_step(
        current_ckpt=current,
        next_ckpt=next_ckpt,
        model=make_model(seed=13),
        dataloader=dl,
        criterion=nn.CrossEntropyLoss(),
        optimizer_class=torch.optim.SGD,
        lr=0.05,
        data_indices=indices_true,
    )
    assert ok is False


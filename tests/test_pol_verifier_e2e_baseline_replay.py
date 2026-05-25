import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy

from server.pol.PoLVerifier import PoLVerifier
from client.pol.MerkleTree import MerkleTree


def make_model(seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    return model


def make_data(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 10, generator=g)
    y = torch.randint(0, 2, (n,), generator=g)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=16, shuffle=False)
    return ds, dl


def train_one_step(model, ds, indices, lr=0.1):
    # Single SGD step (no momentum) to match verifier replay
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
    return deepcopy(model.state_dict()), deepcopy(opt.state_dict())


def checkpoint_from(model_state, opt_state, step):
    return {
        'data': {
            'model_state': deepcopy(model_state),
            'optimizer_state': deepcopy(opt_state),
            'epoch': 0,
            'step': step,
            'loss': 0.0,
        }
    }


def test_e2e_baseline_replay_with_data_indices_and_merkle_membership():
    # Verifier configured for strict equality (small delta) and CPU
    args = {'delta': 1e-9, 'distance_metric': 'l2', 'device': 'cpu'}
    verifier = PoLVerifier(args)

    # Model/Data
    base_model = make_model(seed=123)
    ds, dl = make_data(n=64, seed=7)
    indices = [0, 1, 2, 3]  # fixed small batch
    lr = 0.1

    # Build a chain of checkpoints (current -> next) for multiple steps
    # Use plain SGD (no momentum) so optimizer state is minimal/stable
    checkpoints = []
    model_state = deepcopy(base_model.state_dict())
    opt_state = torch.optim.SGD(base_model.parameters(), lr=lr).state_dict()
    steps = 6  # gives 5 transitions; success_rate = 5/6 ≈ 0.83 < 0.9, so use >=10
    steps = 12  # 11 transitions; success_rate = 11/12 ≈ 0.92 >= 0.9

    work_model = make_model(seed=123)
    work_model.load_state_dict(model_state)
    for s in range(steps):
        ckpt = checkpoint_from(model_state, opt_state, step=s)
        checkpoints.append(ckpt)
        # produce next based on previous state
        next_state, _ = train_one_step(work_model, ds, indices, lr=lr)
        work_model.load_state_dict(next_state)
        model_state = deepcopy(next_state)
        # keep optimizer state constant (SGD no momentum)

    # Add final checkpoint
    checkpoints.append(checkpoint_from(model_state, opt_state, step=steps))

    # Build Merkle leaves consistent with verifier hashing
    leaves = [verifier._compute_checkpoint_hash(ck['data']) for ck in checkpoints]
    tree = MerkleTree(leaves)
    commitment = tree.get_root()
    for i, ck in enumerate(checkpoints):
        ck['merkle_proof'] = tree.get_proof(i)

    # Assemble response and a minimal challenge
    response = {
        'checkpoints': checkpoints,
        'data_indices': indices,
    }
    challenge = {'client_id': 'local', 'idx0': 0, 'idx1': 1}

    ok = verifier.verify_response(
        challenge=challenge,
        response=response,
        commitment=commitment,
        model=make_model(seed=123),
        dataloader=dl,
        criterion=nn.CrossEntropyLoss(),
        optimizer_class=torch.optim.SGD,
        lr=lr,
    )
    assert ok is True


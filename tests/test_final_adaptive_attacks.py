import pytest

torch = pytest.importorskip("torch")

from experiments.final.adaptive_attacks import (
    checkpoint_interpolation,
    combined_adaptive_trajectory,
    gradient_mimicry,
    partial_replay,
)
from experiments.final.adaptive_trial_support import (
    REFERENCE_TRAJECTORIES,
    adaptive_profit,
)


def test_adaptive_attacks_are_executable_and_do_not_embed_paper_ratios():
    checkpoints = (torch.tensor([0.0]), torch.tensor([2.0]), torch.tensor([4.0]))
    interpolated = checkpoint_interpolation(checkpoints)
    assert [float(value) for value in interpolated] == [0.0, 2.0, 4.0]
    replayed = partial_replay(
        tuple(torch.tensor([float(index)]) for index in range(10))
    )
    assert [float(value) for value in replayed[:3]] == [0.0, 1.0, 2.0]
    assert float(replayed[3]) == 0.0
    generator = torch.Generator().manual_seed(7)
    mimicked = gradient_mimicry(
        (torch.tensor([0.0, 2.0]), torch.tensor([2.0, 4.0])),
        generator=generator,
    )
    assert mimicked.shape == (2,)
    combined, gradient = combined_adaptive_trajectory(
        checkpoints,
        (torch.tensor([0.0]), torch.tensor([2.0])),
        generator=torch.Generator().manual_seed(9),
    )
    assert len(combined) == len(checkpoints)
    assert gradient.shape == (1,)


def test_adaptive_trial_cost_comes_from_real_reference_work_and_economics():
    assert REFERENCE_TRAJECTORIES == {
        "BaselineNT": 0,
        "CheckpointInterpolation": 3,
        "GradientMimicry": 4,
        "PartialReplay": 2,
        "CombinedAdaptive": 5,
    }
    assert adaptive_profit(
        reward=0.025,
        base_cost=0.015,
        slash=0.145,
        forge_train_ratio=3.0,
    ) < 0.0

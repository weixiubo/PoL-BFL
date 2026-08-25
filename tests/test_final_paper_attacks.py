import pytest

torch = pytest.importorskip("torch")

from experiments.final.attacks import alie_update, minmax_update, model_replacement_update


def _update(value):
    return {"w": torch.tensor([value, value + 1.0]), "counter": torch.tensor(1)}


def test_alie_uses_population_standard_deviation():
    result = alie_update([_update(0.0), _update(2.0), _update(4.0)], z_max=1.0)
    matrix = torch.stack([_update(value)["w"] for value in (0.0, 2.0, 4.0)])
    assert torch.allclose(result["w"], matrix.mean(0) + matrix.std(0, unbiased=False))


def test_minmax_stays_within_benign_diameter_and_moves_from_mean():
    benign = [_update(0.0), _update(1.0), _update(2.0), _update(3.0)]
    result = minmax_update(benign)
    matrix = torch.stack([item["w"].double() for item in benign])
    candidate = result["w"].double()
    assert torch.linalg.vector_norm(matrix - candidate, dim=1).max() <= torch.cdist(matrix, matrix).max() + 1e-5
    assert not torch.allclose(candidate, matrix.mean(0))


def test_model_replacement_is_an_amplified_delta():
    result = model_replacement_update(_update(5.0), _update(1.0), amplification=10)
    assert torch.equal(result["w"], torch.tensor([40.0, 40.0]))

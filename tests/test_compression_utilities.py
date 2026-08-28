from collections import OrderedDict

import torch

from util.compression import (
    Int8Quantizer,
    magnitude_prune,
    quantify_decode,
    quantify_encode,
)


def test_fixed_scale_quantization_is_non_mutating_and_reversible():
    state = OrderedDict(
        weight=torch.tensor([-1.25, 0.0, 1.25]),
        batches=torch.tensor(3, dtype=torch.int64),
    )
    original = state["weight"].clone()

    encoded = quantify_encode(state, scale=20.0)
    decoded = quantify_decode(encoded, scale=20.0)

    assert encoded["weight"].dtype == torch.int8
    assert torch.equal(state["weight"], original)
    assert torch.allclose(decoded["weight"], original, atol=0.025)
    assert decoded["batches"].dtype == torch.int64


def test_per_tensor_quantizer_tracks_scales_for_each_parameter():
    codec = Int8Quantizer()
    state = OrderedDict(
        first=torch.tensor([-10.0, 10.0]),
        second=torch.tensor([-0.1, 0.1]),
    )

    encoded = codec.encode(state)
    decoded = codec.decode(encoded)

    assert set(codec.scales) == set(state)
    assert codec.scales["first"] != codec.scales["second"]
    assert torch.allclose(decoded["first"], state["first"], atol=0.1)
    assert torch.allclose(decoded["second"], state["second"], atol=0.001)


def test_magnitude_pruning_removes_only_the_smallest_values():
    state = OrderedDict(weight=torch.tensor([1.0, 2.0, 3.0, 4.0]))

    pruned = magnitude_prune(state, fraction=0.5)

    assert torch.equal(pruned["weight"], torch.tensor([0.0, 0.0, 3.0, 4.0]))
    assert torch.equal(state["weight"], torch.tensor([1.0, 2.0, 3.0, 4.0]))

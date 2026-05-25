import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from zkp.hash import quantize_to_field, poseidon_fold, DEFAULT_SCALE


REPO_ROOT = Path(__file__).resolve().parents[1]
POSEIDON_JS = REPO_ROOT / 'scripts' / 'poseidon_fold.js'


@pytest.mark.skipif(not POSEIDON_JS.exists(), reason="poseidon_fold.js missing")
def test_poseidon_python_vs_node_on_small_vector():
    # Deterministic small vector with negatives and positives
    vals = torch.tensor([1.234567, -2.5, 0.0, 7.89, -0.000123], dtype=torch.float32)
    q = quantize_to_field(vals, DEFAULT_SCALE)

    # Python API (calls Node under the hood)
    h_py = poseidon_fold(q)

    # Direct Node invocation
    proc = subprocess.run(
        ['node', str(POSEIDON_JS), json.dumps(list(map(int, q)))],
        check=True, capture_output=True, text=True
    )
    h_js = proc.stdout.strip()

    assert h_py == h_js


@pytest.mark.skipif(not POSEIDON_JS.exists(), reason="poseidon_fold.js missing")
def test_poseidon_consistency_random_vectors():
    # Multiple random vectors to guard against regressions
    g = torch.Generator().manual_seed(2025)
    for n in [1, 2, 7, 31, 64, 101]:
        vals = torch.randn(n, generator=g)
        q = quantize_to_field(vals, DEFAULT_SCALE)
        h1 = poseidon_fold(q)
        proc = subprocess.run(
            ['node', str(POSEIDON_JS), json.dumps(list(map(int, q)))],
            check=True, capture_output=True, text=True
        )
        h2 = proc.stdout.strip()
        assert h1 == h2


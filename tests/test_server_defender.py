import numpy as np

from server.defence_alg.defence_util import serverDefender


def test_krum_filter_selects_from_the_consistent_client_cluster():
    defender = serverDefender(num_byzantine=1)
    updates = [
        np.array([0.0, 0.0]),
        np.array([0.1, -0.1]),
        np.array([-0.1, 0.1]),
        np.array([0.05, 0.05]),
        np.array([100.0, -100.0]),
    ]

    selected = defender.krum_filter(updates)

    assert len(selected) == 1
    assert np.linalg.norm(selected[0]) < 1.0
    assert defender.alg1(updates)[0] is selected[0]


def test_robust_norm_filter_rejects_a_large_outlier():
    defender = serverDefender(norm_multiplier=2.5)
    updates = [
        np.array([0.0, 0.0]),
        np.array([0.1, 0.0]),
        np.array([0.0, 0.1]),
        np.array([0.1, 0.1]),
        np.array([50.0, 50.0]),
    ]

    selected = defender.robust_norm_filter(updates)

    assert len(selected) == 4
    assert all(np.linalg.norm(update) < 1.0 for update in selected)
    assert defender.alg2(updates) == selected


def test_defender_rejects_invalid_krum_dimensions():
    defender = serverDefender(num_byzantine=1)
    try:
        defender.krum_filter([np.array([0.0]), np.array([1.0])])
    except ValueError as exc:
        assert "2f + 3" in str(exc)
    else:
        raise AssertionError("Krum must reject an undersized client set")

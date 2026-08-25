import os

from scripts.exclusive_gpu_controller import external_compute_pids


def test_external_gpu_pid_filter_uses_process_group(monkeypatch):
    groups = {10: 100, 11: 100, 20: 200}
    monkeypatch.setattr(os, "getpgid", lambda pid: groups[pid])
    assert external_compute_pids(
        frozenset({10, 11, 20}),
        owned_process_group=100,
    ) == frozenset({20})

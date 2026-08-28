import pytest
import torch

from client.pol.PoLManager import PoLManager


@pytest.mark.parametrize("compress", [False, True])
def test_async_checkpoint_save_is_atomic_and_immediately_loadable(
    tmp_path,
    compress,
):
    manager = PoLManager(
        client_id="async",
        save_dir=str(tmp_path),
        save_freq=1,
        compress=compress,
        async_save=True,
        save_to_disk=True,
    )
    checkpoint = {
        "model_state": {"weight": torch.arange(8, dtype=torch.float32)},
        "optimizer_state": {"step": 1},
    }

    digest = manager.save_checkpoint(1, checkpoint)
    loaded = manager.load_checkpoint(1)
    manager.close()

    assert len(digest) == 64
    assert torch.equal(loaded["model_state"]["weight"], checkpoint["model_state"]["weight"])
    suffix = ".pt.gz" if compress else ".pt"
    assert (tmp_path / "client_async" / "checkpoints" / f"ckpt_step_1{suffix}").is_file()
    assert not list((tmp_path / "client_async" / "checkpoints").glob("*.tmp-*"))


def test_flush_waits_for_all_scheduled_checkpoints(tmp_path):
    manager = PoLManager(
        client_id="flush",
        save_dir=str(tmp_path),
        save_freq=1,
        async_save=True,
        save_to_disk=True,
    )
    for step in range(4):
        manager.save_checkpoint(
            step,
            {"model_state": {"weight": torch.tensor([float(step)])}},
        )

    manager.flush_pending_saves()
    manager.close()

    checkpoint_dir = tmp_path / "client_flush" / "checkpoints"
    assert len(list(checkpoint_dir.glob("ckpt_step_*.pt.gz"))) == 4

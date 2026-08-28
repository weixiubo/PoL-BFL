import os
import shutil
import tempfile
import torch

from client.pol.PoLManager import PoLManager


def test_deferred_cleanup_flag_and_execution():
    tmpdir = tempfile.mkdtemp(prefix="pol_test_")
    try:
        save_dir = os.path.join(tmpdir, "pol_data")
        # Enable automatic cleanup with a short interval to exercise the callback.
        pm = PoLManager(
            client_id="test_client",
            save_dir=save_dir,
            save_freq=1,
            compress=False,
            save_to_disk=True,
            memory_limit=5,
            enable_auto_cleanup=True,
            auto_cleanup_interval=2,
        )

        # Initially no deferred cleanup
        assert getattr(pm, "_deferred_cleanup_pending", False) is False
        assert getattr(pm, "defer_cleanup", True) is True

        # Prepare minimal checkpoint payload
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        def make_ckpt(step):
            return {
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "epoch": 0,
                "step": step,
                "loss": 0.0,
            }

        # Save two checkpoints to hit the interval (2)
        pm.save_checkpoint(step=1, checkpoint_data=make_ckpt(1))
        assert getattr(pm, "_deferred_cleanup_pending", False) is False

        pm.save_checkpoint(step=2, checkpoint_data=make_ckpt(2))
        # Now deferred cleanup should be scheduled (but not executed yet)
        assert getattr(pm, "_deferred_cleanup_pending", False) is True

        # Execute deferred cleanup at a safe point
        pm.run_deferred_cleanup_if_any()
        assert getattr(pm, "_deferred_cleanup_pending", False) is False

        # Metadata must remain coherent (all entries point to existing files or memory)
        for meta in pm.metadata.get("checkpoints", []):
            path = meta.get("path")
            if path and path != "memory":
                assert os.path.exists(path), f"Missing checkpoint file after cleanup: {path}"

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

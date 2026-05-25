"""
Regression test: final checkpoint consistency in PoLTrainer.finalize_pol
Ensures that when the last saved checkpoint step != trainer.batch_counter,
finalize_pol will save a final checkpoint at the current batch_counter.
"""
import os
import shutil
import tempfile
import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset

from client.trainer.PoLTrainer import PoLTrainer


class TestFinalizeCheckpointConsistency(unittest.TestCase):
    def setUp(self):
        # temp dir for PoL data
        self.tmpdir = tempfile.mkdtemp(prefix="pol_final_ckpt_")
        # tiny model and data
        self.model = torch.nn.Linear(8, 3)
        x = torch.randn(4, 8)
        y = torch.randint(0, 3, (4,))
        self.loader = DataLoader(TensorDataset(x, y), batch_size=2)
        self.args = {
            'device': 'cpu',
            'lr': 0.01,
            'weight_decay': 0.0,
            'optimizer': 'SGD',
            'enable_pol': True,
            'pol_save_freq': 10,  # intentionally larger than batches to not align
            'pol_save_dir': self.tmpdir,
            'pol_compress': True,
            'client_id': 'test_client_final'
        }
        self.trainer = PoLTrainer(self.model, self.loader, torch.nn.CrossEntropyLoss(), self.args)
        # construct optimizer explicitly (we're not running train())
        self.trainer.construct_optimizer()

    def tearDown(self):
        try:
            shutil.rmtree(self.tmpdir)
        except Exception:
            pass

    def test_finalize_saves_last_checkpoint_at_batch_counter(self):
        # Arrange: create one earlier checkpoint at step 0
        self.trainer.batch_counter = 0
        self.trainer._save_checkpoint(epoch=0, batch_idx=0, loss=0.0)
        meta0 = self.trainer.pol_manager.get_metadata()
        self.assertEqual(len(meta0['checkpoints']), 1)
        self.assertEqual(meta0['checkpoints'][-1]['step'], 0)

        # Act: advance to a non-aligned final step and finalize
        final_step = 7
        self.trainer.batch_counter = final_step
        self.trainer.finalize_pol(epoch=0)

        # Assert: a new last checkpoint exists at the final step
        meta = self.trainer.pol_manager.get_metadata()
        self.assertGreaterEqual(len(meta['checkpoints']), 2)
        self.assertEqual(meta['checkpoints'][-1]['step'], final_step)

        # And metadata file is persisted on disk
        meta_path = os.path.join(self.tmpdir, f"client_{self.args['client_id']}", "metadata.json")
        self.assertTrue(os.path.exists(meta_path))


if __name__ == '__main__':
    unittest.main()


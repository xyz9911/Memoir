import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_drama_pretrain import TinyWorld, batch
import torch
from colab_full_pretrain import TrainingRows, backward_update
import drama_pretrain as pre


class FullDataTests(unittest.TestCase):
    def test_full_negative_pool_and_uneven_trajectory_normalizers(self):
        torch.set_num_threads(1)
        torch.manual_seed(177)
        full = TinyWorld().eval().double()
        micro = copy.deepcopy(full)
        data = {key: (val.double() if val.is_floating_point() else val) for key, val in batch().items()}
        # Couple stochastic latent draws between different batch layouts to isolate the objective.
        with patch.object(torch.distributions.Normal, "rsample", lambda dist: dist.mean):
            loss, pieces = pre.wm_loss(full, data, 3)
            loss.backward()
            actual, parts = backward_update(micro, data, 1, 3, {})
        self.assertAlmostEqual(loss.item(), actual, places=6)
        for key, val in pieces.items():
            self.assertAlmostEqual(val.item(), parts[key], places=6)
        for (name, p), (_, other) in zip(full.named_parameters(), micro.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(other.grad, p.grad, rtol=3e-5, atol=3e-6, msg=name)

    def test_every_instruction_and_jsonl_line_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, augmented = Path(tmp)/"human.json", Path(tmp)/"aug.jsonl"
            human.write_text(json.dumps([{"path_id": 1, "scan": "a", "heading": 0, "path": ["a","b"],
                                          "instructions": ["first","second"], "instr_encodings": [[1],[2]]}]))
            augmented.write_text('{"instr_id":"1_0","instr_encoding":[3]}\n{"instr_id":"1_0","instr_encoding":[4]}\n')
            rows = TrainingRows(human, augmented)
            try:
                self.assertEqual(len(rows), 4)
                self.assertEqual([rows[i]["instr_encoding"] for i in [2, 0, 3, 1]], [[3], [1], [4], [2]])
            finally:
                rows.augmented.close()

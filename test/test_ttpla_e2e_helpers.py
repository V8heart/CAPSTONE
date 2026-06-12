# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for TTPLA E2E GT pack, collate, and e2e_train_bridge helpers (no full dataset IO)."""
import os
import sys
import unittest

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from yolino.dataset.dataset_factory import collate_ttpla_with_optional_e2e_gt  # noqa: E402
from yolino.model.e2e_train_bridge import (  # noqa: E402
    bezier_least_squares_ctrl_xy,
    resample_polyline_xy,
)


class TestCollateE2e(unittest.TestCase):
    def test_collate_stacks_e2e_dict(self):
        b = 3
        ni, mp = 4, 8
        batch = []
        for _ in range(b):
            img = torch.randn(3, 16, 16)
            grid = torch.randn(2, 3, 4, 5)
            e2e = {
                "padded": torch.zeros(ni, mp, 2),
                "inst_mask": torch.zeros(ni, dtype=torch.bool),
                "pt_mask": torch.zeros(ni, mp, dtype=torch.bool),
            }
            batch.append((img, grid, "f", {}, {}, e2e))
        out = collate_ttpla_with_optional_e2e_gt(batch)
        self.assertEqual(len(out), 6)
        self.assertEqual(out[0].shape[0], b)
        self.assertEqual(out[5]["padded"].shape, (b, ni, mp, 2))


class TestResampleBezier(unittest.TestCase):
    def test_resample_straight_line_length(self):
        pts = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        valid = torch.tensor([True, True, True])
        out = resample_polyline_xy(pts, valid, 5)
        self.assertEqual(out.shape, (5, 2))
        self.assertTrue(torch.allclose(out[:, 1], torch.zeros(5)))

    def test_bezier_ls_ctrl_degree3(self):
        t = torch.linspace(0, 1, 10).unsqueeze(1)
        pts = torch.cat([t, t * 0.5], dim=1)
        valid = torch.ones(10, dtype=torch.bool)
        ctrl = bezier_least_squares_ctrl_xy(pts, valid, degree=3)
        self.assertEqual(ctrl.shape, (4, 2))


if __name__ == "__main__":
    unittest.main()

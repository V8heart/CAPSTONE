# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for :mod:`yolino.model.e2e_polyline_order` (exp51 GT canonicalization)."""
import math
import unittest

import torch

from yolino.model.e2e_polyline_order import (
    canonicalize_polyline_xy,
    canonicalize_polyline_yx,
    canonicalize_pack_inplace,
)


class TestPolylineOrder(unittest.TestCase):
    def test_horizontal_left_to_right(self):
        # Right-to-left input → left-to-right output.
        pts = torch.tensor([[10.0, 0.0], [5.0, 0.0], [0.0, 0.0]])
        out = canonicalize_polyline_xy(pts)
        self.assertTrue(torch.allclose(out, torch.tensor([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])))

    def test_horizontal_already_ltr(self):
        pts = torch.tensor([[0.0, 0.0], [5.0, 1.0], [10.0, 0.0]])
        out = canonicalize_polyline_xy(pts)
        self.assertTrue(torch.allclose(out, pts))

    def test_pure_vertical_top_to_bottom(self):
        # Bottom-to-top input on a pure vertical → top-to-bottom output (small y first).
        pts = torch.tensor([[0.0, 10.0], [0.0, 5.0], [0.0, 0.0]])
        out = canonicalize_polyline_xy(pts)
        self.assertTrue(torch.allclose(out, torch.tensor([[0.0, 0.0], [0.0, 5.0], [0.0, 10.0]])))

    def test_near_vertical_uses_y_sort(self):
        # Predominantly vertical: tiny horizontal jitter, large vertical range.
        pts = torch.tensor([[5.1, 0.0], [5.0, 5.0], [4.9, 10.0]])  # downward
        out = canonicalize_polyline_xy(pts)
        # y range = 10, x range = 0.2 → dx/total ≈ 0.02 < cos(80°) ≈ 0.174 → use y sort.
        # Already y-ascending, expect unchanged.
        self.assertTrue(torch.allclose(out, pts))

        pts2 = torch.flip(pts, dims=[0])
        out2 = canonicalize_polyline_xy(pts2)
        self.assertTrue(torch.allclose(out2, pts))

    def test_diagonal_uses_x_sort(self):
        # 45°: dx == dy → falls into horizontal bucket (dx_span >= dy_span guard ties).
        pts = torch.tensor([[10.0, 10.0], [5.0, 5.0], [0.0, 0.0]])
        out = canonicalize_polyline_xy(pts)
        self.assertTrue(torch.allclose(out, torch.tensor([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])))

    def test_two_point_input(self):
        pts = torch.tensor([[7.0, 0.0], [3.0, 0.0]])
        out = canonicalize_polyline_xy(pts)
        self.assertTrue(torch.allclose(out, torch.tensor([[3.0, 0.0], [7.0, 0.0]])))

    def test_single_point_returned_unchanged(self):
        pts = torch.tensor([[7.0, 7.0]])
        out = canonicalize_polyline_xy(pts)
        self.assertTrue(torch.allclose(out, pts))

    def test_yx_wrapper(self):
        # (y, x) bottom-to-top vertical → (y, x) top-to-bottom after canonicalize.
        pts_yx = torch.tensor([[10.0, 0.0], [5.0, 0.0], [0.0, 0.0]])
        out = canonicalize_polyline_yx(pts_yx)
        self.assertTrue(torch.allclose(out, torch.tensor([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])))

    def test_pack_inplace_masks_padding(self):
        ni, mp = 3, 5
        padded = torch.zeros((ni, mp, 2))
        pt_mask = torch.zeros((ni, mp), dtype=torch.bool)
        inst_mask = torch.zeros((ni,), dtype=torch.bool)

        # Instance 0: 3-pt horizontal RTL.
        padded[0, :3] = torch.tensor([[10.0, 0.0], [5.0, 0.0], [0.0, 0.0]])
        pt_mask[0, :3] = True
        inst_mask[0] = True
        # Instance 1: invalid (inst_mask False) — must not be touched.
        padded[1, :2] = torch.tensor([[9.0, 9.0], [-1.0, -1.0]])
        # Instance 2: 4-pt near-vertical bottom-to-top.
        padded[2, :4] = torch.tensor([[1.0, 0.0], [1.0, 3.0], [1.0, 7.0], [1.0, 10.0]])
        pt_mask[2, :4] = True
        inst_mask[2] = True

        canonicalize_pack_inplace(padded, pt_mask, inst_mask)

        # Inst 0 flipped to LTR.
        self.assertTrue(
            torch.allclose(padded[0, :3], torch.tensor([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]]))
        )
        # Inst 1 untouched (still original sentinel).
        self.assertTrue(torch.allclose(padded[1, 0], torch.tensor([9.0, 9.0])))
        self.assertTrue(torch.allclose(padded[1, 1], torch.tensor([-1.0, -1.0])))
        # Inst 2 already top-to-bottom (y ascending), no change.
        self.assertTrue(
            torch.allclose(
                padded[2, :4],
                torch.tensor([[1.0, 0.0], [1.0, 3.0], [1.0, 7.0], [1.0, 10.0]]),
            )
        )

    def test_canonical_idempotent(self):
        pts = torch.tensor([[3.0, 0.0], [0.0, 1.0], [7.0, 2.0]])
        once = canonicalize_polyline_xy(pts)
        twice = canonicalize_polyline_xy(once)
        self.assertTrue(torch.allclose(once, twice))

    def test_dense_polyline_preserves_arc_length(self):
        """Flip-only canonicalize must not reorder interior vertices (argsort bug)."""
        theta = torch.linspace(0.0, math.pi, 80)
        pts = torch.stack(
            [
                400.0 * torch.cos(theta) + 500.0,
                120.0 * torch.sin(theta) + 500.0,
            ],
            dim=1,
        )

        def _arc_len(p: torch.Tensor) -> float:
            d = torch.diff(p, dim=0)
            return float(torch.norm(d, p=2, dim=1).sum().item())

        before = _arc_len(pts)
        out = canonicalize_polyline_xy(pts)
        after = _arc_len(out)
        self.assertAlmostEqual(before, after, places=3)
        self.assertGreater(after, 400.0)

    def test_dense_polyline_flip_when_endpoints_reversed(self):
        theta = torch.linspace(0.0, math.pi, 40)
        pts = torch.stack(
            [400.0 * torch.cos(theta) + 200.0, 80.0 * torch.sin(theta) + 300.0],
            dim=1,
        )
        fwd = canonicalize_polyline_xy(pts)
        rev = canonicalize_polyline_xy(pts.flip(0))
        self.assertTrue(torch.allclose(fwd, rev, atol=1e-4))


if __name__ == "__main__":
    unittest.main()

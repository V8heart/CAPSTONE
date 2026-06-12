# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for :mod:`yolino.model.yolino_hough_dn` (exp51 DN / LCDN)."""
import math
import unittest

import torch

from yolino.model.yolino_hough_dn import (
    build_dn_self_attn_mask,
    build_lcdn,
    build_simple_dn,
    gt_5pt_anchor_params,
)


def _horizontal_5pt(cx, cy, L):
    xs = torch.linspace(cx - L / 2, cx + L / 2, 5)
    ys = torch.full((5,), float(cy))
    return torch.stack([xs, ys], dim=-1)


class TestGTAnchorParams(unittest.TestCase):
    def test_horizontal_anchor(self):
        pts = _horizontal_5pt(0.5, 0.5, 0.4).view(1, 1, 5, 2)
        cx, cy, theta, L = gt_5pt_anchor_params(pts)
        self.assertAlmostEqual(float(cx.item()), 0.5, places=5)
        self.assertAlmostEqual(float(cy.item()), 0.5, places=5)
        self.assertAlmostEqual(float(theta.item()), 0.0, places=5)
        self.assertAlmostEqual(float(L.item()), 0.4, places=5)


class TestSimpleDN(unittest.TestCase):
    def test_shapes_and_target_equality(self):
        b, n, g = 2, 3, 4
        gt = torch.zeros((b, n, 5, 2))
        gt[0, 0] = _horizontal_5pt(0.5, 0.5, 0.3)
        gt[0, 1] = _horizontal_5pt(0.3, 0.7, 0.2)
        gt[1, 2] = _horizontal_5pt(0.6, 0.4, 0.4)
        mask = torch.zeros((b, n), dtype=torch.bool)
        mask[0, 0] = True
        mask[0, 1] = True
        mask[1, 2] = True

        gen = torch.Generator().manual_seed(0)
        out = build_simple_dn(gt, mask, num_groups=g, sigma_xy=0.05, generator=gen)
        self.assertEqual(tuple(out["refs"].shape), (b, n * g, 5, 2))
        self.assertEqual(tuple(out["targets"].shape), (b, n * g, 5, 2))
        self.assertEqual(tuple(out["valid"].shape), (b, n * g))
        self.assertFalse(out["refs"].requires_grad)
        self.assertFalse(out["targets"].requires_grad)
        # valid slots are repeated G times per inst.
        # In batch 0, instances 0 and 1 are valid → first n*g elements have
        # valid = [T,T,T,T, T,T,T,T, F,F,F,F] for n=3.
        expected_valid_b0 = torch.zeros((n * g,), dtype=torch.bool)
        expected_valid_b0[0 : 2 * g] = True
        self.assertTrue(torch.equal(out["valid"][0], expected_valid_b0))

    def test_targets_are_unmodified_gt(self):
        b, n, g = 1, 2, 2
        gt = torch.zeros((b, n, 5, 2))
        gt[0, 0] = _horizontal_5pt(0.5, 0.5, 0.3)
        gt[0, 1] = _horizontal_5pt(0.3, 0.6, 0.1)
        mask = torch.ones((b, n), dtype=torch.bool)

        out = build_simple_dn(gt, mask, num_groups=g, sigma_xy=0.05,
                              generator=torch.Generator().manual_seed(1))
        # Targets for slots 0..1 must equal gt[0, 0], slots 2..3 must equal gt[0, 1].
        self.assertTrue(torch.allclose(out["targets"][0, 0], gt[0, 0]))
        self.assertTrue(torch.allclose(out["targets"][0, 1], gt[0, 0]))
        self.assertTrue(torch.allclose(out["targets"][0, 2], gt[0, 1]))
        self.assertTrue(torch.allclose(out["targets"][0, 3], gt[0, 1]))

    def test_noise_has_bounded_deviation(self):
        b, n, g = 1, 1, 4
        gt = torch.zeros((b, n, 5, 2))
        gt[0, 0] = _horizontal_5pt(0.5, 0.5, 0.4)
        mask = torch.ones((b, n), dtype=torch.bool)
        sigma = 0.02
        out = build_simple_dn(gt, mask, num_groups=g, sigma_xy=sigma,
                              generator=torch.Generator().manual_seed(7))
        diff = (out["refs"][0] - out["targets"][0]).abs().max().item()
        # 6-sigma bound should easily hold.
        self.assertLess(diff, 6 * sigma)


class TestLCDN(unittest.TestCase):
    def test_targets_are_clean_gt(self):
        b, n, g = 1, 2, 2
        gt = torch.zeros((b, n, 5, 2))
        gt[0, 0] = _horizontal_5pt(0.5, 0.5, 0.3)
        gt[0, 1] = _horizontal_5pt(0.3, 0.6, 0.2)
        mask = torch.ones((b, n), dtype=torch.bool)
        out = build_lcdn(gt, mask, num_groups=g, sigma_xy=0.01,
                        scale_range=0.1, rot_deg=5.0,
                        generator=torch.Generator().manual_seed(2))
        for slot in range(n * g):
            inst_id = slot // g
            self.assertTrue(torch.allclose(out["targets"][0, slot], gt[0, inst_id]))

    def test_refs_are_clamped(self):
        b, n, g = 1, 1, 1
        gt = torch.zeros((b, n, 5, 2))
        gt[0, 0] = _horizontal_5pt(0.9, 0.5, 0.4)
        mask = torch.ones((b, n), dtype=torch.bool)
        out = build_lcdn(gt, mask, num_groups=g, sigma_xy=0.05,
                        scale_range=0.3, rot_deg=20.0,
                        generator=torch.Generator().manual_seed(3))
        self.assertTrue(torch.all(out["refs"] >= 0.0))
        self.assertTrue(torch.all(out["refs"] <= 1.0))


class TestDNAttnMask(unittest.TestCase):
    def test_block_diagonal(self):
        # 4 matching + DN groups of sizes (2, 2) → 8x8.
        mask = build_dn_self_attn_mask(n_matching=4, dn_group_sizes=(2, 2), device=torch.device("cpu"))
        self.assertEqual(tuple(mask.shape), (8, 8))
        # Diagonal blocks (matching×matching, dn0×dn0, dn1×dn1) are False.
        self.assertTrue(bool((~mask[0:4, 0:4]).all().item()))
        self.assertTrue(bool((~mask[4:6, 4:6]).all().item()))
        self.assertTrue(bool((~mask[6:8, 6:8]).all().item()))
        # Off-diagonal blocks are True (blocked).
        self.assertTrue(bool(mask[0:4, 4:6].all().item()))
        self.assertTrue(bool(mask[0:4, 6:8].all().item()))
        self.assertTrue(bool(mask[4:6, 6:8].all().item()))


if __name__ == "__main__":
    unittest.main()

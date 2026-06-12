# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for :mod:`yolino.model.yolino_hough_cluster` (exp51 Path A)."""
import math
import unittest

import torch

from yolino.model.yolino_hough_cluster import (
    anchor_to_5pt_ref,
    segments_to_hough_anchors,
)


def _make_synthetic_geom_mid_dir(
    seg_specs,
    hf: int = 8,
    wf: int = 8,
    p: int = 1,
    v_total: int = 5,
    conf_channel: int = 4,
    stride: float = 32.0,
):
    """Build a ``geom_act`` tensor with a few synthetic high-conf segments.

    seg_specs: list of (cell_row, cell_col, dy_per_cell, dx_per_cell, conf).
        dy_per_cell / dx_per_cell are MID_DIR slot offsets (fractions of stride).
    Returns:
        geom_act ``[1, hf*wf, p, v_total]`` with first four channels = MID_DIR.
    """
    geom = torch.zeros((1, hf * wf, p, v_total))
    # Default mid offsets: cell center (0.5, 0.5) and zero direction (won't be a peak).
    geom[..., 0] = 0.5
    geom[..., 1] = 0.5
    geom[..., 2] = 0.0
    geom[..., 3] = 0.0
    geom[..., conf_channel] = 0.01

    for (r, c, dv, dh, conf) in seg_specs:
        cell = r * wf + c
        geom[0, cell, 0, 0] = 0.5
        geom[0, cell, 0, 1] = 0.5
        geom[0, cell, 0, 2] = dv
        geom[0, cell, 0, 3] = dh
        geom[0, cell, 0, conf_channel] = conf
    return geom


class TestHoughCluster(unittest.TestCase):
    def test_two_lines_two_clusters(self):
        # Build two near-horizontal "wires" at different y, both having multiple
        # high-conf segments. DBSCAN should yield K=2 clusters.
        hf = wf = 8
        stride = 32.0
        specs = []
        # Wire 1: y ≈ 64 (row 2), high horizontal direction.
        for c in range(2, 7):
            specs.append((2, c, 0.0, 1.0, 0.9))
        # Wire 2: y ≈ 192 (row 6), high horizontal direction.
        for c in range(1, 6):
            specs.append((6, c, 0.0, 1.0, 0.9))

        geom = _make_synthetic_geom_mid_dir(specs, hf=hf, wf=wf)
        out = segments_to_hough_anchors(
            geom, stride=stride, img_h=256, img_w=256,
            conf_channel=4, hf=hf, wf=wf,
            num_anchors=10, conf_thresh=0.3, max_segments=512,
            dbscan_eps=0.05, dbscan_min_samples=2,
        )
        self.assertEqual(tuple(out["cx"].shape), (1, 10))
        self.assertEqual(tuple(out["cy"].shape), (1, 10))
        self.assertEqual(tuple(out["theta"].shape), (1, 10))
        self.assertEqual(tuple(out["L_init"].shape), (1, 10))
        n_valid = int(out["valid"][0].sum().item())
        self.assertEqual(n_valid, 2, msg="Expected exactly 2 valid clusters (horizontal wires)")
        # cy should differ between the two clusters.
        cy_valid = out["cy"][0][out["valid"][0]]
        self.assertGreater(float((cy_valid.max() - cy_valid.min()).item()), 0.1)
        # theta should be ≈ 0 (horizontal).
        theta_valid = out["theta"][0][out["valid"][0]]
        for th in theta_valid.tolist():
            self.assertLess(abs(th), 0.2, msg="theta should be ~0 for horizontal wires")

    def test_l_init_minimum_bound(self):
        # Single high-conf segment: span = 0 (only one segment in cluster) → L_init must
        # fall back to L_init_default * 0.3 to prevent 5-pt collapse.
        specs = [(4, 4, 0.0, 1.0, 0.9)]
        hf = wf = 8
        geom = _make_synthetic_geom_mid_dir(specs, hf=hf, wf=wf)
        out = segments_to_hough_anchors(
            geom, stride=32.0, img_h=256, img_w=256,
            conf_channel=4, hf=hf, wf=wf,
            num_anchors=4, conf_thresh=0.3,
            dbscan_eps=0.05, dbscan_min_samples=1,
            L_init_default=0.3,
        )
        valid = out["valid"][0]
        self.assertGreaterEqual(int(valid.sum().item()), 1)
        L_init = out["L_init"][0][valid]
        # Even for a single-segment cluster, L_init >= 0.3 * 0.3 = 0.09.
        self.assertTrue(torch.all(L_init >= 0.09 - 1e-6))

    def test_outputs_are_detached(self):
        # Even with require_grad on the input, the returned anchors must be detached.
        specs = [(2, 3, 0.0, 1.0, 0.8)]
        hf = wf = 4
        geom = _make_synthetic_geom_mid_dir(specs, hf=hf, wf=wf)
        geom.requires_grad_(True)
        out = segments_to_hough_anchors(
            geom, stride=32.0, img_h=128, img_w=128,
            conf_channel=4, hf=hf, wf=wf, num_anchors=2,
            conf_thresh=0.3, dbscan_min_samples=1,
        )
        for key in ("cx", "cy", "theta", "L_init", "valid"):
            self.assertFalse(out[key].requires_grad, msg=f"{key} must not require grad")

    def test_empty_image_returns_no_valid_anchors(self):
        # No segment passes the conf threshold → all slots invalid.
        hf = wf = 4
        geom = _make_synthetic_geom_mid_dir([], hf=hf, wf=wf)
        out = segments_to_hough_anchors(
            geom, stride=32.0, img_h=128, img_w=128,
            conf_channel=4, hf=hf, wf=wf, num_anchors=3,
            conf_thresh=0.3, dbscan_min_samples=1,
        )
        self.assertEqual(int(out["valid"][0].sum().item()), 0)


class TestAnchorTo5pt(unittest.TestCase):
    def test_shape(self):
        cx = torch.tensor([[0.3, 0.7]])
        cy = torch.tensor([[0.5, 0.5]])
        theta = torch.tensor([[0.0, math.pi / 2]])
        L = torch.tensor([[0.2, 0.4]])
        out = anchor_to_5pt_ref(cx, cy, theta, L)
        self.assertEqual(tuple(out.shape), (1, 2, 5, 2))
        self.assertFalse(out.requires_grad)

    def test_horizontal_anchor_p3_is_center(self):
        cx = torch.tensor([[0.5]])
        cy = torch.tensor([[0.5]])
        theta = torch.tensor([[0.0]])
        L = torch.tensor([[0.2]])
        out = anchor_to_5pt_ref(cx, cy, theta, L)
        # P3 (index=2) == center.
        self.assertTrue(torch.allclose(out[0, 0, 2], torch.tensor([0.5, 0.5])))
        # P1 ≈ (cx - L/2, cy), P5 ≈ (cx + L/2, cy).
        self.assertTrue(torch.allclose(out[0, 0, 0], torch.tensor([0.4, 0.5])))
        self.assertTrue(torch.allclose(out[0, 0, 4], torch.tensor([0.6, 0.5])))

    def test_vertical_anchor(self):
        cx = torch.tensor([[0.5]])
        cy = torch.tensor([[0.5]])
        theta = torch.tensor([[math.pi / 2]])
        L = torch.tensor([[0.2]])
        out = anchor_to_5pt_ref(cx, cy, theta, L)
        # P1 ≈ (cx, cy - L/2), P5 ≈ (cx, cy + L/2).
        # Note sin(pi/2)=1 → dy goes positive for positive offset → P5 has larger y.
        self.assertAlmostEqual(float(out[0, 0, 0, 0].item()), 0.5, places=4)
        self.assertAlmostEqual(float(out[0, 0, 4, 0].item()), 0.5, places=4)
        # y endpoints: 0.5 - 0.1 = 0.4, 0.5 + 0.1 = 0.6.
        self.assertAlmostEqual(float(out[0, 0, 0, 1].item()), 0.4, places=4)
        self.assertAlmostEqual(float(out[0, 0, 4, 1].item()), 0.6, places=4)

    def test_clamp_to_unit_square(self):
        # Anchor at corner with large L → ref must clamp into [0,1].
        cx = torch.tensor([[0.95]])
        cy = torch.tensor([[0.5]])
        theta = torch.tensor([[0.0]])
        L = torch.tensor([[0.4]])
        out = anchor_to_5pt_ref(cx, cy, theta, L)
        self.assertTrue(torch.all((out >= 0.0) & (out <= 1.0)))
        # P5 saturates at 1.0.
        self.assertAlmostEqual(float(out[0, 0, 4, 0].item()), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()

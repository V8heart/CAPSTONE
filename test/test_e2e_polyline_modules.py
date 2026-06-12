# SPDX-License-Identifier: GPL-3.0-or-later
import unittest

import torch

from yolino.model.e2e_polyline_criterion import E2EPolylineSetCriterion, chamfer_distance_l1
from yolino.utils.enums import LINE
from yolino.model.e2e_polyline_modules import (
    E2EDifferentiablePostHead,
    bezier_sample_points,
    bernstein_matrix,
)


class TestE2EPolyline(unittest.TestCase):
    def test_bezier_matrix_shape(self):
        t = bernstein_matrix(3, 16, device=torch.device("cpu"), dtype=torch.float32)
        self.assertEqual(tuple(t.shape), (16, 4))

    def test_bezier_grad(self):
        k, t_n = 4, 8
        bern = bernstein_matrix(3, t_n, device=torch.device("cpu"), dtype=torch.float32)
        ctrl = torch.randn(2, 5, k, 2, requires_grad=True)
        pts = bezier_sample_points(ctrl, bern)
        self.assertEqual(tuple(pts.shape), (2, 5, t_n, 2))
        pts.sum().backward()
        self.assertIsNotNone(ctrl.grad)

    def test_e2e_head_forward(self):
        head = E2EDifferentiablePostHead(
            line_rep=LINE.MID_DIR,
            fpn_channels=64,
            window_size=3,
            token_dim=32,
            transformer_heads=4,
            transformer_layers=1,
            transformer_ff=64,
            bezier_degree=2,
            bezier_num_samples=8,
            feature_aware=True,
            cross_image_context=False,
        )
        b, h, w, p, v = 2, 4, 4, 2, 5
        geom = torch.rand(b, h * w, p, v)
        geom[..., 4] = torch.sigmoid(torch.randn(b, h * w, p))  # conf
        feat = torch.randn(b, 64, h, w, requires_grad=True)
        out = head(geom, feat, stride=32.0, img_h=128, img_w=128, conf_channel=4)
        out["bezier_curve_px"].sum().backward()
        self.assertIsNotNone(feat.grad)

    def test_chamfer(self):
        a = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        b = torch.tensor([[0.1, 0.0], [2.0, 0.0]])
        d = chamfer_distance_l1(a, b)
        self.assertGreater(d.item(), 0.0)

    def test_criterion_smoke(self):
        crit = E2EPolylineSetCriterion(curve_loss="smooth_l1_ctrl", w_ctrl=1.0, w_chamfer=0.0)
        m, n, k, t = 3, 3, 4, 6
        pc = torch.randn(m, k, 2)
        gc = torch.randn(n, k, 2)
        pcr = torch.randn(m, t, 2)
        gcr = torch.randn(n, t, 2)
        loss, _ = crit(pcr, pc, gcr, gc)
        self.assertEqual(loss.ndim, 0)


if __name__ == "__main__":
    unittest.main()

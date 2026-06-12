# SPDX-License-Identifier: GPL-3.0-or-later
"""Forward-shape + gradient smoke tests for ``YolinoHoughDetrPolyHead`` (exp51)."""
import unittest

import torch

from yolino.model.yolino_hough_detr_head import YolinoHoughDetrPolyHead
from yolino.utils.enums import LINE


def _dummy_geom(b: int = 2, hf: int = 8, wf: int = 8, p: int = 4, v: int = 5):
    """Synthetic MID_DIR geom_act with a sprinkle of high-conf segments."""
    geom = torch.zeros((b, hf * wf, p, v))
    geom[..., 0] = 0.5
    geom[..., 1] = 0.5
    geom[..., 2] = 0.0
    geom[..., 3] = 1.0  # all "horizontal" so anchors at theta ≈ 0
    geom[..., 4] = 0.01
    # Plant several high-confidence segments along horizontal rows (clamped to grid).
    r1 = max(1, min(hf - 1, 2))
    r2 = max(1, min(hf - 1, 5))
    for c in range(min(wf, 2), min(wf, 6)):
        geom[0, r1 * wf + c, 0, 4] = 0.95
        geom[0, r2 * wf + c, 0, 4] = 0.95
    if b > 1:
        r3 = max(1, min(hf - 1, 3))
        for c in range(min(wf, 1), min(wf, 5)):
            geom[1, r3 * wf + c, 0, 4] = 0.9
    return geom


class TestHoughDetrHeadForward(unittest.TestCase):
    def _make_head(self, **kwargs):
        defaults = dict(
            line_rep=LINE.MID_DIR,
            fpn_channels=64,
            token_dim=32,
            num_queries=8,
            decoder_layers=2,
            decoder_heads=4,
            decoder_ff=64,
            dropout=0.0,
            sine_freq_bands=4,
            conf_thresh=0.3,
            max_segments=128,
            dbscan_eps=0.05,
            dbscan_min_samples=1,
            pt_radius_norm=0.15,
            max_step_norm=0.05,
            mem_encoder_layers=0,
            dn_mode="none",
            dn_groups=0,
        )
        defaults.update(kwargs)
        return YolinoHoughDetrPolyHead(**defaults)

    def test_forward_shapes_no_dn(self):
        head = self._make_head()
        head.eval()
        b, hf, wf, p, v = 2, 8, 8, 4, 5
        geom = _dummy_geom(b=b, hf=hf, wf=wf, p=p, v=v)
        feat = torch.randn(b, 64, hf, wf)
        out = head(
            geom, feat, stride=32.0, img_h=256, img_w=256,
            conf_channel=4, e2e_gt_pack=None,
        )
        K = head.K
        self.assertEqual(tuple(out["polylines_px"].shape), (b, K, 5, 2))
        self.assertEqual(tuple(out["bezier_curve_px"].shape), (b, K, 5, 2))
        self.assertEqual(tuple(out["objectness_logits"].shape), (b, K))
        self.assertEqual(tuple(out["ref_points_px_init"].shape), (b, K, 5, 2))
        self.assertEqual(tuple(out["ref_pts_layers"].shape), (head.decoder_layers, b, K, 5, 2))
        # DN keys exist but are length-0.
        self.assertEqual(tuple(out["dn_polylines_px"].shape), (b, 0, 5, 2))
        self.assertEqual(tuple(out["dn_targets_px"].shape), (b, 0, 5, 2))
        # Polylines stay within image bounds.
        for q in range(K):
            for v in range(5):
                for d in (0, 1):
                    val = float(out["polylines_px"][0, q, v, d].item())
                    self.assertGreaterEqual(val, 0.0)
                    self.assertLessEqual(val, 256.0)

    def test_forward_shapes_with_simple_dn(self):
        head = self._make_head(dn_mode="simple", dn_groups=2)
        head.train()
        b, hf, wf, p, v = 2, 8, 8, 4, 5
        geom = _dummy_geom(b=b, hf=hf, wf=wf, p=p, v=v)
        feat = torch.randn(b, 64, hf, wf)
        # Tiny GT pack: 2 instances, both horizontal.
        ni, mp = 4, 8
        padded = torch.zeros((b, ni, mp, 2))
        inst_mask = torch.zeros((b, ni), dtype=torch.bool)
        pt_mask = torch.zeros((b, ni, mp), dtype=torch.bool)
        # Batch 0, inst 0: horizontal at y=64.
        padded[0, 0, 0] = torch.tensor([20.0, 64.0])
        padded[0, 0, 1] = torch.tensor([220.0, 64.0])
        pt_mask[0, 0, :2] = True
        inst_mask[0, 0] = True
        # Batch 1, inst 0.
        padded[1, 0, 0] = torch.tensor([30.0, 96.0])
        padded[1, 0, 1] = torch.tensor([200.0, 96.0])
        pt_mask[1, 0, :2] = True
        inst_mask[1, 0] = True
        gt_pack = {"padded": padded, "inst_mask": inst_mask, "pt_mask": pt_mask}
        out = head(
            geom, feat, stride=32.0, img_h=256, img_w=256,
            conf_channel=4, e2e_gt_pack=gt_pack,
        )
        K = head.K
        n_dn = ni * head.dn_groups
        self.assertEqual(tuple(out["polylines_px"].shape), (b, K, 5, 2))
        self.assertEqual(tuple(out["dn_polylines_px"].shape), (b, n_dn, 5, 2))
        self.assertEqual(tuple(out["dn_targets_px"].shape), (b, n_dn, 5, 2))
        self.assertEqual(tuple(out["dn_valid"].shape), (b, n_dn))

    def test_grad_flows_into_delta_mlp(self):
        # Critical: with the corrected detach policy, last-layer Δref MLP weights
        # MUST receive a gradient when we backprop from ``polylines_px``.
        head = self._make_head()
        head.train()
        b, hf, wf, p, v = 1, 6, 6, 2, 5
        geom = _dummy_geom(b=b, hf=hf, wf=wf, p=p, v=v)
        feat = torch.randn(b, 64, hf, wf, requires_grad=True)
        out = head(geom, feat, stride=32.0, img_h=192, img_w=192,
                   conf_channel=4, e2e_gt_pack=None)
        loss = out["polylines_px"].sum()
        loss.backward()
        last_layer = head.layers[-1]
        any_grad = any(p.grad is not None and float(p.grad.abs().sum().item()) > 0.0
                       for p in last_layer.delta_mlp.parameters())
        self.assertTrue(any_grad, msg="Last-layer Δref MLP must receive gradient")

    def test_anchor_outputs_are_detached(self):
        head = self._make_head()
        head.eval()
        b, hf, wf, p, v = 1, 4, 4, 2, 5
        geom = _dummy_geom(b=b, hf=hf, wf=wf, p=p, v=v)
        # Even with requires_grad on geom_act, anchors must not require grad.
        geom.requires_grad_(True)
        feat = torch.randn(b, 64, hf, wf)
        out = head(geom, feat, stride=32.0, img_h=128, img_w=128,
                   conf_channel=4, e2e_gt_pack=None)
        for key in ("cx", "cy", "theta", "L_init", "valid"):
            self.assertFalse(
                out["hough_anchors"][key].requires_grad,
                msg=f"hough_anchors[{key!r}] must not require grad",
            )
        # ref_points_px_init carries (cx, cy, ...) so it should also be detached.
        self.assertFalse(out["ref_points_px_init"].requires_grad)


if __name__ == "__main__":
    unittest.main()

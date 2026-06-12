# SPDX-License-Identifier: GPL-3.0-or-later
"""Forward-shape, DN-gate, mask-isolation, and gradient tests for
:class:`YolinoLearnableDetrPolyHead` (exp53)."""
import unittest

import torch

from yolino.model.yolino_learnable_detr_head import YolinoLearnableDetrPolyHead


def _tiny_gt_pack(b: int = 2, ni: int = 2, mp: int = 8, img: int = 256):
    """Two horizontal GT polylines per batch element for predictable DN refs."""
    padded = torch.zeros((b, ni, mp, 2))
    inst_mask = torch.zeros((b, ni), dtype=torch.bool)
    pt_mask = torch.zeros((b, ni, mp), dtype=torch.bool)

    padded[0, 0, 0] = torch.tensor([20.0, 64.0])
    padded[0, 0, 1] = torch.tensor([220.0, 64.0])
    pt_mask[0, 0, :2] = True
    inst_mask[0, 0] = True

    if b > 1:
        padded[1, 0, 0] = torch.tensor([30.0, 96.0])
        padded[1, 0, 1] = torch.tensor([200.0, 96.0])
        pt_mask[1, 0, :2] = True
        inst_mask[1, 0] = True
    return {"padded": padded, "inst_mask": inst_mask, "pt_mask": pt_mask}


class TestLearnableDetrHeadForward(unittest.TestCase):
    def _make_head(self, **kwargs):
        defaults = dict(
            fpn_channels=64,
            token_dim=32,
            num_queries=8,
            decoder_layers=2,
            decoder_heads=4,
            decoder_ff=64,
            dropout=0.0,
            sine_freq_bands=4,
            max_step_norm=0.1,
            mem_encoder_layers=1,
            dn_mode="none",
            dn_groups=0,
            dn_off_epoch=-1,
        )
        defaults.update(kwargs)
        return YolinoLearnableDetrPolyHead(**defaults)

    # ------------------------------------------------------------------ #
    # Shape contract                                                      #
    # ------------------------------------------------------------------ #
    def test_forward_shapes_no_dn(self):
        head = self._make_head()
        head.eval()
        b, hf, wf = 2, 8, 8
        feat = torch.randn(b, 64, hf, wf)
        out = head(feat, stride=32.0, img_h=256, img_w=256, e2e_gt_pack=None)
        K = head.K
        self.assertEqual(tuple(out["polylines_px"].shape), (b, K, 5, 2))
        self.assertEqual(tuple(out["bezier_curve_px"].shape), (b, K, 5, 2))
        self.assertEqual(tuple(out["objectness_logits"].shape), (b, K))
        self.assertEqual(tuple(out["ref_points_px_init"].shape), (b, K, 5, 2))
        self.assertEqual(
            tuple(out["ref_pts_layers"].shape),
            (head.decoder_layers_n, b, K, 5, 2),
        )
        self.assertEqual(tuple(out["dn_polylines_px"].shape), (b, 0, 5, 2))
        self.assertEqual(tuple(out["dn_targets_px"].shape), (b, 0, 5, 2))
        # Polylines stay in image.
        self.assertGreaterEqual(float(out["polylines_px"].min().item()), 0.0)
        self.assertLessEqual(float(out["polylines_px"].max().item()), 256.0)

    def test_initial_5pt_collapsed_to_one_point(self):
        """Spec: ``ref_pts_2d.unsqueeze(2).expand(-1,-1,5,-1)`` ⇒ at init all 5 pts equal."""
        head = self._make_head()
        head.eval()
        b, hf, wf = 1, 4, 4
        feat = torch.randn(b, 64, hf, wf)
        out = head(feat, stride=32.0, img_h=128, img_w=128, e2e_gt_pack=None)
        ref0 = out["ref_points_px_init"][0]
        for q in range(head.K):
            for v in range(1, 5):
                self.assertAlmostEqual(
                    float(ref0[q, 0, 0].item()), float(ref0[q, v, 0].item()),
                    places=4, msg="x of init 5-pt must be identical (spec)",
                )
                self.assertAlmostEqual(
                    float(ref0[q, 0, 1].item()), float(ref0[q, v, 1].item()),
                    places=4, msg="y of init 5-pt must be identical (spec)",
                )

    def test_layer_delta_spreads_5pt(self):
        """After ≥1 decoder layer, the 5 keypoints must no longer be identical
        (delta head outputs distinct per-keypoint Δ)."""
        head = self._make_head(decoder_layers=2)
        head.eval()
        b = 1
        feat = torch.randn(b, 64, 4, 4)
        out = head(feat, stride=32.0, img_h=128, img_w=128, e2e_gt_pack=None)
        per_layer = out["ref_pts_layers"]    # [L, B, K, 5, 2]
        # At least one query in the last layer must have non-trivial spread.
        last = per_layer[-1, 0]              # [K, 5, 2]
        spread = (last.amax(dim=1) - last.amin(dim=1)).abs().max().item()
        self.assertGreater(spread, 1e-4, msg="Δref MLP must spread the 5 keypoints")

    # ------------------------------------------------------------------ #
    # DN branch                                                            #
    # ------------------------------------------------------------------ #
    def test_forward_shapes_with_simple_dn(self):
        head = self._make_head(dn_mode="simple", dn_groups=2)
        head.train()
        b = 2
        feat = torch.randn(b, 64, 8, 8)
        gt_pack = _tiny_gt_pack(b=b, ni=4, mp=8, img=256)
        out = head(feat, stride=32.0, img_h=256, img_w=256,
                   e2e_gt_pack=gt_pack, current_epoch=0)
        K = head.K
        n_dn = 4 * head.dn_groups
        self.assertEqual(tuple(out["polylines_px"].shape), (b, K, 5, 2))
        self.assertEqual(tuple(out["dn_polylines_px"].shape), (b, n_dn, 5, 2))
        self.assertEqual(tuple(out["dn_targets_px"].shape), (b, n_dn, 5, 2))
        self.assertEqual(tuple(out["dn_valid"].shape), (b, n_dn))
        self.assertEqual(
            tuple(out["dn_polylines_layers"].shape),
            (head.decoder_layers_n, b, n_dn, 5, 2),
        )

    def test_dn_off_epoch_skips_dn(self):
        """current_epoch >= dn_off_epoch ⇒ DN slots are skipped even in train mode."""
        head = self._make_head(dn_mode="simple", dn_groups=3, dn_off_epoch=10)
        head.train()
        b = 1
        feat = torch.randn(b, 64, 6, 6)
        gt_pack = _tiny_gt_pack(b=b, ni=2, mp=8)

        on_out = head(feat, stride=32.0, img_h=192, img_w=192,
                      e2e_gt_pack=gt_pack, current_epoch=5)
        self.assertGreater(on_out["dn_polylines_px"].shape[1], 0)

        off_out = head(feat, stride=32.0, img_h=192, img_w=192,
                       e2e_gt_pack=gt_pack, current_epoch=10)
        self.assertEqual(off_out["dn_polylines_px"].shape[1], 0)
        # Boundary check ABOVE cutoff too.
        off2_out = head(feat, stride=32.0, img_h=192, img_w=192,
                        e2e_gt_pack=gt_pack, current_epoch=999)
        self.assertEqual(off2_out["dn_polylines_px"].shape[1], 0)

    def test_dn_in_eval_is_skipped(self):
        head = self._make_head(dn_mode="simple", dn_groups=2)
        head.eval()
        b = 1
        feat = torch.randn(b, 64, 6, 6)
        gt_pack = _tiny_gt_pack(b=b, ni=2, mp=8)
        out = head(feat, stride=32.0, img_h=192, img_w=192,
                   e2e_gt_pack=gt_pack, current_epoch=0)
        self.assertEqual(out["dn_polylines_px"].shape[1], 0)

    # ------------------------------------------------------------------ #
    # Attention mask isolation                                            #
    # ------------------------------------------------------------------ #
    def test_matching_path_independent_of_dn_via_self_attn_mask(self):
        """The block-diagonal self-attn mask must isolate matching from DN: even
        if DN refs are perturbed, the matching predictions stay identical."""
        torch.manual_seed(0)
        head = self._make_head(dn_mode="simple", dn_groups=2, dropout=0.0)
        head.eval()      # disable dropout for determinism (we still pass gt_pack manually)
        # We bypass the DN-off-in-eval gate by calling _dn_init directly via train mode.
        head.train()

        b = 1
        torch.manual_seed(42)
        feat = torch.randn(b, 64, 6, 6)

        gt_pack_a = _tiny_gt_pack(b=b, ni=2, mp=8)
        gt_pack_b = _tiny_gt_pack(b=b, ni=2, mp=8)
        gt_pack_b["padded"] = gt_pack_b["padded"] + 1.0   # perturb DN GT only

        torch.manual_seed(7)
        out_a = head(feat, stride=32.0, img_h=192, img_w=192,
                     e2e_gt_pack=gt_pack_a, current_epoch=0)
        torch.manual_seed(7)
        out_b = head(feat, stride=32.0, img_h=192, img_w=192,
                     e2e_gt_pack=gt_pack_b, current_epoch=0)

        # Matching predictions must be unchanged thanks to the block-diag mask.
        diff = (out_a["polylines_px"] - out_b["polylines_px"]).abs().max().item()
        self.assertLess(diff, 1e-5,
                        msg="Matching path leaked DN info despite block-diag mask")
        # And DN predictions must differ (otherwise the whole DN branch is dead).
        dn_diff = (out_a["dn_polylines_px"] - out_b["dn_polylines_px"]).abs().max().item()
        self.assertGreater(dn_diff, 1e-3,
                           msg="DN predictions did not respond to GT perturbation")

    # ------------------------------------------------------------------ #
    # Gradient flow                                                       #
    # ------------------------------------------------------------------ #
    def test_grad_flows_into_delta_mlp(self):
        head = self._make_head()
        head.train()
        b = 1
        feat = torch.randn(b, 64, 6, 6, requires_grad=True)
        out = head(feat, stride=32.0, img_h=192, img_w=192, e2e_gt_pack=None)
        loss = out["polylines_px"].sum()
        loss.backward()
        last_layer = head.layers[-1]
        any_grad = any(
            p.grad is not None and float(p.grad.abs().sum().item()) > 0.0
            for p in last_layer.delta_mlp.parameters()
        )
        self.assertTrue(any_grad, msg="Last-layer Δref MLP must receive gradient")

    def test_grad_flows_into_learnable_queries(self):
        head = self._make_head()
        head.train()
        b = 1
        feat = torch.randn(b, 64, 6, 6)
        out = head(feat, stride=32.0, img_h=192, img_w=192, e2e_gt_pack=None)
        loss = (out["polylines_px"] ** 2).sum() + out["objectness_logits"].sum()
        loss.backward()
        self.assertIsNotNone(head.query_content.weight.grad)
        self.assertGreater(float(head.query_content.weight.grad.abs().sum().item()), 0.0)
        # query_ref_pts: only path is via the (clamped+detached) ref → sine PE → query PE,
        # plus the initial-ref-to-loss path before the first detach. Spec says queries
        # train through later iterative refinement gradients via the PE; check non-None.
        self.assertIsNotNone(head.query_ref_pts.weight.grad)


if __name__ == "__main__":
    unittest.main()

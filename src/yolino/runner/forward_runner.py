# Copyright 2023 Karlsruhe Institute of Technology, Institute for Measurement
# and Control Systems
#
# This file is part of YOLinO.
#
# YOLinO is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# YOLinO is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# YOLinO. If not, see <https://www.gnu.org/licenses/>.
#
# ---------------------------------------------------------------------------- #
# ----------------------------- COPYRIGHT ------------------------------------ #
# ---------------------------------------------------------------------------- #
import timeit

from yolino.dataset.dataset_factory import DatasetFactory
from yolino.model.activations import get_activations
from yolino.model.model_factory import load_checkpoint
from yolino.model.variable_structure import VariableStructure
from yolino.utils.logger import Log

import torch

class ForwardRunner:
    def __init__(self, args, preloaded_model=None, start_epoch=-1, coords: VariableStructure = None,
                 load_best=False) -> None:
        self.args = args

        if coords is None:
            if preloaded_model is None:
                coords = DatasetFactory.get_coords("train", args)  # split does not matter
            else:
                # DDP wraps the original nn.Module under `.module`.
                if hasattr(preloaded_model, "coords"):
                    coords = preloaded_model.coords
                elif hasattr(preloaded_model, "module") and hasattr(preloaded_model.module, "coords"):
                    coords = preloaded_model.module.coords
                else:
                    raise AttributeError("Could not resolve coords from preloaded model. "
                                         "Expected `.coords` or `.module.coords`.")

        if preloaded_model:
            self.model = preloaded_model
            self.start_epoch = start_epoch
        else:
            self.model, _, self.start_epoch = load_checkpoint(args, coords, allow_failure=False, load_best=load_best)

        _tgt = torch.device(self.args.cuda)
        _p_dev = next(self.model.parameters()).device
        if _p_dev != _tgt:
            self.model = self.model.to(_tgt)

        self.activations = get_activations(self.args.activations, coords, self.args.linerep)

    def __call__(self, images, is_train, epoch, first_run=False, e2e_gt_pack=None):
        """

        Args:
            images (torch.tensor):
                [batch, 3, height, width]
            e2e_gt_pack: optional GT polyline pack (TTPLA collate). Required only by
                heads that need GT at forward time — e.g. ``--e2e_mode=hough_detr``
                with denoising training enabled (``e2e_hough_dn_mode != "none"``).
                Pass ``None`` for inference / heads that don't consume GT.

        Returns:
            tuple: ``(geom_act, embed_act, geom_logits, e2e_out)`` — activated geometry, embedding preds,
            pre-activation geometry logits, and optional E2E head dict (or None).
        """
        # Do not compare with substring checks: e.g. args.cuda=="cuda" is contained in str(cuda:0).
        tgt = next(self.model.parameters()).device
        if images.device != tgt:
            Log.debug("Moved images from %s to %s" % (images.device, tgt))
            images = images.to(tgt)

        inference_start = timeit.default_timer()
        self.model.train(is_train)
        # Only heads that consume GT inside forward (hough_detr DN) need this stash.
        # The model reads + clears it at the end of forward so it never leaks across
        # backward boundaries.
        net = self.model.module if hasattr(self.model, "module") else self.model
        if e2e_gt_pack is not None and getattr(net, "e2e_head", None) is not None:
            net._pending_e2e_gt_pack = e2e_gt_pack
        # Heads that gate behavior on the current training epoch (e.g.
        # ``learnable_detr`` with ``--e2e_dn_off_epoch``) read this transient
        # stash; cleared by the model at the end of forward to avoid leakage.
        if getattr(net, "e2e_head", None) is not None:
            net._pending_e2e_epoch = epoch
        if is_train:
            logits = self.model(images)
        else:
            with torch.no_grad():
                logits = net(images)

        e2e_pack = None
        e2e_out = None
        if isinstance(logits, tuple) and len(logits) == 3:
            geom_logits, embed_logits, third = logits
            logits_for_act = (geom_logits, embed_logits)
            # Legacy: pack with head_feat for run_e2e_post outside forward (breaks DDP).
            # Current YolinoNet returns the E2E head dict from forward (keys include bezier_curve_px).
            if isinstance(third, dict) and "head_feat" in third:
                e2e_pack = third
            else:
                e2e_out = third
        else:
            logits_for_act = logits

        outputs = self.activations(logits_for_act)

        Log.time(key="raw_infer", value=timeit.default_timer() - inference_start, epoch=epoch)

        if first_run:
            Log.graph(self.model, images)

        # Pre-activation geometry head logits (for focal BCE on logits). Embedding branch unchanged.
        if isinstance(logits, tuple):
            geom_logits = logits[0]
        else:
            geom_logits = logits
        if isinstance(outputs, tuple):
            geom_act, embed_act = outputs
        else:
            geom_act, embed_act = outputs, None

        if e2e_pack is not None:
            net = self.model.module if hasattr(self.model, "module") else self.model
            runner = getattr(net, "run_e2e_post", None)
            if runner is not None:
                e2e_out = runner(geom_act, e2e_pack)
        return geom_act, embed_act, geom_logits, e2e_out

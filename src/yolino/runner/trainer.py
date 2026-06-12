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
import os
import math

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from matplotlib import pyplot as plt
from tqdm import tqdm
from yolino.dataset.dataset_factory import DatasetFactory, dataloader_worker_init_fn
from yolino.grid.grid_factory import GridFactory
from yolino.model.activations import get_activations
from yolino.model.loss import get_loss
from yolino.model.loss_container import LossContainer
from yolino.model.yolino_detr_criterion import compute_detr_e2e_loss
from yolino.model.yolino_gnn_criterion import compute_gnn_e2e_loss
from yolino.model.yolino_center_criterion import compute_center_e2e_loss
from yolino.model.model_factory import load_checkpoint, save_best_checkpoint, save_checkpoint
from yolino.model.optimizer_factory import get_optimizer
from yolino.runner.evaluator import Evaluator
from yolino.runner.forward_runner import ForwardRunner
from yolino.utils.enums import ColorStyle, CoordinateSystem, ImageIdx, Variables, LossWeighting, AnchorDistribution
from yolino.utils.logger import Log
from yolino.viz.plot import plot, plot_cell_class, convert_to_torch_image, plot_style_grid

VAL_TAG = "val"
TRAIN_TAG = "train"


class TrainHandler:
    def __init__(self, args) -> None:
        self.args = args

        # Load data
        self.dataset, self.loader = DatasetFactory.get(args.dataset, only_available=True, split="train", args=args,
                                                       shuffle=True, augment=True, ignore_duplicates=False)
        self.val_dataset, self.val_loader = DatasetFactory.get(args.dataset, only_available=True, split="val",
                                                               args=args, shuffle=False, augment=False,
                                                               ignore_duplicates=False)
        self.train_sampler = None
        self.val_sampler = None
        if getattr(args, "distributed", False):
            self.train_sampler = DistributedSampler(self.dataset, num_replicas=args.world_size,
                                                    rank=args.rank, shuffle=True, drop_last=True)
            self.val_sampler = DistributedSampler(self.val_dataset, num_replicas=args.world_size,
                                                  rank=args.rank, shuffle=False, drop_last=False)
            _dl_kw = {}
            if int(args.loading_workers) > 0:
                _dl_kw["persistent_workers"] = True
                _dl_kw["prefetch_factor"] = 2
                _dl_kw["worker_init_fn"] = dataloader_worker_init_fn
            self.loader = DataLoader(self.dataset, batch_size=args.batch_size, sampler=self.train_sampler,
                                     shuffle=False, drop_last=True, num_workers=args.loading_workers,
                                     pin_memory=args.gpu, **_dl_kw)
            self.val_loader = DataLoader(self.val_dataset, batch_size=args.batch_size, sampler=self.val_sampler,
                                         shuffle=False, drop_last=False, num_workers=args.loading_workers,
                                         pin_memory=args.gpu, **_dl_kw)
        # Used by LR schedulers (especially warmup+cosine with step-wise updates).
        self.args.iters_per_epoch = len(self.loader)
        self.args.total_train_steps = max(1, (int(args.epoch) * int(self.args.iters_per_epoch)))
        Log.upload_params({"train_imgs": len(self.dataset), "val_imgs": len(self.val_dataset),
                           "dataset_path": self.dataset.dataset_path})

        # model
        self.model, scheduler_checkpoint, self.model_epoch = load_checkpoint(args, self.dataset.coords)
        if getattr(args, "distributed", False):
            # If embedding supervision is disabled, embed-head parameters remain unused by loss.
            # DDP must then track unused params to avoid reducer bucket rebuild failures.
            ddp_find_unused = bool(getattr(args, "ddp_find_unused_parameters", False))
            if not getattr(args, "train_instance_embedding", False) and not ddp_find_unused:
                ddp_find_unused = True
                Log.warning("Forcing ddp_find_unused_parameters=True because instance embedding loss is disabled.")
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[args.local_rank] if args.gpu else None,
                output_device=args.local_rank if args.gpu else None,
                find_unused_parameters=ddp_find_unused,
            )

        # geom head terms (+ optional instance embedding term)
        num_geom_terms = len(self.dataset.coords.train_vars())
        num_train_vars = num_geom_terms + (1 if getattr(args, "train_instance_embedding", False) else 0)
        num_cells = np.prod(args.grid_shape)

        # Argoverse has (per scale) 32: 106; 16: 201; 8: 389 line segments in an image (on average)
        num_lines = 106 if args.scale == 32 else (201 if args.scale == 16 else 389)

        activation_is_exp = [
            get_activations(args.activations, coords=self.dataset.coords,
                            linerep=self.dataset.coords.line_representation.enum).activations[i].is_exp()
            for i in range(num_geom_terms)
        ]
        if getattr(args, "train_instance_embedding", False):
            # embedding loss는 exp-activation weight scheme에 묶지 않음 (learned weighting에서도 안정적)
            activation_is_exp.append(False)
        init_weights = args.weights
        if init_weights is None:
            init_weights = [1] * num_geom_terms
        else:
            init_weights = list(init_weights)
        if getattr(args, "train_instance_embedding", False):
            init_weights.append(getattr(args, "instance_embedding_weight", None) or 1)

        self.loss_weights = self.__init_loss_weights__(num_train_vars=num_train_vars,
                                                       cuda=self.args.cuda, init_weights=init_weights,
                                                       loss_weighting=args.loss_weight_strategy,
                                                       average_num_lines=num_lines, num_cells=int(num_cells),
                                                       num_preds=args.num_predictors, is_exponential=activation_is_exp)

        self.conf_loss_weights = self.__init_conf_loss_weights__(cuda=self.args.cuda,
                                                                 init_weights=args.conf_match_weight,
                                                                 loss_weighting=args.loss_weight_strategy,
                                                                 average_num_lines=num_lines, num_cells=int(num_cells),
                                                                 num_preds=args.num_predictors,
                                                                 is_exponential=[activation_is_exp[
                                                                                     -1]] * num_train_vars)

        self.forward = ForwardRunner(args, self.model, self.model_epoch)
        self.loss_fct = get_loss(losses=args.loss, args=args, coords=self.dataset.coords,
                                 weights=self.loss_weights, anchors=self.dataset.anchors,
                                 conf_weights=self.conf_loss_weights)
        self.optimizer, self.scheduler = get_optimizer(args, net=self.forward.model,
                                                       loss_weights=[*self.loss_weights, *self.conf_loss_weights])
        self.scaler = torch.cuda.amp.GradScaler(enabled=bool(getattr(self.args, "amp", True) and self.args.gpu))
        self.debug_anomaly = os.environ.get("YOLINO_DEBUG_ANOMALY", "").lower() in ("1", "true", "yes")
        Log.gradients(model=self.model)

        self.evaluator = Evaluator(args, coords=self.dataset.coords, prepare_forward=False,
                                   anchors=self.dataset.anchors)
        self.evaluator.forward = self.forward

        self.best_mean_loss = np.inf
        self.best_gradient_loss = np.inf
        self.best_epoch = -1
        self.last_loss = torch.inf

        report_vars = getattr(self.args, "training_variables_all", self.args.training_variables)
        if getattr(self.args, "train_instance_embedding", False) and Variables.INSTANCE not in report_vars:
            report_vars = list(report_vars) + [Variables.INSTANCE]
        self.losses = {}
        self.losses[TRAIN_TAG] = LossContainer(report_vars, len(self.loader), self.loss_weights)
        self.losses[VAL_TAG] = LossContainer(report_vars, len(self.val_loader), self.loss_weights)
        self.fixed_val_viz_files = []

        # DETR-head bookkeeping for component-wise TB logging and val-overlay viz.
        # _detr_loss_buf accumulates per-batch scalars within an epoch; flushed at epoch end.
        self._detr_loss_buf = {TRAIN_TAG: {}, VAL_TAG: {}}
        # GNN-head equivalent accumulator (only used when --e2e_mode=gnn).
        self._gnn_loss_buf = {TRAIN_TAG: {}, VAL_TAG: {}}
        self._soft_nms_buf = {TRAIN_TAG: {}, VAL_TAG: {}}
        self._soft_geom_buf = {TRAIN_TAG: {}, VAL_TAG: {}}
        self._gnn_adj_buf = {TRAIN_TAG: {}, VAL_TAG: {}}
        # Center-head accumulator (used when --e2e_mode=center).
        self._center_loss_buf = {TRAIN_TAG: {}, VAL_TAG: {}}
        # _last_e2e_out_by_file caches the latest e2e_out dict per filename in the
        # current val pass so plot_val_summary can overlay either K=20 Bézier curves
        # (DETR) or the kNN connectivity graph (GNN).
        self._last_e2e_out_by_file = {}
        self._e2e_mode = str(getattr(self.args, "e2e_mode", "detr")).lower()

    @staticmethod
    def __init_loss_weights__(num_train_vars, cuda, loss_weighting: LossWeighting, is_exponential,
                              init_weights=None, average_num_lines=120, num_cells=120, num_preds=8):
        return TrainHandler.__init_general_loss_weights(num_train_vars, cuda, loss_weighting, is_exponential,
                                                        init_weights, average_num_lines,
                                                        num_cells, num_preds, ignore_weight=-1)

    @staticmethod
    def __init_general_loss_weights(num_train_vars, cuda, loss_weighting: LossWeighting, is_exponential,
                                    init_weight_factor=None, average_num_lines=120, num_cells=120, num_preds=8,
                                    ignore_weight=None):
        # initialize
        if init_weight_factor is None:
            init_weight_factor = [1] * num_train_vars

        if "calculate" in init_weight_factor:
            # reduce by number of geom vars = 4
            init_weight_factor = [((num_cells * num_preds) / average_num_lines) / 4, *[1.] * (num_train_vars - 1)]
            Log.info(f"We calculated the weights={init_weight_factor}, "
                        f"we had ({num_cells} * {num_preds}) / {average_num_lines}")

        init_weight_factor = torch.tensor(init_weight_factor)

        # calculate trainable weight
        if loss_weighting == LossWeighting.FIXED_NORM:
            train_weights = init_weight_factor / sum(init_weight_factor)
            Log.info(f"Final weights={train_weights} with strategy {loss_weighting}")

        elif loss_weighting == LossWeighting.FIXED:
            train_weights = init_weight_factor
            Log.info(f"Final weights={train_weights} with strategy {loss_weighting}")

        elif loss_weighting == LossWeighting.LEARN_LOG or loss_weighting == LossWeighting.LEARN_LOG_NORM:
            train_weights = torch.zeros((num_train_vars), dtype=float)
            for i in range(num_train_vars):

                if loss_weighting == LossWeighting.LEARN_LOG_NORM:
                    wi = init_weight_factor[i] / sum(init_weight_factor)  # init also with normed values
                else:
                    wi = init_weight_factor[i]

                if is_exponential[i]:
                    train_weights[i] = torch.log(1 / wi)
                else:
                    train_weights[i] = torch.log(1 / (2. * wi))

            # conf weight should not be trained if we also have individual conf weights
            if ignore_weight is not None:
                if ignore_weight == -1:
                    ignore_weight = num_train_vars - 1
                train_weights[ignore_weight] = 1

            Log.info(f"Final trainable weights s=log(sigma**2)={train_weights.numpy()} "
                        f"with strategy {loss_weighting}")
            Log.info(f"Final weight factor w=(1 / (2*e^s))={(1 / (2 * torch.exp(train_weights))).numpy()} "
                        f"with strategy {loss_weighting}")

        elif loss_weighting == LossWeighting.LEARN:
            train_weights = torch.ones((num_train_vars), dtype=float)

            # conf weight should not be trained if we also have individual conf weights
            if ignore_weight is not None:
                if ignore_weight == -1:
                    ignore_weight = num_train_vars - 1
                train_weights[ignore_weight] = 1

            Log.info(f"Final trainable weights s={train_weights.numpy()} "
                        f"with strategy {loss_weighting}")
            Log.info(f"Final weight factor w=(1 / (2*s^2))={(1 / (2 * (train_weights ** 2))).numpy()} "
                        f"with strategy {loss_weighting}")
        else:
            raise NotImplementedError(f"We do not know {loss_weighting}")

        if len(train_weights) != num_train_vars:
            raise ValueError("You want %s variables, but provided a list of %s" % (num_train_vars, train_weights))

        # make optimizable if necessary
        do_training = [(loss_weighting == LossWeighting.LEARN_LOG or loss_weighting == LossWeighting.LEARN
                        or loss_weighting == LossWeighting.LEARN_LOG_NORM)
                       and (ignore_weight is None or ignore_weight != i)
                       for i in range(num_train_vars)]
        grad_train_weights = [torch.tensor((train_weights[i]), dtype=torch.float,
                                           # conf weight should not be trained if we also have individual conf weights
                                           requires_grad=do_training[i],
                                           device=cuda) for i in range(num_train_vars)]
        return grad_train_weights

    @staticmethod
    def __init_conf_loss_weights__(cuda, loss_weighting: LossWeighting, is_exponential,
                                   init_weights=None, average_num_lines=120, num_cells=120, num_preds=8):
        return TrainHandler.__init_general_loss_weights(num_train_vars=2, cuda=cuda, loss_weighting=loss_weighting,
                                                        init_weight_factor=init_weights,
                                                        average_num_lines=average_num_lines,
                                                        num_cells=num_cells, num_preds=num_preds,
                                                        is_exponential=is_exponential)

    def __call__(self, filenames, images, grid_tensor, epoch, image_idx_in_batch, is_train=True, first_run=False,
                 e2e_gt_pack=None):
        if is_train:
            self.optimizer.zero_grad()
        
        # 1. 모델이 반환하는 2개의 텐서를 각각 받습니다.
        use_amp = bool(getattr(self.args, "amp", True) and self.args.gpu)
        with torch.cuda.amp.autocast(enabled=use_amp):
            geom_preds, embed_preds, geom_logits, _e2e_out = self.forward(images, is_train=is_train, epoch=epoch,
                                                                   first_run=first_run,
                                                                   e2e_gt_pack=e2e_gt_pack)

        if image_idx_in_batch == 0 and epoch == 0:
            # 시각화/디버깅은 형태(Geometry) 정보만 필요하므로 geom_preds를 넘깁니다.
            self.on_data_loaded(filenames[0], images[0], grid_tensor[0], geom_preds[0].detach().cpu(), is_train=is_train)
            Log.debug("Data reporting finished..")

        # Align tensors with model device (substring checks fail: e.g. "cuda" in "cuda:0").
        tgt = next(self.forward.model.parameters()).device
        if geom_preds.device != tgt:
            geom_preds = geom_preds.to(tgt)
            if embed_preds is not None:
                embed_preds = embed_preds.to(tgt)
        if grid_tensor.device != tgt:
            grid_tensor = grid_tensor.to(tgt)

        # (필요 없는 기존 시각화 블록 False 처리된 부분은 그대로 두거나 삭제하셔도 무방합니다)

        loss_weight_dict = {}
        # Keep defaults so eval can continue even if one batch hits indef/nan handling.
        losses, mean_losses = {}, {}
        sum_loss = torch.tensor(float("nan"), device=geom_preds.device)
        try:
            # 3. 로스 래퍼 함수(self.loss)에 두 텐서를 모두 넘겨줍니다.
            with torch.cuda.amp.autocast(enabled=use_amp):
                losses, sum_loss, mean_losses = self.loss(
                    grid_tensor, geom_preds, embed_preds, epoch=epoch, filenames=filenames,
                    tag=TRAIN_TAG if is_train else VAL_TAG,
                    geom_logits=geom_logits)
            if _e2e_out is not None:
                # Dispatch by output keys: DETR head emits bezier_curve_px, GNN head emits edge_logits,
                # Center head emits center_logits + polylines_px.
                is_gnn = "edge_logits" in _e2e_out
                is_center = (not is_gnn) and ("center_logits" in _e2e_out) and ("polylines_px" in _e2e_out)
                if is_gnn:
                    e2e_aux = (
                        _e2e_out["edge_logits"].sum()
                        + _e2e_out["node_feat"].sum()
                    ) * 0.0
                    w_gnn = float(getattr(self.args, "gnn_loss_weight", 0.0) or 0.0)
                    use_matched_gnn = bool(getattr(self.args, "gnn_use_matched_gt_from_geom", False))
                    gnn_sup = (
                        self._build_gnn_matched_supervision(_e2e_out, geom_preds.device)
                        if use_matched_gnn else None
                    )
                    can_gnn_loss = w_gnn > 0.0 and (
                        e2e_gt_pack is not None
                        or (use_matched_gnn and gnn_sup is not None)
                    )
                    if can_gnn_loss:
                        pack_on_dev = None
                        if e2e_gt_pack is not None:
                            pack_on_dev = {
                                k: v.to(geom_preds.device, non_blocking=True)
                                for k, v in e2e_gt_pack.items()
                            }
                        gnn_d = compute_gnn_e2e_loss(
                            _e2e_out, pack_on_dev,
                            int(images.shape[2]), int(images.shape[3]),
                            int(epoch), self.args,
                            matched_supervision=gnn_sup,
                        )
                        sum_loss = sum_loss + gnn_d["total"]
                        self._record_gnn_components(gnn_d, tag=TRAIN_TAG if is_train else VAL_TAG)
                    else:
                        sum_loss = sum_loss + e2e_aux
                    if isinstance(_e2e_out.get("soft_nms_stats"), dict):
                        self._record_soft_nms_stats(
                            _e2e_out["soft_nms_stats"],
                            tag=TRAIN_TAG if is_train else VAL_TAG,
                        )
                    if isinstance(_e2e_out.get("soft_geom_stats"), dict):
                        self._record_soft_geom_stats(
                            _e2e_out["soft_geom_stats"],
                            tag=TRAIN_TAG if is_train else VAL_TAG,
                        )
                    if isinstance(_e2e_out.get("gnn_adj_stats"), dict):
                        self._record_gnn_adj_stats(
                            _e2e_out["gnn_adj_stats"],
                            tag=TRAIN_TAG if is_train else VAL_TAG,
                        )
                elif is_center:
                    # DDP autograd stub: ensure every center-head parameter sees a gradient
                    # even when GT/warmup isn't available yet.
                    e2e_aux = (
                        _e2e_out["center_logits"].sum()
                        + _e2e_out["polylines_px"].sum()
                        + _e2e_out["node_feat"].sum()
                    ) * 0.0
                    w_center = float(getattr(self.args, "center_loss_weight", 0.0) or 0.0)
                    # Warmup ramp on center loss weight.
                    warm_start = int(getattr(self.args, "center_warmup_start_epoch", 0) or 0)
                    warm_eps = int(getattr(self.args, "center_warmup_epochs", 0) or 0)
                    if warm_eps > 0 and int(epoch) >= warm_start:
                        scale = min(1.0, max(0.0, (int(epoch) - warm_start + 1) / float(warm_eps)))
                    elif warm_eps > 0 and int(epoch) < warm_start:
                        scale = 0.0
                    else:
                        scale = 1.0
                    w_center_eff = w_center * scale
                    if e2e_gt_pack is not None and w_center_eff > 0.0:
                        pack_on_dev = {k: v.to(geom_preds.device, non_blocking=True) for k, v in e2e_gt_pack.items()}
                        # Build a parameter list view for the DDP zero-stub touch inside the criterion.
                        head = getattr(self.forward.model, "e2e_head", None)
                        params_view = list(head.parameters()) if head is not None else None
                        stride_f = float(
                            getattr(self.forward.model, "scale", 0.0)
                            or (int(images.shape[2]) // max(1, int(_e2e_out["center_logits"].shape[2])))
                        )
                        center_total, center_d = compute_center_e2e_loss(
                            _e2e_out, pack_on_dev, self.args,
                            int(images.shape[2]), int(images.shape[3]),
                            stride=stride_f, model_params=params_view,
                        )
                        sum_loss = sum_loss + w_center_eff * center_total
                        center_d["center/total"] = float(center_total.detach().item())
                        center_d["center/weight_eff"] = float(w_center_eff)
                        self._record_center_components(center_d, tag=TRAIN_TAG if is_train else VAL_TAG)
                    else:
                        sum_loss = sum_loss + e2e_aux
                else:
                    # DETR head path (unchanged for legacy detr/seg_detr; also serves
                    # the exp51 ``hough_detr`` head — it emits ``polylines_px`` /
                    # ``bezier_curve_px`` / ``objectness_logits`` plus DN extras that
                    # ``compute_detr_e2e_loss`` consumes when present).
                    e2e_aux_terms = [
                        _e2e_out["bezier_curve_px"].sum(),
                        _e2e_out["objectness_logits"].sum(),
                    ]
                    if isinstance(_e2e_out.get("dn_polylines_px", None), torch.Tensor) and _e2e_out["dn_polylines_px"].numel() > 0:
                        e2e_aux_terms.append(_e2e_out["dn_polylines_px"].sum())
                    if isinstance(_e2e_out.get("ref_pts_layers", None), torch.Tensor) and _e2e_out["ref_pts_layers"].numel() > 0:
                        e2e_aux_terms.append(_e2e_out["ref_pts_layers"].sum())
                    e2e_aux = sum(e2e_aux_terms) * 0.0
                    w_e2e = float(getattr(self.args, "e2e_loss_weight", 0.0) or 0.0)
                    if e2e_gt_pack is not None and w_e2e > 0.0:
                        pack_on_dev = {k: v.to(geom_preds.device, non_blocking=True) for k, v in e2e_gt_pack.items()}
                        detr_d = compute_detr_e2e_loss(
                            _e2e_out, pack_on_dev,
                            int(images.shape[2]), int(images.shape[3]),
                            int(epoch), self.args,
                        )
                        sum_loss = sum_loss + detr_d["total"]
                        self._record_detr_components(detr_d, tag=TRAIN_TAG if is_train else VAL_TAG)
                    else:
                        sum_loss = sum_loss + e2e_aux

                # Cache the e2e dict per filename for plot_val_summary overlay.
                if not is_train:
                    if is_center:
                        # Move to CPU lazily and slice per-batch for the overlay drawer.
                        center_cached = {
                            "center_logits": _e2e_out["center_logits"].detach().cpu(),
                            "center_peaks_xy": _e2e_out["center_peaks_xy"].detach().cpu(),
                            "peak_score": _e2e_out["peak_score"].detach().cpu(),
                            "peak_valid": _e2e_out["peak_valid"].detach().cpu(),
                            "polylines_px": _e2e_out["polylines_px"].detach().cpu(),
                        }
                        for bi, fname in enumerate(filenames):
                            self._last_e2e_out_by_file[str(fname)] = {
                                "kind": "center",
                                "center_logits": center_cached["center_logits"][bi],
                                "center_peaks_xy": center_cached["center_peaks_xy"][bi],
                                "peak_score": center_cached["peak_score"][bi],
                                "peak_valid": center_cached["peak_valid"][bi],
                                "polylines_px": center_cached["polylines_px"][bi],
                            }
                    elif is_gnn:
                        gnn_cached = {
                            "node_mid_px": _e2e_out["node_mid_px"].detach().cpu(),
                            "node_end_a_px": _e2e_out["node_end_a_px"].detach().cpu(),
                            "node_end_b_px": _e2e_out["node_end_b_px"].detach().cpu(),
                            "node_valid": _e2e_out["node_valid"].detach().cpu(),
                            "node_conf": _e2e_out["node_conf"].detach().cpu(),
                            "neighbors": _e2e_out["neighbors"].detach().cpu(),
                            "neigh_valid": _e2e_out["neigh_valid"].detach().cpu(),
                            "edge_logits": _e2e_out["edge_logits"].detach().cpu(),
                        }
                        if "line_neighbors" in _e2e_out:
                            gnn_cached["line_neighbors"] = _e2e_out["line_neighbors"].detach().cpu()
                            gnn_cached["line_neigh_valid"] = _e2e_out["line_neigh_valid"].detach().cpu()
                        soft_dbg = _e2e_out.get("soft_nms_debug", None)
                        for bi, fname in enumerate(filenames):
                            entry = {
                                "kind": "gnn",
                                "node_mid_px": gnn_cached["node_mid_px"][bi],
                                "node_end_a_px": gnn_cached["node_end_a_px"][bi],
                                "node_end_b_px": gnn_cached["node_end_b_px"][bi],
                                "node_valid": gnn_cached["node_valid"][bi],
                                "neighbors": gnn_cached["neighbors"][bi],
                                "neigh_valid": gnn_cached["neigh_valid"][bi],
                                "edge_logits": gnn_cached["edge_logits"][bi],
                                "node_conf": gnn_cached["node_conf"][bi],
                            }
                            if "line_neighbors" in gnn_cached:
                                entry["line_neighbors"] = gnn_cached["line_neighbors"][bi]
                                entry["line_neigh_valid"] = gnn_cached["line_neigh_valid"][bi]
                            if isinstance(soft_dbg, dict):
                                entry["soft_nms_debug"] = {
                                    k: (v[bi].detach().cpu() if isinstance(v, torch.Tensor) else v)
                                    for k, v in soft_dbg.items()
                                }
                            self._last_e2e_out_by_file[str(fname)] = entry
                    else:
                        is_hough = (
                            isinstance(_e2e_out.get("hough_anchors", None), dict)
                            and "ref_points_px_init" in _e2e_out
                        )
                        if is_hough:
                            ha = _e2e_out["hough_anchors"]
                            hough_cached = {
                                "polylines_px": _e2e_out["polylines_px"].detach().cpu(),
                                "objectness_logits": _e2e_out["objectness_logits"].detach().cpu(),
                                "ref_points_px_init": _e2e_out["ref_points_px_init"].detach().cpu(),
                                "anchors_valid": ha["valid"].detach().cpu(),
                            }
                            for bi, fname in enumerate(filenames):
                                self._last_e2e_out_by_file[str(fname)] = {
                                    "kind": "hough",
                                    "polylines_px": hough_cached["polylines_px"][bi],
                                    "objectness_logits": hough_cached["objectness_logits"][bi],
                                    "ref_points_px_init": hough_cached["ref_points_px_init"][bi],
                                    "anchors_valid": hough_cached["anchors_valid"][bi],
                                }
                        else:
                            e2e_cached = {
                                "bezier_curve_px": _e2e_out["bezier_curve_px"].detach().cpu(),
                                "objectness_logits": _e2e_out["objectness_logits"].detach().cpu(),
                            }
                            for bi, fname in enumerate(filenames):
                                self._last_e2e_out_by_file[str(fname)] = {
                                    "kind": "detr",
                                    "bezier_curve_px": e2e_cached["bezier_curve_px"][bi],
                                    "objectness_logits": e2e_cached["objectness_logits"][bi],
                                }
            self._log_loss_diagnostics(losses, mean_losses, sum_loss, filenames, epoch, image_idx_in_batch)
            if is_train:
                self.backward(sum_loss, epoch=epoch)

                for j, v in enumerate(self.dataset.coords.train_vars()):
                    std_1 = torch.exp(self.loss_weights[j]) ** 0.5
                    Log.debug("Weight for %s is %f with std %f" % (v, self.loss_weights[j].item(), std_1.item()))
                    loss_weight_dict[os.path.join("loss_" + "conf" if v == Variables.CONF else str(v), "weight")] = \
                        self.loss_weights[j]

            self.losses[TRAIN_TAG if is_train else VAL_TAG].add_backprop(epoch=epoch,
                                                                         loss=sum_loss.detach().cpu().item(),
                                                                         i=image_idx_in_batch)
            self.losses[TRAIN_TAG if is_train else VAL_TAG].add(epoch=epoch, mean_losses=mean_losses,
                                                                config_losses=losses, i=image_idx_in_batch)
        except (ValueError, NotImplementedError, RuntimeError) as e:
            Log.debug(e)
            pred_grid, _ = GridFactory.get(torch.unsqueeze(geom_preds[0].detach().cpu(), dim=0), [],
                                           CoordinateSystem.CELL_SPLIT, self.args,
                                           input_coords=self.dataset.coords, threshold=self.args.confidence,
                                           only_train_vars=True, anchors=self.dataset.anchors)
            grid, _ = GridFactory.get(torch.unsqueeze(grid_tensor[0].cpu(), dim=0), [],
                                      CoordinateSystem.CELL_SPLIT, self.args,
                                      input_coords=self.dataset.coords, threshold=self.args.confidence,
                                      only_train_vars=False, anchors=self.dataset.anchors)

            pred_grid_uv_lines = pred_grid.get_image_lines(coords=self.dataset.coords,
                                                           image_height=images[0].shape[1])

            if is_train:
                full_size_img = self.dataset.get_full_size_image(filename=filenames[0])
            else:
                full_size_img = self.val_dataset.get_full_size_image(filename=filenames[0])

            self.plot_debug_images(filenames[0], full_size_img,
                                   pred_grid_uv_lines, epoch,
                                   tag=TRAIN_TAG if is_train else VAL_TAG,
                                   suffix="indef",
                                   imageidx=ImageIdx.GRID,
                                   coordinates=CoordinateSystem.UV_SPLIT,
                                   gt=grid.get_image_lines(coords=self.dataset.coords,
                                                           image_height=full_size_img.shape[1]))

            Log.debug("Indef loss")
            self.on_indef(epoch, filenames, exception=e, is_train=is_train, batch_index=image_idx_in_batch)
            import math
        except AttributeError as e:
            Log.error("Something is wrong with these losses: %s. We continue..." % str(e))
            self.losses[TRAIN_TAG if is_train else VAL_TAG].add(epoch, mean_losses=mean_losses, config_losses=losses,
                                                                i=image_idx_in_batch)
        # sum_loss는 try 블록에서 계산되며, 실패 시 NaN 기본값을 반환합니다.
        return sum_loss.detach().cpu().item(), (geom_preds, embed_preds, _e2e_out)

    def _record_detr_components(self, detr_d, tag):
        """Append per-batch DETR loss components into the per-epoch accumulator."""
        buf = self._detr_loss_buf.setdefault(tag, {})
        for k in (
            "total", "l1", "endpoint", "obj", "chamfer", "straightness", "dn", "aux_layer",
            "n_matched", "n_gt", "lam",
        ):
            if k not in detr_d:
                continue
            v = detr_d[k]
            if isinstance(v, torch.Tensor):
                try:
                    v = float(v.detach().float().cpu().item())
                except Exception:
                    continue
            else:
                v = float(v)
            buf.setdefault(k, []).append(v)

    def _flush_detr_components(self, tag, epoch):
        """Publish mean DETR loss components for the epoch to TB, then reset the buffer."""
        buf = self._detr_loss_buf.get(tag, {})
        if not buf:
            return
        means = {}
        for k, vs in buf.items():
            if not vs:
                continue
            means["detr_loss/" + k] = float(np.mean(vs))
        if means:
            Log.scalars(tag=tag, dict=means, epoch=epoch)
        self._detr_loss_buf[tag] = {}

    def _record_gnn_components(self, gnn_d, tag):
        """Per-batch GNN loss/metric accumulator (mirrors _record_detr_components)."""
        buf = self._gnn_loss_buf.setdefault(tag, {})
        keys = (
            "total", "bce", "lam",
            "n_pos", "n_neg", "n_cross", "n_ignore", "n_kept",
            "n_pos_remote",
            "mean_edge_prob", "frac_pos_edges",
            "n_nodes_fg", "n_nodes_total", "n_nodes_matched", "n_nodes_unmatched",
            "n_edges_orphan_half_gt",
            "degree_loss", "rw_topology_loss", "chamfer",
            "instance_iou_mean",
            "n_pos_strict", "frac_pos_strict",
            "edge_feat_d_end_mean",
        )
        extra = [k for k in gnn_d if k.startswith(("edge_f1_", "edge_prec_", "edge_rec_"))]
        for k in keys + tuple(extra):
            if k not in gnn_d:
                continue
            v = gnn_d[k]
            if isinstance(v, torch.Tensor):
                try:
                    v = float(v.detach().float().cpu().item())
                except Exception:
                    continue
            else:
                v = float(v)
            buf.setdefault(k, []).append(v)

    def _record_soft_nms_stats(self, stats: dict, tag: str) -> None:
        """Accumulate per-batch soft-NMS counters for TensorBoard."""
        buf = self._soft_nms_buf.setdefault(tag, {})
        for k in ("n_prefilter", "n_processed", "n_above_floor_after", "mean_decay_ratio", "n_total"):
            if k not in stats:
                continue
            t = stats[k]
            if isinstance(t, torch.Tensor):
                vals = t.detach().float().cpu().tolist()
                if isinstance(vals, float):
                    vals = [vals]
            else:
                vals = [float(t)]
            buf.setdefault(k, []).extend(vals)

    def _record_soft_geom_stats(self, stats: dict, tag: str) -> None:
        """Accumulate soft geometric prior stats for TensorBoard."""
        buf = self._soft_geom_buf.setdefault(tag, {})
        for k in ("mean_prior", "frac_prior_gt_0.1"):
            if k not in stats:
                continue
            t = stats[k]
            if isinstance(t, torch.Tensor):
                buf.setdefault(k, []).append(float(t.detach().float().cpu().item()))
            else:
                buf.setdefault(k, []).append(float(t))

    def _flush_soft_geom_stats(self, tag, epoch) -> None:
        buf = self._soft_geom_buf.get(tag, {})
        if not buf:
            return
        means = {}
        for k, vs in buf.items():
            if vs:
                means["soft_geom/" + k] = float(np.mean(vs))
        if means:
            Log.scalars(tag=tag, dict=means, epoch=epoch)
        self._soft_geom_buf[tag] = {}

    def _flush_soft_nms_stats(self, tag, epoch) -> None:
        buf = self._soft_nms_buf.get(tag, {})
        if not buf:
            return
        means = {}
        for k, vs in buf.items():
            if vs:
                means["soft_nms/" + k] = float(np.mean(vs))
        if means:
            Log.scalars(tag=tag, dict=means, epoch=epoch)
        self._soft_nms_buf[tag] = {}

    def _record_gnn_adj_stats(self, stats: dict, tag: str) -> None:
        """Accumulate directional2_ctx adjacency stats for TensorBoard."""
        buf = self._gnn_adj_buf.setdefault(tag, {})
        for k in (
            "k_line", "k_gat", "k_ctx",
            "mean_line_neighbors", "mean_ctx_neighbors",
            "frac_nodes_with_ctx", "frac_ctx_slots_used",
        ):
            if k not in stats:
                continue
            v = stats[k]
            if isinstance(v, torch.Tensor):
                v = float(v.detach().float().cpu().item())
            else:
                v = float(v)
            buf.setdefault(k, []).append(v)

    def _flush_gnn_adj_stats(self, tag, epoch):
        buf = self._gnn_adj_buf.get(tag, {})
        if not buf:
            return
        means = {}
        for k, vs in buf.items():
            if vs:
                means["gnn_adj/" + k] = float(np.mean(vs))
        if means:
            Log.scalars(tag=tag, dict=means, epoch=epoch)
        self._gnn_adj_buf[tag] = {}

    def _flush_gnn_components(self, tag, epoch):
        """Publish per-epoch GNN means (parallels _flush_detr_components)."""
        buf = self._gnn_loss_buf.get(tag, {})
        if not buf:
            return
        means = {}
        edge_metrics = {}
        topo = {}
        for k, vs in buf.items():
            if not vs:
                continue
            val = float(np.mean(vs))
            if k == "instance_iou_mean":
                topo["gnn_topology/instance_iou_mean"] = val
            elif k.startswith(("edge_f1_", "edge_prec_", "edge_rec_")):
                edge_metrics["gnn_edge_metrics/" + k] = val
            else:
                means["gnn_loss/" + k] = val
        if means:
            Log.scalars(tag=tag, dict=means, epoch=epoch)
        if edge_metrics:
            Log.scalars(tag=tag, dict=edge_metrics, epoch=epoch)
        if topo:
            Log.scalars(tag=tag, dict=topo, epoch=epoch)
        self._gnn_loss_buf[tag] = {}

    def _record_center_components(self, center_d, tag):
        """Per-batch Center loss/metric accumulator (mirrors _record_detr_components)."""
        buf = self._center_loss_buf.setdefault(tag, {})
        for k, v in center_d.items():
            if isinstance(v, torch.Tensor):
                try:
                    v = float(v.detach().float().cpu().item())
                except Exception:
                    continue
            else:
                v = float(v)
            buf.setdefault(k, []).append(v)

    def _flush_center_components(self, tag, epoch):
        """Publish per-epoch Center means."""
        buf = self._center_loss_buf.get(tag, {})
        if not buf:
            return
        means = {}
        for k, vs in buf.items():
            if not vs:
                continue
            means[k] = float(np.mean(vs))
        if means:
            Log.scalars(tag=tag, dict=means, epoch=epoch)
        self._center_loss_buf[tag] = {}

    def on_indef(self, epoch, filenames, exception, batch_index, is_train=True):
        if self.best_mean_loss != np.inf or epoch > 10 or is_train:
            epoch_losses = self.losses[TRAIN_TAG if is_train else VAL_TAG]._backprops_[-1]
            Log.fail_summary(epoch, epoch_losses if len(epoch_losses) > 0 else [-1],
                             tag=TRAIN_TAG if is_train else VAL_TAG, level=1)
            raise Exception(filenames) from exception
        else:
            if epoch > 0:
                Log.warning("We ran into nan - continue")
            self.losses[TRAIN_TAG if is_train else VAL_TAG].add(epoch=epoch, mean_losses=torch.nan,
                                                                config_losses=torch.nan, i=batch_index)

    def on_data_loaded(self, filename, image, grid_tensor, preds, is_train=True):
        epoch = 0
        if image.shape[1] != self.args.img_size[0] or image.shape[2] != self.args.img_size[1]:
            raise ValueError("Image has %s, we want %s" % (image.shape, self.args.img_size))

        if filename in self.dataset.file_names:
            full_size_img = self.dataset.get_full_size_image(filename=filename)
        elif filename in self.val_dataset.file_names:
            full_size_img = self.val_dataset.get_full_size_image(filename=filename)
        else:
            full_size_img = image

        grid, _ = GridFactory.get(torch.unsqueeze(grid_tensor, dim=0), [], CoordinateSystem.CELL_SPLIT, self.args,
                                  input_coords=self.dataset.coords, threshold=self.args.confidence,
                                  anchors=self.dataset.anchors)

        grid_uv_lines = grid.get_image_lines(coords=self.dataset.coords, image_height=full_size_img.shape[1])

        if preds is not None:
            pred_grid, _ = GridFactory.get(torch.unsqueeze(preds, dim=0), [], CoordinateSystem.CELL_SPLIT,
                                           self.args, anchors=self.dataset.anchors,
                                           input_coords=self.dataset.coords, threshold=self.args.confidence,
                                           only_train_vars=True)
            pred_grid_uv_lines = pred_grid.get_image_lines(coords=self.dataset.coords,
                                                           image_height=full_size_img.shape[1], is_training_data=True)
            ok = self.plot_debug_images(filename, full_size_img,
                                        pred_grid_uv_lines, epoch,
                                        tag=TRAIN_TAG if is_train else VAL_TAG, suffix="preview",
                                        imageidx=ImageIdx.PRED,
                                        coordinates=CoordinateSystem.UV_SPLIT, gt=grid_uv_lines,
                                        training_vars_only=True, cell_size=grid.get_cell_size(full_size_img.shape[1]))
        else:
            ok = self.plot_debug_images(filename, full_size_img,
                                        grid_uv_lines, epoch, tag=TRAIN_TAG if is_train else VAL_TAG, suffix="preview",
                                        imageidx=ImageIdx.GRID,
                                        coordinates=CoordinateSystem.UV_SPLIT,
                                        cell_size=grid.get_cell_size(full_size_img.shape[1]))

        del grid

    def on_train_epoch_finished(self, epoch, filenames, images, preds, grid_tensors):
        if self.args.gpu:
            preds = preds.to("cpu")
            images = images.to("cpu")

        if bool(getattr(self.args, "log_cuda_mem_after_epoch", False)) and getattr(
                self.args, "is_main_process", True) and self.args.gpu:
            dev = torch.device(self.args.cuda)
            alloc_mb = torch.cuda.memory_allocated(dev) / 1e6
            peak_mb = torch.cuda.max_memory_allocated(dev) / 1e6
            Log.scalars(
                tag=TRAIN_TAG,
                dict={"cuda/mem_alloc_mb": alloc_mb, "cuda/mem_peak_mb": peak_mb},
                epoch=epoch,
            )
            torch.cuda.reset_peak_memory_stats(dev)

        if self.losses[TRAIN_TAG].current_epoch == epoch:
            self.losses[TRAIN_TAG].log(epoch=epoch, tag=TRAIN_TAG)
            sum_loss = self.losses[TRAIN_TAG].sum(epoch=epoch)
            if sum_loss > 4 * abs(self.last_loss):
                Log.glitch(self.args, preds)
            self.last_loss = sum_loss
        else:
            Log.error("Wrong epoch")

        if self.is_time_for_image(epoch):
            self.plot_prediction(epoch, filenames, grid_tensors, images, preds, indices=[0])  # range(len(filenames)))

        scores = []
        if self.is_time_for_val(epoch):
            scores = self.evaluator.publish_scores(epoch=epoch, tag=TRAIN_TAG)

        # Publish per-component DETR / GNN / Center losses (means over the epoch).
        self._flush_detr_components(TRAIN_TAG, epoch)
        self._flush_gnn_components(TRAIN_TAG, epoch)
        self._flush_gnn_adj_stats(TRAIN_TAG, epoch)
        self._flush_soft_nms_stats(TRAIN_TAG, epoch)
        self._flush_soft_geom_stats(TRAIN_TAG, epoch)
        self._flush_center_components(TRAIN_TAG, epoch)

        # Checkpoint
        save_checkpoint(self.args, self.model, self.optimizer, self.scheduler, epoch, self.args.id)

        if False:
            self.plot_heatmap(self.dataset.anchors.conf_heatmap, "conf")
            self.plot_heatmap(self.dataset.anchors.heatmap)

        Log.push(epoch)
        return scores

    def plot_heatmap(self, heatmap, title=""):
        import seaborn as sns
        sns_dict = {"anchor": [], "heat": [], "row": [], "col": []}
        for r in range(len(heatmap)):
            for c in range(len(heatmap[r])):
                for a in range(len(heatmap[r, c])):
                    sns_dict["heat"].append(heatmap[r, c, a].item())
                    sns_dict["anchor"].append(a)
                    sns_dict["row"].append(r)
                    sns_dict["col"].append(c)
        import pandas as pd
        sns_df = pd.DataFrame(sns_dict)
        g = sns.FacetGrid(sns_df, col="anchor")

        def draw_heatmap(*args, **kwargs):
            data = kwargs.pop('data')
            d = data.pivot(index=args[1], columns=args[0], values=args[2])
            sns.heatmap(d, **kwargs)

        g.map_dataframe(draw_heatmap, "col", "row", "heat")
        name = "/tmp/seaborn_heatmap_%s.png" % title
        Log.info("Log to file://%s" % name)
        plt.savefig(name)

    def plot_prediction(self, epoch, filenames, grid_tensors, images, preds, indices,
                        image_idx: ImageIdx = ImageIdx.PRED):
        for idx in indices:
            grid, _ = GridFactory.get(data=torch.unsqueeze(preds[idx], dim=0), variables=[],
                                      coordinate=CoordinateSystem.CELL_SPLIT, args=self.args,
                                      input_coords=self.dataset.coords, only_train_vars=True,
                                      anchors=self.dataset.anchors)
            gt_grid, _ = GridFactory.get(data=torch.unsqueeze(grid_tensors[idx], dim=0), variables=[],
                                         coordinate=CoordinateSystem.CELL_SPLIT, args=self.args,
                                         input_coords=self.dataset.coords, only_train_vars=False,
                                         anchors=self.dataset.anchors)

            if Variables.CLASS in self.dataset.coords.train_vars():
                ok = self.plot_debug_class_image(filenames[idx], images[idx], grid, epoch=epoch, tag=TRAIN_TAG,
                                                 imageidx=image_idx, ignore_classes=[0])
            try:
                full_size_img = self.dataset.get_full_size_image(filename=filenames[idx])
            except FileNotFoundError as e:
                full_size_img = images[idx]
            self.plot_debug_images(full_basename=filenames[idx], image=full_size_img,
                                   uv_lines=grid.get_image_lines(coords=self.dataset.coords,
                                                                 image_height=full_size_img.shape[1],
                                                                 is_training_data=True),
                                   epoch=epoch,
                                   tag=TRAIN_TAG, show_grid=True, imageidx=image_idx,
                                   coordinates=CoordinateSystem.UV_SPLIT,
                                   gt=gt_grid.get_image_lines(coords=self.dataset.coords,
                                                              image_height=full_size_img.shape[1]),
                                   training_vars_only=True, cell_size=grid.get_cell_size(full_size_img.shape[1]))

    def on_images_finished(self, preds, grid_tensor, filenames, images, epoch, is_train, num_duplicates):
        if self.is_time_for_val(epoch):
            if torch.all(preds.isnan()):
                Log.error("Prediction is nan. Continue.")
                return

            # When eval_iteration==1, is_time_for_val is true every epoch; running full cell matching + metrics on
            # every *training* batch made each step take minutes (rank 0 CPU-bound, other ranks wait on DDP sync).
            # Metrics for monitoring are still computed in the validation loop (is_train=False) below.
            if is_train:
                return

            if self.args.full_eval:
                try:
                    preds_uv, gt_uv = self.evaluator.prepare_uv(preds=preds, grid_tensors=grid_tensor,
                                                                filenames=filenames,
                                                                images=images)
                except ValueError as ex:
                    if epoch <= 5:
                        Log.error("We ran in to nan - continue for now")
                        return
                    else:
                        raise ex

                self.evaluator.get_scores_uv(gt_uv=gt_uv, preds_uv=preds_uv, epoch=epoch, filenames=filenames,
                                             num_duplicates=num_duplicates, tag=TRAIN_TAG if is_train else VAL_TAG,
                                             do_matching=True)
            else:
                self.evaluator.get_scores_in_cell(grid_tensor, preds, epoch, filenames, num_duplicates=num_duplicates,
                                                  tag=TRAIN_TAG if is_train else VAL_TAG,
                                                  do_matching=self.args.anchors == AnchorDistribution.NONE)

        if not is_train:
            self.plot_val_summary(epoch, filenames, grid_tensor, images, preds)

    def plot_val_summary(self, epoch, filenames, grid_tensor, images, preds):
        if self.args.gpu:
            preds = preds.to("cpu")
            images = images.to("cpu")
        if len(self.fixed_val_viz_files) < 10:
            for fname in filenames:
                if fname not in self.fixed_val_viz_files:
                    self.fixed_val_viz_files.append(fname)
                if len(self.fixed_val_viz_files) >= 10:
                    break

        for viz_file_name in self.fixed_val_viz_files:
            if viz_file_name in filenames:
                idx = np.where(viz_file_name == np.asarray(filenames))[0][0]
            else:
                continue

            grid, _ = GridFactory.get(data=torch.unsqueeze(preds[idx], dim=0), variables=[],
                                      coordinate=CoordinateSystem.CELL_SPLIT, args=self.args,
                                      input_coords=self.dataset.coords, only_train_vars=True,
                                      anchors=self.dataset.anchors)
            gt_grid, _ = GridFactory.get(data=torch.unsqueeze(grid_tensor[idx], dim=0), variables=[],
                                         coordinate=CoordinateSystem.CELL_SPLIT, args=self.args,
                                         input_coords=self.dataset.coords, only_train_vars=False,
                                         anchors=self.dataset.anchors)
            cell_size = grid.get_cell_size(images[idx].shape[1])
            self.plot_debug_images(full_basename=filenames[idx], image=images[idx],
                                   uv_lines=grid.get_image_lines(coords=self.dataset.coords,
                                                                 image_height=images[idx].shape[1],
                                                                 is_training_data=True), epoch=epoch,
                                   tag=VAL_TAG, show_grid=True, imageidx=ImageIdx.PRED,
                                   coordinates=CoordinateSystem.UV_SPLIT,
                                   gt=gt_grid.get_image_lines(coords=self.dataset.coords,
                                                              image_height=images[idx].shape[1]),
                                   training_vars_only=True, cell_size=cell_size)

            # Additional overlay: dispatch by head kind cached in the val pass.
            cached = self._last_e2e_out_by_file.get(str(filenames[idx]), None)
            if cached is not None:
                kind = cached.get("kind", "detr") if isinstance(cached, dict) else "detr"
                if kind == "detr" and "bezier_curve_px" in cached:
                    self._plot_detr_overlay(
                        full_basename=filenames[idx], image=images[idx], e2e_single=cached, epoch=epoch
                    )
                elif kind == "gnn" and "edge_logits" in cached:
                    if cached.get("soft_nms_debug") is not None:
                        self._plot_gnn_soft_nms(
                            full_basename=filenames[idx], image=images[idx], gnn_single=cached, epoch=epoch
                        )
                    self._plot_gnn_edges(
                        full_basename=filenames[idx], image=images[idx], gnn_single=cached, epoch=epoch
                    )
                elif kind == "center" and "polylines_px" in cached:
                    self._plot_center_overlay(
                        full_basename=filenames[idx], image=images[idx], center_single=cached, epoch=epoch
                    )
                elif kind == "hough" and "polylines_px" in cached:
                    self._plot_hough_overlay(
                        full_basename=filenames[idx], image=images[idx], hough_single=cached, epoch=epoch
                    )

        # Clear per-batch e2e cache once we've drawn what we needed; the val pass will
        # re-populate it on the next epoch.
        self._last_e2e_out_by_file.clear()

    def _plot_hough_overlay(self, full_basename, image, hough_single, epoch):
        """Draw exp51 5-pt polylines (cyan, final) + initial Hough 5-pt ref (yellow, dashed)."""
        viz_image = self._denormalize_for_viz(image)
        pl = hough_single["polylines_px"]
        if isinstance(pl, torch.Tensor):
            pl = pl.detach().cpu().numpy()
        ref_init = hough_single.get("ref_points_px_init", None)
        if isinstance(ref_init, torch.Tensor):
            ref_init = ref_init.detach().cpu().numpy()
        obj_t = hough_single.get("objectness_logits", None)
        obj_prob = None
        if obj_t is not None:
            if isinstance(obj_t, torch.Tensor):
                obj_t = obj_t.detach().cpu().float()
            else:
                obj_t = torch.tensor(obj_t, dtype=torch.float32)
            obj_prob = torch.sigmoid(obj_t).numpy()
        valid_anchors = hough_single.get("anchors_valid", None)
        if isinstance(valid_anchors, torch.Tensor):
            valid_anchors = valid_anchors.detach().cpu().numpy().astype(bool)
        ov_thresh = float(getattr(self.args, "e2e_overlay_objectness_thresh", 0.35) or 0.0)

        # --- final polylines (cyan) ---
        valid_curves = []
        for q in range(pl.shape[0]):
            if obj_prob is not None and ov_thresh > 0.0 and float(obj_prob[q]) < ov_thresh:
                continue
            if valid_anchors is not None and not bool(valid_anchors[q]):
                # Slot had no Hough cluster → skip to keep the overlay focused on real preds.
                continue
            pts = pl[q]
            if np.all(pts == 0) or np.any(np.isnan(pts)):
                continue
            valid_curves.append(pts)
        if valid_curves:
            img_path = self.args.paths.generate_debug_image_file_path(
                full_basename, ImageIdx.PRED, suffix="hough_e2e_" + VAL_TAG
            )
            plot(
                [valid_curves], name=img_path,
                image=viz_image.clone() if isinstance(viz_image, torch.Tensor) else viz_image,
                coords=self.dataset.coords, show_grid=False, cell_size=None,
                threshold=0.0, coordinates=CoordinateSystem.UV_CONTINUOUS,
                colorstyle=ColorStyle.UNIFORM, color=(0, 255, 255), thickness=2,
                tag=VAL_TAG + "/hough_e2e", epoch=epoch, imageidx=ImageIdx.PRED,
                anchors=self.dataset.anchors,
            )

        # --- initial Hough 5-pt refs (yellow) ---
        if ref_init is not None:
            init_curves = []
            for q in range(ref_init.shape[0]):
                if valid_anchors is not None and not bool(valid_anchors[q]):
                    continue
                pts = ref_init[q]
                if np.all(pts == 0) or np.any(np.isnan(pts)):
                    continue
                init_curves.append(pts)
            if init_curves:
                img_path = self.args.paths.generate_debug_image_file_path(
                    full_basename, ImageIdx.PRED, suffix="hough_anchor_" + VAL_TAG
                )
                plot(
                    [init_curves], name=img_path,
                    image=viz_image.clone() if isinstance(viz_image, torch.Tensor) else viz_image,
                    coords=self.dataset.coords, show_grid=False, cell_size=None,
                    threshold=0.0, coordinates=CoordinateSystem.UV_CONTINUOUS,
                    colorstyle=ColorStyle.UNIFORM, color=(0, 220, 255), thickness=1,
                    tag=VAL_TAG + "/hough_anchor", epoch=epoch, imageidx=ImageIdx.PRED,
                    anchors=self.dataset.anchors,
                )

    def _plot_detr_overlay(self, full_basename, image, e2e_single, epoch):
        """Draw Bézier curves (cyan) for queries passing objectness + sanity checks."""
        viz_image = self._denormalize_for_viz(image)
        # bezier_curve_px is [K, T, 2] in pixel (x, y); plot() expects a "batch of
        # instances" so wrap in another list at axis 0.
        cr = e2e_single["bezier_curve_px"]
        if isinstance(cr, torch.Tensor):
            cr = cr.detach().cpu().numpy()
        obj_t = e2e_single.get("objectness_logits", None)
        obj_prob = None
        if obj_t is not None:
            if isinstance(obj_t, torch.Tensor):
                obj_t = obj_t.detach().cpu().float()
            else:
                obj_t = torch.tensor(obj_t, dtype=torch.float32)
            obj_prob = torch.sigmoid(obj_t).numpy()
        ov_thresh = float(getattr(self.args, "e2e_overlay_objectness_thresh", 0.35) or 0.0)
        # Filter out queries with all-zero / NaN curves (defensive).
        valid_curves = []
        for q in range(cr.shape[0]):
            if obj_prob is not None and ov_thresh > 0.0 and float(obj_prob[q]) < ov_thresh:
                continue
            pts = cr[q]
            if np.all(pts == 0) or np.any(np.isnan(pts)):
                continue
            valid_curves.append(pts)
        if not valid_curves:
            return
        e2e_lines = [valid_curves]  # [[K_valid, T, 2]] -> plot() iterates [b][i]
        img_path = self.args.paths.generate_debug_image_file_path(
            full_basename, ImageIdx.PRED, suffix="detr_e2e_" + VAL_TAG
        )
        plot(
            e2e_lines, name=img_path, image=viz_image.clone() if isinstance(viz_image, torch.Tensor) else viz_image,
            coords=self.dataset.coords, show_grid=False, cell_size=None,
            threshold=0.0, coordinates=CoordinateSystem.UV_CONTINUOUS,
            colorstyle=ColorStyle.UNIFORM, color=(0, 255, 255), thickness=2,
            tag=VAL_TAG + "/detr_e2e", epoch=epoch, imageidx=ImageIdx.PRED,
            anchors=self.dataset.anchors,
        )

    def _plot_gnn_soft_nms(self, full_basename, image, gnn_single, epoch):
        """TensorBoard: before/after soft-NMS segment overlays (conf as alpha)."""
        import cv2

        dbg = gnn_single.get("soft_nms_debug")
        if not isinstance(dbg, dict):
            return

        mid = dbg["mid_px"]
        ea = dbg["end_a_px"]
        eb = dbg["end_b_px"]
        conf_b = dbg["conf_before"]
        conf_a = dbg["conf_after"]
        thresh = float(dbg.get("viz_thresh", 0.1))

        if isinstance(mid, torch.Tensor):
            mid = mid.numpy()
            ea = ea.numpy()
            eb = eb.numpy()
            conf_b = conf_b.numpy()
            conf_a = conf_a.numpy()

        viz = self._denormalize_for_viz(image)
        if isinstance(viz, torch.Tensor):
            chw = viz.detach().cpu().float().numpy()
            canvas = np.transpose(chw, (1, 2, 0)).copy()
        else:
            canvas = np.asarray(viz).copy()
        if canvas.dtype != np.uint8:
            canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        if canvas.shape[-1] == 3:
            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        else:
            canvas_bgr = canvas

        def _draw_panel(bgr, conf, title_suffix, color_bgr):
            out = bgr.copy()
            n_draw = 0
            for i in range(int(conf.shape[0])):
                c = float(conf[i])
                if c < thresh:
                    continue
                n_draw += 1
                a = (int(round(float(ea[i, 0]))), int(round(float(ea[i, 1]))))
                b = (int(round(float(eb[i, 0]))), int(round(float(eb[i, 1]))))
                alpha = float(np.clip(c, 0.15, 1.0))
                overlay = out.copy()
                cv2.line(overlay, a, b, color_bgr, 2, lineType=cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, dst=out)
            cv2.putText(
                out, "%s n>=%.2f: %d" % (title_suffix, thresh, n_draw),
                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA,
            )
            return out, n_draw

        left, n_b = _draw_panel(canvas_bgr, conf_b, "before soft-NMS", (80, 200, 255))
        right, n_a = _draw_panel(canvas_bgr, conf_a, "after soft-NMS", (80, 255, 120))
        combo = np.concatenate([left, right], axis=1)
        cv2.putText(
            combo, "dup removed via conf decay (not hard NMS)",
            (8, combo.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA,
        )

        img_path = self.args.paths.generate_debug_image_file_path(
            full_basename, ImageIdx.PRED, suffix="gnn_soft_nms_" + VAL_TAG
        )
        if not os.path.exists(os.path.dirname(img_path)):
            os.makedirs(os.path.dirname(img_path))
        cv2.imwrite(img_path, combo)
        Log.info(
            "GNN soft-NMS viz %s: segs conf>=%.2f before=%d after=%d -> file://%s"
            % (full_basename, thresh, n_b, n_a, os.path.abspath(img_path)),
            level=2,
        )
        Log.img(
            img_path,
            combo[..., ::-1],
            epoch,
            tag=VAL_TAG + "/gnn_soft_nms",
            imageidx=ImageIdx.PRED,
            level=1,
        )

    def _plot_gnn_edges(self, full_basename, image, gnn_single, epoch):
        """TensorBoard val overlay: YOLinO segments + predicted links, grouped by connected component.

        Builds an **undirected** graph from edges whose ``sigmoid(edge_logit)`` is at
        least ``gnn_overlay_edge_thresh``, runs connected-components (union–find),
        and draws each component in its own saturated color: segment endpoints
        ``end_a–end_b`` (thick) and mid–mid connectors (thin). Valid nodes with no
        selected incident edge are drawn in neutral gray (optional).
        """
        import colorsys
        import cv2

        node_mid = gnn_single["node_mid_px"]
        node_valid = gnn_single["node_valid"]
        line_n = gnn_single.get("line_neighbors")
        if isinstance(line_n, torch.Tensor):
            neighbors = line_n
            neigh_valid = gnn_single["line_neigh_valid"]
            k_line = int(line_n.shape[-1])
            el = gnn_single["edge_logits"]
            edge_logits = el[..., :k_line] if isinstance(el, torch.Tensor) else el[:, :k_line]
        else:
            neighbors = gnn_single["neighbors"]
            neigh_valid = gnn_single["neigh_valid"]
            edge_logits = gnn_single["edge_logits"]
        node_ea = gnn_single.get("node_end_a_px", None)
        node_eb = gnn_single.get("node_end_b_px", None)
        node_conf = gnn_single.get("node_conf", None)

        if isinstance(node_mid, torch.Tensor):
            node_mid = node_mid.detach().cpu().numpy()
        if isinstance(node_valid, torch.Tensor):
            node_valid = node_valid.detach().cpu().numpy().astype(bool)
        if isinstance(neighbors, torch.Tensor):
            neighbors = neighbors.detach().cpu().numpy().astype(np.int64)
        if isinstance(neigh_valid, torch.Tensor):
            neigh_valid = neigh_valid.detach().cpu().numpy().astype(bool)
        if isinstance(edge_logits, torch.Tensor):
            edge_logits = edge_logits.detach().cpu().float().numpy()
        if node_ea is not None and isinstance(node_ea, torch.Tensor):
            node_ea = node_ea.detach().cpu().numpy()
        if node_eb is not None and isinstance(node_eb, torch.Tensor):
            node_eb = node_eb.detach().cpu().numpy()
        if node_conf is not None and isinstance(node_conf, torch.Tensor):
            node_conf = node_conf.detach().cpu().numpy().astype(np.float32)
        elif node_conf is not None:
            node_conf = np.asarray(node_conf, dtype=np.float32)

        raw_viz = getattr(self.args, "gnn_tb_viz_conf_thresh", None)
        viz_conf = float(raw_viz) if raw_viz is not None else float(self.args.confidence)
        if node_conf is not None:
            node_valid = node_valid & (node_conf >= viz_conf)

        if not np.any(node_valid):
            return

        n_nodes, k = neighbors.shape
        thresh = float(getattr(self.args, "gnn_overlay_edge_thresh", 0.5) or 0.0)
        probs = 1.0 / (1.0 + np.exp(-edge_logits))
        show_lonely = bool(getattr(self.args, "gnn_tb_show_lonely_segments", True))
        t_seg = max(1, int(getattr(self.args, "gnn_tb_segment_thickness", 3)))
        t_conn = max(1, int(getattr(self.args, "gnn_tb_connector_thickness", 2)))

        # --- Undirected edges above threshold (dedupe i<j) ---
        edge_pairs = set()
        for ni in range(n_nodes):
            if not bool(node_valid[ni]):
                continue
            for kj in range(k):
                if not bool(neigh_valid[ni, kj]):
                    continue
                if float(probs[ni, kj]) < thresh:
                    continue
                nj = int(neighbors[ni, kj])
                if nj < 0 or nj >= n_nodes or not bool(node_valid[nj]):
                    continue
                a, b = (ni, nj) if ni < nj else (nj, ni)
                edge_pairs.add((a, b))

        in_graph = np.zeros((n_nodes,), dtype=bool)
        for a, b in edge_pairs:
            in_graph[a] = True
            in_graph[b] = True

        parent = np.arange(n_nodes, dtype=np.int32)

        def _uf_find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = int(parent[x])
            return int(x)

        def _uf_union(a: int, b: int) -> None:
            ra, rb = _uf_find(a), _uf_find(b)
            if ra != rb:
                parent[rb] = ra

        for a, b in edge_pairs:
            _uf_union(a, b)

        roots = sorted({_uf_find(i) for i in range(n_nodes) if in_graph[i]})
        root_to_color = {}
        golden = 0.618033988749895
        for j, r in enumerate(roots):
            h = (j * golden) % 1.0
            r_rgb, g_rgb, b_rgb = colorsys.hsv_to_rgb(h, 0.88, 0.96)
            # OpenCV scalar is (B, G, R).
            root_to_color[r] = (int(255 * b_rgb), int(255 * g_rgb), int(255 * r_rgb))

        lonely_bgr = (160, 160, 160)
        faint_bgr = (210, 210, 210)

        viz = self._denormalize_for_viz(image)
        if isinstance(viz, torch.Tensor):
            chw = viz.detach().cpu().float().numpy()
        else:
            chw = np.asarray(viz, dtype=np.float32)
        if chw.ndim != 3 or chw.shape[0] > 3:
            return
        hwc_rgb = np.transpose(np.clip(chw, 0.0, 1.0), (1, 2, 0))
        hwc_rgb = (hwc_rgb * 255.0).round().astype(np.uint8)
        canvas_bgr = cv2.cvtColor(hwc_rgb, cv2.COLOR_RGB2BGR)

        def _line_bgr(p0, p1, col, thickness, line_type=cv2.LINE_AA):
            x0, y0 = float(p0[0]), float(p0[1])
            x1, y1 = float(p1[0]), float(p1[1])
            if np.any(np.isnan((x0, y0, x1, y1))):
                return
            cv2.line(
                canvas_bgr,
                (int(round(x0)), int(round(y0))),
                (int(round(x1)), int(round(y1))),
                col,
                thickness,
                lineType=line_type,
            )

        # Lonely valid nodes (no selected edge).
        if show_lonely:
            for i in range(n_nodes):
                if not bool(node_valid[i]) or bool(in_graph[i]):
                    continue
                if node_ea is not None and node_eb is not None:
                    _line_bgr(node_ea[i], node_eb[i], lonely_bgr, max(1, t_seg - 1))
                else:
                    _line_bgr(node_mid[i], node_mid[i], lonely_bgr, 3)

        # 3) Connectors + segments per merged component.
        for a, b in edge_pairs:
            ra = _uf_find(a)
            col = root_to_color.get(ra, (0, 255, 255))
            _line_bgr(node_mid[a], node_mid[b], col, t_conn)

        for i in range(n_nodes):
            if not bool(node_valid[i]) or not bool(in_graph[i]):
                continue
            col = root_to_color.get(_uf_find(i), (0, 255, 255))
            if node_ea is not None and node_eb is not None:
                _line_bgr(node_ea[i], node_eb[i], col, t_seg)
            else:
                _line_bgr(node_mid[i], node_mid[i], col, max(2, t_seg))

        n_comp = len(roots)
        n_lonely = int(np.sum(node_valid & (~in_graph)))
        legend = "gnn comp=%d lonely=%d edge>=%.2f conf>=%.2f" % (n_comp, n_lonely, thresh, viz_conf)
        cv2.rectangle(canvas_bgr, (4, 4), (4 + 420, 4 + 26), (255, 255, 255), thickness=-1)
        cv2.putText(
            canvas_bgr, legend, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA,
        )

        img_path = self.args.paths.generate_debug_image_file_path(
            full_basename, ImageIdx.PRED, suffix="gnn_e2e_components_" + VAL_TAG
        )
        if not os.path.exists(os.path.dirname(img_path)):
            os.makedirs(os.path.dirname(img_path))
        cv2.imwrite(img_path, canvas_bgr)
        Log.info(
            "Export GNN component overlay (components=%d) to file://%s" % (n_comp, os.path.abspath(img_path)),
            level=2,
        )
        Log.img(
            img_path,
            canvas_bgr[..., ::-1],
            epoch,
            tag=VAL_TAG + "/gnn_e2e_components",
            imageidx=ImageIdx.PRED,
            level=1,
        )

    def _plot_center_overlay(self, full_basename, image, center_single, epoch):
        """TensorBoard val overlay for center-DETR head.

        Renders three layers on a denormalised RGB canvas:
          1. Optional translucent center heatmap (sigmoid logits, upsampled). Colormap
             and alpha come from ``--center_tb_heatmap_colormap`` / ``--center_tb_heatmap_alpha``
             (default HOT + low alpha to avoid Jet's blue cast).
          2. The K predicted peak markers (◆) coloured per peak with score legend.
          3. The K predicted N-point polylines drawn in the same colour, gated by an
             effective score threshold ``min(center_tb_peak_thresh, center_peak_thresh)``
             (unless ``center_tb_peak_thresh <= 0`` to disable the TB score gate).
        """
        import colorsys
        import cv2

        polylines_px = center_single["polylines_px"]      # [K, N, 2]
        peaks_xy = center_single["center_peaks_xy"]       # [K, 2]
        peak_score = center_single["peak_score"]          # [K]
        peak_valid = center_single.get("peak_valid", None)
        center_logits = center_single.get("center_logits", None)

        if isinstance(polylines_px, torch.Tensor):
            polylines_px = polylines_px.detach().cpu().numpy()
        if isinstance(peaks_xy, torch.Tensor):
            peaks_xy = peaks_xy.detach().cpu().numpy()
        if isinstance(peak_score, torch.Tensor):
            peak_score = peak_score.detach().cpu().float().numpy()
        if peak_valid is not None and isinstance(peak_valid, torch.Tensor):
            peak_valid = peak_valid.detach().cpu().numpy().astype(bool)
        if center_logits is not None and isinstance(center_logits, torch.Tensor):
            center_logits = center_logits.detach().cpu().float().numpy()

        thr_tb = float(getattr(self.args, "center_tb_peak_thresh", 0.3) or 0.0)
        thr_pk = float(getattr(self.args, "center_peak_thresh", 0.05) or 0.0)
        # TB overlay: never stricter than peak extraction (unless user disables TB gate with <=0).
        if thr_tb <= 0.0:
            thr = 0.0
        elif thr_pk <= 0.0:
            thr = thr_tb
        else:
            thr = min(thr_tb, thr_pk)
        thickness = max(1, int(getattr(self.args, "center_tb_thickness", 3)))
        show_heat = bool(getattr(self.args, "center_tb_show_heatmap", True))
        heat_alpha = float(getattr(self.args, "center_tb_heatmap_alpha", 0.12) or 0.0)
        heat_alpha = max(0.0, min(1.0, heat_alpha))
        cmap_name = str(getattr(self.args, "center_tb_heatmap_colormap", "hot") or "hot").upper()
        cmap_id = getattr(cv2, "COLORMAP_" + cmap_name, cv2.COLORMAP_HOT)

        viz = self._denormalize_for_viz(image)
        if isinstance(viz, torch.Tensor):
            chw = viz.detach().cpu().float().numpy()
        else:
            chw = np.asarray(viz, dtype=np.float32)
        if chw.ndim != 3 or chw.shape[0] > 3:
            return
        hwc_rgb = np.transpose(np.clip(chw, 0.0, 1.0), (1, 2, 0))
        hwc_rgb = (hwc_rgb * 255.0).round().astype(np.uint8)
        canvas_bgr = cv2.cvtColor(hwc_rgb, cv2.COLOR_RGB2BGR)
        H_img, W_img = canvas_bgr.shape[:2]

        if show_heat and heat_alpha > 0.0 and center_logits is not None and center_logits.ndim == 3 and center_logits.shape[0] == 1:
            heat = 1.0 / (1.0 + np.exp(-center_logits[0]))
            heat = np.clip(heat, 0.0, 1.0)
            heat = (heat * 255.0).astype(np.uint8)
            heat_up = cv2.resize(heat, (W_img, H_img), interpolation=cv2.INTER_LINEAR)
            heat_col = cv2.applyColorMap(heat_up, cmap_id)
            canvas_bgr = cv2.addWeighted(canvas_bgr, 1.0 - heat_alpha, heat_col, heat_alpha, 0.0)

        K = int(polylines_px.shape[0])
        golden = 0.618033988749895
        kept = 0
        for q in range(K):
            score_q = float(peak_score[q]) if q < len(peak_score) else 0.0
            if peak_valid is not None and q < len(peak_valid) and not bool(peak_valid[q]):
                continue
            if score_q < thr:
                continue
            h = (q * golden) % 1.0
            r_rgb, g_rgb, b_rgb = colorsys.hsv_to_rgb(h, 0.88, 0.96)
            col = (int(255 * b_rgb), int(255 * g_rgb), int(255 * r_rgb))
            pts = polylines_px[q]
            if pts.shape[0] < 2:
                continue
            pts_int = np.round(pts).astype(np.int32)
            cv2.polylines(canvas_bgr, [pts_int], isClosed=False, color=col, thickness=thickness, lineType=cv2.LINE_AA)
            for j in range(pts_int.shape[0]):
                cv2.circle(canvas_bgr, (int(pts_int[j, 0]), int(pts_int[j, 1])), 3, col, -1, lineType=cv2.LINE_AA)
            if q < peaks_xy.shape[0]:
                px, py = float(peaks_xy[q, 0]), float(peaks_xy[q, 1])
                cv2.drawMarker(
                    canvas_bgr, (int(round(px)), int(round(py))), col,
                    markerType=cv2.MARKER_DIAMOND, markerSize=14, thickness=2,
                )
            kept += 1

        legend = "center K=%d/%d thr_eff=%.2f" % (kept, K, thr)
        cv2.rectangle(canvas_bgr, (4, 4), (4 + 420, 4 + 26), (255, 255, 255), thickness=-1)
        cv2.putText(
            canvas_bgr, legend, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA,
        )

        img_path = self.args.paths.generate_debug_image_file_path(
            full_basename, ImageIdx.PRED, suffix="center_e2e_" + VAL_TAG
        )
        if not os.path.exists(os.path.dirname(img_path)):
            os.makedirs(os.path.dirname(img_path))
        cv2.imwrite(img_path, canvas_bgr)
        Log.info(
            "Export Center overlay (K_kept=%d) to file://%s" % (kept, os.path.abspath(img_path)),
            level=2,
        )
        Log.img(
            img_path,
            canvas_bgr[..., ::-1],
            epoch,
            tag=VAL_TAG + "/center_e2e",
            imageidx=ImageIdx.PRED,
            level=1,
        )

    def on_val_epoch_finished(self, epoch):
        self.losses[VAL_TAG].log(epoch=epoch, tag=VAL_TAG)

        # Publish per-component DETR / GNN / Center losses for the val pass.
        self._flush_detr_components(VAL_TAG, epoch)
        self._flush_gnn_components(VAL_TAG, epoch)
        self._flush_gnn_adj_stats(VAL_TAG, epoch)
        self._flush_soft_nms_stats(VAL_TAG, epoch)
        self._flush_soft_geom_stats(VAL_TAG, epoch)
        self._flush_center_components(VAL_TAG, epoch)

        new_best_mean_loss = self.losses[VAL_TAG].mean(epoch)
        gradient_loss = self.losses[VAL_TAG].backprop(epoch)

        if self.args.best_mean_loss:
            if new_best_mean_loss < self.best_mean_loss - 0.0001:
                self.best_gradient_loss = gradient_loss
                self.best_mean_loss = new_best_mean_loss
                self.best_epoch = epoch
                Log.debug('Best mean loss: %f' % self.best_mean_loss)
                self.save_new_best(epoch)
        else:
            if gradient_loss < self.best_gradient_loss - 0.0001:
                self.best_gradient_loss = gradient_loss
                self.best_mean_loss = new_best_mean_loss
                self.best_epoch = epoch
                Log.debug('Best gradient loss: %f' % self.best_gradient_loss)
                self.save_new_best(epoch)

        self.evaluator.publish_scores(epoch=epoch, tag=VAL_TAG)
        Log.push(epoch)

    def save_new_best(self, epoch):
        if self.args.keep:
            save_best_checkpoint(self.args, self.forward.model, self.optimizer, self.scheduler, epoch, self.args.id)
        Log.scalars(tag=VAL_TAG, dict={"loss/best/mean": self.best_mean_loss,
                                       "loss/best/gradient": self.best_gradient_loss,
                                       "epoch/best": self.best_epoch}, epoch=epoch)

    def is_converged(self, epoch):
        if epoch < self.args.earliest_stop:
            return False

        if self.args.patience <= 0:
            return False

        converged = (epoch - self.best_epoch) / self.args.eval_iteration >= self.args.patience
        if converged:
            Log.warning("No improvement for %d epochs. We stop with patience=%d." % (
                epoch - self.best_epoch, self.args.patience))
            Log.tag("early_stop")
        return converged

    def on_training_finished(self, epoch, do_nms):
        skip_best = bool(getattr(self.args, "skip_best_model_eval", False))
        best_model_exists = bool(getattr(self.args, "keep", False)) and os.path.exists(str(self.args.paths.best_model))
        if skip_best:
            Log.info("Skipping best-model end-of-training eval (--skip_best_model_eval): "
                     "that pass uses the full train loader twice per batch, not val_loader.")
        elif best_model_exists:
            Log.print('**** Best Model Eval %s ****' % (self.args.id))
            best_evaluator = Evaluator(args=self.evaluator.args, anchors=self.evaluator.anchors,
                                       coords=self.evaluator.coords, load_best_model=True)
            # best_evaluator.plot = True
            for i, data in tqdm(enumerate(self.loader), total=len(self.loader)):
                images, grid_tensor, fileinfo, dupl, params = data

                num_duplicates = int(sum(dupl["total_duplicates_in_image"]).item())
                preds, _ = best_evaluator(images=images, grid_tensor=grid_tensor, idx=i, filenames=fileinfo,
                                          tag="best/val", epoch=None, num_duplicates=num_duplicates)
                self.plot_val_summary(epoch, filenames=fileinfo, grid_tensor=grid_tensor, images=images, preds=preds)
                preds, _ = best_evaluator(images=images, grid_tensor=grid_tensor, idx=i, filenames=fileinfo,
                                          tag="best/val", do_in_uv=False, epoch=None, num_duplicates=num_duplicates)
            best_evaluator.publish_scores(epoch=None, tag="best/" + VAL_TAG)
            Log.push(None)
        else:
            Log.warning("Skip best-model evaluation: checkpoint missing (keep=%s, best_model=%s)"
                        % (str(getattr(self.args, "keep", False)), str(self.args.paths.best_model)))

        if do_nms:
            Log.print('**** NMS Training Data Eval %s ****' % (self.args.id))
            for i, data in tqdm(enumerate(self.loader), total=len(self.loader)):
                images, grid_tensor, fileinfo, dupl, params = data
                num_duplicates = int(sum(dupl["total_duplicates_in_image"]).item())
                self.evaluator(images=images, grid_tensor=grid_tensor, idx=i, filenames=fileinfo, tag="nms_train",
                               apply_nms=True, epoch=epoch, num_duplicates=num_duplicates)
            self.evaluator.publish_scores(epoch=epoch, tag=TRAIN_TAG)

            Log.print('**** NMS Validation Data Eval %s ****' % (self.args.id))
            for i, data in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
                images, grid_tensor, fileinfo, dupl, params = data
                num_duplicates = int(sum(dupl["total_duplicates_in_image"]).item())
                self.evaluator(images=images, grid_tensor=grid_tensor, idx=i, filenames=fileinfo, tag="nms_val",
                               apply_nms=True, epoch=epoch, num_duplicates=num_duplicates)

            self.evaluator.publish_scores(epoch=epoch, tag=VAL_TAG)

        Log.debug('Finished Training')
        Log.finish()

    def loss(self, grid_tensor, geom_preds, embed_preds, filenames, epoch, tag="dummy_trainer", geom_logits=None):
        """
        Args:
            grid_tensor (torch.tensor): [batch, cells, preds, vars]
            geom_preds (torch.tensor): [batch, cells, preds, num_train_vars]
            embed_preds (torch.tensor): [batch, cells, preds, embed_dim]
            geom_logits (torch.tensor, optional): pre-sigmoid geometry head; same layout as geom_preds (for focal CE).
        """
        losses, sum_loss, mean_losses = self.loss_fct(
            geom_preds, embed_preds, grid_tensor, filenames=filenames, epoch=epoch,
            tag=tag, geom_logits=geom_logits)
        if torch.isnan(sum_loss) or torch.isinf(sum_loss):
            raise ValueError("Loss ran into NaN or infinity")

        return losses, sum_loss, mean_losses

    def _build_gnn_matched_supervision(self, gnn_out, device):
        """Build node-level GT from the cell-matched grid tensor (exp61 option)."""
        if not bool(getattr(self.args, "gnn_use_matched_gt_from_geom", False)):
            return None
        if not isinstance(gnn_out, dict):
            return None
        node_src = gnn_out.get("node_src_flat_idx", None)
        matched = getattr(self.loss_fct, "last_matched_grid_tensor", None)
        if not isinstance(node_src, torch.Tensor) or not isinstance(matched, torch.Tensor):
            return None
        if matched.ndim != 3:
            return None

        node_src = node_src.to(device=device, dtype=torch.long)
        matched = matched.to(device=device)
        b, n = node_src.shape
        v = int(matched.shape[-1])
        gather_idx = node_src.unsqueeze(-1).expand(-1, -1, v)
        node_gt = torch.gather(matched, 1, gather_idx)

        geom_idx = self.loss_fct.coords.get_position_of(Variables.GEOMETRY)
        gt_geom = node_gt[:, :, geom_idx]
        node_gt_valid = ~torch.any(torch.isnan(gt_geom), dim=-1)

        node_gt_instance = torch.full((b, n), -1, dtype=torch.long, device=device)
        try:
            inst_idx = self.loss_fct.coords.get_position_of(Variables.INSTANCE)
            gt_inst = node_gt[:, :, inst_idx]
            if gt_inst.ndim == 3:
                gt_inst = gt_inst[..., 0]
            gt_inst = torch.where(torch.isnan(gt_inst), torch.full_like(gt_inst, -1.0), gt_inst)
            node_gt_instance = torch.round(gt_inst).to(dtype=torch.long)
            node_gt_instance = torch.where(
                node_gt_valid, node_gt_instance, torch.full_like(node_gt_instance, -1)
            )
        except Exception:
            pass

        return {
            "node_gt_valid": node_gt_valid,
            "node_gt_instance": node_gt_instance,
        }

    def backward(self, loss, epoch=None):
        if self.debug_anomaly:
            with torch.autograd.detect_anomaly():
                self._backward_impl(loss, epoch=epoch)
            return
        self._backward_impl(loss, epoch=epoch)

    def _backward_impl(self, loss, epoch=None):
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            if getattr(self.args, "grad_clip_norm", 0.0) and self.args.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip_norm)
            if epoch is not None:
                self._log_gradient_norms(epoch)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if getattr(self.args, "grad_clip_norm", 0.0) and self.args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip_norm)
            if epoch is not None:
                self._log_gradient_norms(epoch)
            self.optimizer.step()
        if self.scheduler is not None and getattr(self.args, "scheduler_step_per_batch", True):
            self.scheduler.step()

    def _log_loss_diagnostics(self, losses, mean_losses, sum_loss, filenames, epoch, image_idx_in_batch):
        def _as_float(v):
            if isinstance(v, torch.Tensor):
                if v.numel() == 0:
                    return float("nan")
                return float(v.detach().reshape(-1)[0].item())
            try:
                return float(v)
            except (TypeError, ValueError):
                return float("nan")

        sum_v = _as_float(sum_loss)
        if math.isfinite(sum_v):
            return

        comp = [_as_float(v) for v in losses]
        comp_mean = [_as_float(v) for v in mean_losses]
        msg = (
            "Non-finite sum_loss detected: sum=%s epoch=%s iter=%s files=%s "
            "components=%s components_finite=%s means=%s means_finite=%s"
            % (
                sum_v,
                epoch,
                image_idx_in_batch,
                filenames,
                comp,
                [math.isfinite(v) for v in comp],
                comp_mean,
                [math.isfinite(v) for v in comp_mean],
            )
        )
        Log.error(msg)

    def _log_gradient_norms(self, epoch):
        # Each .item() forces a GPU sync. The old loop synced once per parameter with a grad (~hundreds per batch),
        # on every DDP rank — destroying step time. Only rank 0 logs; only match params we actually report.
        if not getattr(self.args, "is_main_process", True):
            return
        patterns = (
            ("embed_head", "embed_head"),
            ("attention", "attention"),
            ("cbam", "cbam"),
            ("yolo", "yolo_head"),
        )
        buckets = {key: [] for _, key in patterns}
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            for substr, key in patterns:
                if substr in name:
                    buckets[key].append(param.grad.detach().norm(2).item())
                    break
        for group, norms in buckets.items():
            if not norms:
                continue
            avg_norm = sum(norms) / len(norms)
            max_norm = max(norms)
            Log.scalars(tag=TRAIN_TAG, epoch=epoch,
                        dict={f"grad_norm/{group}/avg": avg_norm,
                              f"grad_norm/{group}/max": max_norm})

    def is_time_for_val(self, epoch):
        return epoch % self.args.eval_iteration == 0

    def _denormalize_for_viz(self, image):
        """Return an image suitable for visualization without changing training tensors."""
        if not isinstance(image, torch.Tensor):
            return image
        if image.ndim != 3 or image.shape[0] > 3:
            return image
        augmentor = getattr(self.dataset, "augmentor", None)
        if augmentor is None:
            return image
        mean = torch.tensor(augmentor.norm_mean, dtype=image.dtype, device=image.device).view(3, 1, 1)
        std = torch.tensor(augmentor.norm_std, dtype=image.dtype, device=image.device).view(3, 1, 1)
        return torch.clamp(image * std + mean, 0.0, 1.0)

    def plot_debug_class_image(self, full_basename, image, grid, epoch, tag="unknown", imageidx=ImageIdx.DEFAULT,
                               ignore_classes=[]):
        img_path = self.args.paths.generate_debug_image_file_path(full_basename + "_class.png", imageidx, suffix=epoch)
        _, ok = plot_cell_class(grid=grid, image=image, name=img_path, epoch=epoch, tag=tag + "_class",
                                imageidx=imageidx,
                                max_class=self.dataset.num_classes(), ignore_classes=ignore_classes,
                                threshold=self.args.confidence)
        return ok

    def plot_debug_images(self, full_basename, image, uv_lines, epoch=-1, tag="train",
                          show_grid=True, imageidx: ImageIdx = ImageIdx.LABEL,
                          coordinates=CoordinateSystem.UV_CONTINUOUS, gt=None, training_vars_only=False, suffix="",
                          cell_size=None):
        viz_image = self._denormalize_for_viz(image)
        img_path = self.args.paths.generate_debug_image_file_path(full_basename, imageidx, suffix=suffix + "_" + tag)
        ok = plot_style_grid(uv_lines, img_path, viz_image, coords=self.dataset.coords,
                             cell_size=self.args.cell_size if cell_size is None else cell_size,
                             show_grid=show_grid, coordinates=coordinates, epoch=epoch, tag=tag,
                             imageidx=imageidx, threshold=self.args.confidence, gt=gt,
                             training_vars_only=training_vars_only,
                             level=1)
        return ok

    def plot_debug_class_grid(self, preds, images, epoch, filenames, tag, imageidx):
        plot_images = []
        for idx in range(len(filenames)):
            grid, _ = GridFactory.get(data=torch.unsqueeze(preds[idx], dim=0), variables=[],
                                      coordinate=CoordinateSystem.CELL_SPLIT, args=self.args,
                                      input_coords=self.dataset.coords, only_train_vars=True,
                                      anchors=self.dataset.anchors)

            img, ok = plot_cell_class(grid=grid, image=images[idx], name=None, epoch=epoch, tag=tag + "_class",
                                      imageidx=imageidx, max_class=self.dataset.num_classes(), ignore_classes=[0],
                                      threshold=self.args.confidence)
            plot_images.append(convert_to_torch_image(img))

        name = self.args.paths.generate_debug_image_file_path("class", imageidx)
        Log.grid(name=name, images=plot_images, epoch=epoch, imageidx=imageidx, tag=tag + "_class")

    def is_time_for_image(self, epoch):
        return epoch % 5 == 0

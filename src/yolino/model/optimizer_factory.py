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
import torch

from yolino.utils.enums import Optimizer, Scheduler
from yolino.utils.logger import Log


# Parameter-name prefixes used by the new YolinoNet (ConvNeXt-Tiny + FPN).
# Kept here so it's easy to keep in sync if the model layout is renamed.
_BACKBONE_BODY_PREFIXES = ("backbone.body.", "module.backbone.body.")  # DDP adds "module."
_FPN_PREFIXES = (                                 # FPN lateral + smooth convs + norms (from-scratch)
    "backbone.lateral_c2.",
    "backbone.lateral_c3.",
    "backbone.lateral_c4.",
    "backbone.smooth_p2.",
    "backbone.smooth_p3.",
    "backbone.smooth_p4.",
    "backbone.smooth_bn_p2.",
    "backbone.smooth_bn_p3.",
    "backbone.smooth_bn_p4.",
    "backbone.smooth_norm_p2.",
    "backbone.smooth_norm_p3.",
    "backbone.smooth_norm_p4.",
    "module.backbone.lateral_c2.",
    "module.backbone.lateral_c3.",
    "module.backbone.lateral_c4.",
    "module.backbone.smooth_p2.",
    "module.backbone.smooth_p3.",
    "module.backbone.smooth_p4.",
    "module.backbone.smooth_bn_p2.",
    "module.backbone.smooth_bn_p3.",
    "module.backbone.smooth_bn_p4.",
    "module.backbone.smooth_norm_p2.",
    "module.backbone.smooth_norm_p3.",
    "module.backbone.smooth_norm_p4.",
    "backbone.bottom_up.",
    "module.backbone.bottom_up.",
)
_GEOM_HEAD_PREFIXES = ("yolo.", "module.yolo.", "std_head.geom_dcns.", "module.std_head.geom_dcns.")
_EMBED_HEAD_PREFIXES = (
    "embed_head.", "module.embed_head.",
    "std_head.embed_convs.", "module.std_head.embed_convs.",
)
_E2E_HEAD_PREFIXES = ("e2e_head.", "module.e2e_head.")
_STD_FEAT_DCN_PREFIXES = ("std_feat_dcn.", "module.std_feat_dcn.")
_REFINE_PREFIXES = ("attention.", "cbam.", "module.attention.", "module.cbam.")  # GlobalSelfAttention / CBAM


def _classify(name: str, refine_shared: bool) -> str:
    """Map a parameter name to one of: 'backbone', 'fpn', 'geom', 'embed', 'e2e'."""
    if any(name.startswith(p) for p in _BACKBONE_BODY_PREFIXES):
        return "backbone"
    for p in _FPN_PREFIXES:
        if name.startswith(p):
            return "fpn"
    if any(name.startswith(p) for p in _GEOM_HEAD_PREFIXES):
        return "geom"
    if any(name.startswith(p) for p in _E2E_HEAD_PREFIXES):
        return "e2e"
    if any(name.startswith(p) for p in _STD_FEAT_DCN_PREFIXES):
        return "e2e"
    if any(name.startswith(p) for p in _EMBED_HEAD_PREFIXES):
        return "embed"
    if any(name.startswith(p) for p in _REFINE_PREFIXES):
        # Shared refine (sa_shared / cbam_shared) → bias to FPN LR (mid-level feature stage).
        # Embed-only refine (sa_embed_only) → tied to embed LR.
        return "fpn" if refine_shared else "embed"
    # Anything not matched (e.g. legacy modules) → treat as backbone.
    return "backbone"


def _collect_lr_groups(args, net, loss_weights):
    lr_global = args.learning_rate
    lr_backbone = getattr(args, "lr_backbone", None)
    lr_fpn = getattr(args, "lr_fpn", None)
    lr_geom = getattr(args, "lr_geom", None)
    lr_embed = getattr(args, "lr_embed", None)
    lr_e2e = getattr(args, "lr_e2e", None)

    # Backward-compatible path: if NONE of the per-group LRs is set, use a single flat LR.
    if all(x is None for x in (lr_backbone, lr_fpn, lr_geom, lr_embed, lr_e2e)):
        flat = [p for p in net.parameters() if p.requires_grad] \
               + [l for l in loss_weights if l.requires_grad]
        return flat, {"mode": "flat", "lr": lr_global}

    if lr_backbone is None:
        lr_backbone = lr_global
    if lr_fpn is None:
        lr_fpn = lr_global
    if lr_geom is None:
        lr_geom = lr_global
    if lr_embed is None:
        lr_embed = lr_global
    if lr_e2e is None:
        lr_e2e = lr_embed

    refine_shared = getattr(args, "feature_refine", "sa_embed_only") in ("sa_shared", "cbam_shared")

    buckets = {"backbone": [], "fpn": [], "geom": [], "embed": [], "e2e": []}
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        bucket = _classify(name, refine_shared=refine_shared)
        buckets[bucket].append(p)

    groups = []
    if buckets["backbone"]:
        groups.append({"params": buckets["backbone"], "lr": lr_backbone, "name": "backbone"})
    if buckets["fpn"]:
        groups.append({"params": buckets["fpn"], "lr": lr_fpn, "name": "fpn"})
    if buckets["geom"]:
        groups.append({"params": buckets["geom"], "lr": lr_geom, "name": "geom"})
    if buckets["embed"]:
        groups.append({"params": buckets["embed"], "lr": lr_embed, "name": "embed"})
    if buckets["e2e"]:
        groups.append({"params": buckets["e2e"], "lr": lr_e2e, "name": "e2e"})

    learnable_loss_weights = [l for l in loss_weights if l.requires_grad]
    if learnable_loss_weights:
        groups.append({"params": learnable_loss_weights, "lr": lr_global, "name": "loss_weights"})

    summary = {
        "mode": "grouped",
        "lr_backbone": lr_backbone, "lr_fpn": lr_fpn,
        "lr_geom": lr_geom, "lr_embed": lr_embed, "lr_e2e": lr_e2e,
        "n_backbone": len(buckets["backbone"]), "n_fpn": len(buckets["fpn"]),
        "n_geom": len(buckets["geom"]), "n_embed": len(buckets["embed"]),
        "n_e2e": len(buckets["e2e"]),
        "n_loss_weights": len(learnable_loss_weights),
    }
    Log.info("Param-group LR: backbone=%.2e fpn=%.2e geom=%.2e embed=%.2e e2e=%.2e "
             "(loss_weights LR=%.2e)" % (lr_backbone, lr_fpn, lr_geom, lr_embed, lr_e2e, lr_global))
    Log.debug("Param-group sizes: backbone=%d fpn=%d geom=%d embed=%d e2e=%d loss_weights=%d"
              % (summary["n_backbone"], summary["n_fpn"],
                 summary["n_geom"], summary["n_embed"], summary["n_e2e"], summary["n_loss_weights"]))
    return groups, summary


def maybe_freeze_backbone(args, net, epoch: int):
    """
    Freeze/unfreeze ConvNeXt body based on `args.backbone_freeze_epochs`.

    Call this at the start of every epoch (e.g. from the train loop). Idempotent.
    """
    n_freeze = int(getattr(args, "backbone_freeze_epochs", 0) or 0)
    if n_freeze <= 0:
        return False  # no freezing requested
    should_freeze = epoch < n_freeze
    changed = False
    for name, p in net.named_parameters():
        if any(name.startswith(p) for p in _BACKBONE_BODY_PREFIXES):
            new_state = (not should_freeze)
            if p.requires_grad != new_state:
                p.requires_grad = new_state
                changed = True
    if changed:
        Log.info("Backbone body %s at epoch=%d (backbone_freeze_epochs=%d)"
                 % ("FROZEN" if should_freeze else "UNFROZEN", epoch, n_freeze))
    return should_freeze


def get_optimizer(args, net, loss_weights):
    Log.debug("Optimizer %s" % str(args.optimizer))

    # Apply initial freeze BEFORE collecting param groups so frozen weights are excluded.
    maybe_freeze_backbone(args, net, epoch=0)

    params, _summary = _collect_lr_groups(args, net, loss_weights)

    if args.optimizer == Optimizer.ADAM:
        if args.learning_rate != 0.001 or args.decay_rate != 0:
            Log.info("The optimizer by default uses lr=0.001 and weight_decay=0, you set lr=%f and weight_decay=%f."
                     % (args.learning_rate, args.decay_rate))
        optimizer = torch.optim.Adam(
            params,
            lr=args.learning_rate,
            weight_decay=args.decay_rate,
        )
    elif args.optimizer == Optimizer.SGD:
        optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=args.momentum)
    elif args.optimizer == Optimizer.RMS_PROP:
        optimizer = torch.optim.RMSprop(params, lr=args.learning_rate,
                                        alpha=0.99, eps=1e-08, weight_decay=args.decay_rate,
                                        momentum=args.momentum, centered=False)
    elif args.optimizer == Optimizer.ADA_DELTA:
        optimizer = torch.optim.Adadelta(params, lr=10 * args.learning_rate, rho=0.9, eps=1e-06,
                                         weight_decay=args.decay_rate)
    else:
        raise NotImplementedError("Unknown optimizer %s" % args.optimizer)

    if args.scheduler == Scheduler.NONE:
        scheduler = None
    elif args.scheduler in (Scheduler.COSINE, Scheduler.WARMUP_COSINE):
        # Per-step schedule (works with param groups preserving individual base LR).
        total_steps = int(getattr(args, "total_train_steps", 0))
        if total_steps <= 0:
            # conservative fallback; caller should set this from dataloader length.
            total_steps = max(1, int(getattr(args, "epoch", 1)))
        warmup_epochs = int(getattr(args, "warmup_epochs", 0) or 0)
        iters_per_epoch = int(getattr(args, "iters_per_epoch", 1) or 1)
        warmup_steps = max(0, warmup_epochs * iters_per_epoch)
        min_lr_ratio = float(getattr(args, "min_lr_ratio", 0.05))
        min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))

        def lr_lambda(step: int):
            step = max(0, int(step))
            if args.scheduler == Scheduler.WARMUP_COSINE and warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)

            cosine_total = max(1, total_steps - warmup_steps)
            cosine_step = max(0, min(step - warmup_steps, cosine_total))
            cos_factor = 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * cosine_step / cosine_total))).item()
            return min_lr_ratio + (1.0 - min_lr_ratio) * cos_factor

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        Log.info("Scheduler=%s warmup_epochs=%d total_steps=%d min_lr_ratio=%.3f"
                 % (args.scheduler, warmup_epochs, total_steps, min_lr_ratio))
    else:
        raise NotImplementedError("Unknown scheduler %s" % args.scheduler)

    return optimizer, scheduler

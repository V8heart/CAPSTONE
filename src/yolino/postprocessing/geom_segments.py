# Copyright 2023 Karlsruhe Institute of Technology, Institute for Measurement
# and Control Systems
"""Fast segment extraction from activated geom tensors (bypass GridFactory)."""
from __future__ import annotations

import numpy as np
import torch

from yolino.model.e2e_polyline_modules import mid_dir_geom_to_midpoints_pixels
from yolino.model.variable_structure import VariableStructure
from yolino.utils.enums import LINE, Variables


def geom_act_to_uv_lines(
    geom_act: torch.Tensor,
    coords: VariableStructure,
    grid_shape: tuple[int, int],
    stride: float,
    img_h: int,
    img_w: int,
    confidence_threshold: float,
) -> np.ndarray:
    """Build UV_SPLIT line array from ``geom_act`` without :class:`GridFactory`.

    Matches :meth:`Grid.get_image_lines` + conf filter for MID_DIR (``linerep=md``):
    geometry columns are ``[V0, H0, V1, H1]`` in pixel row/col indices; conf in train CONF slot.

    Args:
        geom_act: ``[B, cells, P, V]`` activated geometry (``ForwardRunner`` output).
        coords: Dataset coordinate layout (training vars).
        grid_shape: ``(rows, cols)`` e.g. ``args.grid_shape``.
        stride: Head stride in pixels (``args.scale``).
        img_h, img_w: Tile size in pixels.
        confidence_threshold: Keep segments with ``conf > threshold`` (same as ``fit_lines``).

    Returns:
        ``[1, N, num_train_vars]`` float32, or ``[1, 0, num_train_vars]`` if empty.
    """
    if coords.line_representation.enum != LINE.MID_DIR:
        raise NotImplementedError(
            "geom_act_to_uv_lines supports MID_DIR (linerep=md) only, got %s"
            % coords.line_representation.enum
        )

    b, ncell, p, _ = geom_act.shape
    rows, cols = int(grid_shape[0]), int(grid_shape[1])
    if rows * cols != ncell:
        raise ValueError("grid_shape %s does not match cells=%d" % (grid_shape, ncell))

    g = geom_act.view(b, rows, cols, p, -1)
    _, end_a, end_b = mid_dir_geom_to_midpoints_pixels(
        g, stride=float(stride), img_h=int(img_h), img_w=int(img_w),
    )

    conf_cols = np.asarray(coords.get_position_within_prediction(Variables.CONF)).ravel()
    if conf_cols.size == 0:
        raise ValueError("CONF not in training variables")
    conf_i = int(conf_cols[0])
    conf = g[..., conf_i].reshape(b, -1)

    thr = float(confidence_threshold)
    mask = conf[0] > thr
    length = int(coords.num_vars_to_train())
    if not bool(mask.any().item()):
        return np.zeros((0, length), dtype=np.float32)

    ea = end_a.reshape(b, -1, 2)[0][mask].detach().cpu().numpy()
    eb = end_b.reshape(b, -1, 2)[0][mask].detach().cpu().numpy()
    cf = conf[0][mask].detach().cpu().numpy()

    # (x, y) = (H, V) pixels -> UV_SPLIT [V, H, V, H]
    geom_uv = np.stack([ea[:, 1], ea[:, 0], eb[:, 1], eb[:, 0]], axis=1).astype(np.float32)

    length = int(coords.num_vars_to_train())
    geom_pos = np.asarray(coords.get_position_within_prediction(Variables.GEOMETRY)).ravel()
    out = np.zeros((geom_uv.shape[0], length), dtype=np.float32)
    out[:, geom_pos] = geom_uv
    out[:, conf_i] = cf.astype(np.float32)
    return out

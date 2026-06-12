# SPDX-License-Identifier: GPL-3.0-or-later
"""Conservative segment merging (geometric, deterministic).

Unlike :mod:`yolino.model.segment_soft_nms` which only **decays the confidence**
of duplicates, this module **physically fuses** segments that visually overlap
or sit immediately next to each other into a single representative segment.

Use case (exp58): apply this on the post-conf-filter raw geom output **before**
GNN node selection. The GNN then receives one node per real wire chunk rather
than 3-5 near-duplicate predictors covering the same pixels.

Algorithm (per image, ``numpy``)
================================
1. From each segment compute ``mid``, unit direction ``dn``, length ``L``.
2. Build the compatibility graph (symmetric):

   - ``sym_lat[i,j] = max(d_lat(i->j), d_lat(j->i))`` — perpendicular distance
     between the two segment lines, measured both ways and symmetrized
     (the two segments must be on *almost the same line*).
   - ``dir_dot[i,j] = |dn_i · dn_j|`` — unsigned parallelism (0 = perpendicular,
     1 = parallel).
   - ``end_gap[i,j]`` — smallest pairwise distance between
     ``{ea_i, eb_i} × {ea_j, eb_j}`` (endpoint touching score).
   - ``along_overlap[i,j]`` — projection overlap of segment ``j`` onto the
     line through segment ``i`` (positive when the segments visually overlap).

   Pair ``(i, j)`` is connected iff:

   ::

       dir_dot >= dir_dot_min
       sym_lat <= lat_px
       (along_overlap > 0)  OR  (end_gap <= end_gap_px)

3. Union-find clusters on this graph.
4. For each cluster of size ``>= 2`` produce a single replacement segment:

   - Principal direction = conf-weighted unit vector
     (signs aligned to the highest-conf seed so the resulting direction is stable).
   - Project all 2N endpoints onto the principal direction → ``(t_min, t_max)``
     define the new extent.
   - Lateral anchor = conf-weighted mean of mid-points projected onto the
     direction's normal.
   - ``conf_new = max(cluster confs)`` (conservative: keep the strongest seed).

5. Optionally iterate up to ``iters`` times: a freshly merged segment may now
   touch another previously isolated segment.

The function is fully deterministic and does not require gradient flow; it is
applied at inference / pre-GNN time.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Union-Find
# --------------------------------------------------------------------------- #
class _DSU:
    __slots__ = ("p",)

    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


# --------------------------------------------------------------------------- #
# Geometric helpers (numpy, vectorised over pairs)
# --------------------------------------------------------------------------- #
def _unit_dir(ea: np.ndarray, eb: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Return (dn, length) for each row in (ea, eb)."""
    vec = eb - ea
    length = np.linalg.norm(vec, axis=-1)
    safe = np.maximum(length, eps)
    dn = vec / safe[..., None]
    return dn, length


def _pairwise_lateral(mid: np.ndarray, dn: np.ndarray) -> np.ndarray:
    """``out[i, j] = |((mid_j - mid_i) x dn_i)|``  (perp distance from j to line through i)."""
    diff = mid[None, :, :] - mid[:, None, :]
    return np.abs(diff[..., 0] * dn[:, None, 1] - diff[..., 1] * dn[:, None, 0])


def _pairwise_endpoint_gap(ea: np.ndarray, eb: np.ndarray) -> np.ndarray:
    """Smallest pairwise endpoint distance between segments i and j (4 candidate pairs)."""
    n = ea.shape[0]
    # Stack endpoints as [N, 2, 2]: row 0 = ea, row 1 = eb, last dim = xy.
    pts = np.stack([ea, eb], axis=1)  # [N, 2, 2]
    # [N, 1, 2, 1, 2] - [1, N, 1, 2, 2] -> [N, N, 2, 2, 2]
    d = pts[:, None, :, None, :] - pts[None, :, None, :, :]
    dist = np.linalg.norm(d, axis=-1)            # [N, N, 2, 2]
    return dist.reshape(n, n, 4).min(axis=-1)    # [N, N]


def _pairwise_along_overlap(
    mid: np.ndarray, dn: np.ndarray, length: np.ndarray
) -> np.ndarray:
    """Signed projection overlap of segment j onto the line through segment i.

    Positive value = segments visually overlap along i's direction; negative =
    the projected gap between them (j projects outside i's [-L_i/2, L_i/2]).
    """
    diff = mid[None, :, :] - mid[:, None, :]                # [N, N, 2]
    t_j = diff[..., 0] * dn[:, None, 0] + diff[..., 1] * dn[:, None, 1]  # [N, N]
    cos_ij = np.abs(dn[:, None, 0] * dn[None, :, 0] + dn[:, None, 1] * dn[None, :, 1])
    half_proj_j = 0.5 * length[None, :] * cos_ij
    half_i = 0.5 * length[:, None]
    j_lo = t_j - half_proj_j
    j_hi = t_j + half_proj_j
    overlap = np.minimum(half_i, j_hi) - np.maximum(-half_i, j_lo)
    return overlap


# --------------------------------------------------------------------------- #
# Cluster merger
# --------------------------------------------------------------------------- #
def _merge_one_cluster(
    cluster_idx: np.ndarray,
    ea: np.ndarray,
    eb: np.ndarray,
    conf: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fuse a cluster of segments into a single (ea_new, eb_new, conf_new)."""
    if cluster_idx.size == 1:
        i = int(cluster_idx[0])
        return ea[i].copy(), eb[i].copy(), float(conf[i])

    dn, _ = _unit_dir(ea[cluster_idx], eb[cluster_idx])
    c = conf[cluster_idx].astype(np.float64)
    seed = int(cluster_idx[int(np.argmax(c))])
    dn_seed, _ = _unit_dir(ea[seed : seed + 1], eb[seed : seed + 1])
    dn_seed = dn_seed[0]
    signs = np.sign(dn @ dn_seed)
    signs[signs == 0.0] = 1.0
    dn_aligned = dn * signs[:, None]
    w = c[:, None]
    dn_avg = (dn_aligned * w).sum(axis=0)
    n = np.linalg.norm(dn_avg)
    dn_avg = dn_avg / max(n, 1e-8)

    pts = np.concatenate([ea[cluster_idx], eb[cluster_idx]], axis=0)  # [2K, 2]
    pts_w = np.repeat(c, 2)
    mid_anchor = (pts * pts_w[:, None]).sum(axis=0) / max(pts_w.sum(), 1e-8)
    t = (pts - mid_anchor) @ dn_avg
    t_min, t_max = float(t.min()), float(t.max())
    ea_new = mid_anchor + t_min * dn_avg
    eb_new = mid_anchor + t_max * dn_avg
    conf_new = float(c.max())
    return ea_new.astype(ea.dtype), eb_new.astype(eb.dtype), conf_new


def _merge_once(
    ea: np.ndarray,
    eb: np.ndarray,
    conf: np.ndarray,
    *,
    lat_px: float,
    dir_dot_min: float,
    end_gap_px: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    n = ea.shape[0]
    if n <= 1:
        return ea, eb, conf, False

    mid = 0.5 * (ea + eb)
    dn, length = _unit_dir(ea, eb)

    lat_ij = _pairwise_lateral(mid, dn)
    lat_ji = lat_ij.T
    sym_lat = np.maximum(lat_ij, lat_ji)
    dir_dot = np.abs(dn @ dn.T)
    end_gap = _pairwise_endpoint_gap(ea, eb)
    along = _pairwise_along_overlap(mid, dn, length)
    along_sym = np.maximum(along, along.T)

    along_branch = (along_sym > 0.0)
    if float(end_gap_px) > 0.0:
        along_branch = along_branch | (end_gap <= float(end_gap_px))
    connect = (
        (dir_dot >= float(dir_dot_min))
        & (sym_lat <= float(lat_px))
        & along_branch
    )
    np.fill_diagonal(connect, False)

    dsu = _DSU(n)
    ii, jj = np.where(connect)
    for a, b in zip(ii.tolist(), jj.tolist()):
        if a < b:
            dsu.union(a, b)

    roots = np.array([dsu.find(i) for i in range(n)])
    uniq, inv = np.unique(roots, return_inverse=True)
    if uniq.size == n:
        return ea, eb, conf, False

    new_ea = np.empty((uniq.size, 2), dtype=ea.dtype)
    new_eb = np.empty((uniq.size, 2), dtype=eb.dtype)
    new_cf = np.empty((uniq.size,), dtype=conf.dtype)
    for k in range(uniq.size):
        members = np.where(inv == k)[0]
        a_pt, b_pt, c_val = _merge_one_cluster(members, ea, eb, conf)
        new_ea[k] = a_pt
        new_eb[k] = b_pt
        new_cf[k] = c_val
    return new_ea, new_eb, new_cf, True


def merge_segments(
    ea: np.ndarray,
    eb: np.ndarray,
    conf: np.ndarray,
    *,
    lat_px: float = 6.0,
    dir_dot_min: float = 0.98,
    end_gap_px: float = 8.0,
    iters: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conservative, iterative cluster-merge of nearby colinear segments.

    Args:
        ea, eb: ``[N, 2]`` segment endpoint arrays (pixel coordinates).
        conf:   ``[N]`` confidence scores.
        lat_px: max perpendicular distance between segment lines to be considered
            visually overlapping; conservative defaults assume MID_DIR head at
            32-px cell stride (lat 6 px ≈ ¼ cell).
        dir_dot_min: minimum ``|dn_i · dn_j|`` (cosine of half-angle) to be
            considered parallel. ``0.98`` ≈ 11° tolerance.
        end_gap_px: maximum pairwise endpoint distance to bridge segments that
            do not visually overlap (i.e. end-to-end stitch within this gap).
        iters: max number of fixed-point iterations.
    """
    if ea.shape[0] != eb.shape[0] or ea.shape[0] != conf.shape[0]:
        raise ValueError("ea/eb/conf row counts must match")
    if ea.shape[0] == 0:
        return ea.copy(), eb.copy(), conf.copy()

    cur_ea = np.asarray(ea, dtype=np.float64)
    cur_eb = np.asarray(eb, dtype=np.float64)
    cur_cf = np.asarray(conf, dtype=np.float64)
    for _ in range(max(1, int(iters))):
        cur_ea, cur_eb, cur_cf, changed = _merge_once(
            cur_ea, cur_eb, cur_cf,
            lat_px=lat_px,
            dir_dot_min=dir_dot_min,
            end_gap_px=end_gap_px,
        )
        if not changed:
            break

    return cur_ea.astype(np.float32), cur_eb.astype(np.float32), cur_cf.astype(np.float32)


def merge_uv_split_array(
    uv_arr: np.ndarray,
    *,
    geom_cols: Sequence[int],
    conf_col: int,
    lat_px: float = 6.0,
    dir_dot_min: float = 0.98,
    end_gap_px: float = 8.0,
    iters: int = 3,
) -> np.ndarray:
    """Convenience wrapper: merge a single image's UV_SPLIT segment table.

    ``uv_arr`` is ``[N, K]`` (no batch dim). The 4-column geometry block is
    expected to be packed as ``(V0, H0, V1, H1)`` (i.e. the YOLinO MID_DIR pixel
    layout used by ``Grid.get_image_lines`` for ``linerep=md``).

    Returns ``[M, K]`` with merged rows. Non-geometry, non-confidence columns
    are taken from the highest-confidence cluster seed.
    """
    if uv_arr.ndim != 2:
        raise ValueError("uv_arr must be [N, K]; got %s" % (uv_arr.shape,))
    cols = list(geom_cols)
    if len(cols) != 4:
        raise ValueError("Expected 4 geom columns (V0, H0, V1, H1); got %d" % len(cols))
    n, k = uv_arr.shape
    if n == 0:
        return uv_arr.copy()

    v0, h0, v1, h1 = cols
    ea_xy = np.stack([uv_arr[:, h0], uv_arr[:, v0]], axis=1).astype(np.float64)
    eb_xy = np.stack([uv_arr[:, h1], uv_arr[:, v1]], axis=1).astype(np.float64)
    cf = uv_arr[:, int(conf_col)].astype(np.float64)

    cur_ea = ea_xy
    cur_eb = eb_xy
    cur_cf = cf
    cur_seed = np.arange(n)
    for _ in range(max(1, int(iters))):
        new_ea, new_eb, new_cf, changed = _merge_once(
            cur_ea, cur_eb, cur_cf,
            lat_px=lat_px,
            dir_dot_min=dir_dot_min,
            end_gap_px=end_gap_px,
        )
        if not changed:
            break
        new_seed = _propagate_seeds(cur_ea, cur_eb, cur_cf, new_ea, new_eb, cur_seed)
        cur_ea, cur_eb, cur_cf, cur_seed = new_ea, new_eb, new_cf, new_seed

    out = np.zeros((cur_ea.shape[0], k), dtype=uv_arr.dtype)
    out[:, int(conf_col)] = cur_cf.astype(uv_arr.dtype)
    out[:, v0] = cur_ea[:, 1].astype(uv_arr.dtype)
    out[:, h0] = cur_ea[:, 0].astype(uv_arr.dtype)
    out[:, v1] = cur_eb[:, 1].astype(uv_arr.dtype)
    out[:, h1] = cur_eb[:, 0].astype(uv_arr.dtype)
    other_cols = [c for c in range(k) if c not in {int(conf_col), v0, h0, v1, h1}]
    if other_cols:
        for new_i in range(cur_ea.shape[0]):
            seed = int(cur_seed[new_i])
            out[new_i, other_cols] = uv_arr[seed, other_cols]
    return out


def _propagate_seeds(
    prev_ea: np.ndarray,
    prev_eb: np.ndarray,
    prev_cf: np.ndarray,
    new_ea: np.ndarray,
    new_eb: np.ndarray,
    prev_seed: np.ndarray,
) -> np.ndarray:
    """For each new (merged) row, return the previous seed index of the
    highest-confidence member that maps into it (preserves "other" columns)."""
    new_mid = 0.5 * (new_ea + new_eb)
    prev_mid = 0.5 * (prev_ea + prev_eb)
    d = np.linalg.norm(prev_mid[:, None, :] - new_mid[None, :, :], axis=-1)
    nearest_new = d.argmin(axis=1)
    out = np.empty((new_ea.shape[0],), dtype=np.int64)
    for new_i in range(new_ea.shape[0]):
        members = np.where(nearest_new == new_i)[0]
        if members.size == 0:
            out[new_i] = 0
            continue
        best = int(members[np.argmax(prev_cf[members])])
        out[new_i] = int(prev_seed[best])
    return out

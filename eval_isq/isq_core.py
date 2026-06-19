"""Grid-based Instance Segmentation Quality (ISQ) metrics."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]
Segment = Tuple[Point, Point]
SegmentTuple = Tuple[float, float, float, float, int]

GEOM_CONF = 0.7
GNN_EDGE_THRESH = 0.2
GNN_NODE_CONF = 0.7

CELL_SIZE = 32
GRID_SHAPE = (32, 32)
MATCH_MAX_DIST_PX = 24.0
MATCH_MAX_ANGLE_DEG = 15.0
MEANINGFUL_COVERAGE_THRESH = 0.3
MIN_SEGMENTS_PER_PRED_POLYLINE = 2
DEFAULT_ADJACENCY_THRESHOLD = 768.0  # squared endpoint distance (YOLinO default_params)
DEFAULT_MIN_SEGMENTS_FOR_POLYLINE = 5


@dataclass(frozen=True)
class GtSegment:
    x1: float
    y1: float
    x2: float
    y2: float
    polyline_id: int
    seg_idx: int

    def as_endpoints(self) -> Segment:
        return ((self.x1, self.y1), (self.x2, self.y2))


@dataclass
class PredSegmentMatch:
    pred_polyline_idx: int
    pred_seg_idx: int
    matched_gt_seg_idx: Optional[int]
    matched_polyline_id: Optional[int]
    match_distance: float


def _seg_endpoints(seg: Segment) -> Tuple[np.ndarray, np.ndarray]:
    return np.asarray(seg[0], dtype=float), np.asarray(seg[1], dtype=float)


def _segment_length(seg: Segment) -> float:
    a, b = _seg_endpoints(seg)
    return float(np.linalg.norm(b - a))


def _point_to_segment_distance(point: np.ndarray, seg: Segment) -> float:
    a, b = _seg_endpoints(seg)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(point - proj))


def segment_distance(seg_a: Segment, seg_b: Segment) -> float:
    a0, a1 = _seg_endpoints(seg_a)
    b0, b1 = _seg_endpoints(seg_b)
    return max(
        _point_to_segment_distance(a0, seg_b),
        _point_to_segment_distance(a1, seg_b),
        _point_to_segment_distance(b0, seg_a),
        _point_to_segment_distance(b1, seg_a),
    )


def segment_angle(seg: Segment) -> float:
    (x1, y1), (x2, y2) = seg
    return float(np.arctan2(y2 - y1, x2 - x1))


def angle_diff_deg(a: float, b: float) -> float:
    d = abs(a - b)
    d = min(d, 2.0 * np.pi - d)
    d = min(d, np.pi - d)  # undirected line
    return float(np.degrees(d))


def _clip_segment_aabb(
    p0: np.ndarray,
    p1: np.ndarray,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Clip line segment to axis-aligned box; return endpoints inside the box."""
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, float(p0[0] - xmin)),
        (dx, float(xmax - p0[0])),
        (-dy, float(p0[1] - ymin)),
        (dy, float(ymax - p0[1])),
    ):
        if abs(p) < 1e-12:
            if q < 0.0:
                return None
        else:
            t = q / p
            if p < 0.0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
    if t0 > t1:
        return None
    a = p0 + t0 * (p1 - p0)
    b = p0 + t1 * (p1 - p0)
    return a, b


def _append_ordered(ordered: List[np.ndarray], pt: np.ndarray, *, eps: float = 1e-3) -> None:
    if not ordered:
        ordered.append(pt)
        return
    if float(np.linalg.norm(ordered[-1] - pt)) > eps:
        ordered.append(pt)


def polyline_segment_in_cell(
    polyline: Sequence[Sequence[float]],
    row: int,
    col: int,
    *,
    cell_size: int = CELL_SIZE,
) -> Optional[Segment]:
    """At most one segment per (cell, polyline): cell-boundary chord of the clipped polyline."""
    pts = [np.asarray(p[:2], dtype=float) for p in polyline if len(p) >= 2]
    if len(pts) < 2:
        return None

    xmin = float(col * cell_size)
    ymin = float(row * cell_size)
    xmax = xmin + float(cell_size)
    ymax = ymin + float(cell_size)

    ordered: List[np.ndarray] = []
    for i in range(len(pts) - 1):
        clipped = _clip_segment_aabb(pts[i], pts[i + 1], xmin, ymin, xmax, ymax)
        if clipped is None:
            continue
        a, b = clipped
        _append_ordered(ordered, a)
        _append_ordered(ordered, b)

    if len(ordered) < 2:
        return None
    p0, p1 = ordered[0], ordered[-1]
    if float(np.linalg.norm(p1 - p0)) < 1e-3:
        return None
    return ((float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])))


def polyline_to_cell_segments(
    polyline: Sequence[Sequence[float]],
    polyline_id: int,
    *,
    cell_size: int = CELL_SIZE,
    image_size: int = 1024,
    start_idx: int = 0,
) -> Tuple[List[GtSegment], int]:
    """One GT segment per grid cell crossed by this polyline (cell x polyline_id)."""
    n_rows = int(image_size // cell_size)
    n_cols = int(image_size // cell_size)
    out: List[GtSegment] = []
    idx = start_idx
    for row in range(n_rows):
        for col in range(n_cols):
            seg = polyline_segment_in_cell(polyline, row, col, cell_size=cell_size)
            if seg is None:
                continue
            (x1, y1), (x2, y2) = seg
            out.append(
                GtSegment(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    polyline_id=int(polyline_id),
                    seg_idx=idx,
                )
            )
            idx += 1
    return out, idx


def gt_polylines_to_segments(
    polylines: Sequence[Sequence[Sequence[float]]],
    instance_ids: Sequence[int],
    *,
    cell_size: int = CELL_SIZE,
    image_size: int = 1024,
) -> List[GtSegment]:
    """GT segments: at most one per (32x32 cell, GT polyline_id)."""
    all_segs: List[GtSegment] = []
    next_idx = 0
    for poly, pid in zip(polylines, instance_ids):
        segs, next_idx = polyline_to_cell_segments(
            poly, int(pid), cell_size=cell_size, image_size=image_size, start_idx=next_idx
        )
        all_segs.extend(segs)
    return all_segs


def _endpoint_key(pt: Point, quant: float = 0.5) -> Tuple[int, int]:
    return (int(round(pt[0] / quant)), int(round(pt[1] / quant)))


def _geom_segments_to_fitlines_cc(
    segments: Sequence[Segment],
    *,
    adjacency_threshold: float = DEFAULT_ADJACENCY_THRESHOLD,
    min_segments_for_polyline: int = DEFAULT_MIN_SEGMENTS_FOR_POLYLINE,
) -> List[List[Segment]]:
    """YOLinO rule-based grouping: adjacency graph + connected components only (no smooth/spline)."""
    if not segments:
        return []
    from yolino.postprocessing.line_fit import get_adjacency_list, get_connected_components

    startpoints = np.asarray([seg[0] for seg in segments], dtype=float)
    endpoints = np.asarray([seg[1] for seg in segments], dtype=float)
    confidences = np.ones((len(segments),), dtype=float)

    adjacency, reversed_adjacency = get_adjacency_list(
        float(adjacency_threshold), endpoints, startpoints
    )
    roots = [node for node in adjacency.keys() if not adjacency[node]]

    class _Paths:
        @staticmethod
        def generate_debug_image_file_path(**_kwargs) -> str:
            return "/tmp/isq_fitlines_cc.png"

    class _Args:
        paths = _Paths()

    _, seg_id_groups = get_connected_components(
        confidences,
        "isq_geom",
        np.zeros((1, 1, 3), dtype=np.uint8),
        int(min_segments_for_polyline),
        _Args(),
        reversed_adjacency,
        roots,
        list(segments),
        write_debug_images=False,
    )

    out: List[List[Segment]] = []
    for seg_ids in seg_id_groups:
        group = [segments[int(sid)] for sid in seg_ids]
        if group:
            out.append(list(group))
    return out


def polylines_to_segment_lists(
    polylines: Sequence[Sequence[Sequence[float]]],
    *,
    adjacency_threshold: float = DEFAULT_ADJACENCY_THRESHOLD,
    min_segments_for_polyline: int = DEFAULT_MIN_SEGMENTS_FOR_POLYLINE,
) -> List[List[Segment]]:
    segs: List[Segment] = []
    for poly in polylines:
        if len(poly) < 2:
            continue
        segs.append(
            ((float(poly[0][0]), float(poly[0][1])), (float(poly[1][0]), float(poly[1][1])))
        )
    return _geom_segments_to_fitlines_cc(
        segs,
        adjacency_threshold=adjacency_threshold,
        min_segments_for_polyline=min_segments_for_polyline,
    )


def _to_numpy(x: object) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()  # type: ignore[union-attr]
    return np.asarray(x)


def geom_polylines_to_pred_polylines(
    geom_polylines: Sequence[Sequence[Sequence[float]]],
    *,
    adjacency_threshold: float = DEFAULT_ADJACENCY_THRESHOLD,
    min_segments_for_polyline: int = DEFAULT_MIN_SEGMENTS_FOR_POLYLINE,
) -> List[List[Segment]]:
    """Group raw geom cell segments via YOLinO ``fit_lines`` CC (no smoothing / spline)."""
    segments: List[Segment] = []
    for poly in geom_polylines:
        if len(poly) < 2:
            continue
        segments.append(
            ((float(poly[0][0]), float(poly[0][1])), (float(poly[1][0]), float(poly[1][1])))
        )
    return _geom_segments_to_fitlines_cc(
        segments,
        adjacency_threshold=adjacency_threshold,
        min_segments_for_polyline=min_segments_for_polyline,
    )


def gnn_segments_to_polylines(
    segments: Sequence[Segment],
    gnn_single: dict,
    *,
    edge_thresh: float = GNN_EDGE_THRESH,
    node_conf_thresh: float = GNN_NODE_CONF,
) -> List[List[Segment]]:
    """Group GNN **node** segments (end_a→end_b, same as geom cells) into polylines.

    GNN edges (midpoint connectors) are used only for union-find grouping, not as
    extra ISQ segments — otherwise segment count exceeds geom (~N nodes + ~E edges).
    ``segments`` from :func:`extract_gnn_segments` is ignored; grouping uses ``gnn_single``.
    """
    node_mid = _to_numpy(gnn_single["node_mid_px"])
    node_valid = _to_numpy(gnn_single["node_valid"]).astype(bool)
    neighbors = _to_numpy(gnn_single["neighbors"]).astype(np.int64)
    neigh_valid = _to_numpy(gnn_single["neigh_valid"]).astype(bool)
    edge_logits = _to_numpy(gnn_single["edge_logits"]).astype(np.float32)
    node_ea = _to_numpy(gnn_single.get("node_end_a_px"))
    node_eb = _to_numpy(gnn_single.get("node_end_b_px"))
    node_conf = gnn_single.get("node_conf")
    if node_conf is not None:
        node_conf = _to_numpy(node_conf).astype(np.float32)
        node_valid = node_valid & (node_conf >= float(node_conf_thresh))

    n_nodes = int(node_mid.shape[0])
    parent = np.arange(n_nodes, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return int(x)

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    probs = 1.0 / (1.0 + np.exp(-edge_logits))
    edge_pairs = set()
    for ni in range(n_nodes):
        if not bool(node_valid[ni]):
            continue
        for kj in range(neighbors.shape[1]):
            if not bool(neigh_valid[ni, kj]):
                continue
            if float(probs[ni, kj]) < float(edge_thresh):
                continue
            nj = int(neighbors[ni, kj])
            if nj < 0 or nj >= n_nodes or not bool(node_valid[nj]):
                continue
            a, b = (ni, nj) if ni < nj else (nj, ni)
            edge_pairs.add((a, b))
            union(a, b)

    in_graph = np.zeros((n_nodes,), dtype=bool)
    for a, b in edge_pairs:
        in_graph[a] = True
        in_graph[b] = True

    comp_segments: Dict[int, List[Segment]] = {}
    lonely: List[Segment] = []

    for i in range(n_nodes):
        if not bool(node_valid[i]):
            continue
        seg = ((float(node_ea[i][0]), float(node_ea[i][1])), (float(node_eb[i][0]), float(node_eb[i][1])))
        if not bool(in_graph[i]):
            lonely.append(seg)
            continue
        root = find(i)
        comp_segments.setdefault(root, []).append(seg)

    polylines = list(comp_segments.values())
    if lonely:
        polylines.extend([[seg] for seg in lonely])
    return polylines


def pred_polylines_to_segment_tuples(
    pred_polylines: Sequence[Sequence[Segment]],
) -> List[SegmentTuple]:
    out: List[SegmentTuple] = []
    for pid, poly in enumerate(pred_polylines):
        for (x1, y1), (x2, y2) in poly:
            out.append((float(x1), float(y1), float(x2), float(y2), int(pid)))
    return out


def segment_tuples_to_pred_polylines(
    segment_tuples: Sequence[SegmentTuple],
) -> List[List[Segment]]:
    by_id: Dict[int, List[Segment]] = defaultdict(list)
    for x1, y1, x2, y2, pid in segment_tuples:
        by_id[int(pid)].append(((float(x1), float(y1)), (float(x2), float(y2))))
    return [by_id[k] for k in sorted(by_id)]


def _segment_bbox(seg: Segment, margin: float) -> Tuple[float, float, float, float]:
    xs = [seg[0][0], seg[1][0]]
    ys = [seg[0][1], seg[1][1]]
    return (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)


def _bbox_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def build_gt_spatial_index(
    gt_segments: Sequence[GtSegment],
    *,
    bucket_size: int = CELL_SIZE,
) -> Dict[Tuple[int, int], List[GtSegment]]:
    buckets: Dict[Tuple[int, int], List[GtSegment]] = defaultdict(list)
    for gt in gt_segments:
        cx = 0.5 * (gt.x1 + gt.x2)
        cy = 0.5 * (gt.y1 + gt.y2)
        buckets[(int(cx // bucket_size), int(cy // bucket_size))].append(gt)
    return buckets


def _gt_candidates_for_segment(
    pred_seg: Segment,
    gt_index: Dict[Tuple[int, int], List[GtSegment]],
    *,
    bucket_size: int = CELL_SIZE,
    margin: float = MATCH_MAX_DIST_PX,
) -> List[GtSegment]:
    pred_bb = _segment_bbox(pred_seg, margin=margin)
    x0, y0, x1, y1 = pred_bb
    ix0 = int(x0 // bucket_size)
    iy0 = int(y0 // bucket_size)
    ix1 = int(x1 // bucket_size)
    iy1 = int(y1 // bucket_size)
    out: List[GtSegment] = []
    seen: set[int] = set()
    for ix in range(ix0, ix1 + 1):
        for iy in range(iy0, iy1 + 1):
            for gt in gt_index.get((ix, iy), []):
                if gt.seg_idx in seen:
                    continue
                seen.add(gt.seg_idx)
                if _bbox_overlap(pred_bb, _segment_bbox(gt.as_endpoints(), margin=margin)):
                    out.append(gt)
    return out


def match_pred_segment_to_gt(
    pred_seg: Segment,
    gt_segments: Sequence[GtSegment],
    *,
    gt_index: Optional[Dict[Tuple[int, int], List[GtSegment]]] = None,
    max_dist: float = MATCH_MAX_DIST_PX,
    max_angle_deg: float = MATCH_MAX_ANGLE_DEG,
) -> Tuple[Optional[int], Optional[int], float]:
    """Return (gt_seg_idx, polyline_id, distance) for closest qualifying GT segment."""
    pred_ang = segment_angle(pred_seg)
    pred_bb = _segment_bbox(pred_seg, margin=max_dist)
    candidates = (
        _gt_candidates_for_segment(pred_seg, gt_index, margin=max_dist)
        if gt_index is not None
        else list(gt_segments)
    )
    best_idx: Optional[int] = None
    best_dist = float("inf")
    for gt in candidates:
        gt_seg = gt.as_endpoints()
        if gt_index is None and not _bbox_overlap(pred_bb, _segment_bbox(gt_seg, margin=max_dist)):
            continue
        dist = segment_distance(pred_seg, gt_seg)
        if dist >= max_dist:
            continue
        if angle_diff_deg(pred_ang, segment_angle(gt_seg)) >= max_angle_deg:
            continue
        if dist < best_dist:
            best_dist = dist
            best_idx = gt.seg_idx
    if best_idx is None:
        return None, None, float("inf")
    gt = next(g for g in gt_segments if g.seg_idx == best_idx)
    return best_idx, gt.polyline_id, best_dist


def _dominant_gt_id_for_pred(
    pi: int, matches: Sequence[PredSegmentMatch]
) -> Optional[int]:
    ids = [
        m.matched_polyline_id
        for m in matches
        if m.pred_polyline_idx == pi and m.matched_polyline_id is not None
    ]
    if not ids:
        return None
    return Counter(ids).most_common(1)[0][0]


def _pred_coverage_on_gt(
    pi: int,
    gid: int,
    matches: Sequence[PredSegmentMatch],
    gt_by_idx: Dict[int, GtSegment],
    n_gt: int,
) -> float:
    """Recall-style coverage: distinct GT segments of ``gid`` matched by pred polyline ``pi``."""
    if n_gt <= 0:
        return 0.0
    hit_gt: set[int] = set()
    for m in matches:
        if m.pred_polyline_idx != pi or m.matched_gt_seg_idx is None:
            continue
        gt = gt_by_idx[m.matched_gt_seg_idx]
        if gt.polyline_id == gid:
            hit_gt.add(m.matched_gt_seg_idx)
    return len(hit_gt) / float(n_gt)


def _count_over_split_instances(
    gt_ids: Sequence[int],
    gt_seg_by_id: Dict[int, List[int]],
    pred_polylines: Sequence[Sequence[Segment]],
    matches: Sequence[PredSegmentMatch],
    gt_by_idx: Dict[int, GtSegment],
) -> int:
    """GT-centric over-split: >=2 pred polylines with dominant_id==gid and coverage >= thresh."""
    n_over = 0
    n_preds = len(pred_polylines)
    for gid in gt_ids:
        n_gt = len(gt_seg_by_id[gid])
        if n_gt == 0:
            continue
        meaningful_preds = 0
        for pi in range(n_preds):
            if len(pred_polylines[pi]) == 0:
                continue
            if _dominant_gt_id_for_pred(pi, matches) != gid:
                continue
            if _pred_coverage_on_gt(pi, gid, matches, gt_by_idx, n_gt) >= MEANINGFUL_COVERAGE_THRESH:
                meaningful_preds += 1
        if meaningful_preds >= 2:
            n_over += 1
    return n_over


def compute_isq_image(
    gt_segments: Sequence[GtSegment],
    pred_polylines: Sequence[Sequence[Segment]],
) -> Dict[str, object]:
    """Compute ISQ counts and auxiliary metrics for one image."""
    gt_index = build_gt_spatial_index(gt_segments)
    gt_by_idx = {g.seg_idx: g for g in gt_segments}
    matches: List[PredSegmentMatch] = []
    for pi, poly in enumerate(pred_polylines):
        for si, seg in enumerate(poly):
            gt_idx, gt_pid, dist = match_pred_segment_to_gt(seg, gt_segments, gt_index=gt_index)
            matches.append(
                PredSegmentMatch(
                    pred_polyline_idx=pi,
                    pred_seg_idx=si,
                    matched_gt_seg_idx=gt_idx,
                    matched_polyline_id=gt_pid,
                    match_distance=dist,
                )
            )

    eval_poly_indices = [
        pi
        for pi, poly in enumerate(pred_polylines)
        if len(poly) >= MIN_SEGMENTS_PER_PRED_POLYLINE
    ]
    dominant_ids: Dict[int, Optional[int]] = {}
    for pi in eval_poly_indices:
        ids = [
            m.matched_polyline_id
            for m in matches
            if m.pred_polyline_idx == pi and m.matched_polyline_id is not None
        ]
        if ids:
            dominant_ids[pi] = Counter(ids).most_common(1)[0][0]
        else:
            dominant_ids[pi] = None

    claimed_gt: set[int] = set()
    tp = fp = 0
    for m in matches:
        if m.pred_polyline_idx not in dominant_ids:
            continue
        dom = dominant_ids[m.pred_polyline_idx]
        if dom is None or m.matched_polyline_id != dom or m.matched_gt_seg_idx is None:
            fp += 1
            continue
        if m.matched_gt_seg_idx in claimed_gt:
            fp += 1
            continue
        claimed_gt.add(m.matched_gt_seg_idx)
        tp += 1

    fn = sum(1 for g in gt_segments if g.seg_idx not in claimed_gt)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)

    gt_ids = sorted({g.polyline_id for g in gt_segments})
    gt_seg_by_id: Dict[int, List[int]] = defaultdict(list)
    for g in gt_segments:
        gt_seg_by_id[g.polyline_id].append(g.seg_idx)

    # Pairwise coverage for over-split / under-merge.
    pred_to_gt_hits: Dict[Tuple[int, int], set[int]] = defaultdict(set)
    gt_to_pred_hits: Dict[Tuple[int, int], set[int]] = defaultdict(set)
    for m in matches:
        if m.matched_gt_seg_idx is None or m.matched_polyline_id is None:
            continue
        gt = gt_by_idx[m.matched_gt_seg_idx]
        pred_to_gt_hits[(m.pred_polyline_idx, gt.polyline_id)].add(m.matched_gt_seg_idx)
        gt_to_pred_hits[(gt.polyline_id, m.pred_polyline_idx)].add(m.matched_gt_seg_idx)

    over_split_count = _count_over_split_instances(
        gt_ids, gt_seg_by_id, pred_polylines, matches, gt_by_idx
    )

    under_merge = 0
    for pi, poly in enumerate(pred_polylines):
        n_pred = len(poly)
        if n_pred == 0:
            continue
        meaningful_gts = 0
        for gid in gt_ids:
            n_hit = len(pred_to_gt_hits.get((pi, gid), set()))
            precision_pg = n_hit / n_pred
            recall_pg = n_hit / max(len(gt_seg_by_id[gid]), 1)
            if precision_pg > MEANINGFUL_COVERAGE_THRESH or recall_pg > MEANINGFUL_COVERAGE_THRESH:
                meaningful_gts += 1
        if meaningful_gts >= 2:
            under_merge += 1

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "over_split": int(over_split_count),
        "over_split_count": int(over_split_count),
        "under_merge": int(under_merge),
        "n_gt_segments": len(gt_segments),
        "n_pred_polylines": len(pred_polylines),
        "n_eval_pred_polylines": len(eval_poly_indices),
        "n_gt_instances": len(gt_ids),
    }


def summarize_isq(per_image: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not per_image:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "over_split": 0.0,
            "over_split_count": 0,
            "over_split_images": {},
            "under_merge": 0.0,
            "n_images": 0,
        }
    over_split_images = {
        str(m["stem"]): int(m.get("over_split_count", m.get("over_split", 0)))
        for m in per_image
        if m.get("stem") is not None
    }
    return {
        "precision": float(np.mean([m["precision"] for m in per_image])),
        "recall": float(np.mean([m["recall"] for m in per_image])),
        "f1": float(np.mean([m["f1"] for m in per_image])),
        "over_split": float(np.mean([m.get("over_split_count", m["over_split"]) for m in per_image])),
        "over_split_count": int(sum(m.get("over_split_count", m["over_split"]) for m in per_image)),
        "over_split_images": over_split_images,
        "under_merge": float(np.mean([m["under_merge"] for m in per_image])),
        "n_images": len(per_image),
    }

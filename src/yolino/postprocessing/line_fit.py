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
from copy import deepcopy

import numpy as np
from scipy import interpolate
from yolino.grid.coordinates import validate_input_structure
from yolino.model.variable_structure import VariableStructure
from yolino.utils.enums import CoordinateSystem, Variables, ImageIdx, ColorStyle, LINE
from yolino.viz.plot import plot, get_color


def breadth_first_connected_component(graph, start):
    # keep track of all visited nodes
    explored = []
    explored_tuple = []
    # keep track of nodes to be checked
    level = 0
    queue = [(start, level)]
    # keep looping until there are nodes still to be checked
    while queue:
        # pop shallowest node (first node) from queue
        node, level = queue.pop(0)
        if node not in explored:
            # add node to list of checked nodes
            explored.append(node)
            explored_tuple.append((node, level))
            neighbours = graph[node]
            level += 1
            # add neighbours of node to queue
            for neighbour in neighbours:
                queue.append((neighbour, level))
    return explored_tuple


def fit_lines(lines_uv, coords: VariableStructure, confidence_threshold, adjacency_threshold, grid_shape,
              min_segments_for_polyline,
              cell_size, image, file_name, paths, args, split,
              write_debug_images: bool = True,
              spline_s: float = 0.05,
              angle_thr_deg: float | None = None,
              collinear_thr_px: float | None = None,
              second_pass_merge: bool = False,
              second_pass_gap_px: float | None = None,
              skip_spline: bool = False):
    lines_uv = np.asarray(lines_uv, dtype=np.float32)
    if lines_uv.ndim == 2:
        lines_uv = np.expand_dims(lines_uv, axis=0)
    validate_input_structure(lines_uv, CoordinateSystem.UV_SPLIT)
    conf_cols = np.asarray(coords.get_position_within_prediction(Variables.CONF)).ravel()
    geom_pos = coords.get_position_within_prediction(Variables.GEOMETRY)

    # get_position_within_prediction may return multiple column indices — threshold on first CONF column.
    if conf_cols.size:
        conf_i = int(conf_cols[0])
        mask = (lines_uv[0, :, conf_i] > confidence_threshold)
        lines_uv = lines_uv[:, mask, :]
    else:
        conf_i = -1

    if lines_uv.shape[1] == 0:
        return []

    if write_debug_images:
        name = paths.generate_debug_image_file_path(file_name=file_name, idx=ImageIdx.PRED, suffix="filter")
        img = deepcopy(image)
        _, _ = plot(np.expand_dims(lines_uv, axis=0), name, img, coords=coords,
                    colorstyle=ColorStyle.UNIFORM,
                    coordinates=CoordinateSystem.UV_SPLIT, imageidx=ImageIdx.PRED, training_vars_only=True)

    lv = lines_uv[0]
    segments = [((float(x1), float(y1)), (float(x2), float(y2)))
                for x1, y1, x2, y2 in lv[:, geom_pos]]
    startpoints = np.asarray(lv[:, 0:2], dtype=float)
    endpoints = np.asarray(lv[:, 2:4], dtype=float)
    if conf_i >= 0:
        confidences = np.asarray(lv[:, conf_i], dtype=float)
    else:
        confidences = np.ones((lv.shape[0],), dtype=float)

    # build adjacency list
    adjacency, reversed_adjacency = get_adjacency_list(
        adjacency_threshold, endpoints, startpoints,
        angle_thr_deg=angle_thr_deg,
        collinear_thr_px=collinear_thr_px,
    )

    # Infer roots
    roots = [node for node in adjacency.keys() if not adjacency[node]]

    # Find connected components
    polylines, polylines_as_segment_ids = get_connected_components(
        confidences, file_name, image,
        min_segments_for_polyline, paths, reversed_adjacency,
        roots, segments,
        write_debug_images=write_debug_images)

    # Smooth poylines
    end_start_distances, smooth_polylines = smoothing_polylines(
        file_name, image, paths, polylines,
        write_debug_images=write_debug_images)

    # Optional second pass: merge 1st-pass polylines by angle+collinearity(+gap),
    # then spline-fit the merged tracks.
    if second_pass_merge and len(smooth_polylines) > 1 and angle_thr_deg is not None and collinear_thr_px is not None:
        smooth_polylines = merge_polylines_second_pass(
            smooth_polylines,
            angle_thr_deg=float(angle_thr_deg),
            collinear_thr_px=float(collinear_thr_px),
            gap_px=float(second_pass_gap_px) if second_pass_gap_px is not None else float(adjacency_threshold) ** 0.5,
        )

    # if more than one polylines contain the same segments, only one can remain (possibly based on connectedness)
    to_be_removed = remove_duplicates(end_start_distances, polylines_as_segment_ids)

    if skip_spline:
        return smooth_polylines

    # make spline
    splines = fit_spline(cell_size, file_name, image, paths, smooth_polylines, to_be_removed,
                        write_debug_images=write_debug_images, spline_s=spline_s)

    # # Create submission
    # y_samples = np.linspace(160, 710, 56)
    # splines_reformatted = [[spline[0] * 2, spline[1] * 2 + 80] for spline in splines]
    # submission = []
    # for spline in splines_reformatted:
    #     x = spline[0]
    #     y = spline[1]
    #     x_result_values = np.interp(y_samples, y, x, -2, -2)
    #     submission.append(x_result_values.tolist())

    # # Evaluate
    # tus = TusimpleDataset(split=split, args=args, load_only_labels=True)
    # gt_lanes = tus.lanes

    # # TODO fix tus benchmark
    # le = LaneEval()
    # accuracy, fp, fn = le.bench(submission, gt_lanes, y_samples, 0)
    # results_string = "(Acc %.3f, FP %.2f, FN %.2f)" % (accuracy, fp, fn)

    # from skimage.transform import resize
    # upscaled_image = resize(image.copy() / 255, (640, 1280))
    # mask = np.zeros((720, 1280, 3))
    # mask[80:, :, :] += upscaled_image

    return splines


def fit_spline(cell_size, file_name, image, paths, smooth_polylines, to_be_removed,
               write_debug_images: bool = True,
               spline_s: float = 0.05):
    splines = []
    for idx, smooth_polyline in enumerate(smooth_polylines):
        if idx in to_be_removed:
            continue
        lp = [l[0] for l in smooth_polyline]
        lp.append(smooth_polyline[-1][1])
        lp = np.array(lp)
        x = lp[:, 0] / cell_size[0]
        y = lp[:, 1] / cell_size[1]
        n = int(len(x))
        if n < 2:
            continue
        # splprep default k=3 needs len > k; degenerate / duplicate geometry raises ValueError.
        k = min(3, max(1, n - 1))
        tck = None
        s_candidates = (
            float(spline_s),
            float(spline_s) * (10.0 + float(n)),
            float(spline_s) * (100.0 + 10.0 * float(n)),
            float(max(n, 4) ** 2),
        )
        for s_try in s_candidates:
            try:
                tck, _u = interpolate.splprep([x, y], s=max(s_try, 1e-12), k=k)
                break
            except ValueError:
                continue
        if tck is None and k > 1:
            try:
                tck, _u = interpolate.splprep([x, y], s=max(float(spline_s), 1e-9) * 1e3, k=1)
            except ValueError:
                tck = None
        if tck is None:
            try:
                tck, _u = interpolate.splprep([x, y], s=0.0, k=1)
            except ValueError:
                splines.append(np.stack([lp[:, 0], lp[:, 1]], axis=0))
                continue
        unew = np.arange(0, 1.01, 0.01)
        out = interpolate.splev(unew, tck)
        cs0 = float(cell_size[0])
        cs1 = float(cell_size[1]) if len(np.asarray(cell_size).reshape(-1)) > 1 else cs0
        # out[0]/out[1] were fit on row/col normalized by cs0/cs1; restore pixel row then col.
        row_px = np.asarray(out[0], dtype=np.float64) * cs0
        col_px = np.asarray(out[1], dtype=np.float64) * cs1
        splines.append(np.stack([row_px, col_px], axis=0))
    if write_debug_images and splines:
        name = paths.generate_debug_image_file_path(file_name=file_name, idx=ImageIdx.PRED, suffix="spline")
        img = deepcopy(image)
        plot_coords = VariableStructure(line_representation_enum=LINE.POINTS, num_conf=0,
                                        vars_to_train=[Variables.GEOMETRY])
        converted_splines = np.expand_dims(
            np.asarray([[[x, y] for x, y in zip(instance[0], instance[1])] for instance in splines]), axis=0)
        plot(converted_splines, name, img, coords=plot_coords,
             colorstyle=ColorStyle.ID,
             coordinates=CoordinateSystem.UV_CONTINUOUS, imageidx=ImageIdx.PRED, training_vars_only=True)
    return splines


def remove_duplicates(end_start_distances, polylines_as_segment_ids):
    duplicates = set([])
    for id, pl in enumerate(polylines_as_segment_ids):
        others = range(len(polylines_as_segment_ids))
        for seg_id in pl:
            for other in others:
                if other != id:
                    if seg_id in polylines_as_segment_ids[other]:
                        duplicates.add((id, other))
    to_be_removed = []
    for duplicate in duplicates:
        if end_start_distances[duplicate[0]] <= end_start_distances[duplicate[1]]:
            to_be_removed.append(duplicate[0])
        else:
            to_be_removed.append(duplicate[1])
    if to_be_removed:
        print("to_be_removed", to_be_removed)
    return to_be_removed


def smoothing_polylines(file_name, image, paths, polylines, write_debug_images: bool = True):
    smooth_polylines = []
    end_start_distances = []
    for polyline in polylines:
        smoothed = []
        prev = None
        end_start_distance = 0
        for l_idx, line in enumerate(polyline):
            if prev is not None:
                end_start_distance += np.linalg.norm(np.array(prev[1]) - np.array(line[0]))
                midpoint = tuple((np.array(prev[1]) + np.array(line[0])) / 2)
                smoothed.append((prev[0], midpoint))
                prev = (midpoint, line[1])
            else:
                prev = line
        smoothed.append(prev)
        end_start_distances.append(end_start_distance)
        smooth_polylines.append(smoothed)
    if write_debug_images:
        name = paths.generate_debug_image_file_path(file_name=file_name, idx=ImageIdx.PRED, suffix="smooth")
        img = deepcopy(image)
        plot_coords = VariableStructure(line_representation_enum=LINE.POINTS, num_conf=0,
                                        vars_to_train=[Variables.GEOMETRY])
        for i, instance in enumerate(smooth_polylines):
            color = get_color(colorstyle=ColorStyle.ID, idx=i)
            img, _ = plot(np.asarray([instance]).reshape((1, -1, 4)), name, img, coords=plot_coords,
                          colorstyle=ColorStyle.UNIFORM, color=color, coordinates=CoordinateSystem.UV_SPLIT,
                          imageidx=ImageIdx.PRED, training_vars_only=True)
    return end_start_distances, smooth_polylines


def get_connected_components(confidences, file_name, image, min_segments_for_polyline, paths, reversed_adjacency,
                             roots,
                             segments, write_debug_images: bool = True):
    # visits all the nodes of a graph (connected component) using BFS
    polylines = []
    polylines_as_segment_ids = []
    for root in roots:
        polyline = []
        polyline_as_segment_ids = []
        bfs_result = breadth_first_connected_component(reversed_adjacency, root)
        current_level = 0
        merge_segments = []
        merge_confidences = []
        for segment_id, level in bfs_result:
            polyline_as_segment_ids.append(segment_id)
            if level == current_level:
                merge_segments.append(np.array(segments[segment_id]))
                merge_confidences.append(confidences[segment_id])
            else:
                weights = np.array(merge_confidences) / sum(merge_confidences)
                merged_segment = np.sum(
                    np.array([merge_segment * weight for merge_segment, weight in zip(merge_segments, weights)]),
                    axis=0)
                merged_segment = tuple([tuple(x) for x in merged_segment])
                polyline.append(merged_segment)
                merge_segments = [np.array(segments[segment_id])]
                merge_confidences = [(confidences[segment_id])]
                current_level = level
        weights = np.array(merge_confidences) / sum(merge_confidences)
        merge_segments[0] * weights[0]
        merged_segment = np.sum(
            np.array([merge_segment * weight for merge_segment, weight in zip(merge_segments, weights)]), axis=0)
        merged_segment = tuple([tuple(x) for x in merged_segment])
        polyline.append(merged_segment)
        polylines.append(polyline[::-1])
        polylines_as_segment_ids.append(polyline_as_segment_ids)
    polyline_num_threshold = int(min_segments_for_polyline)
    polylines = [pl for pl in polylines if len(pl) >= polyline_num_threshold]
    polylines_as_segment_ids = [pl for pl in polylines_as_segment_ids if len(pl) >= polyline_num_threshold]

    if write_debug_images:
        name = paths.generate_debug_image_file_path(file_name=file_name, idx=ImageIdx.PRED, suffix="cc")
        img = deepcopy(image)
        plot_coords = VariableStructure(line_representation_enum=LINE.POINTS, num_conf=0,
                                        vars_to_train=[Variables.GEOMETRY])
        for i_idx, instance in enumerate(polylines):
            c = get_color(ColorStyle.ID, idx=i_idx)
            img, _ = plot(np.asarray(instance).reshape((1, -1, 4)), name, img, coords=plot_coords,
                          colorstyle=ColorStyle.UNIFORM,
                          color=c,
                          coordinates=CoordinateSystem.UV_SPLIT, imageidx=ImageIdx.PRED,
                          training_vars_only=True)
    return polylines, polylines_as_segment_ids


def _point_line_distance_2d(point, line_p0, line_p1):
    p = np.asarray(point, dtype=float)
    a = np.asarray(line_p0, dtype=float)
    b = np.asarray(line_p1, dtype=float)
    ab = b - a
    denom = np.linalg.norm(ab)
    if denom < 1e-6:
        return float(np.linalg.norm(p - a))
    # 2D cross product magnitude over |ab|
    num = abs(ab[0] * (a[1] - p[1]) - (a[0] - p[0]) * ab[1])
    return float(num / denom)


def _segment_angle(seg):
    (x1, y1), (x2, y2) = seg
    return float(np.arctan2(y2 - y1, x2 - x1))


def _angle_diff_rad(a, b):
    d = abs(a - b)
    return min(d, 2 * np.pi - d)


def merge_polylines_second_pass(smooth_polylines, angle_thr_deg: float, collinear_thr_px: float, gap_px: float):
    """Merge first-pass polylines that are likely parts of the same wire."""
    if len(smooth_polylines) <= 1:
        return smooth_polylines

    angle_thr = float(np.deg2rad(angle_thr_deg))
    used = [False] * len(smooth_polylines)
    out = []

    for i in range(len(smooth_polylines)):
        if used[i]:
            continue
        cur = list(smooth_polylines[i])
        used[i] = True
        changed = True

        while changed:
            changed = False
            if len(cur) == 0:
                break
            end_seg = cur[-1]
            end_pt = np.asarray(end_seg[1], dtype=float)
            end_ang = _segment_angle(end_seg)

            best_j = None
            best_gap = float("inf")
            for j in range(len(smooth_polylines)):
                if used[j]:
                    continue
                cand = smooth_polylines[j]
                if len(cand) == 0:
                    continue
                start_seg = cand[0]
                start_pt = np.asarray(start_seg[0], dtype=float)
                gap = float(np.linalg.norm(start_pt - end_pt))
                if gap > gap_px:
                    continue

                ang = _segment_angle(start_seg)
                if _angle_diff_rad(end_ang, ang) > angle_thr:
                    continue

                # Collinearity: candidate startpoint close to current tail supporting line.
                perp = _point_line_distance_2d(start_pt, np.asarray(end_seg[0], dtype=float), end_pt)
                if perp > collinear_thr_px:
                    continue

                if gap < best_gap:
                    best_gap = gap
                    best_j = j

            if best_j is not None:
                cur.extend(smooth_polylines[best_j])
                used[best_j] = True
                changed = True

        out.append(cur)

    return out


def get_adjacency_list(adjacency_threshold, endpoints, startpoints,
                       angle_thr_deg: float | None = None,
                       collinear_thr_px: float | None = None):
    adjacency_d_threshold = adjacency_threshold
    angle_thr = None if angle_thr_deg is None else float(np.deg2rad(angle_thr_deg))
    col_thr = None if collinear_thr_px is None else float(collinear_thr_px)
    adjacency = dict()
    dirs = endpoints - startpoints
    seg_angles = np.arctan2(dirs[:, 1], dirs[:, 0])
    for idx, endpoint in enumerate(endpoints):
        distances = np.sum((startpoints - endpoint) ** 2, axis=1)
        d_argsort = distances.argsort()
        chosen = None
        for cand in d_argsort:
            if cand == idx:
                continue
            d_value = distances[cand]
            if d_value > adjacency_d_threshold:
                break
            y_distance = startpoints[cand][1] - endpoint[1]
            if y_distance < -0.25 * 32 * 32:
                continue

            if angle_thr is not None:
                da = abs(seg_angles[idx] - seg_angles[cand])
                da = min(da, 2 * np.pi - da)
                if da > angle_thr:
                    continue

            if col_thr is not None:
                # Candidate startpoint should lie close to idx segment's supporting line.
                p0 = startpoints[idx]
                p1 = endpoints[idx]
                perp = _point_line_distance_2d(startpoints[cand], p0, p1)
                if perp > col_thr:
                    continue

            chosen = int(cand)
            break

        adjacency[idx] = [chosen] if chosen is not None else []

    # Build reveserd adjacency lsit
    revesed_adjacency = {}
    for key, value in adjacency.items():
        if value:
            value = value[0]
            if value not in revesed_adjacency.keys():
                revesed_adjacency[value] = [key]
            else:
                revesed_adjacency[value].append(key)
    for key, value in adjacency.items():
        if key not in revesed_adjacency.keys():
            revesed_adjacency[key] = []

    return adjacency, revesed_adjacency

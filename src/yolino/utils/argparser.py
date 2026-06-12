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
import argparse
import os
import pwd

from yolino.utils.enums import Dataset, Network, Level, Logger, Optimizer, LINE, LOSS, ACTIVATION, Scheduler, Metric, \
    Augmentation, Variables, Distance, LossWeighting, AnchorDistribution, AnchorVariables
from yolino.utils.logger import Log


class AbstractParser(argparse.Action):
    def parse_fct(self, namespace, values, variable_type):  #
        if values == "":
            values = list()
        elif "," in values:
            values = list(map(variable_type, values.replace(" ", "").split(",")))
        else:
            values = list(map(variable_type, values.replace(",", "").split(" ")))
        setattr(namespace, self.dest, values)


class ParseActivation(AbstractParser):
    def __call__(self, parser, namespace, values, option_string=None):
        self.parse_fct(namespace, values, ACTIVATION)


class ParseLoss(AbstractParser):
    def __call__(self, parser, namespace, values, option_string=None):
        self.parse_fct(namespace, values, LOSS)


class ParseAnchorVariables(AbstractParser):
    def __call__(self, parser, namespace, values, option_string=None):
        self.parse_fct(namespace, values, AnchorVariables)


class ParseBool(AbstractParser):
    def __call__(self, parser, namespace, values, option_string=None):

        if values == "":
            values = True  # use as a flag
        elif values.lower() == "false":
            values = False
        elif values.lower() == "true":
            values = True
        else:
            raise ValueError(values)

        setattr(namespace, self.dest, values)


class ParseVariables(AbstractParser):
    def __call__(self, parser, namespace, values, option_string=None):
        self.parse_fct(namespace, values, Variables)


class ParseAugmentation(AbstractParser):
    def __call__(self, parser, namespace, values, option_string=None):
        self.parse_fct(namespace, values, Augmentation)


class ParseWeight(AbstractParser):
    def __call__(self, parser, namespace, values, option_string=None):
        if "calculate" in values:
            self.parse_fct(namespace, values, str)
        else:
            self.parse_fct(namespace, values, float)


class ParseFloat(AbstractParser):
    def __call__(self, parser, namespace, values, option_string=None):
        self.parse_fct(namespace, values, float)


def generate_argparse(name, config_file="params.yaml", default_config="default_params.yaml",
                      ignore_cmd_args=False, alternative_args=[], preloaded_argparse=None):
    CONFIG_AVAILABLE, parser = define_argparse(config_file, default_config, name, preloaded_argparse)

    if not CONFIG_AVAILABLE:
        import yaml
        with open(config_file) as f:
            data = yaml.safe_load(f)
            str_data = []
            for k, v in data.items():

                if v == False:  # remove flag
                    continue
                elif v == True:
                    str_data.append("--" + k)  # set flag
                    continue

                str_data.append("--" + k)
                str_data.append(str(v))
            args = parser.parse_args(str_data)
    elif ignore_cmd_args:  # inside a unittest we do not read the cmd args
        args, _ = parser.parse_known_args(alternative_args)
    else:
        args = parser.parse_args()

    args.debug_tool_name = "_".join(name.lower().split(" "))
    return args


def define_argparse(config_file="params.yaml", default_config="default_params.yaml", name="no_name",
                    preloaded_argparse=None):
    CONFIG_AVAILABLE, parser = setup_argparse(config_file, default_config, name, preloaded_argparse)
    # ------ Individual Run Params ---------
    run_group = parser.add_argument_group("Invidiual Run Parameters")
    add_dataset(run_group)
    add_root(run_group)
    add_dvc(run_group)
    run_group.add_argument("--gpu", action="store_true", help="Enable GPU usage")  # TODO: merge with self.cuda
    run_group.add_argument("--gpu_id", type=int, help="Provide GPU ID for CUDA_VISIBLE_DEVICES.", default=-1)
    run_group.add_argument("--nondeterministic", action="store_true",
                           help="Do not set a deterministic seed; ATTENTION: bad reproducibility")
    add_ignore_missing(run_group)
    add_loading_workers(run_group)
    run_group.add_argument("--show_params", action="store_true",
                           help="Flag to show the chosen parameters of the script.")
    # Evaluation / Prediction
    file_group = parser.add_argument_group("File Handling")
    add_max_n(file_group)
    add_plot(file_group)
    add_explicit(file_group)
    # Logging
    log_group = parser.add_argument_group("Logging")
    add_level(log_group)
    add_loggers(log_group)
    add_tags(log_group)
    log_group.add_argument("--resume_log", action="store_true", help="Resume logging jobs if available.\n"
                                                                     "Wandb: https://docs.wandb.ai/ref/python/init resume='auto'.\n"
                                                                     "ClearML: https://clear.ml/docs/latest/docs/references/sdk/task/#taskinit continue_last_task=True.")
    log_group.add_argument("--run_name", type=str, default=None,
                           help="Custom name for this run. All outputs (checkpoints, debug images, "
                                "TensorBoard, cmd log) are stored under this name instead of an "
                                "auto-generated timestamp. E.g. --run_name ep11_embed_v1. "
                                "If not set, a timestamp is used automatically.")
    # ------ Experiment Params ---------
    file_group.add_argument("--log_dir", type=str, required=True,
                            help="Name of the experiment e.g. tus_po_8p_dn19_up. Should also be the branch name and the folder name.")
    # Dataset
    dataset_group = parser.add_argument_group("Dataset")
    add_img_height(dataset_group)
    add_split(dataset_group)
    add_subsample(dataset_group)
    # Model
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--model", type=Network, choices=list(Network), default=Network.YOLO_CLASS,
                             help="Provide network model name")
    model_group.add_argument("--backbone", type=str, default="convnext",
                             choices=["convnext", "darknet", "timm"],
                             help="Backbone family for YolinoNet. "
                                  "'convnext' = ConvNeXt-Tiny + FPN (default). "
                                  "'darknet'  = Darknet-19 (cfg via --darknet_cfg, optional dilation) + FPN. "
                                  "'timm' = timm backbone (e.g. resnet50_dilated/hrnet_w32) + FPN. "
                                  "Both expose `backbone.body.*` so layer-wise LR groups "
                                  "(`--lr_backbone/--lr_fpn/...`) and freeze logic behave identically.")
    model_group.add_argument("--timm_model_name", type=str, default="resnet50_dilated",
                             choices=["resnet50_dilated", "resnet50", "hrnet_w32"],
                             help="Only used when --backbone=timm. "
                                  "resnet50_dilated=ResNet-50 output_stride=16 (atrous stage-4), "
                                  "resnet50=ResNet-50 output_stride=32 (no dilated stage-4; C4 from final stage), "
                                  "hrnet_w32=standard HRNet-W32.")
    model_group.add_argument("--darknet_cfg", type=str, default="model/cfg/darknet19_448_d2.cfg",
                             help="Path to the darknet config. Will be appended to the root path.")
    model_group.add_argument("--darknet_weights", type=str, default="model/cfg/darknet19_448.weights",
                             help="Path to the darknet weights. Will be appended to the root path. Use e.g. model/cfg/darknet19_448.weights.")
    model_group.add_argument("--linerep", type=LINE, choices=list(LINE), required=True,
                             help="Provide the line representation.")
    add_num_predictors(model_group)
    model_group.add_argument("--embed_dim", type=int, default=8,
                             help="Embedding head output channels per predictor (D).")
    model_group.add_argument(
        "--e2e_differentiable_postproc", action=ParseBool, default=False,
        help="If true, attach differentiable E2E head (soft-argmax tokens, optional feature-aware "
             "transformer affinity, Bézier samples). Original scipy/BFS postproc in line_fit is unchanged.")
    model_group.add_argument(
        "--e2e_mode", type=str, default="gnn",
        choices=["gnn"],
        help="E2E head when --e2e_differentiable_postproc=true: "
             "YolinoGnnSegmentGraphHead (segment graph → GAT → edge probs).",
    )
    # ----- GNN E2E head options -----
    model_group.add_argument("--gnn_max_nodes", type=int, default=256,
                             help="GNN: maximum number of segment nodes per image (top-K by conf among all "
                                  "cell*predictor segments).")
    model_group.add_argument("--gnn_node_conf_thresh", type=float, default=0.1,
                             help="GNN: nodes whose YOLinO conf is below this are kept structurally (so we always "
                                  "have gnn_max_nodes slots) but flagged invalid for message-passing and loss.")
    model_group.add_argument(
        "--gnn_node_use_visual_feat", action=ParseBool, default=True,
        help="GNN: if false, node tokens are built from segment geometry PE only (no FPN grid_sample feature).",
    )
    model_group.add_argument(
        "--gnn_use_matched_gt_from_geom", action=ParseBool, default=False,
        help="GNN criterion: reuse geom-head cell-matcher GT assignments per predictor slot as node supervision "
             "instead of nearest-polyline-vertex assignment.",
    )
    model_group.add_argument(
        "--gnn_supervise_matched_nodes_only", action=ParseBool, default=True,
        help="GNN edge loss: if true, supervise only edges where BOTH endpoints have a matched GT instance "
             "(seg_inst>=0). If false (legacy), also supervise edges with one unmatched endpoint as negatives.",
    )
    model_group.add_argument(
        "--gnn_soft_nms", action=ParseBool, default=False,
        help="GNN: apply segment soft-NMS (lateral d_perp only) on conf before top-k node selection.",
    )
    model_group.add_argument("--gnn_soft_nms_mid_sigma_px", type=float, default=16.0,
                             help="Soft-NMS: sigma_lat for exp(-d_lateral^2/(2*sigma^2)); along-track ignored.")
    model_group.add_argument("--gnn_soft_nms_min_dir_dot", type=float, default=0.96,
                             help="Soft-NMS: unused (lateral-only); kept for config compatibility.")
    model_group.add_argument(
        "--gnn_soft_nms_decay", type=str, default="linear", choices=["linear", "gaussian"],
        help="Soft-NMS: 'linear' => score *= (1-overlap); 'gaussian' => score *= exp(-overlap^2).",
    )
    model_group.add_argument("--gnn_soft_nms_score_floor", type=float, default=0.001,
                             help="Soft-NMS: stop propagating decay below this score.")
    model_group.add_argument("--gnn_soft_nms_prefilter_conf", type=float, default=0.05,
                             help="Soft-NMS: only run on segments with conf >= this.")
    model_group.add_argument("--gnn_soft_nms_max_segments", type=int, default=1024,
                             help="Soft-NMS: max segments per image (top by conf before NMS).")
    # ----- exp58: conservative segment merge (pre-GNN, geometric fusion) -----
    model_group.add_argument(
        "--gnn_segment_merge", action=ParseBool, default=False,
        help="GNN: physically MERGE near-overlapping/touching parallel segments into a single "
             "representative before node selection (exp58). Unlike --gnn_soft_nms (conf decay only), "
             "this clusters segments and replaces each cluster with one segment fitted to the cluster.",
    )
    model_group.add_argument("--gnn_segment_merge_lat_px", type=float, default=6.0,
                             help="Segment merge: max symmetric perpendicular distance between segment "
                                  "lines (pixels). Conservative default ~¼ of a 32-px cell.")
    model_group.add_argument("--gnn_segment_merge_dir_dot_min", type=float, default=0.98,
                             help="Segment merge: minimum |dn_i·dn_j| (cosine) to be considered parallel "
                                  "(0.98 ≈ 11° tolerance).")
    model_group.add_argument("--gnn_segment_merge_end_gap_px", type=float, default=8.0,
                             help="Segment merge: max endpoint-to-endpoint distance to bridge non-overlapping "
                                  "co-linear segments (along-track stitch).")
    model_group.add_argument("--gnn_segment_merge_iters", type=int, default=3,
                             help="Segment merge: max fixed-point iterations.")
    model_group.add_argument("--gnn_segment_merge_prefilter_conf", type=float, default=0.05,
                             help="Segment merge: only segments with conf >= this enter the merge pool.")
    model_group.add_argument(
        "--gnn_use_hard_geom_gate", action=ParseBool, default=True,
        help="GNN: if true, apply hard lateral/along/end/dir geometry masks to neigh_valid (exp43). "
             "If false, keep all valid node pairs and use soft geom prior instead (exp55).",
    )
    model_group.add_argument(
        "--gnn_soft_geom_gate", action=ParseBool, default=False,
        help="GNN: add log(edge_prior) bias from lateral×dir soft gate (exp55).",
    )
    model_group.add_argument("--gnn_soft_geom_sigma_lat_px", type=float, default=32.0,
                             help="Soft geom prior: lateral decay sigma (pixels).")
    model_group.add_argument("--gnn_soft_geom_dir_floor", type=float, default=0.5,
                             help="Soft geom prior: ReLU floor on |dir_i·dir_j|.")
    model_group.add_argument("--gnn_soft_geom_prior_eps", type=float, default=1e-6,
                             help="Soft geom prior: clamp/log stability epsilon.")
    model_group.add_argument(
        "--gnn_adjacency_mode", type=str, default="global",
        choices=["global", "knn", "directional2", "directional2_ctx", "directional2_global"],
        help="GNN: 'global' = all N×N directed edges (subject to masks); 'directional2' = at most two "
             "collinear candidates per node (forward/back along segment direction, K=2); "
             "'directional2_ctx' = GAT+edge+loss on cat(on-line,context) K; CC/viz/degree/RW on on-line only; "
             "'directional2_global' = GAT on global N×N; edge+loss on Euclidean top-gnn_knn_k "
             "(optional gnn_knn_min_dir_dot direction filter); "
             "'knn' = legacy top-k neighbours by Euclidean midpoint distance only.",
    )
    model_group.add_argument("--gnn_knn_k", type=int, default=16,
                             help="GNN: when gnn_adjacency_mode=knn, number of nearest neighbours per node; "
                             "when directional2_global, connection/edge candidate count (Euclidean).")
    model_group.add_argument(
        "--gnn_knn_min_dir_dot", type=float, default=0.0,
        help="GNN: when directional2_global (or kNN with dir filter), require |dir_i·dir_j| >= this "
             "before counting a midpoint neighbour (0=off). 0.8 ≈ ~20%% direction mismatch allowed.",
    )
    model_group.add_argument(
        "--gnn_directional_min_sep_px", type=float, default=8.0,
        help="GNN: when gnn_adjacency_mode=directional2, minimum signed along-track separation (px) "
             "to count as forward/backward candidate.",
    )
    model_group.add_argument(
        "--gnn_directional_k", type=int, default=2,
        help="GNN: when gnn_adjacency_mode=directional2, total directed neighbour slots per node "
             "(ceil(K/2) forward + floor(K/2) backward along segment direction, after hard geom gate). "
             "Ignored when gnn_directional_include_all=true.",
    )
    model_group.add_argument(
        "--gnn_directional_include_all", action=ParseBool, default=False,
        help="GNN: include ALL on-line hard-gate neighbours per node (variable K padded to batch max) "
             "instead of top gnn_directional_k directional slots.",
    )
    model_group.add_argument(
        "--gnn_context_k", type=int, default=4,
        help="GNN: when gnn_adjacency_mode=directional2_ctx, parallel-wire context slots per node "
             "(GAT only; excluded from edge MLP and BCE).",
    )
    model_group.add_argument(
        "--gnn_context_lat_min_px", type=float, default=12.0,
        help="GNN context slots: minimum perpendicular offset (px) from source line.",
    )
    model_group.add_argument(
        "--gnn_context_lat_max_px", type=float, default=40.0,
        help="GNN context slots: maximum perpendicular offset (px) from source line.",
    )
    model_group.add_argument(
        "--gnn_context_max_along_px", type=float, default=200.0,
        help="GNN context slots: max |along-track| separation (px); 0 disables.",
    )
    model_group.add_argument(
        "--gnn_context_min_dir_dot", type=float, default=0.85,
        help="GNN context slots: minimum |dir_i·dir_j| for parallel-wire candidates.",
    )
    model_group.add_argument("--gnn_max_lateral_px", type=float, default=48.0,
                             help="GNN (global adjacency): hard-mask edges where perpendicular distance from "
                                  "destination midpoint to source segment line exceeds this (pixels). "
                                  "0 disables lateral masking (not recommended for global mode).")
    model_group.add_argument(
        "--gnn_max_lateral_sym", action=ParseBool, default=False,
        help="GNN: if true, also require source midpoint within gnn_max_lateral_px of dest segment line "
             "(symmetric same-line test).",
    )
    model_group.add_argument(
        "--gnn_lateral_on_overlap_only", action=ParseBool, default=False,
        help="GNN hard gate: apply lateral distance threshold only when |along-track s| is within "
             "gnn_lateral_overlap_window_px (overlap zone). Outside overlap zone, lateral gate is skipped.",
    )
    model_group.add_argument(
        "--gnn_lateral_overlap_window_px", type=float, default=24.0,
        help="GNN hard gate with gnn_lateral_on_overlap_only=true: overlap-zone half-width on |s| (px).",
    )
    model_group.add_argument(
        "--gnn_max_along_px", type=float, default=0.0,
        help="GNN: hard-mask edges when |along-track s| from source mid to dest mid exceeds this (px). "
             "0 disables. Reduces long-range links on the same infinite line.",
    )
    model_group.add_argument(
        "--gnn_max_end_gap_px", type=float, default=0.0,
        help="GNN: hard-mask edges when min endpoint distance (four pairings) exceeds this (px). "
             "0 disables. Enforces stitchable segment ends.",
    )
    model_group.add_argument(
        "--gnn_min_dir_dot", type=float, default=0.0,
        help="GNN: hard-mask edges when |dir_i·dir_j| is below this (0 disables). e.g. 0.96 ≈ 16° max misalignment.",
    )
    model_group.add_argument("--gnn_edge_radius_px", type=float, default=0.0,
                             help="GNN: optional pixel cap on midpoint distance; masked edges are excluded. "
                                  "0 disables. Applies to both adjacency modes.")
    model_group.add_argument("--gnn_gat_layers", type=int, default=3,
                             help="GNN: number of stacked GAT message-passing layers.")
    model_group.add_argument(
        "--gnn_token_dim", type=int, default=None,
        help="GNN: hidden dim for segment node encoding (geom PE + optional visual fuse, GAT, edge MLP). "
             "Defaults to 256 when unset.",
    )
    model_group.add_argument("--gnn_heads", type=int, default=4,
                             help="GNN: number of attention heads per GAT layer (must divide gnn token dim).")
    model_group.add_argument("--gnn_dropout", type=float, default=0.1,
                             help="GNN: dropout on attention weights and FFN inside GAT layers.")
    model_group.add_argument(
        "--gnn_edge_feat_signed", action=ParseBool, default=False,
        help="GNN edge MLP: use signed dir_dot/along + segment lengths (11-D) instead of "
             "180°-safe |dir_dot|/|along| (9-D legacy).",
    )
    model_group.add_argument(
        "--gnn_gat_geom_bias", action=ParseBool, default=False,
        help="GNN GAT: add fixed geometric bias to attention logits "
             "(exp(-|along|/tau_a) - w_lat*lateral/tau_l).",
    )
    model_group.add_argument(
        "--gnn_gat_geom_bias_w_along", type=float, default=1.0,
        help="GNN GAT geom bias: weight on exp(-|along|/tau_along) term.",
    )
    model_group.add_argument(
        "--gnn_gat_geom_bias_w_lat", type=float, default=0.5,
        help="GNN GAT geom bias: weight on lateral/tau_lat penalty term.",
    )
    model_group.add_argument(
        "--gnn_gat_geom_bias_tau_along", type=float, default=40.0,
        help="GNN GAT geom bias: along-track decay scale in pixels.",
    )
    model_group.add_argument(
        "--gnn_gat_geom_bias_tau_lat", type=float, default=20.0,
        help="GNN GAT geom bias: lateral offset scale in pixels.",
    )
    model_group.add_argument("--gnn_loss_weight", type=float, default=0.0,
                             help="GNN: weight on the connectivity BCE term (0 disables supervised GNN loss).")
    model_group.add_argument("--gnn_warmup_epochs", type=int, default=0,
                             help="GNN: linear ramp epochs for gnn_loss_weight after gnn_warmup_start_epoch.")
    model_group.add_argument("--gnn_warmup_start_epoch", type=int, default=0,
                             help="GNN: first epoch index (0-based) at which the GNN loss ramp may begin.")
    model_group.add_argument("--gnn_pos_weight", type=float, default=1.0,
                             help="GNN: pos_weight for BCEWithLogits on the positive class (same-instance edges). "
                                  "Also multiplies the positive term when --gnn_edge_loss_type=focal.")
    model_group.add_argument("--gnn_neg_weight", type=float, default=1.0,
                             help="GNN: extra multiplicative weight for negative edges in BCE/focal. "
                                  "Use >1 to penalize false connections more strongly.")
    model_group.add_argument(
        "--gnn_cross_instance_weight", type=float, default=10.0,
        help="GNN cross_ignore loss: BCE weight on cross-instance edges (label 0). "
             "Same-instance non-adjacent pairs are ignored unless --gnn_same_inst_ignore_as_pos.",
    )
    model_group.add_argument(
        "--gnn_same_inst_ignore_as_pos", action=ParseBool, default=False,
        help="GNN cross_ignore + polyline_adjacent: train same-instance non-adjacent edges as "
             "positives (label 1) with --gnn_pos_weight_remote instead of ignoring them.",
    )
    model_group.add_argument(
        "--gnn_pos_weight_remote", type=float, default=1.0,
        help="GNN cross_ignore: BCE weight for same-instance non-adjacent positives when "
             "--gnn_same_inst_ignore_as_pos is true.",
    )
    model_group.add_argument("--gnn_neg_per_pos", type=int, default=8,
                             help="GNN: max ratio of sampled negative edges to positives per batch (0 disables "
                                  "subsampling — all valid negatives contribute).")
    model_group.add_argument(
        "--gnn_neg_min_kept", type=int, default=8,
        help="GNN: when subsampling negatives, keep at least this many negatives per batch item "
             "(floor on neg_per_pos * n_pos). Set 2–3 for lighter negative pressure.",
    )
    model_group.add_argument(
        "--gnn_edge_loss_type", type=str, default="bce",
        choices=["bce", "focal", "cross_ignore"],
        help="GNN edge loss: 'bce'/'focal' = all non-positives as negatives; "
             "'cross_ignore' = pos=polyline_adjacent, ignore=same-inst non-adj, "
             "cross-inst negatives with gnn_cross_instance_weight.",
    )
    model_group.add_argument(
        "--gnn_focal_alpha", type=float, default=0.75,
        help="GNN focal: α on positive edges (1-α on negatives). Use >0.5 when positives are rare.",
    )
    model_group.add_argument(
        "--gnn_focal_gamma", type=float, default=2.0,
        help="GNN focal: γ focusing parameter (down-weights easy negatives).",
    )
    model_group.add_argument("--gnn_node_assign_radius_px", type=float, default=24.0,
                             help="GNN criterion: max pixel distance from a node midpoint to a GT polyline vertex "
                                  "for the node to be labelled as foreground (-1 otherwise).")
    model_group.add_argument(
        "--gnn_edge_supervision", type=str, default="matched_instance",
        choices=["matched_instance", "polyline_adjacent"],
        help="GNN edge labels: matched_instance = all same-instance candidate pairs; "
             "polyline_adjacent = same instance AND endpoint stitch within gnn_polyline_adjacent_end_px.",
    )
    model_group.add_argument(
        "--gnn_polyline_adjacent_end_px", type=float, default=12.0,
        help="GNN polyline_adjacent: max endpoint distance (px) for a positive edge.",
    )
    model_group.add_argument(
        "--gnn_eval_edge_threshs", type=str, default="0.35,0.5",
        help="GNN val: comma-separated sigmoid thresholds for edge precision/recall/F1 (logged as edge_f1_*).",
    )
    model_group.add_argument(
        "--gnn_use_strict_next_hop", action=ParseBool, default=False,
        help="GNN: if true, BCE positives only for GT-polyline vertex-adjacent segment pairs (|Δvidx|==1); "
             "otherwise keep legacy 'same instance' positives.",
    )
    model_group.add_argument(
        "--gnn_next_hop_allow_closed", action=ParseBool, default=False,
        help="GNN strict next-hop: also treat first/last GT vertex as adjacent (closed polylines).",
    )
    model_group.add_argument(
        "--gnn_rw_steps", type=int, default=6,
        help="GNN RW topology / soft_rw Chamfer: number of random-walk steps (Tk = T^steps).",
    )
    model_group.add_argument(
        "--gnn_rw_pos_weight", type=float, default=20.0,
        help="GNN RW topology loss: positive-class weight for same-instance pairs.",
    )
    model_group.add_argument(
        "--gnn_rw_topology_weight", type=float, default=0.0,
        help="GNN: weight on random-walk topology BCE loss (0 disables).",
    )
    model_group.add_argument(
        "--gnn_rw_start_epoch", type=int, default=0,
        help="GNN RW topology: first epoch index (0-based) to enable RW loss weight.",
    )
    model_group.add_argument(
        "--gnn_cc_edge_thresh", type=float, default=0.3,
        help="GNN val instance IoU: CC edge threshold on sigmoid(edge_logit).",
    )
    model_group.add_argument(
        "--gnn_degree_loss_weight", type=float, default=0.0,
        help="GNN: weight on soft out-degree / in-degree penalty (0 disables). Uses sigmoid(edge_logit).",
    )
    model_group.add_argument("--gnn_degree_row_cap", type=float, default=1.0,
                             help="GNN degree loss: ReLU margin for sum_j sigmoid(logit_ij) per source row.")
    model_group.add_argument("--gnn_degree_col_cap", type=float, default=1.0,
                             help="GNN degree loss: ReLU margin for sum_i sigmoid(logit_ij) per destination column.")
    model_group.add_argument(
        "--gnn_chamfer_weight", type=float, default=0.0,
        help="GNN: weight on differentiable soft-assembly Chamfer term (0 disables).",
    )
    model_group.add_argument(
        "--gnn_assembly_max_nodes_per_inst", type=int, default=32,
        help="GNN Chamfer: max nodes per GT instance considered in assembly (cap memory).",
    )
    model_group.add_argument(
        "--gnn_assembly_max_instances", type=int, default=8,
        help="GNN Chamfer: max GT instances per batch item to supervise (cap memory).",
    )
    model_group.add_argument(
        "--gnn_chamfer_warmup_epochs", type=int, default=0,
        help="GNN Chamfer: linear ramp epochs for gnn_chamfer_weight after gnn_chamfer_warmup_start_epoch.",
    )
    model_group.add_argument(
        "--gnn_chamfer_warmup_start_epoch", type=int, default=0,
        help="GNN Chamfer: first epoch (0-based) when Chamfer weight ramp may begin.",
    )
    model_group.add_argument(
        "--gnn_assembly_tier", type=str, default="soft_chain",
        choices=["soft_chain", "soft_rw", "none"],
        help="GNN assembly: soft_chain | soft_rw (Tk@mids) | none.",
    )
    model_group.add_argument(
        "--gnn_tb_viz_conf_thresh", type=float, default=None,
        help="GNN val TensorBoard overlay: minimum node confidence to draw segments (default: use --confidence). "
             "Set lower than gnn_node_conf_thresh to visualize more nodes during training.",
    )
    model_group.add_argument("--gnn_overlay_edge_thresh", type=float, default=0.5,
                             help="GNN val overlay: draw an edge in TB only if sigmoid(edge_logit) >= this value.")
    model_group.add_argument(
        "--gnn_tb_show_lonely_segments", action=ParseBool, default=True,
        help="GNN val TensorBoard overlay: draw valid nodes that have no edge above "
             "--gnn_overlay_edge_thresh in neutral gray (unmerged segments).",
    )
    model_group.add_argument(
        "--gnn_tb_segment_thickness", type=int, default=3,
        help="GNN val TB overlay: line thickness for YOLinO segment endpoints (end_a–end_b) per node.",
    )
    model_group.add_argument(
        "--gnn_tb_connector_thickness", type=int, default=2,
        help="GNN val TB overlay: line thickness for predicted mid–mid connector edges.",
    )
    # GT canonicalization for the 5-pt head (also useful for other heads).
    model_group.add_argument(
        "--e2e_gt_canonicalize", action=ParseBool, default=True,
        help="TTPLA: re-order each GT polyline after augmentation so vertices go "
             "left→right (or top→bottom when nearly vertical). Source of truth for "
             "the 5-pt head's ordering. Disable to fall back to legacy load-time flip only.",
    )
    model_group.add_argument(
        "--e2e_gt_vertical_angle_deg", type=float, default=80.0,
        help="TTPLA canonicalize: angle (degrees from horizontal) at or above which a "
             "polyline is treated as vertical (use y-sort instead of x-sort).",
    )
    model_group.add_argument(
        "--e2e_train_with_gt_polylines", action=ParseBool, default=False,
        help="TTPLA: return fixed-size GT poly packs for GNN edge supervision.")
    model_group.add_argument("--e2e_gt_resample_t", type=int, default=32,
                             help="Uniform arc-length resample count for GT polylines.")
    model_group.add_argument("--e2e_gt_max_instances", type=int, default=64,
                             help="Max GT instances per image (padding).")
    model_group.add_argument("--e2e_gt_max_points", type=int, default=256,
                             help="Max vertices per GT instance (padding).")
    model_group.add_argument(
        "--feature_refine", type=str, default="sa_embed_only",
        choices=["none", "sa_embed_only", "sa_shared", "cbam_shared"],
        help="Trunk feature refinement before heads: none | sa_embed_only (default, SA only on embed path) | "
             "sa_shared (SA on shared map → both heads) | cbam_shared (CBAM on shared map → both heads).")
    model_group.add_argument("--cbam_reduction_ratio", type=int, default=16,
                             help="CBAM channel MLP reduction ratio when feature_refine=cbam_shared.")
    # ----- ConvNeXt + FPN backbone options -----
    model_group.add_argument("--fpn_out_channels", type=int, default=256,
                             help="FPN lateral/smooth conv output channels. Heads consume this directly. "
                                  "Default 256.")
    model_group.add_argument("--head_level", type=str, default="P3",
                             choices=["P2", "P3", "P4", "P5"],
                             help="FPN level fed to geometry/embedding heads. "
                                  "P2=stride 8, P3=stride 16, P4=stride 32, P5=stride 32 (when provided). "
                                  "MUST match --scale (P2:8, P3:16, P4/P5:32). Ignored when --std=true "
                                  "(always uses P3 + PixelUnshuffle -> stride 32).")
    model_group.add_argument(
        "--std", action=ParseBool, default=False,
        help="If true, use Space-to-Depth head: feats['P3'] -> PixelUnshuffle(2) -> [B,4*C,H/32,W/32], "
             "then per-predictor modulated DCNv2 for geometry (+ per-predictor 1x1 embed). "
             "Requires --scale 32. Replaces shared 1x1 self.yolo.",
    )
    model_group.add_argument(
        "--std_skip_geom_head", action=ParseBool, default=False,
        help="With --std: skip per-predictor geom DCNv2 (zeros geom for grid loss stub). "
             "Use when geom loss weight is 0.",
    )
    model_group.add_argument(
        "--std_feat_dcn_before_e2e", action=ParseBool, default=False,
        help="With --std: apply shared StdFeatRefineDcn on the STD map before E2E memory.",
    )
    model_group.add_argument("--std_feat_dcn_kernel", type=int, default=3,
                             help="Kernel size for --std_feat_dcn_before_e2e.")
    model_group.add_argument(
        "--std_feat_dcn_channels", type=int, default=0,
        help="Output channels for feat DCN (0 = same as STD channels, 1024 for default fpn_out=256).",
    )
    model_group.add_argument("--backbone_pretrained", action=ParseBool, default=True,
                             help="Load ImageNet-pretrained ConvNeXt-Tiny weights (True/False).")
    model_group.add_argument("--fpn_upsample_mode", type=str, default="nearest",
                             choices=["nearest", "bilinear"],
                             help="Interpolation mode used inside the FPN top-down path.")
    model_group.add_argument(
        "--fpn_norm", type=str, default="groupnorm",
        choices=["groupnorm", "batchnorm", "syncbatchnorm", "none"],
        help="Normalization after FPN 3x3 smooth and bottom-up smooth convs. "
             "Default groupnorm (legacy). exp50: syncbatchnorm for DDP (classic FPN-style BN stats). "
             "Use batchnorm for single-GPU; none disables norm.",
    )
    model_group.add_argument("--use_fpn", action=ParseBool, default=True,
                             help="If true, use top-down FPN fusion. "
                                  "If false, skip top-down fusion and use per-level projected backbone features "
                                  "(same P2/P3/P4 strides, no cross-level fusion).")
    model_group.add_argument("--use_bottom_up", action=ParseBool, default=False,
                             help="If true, rebuild P4 by bottom-up fusion (downsample exported P2, add pre-smooth "
                                  "m3/m4). Recommended with use_fpn=True for FPN+PANet-like semantics; "
                                  "use_fpn=False gives lateral-only m3/m4 (different ablation).")
    model_group.add_argument("--timm_force_stride32_head", action=ParseBool, default=False,
                             help="Only for --backbone=timm. If true, apply an extra stride-2 down-projection "
                                  "(3x3 conv + BN + GELU) on P4 and export it as P5 so head_level=P5 can "
                                  "use stride-32 semantics while keeping high-res backbone features.")
    model_group.add_argument("--backbone_freeze_epochs", type=int, default=0,
                             help="Freeze the ConvNeXt body for the first N epochs (only FPN+heads train). "
                                  "0 disables freezing. Useful to stabilize new heads on a pretrained backbone.")
    model_group.add_argument("--activations", type=str, required=True,
                             help="Provide the activation for each block in the training variables. Choose from %s" % [
                                 a.value for a in ACTIVATION],
                             action=ParseActivation)
    model_group.add_argument("--training_variables", type=str, required=True,
                             help="Provide variables to be predicted by the network. Remaining will only be used for visualization, but not learned"
                                  "Choose from %s" % [a.value for a in Variables], action=ParseVariables)
    add_scale(model_group)
    # Augmentation
    augment_group = parser.add_argument_group("Augmentation")
    augment_group.add_argument("--crop_range", type=float, required=True,
                               help="Range to sample crop portion from for the augmentation during training")
    augment_group.add_argument("--rotation_range", type=float, required=True,
                               help="Range of radians to sample the rotation from for the augmentation during training")
    augment_group.add_argument("--augment", type=str, action=ParseAugmentation, required=True,
                               help="Provide list of all augmentations to apply. The methods will be applied in order."
                                    "Choose from %s" % [a.value for a in Augmentation])
    augment_group.add_argument("--noise_std", type=float, default=0.01,
                               help="Std for AddGaussianNoise augmentation.")
    augment_group.add_argument("--noise_p", type=float, default=0.1,
                               help="Probability for AddGaussianNoise augmentation.")
    # Training
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--retrain", action="store_true", help="Does not load weights even if available")
    add_batch_size(train_group)
    train_group.add_argument("--decay_rate", type=float, default=0.0001,
                             help="Decay rate for Adam Optimizer")
    train_group.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")
    train_group.add_argument("--epoch", type=int, required=True, help="Maximum epoch")
    train_group.add_argument("--checkpoint_iteration", type=int, default=10,
                             help="Save separate .pth file after several epochs")
    add_keep(train_group)
    train_group.add_argument("--skip_best_model_eval", action=ParseBool, default=False,
                             help="If true, skip end-of-training evaluation that loads best_model.pth and runs inference "
                                  "over the full TRAINING loader (two forward passes per batch + plots). "
                                  "This is not the same as periodic validation on val_loader; it is often much slower.")
    train_group.add_argument("--eval_iteration", type=int, default=3,
                             help="Run model on full validation set after several epochs")
    train_group.add_argument("--full_eval", action="store_true", help="Apply UV evaluation also during training")
    train_group.add_argument("--optimizer", type=Optimizer, choices=list(Optimizer), default=Optimizer.ADAM,
                             help="Speciy Optimizer")
    train_group.add_argument("--scheduler", type=Scheduler, choices=list(Scheduler), default=Scheduler.NONE,
                             help="Specify Scheduler for the learning rate")
    train_group.add_argument("--warmup_epochs", type=int, default=0,
                             help="Warmup length in epochs for warmup_cosine scheduler.")
    train_group.add_argument("--min_lr_ratio", type=float, default=0.05,
                             help="Final LR ratio for cosine schedules. "
                                  "effective_min_lr = base_lr * min_lr_ratio.")
    train_group.add_argument("--scheduler_step_per_batch", action=ParseBool, default=True,
                             help="If true, step scheduler every optimizer step (recommended for warmup_cosine).")
    train_group.add_argument("--ddp_find_unused_parameters", action=ParseBool, default=False,
                             help="DDP option for models with conditionally unused params. "
                                  "Keep False for best performance unless needed.")
    train_group.add_argument(
        "--log_cuda_mem_after_epoch", action=ParseBool, default=False,
        help="Rank 0 only: after each training epoch, log torch.cuda memory_allocated and max_memory_allocated (MB) "
             "to TensorBoard (requires --gpu).",
    )
    train_group.add_argument(
        "--profile_dataloader", action=ParseBool, default=False,
        help="Rank 0 only: log per-epoch mean data_wait (iterator) and step_compute (forward+backward) "
             "to TensorBoard under dataloader/*.",
    )
    train_group.add_argument("--amp", action=ParseBool, default=True,
                             help="Enable mixed precision training (torch.cuda.amp) for stability/performance.")
    train_group.add_argument("--grad_clip_norm", type=float, default=0.0,
                             help="If >0, clip gradient norm to this value before optimizer step.")
    train_group.add_argument("--patience", type=int, default=5,
                             help="Number of validation epochs to wait for early convergence. "
                                  "If the last p epochs are worse than a previous one, we stop.")
    train_group.add_argument("--best_mean_loss", action=ParseBool, required=True,
                             help="If patience is set, activate this in order to regard the mean of the losses"
                                  " for convergence instead of the actual sum loss used for backpropagation.")
    train_group.add_argument("--earliest_stop", type=int, default=5,
                             help="Minimum number of epochs to run before appyl early stopping criteria.")
    train_group.add_argument("--learning_rate", required=True, type=float, help="The learning rate for the training")
    # Optional: layer-wise LR for decoupled heads (uses param_groups)
    train_group.add_argument("--lr_backbone", required=False, type=float, default=None,
                             help="Optional LR for the ConvNeXt backbone body (pretrained features). "
                                  "Recommend a small value (e.g. 1e-5..5e-5). "
                                  "If unset, falls back to --learning_rate.")
    train_group.add_argument("--lr_fpn", required=False, type=float, default=None,
                             help="Optional LR for FPN (lateral/smooth convs, non-pretrained). "
                                  "Recommend a moderate value (e.g. 1e-4..5e-4). "
                                  "If unset, falls back to --learning_rate.")
    train_group.add_argument("--lr_geom", required=False, type=float, default=None,
                             help="Optional LR for geometry head (yolo 1x1 conv). "
                                  "If unset, falls back to --learning_rate.")
    train_group.add_argument("--lr_embed", required=False, type=float, default=None,
                             help="Optional LR for embedding head (attention/cbam + embed_head). "
                                  "If unset, falls back to --learning_rate.")
    train_group.add_argument("--lr_e2e", required=False, type=float, default=None,
                             help="Optional LR for e2e_head.* (differentiable postproc). "
                                  "If unset, falls back to --lr_embed then --learning_rate.")
    # Loss
    loss_group = parser.add_argument_group("Loss")
    loss_group.add_argument("--loss", type=str, required=True,
                            help="Specify loss functions. For each entry in --training_variables a loss has to be specified"
                                 "Choose from %s" % [a.value for a in LOSS], action=ParseLoss)
    loss_group.add_argument("--weights", type=str, action=ParseWeight,
                            help="Specify the initial weights of the loss function. By default each weight is set to 1. "
                                 "If specified, for each entry in --training_variables a loss weight has to be given."
                                 "The weights will be adapted acording the loss' variance (Kendall et al.) or stay fixed "
                                 "depending on your --loss_weight_strategy. Fixed weights are recommended to be 4,1 from "
                                 "geom:conf. Pass `calculate` to let us calculate the weights.")
    loss_group.add_argument("--loss_weight_strategy", type=LossWeighting, required=True, choices=list(LossWeighting),
                            help="As weight for the multi-task loss terms we either learn log(sigma^2) or sigma of the "
                                 "data distribution for each task or set the provided values fixed."
                                 "The log version should be more stable compared to pure sigma according to Kendall et al. ")
    loss_group.add_argument("--conf_match_weight", type=str, default=[5, 1], action=ParseWeight, required=True,
                            help="Specify the weight for all matches on the conf loss and the unmatched. "
                                 "Helps with the imbalance between number of matched predictors and unmatched predictors."
                                 "Pass `calculate` to let us calculate the weights.")
    loss_group.add_argument("--conf_negative_weight", type=float, default=1.0,
                            help="Additional multiplier for unmatched confidence loss.")
    loss_group.add_argument("--focal_gamma", type=float, default=2.0,
                            help="Gamma parameter for focal confidence loss.")
    loss_group.add_argument("--focal_alpha", type=float, default=0.25,
                            help="Alpha parameter for focal confidence loss.")
    loss_group.add_argument("--qfl_beta", type=float, default=2.0,
                            help="Quality Focal Loss beta (GFL): |y - sigmoid(logit)|^beta scaling.")
    loss_group.add_argument("--qfl_iou_floor", type=float, default=0.0,
                            help="Minimum IoU soft target for matched confidence (QFL).")
    loss_group.add_argument("--match_by_conf_first", action=ParseBool,
                            help="Apply two-stage matching. 1. Only match predictions with confidence > --confidence. "
                                 "2. Match all remaining. This only affects the loss matching, not the evaluation.")
    loss_group.add_argument("--loss_hard_matching", action=ParseBool, default=True,
                            help="If true, use Hungarian bipartite matching for loss assignment. "
                                 "If false, skip matching and keep predictor order as-is for loss.")
    loss_group.add_argument("--use_conf_in_loss_matching", action=ParseBool,
                            help="Use the confidence variable in matching line segments for the loss. "
                                 "Only for --anchors=none.")
    loss_group.add_argument("--association_metric", type=Distance, choices=list(Distance), default=Distance.EUCLIDEAN,
                            help="Specify metric to associate two line segments geometrically. "
                                 "Will determine the responsibility in the loss function.", required=True)
    # Optional: discriminative embedding hyperparameters
    loss_group.add_argument("--embedding_delta_v", type=float, default=0.5,
                            help="Discriminative embedding loss: pull margin delta_v")
    loss_group.add_argument("--embedding_delta_d", type=float, default=3.0,
                            help="Discriminative embedding loss: push margin delta_d")
    loss_group.add_argument("--embedding_lambda_reg", type=float, default=0.0,
                            help="Discriminative loss: weight for L_reg (centroid L2 toward 0) on pure embeddings; 0=off.")
    loss_group.add_argument("--embedding_concat_geom_dims", type=int, default=0,
                            help="If >0, concat first N GT geometry channels (detached) to embedding vectors for pull/push only.")
    loss_group.add_argument("--embedding_loss_warmup_epochs", type=int, default=0,
                            help="Linearly scale discriminative embedding loss from 0→1 over this many epochs (0=off).")
    # Anchors
    anchor_group = parser.add_argument_group("Anchors")
    add_anchors(anchor_group)
    anchor_group.add_argument("--offset", action=ParseBool, required=True,
                              help="Predict offset values to an anchor instead of absolute values. "
                                   "--anchors has to be set to something other than 'none'.")
    anchor_group.add_argument("--anchor_vars", type=str, action=ParseAnchorVariables, required=True,
                              help="Specify which aspect of a line will be used to define an anchor. You can en-/disable anchors "
                                   "and define the distribution with --anchors. "
                                   "Choose from %s" % [a.value for a in AnchorVariables])
    # # NMS
    nms_group = parser.add_argument_group("Non-Maximum-Suppression")
    nms_group.add_argument("--nms", action="store_true", help="Apply non maximum suppression")
    nms_group.add_argument("--eps", type=float, required=True,  # 0.02,
                           help="NMS: Epsilon for DBSCAN")
    nms_group.add_argument("--min_samples", type=int, required=True,
                           help="NMS: Minimum number of samples required for main points in DBSCAN Cluster")
    nms_group.add_argument("--nxw", type=float, required=True,  # 0.05,
                           help="NMS: Weight for the normed x-widths in the DBSCAN Clustering")
    nms_group.add_argument("--confidence", type=float, required=True,  # 0.9,
                           help="Confidence threshold", )
    nms_group.add_argument("--lw", type=float, required=True,  # 0.05,
                           help="NMS: Weight for the length in the DBSCAN Clustering")
    nms_group.add_argument("--mpxw", type=float, required=True,  # 0.016,
                           help="NMS: Weight for the midpoint in the DBSCAN Clustering")
    # Eval
    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument("--metrics", type=Metric, nargs="*", default=list(Metric),
                            choices=list(Metric), help="Select metrics that should be calculated")
    eval_group.add_argument("--matching_gate", type=float, required=True,
                            help="Provide ratio of cell_sizes to be included in the matching of start-/endpoint in the "
                                 "evaluation. E.g. in order to match with the squared association_metric a specific number "
                                 "of cells 'no_cells' each with a 'cell_size' choose matching_gate=cell_size "
                                 "and (no_cells * cell_size)^2 will be the radius in px.")
    eval_group.add_argument("--explicit_model", type=str,  # default="log/checkpoints/model.pth",
                            help="Provide a path to an alternative model.pth file. By default we use "
                                 "log/checkpoints/model.path (training continued) or log/checkpoints/best_model.pth (prediction)"
                                 "in the --dvc folder.")
    eval_group.add_argument("--log_duplicate_metrics", action=ParseBool, default=False,
                            help="Log duplicate-adjusted matching metrics (suffix _dupl).")
    eval_group.add_argument("--log_strict_metrics", action=ParseBool, default=False,
                            help="Log strict matching metrics without confidence rematch/filter.")
    # Tusimple Benchmark / Connection
    postproc_group = parser.add_argument_group("Postprocessing")
    postproc_group.add_argument("--min_segments_for_polyline", type=int, required=True,
                                help="Minimum number of segments to build a valid polyline in the line fitting for tusimple")
    postproc_group.add_argument("--adjacency_threshold", required=True, type=float)
    return CONFIG_AVAILABLE, parser


def add_batch_size(parser):
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training")


def add_keep(parser):
    parser.add_argument("--keep", action="store_true",
                        help="Set true if you would like to keep the checkpoint in log/checkpoints/<ID>.pt")


def add_loading_workers(run_group):
    run_group.add_argument("--loading_workers", type=int, required=True,
                           help="How many cpu cores (?) / threads (?) will be used by torch dataloader")


def add_anchors(parser):
    parser.add_argument("--anchors", type=AnchorDistribution, choices=list(AnchorDistribution),
                        help="Anchors fix the GT to specific positions based on th anchors position (--anchors_vars). "
                             "Set to 'none' to allow the network to learn specific predictors on random positions. "
                             "ATTENTION: this heavily increases computation time, as each loss calculation needs a "
                             "matching. Other choices define the distribution of the anchors. '%s' loads the values "
                             "from the anchors file." % AnchorDistribution.KMEANS)


def add_scale(parser):
    parser.add_argument("-up", "--scale", type=int, choices=[32, 16, 8], required=True,
                        help="Provide the downsampling factor we should use to add upsampling layer. "
                             "With -up=32 the output will have 32x32 px per cell.")


def add_img_height(parser):
    parser.add_argument("--img_height", type=int, required=False,
                        help="Expected image height for the training, width will be calculated. "
                             "Input will be cropped/scaled to that if not valid.")


def setup_argparse(config_file="params.yaml", default_config="default_params.yaml", name="no_name",
                   preloaded_argparse=None):
    CONFIG_AVAILABLE = True
    print("Config file: file://%s" % os.path.abspath(config_file))
    print("Default config file: file://%s" % os.path.abspath(default_config))
    import argparse
    if preloaded_argparse is None or not isinstance(preloaded_argparse, argparse.ArgumentParser):
        try:
            # Test if configargparse is available (not avail on unittests in CI)
            import configargparse
            parser = configargparse.ArgumentParser(name, default_config_files=[default_config])
        except ModuleNotFoundError as ex:
            import socket
            host = socket.gethostname()
            user = pwd.getpwuid(os.getuid())[0]

            if "mrtbuild" != user:  # this should only happen in CI
                Log.error("%s with %s: No configargparse available" % (host, user))
                raise ex

            Log.warning("%s with %s: No configargparse available" % (host, user))
            CONFIG_AVAILABLE = False

            parser = argparse.ArgumentParser(name)
    else:
        parser = preloaded_argparse
    parser.add_argument("-c", "--config", required=False, is_config_file=True, default=config_file,
                        help="Config file path e.g. params.yaml")
    return CONFIG_AVAILABLE, parser


def add_plot(parser):
    parser.add_argument("--plot", action="store_true", help="Plot a lot of debug images")


def add_loggers(parser):
    parser.add_argument("--loggers", type=Logger, choices=list(Logger), nargs="*",
                        help="Specify logging services")


def add_tags(parser):
    parser.add_argument("--tags", type=str, nargs="*", help="Add tags for the experiments")


def add_ignore_missing(run_group):
    run_group.add_argument("--ignore_missing", action="store_true",
                           help="Do not abort on missing files. Use this for test environments with only subset of the dataset.")


def add_root(parser):
    parser.add_argument("--root", default="../yolino", help="Folder containing source code e.g. src")


def add_subsample(parser):
    parser.add_argument("-sdr", "--subsample_dataset_rhythm", type=int, default=-1,
                        help="Especially training with sequence based datasets might benefit from using only a subset of the images. The subsample rhythm will be used to select those.")


def add_max_n(parser):
    parser.add_argument('--max_n', type=int, default=-1, help='Runs for only max_n images')


def add_explicit(parser):
    parser.add_argument("--explicit", type=str, nargs="+",
                        help="Provide explicit filenames from the set that is chosen with --split, e.g. " \
                             "'driver_23_30frame/05161540_0603.MP4/05275.jpg' to process. This will ignore --max_n.")


def add_dvc(parser):
    parser.add_argument("--dvc", type=str, default=".",
                        help="DVC folder where the log, checkpoints and eval data will be stored.")


def add_level(parser):
    parser.add_argument("--level", type=Level, choices=list(Level), default=Level.INFO,
                        help="Choosing logging level. Only the chose and more severe levels are vizualized.")


def add_split(parser):
    parser.add_argument("--split", required=True, help="Provide split folder name (train, val, test)",
                        choices=["train", "val", "test"])


def add_dataset(parser):
    parser.add_argument("--dataset", type=Dataset, choices=list(Dataset), required=True,
                        help="Specify the dataset here")
    parser.add_argument(
        "--dataset_ttpla",
        type=str,
        default=None,
        help="TTPLA root with images/{train,val,test}/ and labels/... Uses DATASET_TTPLA env when set; "
             "otherwise this path is applied (experiment yaml friendly).",
    )


def add_num_predictors(parser):
    parser.add_argument("--num_predictors", type=int, required=True,
                        help="The number of allowed predictors for each cell")

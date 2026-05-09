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
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from yolino.eval.distances import linesegment_euclidean_distance
from yolino.eval.matcher_cell import CellMatcher
from yolino.model.activations import get_activations
from yolino.model.variable_structure import VariableStructure
from yolino.utils.enums import LOSS, Variables, LossWeighting, AnchorDistribution, ImageIdx
from yolino.utils.logger import Log


class AbstractLoss:
    def __init__(self, coords: VariableStructure, gpu, cuda, one_hot, variable, function, conf_match_weight, reduction,
                 activation_is_exp, loss_weight_strategy):
        self.loss_weight_strategy = loss_weight_strategy
        self.coords = coords
        self.variable = variable
        self.gpu = gpu
        self.cuda = cuda
        self.one_hot = one_hot
        self.function = function
        if self.gpu:
            self.function.cuda()
        self.conf_match_weight = conf_match_weight
        self.conf_negative_weight = 1.0
        self.reduction = reduction
        self.activation_is_exp = activation_is_exp

    # removes nan
    # Creates a torch view on the relevant variables only
    # returns a tensor of shape (?, num_variables)
    def __prepare_data__(self, preds, grid_tensor):
        breaks = self.__get_breaks_in_coords__()
        _, labels_for_loss, _ = torch.tensor_split(grid_tensor, torch.tensor(breaks), dim=1)
        if not self.one_hot:
            labels_for_loss = torch.argmax(labels_for_loss, dim=1)

        breaks = self.__get_breaks_in_coords__(only_training=True)
        _, preds_for_loss, _ = torch.tensor_split(preds, torch.tensor(breaks), dim=1)

        return preds_for_loss, labels_for_loss

    def __get_breaks_in_coords__(self, only_training=False):
        breaks = []
        position = 0
        included_idx = -1
        for idx, var in enumerate(self.coords):
            if self.coords[var] == 0 or (only_training and not var in self.coords.train_vars()):
                continue

            if var == self.variable:
                included_idx = idx

            position += self.coords[var]
            if included_idx == -1:
                if len(breaks) == 0:
                    breaks.append(position)
                else:
                    breaks[0] = position  # collect all excluded up to the included
            elif included_idx == idx:
                if len(breaks) == 0:
                    breaks.append(0)
                breaks.append(position)
                break
            elif included_idx + 1 == idx:
                breaks.append(position)
            else:
                if idx < len(self.coords) - 1:
                    breaks[-1] = position  # collect all excluded after the included
        return breaks

    @staticmethod
    def __get_flat__(preds, grid_tensor):
        grid_tensor = grid_tensor.flatten()
        preds = preds.flatten()
        return preds, grid_tensor

    def split_nans(self, preds, grid_tensor):
        """

        Args:
            preds (torch.tensor): with shape [batch, cells, preds, vars]

        """
        invalid_flags = torch.any(grid_tensor[:, self.coords.get_position_of_training_vars()].isnan(), dim=-1)
        valid_flags = ~invalid_flags

        return preds[valid_flags], grid_tensor[valid_flags], preds[invalid_flags], grid_tensor[invalid_flags]

    def replace_nans(self, grid_tensor, variable):
        indices = torch.any(grid_tensor[:, self.coords.get_position_of(variable)].isnan(), dim=1)
        grid_tensor[indices, self.coords.get_position_of(variable)] = 0
        return grid_tensor

    def __call__(self, preds, grid_tensor):
        if self.cuda not in str(grid_tensor.device):
            Log.debug("Moved labels from %s to %s" % (grid_tensor.device, self.cuda))
            grid_tensor.to(self.cuda)

        if self.cuda not in str(preds.device):
            Log.debug("Moved prediction from %s to %s" % (preds.device, self.cuda))
            preds.to(self.cuda)

        # for the confidence we also punish unmatched predictions which are assigned a nan GT line and so we
        # want to split them into matched and unmatched
        preds, grid_tensor, preds_unmatched, grid_tensor_unmatched = self.split_nans(preds, grid_tensor)

        if self.variable == Variables.CONF:
            # for the confidence all nans in the GT should have a confidence of 0
            grid_tensor_unmatched = self.replace_nans(grid_tensor_unmatched, variable=self.variable)
            preds_unmatched, grid_tensor_unmatched = self.__prepare_data__(preds_unmatched, grid_tensor_unmatched)

        preds, grid_tensor = self.__prepare_data__(preds, grid_tensor)

        if not self.variable == Variables.CONF and (len(preds) == 0 or len(grid_tensor) == 0):
            msg = "No valid data for loss %s in variable %s" % (type(self), self.variable)
            Log.debug(msg)
            raise ValueError(msg)

        return preds, grid_tensor, preds_unmatched, grid_tensor_unmatched

    def __str__(self):
        return "%s for %s %s" \
               % (str(self.__class__).replace("<", "").replace(">", "").replace("'", "").replace(
            "class yolino.model.loss.", ""),
                  self.variable, self.coords.get_position_of(self.variable))

    def __apply_function__(self, preds, grid_tensor, p_unmatched, gt_unmatched, tag="none", epoch=None):
        if len(preds) > 0 and len(grid_tensor) > 0:
            Log.debug("%s on e.g. %s vs gt=%s" % (str(self), preds[0], grid_tensor[0]))
        elif self.variable == Variables.CONF and len(p_unmatched) > 0 and len(gt_unmatched) > 0:
            Log.debug("Unmatched %s on e.g. %s vs gt=%s" % (str(self), p_unmatched[0], gt_unmatched[0]))

        matched_loss = torch.tensor(0, dtype=grid_tensor.dtype, device=grid_tensor.device)
        unmatched_loss = torch.tensor(0, dtype=grid_tensor.dtype, device=grid_tensor.device)

        # handle only not nan labels
        if not torch.all(grid_tensor.isnan()):
            matched_loss = self.function(preds, grid_tensor)

        normalizing_matched = len(preds) if self.reduction == "sum" else 1
        normalizing_unmatched = len(p_unmatched) if self.reduction == "sum" else 1

        mean_matched_loss = matched_loss / normalizing_matched
        mean_unmatched_loss = 0

        if self.variable == Variables.CONF:
            if not torch.all(gt_unmatched.isnan()):
                unmatched_loss = self.function(p_unmatched, gt_unmatched)

            mean_unmatched_loss = unmatched_loss / normalizing_unmatched
            Log.scalars(tag=tag, dict={"loss_conf_batch/match/mean": mean_matched_loss,
                                       "loss_conf_batch/unmatch/mean": mean_unmatched_loss},
                        epoch=epoch)

            add_weights, weight_factors = get_actual_weight(epoch, "conf/match",
                                                            weight_strategy=self.loss_weight_strategy,
                                                            weight=self.conf_match_weight[0],
                                                            activation_is_exponential=self.activation_is_exp)
            weighted_loss = weight_factors * matched_loss + add_weights
            add_weights, weight_factors = get_actual_weight(epoch, "conf/nomatch",
                                                            weight_strategy=self.loss_weight_strategy,
                                                            weight=self.conf_match_weight[1],
                                                            activation_is_exponential=self.activation_is_exp)
            weighted_loss += weight_factors * (self.conf_negative_weight * unmatched_loss) + add_weights
            mean_loss = (mean_matched_loss + mean_unmatched_loss) * 0.5
        else:
            weighted_loss = mean_matched_loss
            mean_loss = mean_matched_loss

        return weighted_loss, mean_loss


class CrossEntropyCellLoss(AbstractLoss):

    def __init__(self, gpu, coords: VariableStructure, cuda, variable: Variables, conf_match_weight, reduction,
                 activation_is_exp, loss_weight_strategy):
        super().__init__(coords=coords, gpu=gpu, cuda=cuda, one_hot=False, variable=variable,
                         function=nn.CrossEntropyLoss(reduction=reduction), conf_match_weight=conf_match_weight,
                         reduction=reduction, activation_is_exp=activation_is_exp,
                         loss_weight_strategy=loss_weight_strategy)

    def __call__(self, preds, grid_tensor, tag="none", epoch=None):
        preds, grid_tensor, p_unmatched, gt_unmatched = super().__call__(preds, grid_tensor)  # output shape (?, 10)
        Log.debug("%s on e.g. %s vs gt=%s; Shapes pred=%s vs gt=%s" % (str(self), preds[0], grid_tensor[0], preds.shape,
                                                                       grid_tensor.shape))

        try:
            return self.__apply_function__(preds, grid_tensor, p_unmatched, gt_unmatched, tag=tag, epoch=epoch)
        except Exception as ex:
            Log.error("Calculate loss with shapes %s and %s" % (preds.shape, grid_tensor.shape))
            raise ex


class BinaryCrossEntropyCellLoss(AbstractLoss):
    def __init__(self, gpu, coords: VariableStructure, cuda, variable: Variables, conf_match_weight, reduction,
                 activation_is_exp, loss_weight_strategy):
        super().__init__(coords=coords, gpu=gpu, cuda=cuda, one_hot=True, variable=variable,
                         function=nn.BCELoss(reduction=reduction), conf_match_weight=conf_match_weight,
                         reduction=reduction, activation_is_exp=activation_is_exp,
                         loss_weight_strategy=loss_weight_strategy)

    def __call__(self, preds, grid_tensor, tag="none", epoch=None):
        preds, grid_tensor, p_unmatched, gt_unmatched = super().__call__(preds, grid_tensor)  # shape (?, 10)

        if preds.shape[1] > 1 and self.variable == Variables.CLASS:
            raise NotImplementedError(
                "Calculate binary cross entropy for %d classes! Full shape %s" % (preds.shape[1], preds.shape))

        Log.debug("%s on e.g. %s vs gt=%s" % (str(self), preds[0], grid_tensor[0]))
        preds, grid_tensor = self.__get_flat__(preds, grid_tensor)

        return self.__apply_function__(preds, grid_tensor, p_unmatched, gt_unmatched, tag=tag, epoch=epoch)


class FocalConfidenceLoss(AbstractLoss):
    """Binary focal loss on confidence.

    Preferred path (training): **raw geometry-head logits** + ``binary_cross_entropy_with_logits``
    so gradients flow through a single sigmoid (RetinaNet-style). Fallback: probabilities +
    ``logit(prob)`` chain when ``conf_logits_flat`` is absent (legacy / tests).
    """

    def __init__(self, coords: VariableStructure, gpu, cuda, variable: Variables, conf_match_weight,
                 activation_is_exp, loss_weight_strategy, gamma=2.0, alpha=0.25):
        class focal_bce_logits:
            def __init__(self, gamma, alpha):
                self.gamma = float(gamma)
                self.alpha = float(alpha)

            def __call__(self, logits, gts):
                gts = torch.clamp(gts.to(dtype=logits.dtype), min=0.0, max=1.0)
                ce = F.binary_cross_entropy_with_logits(logits, gts, reduction="none")
                pt = torch.exp(-ce)
                alpha_t = torch.where(gts > 0.5,
                                      torch.full_like(gts, self.alpha),
                                      torch.full_like(gts, 1.0 - self.alpha))
                focal = alpha_t * ((1 - pt) ** self.gamma) * ce
                focal = torch.where(torch.isfinite(focal), focal, torch.zeros_like(focal))
                return focal.mean()

            def cuda(self):
                return self

        class focal_prob_via_inverse_logit:
            """Legacy: probs → logit → BCEWithLogits (two sigmoids in autograd); avoid when logits available."""

            def __init__(self, gamma, alpha):
                self.gamma = float(gamma)
                self.alpha = float(alpha)

            def __call__(self, preds, gts):
                probs = torch.clamp(preds, min=1e-6, max=1 - 1e-6)
                gts = torch.clamp(gts, min=0.0, max=1.0)
                logits = torch.logit(probs)
                ce = F.binary_cross_entropy_with_logits(logits, gts, reduction="none")
                pt = torch.exp(-ce)
                alpha_t = torch.where(gts > 0.5,
                                      torch.full_like(gts, self.alpha),
                                      torch.full_like(gts, 1.0 - self.alpha))
                focal = alpha_t * ((1 - pt) ** self.gamma) * ce
                focal = torch.where(torch.isfinite(focal), focal, torch.zeros_like(focal))
                return focal.mean()

            def cuda(self):
                return self

        self._focal_logits_fn = focal_bce_logits(gamma=gamma, alpha=alpha)
        self._focal_prob_fn = focal_prob_via_inverse_logit(gamma=gamma, alpha=alpha)

        super().__init__(coords=coords, gpu=gpu, cuda=cuda, one_hot=True, variable=variable,
                         function=self._focal_logits_fn,
                         conf_match_weight=conf_match_weight, reduction="mean",
                         activation_is_exp=activation_is_exp, loss_weight_strategy=loss_weight_strategy)

    def _prepare_conf_logits_slices(self, logits_full: torch.Tensor, grid_tensor: torch.Tensor):
        """Same masking / splits as AbstractLoss for CONF, but parallel logits tensor."""
        device = grid_tensor.device
        breaks_py = self.__get_breaks_in_coords__()
        breaks_t = torch.tensor(breaks_py, device=device, dtype=torch.long)
        breaks_train_py = self.__get_breaks_in_coords__(only_training=True)
        breaks_train_t = torch.tensor(breaks_train_py, device=device, dtype=torch.long)

        invalid_flags = torch.any(grid_tensor[:, self.coords.get_position_of_training_vars()].isnan(), dim=-1)
        valid_flags = ~invalid_flags

        logits_tm = logits_full[valid_flags]
        logits_tum = logits_full[~valid_flags]
        grid_tm = grid_tensor[valid_flags]
        grid_tum = grid_tensor[~valid_flags]

        grid_tum = self.replace_nans(grid_tum, variable=self.variable)

        # tensor_split indices must live on CPU (PyTorch API); tensors may be CUDA.
        br_cpu, br_train_cpu = breaks_t.cpu(), breaks_train_t.cpu()
        _, labels_m, _ = torch.tensor_split(grid_tm, br_cpu, dim=1)
        _, labels_um, _ = torch.tensor_split(grid_tum, br_cpu, dim=1)

        _, logits_m, _ = torch.tensor_split(logits_tm, br_train_cpu, dim=1)
        _, logits_um, _ = torch.tensor_split(logits_tum, br_train_cpu, dim=1)

        logits_m, gt_m = self.__get_flat__(logits_m, labels_m)
        logits_um, gt_um = self.__get_flat__(logits_um, labels_um)
        return logits_m, gt_m, logits_um, gt_um

    def __call__(self, preds, grid_tensor, tag="none", epoch=None, conf_logits_flat=None):
        if conf_logits_flat is None:
            self.function = self._focal_prob_fn
            preds, grid_tensor, p_unmatched, gt_unmatched = super().__call__(preds, grid_tensor)
            return self.__apply_function__(preds, grid_tensor, p_unmatched, gt_unmatched, tag=tag, epoch=epoch)

        self.function = self._focal_logits_fn
        logits_m, gt_m, logits_um, gt_um = self._prepare_conf_logits_slices(conf_logits_flat, grid_tensor)

        matched_loss = torch.tensor(0.0, dtype=logits_m.dtype, device=logits_m.device)
        if logits_m.numel() > 0 and not torch.all(gt_m.isnan()):
            matched_loss = self.function(logits_m, gt_m)

        unmatched_loss = torch.tensor(0.0, dtype=logits_um.dtype, device=logits_um.device)
        if logits_um.numel() > 0 and not torch.all(gt_um.isnan()):
            unmatched_loss = self.function(logits_um, gt_um)

        normalizing_matched = len(logits_m) if self.reduction == "sum" else 1
        normalizing_unmatched = len(logits_um) if self.reduction == "sum" else 1
        mean_matched_loss = matched_loss / normalizing_matched
        mean_unmatched_loss = unmatched_loss / normalizing_unmatched

        Log.scalars(tag=tag, dict={"loss_conf_batch/match/mean": mean_matched_loss,
                                   "loss_conf_batch/unmatch/mean": mean_unmatched_loss},
                    epoch=epoch)

        add_weights, weight_factors = get_actual_weight(epoch, "conf/match",
                                                        weight_strategy=self.loss_weight_strategy,
                                                        weight=self.conf_match_weight[0],
                                                        activation_is_exponential=self.activation_is_exp)
        weighted_loss = weight_factors * matched_loss + add_weights
        add_weights, weight_factors = get_actual_weight(epoch, "conf/nomatch",
                                                        weight_strategy=self.loss_weight_strategy,
                                                        weight=self.conf_match_weight[1],
                                                        activation_is_exponential=self.activation_is_exp)
        weighted_loss += weight_factors * (self.conf_negative_weight * unmatched_loss) + add_weights
        mean_loss = (mean_matched_loss + mean_unmatched_loss) * 0.5

        return weighted_loss, mean_loss


class MeanSquaredErrorLoss(AbstractLoss):
    def __init__(self, reduction, coords: VariableStructure, gpu, cuda, variable: Variables, batch_size: int,
                 conf_match_weight, activation_is_exp, loss_weight_strategy) -> None:
        super().__init__(coords=coords, gpu=gpu, cuda=cuda, one_hot=True, variable=variable,
                         function=torch.nn.MSELoss(reduction=reduction), conf_match_weight=conf_match_weight,
                         reduction=reduction, activation_is_exp=activation_is_exp,
                         loss_weight_strategy=loss_weight_strategy)
        self.batch_size = batch_size

    def __call__(self, preds, grid_tensor, tag="none", epoch=None):
        preds, grid_tensor, p_unmatched, gt_unmatched = super().__call__(preds, grid_tensor)
        return self.__apply_function__(preds, grid_tensor, p_unmatched, gt_unmatched, tag=tag, epoch=epoch)

    def __str__(self):
        string = super().__str__()
        return self.function.reduction.capitalize() + string
####################################################################################################################################
class TangentCosineLoss(AbstractLoss):
    def __init__(self, coords: VariableStructure, gpu, cuda, variable: Variables, conf_match_weight, activation_is_exp,
                 loss_weight_strategy, reduction="mean") -> None:
        if variable != Variables.GEOMETRY:
            raise NotImplementedError("Tangent loss is only applicable to geometry.")

        class tangent_loss_fct:
            def __init__(self, reduction):
                self.reduction = reduction

            def __call__(self, preds, gts):
                # 1. 예측 선분(preds)과 정답 선분(gts)에서 각각 방향 벡터 추출
                # (주의: coords.py에 정의된 방식에 따라 x, y 좌표 인덱싱 필요)
                pred_vec = preds[:, 2:4] - preds[:, 0:2] # 예시: (x2, y2) - (x1, y1)
                gt_vec = gts[:, 2:4] - gts[:, 0:2]

                # 2. 벡터 정규화 (크기를 1로 만듦)
                pred_vec_norm = torch.nn.functional.normalize(pred_vec, p=2, dim=1)
                gt_vec_norm = torch.nn.functional.normalize(gt_vec, p=2, dim=1)

                # 3. 코사인 유사도 계산
                # 방향이 완벽히 같으면 1, 정반대면 -1, 직교하면 0
                cos_sim = torch.sum(pred_vec_norm * gt_vec_norm, dim=1)

                # 4. Loss 계산: 1에서 코사인 유사도를 빼서 평행할수록 0에 가까워지도록 함
                # abs()를 쓰는 이유: 전선의 방향이 180도 뒤집혀 있어도 같은 선분으로 취급하기 위함
                losses = 1.0 - torch.abs(cos_sim)

                if self.reduction == "mean":
                    return losses.mean()
                else:
                    return losses.sum()

            def cuda(self):
                pass

        super().__init__(coords=coords, gpu=gpu, cuda=cuda, one_hot=True, variable=variable,
                         function=tangent_loss_fct(reduction=reduction), conf_match_weight=conf_match_weight,
                         reduction=reduction, activation_is_exp=activation_is_exp,
                         loss_weight_strategy=loss_weight_strategy)

    def __call__(self, preds, grid_tensor, tag="none", epoch=None):
        preds, grid_tensor, p_unmatched, gt_unmatched = super().__call__(preds, grid_tensor)
        return self.__apply_function__(preds, grid_tensor, p_unmatched, gt_unmatched, tag=tag, epoch=epoch)

class DiscriminativeEmbeddingLoss(AbstractLoss):
    def __init__(self, coords: VariableStructure, gpu, cuda, variable: Variables, conf_match_weight, activation_is_exp,
                 loss_weight_strategy, delta_v=0.5, delta_d=3.0,
                 lambda_reg: float = 0.0, concat_geom_dims: int = 0, warmup_epochs: int = 0) -> None:
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.lambda_reg = float(lambda_reg)
        self.concat_geom_dims = int(concat_geom_dims)
        self.warmup_epochs = int(warmup_epochs)

        super().__init__(coords=coords, gpu=gpu, cuda=cuda, one_hot=True, variable=variable,
                         function=torch.nn.MSELoss(), conf_match_weight=conf_match_weight, reduction="mean",
                         activation_is_exp=activation_is_exp, loss_weight_strategy=loss_weight_strategy)

    def _warmup_scale(self, loss_tensor, epoch):
        if self.warmup_epochs <= 0 or epoch is None:
            return loss_tensor
        scale = min(1.0, float(epoch + 1) / float(self.warmup_epochs))
        return loss_tensor * scale

    def __call__(self, embed_preds, grid_tensor, tag="none", epoch=None,
                 batch_size=1, items_per_image=None):
        if items_per_image is not None and batch_size > 1:
            return self._per_image_loss(embed_preds, grid_tensor, batch_size, items_per_image, tag, epoch)
        result = self._single_image_loss(embed_preds, grid_tensor)
        result["total"] = self._warmup_scale(result["total"], epoch)
        self._log_stats(result, tag, epoch)
        return result["total"], result["total"]

    def _per_image_loss(self, embed_preds, grid_tensor, batch_size, items_per_image, tag, epoch):
        total_loss = torch.tensor(0.0, device=embed_preds.device, requires_grad=False)
        count = 0
        agg_pull, agg_push, agg_reg = 0.0, 0.0, 0.0
        agg_instances, agg_embeds = 0, 0
        agg_intra, agg_inter = [], []

        for b in range(batch_size):
            s = b * items_per_image
            e = (b + 1) * items_per_image
            result = self._single_image_loss(embed_preds[s:e], grid_tensor[s:e])
            loss_val = result["total"]
            if loss_val.requires_grad or loss_val.item() > 0:
                total_loss = total_loss + loss_val
                count += 1
            agg_pull += result["pull"].item()
            agg_push += result["push"].item()
            agg_reg += result.get("reg", torch.tensor(0.0)).item() if isinstance(result.get("reg"), torch.Tensor) else 0.0
            agg_instances += result["num_instances"]
            agg_embeds += result["num_valid_embeds"]
            if result["mean_intra_dist"] is not None:
                agg_intra.append(result["mean_intra_dist"])
            if result["mean_inter_dist"] is not None:
                agg_inter.append(result["mean_inter_dist"])

        if count == 0:
            zero = torch.tensor(0.0, device=embed_preds.device)
            return zero, zero
        avg = total_loss / count
        avg = self._warmup_scale(avg, epoch)

        summary = {
            "total": avg, "pull": torch.tensor(agg_pull / max(1, count)),
            "push": torch.tensor(agg_push / max(1, count)),
            "reg": torch.tensor(agg_reg / max(1, count)),
            "num_instances": agg_instances // max(1, count),
            "num_valid_embeds": agg_embeds // max(1, count),
            "mean_intra_dist": sum(agg_intra) / len(agg_intra) if agg_intra else None,
            "mean_inter_dist": sum(agg_inter) / len(agg_inter) if agg_inter else None,
        }
        self._log_stats(summary, tag, epoch)
        return avg, avg

    def _log_stats(self, result, tag, epoch):
        log_dict = {
            "embed_loss/total": result["total"].item() if isinstance(result["total"], torch.Tensor) else result["total"],
            "embed_loss/pull": result["pull"].item() if isinstance(result["pull"], torch.Tensor) else result["pull"],
            "embed_loss/push": result["push"].item() if isinstance(result["push"], torch.Tensor) else result["push"],
            "embed_stats/num_instances": result["num_instances"],
            "embed_stats/num_valid_embeds": result["num_valid_embeds"],
        }
        if result["mean_intra_dist"] is not None:
            log_dict["embed_stats/mean_intra_dist"] = result["mean_intra_dist"]
        if result["mean_inter_dist"] is not None:
            log_dict["embed_stats/mean_inter_dist"] = result["mean_inter_dist"]
        if "reg" in result and isinstance(result["reg"], torch.Tensor):
            log_dict["embed_loss/reg"] = float(result["reg"].detach().mean().cpu())
        Log.scalars(tag=tag, epoch=epoch, dict=log_dict)

    def _single_image_loss(self, embed_preds, grid_tensor):
        zero = torch.tensor(0.0, device=embed_preds.device)
        empty_result = {"total": zero, "pull": zero, "push": zero, "reg": zero,
                        "num_instances": 0, "num_valid_embeds": 0,
                        "mean_intra_dist": None, "mean_inter_dist": None}

        pos_geom = self.coords.get_position_of(Variables.GEOMETRY)
        invalid_flags = torch.any(grid_tensor[:, pos_geom].isnan(), dim=-1)
        valid_flags = ~invalid_flags
        
        valid_embeds = embed_preds[valid_flags]
        valid_grids = grid_tensor[valid_flags]
        
        if len(valid_embeds) == 0:
            return empty_result
            
        pos_inst = self.coords.get_position_of(Variables.INSTANCE)
        gts = valid_grids[:, pos_inst[0]]
        instance_valid = torch.isfinite(gts) & (gts > 0)
        valid_embeds = valid_embeds[instance_valid]
        valid_grids = valid_grids[instance_valid]
        gts = gts[instance_valid]

        if len(valid_embeds) == 0:
            return empty_result

        working_embeds = valid_embeds
        if self.concat_geom_dims > 0:
            geom_idx = pos_geom[: min(self.concat_geom_dims, len(pos_geom))]
            geom_feat = valid_grids[:, geom_idx].detach().to(dtype=valid_embeds.dtype, device=valid_embeds.device)
            working_embeds = torch.cat([valid_embeds, geom_feat], dim=-1)
        
        unique_ids, inverse = torch.unique(gts, return_inverse=True)
        num_instances = len(unique_ids)

        pull_loss = torch.tensor(0.0, device=embed_preds.device)
        push_loss = torch.tensor(0.0, device=embed_preds.device)
        reg_loss = torch.tensor(0.0, device=embed_preds.device)
        intra_dists = []
        inter_dists = []

        if num_instances > 0:
            # ---------- Vectorized cluster means ----------
            counts = torch.bincount(inverse, minlength=num_instances).to(working_embeds.dtype).unsqueeze(1)
            sums_w = torch.zeros((num_instances, working_embeds.shape[1]), dtype=working_embeds.dtype,
                                 device=working_embeds.device)
            sums_w = sums_w.index_add(0, inverse, working_embeds)
            means_w = sums_w / torch.clamp(counts, min=1.0)

            # ---------- Pull term ----------
            dists = torch.norm(working_embeds - means_w[inverse], dim=1)
            pull_each = torch.clamp(dists - self.delta_v, min=0.0) ** 2
            pull_sum = torch.zeros((num_instances,), dtype=working_embeds.dtype, device=working_embeds.device)
            pull_sum = pull_sum.index_add(0, inverse, pull_each)
            pull_mean_per_cluster = pull_sum / torch.clamp(counts.squeeze(1), min=1.0)
            pull_loss = pull_mean_per_cluster.mean()

            # Metrics
            intra_sum = torch.zeros((num_instances,), dtype=working_embeds.dtype, device=working_embeds.device)
            intra_sum = intra_sum.index_add(0, inverse, dists)
            intra_mean = intra_sum / torch.clamp(counts.squeeze(1), min=1.0)
            intra_dists = intra_mean.detach().cpu().tolist()

            # ---------- Optional reg term ----------
            if self.lambda_reg > 0:
                sums_e = torch.zeros((num_instances, valid_embeds.shape[1]), dtype=valid_embeds.dtype,
                                     device=valid_embeds.device)
                sums_e = sums_e.index_add(0, inverse, valid_embeds)
                means_e = sums_e / torch.clamp(counts, min=1.0)
                reg_loss = torch.sum(means_e ** 2, dim=1).mean()

            # ---------- Push term (vectorized pairwise center distance) ----------
            if num_instances > 1:
                pairwise = torch.cdist(means_w, means_w, p=2)  # [K, K]
                mask_offdiag = ~torch.eye(num_instances, dtype=torch.bool, device=pairwise.device)
                push_terms = torch.clamp(self.delta_d - pairwise[mask_offdiag], min=0.0) ** 2
                push_loss = push_terms.mean()

                # reporting metric: unique-pair inter distance (upper triangle)
                tri_i, tri_j = torch.triu_indices(num_instances, num_instances, offset=1, device=pairwise.device)
                inter_dists = pairwise[tri_i, tri_j].detach().cpu().tolist()

        total_loss = pull_loss + push_loss + self.lambda_reg * reg_loss
        return {
            "total": total_loss,
            "pull": pull_loss.detach(),
            "push": push_loss.detach(),
            "reg": reg_loss.detach(),
            "num_instances": num_instances,
            "num_valid_embeds": len(valid_embeds),
            "mean_intra_dist": sum(intra_dists) / len(intra_dists) if intra_dists else None,
            "mean_inter_dist": sum(inter_dists) / len(inter_dists) if inter_dists else None,
        }
################################################################################################################################################################################
class NormLoss(AbstractLoss):
    def __init__(self, coords: VariableStructure, gpu, cuda, variable: Variables, conf_match_weight, activation_is_exp,
                 loss_weight_strategy, reduction="mean") -> None:
        if variable != Variables.GEOMETRY:
            raise NotImplementedError("We only implemented the norm loss for geometry.")

        class norm_loss_fct:
            def __init__(self, reduction):
                self.reduction = reduction

            def __call__(self, preds, gts):
                vls = [linesegment_euclidean_distance(gt=g, pred=p.unsqueeze(dim=0), coords=coords, use_conf=False)
                       for p, g in zip(preds, gts)]
                if self.reduction == "mean":
                    return torch.cat(vls).mean()
                else:
                    return torch.cat(vls).sum()

            def cuda(self):
                pass

        super().__init__(coords=coords, gpu=gpu, cuda=cuda, one_hot=True, variable=variable,
                         function=norm_loss_fct(reduction=reduction), conf_match_weight=conf_match_weight,
                         reduction=reduction, activation_is_exp=activation_is_exp,
                         loss_weight_strategy=loss_weight_strategy)

    def __call__(self, preds, grid_tensor, tag="none", epoch=None):
        preds, grid_tensor, p_unmatched, gt_unmatched = super().__call__(preds, grid_tensor)
        return self.__apply_function__(preds, grid_tensor, p_unmatched, gt_unmatched, tag=tag, epoch=epoch)


def get_loss(losses, args, coords: VariableStructure, weights: list, anchors, conf_weights):
    loss_weight_strategy = args.loss_weight_strategy

    disc_kw = dict(
        delta_v=getattr(args, "embedding_delta_v", 0.5),
        delta_d=getattr(args, "embedding_delta_d", 3.0),
        lambda_reg=float(getattr(args, "embedding_lambda_reg", 0.0)),
        concat_geom_dims=int(getattr(args, "embedding_concat_geom_dims", 0)),
        warmup_epochs=int(getattr(args, "embedding_loss_warmup_epochs", 0)),
    )

    functions = []
    assert (len(coords.train_vars()) == len(losses))
    for i, loss in enumerate(losses):
        activation_is_exp = get_activations(args.activations, coords=coords,
                                            linerep=coords.line_representation.enum).activations[i].is_exp()
        variable = coords.train_vars()[i]
        if loss == LOSS.CROSS_ENTROPY_MEAN:
            functions.append(CrossEntropyCellLoss(gpu=args.gpu, coords=coords, cuda=args.cuda, variable=variable,
                                                  conf_match_weight=conf_weights, reduction="mean",
                                                  activation_is_exp=activation_is_exp,
                                                  loss_weight_strategy=loss_weight_strategy))
        elif loss == LOSS.CROSS_ENTROPY_SUM:
            functions.append(CrossEntropyCellLoss(gpu=args.gpu, coords=coords, cuda=args.cuda, variable=variable,
                                                  conf_match_weight=conf_weights, reduction="sum",
                                                  activation_is_exp=activation_is_exp,
                                                  loss_weight_strategy=loss_weight_strategy))
        elif loss == LOSS.BINARY_CROSS_ENTROPY_MEAN:
            functions.append(BinaryCrossEntropyCellLoss(gpu=args.gpu, coords=coords, cuda=args.cuda, variable=variable,
                                                        conf_match_weight=conf_weights, reduction="mean",
                                                        activation_is_exp=activation_is_exp,
                                                        loss_weight_strategy=loss_weight_strategy))
        elif loss == LOSS.BINARY_CROSS_ENTROPY_SUM:
            functions.append(BinaryCrossEntropyCellLoss(gpu=args.gpu, coords=coords, cuda=args.cuda, variable=variable,
                                                        conf_match_weight=conf_weights, reduction="sum",
                                                        activation_is_exp=activation_is_exp,
                                                        loss_weight_strategy=loss_weight_strategy))
        elif loss == LOSS.FOCAL_MEAN:
            functions.append(FocalConfidenceLoss(coords=coords, gpu=args.gpu, cuda=args.cuda, variable=variable,
                                                conf_match_weight=conf_weights,
                                                activation_is_exp=activation_is_exp,
                                                loss_weight_strategy=loss_weight_strategy,
                                                gamma=getattr(args, "focal_gamma", 2.0),
                                                alpha=getattr(args, "focal_alpha", 0.25)))
        elif loss == LOSS.MSE_SUM:
            functions.append(MeanSquaredErrorLoss(reduction="sum", coords=coords, gpu=args.gpu, cuda=args.cuda,
                                                  variable=variable, batch_size=args.batch_size,
                                                  conf_match_weight=conf_weights,
                                                  activation_is_exp=activation_is_exp,
                                                  loss_weight_strategy=loss_weight_strategy))
        elif loss == LOSS.MSE_MEAN:
            functions.append(MeanSquaredErrorLoss(reduction="mean", coords=coords, gpu=args.gpu, cuda=args.cuda,
                                                  variable=variable, batch_size=args.batch_size,
                                                  conf_match_weight=conf_weights,
                                                  activation_is_exp=activation_is_exp,
                                                  loss_weight_strategy=loss_weight_strategy))
        elif loss == LOSS.NORM_MEAN:
            functions.append(
                NormLoss(coords=coords, gpu=args.gpu, cuda=args.cuda, variable=variable, conf_match_weight=conf_weights,
                         activation_is_exp=activation_is_exp, loss_weight_strategy=loss_weight_strategy,
                         reduction="mean"))
        # (기존 코드)
        elif loss == LOSS.NORM_SUM:
            functions.append(
                NormLoss(coords=coords, gpu=args.gpu, cuda=args.cuda, variable=variable, conf_match_weight=conf_weights,
                         activation_is_exp=activation_is_exp, loss_weight_strategy=loss_weight_strategy,
                         reduction="sum"))
    #####################################################################################################################################
        # (신규 추가)
        elif loss == LOSS.TANGENT_COSINE:
            functions.append(
                TangentCosineLoss(coords=coords, gpu=args.gpu, cuda=args.cuda, variable=variable, conf_match_weight=conf_weights,
                         activation_is_exp=activation_is_exp, loss_weight_strategy=loss_weight_strategy,
                         reduction="mean"))
        elif loss == LOSS.DISCRIMINATIVE_EMBEDDING:
            functions.append(
                DiscriminativeEmbeddingLoss(coords=coords, gpu=args.gpu, cuda=args.cuda, variable=variable,
                                           conf_match_weight=conf_weights,
                                           activation_is_exp=activation_is_exp,
                                           loss_weight_strategy=loss_weight_strategy,
                                           **disc_kw))
        ########################################################################################################################################################
        else:
            raise NotImplementedError("Unknown loss type %s" % loss)

    # Option A: INSTANCE는 geom head가 아니라 embed head에서만 학습.
    # coords.train_vars()에는 INSTANCE가 없어도, GT grid_tensor에는 INSTANCE가 존재하므로 loss는 추가 가능.
    if getattr(args, "train_instance_embedding", False) \
            and not any(type(f) is DiscriminativeEmbeddingLoss for f in functions):
        functions.append(
            DiscriminativeEmbeddingLoss(
                coords=coords, gpu=args.gpu, cuda=args.cuda, variable=Variables.INSTANCE,
                conf_match_weight=conf_weights, activation_is_exp=False,
                loss_weight_strategy=loss_weight_strategy,
                **disc_kw,
            )
        )
    composed_loss = LossComposition(losses=functions, args=args, coords=coords, weights=weights, anchors=anchors)
    for fct in functions:
        if getattr(fct, "variable", None) == Variables.CONF:
            fct.conf_negative_weight = float(getattr(args, "conf_negative_weight", 1.0))
    return composed_loss


def get_actual_weight(epoch, variable_str, weight_strategy, weight, activation_is_exponential):
    if weight_strategy == LossWeighting.FIXED or weight_strategy == LossWeighting.FIXED_NORM:
        weight_factor = weight
        add_weight = 0  # add no regularization (=0)
    elif weight_strategy == LossWeighting.LEARN:
        if torch.any(weight > 20) or torch.any(weight < 0.05):
            Log.warning(f"Clamp necessary: {weight}")
            weight = torch.clamp(weight, min=0.05, max=20)

        # this is the thing we actually learn: sigma
        Log.scalars(tag="", epoch=epoch, dict={os.path.join("loss_" + variable_str, "pure_learn_weight"): weight})
        # this is the actual variance sigma ** 2
        Log.scalars(tag="", epoch=epoch, dict={os.path.join("loss_" + variable_str, "var"): math.pow(weight, 2)})

        if activation_is_exponential:
            # else we have 1 / s^2
            weight_factor = torch.pow(weight, -2)
        else:
            # when using softmax/sigmoid we have 1 / (2s^2)
            weight_factor = 1 / (2 * torch.pow(weight, 2))

        # the regularization is on log(sigma)
        add_weight = torch.log(weight)
    elif weight_strategy == LossWeighting.LEARN_LOG or weight_strategy == LossWeighting.LEARN_LOG_NORM:
        if torch.any(weight > math.log(20)) or torch.any(weight < math.log(0.05)):
            Log.warning(f"Clamp necessary: {weight}")
            weight = torch.clamp(weight, min=math.log(0.05), max=math.log(20))

        # this is the thing we acutally learn: log(sigma ** 2)
        Log.scalars(tag="", epoch=epoch, dict={os.path.join("loss_" + variable_str, "pure_learn_weight"): weight})
        # this is the actual varianz sigma ** 2 = e^(log(sigma ** 2))
        Log.scalars(tag="", epoch=epoch, dict={os.path.join("loss_" + variable_str, "var"): math.exp(weight)})

        if activation_is_exponential:
            # else we have e^-s
            weight_factor = torch.exp(-1. * weight)
        else:
            # when using softmax/sigmoid we have 1 / ( 2 * e^s)
            weight_factor = 0.5 * torch.exp(-1. * weight)

        # the regularization is on log(sigma), we train log(sigma ** 2)
        add_weight = 0.5 * weight
    else:
        raise NotImplementedError("We do not know %s" % weight_strategy)

    if weight_strategy == LossWeighting.LEARN \
            or weight_strategy == LossWeighting.LEARN_LOG \
            or weight_strategy == LossWeighting.LEARN_LOG_NORM:
        Log.scalars(tag="", epoch=epoch, dict={os.path.join("loss_" + variable_str, "actual_weight"): weight_factor})
    elif epoch == 0:
        Log.scalars(tag="", epoch=epoch, dict={os.path.join("loss_" + variable_str, "actual_weight"): weight_factor})

    return add_weight, weight_factor


class LossComposition:
    def __init__(self, losses, args, coords: VariableStructure, weights: list, anchors):
        self.add_weights = []
        self.losses = losses

        self.anchors = anchors
        self.args = args
        self.coords = coords
        self.weights = weights
        self.matcher = CellMatcher(coords, args)
        self._conf_dist_logged_epochs = set()
        Log.debug("Weights=%s" % self.weights)
        if len(self.losses) != len(self.weights):
            raise ValueError("Please specify the same number of loss terms as weights, we got %s loss terms, "
                             "but %s weights." % (self.losses, self.weights))

    def _add_conf_stats(self, stats_dict, prefix, values):
        if values is None or values.numel() == 0:
            return
        vals = values.detach().float()
        stats_dict[f"{prefix}/mean"] = float(vals.mean().item())
        stats_dict[f"{prefix}/std"] = float(vals.std(unbiased=False).item())
        stats_dict[f"{prefix}/p50"] = float(torch.quantile(vals, 0.5).item())
        stats_dict[f"{prefix}/p90"] = float(torch.quantile(vals, 0.9).item())
        stats_dict[f"{prefix}/high_ratio"] = float((vals >= float(self.args.confidence)).float().mean().item())

    def _log_confidence_distribution(self, geom_preds_flat, reduced_grid_tensor, epoch, tag):
        if Variables.CONF not in self.coords.train_vars():
            return
        epoch_key = int(epoch) if epoch is not None else -1
        log_key = (str(tag), epoch_key)
        if log_key in self._conf_dist_logged_epochs:
            return
        self._conf_dist_logged_epochs.add(log_key)

        conf_pos_pred = self.coords.get_position_within_prediction(Variables.CONF)
        conf_pos_gt = self.coords.get_position_of(Variables.CONF)
        geom_pos_gt = self.coords.get_position_of(Variables.GEOMETRY)

        pred_conf = geom_preds_flat[:, conf_pos_pred].detach()
        gt_conf = reduced_grid_tensor[:, conf_pos_gt].detach()
        gt_conf = torch.where(torch.isnan(gt_conf), torch.zeros_like(gt_conf), gt_conf)
        matched_mask = ~torch.any(torch.isnan(reduced_grid_tensor[:, geom_pos_gt]), dim=1)
        unmatched_mask = ~matched_mask

        stats = {}
        self._add_conf_stats(stats, "conf_dist/pred/all", pred_conf)
        self._add_conf_stats(stats, "conf_dist/pred/matched", pred_conf[matched_mask])
        self._add_conf_stats(stats, "conf_dist/pred/unmatched", pred_conf[unmatched_mask])
        self._add_conf_stats(stats, "conf_dist/gt/all", gt_conf)
        self._add_conf_stats(stats, "conf_dist/gt/matched", gt_conf[matched_mask])
        self._add_conf_stats(stats, "conf_dist/gt/unmatched", gt_conf[unmatched_mask])

        num_predictors = int(getattr(self.args, "num_predictors", 1))
        if num_predictors > 0:
            predictor_idx = torch.arange(len(pred_conf), device=pred_conf.device) % num_predictors
            for p in range(num_predictors):
                p_mask = predictor_idx == p
                self._add_conf_stats(stats, f"conf_dist/pred/p{p}/all", pred_conf[p_mask])
                self._add_conf_stats(stats, f"conf_dist/pred/p{p}/matched", pred_conf[p_mask & matched_mask])
                self._add_conf_stats(stats, f"conf_dist/pred/p{p}/unmatched", pred_conf[p_mask & unmatched_mask])
                stats[f"conf_dist/pred/p{p}/match_ratio"] = float((p_mask & matched_mask).float().mean().item())

        if len(stats) > 0:
            Log.scalars(tag=tag, dict=stats, epoch=epoch)

    def __call__(self, geom_preds, embed_preds, grid_tensor, filenames, epoch, tag="dummy_loss", geom_logits=None):
        if torch.any(torch.isnan(geom_preds)):
            raise ValueError("Prediction can not contain nans!")
        Log.debug("LossComposition input shapes geom=%s embed=%s gt=%s "
                  "(target geom~[B, 1024, 8, vars_train], embed~[B, 1024, 8, 8])"
                  % (tuple(geom_preds.shape), tuple(embed_preds.shape), tuple(grid_tensor.shape)))

        weighted_losses = torch.zeros((1), device=self.args.cuda, dtype=torch.float32)
        losses = []
        mean_losses = []

        from datetime import datetime
        from datetime import timedelta
        start = datetime.now()
        
        if self.args.anchors == AnchorDistribution.NONE:
            geom_preds_sorted, reduced_grid_tensor = self.matcher.sort_cells_by_geometric_match(
                                                                preds=geom_preds,
                                                                grid_tensor=grid_tensor,
                                                                epoch=epoch, tag=tag,
                                                                filenames=filenames)
            geom_preds_flat = geom_preds_sorted.reshape(-1, self.coords.num_vars_to_train())
            embed_preds_flat = embed_preds.reshape(-1, embed_preds.shape[-1])
            if geom_logits is not None:
                geom_logits_flat = geom_logits.reshape(-1, self.coords.num_vars_to_train())
            else:
                geom_logits_flat = None
        else:
            reduced_grid_tensor = grid_tensor.reshape(-1, self.coords.get_length())
            geom_preds_flat = geom_preds.reshape(-1, self.coords.num_vars_to_train())
            embed_preds_flat = embed_preds.reshape(-1, embed_preds.shape[-1])
            if geom_logits is not None:
                geom_logits_flat = geom_logits.reshape(-1, self.coords.num_vars_to_train())
            else:
                geom_logits_flat = None

        self._log_confidence_distribution(geom_preds_flat=geom_preds_flat,
                                          reduced_grid_tensor=reduced_grid_tensor,
                                          epoch=epoch, tag=tag)

        end = datetime.now()
        seconds = ((end - start) / timedelta(milliseconds=1))
        Log.debug("Matching done in %dms" % (seconds))

        batch_size = geom_preds.shape[0]
        items_per_image = geom_preds.shape[1] * geom_preds.shape[2]

        for i, t in enumerate(self.losses):
            if torch.all(reduced_grid_tensor[:, self.coords.get_position_of(t.variable)].isnan()):
                losses.append(0)
                mean_losses.append(0)
                continue
            elif torch.all(reduced_grid_tensor[:, self.coords.get_position_of(Variables.GEOMETRY)].isnan()) \
                    and t.variable != Variables.CONF:
                losses.append(0)
                mean_losses.append(0)
                continue
            else:
                t: AbstractLoss
                try:
                    if t.variable == Variables.INSTANCE:
                        loss_val, mean_loss_val = t(embed_preds_flat, reduced_grid_tensor, tag=tag, epoch=epoch,
                                                    batch_size=batch_size, items_per_image=items_per_image)
                    elif isinstance(t, FocalConfidenceLoss) and geom_logits_flat is not None:
                        loss_val, mean_loss_val = t(geom_preds_flat, reduced_grid_tensor, tag=tag, epoch=epoch,
                                                    conf_logits_flat=geom_logits_flat)
                    else:
                        loss_val, mean_loss_val = t(geom_preds_flat, reduced_grid_tensor, tag=tag, epoch=epoch)
                except ValueError as e:
                    if "No valid data" in str(e):
                        loss_val = torch.tensor(0.0, device=geom_preds_flat.device, requires_grad=True)
                        mean_loss_val = torch.tensor(0.0, device=geom_preds_flat.device)
                    else:
                        raise e

                variable_strings = ("conf" if self.losses[i].variable == Variables.CONF
                                    else str(self.losses[i].variable))
                activation_list = get_activations(self.args.activations, self.coords, self.args.linerep).activations
                is_exp = activation_list[i].is_exp() if i < len(activation_list) else False

                add_weights, weight_factors = get_actual_weight(epoch, variable_strings,
                                                                weight_strategy=self.args.loss_weight_strategy,
                                                                weight=self.weights[i],
                                                                activation_is_exponential=is_exp)

                mean_losses.append(mean_loss_val.detach().cpu()) 
                losses.append(loss_val.detach().cpu())  
                l = loss_val * weight_factors + add_weights
                
                # 에러나던 len(preds)를 len(geom_preds_flat)으로 수정
                Log.scalars(tag=tag, epoch=epoch,
                            dict={os.path.join("loss_" + variable_strings + "_batch", "sum",
                                               "weighted"): l.item() / max(1, len(geom_preds_flat))}) # 0 나누기 방지용 max 추가
                Log.scalars(tag=tag, epoch=epoch, dict={"loss_batch/sum/weighted": weighted_losses})

                weighted_losses += l

        return losses, weighted_losses, mean_losses

    def __repr__(self):
        string = "Loss Composition <"
        for t in self.losses:
            string += str(t) + ", "
        string += ">"
        return string

    def is_exp_activation(self, index):
        return get_activations(self.args.activations, self.coords, self.args.linerep).activations[index].is_exp()

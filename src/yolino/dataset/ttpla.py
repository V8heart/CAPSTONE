import math
import os
import glob
import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

# Avoid OpenCV internal threading oversubscription when DataLoader workers > 0.
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from yolino.dataset.dataset_base import DatasetInfo
from yolino.model.e2e_polyline_order import canonicalize_polyline_xy
from yolino.model.e2e_train_bridge import resample_polyline_xy
from yolino.utils.duplicates import LineDuplicates
from yolino.utils.enums import Dataset, Variables
from yolino.utils.geometry import t_cart2pol, t_pol2cart
from yolino.utils.logger import Log


def _replay_geom_augment_on_full_lines(
    lines_yx: torch.Tensor,
    params: dict,
    *,
    sky_crop: int,
    side_crop: int,
    out_h: int,
    out_w: int,
    src_h: int,
    src_w: int,
) -> torch.Tensor:
    """Replay image geom augment on **full** polylines (no crop intersection split).

    Grid supervision uses short crop *segments* from :class:`RandomCropWithLabels`;
    E2E / DN GT must keep each wire's full vertex chain so 5-pt resampling spans the
    entire arc length in the final training crop.
    """
    if lines_yx is None or lines_yx.numel() == 0:
        return lines_yx
    out = lines_yx.clone()
    out[..., 0] = out[..., 0] - float(sky_crop)
    out[..., 1] = out[..., 1] - float(side_crop)

    h_pre = max(int(src_h) - int(sky_crop), 1)
    w_pre = max(int(src_w) - 2 * int(side_crop), 1)

    angle = params.get("rrotate_angle", None)
    if angle is not None:
        rad = math.radians(float(angle))
        for i in range(int(out.shape[0])):
            for j in range(int(out.shape[1])):
                if torch.isnan(out[i, j, 0]):
                    continue
                pt = out[i, j]
                pt = pt.clone()
                pt[0] = pt[0] - h_pre / 2.0
                pt[1] = pt[1] - w_pre / 2.0
                pol = t_cart2pol(pt)
                pol[1] = pol[1] + rad
                pt = t_pol2cart(pol)
                pt[0] = pt[0] + h_pre / 2.0
                pt[1] = pt[1] + w_pre / 2.0
                out[i, j] = pt

    if "crop_t" in params:
        out[..., 0] = out[..., 0] - float(params["crop_t"])
        out[..., 1] = out[..., 1] - float(params["crop_l"])

    if "crop_h" in params and "crop_w" in params:
        ch = max(int(params["crop_h"]), 1)
        cw = max(int(params["crop_w"]), 1)
        out[..., 0] = out[..., 0] * (float(out_h) / float(ch))
        out[..., 1] = out[..., 1] * (float(out_w) / float(cw))

    return out

class TTPLADataset(DatasetInfo):
    @classmethod
    def _resolve_dataset_root(cls):
        return os.getenv("DATASET_TTPLA")

    @classmethod
    def _infer_hw_from_dataset(cls):
        dataset_path = cls._resolve_dataset_root()
        if not dataset_path:
            return None
        for split in ["train", "val", "test"]:
            img_dir = os.path.join(dataset_path, "images", split)
            if not os.path.isdir(img_dir):
                continue
            pngs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
            if len(pngs) == 0:
                continue
            with Image.open(pngs[0]) as im:
                w, h = im.size
            return int(h), int(w)
        return None

    @classmethod
    def height(cls) -> int:
        inferred = cls._infer_hw_from_dataset()
        if inferred is not None:
            return inferred[0]
        return 1024

    @classmethod
    def width(cls) -> int:
        inferred = cls._infer_hw_from_dataset()
        if inferred is not None:
            return inferred[1]
        return 1024

    @classmethod
    def get_max_image_size(cls):
        inferred = cls._infer_hw_from_dataset()
        if inferred is not None:
            return inferred
        return 1024, 1024

    def __init__(self, split, args, augment=False, sky_crop=0, side_crop=0, load_only_labels=False,
                 show=False, load_full_dataset=False, lazy=False, ignore_duplicates=False, store_lines=False):
        
        dataset_path = os.getenv("DATASET_TTPLA")
        
        # [핵심] 뼈대 클래스를 초기화하기 전에 폴더의 진짜 파일 개수를 미리 셉니다!
        def count_files(s):
            d = os.path.join(dataset_path, "images", s)
            if os.path.exists(d):
                return len(glob.glob(os.path.join(d, "*.png")))
            return 0

        train_count = count_files("train")
        val_count = count_files("val")
        test_count = count_files("test")

        super().__init__(Dataset.TTPLA, split, args, sky_crop=sky_crop, side_crop=side_crop, augment=augment,
                         num_classes=0, train=train_count, test=test_count, val=val_count,
                         override_dataset_path=dataset_path,
                         load_only_labels=load_only_labels, show=show, load_sequences=load_full_dataset, lazy=lazy,
                         ignore_duplicates=ignore_duplicates, store_lines=store_lines)

        self.file_names = []
        self.img_list = []
        self.label_files = []

        self.gather_files()

    def gather_files(self):
        if self.lazy:
            return

        img_dir = os.path.join(self.dataset_path, "images", self.split)
        label_dir = os.path.join(self.dataset_path, "labels", self.split)

        if not os.path.exists(img_dir) or not os.path.exists(label_dir):
            Log.warning(f"TTPLA split '{self.split}' directory not found.")
            return

        image_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        for img_path in tqdm(image_files, desc=f"Loading TTPLA {self.split}"):
            if self.abort(count=len(self.img_list)):
                break
            basename = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_dir, f"{basename}.npy")
            
            if os.path.exists(label_path):
                self.img_list.append(img_path)
                self.file_names.append(basename)
                self.label_files.append(label_path)
        
        self.on_load()

    def on_load(self):
        if len(self.file_names) != len(self.img_list):
            raise IndexError("Mismatch between images and labels.")
        super().on_load()

    def _build_e2e_gt_pack(self, lines: torch.Tensor) -> dict:
        """Fixed-size tensors for default_collate + E2E loss (x,y in pixels).

        Additional fields for the center-DETR head (--e2e_mode=center):
          - ``center_xy``     ``[NI, 2]``  float; arc-length midpoint per instance (x, y).
                              Zero-padded for invalid slots (mask via ``inst_mask``).
          - ``poly_length``   ``[NI]``     float; total arc length per instance in pixels.

        Centers are **recomputed from the (post-augment) ``lines`` tensor** so they
        stay consistent with any crop/rotation. The baked ``centers`` field saved by
        ``scripts/create_ttpla_yolino_center_detr.py`` is informational only.
        """
        args = self.args
        ni = int(getattr(args, "e2e_gt_max_instances", 32))
        mp = int(getattr(args, "e2e_gt_max_points", 128))
        dev, dt = lines.device, lines.dtype
        padded = torch.zeros((ni, mp, 2), device=dev, dtype=dt)
        inst_m = torch.zeros((ni,), device=dev, dtype=torch.bool)
        pt_m = torch.zeros((ni, mp), device=dev, dtype=torch.bool)
        center_xy = torch.zeros((ni, 2), device=dev, dtype=dt)
        poly_length = torch.zeros((ni,), device=dev, dtype=dt)
        out = {
            "padded": padded,
            "inst_mask": inst_m,
            "pt_mask": pt_m,
            "center_xy": center_xy,
            "poly_length": poly_length,
        }
        if lines is None or lines.dim() < 2:
            return out
        valid_inst = torch.logical_not(torch.isnan(lines[:, 0, 0]))
        idxs = torch.where(valid_inst)[0]
        # Augment-after canonicalization toggle (exp51 5-pt head needs deterministic
        # left→right / top→bottom order regardless of rotation). Default ON; set
        # ``e2e_gt_canonicalize=False`` to fall back to legacy load-time flip only.
        canonicalize = bool(getattr(args, "e2e_gt_canonicalize", True))
        vertical_angle_deg = float(getattr(args, "e2e_gt_vertical_angle_deg", 80.0))
        for k in range(min(int(idxs.numel()), ni)):
            i = int(idxs[k])
            row = lines[i]
            valid_pt = torch.logical_not(torch.isnan(row[:, 0]))
            pj = torch.where(valid_pt)[0]
            npt_full = int(pj.numel())
            if npt_full < 2:
                continue
            # Gather **all** valid vertices as (x, y). Do not take only the first
            # ``mp`` indices — dense TTPLA labels can have 1k+ points per wire and
            # truncating by index collapses DN / 5-pt GT to a short prefix of the arc.
            xy_full = torch.empty((npt_full, 2), device=dev, dtype=dt)
            for t in range(npt_full):
                j = int(pj[t])
                # lines stores (y, x); pack to (x, y).
                xy_full[t, 0] = row[j, 1]  # x
                xy_full[t, 1] = row[j, 0]  # y
            if npt_full > mp:
                xy_pts = resample_polyline_xy(
                    xy_full, torch.ones((npt_full,), device=dev, dtype=torch.bool), mp
                )
                npt = mp
            else:
                xy_pts = xy_full
                npt = npt_full
            if canonicalize:
                xy_pts = canonicalize_polyline_xy(xy_pts, vertical_angle_deg=vertical_angle_deg)
            padded[k, :npt] = xy_pts
            xs = xy_pts[:, 0]
            ys = xy_pts[:, 1]
            inst_m[k] = True
            pt_m[k, :npt] = True
            # Arc-length midpoint + total length on the augmented polyline.
            dx = xs[1:] - xs[:-1]
            dy = ys[1:] - ys[:-1]
            seg = torch.sqrt(dx * dx + dy * dy).clamp(min=0.0)
            total = float(seg.sum().item())
            poly_length[k] = float(total)
            if total <= 0.0:
                center_xy[k, 0] = xs[0]
                center_xy[k, 1] = ys[0]
            else:
                half = 0.5 * total
                cum = torch.cumsum(seg, dim=0).cpu().numpy()
                jj = int(np.searchsorted(cum, half))
                jj = max(0, min(jj, npt - 2))
                prev = float(cum[jj - 1]) if jj > 0 else 0.0
                denom = float(seg[jj].item())
                t_frac = (half - prev) / denom if denom > 1e-9 else 0.0
                center_xy[k, 0] = xs[jj] + t_frac * (xs[jj + 1] - xs[jj])
                center_xy[k, 1] = ys[jj] + t_frac * (ys[jj + 1] - ys[jj])
        return out

    def __len__(self):
        return len(self.img_list)

    def __get_labels__(self, idx):
        """
        Returns:
            gridable_lines: torch.Tensor [num_instances, max_points, 2] storing **(y, x)** in pixel space
                (row/col convention used by the grid pipeline). The E2E pack :meth:`_build_e2e_gt_pack` converts
                to **(x, y)** for Chamfer against head outputs.
            instance_ids: list[int], length=num_instances

        Supported label formats:
            1) Legacy npy: array/list of polylines (no IDs) -> fallback IDs: 1..N
            2) New npy/json-like dict:
               {
                 "polylines": [...],              # list of polylines OR list of {"points": ...}
                 "instance_ids": [int, int, ...]  # optional
               }
               or {"instances": [{"instance_id": ..., "points": ...}, ...]}
        """
        raw = np.load(self.label_files[idx], allow_pickle=True)
        payload = raw
        if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
            payload = raw.item()

        polylines = []
        instance_ids = []

        if isinstance(payload, dict):
            if "instances" in payload:
                for inst in payload.get("instances", []):
                    pts = inst.get("points", [])
                    if len(pts) >= 2:
                        polylines.append(pts)
                        iid = inst.get("instance_id", len(instance_ids) + 1)
                        instance_ids.append(int(iid))
            else:
                raw_polys = payload.get("polylines", [])
                raw_ids = payload.get("instance_ids", None)
                for i, p in enumerate(raw_polys):
                    pts = p.get("points", []) if isinstance(p, dict) else p
                    if len(pts) >= 2:
                        polylines.append(pts)
                        if raw_ids is not None and i < len(raw_ids):
                            instance_ids.append(int(raw_ids[i]))
                        else:
                            if isinstance(p, dict) and "instance_id" in p:
                                instance_ids.append(int(p["instance_id"]))
                            else:
                                instance_ids.append(len(instance_ids) + 1)
        else:
            # Legacy: npy of polylines only
            for p in payload:
                if len(p) >= 2:
                    polylines.append(p)
                    instance_ids.append(len(instance_ids) + 1)

        num_lines = len(polylines)
        max_points = max([len(p) for p in polylines], default=1)
        gridable_lines = torch.ones((max(1, num_lines), max(1, max_points), 2), dtype=torch.float32) * torch.nan

        for i, poly in enumerate(polylines):
            poly_np = np.array(poly)
            if len(poly_np) < 2:
                continue

            # [가장 중요한 드론용 정렬 로직]
            # 위아래(Y) 상관없이 무조건 왼쪽(X가 작은 쪽)에서 오른쪽으로 통일!
            if poly_np[0, 0] > poly_np[-1, 0]:
                poly_np = poly_np[::-1]

            for j, pt in enumerate(poly_np):
                x, y = pt
                gridable_lines[i, j, 0] = float(y)
                gridable_lines[i, j, 1] = float(x)

        return gridable_lines, instance_ids

    def __load_image__(self, idx):
        img = cv2.imread(self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        if self.load_only_labels:
            image = self.dummy_image(self.args.img_size)
        else:
            cv_image = self.__load_image__(idx)
            image = self.__make_torch__(cv_image)
            del cv_image

        lines, instance_ids = self.__get_labels__(idx)
        lines_for_e2e = lines.clone() if lines is not None else None
        image, lines, params = self.__augment__(idx, image, lines)

        # ---------------------------------------------------------
        # Build instance payload aligned to *augmented* line order.
        # We preserve original polygon instance IDs by geometric remap after
        # augmentation, instead of relying on raw index order.
        # ---------------------------------------------------------
        valid_indices = []
        if lines is not None and len(lines.shape) >= 2:
            # valid line: first point's y is finite (line tensor is NaN-padded)
            valid_mask = torch.logical_not(torch.isnan(lines[:, 0, 0]))
            valid_indices = torch.where(valid_mask)[0].tolist()

        # Keep original polygon instance IDs by line index. Random crop may drop lines,
        # but it does not reorder retained line slots in this dataset pipeline.
        instance_list = []
        for line_idx in valid_indices:
            # YOLinO Predictor는 여기서 받은 값을 Geometry 뒤에 순서대로 붙입니다.
            # Variables.INSTANCE 방에 들어갈 ID값만 넘파이로 깔끔하게 넣어줍니다.
            inst_data = np.zeros(4, dtype=np.float32)
            if line_idx < len(instance_ids):
                inst_data[0] = float(instance_ids[line_idx])
            else:
                inst_data[0] = float(len(instance_list) + 1)
            
            instance_list.append(inst_data)
            
        variables = [instance_list]
        # ---------------------------------------------------------

        duplicates = LineDuplicates(filename=self.file_names[idx], grid_shape=self.args.grid_shape,
                                    num_predictors=self.args.num_predictors)
                                    
        # lines를 넘파이로 변환하여 전달 (형식 통일)
        grid_tensor, grid = self.__get_grid_labels__(torch.unsqueeze(lines, dim=0).numpy(), variables,
                                                     idx, image=image,
                                                     duplicates=duplicates)

        if bool(getattr(self.args, "e2e_train_with_gt_polylines", False)):
            src_h = int(self.args.img_size[0])
            src_w = int(self.args.img_size[1])
            out_h = int(image.shape[1])
            out_w = int(image.shape[2])
            lines_e2e = _replay_geom_augment_on_full_lines(
                lines_for_e2e,
                params,
                sky_crop=int(self.augmentor.sky_crop),
                side_crop=int(self.augmentor.side_crop),
                out_h=out_h,
                out_w=out_w,
                src_h=src_h,
                src_w=src_w,
            )
            e2e_gt = self._build_e2e_gt_pack(lines_e2e)
            return image, grid_tensor, self.file_names[idx], duplicates.dict(), params, e2e_gt
        return image, grid_tensor, self.file_names[idx], duplicates.dict(), params

    def check_img_size(self):
        if not np.all(np.mod(self.args.img_size, 32) == 0):
            Log.warning("Image dimensions must be divisible by 32.")
            return False
        return True

    def __construct_filename__(self, filename):
        # YOLinO가 확장자 없이 filename만 던져주면, 실제 .png 파일의 전체 경로를 만들어 줍니다.
        # self.split이 train, val 등 현재 상태를 담고 있습니다.
        return os.path.join(self.dataset_path, "images", self.split, filename + ".png")

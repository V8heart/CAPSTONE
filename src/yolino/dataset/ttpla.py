import math
import os
import glob
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from yolino.dataset.dataset_base import DatasetInfo
from yolino.utils.duplicates import LineDuplicates
from yolino.utils.enums import Dataset, Variables
from yolino.utils.logger import Log

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

    def __len__(self):
        return len(self.img_list)

    def __get_labels__(self, idx):
        """
        Returns:
            gridable_lines: torch.Tensor [num_instances, max_points, 2] (y, x) with NaN padding
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
        import cv2
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
        image, lines, params = self.__augment__(idx, image, lines)

        # ---------------------------------------------------------
        # [최종 수정] 딕셔너리를 빼고, 순수 넘파이 배열(float32)만 담습니다.
        # ---------------------------------------------------------
        # Augmentation (notably random crop) may change the number of line instances.
        # Recompute the currently valid line count from augmented labels to keep variable
        # metadata aligned with `lines` in GridFactory.
        if lines is None or len(lines.shape) < 2:
            num_instances = 0
        else:
            # valid line: first point's y is finite (line tensor is NaN-padded)
            valid_mask = torch.logical_not(torch.isnan(lines[:, 0, 0]))
            num_instances = int(torch.sum(valid_mask).item())

        # Keep original IDs where possible; if augmentation split/removed lines and we
        # cannot map exactly, fall back to a contiguous local ID for remaining lines.
        instance_list = []
        
        for i in range(num_instances):
            # YOLinO Predictor는 여기서 받은 값을 Geometry 뒤에 순서대로 붙입니다.
            # Variables.INSTANCE 방에 들어갈 ID값만 넘파이로 깔끔하게 넣어줍니다.
            inst_data = np.zeros(4, dtype=np.float32)
            # If label has explicit/global IDs, preserve them.
            # Legacy labels still fall back to 1..N from __get_labels__.
            if i < len(instance_ids):
                inst_data[0] = float(instance_ids[i])
            else:
                inst_data[0] = float(i + 1)
            
            instance_list.append(inst_data)
            
        variables = [instance_list]
        # ---------------------------------------------------------

        duplicates = LineDuplicates(filename=self.file_names[idx], grid_shape=self.args.grid_shape,
                                    num_predictors=self.args.num_predictors)
                                    
        # lines를 넘파이로 변환하여 전달 (형식 통일)
        grid_tensor, grid = self.__get_grid_labels__(torch.unsqueeze(lines, dim=0).numpy(), variables,
                                                     idx, image=image,
                                                     duplicates=duplicates)

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

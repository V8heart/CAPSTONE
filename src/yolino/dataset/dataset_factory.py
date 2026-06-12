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

from torch.utils.data import DataLoader, default_collate

from yolino.dataset.argoverse20_pytorch import Argoverse2Dataset
from yolino.dataset.dataset_base import DatasetInfo
from yolino.utils.enums import Dataset
from yolino.utils.logger import Log
from yolino.dataset.ttpla import TTPLADataset


def dataloader_worker_init_fn(_worker_id):
    """Re-apply thread limits inside forked DataLoader workers."""
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    import cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def collate_ttpla_with_optional_e2e_gt(batch):
    """
    Same stacking as ``default_collate``, but asserts batched ``e2e_gt`` dict shapes when present
    (6-tuple samples from :class:`TTPLADataset`).
    """
    if not batch:
        raise ValueError("empty batch")
    out = default_collate(batch)
    if len(out) == 6:
        images, _grid, _fn, _dup, _params, e2e = out
        b = int(images.shape[0])
        if e2e["padded"].shape[0] != b or e2e["inst_mask"].shape[0] != b or e2e["pt_mask"].shape[0] != b:
            raise RuntimeError(
                "e2e_gt batch shape mismatch: B=%d padded=%s inst=%s pt=%s"
                % (b, tuple(e2e["padded"].shape), tuple(e2e["inst_mask"].shape), tuple(e2e["pt_mask"].shape))
            )
    return out


class DatasetFactory:
    from yolino.dataset.tusimple_pytorch import TusimpleDataset
    from yolino.dataset.caltech_pytorch import CaltechDataSet
    from yolino.dataset.cifar10_pytorch import CifarDataSet
    from yolino.dataset.culane_pytorch import CULaneDataSet
    datasets = {
        Dataset.CULANE: CULaneDataSet,
        Dataset.CIFAR: CifarDataSet,
        Dataset.CALTECH: CaltechDataSet,
        Dataset.TUSIMPLE: TusimpleDataset,
        Dataset.ARGOVERSE2: Argoverse2Dataset,
        Dataset.TTPLA: TTPLADataset,
    }

    @classmethod
    def __str__(self) -> str:
        return str(self.datasets.keys())

    @classmethod
    def get_coords(self, split, args):
        if not args.dataset in DatasetFactory.datasets:
            raise NotImplementedError("We did not set class for %s" % args.dataset)

        dataset_class = DatasetFactory.datasets[args.dataset]
        coords = dataset_class(split, args, lazy=True).coords
        return coords

    @classmethod
    def get_path(self, split, args):
        if not args.dataset in DatasetFactory.datasets:
            raise NotImplementedError("We did not set class for %s" % args.dataset)

        dataset_class = DatasetFactory.datasets[args.dataset]
        path = dataset_class(split, args, lazy=True).dataset_path
        img_path = dataset_class(split, args, lazy=True).dataset_img_path
        return path, img_path

    @classmethod
    def get_img_size(self, dataset, img_height):
        if not dataset in DatasetFactory.datasets:
            raise NotImplementedError("We did not setup %s" % dataset)

        dataset_class = DatasetFactory.datasets[dataset]
        width = dataset_class.get_img_width(img_height)
        return [img_height, width]

    @classmethod
    def get_max_image_size(cls, dataset):
        if not dataset in DatasetFactory.datasets:
            raise NotImplementedError("We did not setup %s" % dataset)

        dataset_class = DatasetFactory.datasets[dataset]
        img_size = dataset_class.get_max_image_size()
        return img_size

    @classmethod
    def get(self, dataset_enum: Dataset, only_available, split, args, shuffle, augment, load_only_labels=False,
            show=False, load_full=False, ignore_duplicates=False, store_lines=False, sampler=None) -> (DatasetInfo, DataLoader):
        if dataset_enum in DatasetFactory.datasets:
            dataset_class = DatasetFactory.datasets[dataset_enum]
            dataset = dataset_class(split=split, args=args, augment=augment, load_only_labels=load_only_labels,
                                    show=show, load_full_dataset=load_full, ignore_duplicates=ignore_duplicates,
                                    store_lines=store_lines)

            if dataset.is_available():
                Log.debug("Load data from %s with batch=%d" % (dataset_enum, args.batch_size))
                collate_fn = None
                if dataset_enum == Dataset.TTPLA and bool(getattr(args, "e2e_train_with_gt_polylines", False)):
                    collate_fn = collate_ttpla_with_optional_e2e_gt
                nw = int(args.loading_workers)
                loader_kw = dict(
                    batch_size=args.batch_size,
                    shuffle=(shuffle if sampler is None else False),
                    sampler=sampler,
                    drop_last=True,
                    num_workers=nw,
                    pin_memory=args.gpu,
                )
                # Reduce fork-after-thread deadlock risk: workers are forked only
                # **once** at first iter() and reused across epochs via
                # persistent_workers — instead of being re-forked every epoch end.
                # We cannot use multiprocessing_context='spawn' here because the
                # TTPLA dataset object embeds non-picklable thread locks (yolino
                # Log handlers attached to dataset). The launch script is expected
                # to set OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS=1
                # before main() so the one-shot fork has no BLAS threads to copy.
                if nw > 0:
                    loader_kw["persistent_workers"] = True
                    loader_kw["prefetch_factor"] = 2
                    loader_kw["worker_init_fn"] = dataloader_worker_init_fn
                if collate_fn is not None:
                    loader_kw["collate_fn"] = collate_fn
                loader = DataLoader(dataset, **loader_kw)
                return dataset, loader
            else:
                if only_available:
                    raise FileNotFoundError("Could not find the data for %s" % (dataset_enum))
                else:
                    return dataset, None
        else:
            raise ValueError("%s not found. Please choose from %s" % (dataset_enum, DatasetFactory.datasets.keys()))

#!/usr/bin/env python3
"""Minimal local Detectron2 training smoke test for the site_1 tile dataset.

Supports standard 3-band RGB tiles (default) or 4-band RGB+DSM GeoTIFFs via
``--four-band`` (custom rasterio mapper + 4-channel ResNet stem).

COCO ``iscrowd=1`` polygons (deposits / sub-threshold boulders from
``gpkg_to_coco.py``) are treated as ignore regions: not positives, and not
used as background negatives in RPN / ROI loss (see ``crowd_ignore.py``).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import detectron2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco_eval_with_recall import BoulderCOCOEvaluator  # noqa: E402
from crowd_ignore import (  # noqa: E402
    CrowdAwareDatasetMapper,
    GeneralizedRCNNWithIgnore,
    RPNWithIgnore,
    StandardROIHeadsWithIgnore,
    transform_annotations_with_ignore,
)
from multiband_io import (  # noqa: E402
    FOUR_BAND_PIXEL_MEAN,
    FOUR_BAND_PIXEL_STD,
    load_bgrd_uint8,
    patch_four_band_stem_from_checkpoint,
)

# Ensure custom meta-arch / RPN / ROI heads are registered (side effects).
_ = (GeneralizedRCNNWithIgnore, RPNWithIgnore, StandardROIHeadsWithIgnore)


class FourBandDatasetMapper(CrowdAwareDatasetMapper):
    """Like CrowdAwareDatasetMapper, but loads 4-band BGR+DSM via rasterio."""

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = load_bgrd_uint8(dataset_dict["file_name"])
        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            transform_annotations_with_ignore(
                self, dataset_dict, transforms, image_shape
            )

        return dataset_dict


class BoulderTrainer(DefaultTrainer):
    four_band: bool = False

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
        os.makedirs(output_folder, exist_ok=True)
        return BoulderCOCOEvaluator(dataset_name, cfg, False, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        if cls.four_band:
            mapper = FourBandDatasetMapper(cfg, is_train=True)
        else:
            mapper = CrowdAwareDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if cls.four_band:
            mapper = FourBandDatasetMapper(cfg, is_train=False)
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        return super().build_test_loader(cfg, dataset_name)


def dataset_class_names(base_path: Path) -> list[str]:
    """Read category names from the train COCO JSON (sorted by category id)."""
    data = json.loads((base_path / "train_annotations.json").read_text())
    cats = sorted(data["categories"], key=lambda c: c["id"])
    return [c["name"] for c in cats]


def register_datasets(base_path: Path) -> list[str]:
    class_names = dataset_class_names(base_path)
    # Clear prior registrations so re-runs with a different dataset dir work.
    for name in ("boulder_train", "boulder_valid", "boulder_test"):
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)

    for name, image_dir, ann_file in [
        ("boulder_train", "train", "train_annotations.json"),
        ("boulder_valid", "valid", "validation_annotations.json"),
        ("boulder_test", "test", "testing_annotations.json"),
    ]:
        register_coco_instances(
            name,
            {},
            str(base_path / ann_file),
            str(base_path / image_dir),
        )
        MetadataCatalog.get(name).thing_classes = class_names
    return class_names


def build_cfg(
    base_path: Path,
    output_dir: Path,
    max_iter: int,
    batch_size: int,
    num_workers: int,
    device: str = "cpu",
    num_classes: int = 1,
    four_band: bool = False,
    image_size: int = 2000,
    eval_during_train: bool = True,
) -> detectron2.config.CfgNode:
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.DATASETS.TRAIN = ("boulder_train",)
    # Always override the zoo default (coco_2017_val). Leaving this unset
    # makes EvalHook look for datasets/coco/annotations/instances_val2017.json.
    if eval_during_train:
        cfg.DATASETS.TEST = ("boulder_valid",)
        cfg.TEST.EVAL_PERIOD = max(1, min(50, max(1, max_iter // 2)))
    else:
        cfg.DATASETS.TEST = ()
        cfg.TEST.EVAL_PERIOD = 0
    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

    cfg.INPUT.MAX_SIZE_TRAIN = image_size
    cfg.INPUT.MIN_SIZE_TRAIN = image_size
    cfg.INPUT.MAX_SIZE_TEST = image_size
    cfg.INPUT.MIN_SIZE_TEST = image_size
    cfg.INPUT.FORMAT = "BGR"

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    # Crowd-ignore path for COCO iscrowd=1 (deposits / small boulders).
    cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNNWithIgnore"
    cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RPNWithIgnore"
    cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeadsWithIgnore"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.DEVICE = device

    if four_band:
        # len(PIXEL_MEAN) sets ResNet stem in_channels via ShapeSpec.
        cfg.MODEL.PIXEL_MEAN = FOUR_BAND_PIXEL_MEAN
        cfg.MODEL.PIXEL_STD = FOUR_BAND_PIXEL_STD

    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = max_iter
    # Multi-step LR decay (similar to BoulderCalculator notebook proportions)
    raw_steps = {
        max(1, max_iter // 5),
        max(2, (max_iter * 2) // 5),
        max(3, (max_iter * 3) // 5),
        max(4, (max_iter * 4) // 5),
    }
    cfg.SOLVER.STEPS = tuple(sorted(s for s in raw_steps if s < max_iter))
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.CHECKPOINT_PERIOD = max(1, min(50, max(1, max_iter // 2)))
    cfg.TEST.DETECTIONS_PER_IMAGE = 300

    cfg.OUTPUT_DIR = str(output_dir)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def summarize_dataset(base_path: Path) -> dict:
    summary = {}
    for split, ann_name in [
        ("train", "train_annotations.json"),
        ("valid", "validation_annotations.json"),
        ("test", "testing_annotations.json"),
    ]:
        data = json.loads((base_path / ann_name).read_text())
        n_crowd = sum(1 for a in data["annotations"] if a.get("iscrowd", 0))
        summary[split] = {
            "images": len(data["images"]),
            "annotations": len(data["annotations"]),
            "trainable": len(data["annotations"]) - n_crowd,
            "crowd_ignore": n_crowd,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("segmentation/coco_dataset"),
        help="COCO dataset dir. Default: segmentation/coco_dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("segmentation/training_run"),
        help="Training output dir. Default: segmentation/training_run",
    )
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in --output-dir if present.",
    )
    parser.add_argument(
        "--four-band",
        action="store_true",
        help="Train on 4-band RGB+DSM GeoTIFFs (custom mapper + 4-channel stem).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional local Detectron2 .pkl/.pth to initialize from (skips model zoo download).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=2000,
        help="Square resize used for train/test (default 2000; use smaller for smoke tests).",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help=(
            "Skip periodic validation during training and the final COCO eval. "
            "Use on low-VRAM GPUs when eval spikes memory or stalls."
        ),
    )
    args = parser.parse_args()

    BoulderTrainer.four_band = args.four_band
    class_names = register_datasets(args.dataset_dir)
    cfg = build_cfg(
        args.dataset_dir,
        args.output_dir,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        num_classes=len(class_names),
        four_band=args.four_band,
        image_size=args.image_size,
        eval_during_train=not args.no_eval,
    )
    if args.weights is not None:
        cfg.MODEL.WEIGHTS = str(args.weights)

    print("Classes:", class_names)
    print("Four-band:", args.four_band)
    print("Dataset summary:", json.dumps(summarize_dataset(args.dataset_dir), indent=2))
    print("Train images registered:", len(DatasetCatalog.get("boulder_train")))
    print("Output dir:", cfg.OUTPUT_DIR)
    print("MAX_ITER:", cfg.SOLVER.MAX_ITER)
    print("PIXEL_MEAN:", list(cfg.MODEL.PIXEL_MEAN))

    # Cache zoo weights path before resume_or_load (needed to patch 4-channel stem).
    zoo_weights = cfg.MODEL.WEIGHTS

    trainer = BoulderTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    if args.four_band and not args.resume:
        # COCO stem is 3-channel and was skipped; copy RGB weights and init DSM channel.
        local_zoo = Path(zoo_weights)
        if not local_zoo.exists():
            # model_zoo URL was resolved by checkpointer; use cached file if present
            from detectron2.utils.file_io import PathManager

            local_zoo = Path(PathManager.get_local_path(zoo_weights))
        print(f"Patching 4-channel stem from {local_zoo}")
        patch_four_band_stem_from_checkpoint(trainer.model, local_zoo)

    trainer.train()

    if args.no_eval:
        print("Skipping final validation (--no-eval).")
        return

    # Point TEST at boulder_valid for the post-train eval (may already be set).
    cfg.DATASETS.TEST = ("boulder_valid",)
    evaluator = BoulderTrainer.build_evaluator(cfg, "boulder_valid")
    eval_results = trainer.test(cfg, trainer.model, evaluators=[evaluator])
    (args.output_dir / "metrics_valid.json").write_text(json.dumps(eval_results, indent=2))
    print("Validation metrics:", json.dumps(eval_results, indent=2))


if __name__ == "__main__":
    main()

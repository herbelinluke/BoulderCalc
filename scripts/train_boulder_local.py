#!/usr/bin/env python3
"""Minimal local Detectron2 training smoke test for the site_1 tile dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import detectron2
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import COCOEvaluator
from detectron2 import model_zoo


class BoulderTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
        os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


def register_datasets(base_path: Path) -> None:
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
        MetadataCatalog.get(name).thing_classes = ["Boulder"]


def build_cfg(
    base_path: Path,
    output_dir: Path,
    max_iter: int,
    batch_size: int,
    num_workers: int,
    device: str = "cpu",
) -> detectron2.config.CfgNode:
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.DATASETS.TRAIN = ("boulder_train",)
    cfg.DATASETS.TEST = ("boulder_valid",)
    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

    cfg.INPUT.MAX_SIZE_TRAIN = 2000
    cfg.INPUT.MIN_SIZE_TRAIN = 2000
    cfg.INPUT.MAX_SIZE_TEST = 2000
    cfg.INPUT.MIN_SIZE_TEST = 2000

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.DEVICE = device

    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = max_iter
    # Multi-step LR decay (similar to BoulderCalculator notebook proportions)
    cfg.SOLVER.STEPS = tuple(
        sorted(
            {
                max(1, max_iter // 5),
                max(2, (max_iter * 2) // 5),
                max(3, (max_iter * 3) // 5),
                max(4, (max_iter * 4) // 5),
            }
        )
    )
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.CHECKPOINT_PERIOD = max(50, max_iter // 10)
    cfg.TEST.EVAL_PERIOD = max(50, max_iter // 10)
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
        summary[split] = {
            "images": len(data["images"]),
            "annotations": len(data["annotations"]),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/home/herbs/Documents/tamucc/segmentation/coco_dataset"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/herbs/Documents/tamucc/segmentation/training_run"),
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
    args = parser.parse_args()

    register_datasets(args.dataset_dir)
    cfg = build_cfg(
        args.dataset_dir,
        args.output_dir,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )

    print("Dataset summary:", json.dumps(summarize_dataset(args.dataset_dir), indent=2))
    print("Train images registered:", len(DatasetCatalog.get("boulder_train")))
    print("Output dir:", cfg.OUTPUT_DIR)
    print("MAX_ITER:", cfg.SOLVER.MAX_ITER)

    trainer = BoulderTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

    evaluator = BoulderTrainer.build_evaluator(cfg, "boulder_valid")
    eval_results = trainer.test(cfg, trainer.model, evaluators=[evaluator])
    (args.output_dir / "metrics_valid.json").write_text(json.dumps(eval_results, indent=2))
    print("Validation metrics:", json.dumps(eval_results, indent=2))


if __name__ == "__main__":
    main()

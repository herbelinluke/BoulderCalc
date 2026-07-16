"""Detectron2 helpers so COCO ``iscrowd=1`` boxes are true ignore regions.

Default Detectron2 DatasetMapper drops ``iscrowd`` annotations, which means
those pixels become unlabeled background and can be used as *negatives*.
This module:

1. Keeps non-crowd polygons as normal GT (positives).
2. Stores crowd boxes on the dataset dict as ``ignore_boxes``.
3. ``GeneralizedRCNNWithIgnore`` attaches them onto each ``Instances`` as
   ``_ignore_boxes`` (private attr; length need not match GT count).
4. Marks RPN anchors / ROI proposals that overlap ignore boxes (and are not
   already positive) as ignore label ``-1`` so they contribute neither
   positive nor negative loss.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from detectron2.data import DatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.modeling import (
    META_ARCH_REGISTRY,
    PROPOSAL_GENERATOR_REGISTRY,
    ROI_HEADS_REGISTRY,
)
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.proposal_generator.rpn import RPN
from detectron2.modeling.roi_heads import StandardROIHeads
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom

# IoU with an ignore box at/above this turns a non-positive sample into ignore.
IGNORE_IOU_THRESH = 0.5


def _boxes_xyxy_from_annos(annos: list[dict], image_size: tuple[int, int]) -> Boxes:
    """Build Boxes in XYXY_ABS from annotation dicts (raw COCO or transformed)."""
    if not annos:
        return Boxes(torch.zeros((0, 4), dtype=torch.float32))
    from detectron2.structures import BoxMode

    xyxy = []
    h, w = image_size
    for obj in annos:
        bbox = obj["bbox"]
        mode = obj.get("bbox_mode", BoxMode.XYWH_ABS)
        box = BoxMode.convert(bbox, mode, BoxMode.XYXY_ABS)
        x0 = max(0.0, min(float(box[0]), w))
        y0 = max(0.0, min(float(box[1]), h))
        x1 = max(0.0, min(float(box[2]), w))
        y1 = max(0.0, min(float(box[3]), h))
        if x1 > x0 and y1 > y0:
            xyxy.append([x0, y0, x1, y1])
    if not xyxy:
        return Boxes(torch.zeros((0, 4), dtype=torch.float32))
    return Boxes(torch.as_tensor(xyxy, dtype=torch.float32))


def get_ignore_boxes(inst: Instances, device: torch.device | None = None) -> Boxes:
    """Read private ``_ignore_boxes``; return empty Boxes if missing."""
    boxes = getattr(inst, "_ignore_boxes", None)
    if boxes is None:
        dev = device or (
            inst.gt_boxes.device if inst.has("gt_boxes") else torch.device("cpu")
        )
        return Boxes(torch.zeros((0, 4), dtype=torch.float32, device=dev))
    if device is not None:
        return boxes.to(device)
    return boxes


def mark_ignore_overlaps(
    labels: torch.Tensor,
    sample_boxes: Boxes,
    ignore_boxes: Boxes | None,
    iou_thresh: float = IGNORE_IOU_THRESH,
) -> torch.Tensor:
    """Set labels from 0 (negative) to -1 where IoU with any ignore box >= thresh.

    Positives (label == 1) are left unchanged so a real boulder that touches an
    ignore region can still train.
    """
    if ignore_boxes is None or len(ignore_boxes) == 0 or len(sample_boxes) == 0:
        return labels
    ious = pairwise_iou(ignore_boxes, sample_boxes)  # Ig x N
    max_iou, _ = ious.max(dim=0)
    out = labels.clone()
    out[(out == 0) & (max_iou >= iou_thresh)] = -1
    return out


def transform_annotations_with_ignore(
    mapper: DatasetMapper,
    dataset_dict: dict,
    transforms,
    image_shape: tuple[int, int],
) -> None:
    """Like DatasetMapper._transform_annotations, but keep crowd boxes as ignore."""
    for anno in dataset_dict["annotations"]:
        if not mapper.use_instance_mask:
            anno.pop("segmentation", None)
        if not mapper.use_keypoint:
            anno.pop("keypoints", None)

    raw_annos = dataset_dict.pop("annotations")
    keep = [obj for obj in raw_annos if obj.get("iscrowd", 0) == 0]
    crowd = [obj for obj in raw_annos if obj.get("iscrowd", 0) != 0]

    annos = [
        utils.transform_instance_annotations(
            obj,
            transforms,
            image_shape,
            keypoint_hflip_indices=mapper.keypoint_hflip_indices,
        )
        for obj in keep
    ]
    instances = utils.annotations_to_instances(
        annos, image_shape, mask_format=mapper.instance_mask_format
    )
    if mapper.recompute_boxes and instances.has("gt_masks"):
        instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
    instances = utils.filter_empty_instances(instances)

    crowd_transformed = [
        utils.transform_instance_annotations(
            obj,
            transforms,
            image_shape,
            keypoint_hflip_indices=mapper.keypoint_hflip_indices,
        )
        for obj in crowd
    ]
    # Stored on the dict (not Instances): lengths need not match GT count.
    dataset_dict["ignore_boxes"] = _boxes_xyxy_from_annos(crowd_transformed, image_shape)
    dataset_dict["instances"] = instances


class CrowdAwareDatasetMapper(DatasetMapper):
    """Standard DatasetMapper that preserves ``iscrowd`` as ``ignore_boxes``.

    Training uses the shared boulder aug stack (full-circle rotation, flips,
    scale jitter, coastal photometric) from ``boulder_augmentations`` so both
    the RGB crowd-ignore path and RGB+DSM ``FourBandDatasetMapper`` stay aligned.
    """

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = super().from_config(cfg, is_train)
        # Rich augs are on by default; set INPUT.BOULDER_RICH_AUG=False to fall
        # back to Detectron2's ResizeShortestEdge (+ optional crop/flip).
        use_rich = bool(getattr(cfg.INPUT, "BOULDER_RICH_AUG", True))
        if use_rich:
            from boulder_augmentations import build_boulder_test_augs, build_boulder_train_augs

            image_size = int(cfg.INPUT.MAX_SIZE_TRAIN if is_train else cfg.INPUT.MAX_SIZE_TEST)
            if is_train:
                ret["augmentations"] = build_boulder_train_augs(
                    image_size,
                    scale_min=float(getattr(cfg.INPUT, "BOULDER_SCALE_MIN", 0.5)),
                    scale_max=float(getattr(cfg.INPUT, "BOULDER_SCALE_MAX", 1.5)),
                )
            else:
                ret["augmentations"] = build_boulder_test_augs(image_size)
        return ret

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        transform_annotations_with_ignore(self, dataset_dict, transforms, image_shape)


@META_ARCH_REGISTRY.register()
class GeneralizedRCNNWithIgnore(GeneralizedRCNN):
    """Attach per-image ignore boxes onto GT Instances before RPN / ROI heads."""

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = []
            for x in batched_inputs:
                inst = x["instances"].to(self.device)
                if "ignore_boxes" in x and x["ignore_boxes"] is not None:
                    ib = x["ignore_boxes"].to(self.device)
                else:
                    ib = Boxes(torch.zeros((0, 4), dtype=torch.float32, device=self.device))
                # Private attr: Instances forbids unequally-sized public fields.
                object.__setattr__(inst, "_ignore_boxes", ib)
                gt_instances.append(inst)
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses


@PROPOSAL_GENERATOR_REGISTRY.register()
class RPNWithIgnore(RPN):
    """RPN that ignores anchors overlapping ``gt_instances._ignore_boxes``."""

    @torch.jit.unused
    @torch.no_grad()
    def label_and_sample_anchors(
        self, anchors: List[Boxes], gt_instances: List[Instances]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        anchors_cat = Boxes.cat(anchors)

        gt_boxes = [x.gt_boxes for x in gt_instances]
        ignore_boxes = [get_ignore_boxes(x) for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]

        gt_labels = []
        matched_gt_boxes = []
        for image_size_i, gt_boxes_i, ignore_i in zip(
            image_sizes, gt_boxes, ignore_boxes
        ):
            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(
                gt_boxes_i, anchors_cat
            )
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(
                match_quality_matrix
            )
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

            if self.anchor_boundary_thresh >= 0:
                anchors_inside_image = anchors_cat.inside_box(
                    image_size_i, self.anchor_boundary_thresh
                )
                gt_labels_i[~anchors_inside_image] = -1

            # Apply ignore BEFORE subsampling so ignored anchors are not drawn
            # as negatives into the RPN mini-batch.
            if len(ignore_i):
                ignore_i = ignore_i.to(device=gt_boxes_i.device)
                gt_labels_i = mark_ignore_overlaps(gt_labels_i, anchors_cat, ignore_i)

            gt_labels_i = self._subsample_labels(gt_labels_i)

            if len(gt_boxes_i) == 0:
                matched_gt_boxes_i = torch.zeros_like(anchors_cat.tensor)
            else:
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)
        return gt_labels, matched_gt_boxes


@ROI_HEADS_REGISTRY.register()
class StandardROIHeadsWithIgnore(StandardROIHeads):
    """ROI heads that ignore proposals overlapping ``targets._ignore_boxes``."""

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        from detectron2.modeling.proposal_generator.proposal_utils import (
            add_ground_truth_to_proposals,
        )

        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []
        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)

            ignore_i = get_ignore_boxes(targets_per_image)
            if len(ignore_i):
                matched_labels = mark_ignore_overlaps(
                    matched_labels,
                    proposals_per_image.proposal_boxes,
                    ignore_i,
                )

            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for trg_name, trg_value in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", float(np.mean(num_fg_samples)))
        storage.put_scalar("roi_head/num_bg_samples", float(np.mean(num_bg_samples)))
        return proposals_with_gt

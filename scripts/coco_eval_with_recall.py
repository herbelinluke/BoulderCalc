"""COCO evaluator that persists Average Recall (AR*) alongside AP metrics."""

from __future__ import annotations

from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import create_small_table

# pycocotools COCOeval.stats indices 6-11 (bbox / segm).
_AR_METRICS = ("AR1", "AR10", "AR100", "ARs", "ARm", "ARl")


class BoulderCOCOEvaluator(COCOEvaluator):
    """Like Detectron2's COCOEvaluator, but also saves COCO Average Recall keys.

    Stock Detectron2 only returns AP / AP50 / AP75 / APs / APm / APl. COCO already
    computes AR@1 / AR@10 / AR@100 and size-binned AR; this subclass includes them
    in the dict written to metrics.json / metrics_valid.json / metrics_test.json.
    """

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        results = super()._derive_coco_results(coco_eval, iou_type, class_names)

        if coco_eval is None:
            results.update({metric: float("nan") for metric in _AR_METRICS})
            return results

        # Keypoints use a shorter stats vector; only bbox/segm expose AR* at 6:12.
        if iou_type not in ("bbox", "segm") or len(coco_eval.stats) < 12:
            return results

        ar_results = {
            metric: float(coco_eval.stats[idx + 6] * 100 if coco_eval.stats[idx + 6] >= 0 else "nan")
            for idx, metric in enumerate(_AR_METRICS)
        }
        self._logger.info(
            "Average Recall results for {}: \n".format(iou_type) + create_small_table(ar_results)
        )
        results.update(ar_results)
        return results

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class PriorAwareQCConfig:
    min_prediction_voxels: int = 1
    min_overlap_with_coarse: float = 0.05
    max_volume_ratio_vs_coarse: float = 8.0
    min_volume_ratio_vs_coarse: float = 0.05
    fallback_to_coarse: bool = True


def _centroid(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.full(3, np.nan, dtype=np.float32)
    return coords.mean(axis=0).astype(np.float32)


def binary_prior_qc(
    prediction: np.ndarray,
    coarse: np.ndarray,
    config: PriorAwareQCConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Conservative quality-control hook for calibrated inference.

    It is intentionally not wired into the trainer: thresholds should be
    selected on validation folds before applying it to Gold Atlas.
    """

    pred = np.asarray(prediction > 0, dtype=bool)
    prior = np.asarray(coarse > 0, dtype=bool)
    pred_voxels = int(pred.sum())
    prior_voxels = int(prior.sum())
    overlap = int(np.logical_and(pred, prior).sum())
    overlap_ratio = overlap / max(pred_voxels, 1)
    volume_ratio = pred_voxels / max(prior_voxels, 1)
    failed = (
        pred_voxels < config.min_prediction_voxels
        or overlap_ratio < config.min_overlap_with_coarse
        or volume_ratio > config.max_volume_ratio_vs_coarse
        or volume_ratio < config.min_volume_ratio_vs_coarse
    )
    metrics = {
        "pred_voxels": float(pred_voxels),
        "coarse_voxels": float(prior_voxels),
        "overlap_ratio": float(overlap_ratio),
        "volume_ratio_vs_coarse": float(volume_ratio),
        "pred_centroid_z": float(_centroid(pred)[0]),
        "coarse_centroid_z": float(_centroid(prior)[0]),
        "qc_failed": float(failed),
    }
    if failed and config.fallback_to_coarse:
        return prior.astype(prediction.dtype), metrics
    return pred.astype(prediction.dtype), metrics

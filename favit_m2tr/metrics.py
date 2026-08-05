from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def video_level_metrics(
    frame_probabilities: list[float], frame_labels: list[int], video_ids: list[str]
) -> dict[str, float | int]:
    if not (len(frame_probabilities) == len(frame_labels) == len(video_ids)):
        raise ValueError("probabilities, labels, and video_ids must have equal length")
    grouped_probabilities: dict[str, list[float]] = defaultdict(list)
    grouped_labels: dict[str, set[int]] = defaultdict(set)
    for probability, label, video_id in zip(
        frame_probabilities, frame_labels, video_ids, strict=True
    ):
        grouped_probabilities[video_id].append(float(probability))
        grouped_labels[video_id].add(int(label))
    inconsistent = [key for key, labels in grouped_labels.items() if len(labels) != 1]
    if inconsistent:
        raise ValueError(f"videos have inconsistent labels: {inconsistent[:3]}")

    ordered_ids = sorted(grouped_probabilities)
    probabilities = np.asarray(
        [np.mean(grouped_probabilities[key]) for key in ordered_ids], dtype=np.float64
    )
    labels = np.asarray([next(iter(grouped_labels[key])) for key in ordered_ids])
    if np.unique(labels).size != 2:
        raise ValueError("AUC requires both real and fake videos")
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "video_auc": float(roc_auc_score(labels, probabilities)),
        "video_accuracy": float(accuracy_score(labels, predictions)),
        "num_videos": len(ordered_ids),
        "num_frames": len(frame_probabilities),
    }


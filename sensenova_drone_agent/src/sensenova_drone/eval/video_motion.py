from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import imageio.v3 as iio
import numpy as np


def load_video_frames(video_path: str, max_frames: int | None = None):
    frames = np.asarray(iio.imread(video_path))
    if frames.ndim == 3:
        frames = frames[None, ...]
    if max_frames is not None:
        frames = frames[:max_frames]
    return frames


def compute_frame_difference_stats(frames) -> dict[str, Any]:
    frames = np.asarray(frames)
    if len(frames) < 2:
        return {
            "mean_abs_frame_diff": 0.0,
            "median_abs_frame_diff": 0.0,
            "max_abs_frame_diff": 0.0,
            "static_frame_fraction": 1.0,
            "first_last_abs_diff": 0.0,
        }

    diffs = np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16))
    per_frame = diffs.mean(axis=(1, 2, 3))
    first_last = float(np.abs(frames[-1].astype(np.int16) - frames[0].astype(np.int16)).mean())
    static_threshold = max(0.5, float(np.percentile(per_frame, 25)))
    return {
        "mean_abs_frame_diff": float(per_frame.mean()),
        "median_abs_frame_diff": float(np.median(per_frame)),
        "max_abs_frame_diff": float(per_frame.max()),
        "static_frame_fraction": float(np.mean(per_frame <= static_threshold)),
        "first_last_abs_diff": first_last,
    }


def compute_optical_flow_stats(frames) -> dict[str, Any]:
    frames = np.asarray(frames)
    if len(frames) < 2:
        return {
            "mean_optical_flow_magnitude": 0.0,
            "median_optical_flow_magnitude": 0.0,
            "max_optical_flow_magnitude": 0.0,
        }

    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    magnitudes: list[float] = []
    for prev, nxt in zip(gray_frames[:-1], gray_frames[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            prev,
            nxt,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitudes.append(float(np.mean(mag)))

    if not magnitudes:
        return {
            "mean_optical_flow_magnitude": 0.0,
            "median_optical_flow_magnitude": 0.0,
            "max_optical_flow_magnitude": 0.0,
        }

    values = np.asarray(magnitudes, dtype=np.float32)
    return {
        "mean_optical_flow_magnitude": float(values.mean()),
        "median_optical_flow_magnitude": float(np.median(values)),
        "max_optical_flow_magnitude": float(values.max()),
    }


def estimate_motion_strength(video_path: str) -> dict[str, Any]:
    frames = load_video_frames(video_path)
    diff_stats = compute_frame_difference_stats(frames)
    flow_stats = compute_optical_flow_stats(frames)
    return {
        "video_path": str(Path(video_path)),
        "num_frames": int(len(frames)),
        **diff_stats,
        **flow_stats,
    }

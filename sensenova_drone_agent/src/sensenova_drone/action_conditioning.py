from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from PIL import Image, ImageFilter

from sensenova_drone.actions import DiscreteDroneAction, coerce_discrete_action


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class ActionSeedVideoResult:
    frame_dir: str
    num_frames: int
    metadata: dict


def build_action_seed_video(
    image: Image.Image,
    actions: list[DiscreteDroneAction],
    out_dir: str | Path,
    num_frames: int,
    max_pan_fraction: float = 0.08,
    max_zoom_fraction: float = 0.08,
) -> ActionSeedVideoResult:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    source = image.convert("RGB")
    width, height = source.size
    motion = _resolve_motion_profile(actions, width, height, max_pan_fraction, max_zoom_fraction)

    for frame_index in range(num_frames):
        progress = 0.0 if num_frames <= 1 else frame_index / float(num_frames - 1)
        eased = _ease_in_out(progress)
        pan_x = motion["pan_x_px"] * eased
        pan_y = motion["pan_y_px"] * eased
        zoom = 1.0 + motion["zoom_delta"] * eased
        frame = _render_motion_frame(source, pan_x=pan_x, pan_y=pan_y, zoom=zoom)
        frame.save(out_path / f"frame_{frame_index:04d}.png")

    metadata = {
        "actions": [coerce_discrete_action(action).value for action in actions],
        "num_frames": num_frames,
        "motion_profile": motion,
    }
    return ActionSeedVideoResult(
        frame_dir=str(out_path),
        num_frames=num_frames,
        metadata=metadata,
    )


def _resolve_motion_profile(
    actions: list[DiscreteDroneAction],
    width: int,
    height: int,
    max_pan_fraction: float,
    max_zoom_fraction: float,
) -> dict:
    pan_x = 0.0
    pan_y = 0.0
    zoom_delta = 0.0

    for raw_action in actions:
        action = coerce_discrete_action(raw_action)
        if action == DiscreteDroneAction.YAW_LEFT:
            pan_x += width * max_pan_fraction
        elif action == DiscreteDroneAction.YAW_RIGHT:
            pan_x -= width * max_pan_fraction
        elif action == DiscreteDroneAction.ASCEND:
            pan_y += height * max_pan_fraction
        elif action == DiscreteDroneAction.DESCEND:
            pan_y -= height * max_pan_fraction
        elif action == DiscreteDroneAction.FORWARD:
            zoom_delta += max_zoom_fraction
        elif action == DiscreteDroneAction.BACKWARD:
            zoom_delta -= max_zoom_fraction * 0.5
        elif action == DiscreteDroneAction.STRAFE_LEFT:
            pan_x += width * max_pan_fraction * 0.5
        elif action == DiscreteDroneAction.STRAFE_RIGHT:
            pan_x -= width * max_pan_fraction * 0.5

    return {
        "pan_x_px": float(pan_x),
        "pan_y_px": float(pan_y),
        "zoom_delta": float(zoom_delta),
    }


def _ease_in_out(value: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * value)


def _render_motion_frame(image: Image.Image, pan_x: float, pan_y: float, zoom: float) -> Image.Image:
    if zoom >= 1.0:
        return _render_zoomed_crop(image, pan_x=pan_x, pan_y=pan_y, zoom=zoom)
    return _render_zoomed_out(image, pan_x=pan_x, pan_y=pan_y, zoom=zoom)


def _render_zoomed_crop(image: Image.Image, pan_x: float, pan_y: float, zoom: float) -> Image.Image:
    width, height = image.size
    crop_w = max(1.0, width / zoom)
    crop_h = max(1.0, height / zoom)

    center_x = width / 2.0 - pan_x / max(zoom, 1e-6)
    center_y = height / 2.0 - pan_y / max(zoom, 1e-6)

    left = _clamp(center_x - crop_w / 2.0, 0.0, max(0.0, width - crop_w))
    top = _clamp(center_y - crop_h / 2.0, 0.0, max(0.0, height - crop_h))
    right = left + crop_w
    bottom = top + crop_h

    crop_box = tuple(int(round(value)) for value in (left, top, right, bottom))
    cropped = image.crop(crop_box)
    return cropped.resize((width, height), Image.Resampling.BICUBIC)


def _render_zoomed_out(image: Image.Image, pan_x: float, pan_y: float, zoom: float) -> Image.Image:
    width, height = image.size
    inner_w = max(1, int(round(width * zoom)))
    inner_h = max(1, int(round(height * zoom)))
    resized = image.resize((inner_w, inner_h), Image.Resampling.BICUBIC)

    background = image.resize((width, height), Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(radius=6))
    offset_x = int(round((width - inner_w) / 2.0 + pan_x))
    offset_y = int(round((height - inner_h) / 2.0 + pan_y))
    background.paste(resized, (offset_x, offset_y))
    return background


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))

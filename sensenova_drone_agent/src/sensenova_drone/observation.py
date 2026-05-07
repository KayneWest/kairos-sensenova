from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class Pose:
    position_xyz: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    frame_id: str | None = None
    timestamp_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Pose":
        position = value.get("position_xyz") or value.get("position") or value.get("xyz")
        orientation = value.get("orientation_xyzw") or value.get("orientation") or value.get("xyzw")

        if position is None:
            position = (
                float(value.get("x", 0.0)),
                float(value.get("y", 0.0)),
                float(value.get("z", 0.0)),
            )

        if orientation is None:
            orientation = (
                float(value.get("qx", 0.0)),
                float(value.get("qy", 0.0)),
                float(value.get("qz", 0.0)),
                float(value.get("qw", 1.0)),
            )

        return cls(
            position_xyz=tuple(float(v) for v in position),
            orientation_xyzw=tuple(float(v) for v in orientation),
            frame_id=value.get("frame_id"),
            timestamp_s=_maybe_float(value.get("timestamp_s")),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    frame_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CameraIntrinsics":
        return cls(
            width=int(value.get("width")),
            height=int(value.get("height")),
            fx=_maybe_float(value.get("fx")),
            fy=_maybe_float(value.get("fy")),
            cx=_maybe_float(value.get("cx")),
            cy=_maybe_float(value.get("cy")),
            frame_id=value.get("frame_id"),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass
class Observation:
    frame_rgb: Any
    timestamp_s: float | None = None
    pose: Pose | None = None
    intrinsics: CameraIntrinsics | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)

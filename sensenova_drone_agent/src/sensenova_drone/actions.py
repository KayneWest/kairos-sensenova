from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DiscreteDroneAction(str, Enum):
    HOVER = "hover"
    YAW_LEFT = "yaw_left"
    YAW_RIGHT = "yaw_right"
    ASCEND = "ascend"
    DESCEND = "descend"
    FORWARD = "forward"
    BACKWARD = "backward"
    STRAFE_LEFT = "strafe_left"
    STRAFE_RIGHT = "strafe_right"


@dataclass(frozen=True)
class DroneCommand:
    forward_m_s: float = 0.0
    right_m_s: float = 0.0
    down_m_s: float = 0.0
    yawspeed_deg_s: float = 0.0
    duration_s: float = 0.5
    source_action: DiscreteDroneAction | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


_DEFAULT_ACTION_COMMANDS: dict[DiscreteDroneAction, dict[str, float]] = {
    DiscreteDroneAction.HOVER: {
        "forward_m_s": 0.0,
        "right_m_s": 0.0,
        "down_m_s": 0.0,
        "yawspeed_deg_s": 0.0,
        "duration_s": 0.5,
    },
    DiscreteDroneAction.YAW_LEFT: {
        "forward_m_s": 0.0,
        "right_m_s": 0.0,
        "down_m_s": 0.0,
        "yawspeed_deg_s": -5.0,
        "duration_s": 0.5,
    },
    DiscreteDroneAction.YAW_RIGHT: {
        "forward_m_s": 0.0,
        "right_m_s": 0.0,
        "down_m_s": 0.0,
        "yawspeed_deg_s": 5.0,
        "duration_s": 0.5,
    },
    DiscreteDroneAction.ASCEND: {
        "forward_m_s": 0.0,
        "right_m_s": 0.0,
        "down_m_s": -0.3,
        "yawspeed_deg_s": 0.0,
        "duration_s": 0.5,
    },
    DiscreteDroneAction.DESCEND: {
        "forward_m_s": 0.0,
        "right_m_s": 0.0,
        "down_m_s": 0.3,
        "yawspeed_deg_s": 0.0,
        "duration_s": 0.5,
    },
    DiscreteDroneAction.FORWARD: {
        "forward_m_s": 0.3,
        "right_m_s": 0.0,
        "down_m_s": 0.0,
        "yawspeed_deg_s": 0.0,
        "duration_s": 0.5,
    },
    DiscreteDroneAction.BACKWARD: {
        "forward_m_s": -0.3,
        "right_m_s": 0.0,
        "down_m_s": 0.0,
        "yawspeed_deg_s": 0.0,
        "duration_s": 0.5,
    },
    DiscreteDroneAction.STRAFE_LEFT: {
        "forward_m_s": 0.0,
        "right_m_s": -0.3,
        "down_m_s": 0.0,
        "yawspeed_deg_s": 0.0,
        "duration_s": 0.5,
    },
    DiscreteDroneAction.STRAFE_RIGHT: {
        "forward_m_s": 0.0,
        "right_m_s": 0.3,
        "down_m_s": 0.0,
        "yawspeed_deg_s": 0.0,
        "duration_s": 0.5,
    },
}

_DEFAULT_PROMPT_SUFFIXES: dict[DiscreteDroneAction, str] = {
    DiscreteDroneAction.HOVER: "The camera remains mostly stable.",
    DiscreteDroneAction.YAW_LEFT: "The camera slowly yaws left.",
    DiscreteDroneAction.YAW_RIGHT: "The camera slowly yaws right.",
    DiscreteDroneAction.ASCEND: "The camera rises slightly.",
    DiscreteDroneAction.DESCEND: "The camera descends slightly.",
    DiscreteDroneAction.FORWARD: "The camera moves slowly forward.",
    DiscreteDroneAction.BACKWARD: "The camera moves slowly backward.",
    DiscreteDroneAction.STRAFE_LEFT: "The camera moves slowly left.",
    DiscreteDroneAction.STRAFE_RIGHT: "The camera moves slowly right.",
}

_SEQUENCE_PROMPT_SUFFIXES: dict[tuple[str, ...], str] = {
    (DiscreteDroneAction.FORWARD.value, DiscreteDroneAction.YAW_LEFT.value): (
        "The camera moves slowly forward while yawing left."
    ),
    (DiscreteDroneAction.FORWARD.value, DiscreteDroneAction.YAW_RIGHT.value): (
        "The camera moves slowly forward while yawing right."
    ),
}


def coerce_discrete_action(value: str | DiscreteDroneAction) -> DiscreteDroneAction:
    if isinstance(value, DiscreteDroneAction):
        return value

    normalized = str(value).strip().lower()
    for action in DiscreteDroneAction:
        if normalized == action.value:
            return action

    raise ValueError(f"Unsupported discrete drone action: {value!r}")


def discrete_to_command(
    action: DiscreteDroneAction,
    cfg: dict[str, Any] | None = None,
) -> DroneCommand:
    cfg = cfg or {}
    action = coerce_discrete_action(action)
    defaults = dict(_DEFAULT_ACTION_COMMANDS[action])
    overrides = cfg.get(action.value, {})

    merged = {
        "forward_m_s": float(overrides.get("forward_m_s", defaults["forward_m_s"])),
        "right_m_s": float(overrides.get("right_m_s", defaults["right_m_s"])),
        "down_m_s": float(overrides.get("down_m_s", defaults["down_m_s"])),
        "yawspeed_deg_s": float(overrides.get("yawspeed_deg_s", defaults["yawspeed_deg_s"])),
        "duration_s": float(overrides.get("duration_s", defaults["duration_s"])),
    }

    return DroneCommand(source_action=action, **merged)


def action_to_kairos_prompt_suffix(action: DiscreteDroneAction) -> str:
    action = coerce_discrete_action(action)
    return _DEFAULT_PROMPT_SUFFIXES[action]


def actions_to_kairos_prompt_suffix(actions: list[DiscreteDroneAction]) -> str:
    coerced = [coerce_discrete_action(action) for action in actions]
    key = tuple(action.value for action in coerced)
    if key in _SEQUENCE_PROMPT_SUFFIXES:
        return _SEQUENCE_PROMPT_SUFFIXES[key]
    return " ".join(_DEFAULT_PROMPT_SUFFIXES[action] for action in coerced)


def build_action_cfg(
    *,
    duration_s: float,
    forward_m_s: float = 0.3,
    strafe_m_s: float = 0.3,
    vertical_m_s: float = 0.3,
    yawspeed_deg_s: float = 5.0,
) -> dict[str, dict[str, float]]:
    duration = float(duration_s)
    forward = float(abs(forward_m_s))
    strafe = float(abs(strafe_m_s))
    vertical = float(abs(vertical_m_s))
    yaw = float(abs(yawspeed_deg_s))
    return {
        DiscreteDroneAction.HOVER.value: {
            "duration_s": duration,
        },
        DiscreteDroneAction.YAW_LEFT.value: {
            "duration_s": duration,
            "yawspeed_deg_s": -yaw,
        },
        DiscreteDroneAction.YAW_RIGHT.value: {
            "duration_s": duration,
            "yawspeed_deg_s": yaw,
        },
        DiscreteDroneAction.ASCEND.value: {
            "duration_s": duration,
            "down_m_s": -vertical,
        },
        DiscreteDroneAction.DESCEND.value: {
            "duration_s": duration,
            "down_m_s": vertical,
        },
        DiscreteDroneAction.FORWARD.value: {
            "duration_s": duration,
            "forward_m_s": forward,
        },
        DiscreteDroneAction.BACKWARD.value: {
            "duration_s": duration,
            "forward_m_s": -forward,
        },
        DiscreteDroneAction.STRAFE_LEFT.value: {
            "duration_s": duration,
            "right_m_s": -strafe,
        },
        DiscreteDroneAction.STRAFE_RIGHT.value: {
            "duration_s": duration,
            "right_m_s": strafe,
        },
    }

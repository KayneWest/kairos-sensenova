from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

from sensenova_drone.actions import DroneCommand
from sensenova_drone.planner import CandidatePlan
from sensenova_drone.policy import PolicyOutput
from sensenova_drone.world_state import ObservationEncoding, WorldState


class TelemetryLogger:
    def __init__(self, root_dir: str | Path, image_ext: str = "png"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.image_ext = image_ext
        self._step_counter = 0
        self._active_step_dir: Path | None = None

    def save_real_frame(self, observation) -> str:
        if observation.frame_rgb is None:
            raise RuntimeError("Observation.frame_rgb is required to save a real frame.")

        step_dir = self._ensure_step_dir()
        frame_path = step_dir / f"real_frame.{self.image_ext}"

        frame = observation.frame_rgb
        if hasattr(frame, "save"):
            frame.save(frame_path)
        elif Image is not None:
            Image.fromarray(frame).save(frame_path)
        else:
            raise RuntimeError(
                "Pillow is required to save numpy-like frames. "
                "Install pillow or pass a frame object with a .save() method."
            )

        observation.metadata["frame_path"] = str(frame_path)
        return str(frame_path)

    def make_step_dir(self) -> str:
        return str(self._ensure_step_dir())

    def log_step(
        self,
        observation,
        world_state: WorldState,
        plan,
        executed_command: DroneCommand,
    ) -> str:
        step_dir = self._ensure_step_dir()
        telemetry_path = step_dir / "telemetry.json"

        proposed_command = getattr(plan, "proposed_command", executed_command)
        payload = {
            "step": self._step_counter,
            "real_frame_path": observation.metadata.get("frame_path") or world_state.encoding.frame_path,
            "pose_T_t": _pose_to_dict(world_state.pose),
            "intrinsics_K_t": _intrinsics_to_dict(world_state.intrinsics),
            "observation_encoding": _encoding_to_dict(world_state.encoding),
            "memory_size": world_state.memory_size,
            "candidate_action_sequences": _candidate_action_sequences(plan),
            "candidate_rollouts": _candidate_rollouts(plan),
            "decision_rule": _decision_rule(plan),
            "chosen_action": _chosen_action(plan),
            "proposed_command": _command_to_dict(proposed_command),
            "executed_command": _command_to_dict(executed_command),
            "safety_override": executed_command != proposed_command,
            "generated_rollouts_used_as_state": False,
        }

        telemetry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._active_step_dir = None
        return str(telemetry_path)

    def _ensure_step_dir(self) -> Path:
        if self._active_step_dir is None:
            self._step_counter += 1
            self._active_step_dir = self.root_dir / f"step_{self._step_counter:06d}"
            self._active_step_dir.mkdir(parents=True, exist_ok=True)
        return self._active_step_dir


def _command_to_dict(command: DroneCommand | None) -> dict[str, Any] | None:
    if command is None:
        return None
    return {
        "forward_m_s": command.forward_m_s,
        "right_m_s": command.right_m_s,
        "down_m_s": command.down_m_s,
        "yawspeed_deg_s": command.yawspeed_deg_s,
        "duration_s": command.duration_s,
    }


def _encoding_to_dict(encoding: ObservationEncoding) -> dict[str, Any]:
    return {
        "latent_available": encoding.latent is not None,
        "image_features_available": encoding.image_features is not None,
        "backend": encoding.metadata.get("backend"),
        **encoding.metadata,
    }


def _candidate_action_sequences(plan) -> list[list[str]]:
    if isinstance(plan, CandidatePlan):
        summaries = plan.diagnostics.get("all_candidates", [])
        if summaries:
            return [summary.get("action_sequence", []) for summary in summaries]
        return [[action.value for action in plan.action_sequence.actions]]

    if isinstance(plan, PolicyOutput):
        return [[plan.action.value]]

    return []


def _candidate_rollouts(plan) -> list[dict[str, Any]]:
    if isinstance(plan, CandidatePlan):
        return list(plan.diagnostics.get("all_candidates", []))
    if isinstance(plan, PolicyOutput):
        return []
    return []


def _decision_rule(plan) -> str:
    if isinstance(plan, CandidatePlan):
        return plan.diagnostics.get(
            "decision_rule",
            "argmax_A R(Kairos rollout under candidate action sequence A)",
        )
    if isinstance(plan, PolicyOutput):
        return "argmax(action_logits)"
    return "unknown"


def _chosen_action(plan) -> str | None:
    if isinstance(plan, CandidatePlan):
        if plan.action_sequence.actions:
            return plan.action_sequence.actions[0].value
        return None
    if isinstance(plan, PolicyOutput):
        return plan.action.value
    return None


def _pose_to_dict(value) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "position_xyz": list(value.position_xyz),
        "orientation_xyzw": list(value.orientation_xyzw),
    }


def _intrinsics_to_dict(value) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "fx": value.fx,
        "fy": value.fy,
        "cx": value.cx,
        "cy": value.cy,
        "width": value.width,
        "height": value.height,
    }


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _to_serializable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(inner) for inner in value]
    return value

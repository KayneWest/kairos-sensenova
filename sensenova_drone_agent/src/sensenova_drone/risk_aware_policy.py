from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None

from sensenova_drone.actions import DiscreteDroneAction, DroneCommand, discrete_to_command
from sensenova_drone.bc_data import ACTION_VOCAB


@dataclass
class RiskLabels:
    collision_risk: float
    stall_risk: float
    front_clearance_norm: float
    progress_norm: float
    metadata: dict[str, Any]


@dataclass
class RiskAwarePrediction:
    action: DiscreteDroneAction
    command: DroneCommand
    confidence: float
    probabilities: list[float]
    action_index: int
    collision_risk: float
    stall_risk: float
    front_clearance_m: float
    progress_m: float
    raw_command_prediction: list[float]
    shielded_action: DiscreteDroneAction | None
    shield_reason: str | None
    metadata: dict[str, Any]


if nn is not None:
    class RiskAwareVisualPolicy(nn.Module):
        def __init__(
            self,
            num_actions: int,
            hidden_dim: int = 256,
            goal_feature_dim: int = 4,
            frame_stack: int = 1,
        ):
            super().__init__()
            self.goal_feature_dim = int(goal_feature_dim)
            self.frame_stack = max(1, int(frame_stack))
            input_channels = 3 * self.frame_stack
            self.encoder = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.goal_mlp = None
            trunk_input_dim = hidden_dim
            if self.goal_feature_dim > 0:
                self.goal_mlp = nn.Sequential(
                    nn.Linear(self.goal_feature_dim, 32),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, 32),
                    nn.ReLU(inplace=True),
                )
                trunk_input_dim += 32
            self.trunk = nn.Sequential(
                nn.Linear(trunk_input_dim, hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.action_head = nn.Linear(hidden_dim, num_actions)
            self.command_head = nn.Linear(hidden_dim, 5)
            self.collision_head = nn.Linear(hidden_dim, 1)
            self.stall_head = nn.Linear(hidden_dim, 1)
            self.front_clearance_head = nn.Linear(hidden_dim, 1)
            self.progress_head = nn.Linear(hidden_dim, 1)

        def forward(
            self,
            image: torch.Tensor,
            goal_features: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            features = self.encoder(image)
            features = torch.flatten(features, start_dim=1)
            if self.goal_mlp is not None:
                if goal_features is None:
                    goal_features = torch.zeros(
                        (features.shape[0], self.goal_feature_dim),
                        device=features.device,
                        dtype=features.dtype,
                    )
                features = torch.cat([features, self.goal_mlp(goal_features)], dim=1)
            features = self.trunk(features)
            return {
                "action_logits": self.action_head(features),
                "command_pred": self.command_head(features),
                "collision_logit": self.collision_head(features).squeeze(-1),
                "stall_logit": self.stall_head(features).squeeze(-1),
                "front_clearance_norm": self.front_clearance_head(features).squeeze(-1),
                "progress_norm": self.progress_head(features).squeeze(-1),
            }
else:
    class RiskAwareVisualPolicy:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)
            raise RuntimeError("torch is required to instantiate RiskAwareVisualPolicy.")


class RiskAwarePolicyRunner:
    def __init__(
        self,
        model: RiskAwareVisualPolicy,
        *,
        device: str,
        image_size: int,
        frame_stack: int,
        max_depth_m: float,
        action_vocab: list[str] | None = None,
        action_cfg: dict[str, Any] | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
        shield_collision_threshold: float = 0.55,
        shield_front_clearance_m: float = 1.1,
        shield_enabled: bool = True,
    ):
        if torch is None:
            raise RuntimeError("torch is required to run RiskAwarePolicyRunner.")
        self.model = model
        self.device = device
        self.image_size = int(image_size)
        self.frame_stack = max(1, int(frame_stack))
        self.max_depth_m = float(max_depth_m)
        self.action_vocab = list(action_vocab or ACTION_VOCAB)
        self.action_cfg = action_cfg or {}
        self.checkpoint_metadata = checkpoint_metadata or {}
        self.shield_collision_threshold = float(shield_collision_threshold)
        self.shield_front_clearance_m = float(shield_front_clearance_m)
        self.shield_enabled = bool(shield_enabled)
        self._frame_history = deque(maxlen=max(self.frame_stack - 1, 1))

    def predict(
        self,
        image: Image.Image | np.ndarray | str | Path,
        *,
        goal_features: list[float] | tuple[float, ...] | None = None,
        enabled_actions: list[int] | None = None,
    ) -> RiskAwarePrediction:
        current_frame = _coerce_image(image)
        frame_sequence = self._stack_frames(current_frame)
        image_tensor = _image_sequence_to_tensor(frame_sequence, image_size=self.image_size).to(self.device)
        if goal_features is None:
            goal_features = [0.0, 0.0, 0.0, 0.0]
        goal_tensor = torch.tensor(goal_features, dtype=torch.float32, device=self.device).unsqueeze(0)
        enabled_actions = list(enabled_actions or range(len(self.action_vocab)))

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor, goal_features=goal_tensor)
            probs = torch.softmax(outputs["action_logits"], dim=-1)[0]
            action_index = masked_argmax_tensor(probs, enabled_actions)
            collision_risk = float(torch.sigmoid(outputs["collision_logit"])[0].item())
            stall_risk = float(torch.sigmoid(outputs["stall_logit"])[0].item())
            front_clearance_m = float(
                torch.sigmoid(outputs["front_clearance_norm"])[0].item() * self.max_depth_m
            )
            progress_m = float(outputs["progress_norm"][0].item())
            raw_command = outputs["command_pred"][0].detach().cpu().tolist()

        shielded_index, reason = self._shield_action(
            action_index,
            probs,
            enabled_actions=enabled_actions,
            collision_risk=collision_risk,
            front_clearance_m=front_clearance_m,
        )
        action_name = self.action_vocab[shielded_index]
        action = DiscreteDroneAction(action_name)
        top_indices = torch.argsort(probs, descending=True)[: min(4, probs.shape[0])].detach().cpu().tolist()

        return RiskAwarePrediction(
            action=action,
            command=discrete_to_command(action, self.action_cfg),
            confidence=float(probs[action_index].item()),
            probabilities=[float(value) for value in probs.detach().cpu().tolist()],
            action_index=shielded_index,
            collision_risk=collision_risk,
            stall_risk=stall_risk,
            front_clearance_m=front_clearance_m,
            progress_m=progress_m,
            raw_command_prediction=[float(value) for value in raw_command],
            shielded_action=(action if shielded_index != action_index else None),
            shield_reason=reason,
            metadata={
                "unshielded_action": self.action_vocab[action_index],
                "image_size": self.image_size,
                "frame_stack": self.frame_stack,
                "max_depth_m": self.max_depth_m,
                "top_actions": [
                    {
                        "action": self.action_vocab[index],
                        "probability": float(probs[index].item()),
                    }
                    for index in top_indices
                ],
                **self.checkpoint_metadata,
            },
        )

    def reset_history(self) -> None:
        self._frame_history.clear()

    def _stack_frames(self, current_frame: Image.Image) -> list[Image.Image]:
        frames = list(self._frame_history) + [current_frame]
        while len(frames) < self.frame_stack:
            frames.insert(0, frames[0].copy())
        stacked = [frame.copy() for frame in frames[-self.frame_stack :]]
        if self.frame_stack > 1:
            self._frame_history.append(current_frame.copy())
        return stacked

    def _shield_action(
        self,
        action_index: int,
        probabilities,
        *,
        enabled_actions: list[int],
        collision_risk: float,
        front_clearance_m: float,
    ) -> tuple[int, str | None]:
        if not self.shield_enabled:
            return action_index, None
        action_name = self.action_vocab[action_index]
        risky_forward = action_name == DiscreteDroneAction.FORWARD.value and (
            collision_risk >= self.shield_collision_threshold
            or front_clearance_m <= self.shield_front_clearance_m
        )
        if not risky_forward:
            return action_index, None
        safe_names = {
            DiscreteDroneAction.STRAFE_LEFT.value,
            DiscreteDroneAction.STRAFE_RIGHT.value,
            DiscreteDroneAction.YAW_LEFT.value,
            DiscreteDroneAction.YAW_RIGHT.value,
            DiscreteDroneAction.HOVER.value,
        }
        candidates = [
            index
            for index in enabled_actions
            if self.action_vocab[index] in safe_names
        ]
        if not candidates:
            return action_index, None
        return masked_argmax_tensor(probabilities, candidates), "blocked_forward_risk"


def load_risk_aware_policy_runner(
    checkpoint_path: str | Path,
    *,
    device: str = "auto",
    action_cfg: dict[str, Any] | None = None,
    shield_enabled: bool = True,
    shield_collision_threshold: float | None = None,
    shield_front_clearance_m: float | None = None,
) -> RiskAwarePolicyRunner:
    if torch is None:
        raise RuntimeError("torch is required to load risk-aware checkpoints.")
    resolved_device = _resolve_device(device)
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(checkpoint_file, map_location=resolved_device)
    config = dict(payload.get("config", {}))
    action_vocab = list(payload.get("action_vocab", ACTION_VOCAB))
    model = RiskAwareVisualPolicy(
        num_actions=len(action_vocab),
        goal_feature_dim=int(config.get("goal_feature_dim", 4)),
        frame_stack=int(config.get("frame_stack", 1)),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(resolved_device)
    return RiskAwarePolicyRunner(
        model,
        device=resolved_device,
        image_size=int(config.get("image_size", 96)),
        frame_stack=int(config.get("frame_stack", 1)),
        max_depth_m=float(config.get("max_depth_m", 10.0)),
        action_vocab=action_vocab,
        action_cfg=action_cfg,
        checkpoint_metadata={
            "checkpoint_path": str(checkpoint_file),
            "epoch": int(payload.get("epoch", 0)),
            "metrics": payload.get("metrics"),
        },
        shield_enabled=shield_enabled,
        shield_collision_threshold=(
            float(shield_collision_threshold)
            if shield_collision_threshold is not None
            else float(config.get("shield_collision_threshold", 0.55))
        ),
        shield_front_clearance_m=(
            float(shield_front_clearance_m)
            if shield_front_clearance_m is not None
            else float(config.get("shield_front_clearance_m", 1.1))
        ),
    )


def extract_risk_labels(record: dict[str, Any], *, max_depth_m: float = 10.0) -> RiskLabels:
    metadata = dict(record.get("metadata", {}))
    teacher = dict(metadata.get("teacher", {}))
    goal_features = dict(teacher.get("goal_features", {}))
    current_clearance = dict(teacher.get("depth_clearance_m", {}))
    after = dict(teacher.get("after", {}))
    after_clearance = dict(after.get("clearance_m", {}))
    before_forward = float(goal_features.get("forward_m", 0.0))
    before_right = float(goal_features.get("right_m", 0.0))
    before_distance = math.sqrt(before_forward * before_forward + before_right * before_right)
    after_distance = _maybe_float(after.get("distance_to_goal_m"))
    progress_m = 0.0 if after_distance is None else before_distance - after_distance
    front_m = _maybe_float(current_clearance.get("front_m"))
    if front_m is None:
        front_m = _maybe_float(after_clearance.get("front_m"))
    if front_m is None:
        front_m = float(max_depth_m)
    collision = bool(after.get("collision", False))
    success = bool(after.get("success", False))
    action = str(record.get("action", ""))
    stall = (not collision and not success) and (
        progress_m < 0.015
        or action in {DiscreteDroneAction.HOVER.value, DiscreteDroneAction.ASCEND.value, DiscreteDroneAction.DESCEND.value}
    )
    return RiskLabels(
        collision_risk=1.0 if collision else 0.0,
        stall_risk=1.0 if stall else 0.0,
        front_clearance_norm=float(np.clip(front_m / max(float(max_depth_m), 1e-6), 0.0, 1.0)),
        progress_norm=float(np.clip(progress_m, -1.0, 1.0)),
        metadata={
            "front_clearance_m": float(front_m),
            "progress_m": float(progress_m),
            "before_distance_to_goal_m": float(before_distance),
            "after_distance_to_goal_m": after_distance,
            "collision": collision,
            "success": success,
            "stall": bool(stall),
        },
    )


def masked_argmax_tensor(values, enabled_actions: list[int]) -> int:
    best = int(enabled_actions[0])
    best_value = float("-inf")
    for index in enabled_actions:
        value = float(values[int(index)].item())
        if value > best_value:
            best_value = value
            best = int(index)
    return best


def _resolve_device(requested: str) -> str:
    requested = requested.strip().lower()
    if requested == "auto":
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    return requested


def _coerce_image(image: Image.Image | np.ndarray | str | Path) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")


def _image_to_tensor(image: Image.Image | np.ndarray | str | Path, *, image_size: int):
    if torch is None:
        raise RuntimeError("torch is required to create image tensors.")
    pil_image = _coerce_image(image)
    pil_image = pil_image.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(pil_image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array).unsqueeze(0)


def _image_sequence_to_tensor(images: list[Image.Image], *, image_size: int):
    tensors = [_image_to_tensor(image, image_size=image_size)[0] for image in images]
    return torch.cat(tensors, dim=0).unsqueeze(0)


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)

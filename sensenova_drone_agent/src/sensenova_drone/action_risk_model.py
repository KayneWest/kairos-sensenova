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
class CandidateActionRisk:
    action: DiscreteDroneAction
    action_index: int
    utility: float
    collision_risk: float
    success_prob: float
    out_of_bounds_risk: float
    progress_m: float
    front_clearance_m: float
    reward: float


@dataclass
class ActionRiskDecision:
    action: DiscreteDroneAction
    command: DroneCommand
    action_index: int
    candidates: list[CandidateActionRisk]
    metadata: dict[str, Any]


if nn is not None:
    class ActionRiskVisualScorer(nn.Module):
        def __init__(
            self,
            num_actions: int,
            hidden_dim: int = 256,
            goal_feature_dim: int = 4,
            frame_stack: int = 1,
            action_embed_dim: int = 32,
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
            self.goal_mlp = nn.Sequential(
                nn.Linear(self.goal_feature_dim, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 32),
                nn.ReLU(inplace=True),
            )
            self.action_embedding = nn.Embedding(num_actions, action_embed_dim)
            self.trunk = nn.Sequential(
                nn.Linear(hidden_dim + 32 + action_embed_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.collision_head = nn.Linear(hidden_dim, 1)
            self.success_head = nn.Linear(hidden_dim, 1)
            self.out_of_bounds_head = nn.Linear(hidden_dim, 1)
            self.front_clearance_head = nn.Linear(hidden_dim, 1)
            self.progress_head = nn.Linear(hidden_dim, 1)
            self.reward_head = nn.Linear(hidden_dim, 1)

        def forward(
            self,
            image: torch.Tensor,
            goal_features: torch.Tensor,
            action_index: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            features = torch.flatten(self.encoder(image), start_dim=1)
            goal_embed = self.goal_mlp(goal_features)
            action_embed = self.action_embedding(action_index.long())
            x = self.trunk(torch.cat([features, goal_embed, action_embed], dim=1))
            return {
                "collision_logit": self.collision_head(x).squeeze(-1),
                "success_logit": self.success_head(x).squeeze(-1),
                "out_of_bounds_logit": self.out_of_bounds_head(x).squeeze(-1),
                "front_clearance_norm": self.front_clearance_head(x).squeeze(-1),
                "progress_norm": self.progress_head(x).squeeze(-1),
                "reward_norm": self.reward_head(x).squeeze(-1),
            }
else:
    class ActionRiskVisualScorer:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)
            raise RuntimeError("torch is required to instantiate ActionRiskVisualScorer.")


class ActionRiskPlannerRunner:
    def __init__(
        self,
        model: ActionRiskVisualScorer,
        *,
        device: str,
        image_size: int,
        frame_stack: int,
        max_depth_m: float,
        reward_scale: float,
        action_vocab: list[str] | None = None,
        action_cfg: dict[str, Any] | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
        collision_penalty: float = 8.0,
        out_of_bounds_penalty: float = 5.0,
        success_bonus: float = 5.0,
        progress_weight: float = 2.0,
        clearance_weight: float = 0.5,
        hover_penalty: float = 0.0,
        yaw_penalty: float = 0.0,
    ):
        if torch is None:
            raise RuntimeError("torch is required to run ActionRiskPlannerRunner.")
        self.model = model
        self.device = device
        self.image_size = int(image_size)
        self.frame_stack = max(1, int(frame_stack))
        self.max_depth_m = float(max_depth_m)
        self.reward_scale = float(reward_scale)
        self.action_vocab = list(action_vocab or ACTION_VOCAB)
        self.action_cfg = action_cfg or {}
        self.checkpoint_metadata = checkpoint_metadata or {}
        self.collision_penalty = float(collision_penalty)
        self.out_of_bounds_penalty = float(out_of_bounds_penalty)
        self.success_bonus = float(success_bonus)
        self.progress_weight = float(progress_weight)
        self.clearance_weight = float(clearance_weight)
        self.hover_penalty = float(hover_penalty)
        self.yaw_penalty = float(yaw_penalty)
        self._frame_history = deque(maxlen=max(self.frame_stack - 1, 1))

    def predict(
        self,
        image: Image.Image | np.ndarray | str | Path,
        *,
        goal_features: list[float] | tuple[float, ...] | None = None,
        enabled_actions: list[int] | None = None,
    ) -> ActionRiskDecision:
        current_frame = _coerce_image(image)
        frames = self._stack_frames(current_frame)
        image_tensor = _image_sequence_to_tensor(frames, image_size=self.image_size).to(self.device)
        if goal_features is None:
            goal_features = [0.0, 0.0, 0.0, 0.0]
        enabled_actions = list(enabled_actions or range(len(self.action_vocab)))
        batch_images = image_tensor.repeat(len(enabled_actions), 1, 1, 1)
        goal_tensor = torch.tensor([list(goal_features) for _ in enabled_actions], dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(enabled_actions, dtype=torch.long, device=self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch_images, goal_tensor, action_tensor)
            collision = torch.sigmoid(outputs["collision_logit"]).detach().cpu().numpy()
            success = torch.sigmoid(outputs["success_logit"]).detach().cpu().numpy()
            out_of_bounds = torch.sigmoid(outputs["out_of_bounds_logit"]).detach().cpu().numpy()
            clearance = torch.sigmoid(outputs["front_clearance_norm"]).detach().cpu().numpy() * self.max_depth_m
            progress = outputs["progress_norm"].detach().cpu().numpy()
            reward = outputs["reward_norm"].detach().cpu().numpy() * self.reward_scale

        candidates: list[CandidateActionRisk] = []
        for row, action_index in enumerate(enabled_actions):
            action_name = self.action_vocab[action_index]
            action_penalty = 0.0
            if action_name == DiscreteDroneAction.HOVER.value:
                action_penalty += self.hover_penalty
            if action_name in {DiscreteDroneAction.YAW_LEFT.value, DiscreteDroneAction.YAW_RIGHT.value}:
                action_penalty += self.yaw_penalty
            utility = (
                float(reward[row])
                + self.success_bonus * float(success[row])
                + self.progress_weight * float(progress[row])
                + self.clearance_weight * float(clearance[row] / max(self.max_depth_m, 1e-6))
                - self.collision_penalty * float(collision[row])
                - self.out_of_bounds_penalty * float(out_of_bounds[row])
                - action_penalty
            )
            candidates.append(
                CandidateActionRisk(
                    action=DiscreteDroneAction(self.action_vocab[action_index]),
                    action_index=int(action_index),
                    utility=utility,
                    collision_risk=float(collision[row]),
                    success_prob=float(success[row]),
                    out_of_bounds_risk=float(out_of_bounds[row]),
                    progress_m=float(progress[row]),
                    front_clearance_m=float(clearance[row]),
                    reward=float(reward[row]),
                )
            )
        best = max(candidates, key=lambda item: item.utility)
        return ActionRiskDecision(
            action=best.action,
            command=discrete_to_command(best.action, self.action_cfg),
            action_index=best.action_index,
            candidates=sorted(candidates, key=lambda item: item.utility, reverse=True),
            metadata={
                "image_size": self.image_size,
                "frame_stack": self.frame_stack,
                "max_depth_m": self.max_depth_m,
                "decision_rule": "argmax predicted reward/progress/clearance minus risk",
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


def load_action_risk_planner_runner(
    checkpoint_path: str | Path,
    *,
    device: str = "auto",
    action_cfg: dict[str, Any] | None = None,
    collision_penalty: float | None = None,
    out_of_bounds_penalty: float | None = None,
    success_bonus: float | None = None,
    progress_weight: float | None = None,
    clearance_weight: float | None = None,
    hover_penalty: float | None = None,
    yaw_penalty: float | None = None,
) -> ActionRiskPlannerRunner:
    if torch is None:
        raise RuntimeError("torch is required to load action-risk checkpoints.")
    resolved_device = _resolve_device(device)
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(checkpoint_file, map_location=resolved_device)
    config = dict(payload.get("config", {}))
    action_vocab = list(payload.get("action_vocab", ACTION_VOCAB))
    model = ActionRiskVisualScorer(
        num_actions=len(action_vocab),
        goal_feature_dim=int(config.get("goal_feature_dim", 4)),
        frame_stack=int(config.get("frame_stack", 1)),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(resolved_device)
    return ActionRiskPlannerRunner(
        model,
        device=resolved_device,
        image_size=int(config.get("image_size", 96)),
        frame_stack=int(config.get("frame_stack", 1)),
        max_depth_m=float(config.get("max_depth_m", 10.0)),
        reward_scale=float(config.get("reward_scale", 12.0)),
        action_vocab=action_vocab,
        action_cfg=action_cfg,
        checkpoint_metadata={
            "checkpoint_path": str(checkpoint_file),
            "epoch": int(payload.get("epoch", 0)),
            "metrics": payload.get("metrics"),
        },
        collision_penalty=float(collision_penalty if collision_penalty is not None else config.get("planner_collision_penalty", 8.0)),
        out_of_bounds_penalty=float(out_of_bounds_penalty if out_of_bounds_penalty is not None else config.get("planner_out_of_bounds_penalty", 5.0)),
        success_bonus=float(success_bonus if success_bonus is not None else config.get("planner_success_bonus", 5.0)),
        progress_weight=float(progress_weight if progress_weight is not None else config.get("planner_progress_weight", 2.0)),
        clearance_weight=float(clearance_weight if clearance_weight is not None else config.get("planner_clearance_weight", 0.5)),
        hover_penalty=float(hover_penalty if hover_penalty is not None else config.get("planner_hover_penalty", 0.0)),
        yaw_penalty=float(yaw_penalty if yaw_penalty is not None else config.get("planner_yaw_penalty", 0.0)),
    )


def goal_features_from_info(info: dict[str, Any]) -> list[float]:
    forward, right = info.get("goal_body_xy_m") or [0.0, 0.0]
    heading_error_deg = math.degrees(math.atan2(float(right), max(float(forward), 1e-6)))
    return [
        float(np.clip(float(forward) / 10.0, -2.0, 2.0)),
        float(np.clip(float(right) / 5.0, -2.0, 2.0)),
        0.0,
        float(np.clip(heading_error_deg / 180.0, -1.0, 1.0)),
    ]


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

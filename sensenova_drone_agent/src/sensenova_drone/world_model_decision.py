from __future__ import annotations

from dataclasses import dataclass
from collections import deque
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

from sensenova_drone.action_risk_model import CandidateActionRisk
from sensenova_drone.actions import DiscreteDroneAction, DroneCommand, discrete_to_command
from sensenova_drone.bc_data import ACTION_VOCAB


@dataclass
class WorldModelDecision:
    action: DiscreteDroneAction
    command: DroneCommand
    action_index: int
    candidates: list[CandidateActionRisk]
    policy_probabilities: list[float]
    metadata: dict[str, Any]


if nn is not None:
    class WorldModelDecisionHeads(nn.Module):
        def __init__(
            self,
            world_model,
            *,
            latent_dim: int,
            num_actions: int,
            goal_feature_dim: int = 4,
            hidden_dim: int = 192,
            action_embed_dim: int = 32,
            freeze_world_model: bool = True,
        ):
            super().__init__()
            self.world_model = world_model
            self.latent_dim = int(latent_dim)
            self.num_actions = int(num_actions)
            self.goal_feature_dim = int(goal_feature_dim)
            if freeze_world_model:
                for parameter in self.world_model.parameters():
                    parameter.requires_grad_(False)
            self.state_trunk = nn.Sequential(
                nn.Linear(self.latent_dim + self.goal_feature_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.policy_head = nn.Linear(hidden_dim, self.num_actions)
            self.value_head = nn.Linear(hidden_dim, 1)
            self.action_embedding = nn.Embedding(self.num_actions, action_embed_dim)
            self.action_trunk = nn.Sequential(
                nn.Linear(hidden_dim + action_embed_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.collision_head = nn.Linear(hidden_dim, 1)
            self.success_head = nn.Linear(hidden_dim, 1)
            self.out_of_bounds_head = nn.Linear(hidden_dim, 1)
            self.front_clearance_head = nn.Linear(hidden_dim, 1)
            self.progress_head = nn.Linear(hidden_dim, 1)
            self.utility_head = nn.Linear(hidden_dim, 1)

        def encode_state(self, image, goal_features):
            latent = self.world_model.encode(image)
            return self.state_trunk(torch.cat([latent, goal_features], dim=1))

        def policy(self, image, goal_features):
            state = self.encode_state(image, goal_features)
            return {
                "action_logits": self.policy_head(state),
                "value": self.value_head(state).squeeze(-1),
            }

        def risk(self, image, goal_features, action_index):
            state = self.encode_state(image, goal_features)
            action = self.action_embedding(action_index.long())
            x = self.action_trunk(torch.cat([state, action], dim=1))
            return {
                "collision_logit": self.collision_head(x).squeeze(-1),
                "success_logit": self.success_head(x).squeeze(-1),
                "out_of_bounds_logit": self.out_of_bounds_head(x).squeeze(-1),
                "front_clearance_norm": self.front_clearance_head(x).squeeze(-1),
                "progress_norm": self.progress_head(x).squeeze(-1),
                "utility_norm": self.utility_head(x).squeeze(-1),
            }
else:
    class WorldModelDecisionHeads:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)
            raise RuntimeError("torch is required to instantiate WorldModelDecisionHeads.")


class WorldModelDecisionRunner:
    def __init__(
        self,
        model: WorldModelDecisionHeads,
        *,
        device: str,
        image_width: int,
        image_height: int,
        max_depth_m: float,
        utility_scale: float,
        action_vocab: list[str] | None = None,
        action_cfg: dict[str, Any] | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
        policy_logprob_weight: float = 0.75,
        collision_penalty: float = 8.0,
        out_of_bounds_penalty: float = 5.0,
        success_bonus: float = 5.0,
        progress_weight: float = 3.0,
        clearance_weight: float = 0.4,
        hover_penalty: float = 0.5,
        yaw_penalty: float = 0.05,
    ):
        if torch is None:
            raise RuntimeError("torch is required to run WorldModelDecisionRunner.")
        self.model = model
        self.device = device
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.max_depth_m = float(max_depth_m)
        self.utility_scale = float(utility_scale)
        self.action_vocab = list(action_vocab or ACTION_VOCAB)
        self.action_cfg = action_cfg or {}
        self.checkpoint_metadata = checkpoint_metadata or {}
        self.policy_logprob_weight = float(policy_logprob_weight)
        self.collision_penalty = float(collision_penalty)
        self.out_of_bounds_penalty = float(out_of_bounds_penalty)
        self.success_bonus = float(success_bonus)
        self.progress_weight = float(progress_weight)
        self.clearance_weight = float(clearance_weight)
        self.hover_penalty = float(hover_penalty)
        self.yaw_penalty = float(yaw_penalty)
        self._frame_history = deque(maxlen=1)

    def predict(
        self,
        image: Image.Image | np.ndarray | str | Path,
        *,
        goal_features: list[float] | tuple[float, ...] | None = None,
        enabled_actions: list[int] | None = None,
    ) -> WorldModelDecision:
        pil = _coerce_image(image)
        image_tensor = _image_to_tensor(pil, width=self.image_width, height=self.image_height).to(self.device)
        if goal_features is None:
            goal_features = [0.0, 0.0, 0.0, 0.0]
        goal_tensor = torch.tensor([list(goal_features)], dtype=torch.float32, device=self.device)
        enabled_actions = list(enabled_actions or range(len(self.action_vocab)))

        self.model.eval()
        with torch.no_grad():
            policy_outputs = self.model.policy(image_tensor, goal_tensor)
            logits = policy_outputs["action_logits"][0]
            policy_probs = torch.softmax(logits, dim=-1)
            policy_log_probs = torch.log_softmax(logits, dim=-1)

            batch_images = image_tensor.repeat(len(enabled_actions), 1, 1, 1)
            batch_goals = goal_tensor.repeat(len(enabled_actions), 1)
            action_tensor = torch.tensor(enabled_actions, dtype=torch.long, device=self.device)
            risk_outputs = self.model.risk(batch_images, batch_goals, action_tensor)
            collision = torch.sigmoid(risk_outputs["collision_logit"]).detach().cpu().numpy()
            success = torch.sigmoid(risk_outputs["success_logit"]).detach().cpu().numpy()
            out_of_bounds = torch.sigmoid(risk_outputs["out_of_bounds_logit"]).detach().cpu().numpy()
            clearance = torch.sigmoid(risk_outputs["front_clearance_norm"]).detach().cpu().numpy() * self.max_depth_m
            progress = risk_outputs["progress_norm"].detach().cpu().numpy()
            utility_pred = risk_outputs["utility_norm"].detach().cpu().numpy() * self.utility_scale
            policy_log_probs_np = policy_log_probs.detach().cpu().numpy()

        candidates: list[CandidateActionRisk] = []
        for row, action_index in enumerate(enabled_actions):
            action_name = self.action_vocab[action_index]
            action_penalty = 0.0
            if action_name == DiscreteDroneAction.HOVER.value:
                action_penalty += self.hover_penalty
            if action_name in {DiscreteDroneAction.YAW_LEFT.value, DiscreteDroneAction.YAW_RIGHT.value}:
                action_penalty += self.yaw_penalty
            score = (
                float(utility_pred[row])
                + self.policy_logprob_weight * float(policy_log_probs_np[action_index])
                + self.success_bonus * float(success[row])
                + self.progress_weight * float(progress[row])
                + self.clearance_weight * float(clearance[row] / max(self.max_depth_m, 1e-6))
                - self.collision_penalty * float(collision[row])
                - self.out_of_bounds_penalty * float(out_of_bounds[row])
                - action_penalty
            )
            candidates.append(
                CandidateActionRisk(
                    action=DiscreteDroneAction(action_name),
                    action_index=int(action_index),
                    utility=score,
                    collision_risk=float(collision[row]),
                    success_prob=float(success[row]),
                    out_of_bounds_risk=float(out_of_bounds[row]),
                    progress_m=float(progress[row]),
                    front_clearance_m=float(clearance[row]),
                    reward=float(utility_pred[row]),
                )
            )
        best = max(candidates, key=lambda item: item.utility)
        return WorldModelDecision(
            action=best.action,
            command=discrete_to_command(best.action, self.action_cfg),
            action_index=best.action_index,
            candidates=sorted(candidates, key=lambda item: item.utility, reverse=True),
            policy_probabilities=[float(value) for value in policy_probs.detach().cpu().tolist()],
            metadata={
                "decision_rule": "argmax(world_model_decision_heads policy prior + utility - risk)",
                "image_width": self.image_width,
                "image_height": self.image_height,
                **self.checkpoint_metadata,
            },
        )

    def reset_history(self) -> None:
        self._frame_history.clear()


def load_world_model_decision_runner(
    checkpoint_path: str | Path,
    *,
    device: str = "auto",
    action_cfg: dict[str, Any] | None = None,
    policy_logprob_weight: float | None = None,
    collision_penalty: float | None = None,
    out_of_bounds_penalty: float | None = None,
    success_bonus: float | None = None,
    progress_weight: float | None = None,
    clearance_weight: float | None = None,
    hover_penalty: float | None = None,
    yaw_penalty: float | None = None,
) -> WorldModelDecisionRunner:
    if torch is None:
        raise RuntimeError("torch is required to load world-model decision checkpoints.")
    from scripts.train_gym_drone_game_world_model import ActionConditionedWorldModel

    resolved_device = _resolve_device(device)
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(checkpoint_file, map_location=resolved_device)
    config = dict(payload.get("config", {}))
    wm_config = dict(payload["world_model_config"])
    action_vocab = list(payload.get("action_vocab", ACTION_VOCAB))
    world_model = ActionConditionedWorldModel(
        num_actions=len(action_vocab),
        image_width=int(wm_config["image_width"]),
        image_height=int(wm_config["image_height"]),
        latent_dim=int(wm_config["latent_dim"]),
    ).to(resolved_device)
    state_dict = payload["model_state_dict"]
    world_model_state = {
        key.removeprefix("world_model."): value
        for key, value in state_dict.items()
        if key.startswith("world_model.")
    }
    world_model.load_state_dict(world_model_state)
    model = WorldModelDecisionHeads(
        world_model,
        latent_dim=int(wm_config["latent_dim"]),
        num_actions=len(action_vocab),
        freeze_world_model=True,
    ).to(resolved_device)
    model.load_state_dict(state_dict)
    return WorldModelDecisionRunner(
        model,
        device=resolved_device,
        image_width=int(wm_config["image_width"]),
        image_height=int(wm_config["image_height"]),
        max_depth_m=float(config.get("max_depth_m", 10.0)),
        utility_scale=float(config.get("utility_scale", 12.0)),
        action_vocab=action_vocab,
        action_cfg=action_cfg,
        checkpoint_metadata={
            "checkpoint_path": str(checkpoint_file),
            "epoch": int(payload.get("epoch", 0)),
            "metrics": payload.get("metrics"),
        },
        policy_logprob_weight=float(
            policy_logprob_weight if policy_logprob_weight is not None else config.get("planner_policy_logprob_weight", 0.75)
        ),
        collision_penalty=float(
            collision_penalty if collision_penalty is not None else config.get("planner_collision_penalty", 8.0)
        ),
        out_of_bounds_penalty=float(
            out_of_bounds_penalty if out_of_bounds_penalty is not None else config.get("planner_out_of_bounds_penalty", 5.0)
        ),
        success_bonus=float(
            success_bonus if success_bonus is not None else config.get("planner_success_bonus", 5.0)
        ),
        progress_weight=float(
            progress_weight if progress_weight is not None else config.get("planner_progress_weight", 3.0)
        ),
        clearance_weight=float(
            clearance_weight if clearance_weight is not None else config.get("planner_clearance_weight", 0.4)
        ),
        hover_penalty=float(
            hover_penalty if hover_penalty is not None else config.get("planner_hover_penalty", 0.5)
        ),
        yaw_penalty=float(
            yaw_penalty if yaw_penalty is not None else config.get("planner_yaw_penalty", 0.05)
        ),
    )


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


def _image_to_tensor(image: Image.Image, *, width: int, height: int):
    if torch is None:
        raise RuntimeError("torch is required to create image tensors.")
    image = image.resize((int(width), int(height)), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array).unsqueeze(0)

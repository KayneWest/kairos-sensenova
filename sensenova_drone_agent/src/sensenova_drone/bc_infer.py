from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import torch
except ModuleNotFoundError:
    torch = None

from sensenova_drone.actions import DiscreteDroneAction, DroneCommand, discrete_to_command
from sensenova_drone.bc_data import ACTION_VOCAB
from sensenova_drone.bc_model import ImageBCPolicy


@dataclass
class BCPrediction:
    action: DiscreteDroneAction
    command: DroneCommand
    confidence: float
    probabilities: list[float]
    action_index: int
    raw_command_prediction: list[float]
    metadata: dict[str, Any]


class BCPolicyRunner:
    def __init__(
        self,
        model: ImageBCPolicy,
        *,
        device: str,
        image_size: int,
        frame_stack: int,
        action_vocab: list[str] | None = None,
        action_cfg: dict[str, Any] | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
    ):
        if torch is None:
            raise RuntimeError("torch is required to run BCPolicyRunner.")
        self.model = model
        self.device = device
        self.image_size = int(image_size)
        self.frame_stack = max(1, int(frame_stack))
        self.action_vocab = list(action_vocab or ACTION_VOCAB)
        self.action_cfg = action_cfg or {}
        self.checkpoint_metadata = checkpoint_metadata or {}
        self._frame_history = deque(maxlen=max(self.frame_stack - 1, 1))

    def predict(
        self,
        image: Image.Image | np.ndarray | str | Path,
        *,
        goal_features: list[float] | tuple[float, ...] | None = None,
    ) -> BCPrediction:
        if torch is None:
            raise RuntimeError("torch is required to run BC policy inference.")

        current_frame = _coerce_image(image)
        frame_sequence = self._stack_frames(current_frame)
        image_tensor = _image_sequence_to_tensor(frame_sequence, image_size=self.image_size).to(self.device)
        if goal_features is None:
            goal_features = [0.0, 0.0, 0.0, 0.0]
        goal_tensor = torch.tensor(goal_features, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor, goal_features=goal_tensor)
            logits = outputs["action_logits"]
            probs = torch.softmax(logits, dim=-1)[0]
            action_index = int(torch.argmax(probs).item())
            confidence = float(probs[action_index].item())
            raw_command = outputs["command_pred"][0].detach().cpu().tolist()

        action_name = self.action_vocab[action_index]
        action = DiscreteDroneAction(action_name)
        command = discrete_to_command(action, self.action_cfg)
        top_indices = torch.argsort(probs, descending=True)[: min(3, probs.shape[0])].detach().cpu().tolist()

        return BCPrediction(
            action=action,
            command=command,
            confidence=confidence,
            probabilities=[float(value) for value in probs.detach().cpu().tolist()],
            action_index=action_index,
            raw_command_prediction=[float(value) for value in raw_command],
            metadata={
                "image_size": self.image_size,
                "frame_stack": self.frame_stack,
                "action_vocab": list(self.action_vocab),
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


def load_bc_policy_runner(
    checkpoint_path: str | Path,
    *,
    device: str = "auto",
    action_cfg: dict[str, Any] | None = None,
) -> BCPolicyRunner:
    if torch is None:
        raise RuntimeError("torch is required to load BC checkpoints.")

    resolved_device = _resolve_device(device)
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(checkpoint_file, map_location=resolved_device)

    config = dict(payload.get("config", {}))
    action_vocab = list(payload.get("action_vocab", ACTION_VOCAB))
    goal_feature_dim = 4
    if config.get("goal_feature_dim") is not None:
        goal_feature_dim = int(config["goal_feature_dim"])

    frame_stack = int(config.get("frame_stack", 1))
    model = ImageBCPolicy(
        num_actions=len(action_vocab),
        goal_feature_dim=goal_feature_dim,
        frame_stack=frame_stack,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(resolved_device)

    return BCPolicyRunner(
        model,
        device=resolved_device,
        image_size=int(config.get("image_size", 224)),
        frame_stack=frame_stack,
        action_vocab=action_vocab,
        action_cfg=action_cfg,
        checkpoint_metadata={
            "checkpoint_path": str(checkpoint_file),
            "epoch": int(payload.get("epoch", 0)),
            "metrics": payload.get("metrics"),
        },
    )


def _resolve_device(requested: str) -> str:
    requested = requested.strip().lower()
    if requested == "auto":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"
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
    if torch is None:
        raise RuntimeError("torch is required to create stacked image tensors.")
    tensors = [_image_to_tensor(image, image_size=image_size)[0] for image in images]
    return torch.cat(tensors, dim=0).unsqueeze(0)

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ModuleNotFoundError:
    torch = None
    F = None
    DataLoader = None
    Dataset = None
    WeightedRandomSampler = None

from sensenova_drone.bc_data import ACTION_VOCAB
from sensenova_drone.bc_model import ImageBCPolicy


@dataclass
class BCTrainConfig:
    manifest_path: str
    out_dir: str
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    image_size: int = 224
    command_loss_weight: float = 0.25
    device: str = "auto"
    num_workers: int = 0
    seed: int = 0
    use_class_weights: bool = True
    goal_feature_mode: str = "recorded"
    frame_stack: int = 1
    mirror_lateral_actions: bool = False
    balanced_action_sampler: bool = False


class BCManifestDataset(Dataset if Dataset is not None else object):
    def __init__(
        self,
        records: list[dict[str, Any]],
        image_size: int,
        *,
        goal_feature_mode: str = "recorded",
        frame_stack: int = 1,
        mirror_lateral_actions: bool = False,
    ):
        if torch is None:
            raise RuntimeError("torch is required to instantiate BCManifestDataset.")
        self.records = records
        self.image_size = int(image_size)
        self.goal_feature_mode = str(goal_feature_mode)
        self.frame_stack = max(1, int(frame_stack))
        self._episode_steps = _build_episode_step_index(records)
        self._entries = _build_dataset_entries(records, mirror_lateral_actions=mirror_lateral_actions)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int):
        record, mirrored = self._entries[index]
        image = _load_stacked_image_tensor(
            self._stack_frame_paths(record),
            self.image_size,
            mirror=mirrored,
        )
        action_index = _mirrored_action_index(int(record["action_index"])) if mirrored else int(record["action_index"])
        if self.goal_feature_mode == "zeros":
            goal_features_list = [0.0, 0.0, 0.0, 0.0]
        else:
            goal_features_list = _extract_goal_feature_vector(record, mirrored=mirrored)
        goal_features = torch.tensor(goal_features_list, dtype=torch.float32)
        right_m_s = float(record["command"]["right_m_s"])
        yawspeed_deg_s = float(record["command"]["yawspeed_deg_s"])
        if mirrored:
            right_m_s *= -1.0
            yawspeed_deg_s *= -1.0
        command = torch.tensor(
            [
                float(record["command"]["forward_m_s"]),
                right_m_s,
                float(record["command"]["down_m_s"]),
                yawspeed_deg_s,
                float(record["command"]["duration_s"]),
            ],
            dtype=torch.float32,
        )
        return {
            "image": image,
            "action_index": torch.tensor(action_index, dtype=torch.long),
            "goal_features": goal_features,
            "command": command,
        }

    @property
    def action_indices(self) -> list[int]:
        values: list[int] = []
        for record, mirrored in self._entries:
            action_index = int(record["action_index"])
            values.append(_mirrored_action_index(action_index) if mirrored else action_index)
        return values

    def _stack_frame_paths(self, record: dict[str, Any]) -> list[str]:
        if self.frame_stack <= 1:
            return [str(record["image_path"])]

        episode_id = str(record.get("episode_id", ""))
        step_index = int(record.get("step_index", 0))
        steps = self._episode_steps.get(episode_id, {})
        paths: list[str] = []
        for offset in range(self.frame_stack - 1, -1, -1):
            candidate = steps.get(step_index - offset)
            if candidate is None:
                fallback = steps.get(0) or record
                candidate = fallback
            paths.append(str(candidate["image_path"]))
        return paths


def load_manifest_records(manifest_path: str | Path) -> list[dict[str, Any]]:
    manifest = Path(manifest_path)
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest}")

    records: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def split_manifest_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_records = [record for record in records if record.get("split", "train") == "train"]
    val_records = [record for record in records if record.get("split", "train") == "val"]
    return train_records, val_records


def resolve_device(requested: str) -> str:
    requested = requested.strip().lower()
    if requested == "auto":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return requested


def train_supervised_bc(config: BCTrainConfig) -> dict[str, Any]:
    if torch is None or F is None or DataLoader is None:
        raise RuntimeError(
            "torch is required for BC training. Use a Python environment with torch installed."
        )

    _seed_everything(config.seed)

    records = load_manifest_records(config.manifest_path)
    if not records:
        raise RuntimeError("The manifest is empty. Collect SITL episodes before training.")

    train_records, val_records = split_manifest_records(records)
    if not train_records:
        raise RuntimeError("The manifest has no training records.")

    device = resolve_device(config.device)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = BCManifestDataset(
        train_records,
        config.image_size,
        goal_feature_mode=config.goal_feature_mode,
        frame_stack=config.frame_stack,
        mirror_lateral_actions=config.mirror_lateral_actions,
    )
    train_sampler = None
    if config.balanced_action_sampler:
        train_sampler = _build_balanced_action_sampler(train_dataset.action_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.num_workers,
    )
    val_loader = None
    if val_records:
        val_dataset = BCManifestDataset(
            val_records,
            config.image_size,
            goal_feature_mode=config.goal_feature_mode,
            frame_stack=config.frame_stack,
            mirror_lateral_actions=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

    model = ImageBCPolicy(
        num_actions=len(ACTION_VOCAB),
        goal_feature_dim=4,
        frame_stack=config.frame_stack,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    class_weights = None
    if config.use_class_weights:
        class_weights = _compute_class_weights(train_dataset.action_indices)
        if class_weights is not None:
            class_weights = class_weights.to(device)

    history: list[dict[str, Any]] = []
    best_metric = float("inf")
    best_checkpoint = out_dir / "best.pt"
    last_checkpoint = out_dir / "last.pt"

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            command_loss_weight=config.command_loss_weight,
            optimizer=optimizer,
            class_weights=class_weights,
        )
        val_metrics = None
        if val_loader is not None:
            val_metrics = _run_epoch(
                model,
                val_loader,
                device=device,
                command_loss_weight=config.command_loss_weight,
                optimizer=None,
                class_weights=class_weights,
            )

        epoch_metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_metrics)

        selection_metric = (
            val_metrics["loss"]
            if val_metrics is not None
            else train_metrics["loss"]
        )
        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "action_vocab": ACTION_VOCAB,
            "metrics": epoch_metrics,
        }
        torch.save(checkpoint_payload, last_checkpoint)
        if selection_metric < best_metric:
            best_metric = selection_metric
            torch.save(checkpoint_payload, best_checkpoint)

    summary = {
        "manifest_path": str(Path(config.manifest_path).resolve()),
        "out_dir": str(out_dir.resolve()),
        "device": device,
        "num_train": len(train_records),
        "num_val": len(val_records),
        "action_vocab": ACTION_VOCAB,
        "goal_feature_names": [
            "goal_forward_m_over_10",
            "goal_right_m_over_5",
            "goal_alt_error_m_over_3",
            "goal_heading_error_deg_over_180",
        ],
        "goal_feature_mode": config.goal_feature_mode,
        "frame_stack": config.frame_stack,
        "mirror_lateral_actions": config.mirror_lateral_actions,
        "balanced_action_sampler": config.balanced_action_sampler,
        "class_weights": (
            class_weights.detach().cpu().tolist()
            if class_weights is not None
            else None
        ),
        "best_metric": best_metric,
        "history": history,
        "best_checkpoint": str(best_checkpoint.resolve()),
        "last_checkpoint": str(last_checkpoint.resolve()),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _load_image_tensor(path: str | Path, image_size: int):
    if torch is None:
        raise RuntimeError("torch is required to load image tensors for BC training.")
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


def _load_stacked_image_tensor(paths: list[str | Path], image_size: int, *, mirror: bool = False):
    if torch is None:
        raise RuntimeError("torch is required to load stacked image tensors for BC training.")
    tensors = [_load_image_tensor(path, image_size) for path in paths]
    tensor = torch.cat(tensors, dim=0)
    if mirror:
        tensor = torch.flip(tensor, dims=[2])
    return tensor


def _run_epoch(
    model,
    data_loader,
    *,
    device: str,
    command_loss_weight: float,
    optimizer,
    class_weights,
) -> dict[str, Any]:
    if torch is None or F is None:
        raise RuntimeError("torch is required for BC training.")

    train_mode = optimizer is not None
    model.train(mode=train_mode)

    total_examples = 0
    total_loss = 0.0
    total_action_loss = 0.0
    total_command_loss = 0.0
    total_correct = 0
    total_command_mae = 0.0

    for batch in data_loader:
        images = batch["image"].to(device)
        action_targets = batch["action_index"].to(device)
        goal_features = batch["goal_features"].to(device)
        command_targets = batch["command"].to(device)

        with torch.set_grad_enabled(train_mode):
            outputs = model(images, goal_features=goal_features)
            action_loss = F.cross_entropy(
                outputs["action_logits"],
                action_targets,
                weight=class_weights,
            )
            command_loss = F.mse_loss(outputs["command_pred"], command_targets)
            loss = action_loss + float(command_loss_weight) * command_loss

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = int(images.shape[0])
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_action_loss += float(action_loss.item()) * batch_size
        total_command_loss += float(command_loss.item()) * batch_size
        total_correct += int(
            (torch.argmax(outputs["action_logits"], dim=1) == action_targets).sum().item()
        )
        total_command_mae += float(
            torch.abs(outputs["command_pred"] - command_targets).mean().item()
        ) * batch_size

    return {
        "loss": total_loss / max(total_examples, 1),
        "action_loss": total_action_loss / max(total_examples, 1),
        "command_loss": total_command_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "command_mae": total_command_mae / max(total_examples, 1),
        "num_examples": total_examples,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _compute_class_weights(action_indices: list[int]):
    if torch is None:
        return None

    counts = [0 for _ in ACTION_VOCAB]
    for action_index in action_indices:
        counts[int(action_index)] += 1

    nonzero_counts = [count for count in counts if count > 0]
    if not nonzero_counts:
        return None

    mean_count = float(sum(nonzero_counts)) / float(len(nonzero_counts))
    weights = []
    for count in counts:
        if count <= 0:
            weights.append(0.0)
        else:
            weights.append(mean_count / float(count))
    return torch.tensor(weights, dtype=torch.float32)


def _build_balanced_action_sampler(action_indices: list[int]):
    if torch is None or WeightedRandomSampler is None or not action_indices:
        return None
    counts = [0 for _ in ACTION_VOCAB]
    for action_index in action_indices:
        counts[int(action_index)] += 1
    sample_weights = []
    for action_index in action_indices:
        count = counts[int(action_index)]
        sample_weights.append(0.0 if count <= 0 else 1.0 / float(count))
    weight_tensor = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weight_tensor, num_samples=len(action_indices), replacement=True)


def _build_episode_step_index(records: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    index: dict[str, dict[int, dict[str, Any]]] = {}
    for record in records:
        episode_id = str(record.get("episode_id", ""))
        step_index = int(record.get("step_index", 0))
        index.setdefault(episode_id, {})[step_index] = record
    return index


def _extract_goal_feature_vector(record: dict[str, Any], *, mirrored: bool = False) -> list[float]:
    metadata = dict(record.get("metadata", {}))
    teacher = metadata.get("teacher") or {}
    goal_features = teacher.get("goal_features") or {}

    forward_m = float(goal_features.get("forward_m", 0.0))
    right_m = float(goal_features.get("right_m", 0.0))
    alt_error_m = float(goal_features.get("alt_error_m", 0.0))
    heading_error_deg = float(goal_features.get("heading_error_deg", 0.0))

    if mirrored:
        right_m *= -1.0
        heading_error_deg *= -1.0

    return [
        float(np.clip(forward_m / 10.0, -2.0, 2.0)),
        float(np.clip(right_m / 5.0, -2.0, 2.0)),
        float(np.clip(alt_error_m / 3.0, -2.0, 2.0)),
        float(np.clip(heading_error_deg / 180.0, -1.0, 1.0)),
    ]


def _build_dataset_entries(
    records: list[dict[str, Any]],
    *,
    mirror_lateral_actions: bool,
) -> list[tuple[dict[str, Any], bool]]:
    entries: list[tuple[dict[str, Any], bool]] = []
    for record in records:
        entries.append((record, False))
        if mirror_lateral_actions and _supports_lateral_mirroring(int(record["action_index"])):
            entries.append((record, True))
    return entries


def _supports_lateral_mirroring(action_index: int) -> bool:
    action = ACTION_VOCAB[int(action_index)]
    return action in {"yaw_left", "yaw_right", "strafe_left", "strafe_right"}


def _mirrored_action_index(action_index: int) -> int:
    action = ACTION_VOCAB[int(action_index)]
    mirrored = {
        "yaw_left": "yaw_right",
        "yaw_right": "yaw_left",
        "strafe_left": "strafe_right",
        "strafe_right": "strafe_left",
    }.get(action, action)
    return ACTION_VOCAB.index(mirrored)

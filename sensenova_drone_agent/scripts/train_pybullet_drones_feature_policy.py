#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


TARGET_POS = np.array([0.0, 0.0, 1.0], dtype=np.float32)


@dataclass
class Sample:
    frame: np.ndarray
    state: np.ndarray
    action: np.ndarray
    distance_m: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/evaluate small PyBullet drone hover policies from frozen visual features. "
            "This is the first external-benchmark policy-head path for Kairos features."
        )
    )
    parser.add_argument("--out-dir", default="sensenova_drone_agent/output/pybullet_drones_feature_policy_v1")
    parser.add_argument(
        "--features",
        default="kinematic,rgb_downsample,random_projection,kairos_vae",
        help=(
            "Comma-separated feature families: kinematic,rgb_downsample,random_projection,"
            "resnet18_imagenet,kairos_vae,kairos_vae_flat,cnn_pixels"
        ),
    )
    parser.add_argument("--train-episodes", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=72)
    parser.add_argument("--seed", type=int, default=140000)
    parser.add_argument("--initial-z", type=float, default=0.2)
    parser.add_argument("--initial-xy-range", type=float, default=0.0)
    parser.add_argument("--initial-z-min", type=float, default=None)
    parser.add_argument("--initial-z-max", type=float, default=None)
    parser.add_argument("--success-distance-m", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--eval-trace-frames", type=int, default=8)
    parser.add_argument("--rgb-feature-size", type=int, default=16)
    parser.add_argument("--random-feature-dim", type=int, default=32)
    parser.add_argument("--cnn-image-size", type=int, default=64)
    parser.add_argument("--resnet-image-size", type=int, default=224)
    parser.add_argument(
        "--skip-resnet-if-unavailable",
        action="store_true",
        default=True,
        help="Record ResNet failure instead of failing the whole comparison.",
    )
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument("--kairos-device", default="cpu")
    parser.add_argument("--kairos-dtype", default="float32")
    parser.add_argument("--kairos-height", type=int, default=128)
    parser.add_argument("--kairos-width", type=int, default=128)
    parser.add_argument(
        "--skip-kairos-if-unavailable",
        action="store_true",
        default=True,
        help="Record Kairos failure instead of failing the whole comparison.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = resolve_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imports = import_pybullet_drones()
    torch = import_torch()
    set_seed(args.seed)

    feature_names = [name.strip() for name in args.features.split(",") if name.strip()]
    valid_features = {
        "kinematic",
        "rgb_downsample",
        "random_projection",
        "resnet18_imagenet",
        "kairos_vae",
        "kairos_vae_flat",
        "cnn_pixels",
    }
    invalid = sorted(set(feature_names) - valid_features)
    if invalid:
        raise ValueError(f"Unknown feature type(s): {invalid}")

    start_time = time.time()
    samples = collect_teacher_dataset(args=args, imports=imports)
    dataset_summary = {
        "num_samples": len(samples),
        "train_episodes": args.train_episodes,
        "max_steps": args.max_steps,
        "teacher": "target_velocity",
        "target_pos": TARGET_POS.tolist(),
        "initial_xy_range": args.initial_xy_range,
        "initial_z": args.initial_z,
        "initial_z_min": args.initial_z_min,
        "initial_z_max": args.initial_z_max,
    }
    (out_dir / "teacher_dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2), encoding="utf-8"
    )

    results = []
    for feature_name in feature_names:
        feature_dir = out_dir / feature_name
        feature_dir.mkdir(parents=True, exist_ok=True)
        try:
            set_seed(stable_feature_seed(args.seed, feature_name))
            if feature_name == "cnn_pixels":
                result = train_and_evaluate_cnn_pixels(
                    samples=samples,
                    feature_name=feature_name,
                    args=args,
                    imports=imports,
                    torch=torch,
                    out_dir=feature_dir,
                )
            else:
                encoder = make_feature_encoder(feature_name, args)
                features, actions = encode_dataset(samples, encoder, feature_dir)
                train_result = train_policy(
                    features=features,
                    actions=actions,
                    feature_name=feature_name,
                    args=args,
                    torch=torch,
                    out_dir=feature_dir,
                )
                eval_result = evaluate_learned_policy(
                    feature_name=feature_name,
                    encoder=encoder,
                    checkpoint_path=Path(train_result["best_checkpoint"]),
                    args=args,
                    imports=imports,
                    torch=torch,
                    out_dir=feature_dir,
                )
                result = {
                    "feature": feature_name,
                    "status": "ok",
                    "feature_dim": int(features.shape[1]),
                    "dataset_path": str(feature_dir / "dataset.npz"),
                    "train": train_result,
                    "eval": eval_result,
                }
        except Exception as exc:
            if feature_name in {"kairos_vae", "kairos_vae_flat"} and args.skip_kairos_if_unavailable:
                result = {
                    "feature": feature_name,
                    "status": "failed",
                    "error": repr(exc),
                    "note": "Kairos feature path failed; other baselines remain valid.",
                }
                (feature_dir / "error.txt").write_text(repr(exc), encoding="utf-8")
            elif feature_name == "resnet18_imagenet" and args.skip_resnet_if_unavailable:
                result = {
                    "feature": feature_name,
                    "status": "failed",
                    "error": repr(exc),
                    "note": "ResNet/ImageNet feature path failed; other baselines remain valid.",
                }
                (feature_dir / "error.txt").write_text(repr(exc), encoding="utf-8")
            else:
                raise
        results.append(result)

    summary = {
        "benchmark": "gym-pybullet-drones HoverAviary feature-policy BC",
        "source": "https://github.com/learnsyslab/gym-pybullet-drones",
        "elapsed_s": time.time() - start_time,
        "args": vars(args),
        "dataset": dataset_summary,
        "results": results,
        "ranking_by_success_then_distance": rank_results(results),
        "claim_boundary": (
            "This is an offline behavior-cloned policy-head benchmark. It tests whether frozen "
            "features contain usable control signal; it is not yet RL and not PX4/SITL transfer."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, out_dir / "report.md")
    print(json.dumps(summary["ranking_by_success_then_distance"], indent=2))
    print(f"Wrote {out_dir / 'summary.json'}")
    return 0


def import_torch():
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required. Run inside the PyBullet benchmark Docker image.") from exc
    return {"torch": torch, "nn": nn}


def import_pybullet_drones() -> dict[str, Any]:
    try:
        from gym_pybullet_drones.envs.HoverAviary import HoverAviary
        from gym_pybullet_drones.utils.enums import ActionType, ObservationType, Physics
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "gym-pybullet-drones is missing. Build/run with the PyBullet benchmark Docker scripts."
        ) from exc
    return {
        "HoverAviary": HoverAviary,
        "ActionType": ActionType,
        "ObservationType": ObservationType,
        "Physics": Physics,
    }


def collect_teacher_dataset(args: argparse.Namespace, imports: dict[str, Any]) -> list[Sample]:
    samples: list[Sample] = []
    for episode_idx in range(args.train_episodes):
        seed = args.seed + episode_idx
        set_seed(seed)
        initial_xyz = sample_initial_xyz(args, seed)
        env = make_env(args, imports, initial_xyz=initial_xyz)
        try:
            obs, info = env.reset(seed=seed)
            del info
            for _ in range(args.max_steps):
                state = env._getDroneStateVector(0).astype(np.float32)
                action = target_velocity_action(state)
                frame = rgb_from_obs(obs)
                distance = float(np.linalg.norm(TARGET_POS - state[0:3]))
                samples.append(
                    Sample(
                        frame=frame,
                        state=state.copy(),
                        action=action.reshape(-1).astype(np.float32),
                        distance_m=distance,
                    )
                )
                obs, reward, terminated, truncated, info = env.step(action)
                del reward, info
                if terminated or truncated:
                    break
        finally:
            env.close()
    if not samples:
        raise RuntimeError("Teacher dataset collection produced no samples.")
    return samples


class FeatureEncoder:
    def __init__(self, name: str, args: argparse.Namespace):
        self.name = name
        self.args = args
        self._projection: np.ndarray | None = None
        self._kairos = None
        self._resnet = None
        self._resnet_weights = None
        self._torch = None

    def encode(self, frame: np.ndarray, state: np.ndarray) -> np.ndarray:
        if self.name == "kinematic":
            delta = TARGET_POS - state[0:3]
            return np.concatenate([state[0:3], state[7:10], delta], axis=0).astype(np.float32)

        if self.name == "rgb_downsample":
            return downsample_rgb_feature(frame, size=self.args.rgb_feature_size)

        if self.name == "random_projection":
            raw = downsample_rgb_feature(frame, size=self.args.rgb_feature_size)
            if self._projection is None:
                rng = np.random.default_rng(self.args.seed + 991)
                scale = 1.0 / np.sqrt(max(1, raw.shape[0]))
                self._projection = rng.normal(
                    loc=0.0,
                    scale=scale,
                    size=(raw.shape[0], self.args.random_feature_dim),
                ).astype(np.float32)
            return (raw @ self._projection).astype(np.float32)

        if self.name == "resnet18_imagenet":
            return self._encode_resnet18(frame)

        if self.name in {"kairos_vae", "kairos_vae_flat"}:
            if self._kairos is None:
                from sensenova_drone.kairos_features import KairosVAEFeatureExtractor

                self._kairos = KairosVAEFeatureExtractor(
                    repo_root=REPO_ROOT,
                    device=self.args.kairos_device,
                    dtype=self.args.kairos_dtype,
                    height=self.args.kairos_height,
                    width=self.args.kairos_width,
                    tiled=False,
                )
            payload = self._kairos.encode_image(frame)
            if self.name == "kairos_vae_flat":
                return payload["latent"].reshape(-1).detach().cpu().numpy().astype(np.float32)
            return payload["image_features"].detach().cpu().numpy().astype(np.float32)

        raise ValueError(f"Unknown feature type: {self.name}")

    def _encode_resnet18(self, frame: np.ndarray) -> np.ndarray:
        if self._resnet is None:
            try:
                import torch
                from torchvision.models import ResNet18_Weights, resnet18
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "torchvision is required for resnet18_imagenet. Rebuild the PyBullet Docker image."
                ) from exc
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
            model.fc = torch.nn.Identity()
            model.eval().requires_grad_(False)
            model = model.to(self.args.torch_device)
            self._resnet = model
            self._resnet_weights = weights
            self._torch = torch

        image = Image.fromarray(frame).convert("RGB").resize(
            (self.args.resnet_image_size, self.args.resnet_image_size),
            Image.BILINEAR,
        )
        tensor = self._resnet_weights.transforms()(image).unsqueeze(0).to(self.args.torch_device)
        with self._torch.no_grad():
            features = self._resnet(tensor).detach().cpu().numpy()[0]
        return features.astype(np.float32)


def make_feature_encoder(name: str, args: argparse.Namespace) -> FeatureEncoder:
    return FeatureEncoder(name=name, args=args)


def encode_dataset(samples: list[Sample], encoder: FeatureEncoder, out_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    features = []
    actions = []
    records = []
    for idx, sample in enumerate(samples):
        feature = encoder.encode(sample.frame, sample.state)
        features.append(feature)
        actions.append(sample.action)
        if idx < 8:
            records.append(
                {
                    "sample": idx,
                    "feature_mean": float(np.mean(feature)),
                    "feature_std": float(np.std(feature)),
                    "label_action": sample.action.tolist(),
                    "distance_m": sample.distance_m,
                }
            )
    x = np.stack(features).astype(np.float32)
    y = np.stack(actions).astype(np.float32)
    np.savez_compressed(out_dir / "dataset.npz", features=x, actions=y)
    (out_dir / "dataset_preview.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return x, y


def train_policy(
    features: np.ndarray,
    actions: np.ndarray,
    feature_name: str,
    args: argparse.Namespace,
    torch: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    t = torch["torch"]
    nn = torch["nn"]
    device = t.device(args.torch_device)

    n = features.shape[0]
    indices = np.arange(n)
    rng = np.random.default_rng(args.seed + 17)
    rng.shuffle(indices)
    val_count = max(1, int(0.2 * n))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]
    if len(train_idx) == 0:
        train_idx = val_idx

    x_train = t.from_numpy(features[train_idx]).to(device=device)
    y_train = t.from_numpy(actions[train_idx]).to(device=device)
    x_val = t.from_numpy(features[val_idx]).to(device=device)
    y_val = t.from_numpy(actions[val_idx]).to(device=device)

    model = FeaturePolicy(input_dim=features.shape[1], hidden_dim=args.hidden_dim, nn=nn).to(device)
    optimizer = t.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = float("inf")
    best_path = out_dir / "best.pt"
    metrics = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = t.randperm(x_train.shape[0], device=device)
        losses = []
        for start in range(0, x_train.shape[0], args.batch_size):
            batch_idx = order[start : start + args.batch_size]
            pred = model(x_train[batch_idx])
            loss = nn.functional.mse_loss(pred, y_train[batch_idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with t.no_grad():
            val_pred = model(x_val)
            val_loss = float(nn.functional.mse_loss(val_pred, y_val).detach().cpu().item())
            mean_abs = float((val_pred - y_val).abs().mean().detach().cpu().item())

        record = {
            "epoch": epoch,
            "train_mse": float(np.mean(losses)) if losses else None,
            "val_mse": val_loss,
            "val_mean_abs_action_error": mean_abs,
        }
        metrics.append(record)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(t, model, best_path, feature_name, features.shape[1], args)

    last_path = out_dir / "last.pt"
    save_checkpoint(t, model, last_path, feature_name, features.shape[1], args)
    (out_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(record) for record in metrics) + "\n",
        encoding="utf-8",
    )
    return {
        "epochs": args.epochs,
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "best_val_mse": best_val,
        "last_val_mse": metrics[-1]["val_mse"],
        "last_val_mean_abs_action_error": metrics[-1]["val_mean_abs_action_error"],
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
    }


class FeaturePolicy:
    def __init__(self, input_dim: int, hidden_dim: int, nn: Any):
        super().__setattr__("_module", nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        ))

    def __call__(self, x):
        return self._module(x)

    def to(self, *args, **kwargs):
        self._module = self._module.to(*args, **kwargs)
        return self

    def train(self, *args, **kwargs):
        return self._module.train(*args, **kwargs)

    def eval(self):
        return self._module.eval()

    def parameters(self):
        return self._module.parameters()

    def state_dict(self):
        return self._module.state_dict()

    def load_state_dict(self, state_dict):
        return self._module.load_state_dict(state_dict)


def save_checkpoint(t: Any, model: FeaturePolicy, path: Path, feature_name: str, input_dim: int, args: argparse.Namespace) -> None:
    t.save(
        {
            "model_state": model.state_dict(),
            "feature_name": feature_name,
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "args": vars(args),
        },
        path,
    )


def train_and_evaluate_cnn_pixels(
    samples: list[Sample],
    feature_name: str,
    args: argparse.Namespace,
    imports: dict[str, Any],
    torch: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    frames = np.stack([sample.frame for sample in samples]).astype(np.uint8)
    actions = np.stack([sample.action for sample in samples]).astype(np.float32)
    np.savez_compressed(out_dir / "dataset.npz", frames=frames, actions=actions)
    train_result = train_cnn_policy(
        frames=frames,
        actions=actions,
        feature_name=feature_name,
        args=args,
        torch=torch,
        out_dir=out_dir,
    )
    eval_result = evaluate_cnn_policy(
        checkpoint_path=Path(train_result["best_checkpoint"]),
        args=args,
        imports=imports,
        torch=torch,
        out_dir=out_dir,
    )
    return {
        "feature": feature_name,
        "status": "ok",
        "feature_dim": f"rgb_{args.cnn_image_size}x{args.cnn_image_size}",
        "dataset_path": str(out_dir / "dataset.npz"),
        "train": train_result,
        "eval": eval_result,
    }


def train_cnn_policy(
    frames: np.ndarray,
    actions: np.ndarray,
    feature_name: str,
    args: argparse.Namespace,
    torch: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    t = torch["torch"]
    nn = torch["nn"]
    device = t.device(args.torch_device)
    n = frames.shape[0]
    indices = np.arange(n)
    rng = np.random.default_rng(args.seed + 23)
    rng.shuffle(indices)
    val_count = max(1, int(0.2 * n))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]
    if len(train_idx) == 0:
        train_idx = val_idx

    x_all = frames_to_tensor(t, frames, args.cnn_image_size)
    y_all = t.from_numpy(actions).float()
    x_train = x_all[train_idx].to(device=device)
    y_train = y_all[train_idx].to(device=device)
    x_val = x_all[val_idx].to(device=device)
    y_val = y_all[val_idx].to(device=device)

    model = TinyCNNPolicy(hidden_dim=args.hidden_dim, nn=nn).to(device)
    optimizer = t.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = float("inf")
    best_path = out_dir / "best.pt"
    metrics = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = t.randperm(x_train.shape[0], device=device)
        losses = []
        for start in range(0, x_train.shape[0], args.batch_size):
            batch_idx = order[start : start + args.batch_size]
            pred = model(x_train[batch_idx])
            loss = nn.functional.mse_loss(pred, y_train[batch_idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with t.no_grad():
            val_pred = model(x_val)
            val_loss = float(nn.functional.mse_loss(val_pred, y_val).detach().cpu().item())
            mean_abs = float((val_pred - y_val).abs().mean().detach().cpu().item())
        record = {
            "epoch": epoch,
            "train_mse": float(np.mean(losses)) if losses else None,
            "val_mse": val_loss,
            "val_mean_abs_action_error": mean_abs,
        }
        metrics.append(record)
        if val_loss < best_val:
            best_val = val_loss
            save_cnn_checkpoint(t, model, best_path, feature_name, args)

    last_path = out_dir / "last.pt"
    save_cnn_checkpoint(t, model, last_path, feature_name, args)
    (out_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(record) for record in metrics) + "\n",
        encoding="utf-8",
    )
    return {
        "epochs": args.epochs,
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "best_val_mse": best_val,
        "last_val_mse": metrics[-1]["val_mse"],
        "last_val_mean_abs_action_error": metrics[-1]["val_mean_abs_action_error"],
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
    }


class TinyCNNPolicy:
    def __init__(self, hidden_dim: int, nn: Any):
        super().__setattr__(
            "_module",
            nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
                nn.GELU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 4),
            ),
        )

    def __call__(self, x):
        return self._module(x)

    def to(self, *args, **kwargs):
        self._module = self._module.to(*args, **kwargs)
        return self

    def train(self, *args, **kwargs):
        return self._module.train(*args, **kwargs)

    def eval(self):
        return self._module.eval()

    def parameters(self):
        return self._module.parameters()

    def state_dict(self):
        return self._module.state_dict()

    def load_state_dict(self, state_dict):
        return self._module.load_state_dict(state_dict)


def save_cnn_checkpoint(t: Any, model: TinyCNNPolicy, path: Path, feature_name: str, args: argparse.Namespace) -> None:
    t.save(
        {
            "model_state": model.state_dict(),
            "feature_name": feature_name,
            "hidden_dim": args.hidden_dim,
            "cnn_image_size": args.cnn_image_size,
            "args": vars(args),
        },
        path,
    )


def frames_to_tensor(t: Any, frames: np.ndarray, image_size: int):
    resized = []
    for frame in frames:
        image = Image.fromarray(frame).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(image).astype(np.float32) / 255.0
        resized.append(np.transpose(arr, (2, 0, 1)))
    return t.from_numpy(np.stack(resized).astype(np.float32))


def frame_to_tensor(t: Any, frame: np.ndarray, image_size: int):
    return frames_to_tensor(t, frame[None, ...], image_size)


def evaluate_cnn_policy(
    checkpoint_path: Path,
    args: argparse.Namespace,
    imports: dict[str, Any],
    torch: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    t = torch["torch"]
    nn = torch["nn"]
    device = t.device(args.torch_device)
    checkpoint = t.load(checkpoint_path, map_location=device)
    model = TinyCNNPolicy(hidden_dim=int(checkpoint["hidden_dim"]), nn=nn).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    records = []
    traces_dir = out_dir / "eval_traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    for episode_idx in range(args.eval_episodes):
        seed = args.seed + 1000 + episode_idx
        record = run_cnn_episode(
            model=model,
            args=args,
            imports=imports,
            torch=t,
            device=device,
            seed=seed,
            traces_dir=traces_dir,
        )
        records.append(record)

    episodes_path = out_dir / "eval_episodes.jsonl"
    episodes_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    trace_paths = [Path(record["trace_contact_sheet"]) for record in records if record.get("trace_contact_sheet")]
    combined_trace = None
    if trace_paths:
        combined_trace = out_dir / "eval_contact_sheet.png"
        make_combined_contact_sheet(trace_paths[: min(4, len(trace_paths))], combined_trace)

    n = max(1, len(records))
    return {
        "episodes": len(records),
        "success_rate": sum(1 for record in records if record["success"]) / n,
        "mean_return": float(np.mean([record["return"] for record in records])),
        "mean_length": float(np.mean([record["steps"] for record in records])),
        "mean_final_distance_m": float(np.mean([record["final_distance_m"] for record in records])),
        "mean_min_distance_m": float(np.mean([record["min_distance_m"] for record in records])),
        "episodes_path": str(episodes_path),
        "contact_sheet": str(combined_trace) if combined_trace else None,
    }


def run_cnn_episode(
    model: TinyCNNPolicy,
    args: argparse.Namespace,
    imports: dict[str, Any],
    torch: Any,
    device: Any,
    seed: int,
    traces_dir: Path,
) -> dict[str, Any]:
    set_seed(seed)
    initial_xyz = sample_initial_xyz(args, seed)
    env = make_env(args, imports, initial_xyz=initial_xyz)
    frames = []
    total_reward = 0.0
    min_distance = float("inf")
    final_distance = float("inf")
    terminated = False
    truncated = False
    start_time = time.time()
    try:
        obs, info = env.reset(seed=seed)
        del info
        for step in range(args.max_steps):
            state = env._getDroneStateVector(0).astype(np.float32)
            distance = float(np.linalg.norm(TARGET_POS - state[0:3]))
            min_distance = min(min_distance, distance)
            final_distance = distance
            frame = rgb_from_obs(obs)
            if should_capture_frame(step, args.eval_trace_frames, args.max_steps):
                frames.append(frame)
            with torch.no_grad():
                tensor = frame_to_tensor(torch, frame, args.cnn_image_size).to(device=device)
                raw_action = model(tensor).detach().cpu().numpy()[0]
            action = sanitize_action(raw_action)
            obs, reward, terminated, truncated, info = env.step(action)
            del info
            total_reward += float(reward)
            if terminated or truncated:
                break
    finally:
        env.close()

    trace_path = None
    if frames:
        trace_path = traces_dir / f"eval_{seed}.png"
        make_frame_contact_sheet(frames, trace_path, label=f"eval seed={seed}")

    return {
        "seed": seed,
        "initial_xyz": initial_xyz.reshape(-1).astype(float).tolist(),
        "steps": step + 1 if "step" in locals() else 0,
        "return": total_reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": bool(final_distance <= args.success_distance_m),
        "final_distance_m": final_distance,
        "min_distance_m": min_distance,
        "elapsed_s": time.time() - start_time,
        "trace_contact_sheet": str(trace_path) if trace_path else None,
    }


def evaluate_learned_policy(
    feature_name: str,
    encoder: FeatureEncoder,
    checkpoint_path: Path,
    args: argparse.Namespace,
    imports: dict[str, Any],
    torch: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    t = torch["torch"]
    nn = torch["nn"]
    device = t.device(args.torch_device)
    checkpoint = t.load(checkpoint_path, map_location=device)
    model = FeaturePolicy(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        nn=nn,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    records = []
    traces_dir = out_dir / "eval_traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    for episode_idx in range(args.eval_episodes):
        seed = args.seed + 1000 + episode_idx
        record = run_learned_episode(
            feature_name=feature_name,
            encoder=encoder,
            model=model,
            args=args,
            imports=imports,
            torch=t,
            device=device,
            seed=seed,
            traces_dir=traces_dir,
        )
        records.append(record)

    episodes_path = out_dir / "eval_episodes.jsonl"
    episodes_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    trace_paths = [Path(record["trace_contact_sheet"]) for record in records if record.get("trace_contact_sheet")]
    combined_trace = None
    if trace_paths:
        combined_trace = out_dir / "eval_contact_sheet.png"
        make_combined_contact_sheet(trace_paths[: min(4, len(trace_paths))], combined_trace)

    n = max(1, len(records))
    return {
        "episodes": len(records),
        "success_rate": sum(1 for record in records if record["success"]) / n,
        "mean_return": float(np.mean([record["return"] for record in records])),
        "mean_length": float(np.mean([record["steps"] for record in records])),
        "mean_final_distance_m": float(np.mean([record["final_distance_m"] for record in records])),
        "mean_min_distance_m": float(np.mean([record["min_distance_m"] for record in records])),
        "episodes_path": str(episodes_path),
        "contact_sheet": str(combined_trace) if combined_trace else None,
    }


def run_learned_episode(
    feature_name: str,
    encoder: FeatureEncoder,
    model: FeaturePolicy,
    args: argparse.Namespace,
    imports: dict[str, Any],
    torch: Any,
    device: Any,
    seed: int,
    traces_dir: Path,
) -> dict[str, Any]:
    del feature_name
    set_seed(seed)
    initial_xyz = sample_initial_xyz(args, seed)
    env = make_env(args, imports, initial_xyz=initial_xyz)
    frames = []
    total_reward = 0.0
    min_distance = float("inf")
    final_distance = float("inf")
    terminated = False
    truncated = False
    start_time = time.time()
    try:
        obs, info = env.reset(seed=seed)
        del info
        for step in range(args.max_steps):
            state = env._getDroneStateVector(0).astype(np.float32)
            distance = float(np.linalg.norm(TARGET_POS - state[0:3]))
            min_distance = min(min_distance, distance)
            final_distance = distance
            if should_capture_frame(step, args.eval_trace_frames, args.max_steps):
                frames.append(rgb_from_obs(obs))
            feature = encoder.encode(rgb_from_obs(obs), state)
            with torch.no_grad():
                tensor = torch.from_numpy(feature[None, :]).to(device=device)
                raw_action = model(tensor).detach().cpu().numpy()[0]
            action = sanitize_action(raw_action)
            obs, reward, terminated, truncated, info = env.step(action)
            del info
            total_reward += float(reward)
            if terminated or truncated:
                break
    finally:
        env.close()

    trace_path = None
    if frames:
        trace_path = traces_dir / f"eval_{seed}.png"
        make_frame_contact_sheet(frames, trace_path, label=f"eval seed={seed}")

    return {
        "seed": seed,
        "initial_xyz": initial_xyz.reshape(-1).astype(float).tolist(),
        "steps": step + 1 if "step" in locals() else 0,
        "return": total_reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": bool(final_distance <= args.success_distance_m),
        "final_distance_m": final_distance,
        "min_distance_m": min_distance,
        "elapsed_s": time.time() - start_time,
        "trace_contact_sheet": str(trace_path) if trace_path else None,
    }


def make_env(args: argparse.Namespace, imports: dict[str, Any], initial_xyz: np.ndarray | None = None):
    HoverAviary = imports["HoverAviary"]
    ObservationType = imports["ObservationType"]
    ActionType = imports["ActionType"]
    Physics = imports["Physics"]

    return HoverAviary(
        initial_xyzs=initial_xyz if initial_xyz is not None else np.array([[0.0, 0.0, args.initial_z]], dtype=np.float32),
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=24,
        gui=False,
        record=False,
        obs=ObservationType.RGB,
        act=ActionType.VEL,
    )


def sample_initial_xyz(args: argparse.Namespace, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 12345)
    xy_range = float(args.initial_xy_range)
    if xy_range > 0:
        x = float(rng.uniform(-xy_range, xy_range))
        y = float(rng.uniform(-xy_range, xy_range))
    else:
        x = 0.0
        y = 0.0

    z_min = args.initial_z_min
    z_max = args.initial_z_max
    if z_min is not None or z_max is not None:
        lo = float(args.initial_z if z_min is None else z_min)
        hi = float(args.initial_z if z_max is None else z_max)
        if hi < lo:
            raise ValueError("--initial-z-max must be >= --initial-z-min")
        z = float(rng.uniform(lo, hi)) if hi > lo else lo
    else:
        z = float(args.initial_z)
    return np.array([[x, y, z]], dtype=np.float32)


def target_velocity_action(state: np.ndarray) -> np.ndarray:
    delta = TARGET_POS - state[0:3]
    norm = float(np.linalg.norm(delta))
    direction = np.zeros(3, dtype=np.float32) if norm < 1e-6 else (delta / norm).astype(np.float32)
    speed_fraction = np.float32(np.clip(norm, 0.1, 1.0))
    return np.array([[direction[0], direction[1], direction[2], speed_fraction]], dtype=np.float32)


def sanitize_action(raw_action: np.ndarray) -> np.ndarray:
    action = np.asarray(raw_action, dtype=np.float32).reshape(4)
    direction = np.clip(action[:3], -1.0, 1.0)
    norm = float(np.linalg.norm(direction))
    if norm > 1.0:
        direction = direction / norm
    speed = np.clip(action[3], 0.0, 1.0)
    return np.array([[direction[0], direction[1], direction[2], speed]], dtype=np.float32)


def rgb_from_obs(obs: Any) -> np.ndarray:
    frame = np.asarray(obs[0])
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return np.clip(frame, 0, 255).astype(np.uint8)


def downsample_rgb_feature(frame: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray(frame).resize((size, size), Image.BILINEAR)
    arr = np.asarray(image).astype(np.float32) / 255.0
    return (arr.reshape(-1) - 0.5).astype(np.float32)


def should_capture_frame(step: int, trace_frames: int, max_steps: int) -> bool:
    if trace_frames <= 0:
        return False
    interval = max(1, max_steps // trace_frames)
    return step % interval == 0


def make_frame_contact_sheet(frames: list[np.ndarray], out_path: Path, label: str = "") -> None:
    if not frames:
        return
    pil_frames = [Image.fromarray(frame).resize((160, 120)) for frame in frames]
    label_h = 24
    sheet = Image.new("RGB", (160 * len(pil_frames), 120 + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), label, fill=(0, 0, 0))
    for idx, frame in enumerate(pil_frames):
        sheet.paste(frame, (160 * idx, label_h))
    sheet.save(out_path)


def make_combined_contact_sheet(trace_paths: list[Path], out_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in trace_paths if path.exists()]
    if not images:
        return
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for image in images:
        sheet.paste(image, (0, y))
        y += image.height
    sheet.save(out_path)


def rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        if result.get("status") != "ok":
            rows.append({"feature": result["feature"], "status": result["status"], "error": result.get("error")})
            continue
        eval_result = result["eval"]
        rows.append(
            {
                "feature": result["feature"],
                "status": result["status"],
                "success_rate": eval_result["success_rate"],
                "mean_final_distance_m": eval_result["mean_final_distance_m"],
                "mean_return": eval_result["mean_return"],
                "best_val_mse": result["train"]["best_val_mse"],
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row.get("status") != "ok",
            -float(row.get("success_rate", -1.0)),
            float(row.get("mean_final_distance_m", 1e9)),
        ),
    )


def write_report(summary: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# PyBullet Drone Feature Policy Benchmark",
        "",
        "Source: https://github.com/learnsyslab/gym-pybullet-drones",
        "",
        "## Scope",
        "",
        "This trains small behavior-cloned action heads on frozen features from RGB observations.",
        "The teacher is a privileged target-velocity controller used only to create labels.",
        "",
        "## Results",
        "",
        "| Feature | Status | Success | Final distance m | Mean return | Best val MSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["ranking_by_success_then_distance"]:
        if row.get("status") != "ok":
            lines.append(f"| `{row['feature']}` | `{row['status']}` |  |  |  |  |")
            continue
        lines.append(
            "| `{feature}` | `{status}` | {success_rate:.3f} | {mean_final_distance_m:.4f} | "
            "{mean_return:.3f} | {best_val_mse:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            summary["claim_boundary"],
            "",
            "A positive Kairos result here would mean Kairos/Wan VAE features support a learned policy head "
            "on a recognized drone benchmark. It would still need more seeds and comparison to stronger "
            "pretrained encoders before it becomes paper-grade evidence.",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ModuleNotFoundError:
        pass


def stable_feature_seed(base_seed: int, feature_name: str) -> int:
    return int(base_seed + sum((idx + 1) * ord(ch) for idx, ch in enumerate(feature_name)))


def resolve_out_dir(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


if __name__ == "__main__":
    raise SystemExit(main())

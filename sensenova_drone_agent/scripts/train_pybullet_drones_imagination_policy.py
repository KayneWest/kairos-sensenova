#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SCRIPT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from train_pybullet_drones_feature_policy import (  # noqa: E402
    FeatureEncoder,
    TARGET_POS,
    import_pybullet_drones,
    make_env,
    rgb_from_obs,
    sample_initial_xyz,
    sanitize_action,
    set_seed,
    target_velocity_action,
)


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None


@dataclass
class TransitionBatch:
    z: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    z_next: np.ndarray
    done: np.ndarray
    episode: np.ndarray | None = None
    step: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PyBullet drone policy inside a learned latent simulator."
    )
    parser.add_argument("--out-dir", default="sensenova_drone_agent/output/pybullet_drones_imagination_policy_v1")
    parser.add_argument("--feature", default="kinematic")
    parser.add_argument("--collect-episodes", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--seed", type=int, default=170000)
    parser.add_argument(
        "--eval-seeds",
        default="",
        help="Comma-separated fixed evaluation seeds. Overrides --eval-episodes when provided.",
    )
    parser.add_argument("--eval-seed-offset", type=int, default=1000)
    parser.add_argument(
        "--dataset-cache",
        default="",
        help="Optional transitions .npz cache path. Relative paths resolve from repo root.",
    )
    parser.add_argument(
        "--reuse-dataset-cache",
        action="store_true",
        help="Load --dataset-cache if present instead of recollecting and re-encoding transitions.",
    )
    parser.add_argument("--initial-z", type=float, default=0.2)
    parser.add_argument("--initial-xy-range", type=float, default=0.4)
    parser.add_argument("--initial-z-min", type=float, default=0.15)
    parser.add_argument("--initial-z-max", type=float, default=0.6)
    parser.add_argument("--success-distance-m", type=float, default=0.15)
    parser.add_argument("--behavior", choices=["teacher", "noisy_teacher", "random_mix"], default="random_mix")
    parser.add_argument("--behavior-noise", type=float, default=0.25)
    parser.add_argument("--random-action-prob", type=float, default=0.25)
    parser.add_argument("--world-epochs", type=int, default=120)
    parser.add_argument("--bc-epochs", type=int, default=80)
    parser.add_argument("--imagination-updates", type=int, default=200)
    parser.add_argument("--imagination-horizon", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument(
        "--dynamics-action-conditioning",
        choices=["concat", "action_token"],
        default="concat",
        help=(
            "How the learned dynamics consumes actions. 'concat' is the original MLP baseline; "
            "'action_token' gives actions their own token and fuses state/action with a small transformer."
        ),
    )
    parser.add_argument("--dynamics-action-token-layers", type=int, default=2)
    parser.add_argument("--dynamics-action-token-heads", type=int, default=4)
    parser.add_argument(
        "--world-training-mode",
        choices=["one_step", "sequence"],
        default="one_step",
        help="Train dynamics on isolated transitions or contiguous trajectory windows.",
    )
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--sequence-stride", type=int, default=1)
    parser.add_argument("--world-lr", type=float, default=1e-3)
    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--lambda-return", type=float, default=0.95)
    parser.add_argument("--imagination-objective", choices=["pmpo", "backprop"], default="pmpo")
    parser.add_argument("--policy-std", type=float, default=0.15)
    parser.add_argument("--pmpo-alpha", type=float, default=0.5)
    parser.add_argument("--pmpo-bootstrap-value", action="store_true")
    parser.add_argument("--return-clip", type=float, default=20.0)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--prior-weight", type=float, default=0.2)
    parser.add_argument("--reward-weight", type=float, default=1.0)
    parser.add_argument("--done-weight", type=float, default=0.2)
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument("--eval-trace-frames", type=int, default=6)
    parser.add_argument("--feature-stack", type=int, default=1)
    parser.add_argument("--feature-stack-deltas", action="store_true")
    parser.add_argument("--include-prev-action-in-feature", action="store_true")
    parser.add_argument("--rgb-feature-size", type=int, default=16)
    parser.add_argument("--random-feature-dim", type=int, default=32)
    parser.add_argument("--cnn-image-size", type=int, default=64)
    parser.add_argument("--resnet-image-size", type=int, default=224)
    parser.add_argument("--kairos-device", default="cpu")
    parser.add_argument("--kairos-dtype", default="float32")
    parser.add_argument("--kairos-height", type=int, default=128)
    parser.add_argument("--kairos-width", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    if torch is None or nn is None or F is None:
        raise RuntimeError("PyTorch is required. Run inside the PyBullet benchmark Docker image.")
    args = parse_args()
    set_seed(args.seed)
    out_dir = resolve_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    imports = import_pybullet_drones()
    encoder = AgentObservationEncoder(FeatureEncoder(args.feature, args), args)
    dataset_cache = resolve_optional_path(args.dataset_cache)
    dataset_source = "collection"
    cache_hit = False
    if args.reuse_dataset_cache and dataset_cache and dataset_cache.exists():
        transitions, dataset_records = load_transition_cache(dataset_cache)
        dataset_source = "cache"
        cache_hit = True
    else:
        transitions, dataset_records = collect_transitions(args, imports, encoder)
        if dataset_cache:
            save_transition_cache(dataset_cache, transitions, dataset_records, args)
    normalizer = make_normalizer(transitions.z)
    save_dataset(out_dir, transitions, dataset_records, normalizer)

    device = torch.device(args.torch_device)
    z_dim = int(transitions.z.shape[1])
    dynamics = LatentDynamics(
        z_dim=z_dim,
        hidden_dim=args.hidden_dim,
        action_conditioning=args.dynamics_action_conditioning,
        action_token_layers=args.dynamics_action_token_layers,
        action_token_heads=args.dynamics_action_token_heads,
    ).to(device)
    actor = Actor(z_dim=z_dim, hidden_dim=args.hidden_dim).to(device)
    critic = Critic(z_dim=z_dim, hidden_dim=args.hidden_dim).to(device)

    tensors = transitions_to_tensors(transitions, normalizer, device)
    world_metrics = train_dynamics(dynamics, tensors, args)
    bc_metrics = train_bc_actor(actor, tensors, args)
    prior_actor = copy.deepcopy(actor).eval().requires_grad_(False)

    bc_eval = evaluate_actor(
        actor,
        encoder=encoder,
        normalizer=normalizer,
        args=args,
        imports=imports,
        out_dir=out_dir / "eval_bc_prior",
    )

    imagination_metrics = train_policy_in_imagination(
        dynamics=dynamics,
        actor=actor,
        prior_actor=prior_actor,
        critic=critic,
        tensors=tensors,
        args=args,
    )

    imagined_eval = evaluate_actor(
        actor,
        encoder=encoder,
        normalizer=normalizer,
        args=args,
        imports=imports,
        out_dir=out_dir / "eval_after_imagination",
    )
    policy_selection = select_policy(bc_eval, imagined_eval)
    selected_actor = prior_actor if policy_selection["selected_actor"] == "bc_prior" else actor

    torch.save(
        {
            "feature": args.feature,
            "normalizer": normalizer,
            "dynamics_state": dynamics.state_dict(),
            "actor_state": actor.state_dict(),
            "bc_prior_actor_state": prior_actor.state_dict(),
            "selected_actor_state": selected_actor.state_dict(),
            "critic_state": critic.state_dict(),
            "policy_selection": policy_selection,
            "args": vars(args),
        },
        out_dir / "imagination_policy.pt",
    )

    summary = {
        "benchmark": "PyBullet learned latent simulator imagination policy",
        "elapsed_s": time.time() - started,
        "args": vars(args),
        "dataset": {
            "transitions": int(transitions.z.shape[0]),
            "feature": args.feature,
            "feature_dim": z_dim,
            "feature_stack": args.feature_stack,
            "feature_stack_deltas": args.feature_stack_deltas,
            "include_prev_action_in_feature": args.include_prev_action_in_feature,
            "dynamics_action_conditioning": args.dynamics_action_conditioning,
            "world_training_mode": args.world_training_mode,
            "sequence_length": args.sequence_length,
            "behavior": args.behavior,
            "source": dataset_source,
            "cache_hit": cache_hit,
            "cache_path": str(dataset_cache) if dataset_cache else None,
        },
        "world_model": summarize_last(world_metrics),
        "bc_prior_eval": bc_eval,
        "imagination": summarize_last(imagination_metrics),
        "after_imagination_eval": imagined_eval,
        "policy_selection": policy_selection,
        "claim_boundary": (
            "This trains inside a learned latent simulator. It is a model-based RL scaffold, "
            "not proof of robust drone autonomy or Kairos superiority."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_jsonl(out_dir / "world_metrics.jsonl", world_metrics)
    write_jsonl(out_dir / "imagination_metrics.jsonl", imagination_metrics)
    write_report(summary, out_dir / "report.md")
    print(json.dumps(summary, indent=2))
    return 0


class LatentDynamics(nn.Module):
    def __init__(
        self,
        z_dim: int,
        hidden_dim: int,
        *,
        action_conditioning: str = "concat",
        action_token_layers: int = 2,
        action_token_heads: int = 4,
    ):
        super().__init__()
        self.action_conditioning = action_conditioning
        if action_conditioning == "concat":
            self.trunk = nn.Sequential(
                nn.Linear(z_dim + 4, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
        elif action_conditioning == "action_token":
            if hidden_dim % action_token_heads != 0:
                raise ValueError(
                    f"hidden_dim={hidden_dim} must be divisible by "
                    f"dynamics_action_token_heads={action_token_heads}"
                )
            self.z_token = nn.Sequential(
                nn.Linear(z_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
            )
            self.action_token = nn.Sequential(
                nn.Linear(4, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            self.token_type = nn.Parameter(torch.empty(2, hidden_dim))
            nn.init.normal_(self.token_type, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=action_token_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.trunk = nn.TransformerEncoder(layer, num_layers=action_token_layers)
            self.post = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
        else:
            raise ValueError(f"Unsupported action_conditioning: {action_conditioning}")
        self.delta = nn.Linear(hidden_dim, z_dim)
        self.reward = nn.Linear(hidden_dim, 1)
        self.done = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.action_conditioning == "concat":
            x = self.trunk(torch.cat([z, action], dim=-1))
        else:
            x = self._action_token_sequence_features(z[:, None, :], action[:, None, :])[:, 0]
        return {
            "z_next": z + self.delta(x),
            "reward": self.reward(x).squeeze(-1),
            "done_logit": self.done(x).squeeze(-1),
        }

    def forward_sequence(self, z: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        if z.dim() != 3 or action.dim() != 3:
            raise ValueError("forward_sequence expects z=(B,T,Z) and action=(B,T,4)")
        bsz, steps, z_dim = z.shape
        if self.action_conditioning == "concat":
            x = self.trunk(torch.cat([z, action], dim=-1).reshape(bsz * steps, z_dim + 4))
            x = x.reshape(bsz, steps, -1)
        else:
            x = self._action_token_sequence_features(z, action)
        return {
            "z_next": z + self.delta(x),
            "reward": self.reward(x).squeeze(-1),
            "done_logit": self.done(x).squeeze(-1),
        }

    def _action_token_sequence_features(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        z_token = self.z_token(z) + self.token_type[0].view(1, 1, -1)
        a_token = self.action_token(project_action_tensor(action)) + self.token_type[1].view(1, 1, -1)
        # Order action before observation so the observation token at time t can
        # attend to the action that produces the next observation.
        tokens = torch.stack([a_token, z_token], dim=2).flatten(1, 2)
        seq_len = tokens.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        encoded = self.trunk(tokens, mask=causal_mask)
        z_outputs = encoded[:, 1::2, :]
        return self.post(z_outputs)


class AgentObservationEncoder:
    """
    Agent-visible feature wrapper.

    For pixel/Kairos policies, a single frame is under-observed: it does not
    reliably expose velocity. Stacking recent visual features and appending the
    previous action gives the policy/dynamics a minimal observation history
    without using privileged simulator state.
    """

    def __init__(self, base_encoder: FeatureEncoder, args: argparse.Namespace):
        self.base_encoder = base_encoder
        self.args = args
        self.stack = max(1, int(args.feature_stack))
        self.history: deque[np.ndarray] = deque(maxlen=self.stack)

    def reset(self) -> None:
        self.history.clear()

    def encode(self, frame: np.ndarray, state: np.ndarray, prev_action: np.ndarray | None = None) -> np.ndarray:
        base = self.base_encoder.encode(frame, state).astype(np.float32)
        if not self.history:
            for _ in range(self.stack):
                self.history.append(base.copy())
        else:
            self.history.append(base.copy())

        stacked = list(self.history)
        while len(stacked) < self.stack:
            stacked.insert(0, stacked[0].copy())

        components = [feature.reshape(-1) for feature in stacked]
        if self.args.feature_stack_deltas and len(stacked) > 1:
            components.extend((stacked[idx] - stacked[idx - 1]).reshape(-1) for idx in range(1, len(stacked)))

        if self.args.include_prev_action_in_feature:
            if prev_action is None:
                prev = np.zeros(4, dtype=np.float32)
            else:
                prev = np.asarray(prev_action, dtype=np.float32).reshape(4)
            components.append(prev)

        return np.concatenate(components, axis=0).astype(np.float32)


class Actor(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.net(z)
        direction = torch.tanh(raw[..., :3])
        speed = torch.sigmoid(raw[..., 3:4])
        return project_action_tensor(torch.cat([direction, speed], dim=-1))

    def sample_action(self, z: torch.Tensor, std: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self(z)
        dist = torch.distributions.Normal(mean, torch.full_like(mean, std))
        raw_action = dist.rsample()
        action = project_action_tensor(raw_action)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def log_prob(self, z: torch.Tensor, action: torch.Tensor, std: float) -> torch.Tensor:
        mean = self(z)
        dist = torch.distributions.Normal(mean, torch.full_like(mean, std))
        return dist.log_prob(action).sum(dim=-1)

    def prior_kl(self, z: torch.Tensor, prior_actor: "Actor", std: float) -> torch.Tensor:
        mean = self(z)
        with torch.no_grad():
            prior_mean = prior_actor(z)
        # Same diagonal std for both policies, so KL reduces to squared mean gap.
        return (((mean - prior_mean) ** 2) / (2.0 * std * std)).sum(dim=-1)


def project_action_tensor(action: torch.Tensor) -> torch.Tensor:
    direction = torch.clamp(action[..., :3], -1.0, 1.0)
    direction_norm = direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    direction = direction / torch.clamp(direction_norm, min=1.0)
    speed = torch.clamp(action[..., 3:4], 0.0, 1.0)
    return torch.cat([direction, speed], dim=-1)


class Critic(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


def collect_transitions(args: argparse.Namespace, imports: dict[str, Any], encoder: AgentObservationEncoder) -> tuple[TransitionBatch, list[dict[str, Any]]]:
    zs: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    z_nexts: list[np.ndarray] = []
    dones: list[float] = []
    episodes: list[int] = []
    steps: list[int] = []
    records: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed + 515)

    for episode_idx in range(args.collect_episodes):
        seed = args.seed + episode_idx
        set_seed(seed)
        initial_xyz = sample_initial_xyz(args, seed)
        env = make_env(args, imports, initial_xyz=initial_xyz)
        try:
            encoder.reset()
            obs, info = env.reset(seed=seed)
            del info
            prev_action = np.zeros(4, dtype=np.float32)
            state = env._getDroneStateVector(0).astype(np.float32)
            z = encoder.encode(rgb_from_obs(obs), state, prev_action)
            for step in range(args.max_steps):
                action = behavior_action(args, rng, state)
                next_obs, env_reward, terminated, truncated, info = env.step(action)
                del info
                next_state = env._getDroneStateVector(0).astype(np.float32)
                z_next = encoder.encode(rgb_from_obs(next_obs), next_state, action)
                reward = shaped_reward(state, next_state, action, args, env_reward=float(env_reward))
                done = bool(terminated or truncated or np.linalg.norm(TARGET_POS - next_state[0:3]) <= args.success_distance_m)
                zs.append(z.astype(np.float32))
                actions.append(action.reshape(-1).astype(np.float32))
                rewards.append(float(reward))
                z_nexts.append(z_next.astype(np.float32))
                dones.append(1.0 if done else 0.0)
                episodes.append(episode_idx)
                steps.append(step)
                if len(records) < 16:
                    records.append(
                        {
                            "episode": episode_idx,
                            "step": step,
                            "initial_xyz": initial_xyz.reshape(-1).astype(float).tolist(),
                            "distance_before_m": float(np.linalg.norm(TARGET_POS - state[0:3])),
                            "distance_after_m": float(np.linalg.norm(TARGET_POS - next_state[0:3])),
                            "action": action.reshape(-1).astype(float).tolist(),
                            "reward": float(reward),
                            "done": done,
                        }
                    )
                obs = next_obs
                state = next_state
                z = z_next
                prev_action = action.reshape(-1).astype(np.float32)
                if terminated or truncated:
                    break
        finally:
            env.close()

    if not zs:
        raise RuntimeError("No transitions collected.")
    return (
        TransitionBatch(
            z=np.stack(zs).astype(np.float32),
            action=np.stack(actions).astype(np.float32),
            reward=np.asarray(rewards, dtype=np.float32),
            z_next=np.stack(z_nexts).astype(np.float32),
            done=np.asarray(dones, dtype=np.float32),
            episode=np.asarray(episodes, dtype=np.int64),
            step=np.asarray(steps, dtype=np.int64),
        ),
        records,
    )


def load_transition_cache(path: Path) -> tuple[TransitionBatch, list[dict[str, Any]]]:
    with np.load(path) as data:
        required = {"z", "action", "reward", "z_next", "done"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"Transition cache {path} is missing arrays: {missing}")
        batch = TransitionBatch(
            z=data["z"].astype(np.float32),
            action=data["action"].astype(np.float32),
            reward=data["reward"].astype(np.float32),
            z_next=data["z_next"].astype(np.float32),
            done=data["done"].astype(np.float32),
            episode=data["episode"].astype(np.int64) if "episode" in data.files else None,
            step=data["step"].astype(np.int64) if "step" in data.files else None,
        )
    preview_path = path.with_suffix(".preview.json")
    records = []
    if preview_path.exists():
        records = json.loads(preview_path.read_text(encoding="utf-8"))
    return batch, records


def save_transition_cache(
    path: Path,
    transitions: TransitionBatch,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "z": transitions.z,
        "action": transitions.action,
        "reward": transitions.reward,
        "z_next": transitions.z_next,
        "done": transitions.done,
    }
    if transitions.episode is not None:
        arrays["episode"] = transitions.episode
    if transitions.step is not None:
        arrays["step"] = transitions.step
    np.savez_compressed(path, **arrays)
    path.with_suffix(".preview.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    metadata = {
        "feature": args.feature,
        "feature_stack": args.feature_stack,
        "feature_stack_deltas": args.feature_stack_deltas,
        "include_prev_action_in_feature": args.include_prev_action_in_feature,
        "collect_episodes": args.collect_episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "behavior": args.behavior,
        "behavior_noise": args.behavior_noise,
        "random_action_prob": args.random_action_prob,
        "transitions": int(transitions.z.shape[0]),
        "feature_dim": int(transitions.z.shape[1]),
        "has_episode_metadata": transitions.episode is not None and transitions.step is not None,
    }
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def measure_action_sensitivity(
    model: LatentDynamics,
    tensors: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, float]:
    n = tensors["z"].shape[0]
    if n < 2:
        return {}
    was_training = model.training
    model.eval()
    count = min(n, max(args.batch_size, min(2048, n)))
    idx = torch.randperm(n, device=tensors["z"].device)[:count]
    shuffled_idx = idx[torch.randperm(idx.shape[0], device=tensors["z"].device)]
    with torch.no_grad():
        normal = model(tensors["z"][idx], tensors["action"][idx])
        shuffled = model(tensors["z"][idx], tensors["action"][shuffled_idx])
        normal_z_mse = F.mse_loss(normal["z_next"], tensors["z_next"][idx])
        shuffled_z_mse = F.mse_loss(shuffled["z_next"], tensors["z_next"][idx])
        normal_reward_mse = F.mse_loss(normal["reward"], tensors["reward"][idx])
        shuffled_reward_mse = F.mse_loss(shuffled["reward"], tensors["reward"][idx])
        normal_loss = normal_z_mse + args.reward_weight * normal_reward_mse
        shuffled_loss = shuffled_z_mse + args.reward_weight * shuffled_reward_mse
        action_effect = (normal["z_next"] - shuffled["z_next"]).pow(2).mean().sqrt()
    eps = 1e-8
    metrics = {
        "action_probe_samples": float(count),
        "action_shuffle_z_mse": float(shuffled_z_mse.detach().cpu().item()),
        "action_shuffle_reward_mse": float(shuffled_reward_mse.detach().cpu().item()),
        "action_shuffle_loss_ratio": float((shuffled_loss / normal_loss.clamp_min(eps)).detach().cpu().item()),
        "action_shuffle_z_mse_ratio": float((shuffled_z_mse / normal_z_mse.clamp_min(eps)).detach().cpu().item()),
        "action_effect_rms": float(action_effect.detach().cpu().item()),
    }
    if args.world_training_mode == "sequence":
        metrics.update(measure_sequence_action_sensitivity(model, tensors, args))
    if was_training:
        model.train()
    return metrics


def measure_sequence_action_sensitivity(
    model: LatentDynamics,
    tensors: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, float]:
    sequence_starts = build_sequence_starts(tensors, args.sequence_length, args.sequence_stride)
    if sequence_starts.numel() < 2:
        return {}
    count = min(sequence_starts.shape[0], max(args.batch_size, min(512, sequence_starts.shape[0])))
    selected = sequence_starts[torch.randperm(sequence_starts.shape[0], device=tensors["z"].device)[:count]]
    offsets = torch.arange(args.sequence_length, device=tensors["z"].device)
    idx = selected[:, None] + offsets[None, :]
    z_seq = tensors["z"][idx]
    action_seq = tensors["action"][idx]
    target_seq = tensors["z_next"][idx]
    reward_seq = tensors["reward"][idx]
    shuffled_action_seq = action_seq[torch.randperm(action_seq.shape[0], device=action_seq.device)]
    with torch.no_grad():
        normal = model.forward_sequence(z_seq, action_seq)
        shuffled = model.forward_sequence(z_seq, shuffled_action_seq)
        normal_z_mse = F.mse_loss(normal["z_next"], target_seq)
        shuffled_z_mse = F.mse_loss(shuffled["z_next"], target_seq)
        normal_reward_mse = F.mse_loss(normal["reward"], reward_seq)
        shuffled_reward_mse = F.mse_loss(shuffled["reward"], reward_seq)
        normal_loss = normal_z_mse + args.reward_weight * normal_reward_mse
        shuffled_loss = shuffled_z_mse + args.reward_weight * shuffled_reward_mse
        action_effect = (normal["z_next"] - shuffled["z_next"]).pow(2).mean().sqrt()
    eps = 1e-8
    return {
        "sequence_action_probe_windows": float(count),
        "sequence_action_shuffle_z_mse": float(shuffled_z_mse.detach().cpu().item()),
        "sequence_action_shuffle_reward_mse": float(shuffled_reward_mse.detach().cpu().item()),
        "sequence_action_shuffle_loss_ratio": float((shuffled_loss / normal_loss.clamp_min(eps)).detach().cpu().item()),
        "sequence_action_shuffle_z_mse_ratio": float((shuffled_z_mse / normal_z_mse.clamp_min(eps)).detach().cpu().item()),
        "sequence_action_effect_rms": float(action_effect.detach().cpu().item()),
    }


def behavior_action(args: argparse.Namespace, rng: np.random.Generator, state: np.ndarray) -> np.ndarray:
    teacher = target_velocity_action(state).reshape(-1)
    if args.behavior == "teacher":
        return teacher.reshape(1, 4).astype(np.float32)
    if args.behavior == "random_mix" and rng.random() < args.random_action_prob:
        direction = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
        norm = float(np.linalg.norm(direction))
        if norm > 1.0:
            direction = direction / norm
        speed = np.float32(rng.uniform(0.0, 1.0))
        return np.array([[direction[0], direction[1], direction[2], speed]], dtype=np.float32)
    noisy = teacher + rng.normal(0.0, args.behavior_noise, size=4).astype(np.float32)
    return sanitize_action(noisy)


def shaped_reward(
    state: np.ndarray,
    next_state: np.ndarray,
    action: np.ndarray,
    args: argparse.Namespace,
    *,
    env_reward: float,
) -> float:
    del env_reward
    before = float(np.linalg.norm(TARGET_POS - state[0:3]))
    after = float(np.linalg.norm(TARGET_POS - next_state[0:3]))
    progress = before - after
    action_cost = 0.02 * float(np.linalg.norm(action.reshape(-1)))
    success_bonus = 1.0 if after <= args.success_distance_m else 0.0
    return 4.0 * progress - 0.15 * after - action_cost + success_bonus


def make_normalizer(z: np.ndarray) -> dict[str, list[float]]:
    mean = z.mean(axis=0)
    std = z.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean.astype(np.float32).tolist(), "std": std.astype(np.float32).tolist()}


def normalize_np(z: np.ndarray, normalizer: dict[str, list[float]]) -> np.ndarray:
    mean = np.asarray(normalizer["mean"], dtype=np.float32)
    std = np.asarray(normalizer["std"], dtype=np.float32)
    return ((z - mean) / std).astype(np.float32)


def transitions_to_tensors(batch: TransitionBatch, normalizer: dict[str, list[float]], device: torch.device) -> dict[str, torch.Tensor]:
    n = batch.z.shape[0]
    episode = batch.episode if batch.episode is not None else np.zeros(n, dtype=np.int64)
    step = batch.step if batch.step is not None else np.arange(n, dtype=np.int64)
    return {
        "z": torch.from_numpy(normalize_np(batch.z, normalizer)).to(device),
        "action": torch.from_numpy(batch.action).to(device),
        "reward": torch.from_numpy(batch.reward).to(device),
        "z_next": torch.from_numpy(normalize_np(batch.z_next, normalizer)).to(device),
        "done": torch.from_numpy(batch.done).to(device),
        "episode": torch.from_numpy(episode.astype(np.int64)).to(device),
        "step": torch.from_numpy(step.astype(np.int64)).to(device),
    }


def train_dynamics(model: LatentDynamics, tensors: dict[str, torch.Tensor], args: argparse.Namespace) -> list[dict[str, float]]:
    if args.world_training_mode == "sequence":
        return train_dynamics_sequence(model, tensors, args)
    return train_dynamics_one_step(model, tensors, args)


def train_dynamics_one_step(model: LatentDynamics, tensors: dict[str, torch.Tensor], args: argparse.Namespace) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.world_lr, weight_decay=1e-4)
    metrics: list[dict[str, float]] = []
    n = tensors["z"].shape[0]
    for epoch in range(1, args.world_epochs + 1):
        order = torch.randperm(n, device=tensors["z"].device)
        losses = []
        z_losses = []
        reward_losses = []
        done_losses = []
        model.train()
        for start in range(0, n, args.batch_size):
            idx = order[start : start + args.batch_size]
            out = model(tensors["z"][idx], tensors["action"][idx])
            z_loss = F.mse_loss(out["z_next"], tensors["z_next"][idx])
            reward_loss = F.mse_loss(out["reward"], tensors["reward"][idx])
            done_loss = F.binary_cross_entropy_with_logits(out["done_logit"], tensors["done"][idx])
            loss = z_loss + args.reward_weight * reward_loss + args.done_weight * done_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            z_losses.append(float(z_loss.detach().cpu().item()))
            reward_losses.append(float(reward_loss.detach().cpu().item()))
            done_losses.append(float(done_loss.detach().cpu().item()))
        metrics.append(
            {
                "epoch": float(epoch),
                "loss": float(np.mean(losses)),
                "z_mse": float(np.mean(z_losses)),
                "reward_mse": float(np.mean(reward_losses)),
                "done_bce": float(np.mean(done_losses)),
            }
        )
    model.eval()
    if metrics:
        metrics[-1].update(measure_action_sensitivity(model, tensors, args))
    return metrics


def train_dynamics_sequence(model: LatentDynamics, tensors: dict[str, torch.Tensor], args: argparse.Namespace) -> list[dict[str, float]]:
    sequence_starts = build_sequence_starts(tensors, args.sequence_length, args.sequence_stride)
    if sequence_starts.numel() == 0:
        raise RuntimeError(
            "No contiguous sequence windows found for sequence dynamics training. "
            "Recollect the dataset so transition caches include episode/step metadata."
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.world_lr, weight_decay=1e-4)
    metrics: list[dict[str, float]] = []
    n = sequence_starts.shape[0]
    offsets = torch.arange(args.sequence_length, device=tensors["z"].device)
    for epoch in range(1, args.world_epochs + 1):
        order = torch.randperm(n, device=tensors["z"].device)
        losses = []
        z_losses = []
        reward_losses = []
        done_losses = []
        model.train()
        for start in range(0, n, args.batch_size):
            batch_starts = sequence_starts[order[start : start + args.batch_size]]
            idx = batch_starts[:, None] + offsets[None, :]
            out = model.forward_sequence(tensors["z"][idx], tensors["action"][idx])
            z_loss = F.mse_loss(out["z_next"], tensors["z_next"][idx])
            reward_loss = F.mse_loss(out["reward"], tensors["reward"][idx])
            done_loss = F.binary_cross_entropy_with_logits(out["done_logit"], tensors["done"][idx])
            loss = z_loss + args.reward_weight * reward_loss + args.done_weight * done_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            z_losses.append(float(z_loss.detach().cpu().item()))
            reward_losses.append(float(reward_loss.detach().cpu().item()))
            done_losses.append(float(done_loss.detach().cpu().item()))
        metrics.append(
            {
                "epoch": float(epoch),
                "loss": float(np.mean(losses)),
                "z_mse": float(np.mean(z_losses)),
                "reward_mse": float(np.mean(reward_losses)),
                "done_bce": float(np.mean(done_losses)),
                "sequence_windows": float(n),
                "sequence_length": float(args.sequence_length),
            }
        )
    model.eval()
    if metrics:
        metrics[-1].update(measure_action_sensitivity(model, tensors, args))
    return metrics


def build_sequence_starts(tensors: dict[str, torch.Tensor], sequence_length: int, sequence_stride: int) -> torch.Tensor:
    n = int(tensors["z"].shape[0])
    length = max(1, int(sequence_length))
    stride = max(1, int(sequence_stride))
    if n < length:
        return torch.empty(0, dtype=torch.long, device=tensors["z"].device)
    episode = tensors["episode"].detach().cpu().numpy()
    step = tensors["step"].detach().cpu().numpy()
    starts: list[int] = []
    for start in range(0, n - length + 1, stride):
        end = start + length
        if np.all(episode[start:end] == episode[start]) and np.all(np.diff(step[start:end]) == 1):
            starts.append(start)
    return torch.tensor(starts, dtype=torch.long, device=tensors["z"].device)


def train_bc_actor(actor: Actor, tensors: dict[str, torch.Tensor], args: argparse.Namespace) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(actor.parameters(), lr=args.policy_lr, weight_decay=1e-4)
    metrics: list[dict[str, float]] = []
    n = tensors["z"].shape[0]
    for epoch in range(1, args.bc_epochs + 1):
        order = torch.randperm(n, device=tensors["z"].device)
        losses = []
        actor.train()
        for start in range(0, n, args.batch_size):
            idx = order[start : start + args.batch_size]
            pred = actor(tensors["z"][idx])
            loss = F.mse_loss(pred, tensors["action"][idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        metrics.append({"epoch": float(epoch), "bc_action_mse": float(np.mean(losses))})
    actor.eval()
    return metrics


def train_policy_in_imagination(
    dynamics: LatentDynamics,
    actor: Actor,
    prior_actor: Actor,
    critic: Critic,
    tensors: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> list[dict[str, float]]:
    for parameter in dynamics.parameters():
        parameter.requires_grad_(False)
    dynamics.eval()
    prior_actor.eval()
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=args.policy_lr, weight_decay=1e-5)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr, weight_decay=1e-4)
    metrics: list[dict[str, float]] = []
    n = tensors["z"].shape[0]
    device = tensors["z"].device

    for update in range(1, args.imagination_updates + 1):
        idx = torch.randint(0, n, (min(args.batch_size, n),), device=device)
        z0 = tensors["z"][idx]

        if args.imagination_objective == "pmpo":
            metric = train_pmpo_update(
                dynamics=dynamics,
                actor=actor,
                prior_actor=prior_actor,
                critic=critic,
                z0=z0,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                args=args,
            )
        else:
            metric = train_backprop_imagination_update(
                dynamics=dynamics,
                actor=actor,
                prior_actor=prior_actor,
                critic=critic,
                z0=z0,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                args=args,
            )
        metric["update"] = float(update)
        metrics.append(metric)
    actor.eval()
    critic.eval()
    return metrics


def train_pmpo_update(
    *,
    dynamics: LatentDynamics,
    actor: Actor,
    prior_actor: Actor,
    critic: Critic,
    z0: torch.Tensor,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> dict[str, float]:
    # Dreamer-style imagination: sample actions, unroll frozen world, then
    # update policy probabilities from advantage signs rather than direct
    # reward gradients through the model.
    with torch.no_grad():
        z = z0.detach()
        z_context: list[torch.Tensor] = []
        action_context: list[torch.Tensor] = []
        states = []
        actions = []
        rewards = []
        values = []
        entropies = []
        for _ in range(args.imagination_horizon):
            states.append(z)
            values.append(critic(z))
            action, _, entropy = actor.sample_action(z, args.policy_std)
            actions.append(action)
            entropies.append(entropy)
            z_context, action_context = append_imagination_context(
                z_context,
                action_context,
                z,
                action,
                max_context=max(1, int(args.sequence_length)),
            )
            out = dynamics_context_step(dynamics, z_context, action_context, args)
            rewards.append(out["reward"])
            z = out["z_next"]
        bootstrap = critic(z) if args.pmpo_bootstrap_value else torch.zeros_like(rewards[-1])
        returns = compute_lambda_returns(rewards, values, bootstrap, args.gamma, args.lambda_return)

    values_tensor = torch.stack([critic(state.detach()) for state in states], dim=0)
    returns_tensor = torch.stack([ret.detach() for ret in returns], dim=0)
    returns_tensor = clip_returns(returns_tensor, args.return_clip)
    critic_loss = F.mse_loss(values_tensor, returns_tensor)
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    maybe_clip_grad(critic.parameters(), args.max_grad_norm)
    critic_optimizer.step()

    states_tensor = torch.cat([state.detach() for state in states], dim=0)
    actions_tensor = torch.cat([action.detach() for action in actions], dim=0)
    returns_flat = returns_tensor.detach().reshape(-1)
    with torch.no_grad():
        values_for_advantage = critic(states_tensor).detach()
        advantage = returns_flat - values_for_advantage
        positive = advantage >= 0.0
        negative = ~positive

    log_prob = actor.log_prob(states_tensor, actions_tensor, args.policy_std)
    prior_kl = actor.prior_kl(states_tensor, prior_actor, args.policy_std)
    dist_entropy = torch.distributions.Normal(actor(states_tensor), torch.full_like(actions_tensor, args.policy_std)).entropy().sum(dim=-1)
    policy_loss = torch.zeros((), device=states_tensor.device)
    if bool(positive.any()):
        policy_loss = policy_loss - args.pmpo_alpha * log_prob[positive].mean()
    if bool(negative.any()):
        policy_loss = policy_loss + (1.0 - args.pmpo_alpha) * log_prob[negative].mean()
    actor_loss = policy_loss + args.prior_weight * prior_kl.mean() - args.entropy_weight * dist_entropy.mean()
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    maybe_clip_grad(actor.parameters(), args.max_grad_norm)
    actor_optimizer.step()

    reward_tensor = torch.stack(rewards, dim=0)
    action_tensor = torch.stack(actions, dim=0)
    return {
        "objective": "pmpo",
        "actor_loss": float(actor_loss.detach().cpu().item()),
        "critic_loss": float(critic_loss.detach().cpu().item()),
        "pmpo_policy_loss": float(policy_loss.detach().cpu().item()),
        "imagined_reward_mean": float(reward_tensor.mean().detach().cpu().item()),
        "imagined_return_mean": float(returns_tensor.mean().detach().cpu().item()),
        "advantage_mean": float(advantage.mean().detach().cpu().item()),
        "positive_advantage_fraction": float(positive.float().mean().detach().cpu().item()),
        "prior_kl": float(prior_kl.mean().detach().cpu().item()),
        "entropy": float(dist_entropy.mean().detach().cpu().item()),
        "action_norm": float(action_tensor.norm(dim=-1).mean().detach().cpu().item()),
        "imagination_context_length": float(min(args.sequence_length, args.imagination_horizon)),
    }


def train_backprop_imagination_update(
    *,
    dynamics: LatentDynamics,
    actor: Actor,
    prior_actor: Actor,
    critic: Critic,
    z0: torch.Tensor,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> dict[str, float]:
    with torch.no_grad():
        z = z0.detach()
        z_context: list[torch.Tensor] = []
        action_context: list[torch.Tensor] = []
        imagined_states = []
        rewards = []
        for _ in range(args.imagination_horizon):
            imagined_states.append(z)
            action = actor(z)
            z_context, action_context = append_imagination_context(
                z_context,
                action_context,
                z,
                action,
                max_context=max(1, int(args.sequence_length)),
            )
            out = dynamics_context_step(dynamics, z_context, action_context, args)
            rewards.append(out["reward"])
            z = out["z_next"]
        bootstrap = torch.zeros_like(rewards[-1])
        returns = compute_returns(rewards, bootstrap, args.gamma)
    values_tensor = torch.stack([critic(state.detach()) for state in imagined_states], dim=0)
    returns_tensor = torch.stack([ret.detach() for ret in returns], dim=0)
    critic_loss = F.mse_loss(values_tensor, returns_tensor)
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    maybe_clip_grad(critic.parameters(), args.max_grad_norm)
    critic_optimizer.step()

    z = z0
    z_context = []
    action_context = []
    reward_terms = []
    prior_terms = []
    action_norms = []
    imagined_objective = torch.zeros((), device=z0.device)
    discount = 1.0
    for _ in range(args.imagination_horizon):
        action = actor(z)
        with torch.no_grad():
            prior_action = prior_actor(z)
        z_context, action_context = append_imagination_context(
            z_context,
            action_context,
            z,
            action,
            max_context=max(1, int(args.sequence_length)),
        )
        out = dynamics_context_step(dynamics, z_context, action_context, args)
        reward_mean = out["reward"].mean()
        reward_terms.append(reward_mean)
        prior_terms.append(F.mse_loss(action, prior_action))
        action_norms.append(action.norm(dim=-1).mean())
        imagined_objective = imagined_objective + discount * reward_mean
        discount *= args.gamma
        z = out["z_next"]
    actor_loss = -(imagined_objective / max(1, args.imagination_horizon)) + args.prior_weight * torch.stack(prior_terms).mean()
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    maybe_clip_grad(actor.parameters(), args.max_grad_norm)
    actor_optimizer.step()
    return {
        "objective": "backprop",
        "actor_loss": float(actor_loss.detach().cpu().item()),
        "critic_loss": float(critic_loss.detach().cpu().item()),
        "imagined_reward_mean": float(torch.stack(reward_terms).mean().detach().cpu().item()),
        "imagined_objective": float(imagined_objective.detach().cpu().item()),
        "prior_mse": float(torch.stack(prior_terms).mean().detach().cpu().item()),
        "action_norm": float(torch.stack(action_norms).mean().detach().cpu().item()),
        "imagination_context_length": float(min(args.sequence_length, args.imagination_horizon)),
    }


def append_imagination_context(
    z_context: list[torch.Tensor],
    action_context: list[torch.Tensor],
    z: torch.Tensor,
    action: torch.Tensor,
    *,
    max_context: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    z_next_context = [*z_context, z]
    action_next_context = [*action_context, action]
    if len(z_next_context) > max_context:
        z_next_context = z_next_context[-max_context:]
        action_next_context = action_next_context[-max_context:]
    return z_next_context, action_next_context


def dynamics_context_step(
    dynamics: LatentDynamics,
    z_context: list[torch.Tensor],
    action_context: list[torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    if len(z_context) != len(action_context):
        raise ValueError("z_context and action_context must have the same length")
    if args.world_training_mode == "sequence":
        out = dynamics.forward_sequence(
            torch.stack(z_context, dim=1),
            torch.stack(action_context, dim=1),
        )
        return {
            "z_next": out["z_next"][:, -1],
            "reward": out["reward"][:, -1],
            "done_logit": out["done_logit"][:, -1],
        }
    return dynamics(z_context[-1], action_context[-1])


def compute_returns(rewards: list[torch.Tensor], bootstrap: torch.Tensor, gamma: float) -> list[torch.Tensor]:
    ret = bootstrap
    returns: list[torch.Tensor] = []
    for reward in reversed(rewards):
        ret = reward + gamma * ret
        returns.append(ret)
    returns.reverse()
    return returns


def compute_lambda_returns(
    rewards: list[torch.Tensor],
    values: list[torch.Tensor],
    bootstrap: torch.Tensor,
    gamma: float,
    lambda_return: float,
) -> list[torch.Tensor]:
    ret = bootstrap
    returns: list[torch.Tensor] = []
    for reward, value in reversed(list(zip(rewards, values))):
        ret = reward + gamma * ((1.0 - lambda_return) * value + lambda_return * ret)
        returns.append(ret)
    returns.reverse()
    return returns


def clip_returns(returns: torch.Tensor, limit: float) -> torch.Tensor:
    if limit and limit > 0:
        return returns.clamp(min=-float(limit), max=float(limit))
    return returns


def maybe_clip_grad(parameters, max_norm: float) -> None:
    if max_norm and max_norm > 0:
        torch.nn.utils.clip_grad_norm_(list(parameters), max_norm)


def evaluate_actor(
    actor: Actor,
    *,
    encoder: AgentObservationEncoder,
    normalizer: dict[str, list[float]],
    args: argparse.Namespace,
    imports: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    records = []
    device = next(actor.parameters()).device
    eval_seeds = evaluation_seeds(args)
    for episode_idx, seed in enumerate(eval_seeds):
        set_seed(seed)
        initial_xyz = sample_initial_xyz(args, seed)
        env = make_env(args, imports, initial_xyz=initial_xyz)
        frames = []
        total_reward = 0.0
        final_distance = float("inf")
        min_distance = float("inf")
        terminated = False
        truncated = False
        try:
            encoder.reset()
            obs, info = env.reset(seed=seed)
            del info
            prev_action = np.zeros(4, dtype=np.float32)
            for step in range(args.max_steps):
                state = env._getDroneStateVector(0).astype(np.float32)
                distance = float(np.linalg.norm(TARGET_POS - state[0:3]))
                final_distance = distance
                min_distance = min(min_distance, distance)
                frame = rgb_from_obs(obs)
                if should_capture_frame(step, args.eval_trace_frames, args.max_steps):
                    frames.append(frame)
                z = encoder.encode(frame, state, prev_action)
                z_norm = torch.from_numpy(normalize_np(z[None, :], normalizer)).to(device)
                with torch.no_grad():
                    action_raw = actor(z_norm).detach().cpu().numpy()[0]
                action = sanitize_action(action_raw)
                next_obs, env_reward, terminated, truncated, info = env.step(action)
                del info
                next_state = env._getDroneStateVector(0).astype(np.float32)
                total_reward += shaped_reward(state, next_state, action, args, env_reward=float(env_reward))
                obs = next_obs
                prev_action = action.reshape(-1).astype(np.float32)
                if terminated or truncated:
                    break
        finally:
            env.close()
        trace_path = None
        if frames:
            trace_path = traces_dir / f"eval_{seed}.png"
            make_frame_contact_sheet(frames, trace_path, label=f"eval seed={seed}")
        records.append(
            {
                "seed": seed,
                "initial_xyz": initial_xyz.reshape(-1).astype(float).tolist(),
                "steps": step + 1 if "step" in locals() else 0,
                "return": float(total_reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "success": bool(final_distance <= args.success_distance_m),
                "final_distance_m": final_distance,
                "min_distance_m": min_distance,
                "trace_contact_sheet": str(trace_path) if trace_path else None,
            }
        )
    write_jsonl(out_dir / "episodes.jsonl", records)
    trace_paths = [Path(record["trace_contact_sheet"]) for record in records if record.get("trace_contact_sheet")]
    if trace_paths:
        make_combined_contact_sheet(trace_paths, out_dir / "contact_sheet.png")
    n = max(1, len(records))
    return {
        "episodes": len(records),
        "seeds": [int(seed) for seed in eval_seeds],
        "success_rate": sum(1 for record in records if record["success"]) / n,
        "mean_return": float(np.mean([record["return"] for record in records])),
        "mean_final_distance_m": float(np.mean([record["final_distance_m"] for record in records])),
        "mean_min_distance_m": float(np.mean([record["min_distance_m"] for record in records])),
        "episodes_path": str(out_dir / "episodes.jsonl"),
        "contact_sheet": str(out_dir / "contact_sheet.png") if trace_paths else None,
    }


def evaluation_seeds(args: argparse.Namespace) -> list[int]:
    if args.eval_seeds.strip():
        seeds = [int(item.strip()) for item in args.eval_seeds.split(",") if item.strip()]
        if not seeds:
            raise ValueError("--eval-seeds was provided but no valid seeds were parsed.")
        return seeds
    return [int(args.seed + args.eval_seed_offset + idx) for idx in range(args.eval_episodes)]


def should_capture_frame(step: int, trace_frames: int, max_steps: int) -> bool:
    if trace_frames <= 0:
        return False
    return step % max(1, max_steps // trace_frames) == 0


def make_frame_contact_sheet(frames: list[np.ndarray], out_path: Path, label: str) -> None:
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


def save_dataset(
    out_dir: Path,
    transitions: TransitionBatch,
    records: list[dict[str, Any]],
    normalizer: dict[str, list[float]],
) -> None:
    arrays = {
        "z": transitions.z,
        "action": transitions.action,
        "reward": transitions.reward,
        "z_next": transitions.z_next,
        "done": transitions.done,
    }
    if transitions.episode is not None:
        arrays["episode"] = transitions.episode
    if transitions.step is not None:
        arrays["step"] = transitions.step
    np.savez_compressed(out_dir / "transitions.npz", **arrays)
    (out_dir / "transition_preview.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (out_dir / "normalizer.json").write_text(json.dumps(normalizer, indent=2), encoding="utf-8")


def write_report(summary: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# PyBullet Latent Imagination Policy",
        "",
        "This trains a policy inside a learned latent simulator and evaluates it back in PyBullet.",
        "",
        "## Result",
        "",
        f"- Feature: `{summary['dataset']['feature']}`",
        f"- Transitions: `{summary['dataset']['transitions']}`",
        f"- BC prior success: `{summary['bc_prior_eval']['success_rate']}`",
        f"- BC prior final distance: `{summary['bc_prior_eval']['mean_final_distance_m']}`",
        f"- After imagination success: `{summary['after_imagination_eval']['success_rate']}`",
        f"- After imagination final distance: `{summary['after_imagination_eval']['mean_final_distance_m']}`",
        f"- Selected actor: `{summary['policy_selection']['selected_actor']}`",
        f"- Selection reason: {summary['policy_selection']['reason']}",
        "",
        "## Claim Boundary",
        "",
        summary["claim_boundary"],
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def summarize_last(records: list[dict[str, Any]]) -> dict[str, Any]:
    return dict(records[-1]) if records else {}


def select_policy(bc_eval: dict[str, Any], imagined_eval: dict[str, Any]) -> dict[str, Any]:
    bc_success = float(bc_eval["success_rate"])
    imagined_success = float(imagined_eval["success_rate"])
    bc_distance = float(bc_eval["mean_final_distance_m"])
    imagined_distance = float(imagined_eval["mean_final_distance_m"])
    imagination_better = (
        imagined_success > bc_success
        or (imagined_success == bc_success and imagined_distance <= bc_distance)
    )
    selected = "after_imagination" if imagination_better else "bc_prior"
    return {
        "selected_actor": selected,
        "imagination_promoted": imagination_better,
        "reason": (
            "after_imagination improved success or tied success with lower final distance"
            if imagination_better
            else "after_imagination regressed against BC prior; selected BC prior checkpoint"
        ),
        "bc_success_rate": bc_success,
        "after_imagination_success_rate": imagined_success,
        "bc_mean_final_distance_m": bc_distance,
        "after_imagination_mean_final_distance_m": imagined_distance,
    }


def resolve_out_dir(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def resolve_optional_path(value: str) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


if __name__ == "__main__":
    raise SystemExit(main())

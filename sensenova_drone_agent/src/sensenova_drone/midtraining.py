from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None
    nn = None
    Dataset = object  # type: ignore[assignment]


@dataclass
class SequenceCache:
    """
    Generic phase-2 midtraining cache.

    Required:
        z:      (N, z_dim) frozen world-model/Kairos features
        action: (N, action_dim) behavior action labels

    Optional:
        reward:  (N,) scalar reward/success/hindsight score
        episode: (N,) trajectory id
        step:    (N,) timestep within trajectory
        task_id: (N,) integer task id
        done:    (N,) terminal flag for each sequence row
    """

    z: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    episode: np.ndarray
    step: np.ndarray
    task_id: np.ndarray
    done: np.ndarray
    source_path: str | None = None

    @property
    def num_steps(self) -> int:
        return int(self.z.shape[0])

    @property
    def z_dim(self) -> int:
        return int(self.z.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.action.shape[1])

    @property
    def num_tasks(self) -> int:
        return int(np.max(self.task_id)) + 1 if self.task_id.size else 1


@dataclass
class Normalizer:
    z_mean: list[float]
    z_std: list[float]
    action_mean: list[float]
    action_std: list[float]
    reward_mean: float
    reward_std: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CONTROL_MODES = {
    "normal",
    "shuffle_targets",
    "shuffle_z_context",
    "zero_z_context",
    "zero_prev_actions",
}

DYNAMICS_CONTROL_MODES = {
    "normal",
    "shuffle_future_actions",
    "zero_future_actions",
    "shuffle_z_context",
    "zero_z_context",
    "shuffle_targets",
    "zero_action_context",
}


def load_sequence_cache(path: str | Path) -> SequenceCache:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)
    z_key = first_present(keys, ["z", "latent", "latents", "features", "feature"])
    action_key = first_present(keys, ["action", "actions", "a"])
    if z_key is None:
        raise ValueError(f"{path} must contain one of: z, latent, latents, features, feature")
    if action_key is None:
        raise ValueError(f"{path} must contain one of: action, actions, a")

    z = np.asarray(data[z_key], dtype=np.float32)
    action = np.asarray(data[action_key], dtype=np.float32)
    if z.ndim != 2:
        raise ValueError(f"z/features must be rank-2 (N,D), got {z.shape}")
    if action.ndim == 1:
        action = action[:, None]
    if action.ndim != 2:
        raise ValueError(f"action must be rank-1 or rank-2, got {action.shape}")
    if action.shape[0] != z.shape[0]:
        raise ValueError(f"z/action length mismatch: {z.shape[0]} vs {action.shape[0]}")

    n = int(z.shape[0])
    reward = np.asarray(data["reward"], dtype=np.float32) if "reward" in keys else np.zeros(n, dtype=np.float32)
    if reward.ndim == 2 and reward.shape[-1] == 1:
        reward = reward[:, 0]
    if reward.ndim != 1 or reward.shape[0] != n:
        raise ValueError(f"reward must have shape (N,), got {reward.shape}")

    episode = np.asarray(data["episode"], dtype=np.int64) if "episode" in keys else np.zeros(n, dtype=np.int64)
    step = np.asarray(data["step"], dtype=np.int64) if "step" in keys else np.arange(n, dtype=np.int64)
    task_id = np.asarray(data["task_id"], dtype=np.int64) if "task_id" in keys else np.zeros(n, dtype=np.int64)
    for name, value in [("episode", episode), ("step", step), ("task_id", task_id)]:
        if value.ndim != 1 or value.shape[0] != n:
            raise ValueError(f"{name} must have shape (N,), got {value.shape}")
    if "done" in keys:
        done = np.asarray(data["done"], dtype=bool)
        if done.ndim == 2 and done.shape[-1] == 1:
            done = done[:, 0]
        if done.ndim != 1 or done.shape[0] != n:
            raise ValueError(f"done must have shape (N,), got {done.shape}")
    else:
        done = infer_done_flags(episode=episode, step=step)

    return SequenceCache(
        z=z,
        action=action,
        reward=reward,
        episode=episode,
        step=step,
        task_id=task_id,
        done=done,
        source_path=str(path),
    )


def infer_done_flags(*, episode: np.ndarray, step: np.ndarray) -> np.ndarray:
    """Infer terminal rows for old sequence caches that did not store done."""

    episode = np.asarray(episode)
    step = np.asarray(step)
    if episode.ndim != 1 or step.ndim != 1 or episode.shape[0] != step.shape[0]:
        raise ValueError("episode and step must be rank-1 arrays with matching length")
    n = int(episode.shape[0])
    done = np.zeros(n, dtype=bool)
    if n <= 0:
        return done
    if n > 1:
        done[:-1] = (episode[1:] != episode[:-1]) | (step[1:] <= step[:-1])
    done[-1] = True
    return done


def first_present(keys: set[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in keys:
            return candidate
    return None


def build_valid_anchors(cache: SequenceCache, *, context_len: int, mtp_horizon: int) -> np.ndarray:
    """
    Anchors are timesteps t where the model sees z[t-context+1:t] plus previous
    actions and predicts action/reward targets t..t+H.

    The previous-action context needs action[t-context], so the checked span is:
        t-context .. t+H
    """

    context_len = int(context_len)
    mtp_horizon = int(mtp_horizon)
    if context_len < 1:
        raise ValueError("context_len must be >= 1")
    if mtp_horizon < 0:
        raise ValueError("mtp_horizon must be >= 0")
    anchors: list[int] = []
    n = cache.num_steps
    for anchor in range(context_len, n - mtp_horizon):
        span_start = anchor - context_len
        span_end = anchor + mtp_horizon
        eps = cache.episode[span_start : span_end + 1]
        steps = cache.step[span_start : span_end + 1]
        if eps.size != (context_len + mtp_horizon + 1):
            continue
        if not np.all(eps == eps[0]):
            continue
        if not np.all(np.diff(steps) == 1):
            continue
        anchors.append(anchor)
    return np.asarray(anchors, dtype=np.int64)


def build_valid_dynamics_anchors(
    cache: SequenceCache,
    *,
    context_len: int,
    prediction_horizon: int,
    future_action_offset: int = 0,
) -> np.ndarray:
    """
    Anchors for action-conditioned latent dynamics.

    The model sees z[t-C+1:t], action context a[t-C+1:t], future/candidate
    actions a[t+offset:t+offset+H-1], and predicts z[t+1:t+H].
    Offsets let us audit dataset-specific action/frame alignment.
    """

    context_len = int(context_len)
    prediction_horizon = int(prediction_horizon)
    future_action_offset = int(future_action_offset)
    if context_len < 1:
        raise ValueError("context_len must be >= 1")
    if prediction_horizon < 1:
        raise ValueError("prediction_horizon must be >= 1")
    anchors: list[int] = []
    n = cache.num_steps
    for anchor in range(context_len, n - prediction_horizon):
        span_start = min(anchor - context_len + 1, anchor + future_action_offset)
        span_end = max(anchor + prediction_horizon, anchor + future_action_offset + prediction_horizon - 1)
        if span_start < 0 or span_end >= n:
            continue
        eps = cache.episode[span_start : span_end + 1]
        steps = cache.step[span_start : span_end + 1]
        if eps.size != (span_end - span_start + 1):
            continue
        if not np.all(eps == eps[0]):
            continue
        if not np.all(np.diff(steps) == 1):
            continue
        anchors.append(anchor)
    return np.asarray(anchors, dtype=np.int64)


def split_anchors(
    anchors: np.ndarray,
    *,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(anchors, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    val_count = int(round(len(shuffled) * float(val_ratio)))
    val_count = min(max(val_count, 0), len(shuffled))
    val = np.sort(shuffled[:val_count])
    train = np.sort(shuffled[val_count:])
    return train, val


def split_anchors_by_episode(
    cache: SequenceCache,
    anchors: np.ndarray,
    *,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split anchors by held-out episodes to avoid trajectory leakage."""

    anchors = np.asarray(anchors, dtype=np.int64)
    if anchors.size <= 0:
        return anchors, anchors
    anchor_episodes = cache.episode[anchors]
    episodes = np.unique(anchor_episodes)
    if episodes.size <= 1 or val_ratio <= 0:
        return np.sort(anchors), np.asarray([], dtype=np.int64)

    rng = np.random.default_rng(seed)
    shuffled_episodes = episodes.copy()
    rng.shuffle(shuffled_episodes)
    val_count = int(round(episodes.size * float(val_ratio)))
    val_count = min(max(val_count, 1), episodes.size - 1)
    val_episodes = set(int(ep) for ep in shuffled_episodes[:val_count])
    val_mask = np.asarray([int(ep) in val_episodes for ep in anchor_episodes], dtype=bool)
    train = np.sort(anchors[~val_mask])
    val = np.sort(anchors[val_mask])
    return train, val


def split_anchors_by_task_episode(
    cache: SequenceCache,
    anchors: np.ndarray,
    *,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hold out complete episodes while keeping validation tasks represented in train.

    Small SOAR subsets can contain many task strings with only one trajectory.
    A pure random episode split can therefore evaluate unseen tasks rather than
    held-out behavior for known tasks. Singleton-task episodes stay in train.
    """

    anchors = np.asarray(anchors, dtype=np.int64)
    if anchors.size <= 0:
        return anchors, anchors
    rng = np.random.default_rng(seed)
    episode_ids = np.unique(cache.episode[anchors])
    episode_task: dict[int, int] = {}
    for episode in episode_ids:
        indices = np.flatnonzero(cache.episode == episode)
        if indices.size <= 0:
            continue
        values, counts = np.unique(cache.task_id[indices], return_counts=True)
        episode_task[int(episode)] = int(values[np.argmax(counts)])

    task_to_episodes: dict[int, list[int]] = {}
    for episode, task in episode_task.items():
        task_to_episodes.setdefault(task, []).append(episode)

    val_episodes: set[int] = set()
    for episodes in task_to_episodes.values():
        episodes_array = np.asarray(episodes, dtype=np.int64)
        if episodes_array.size <= 1:
            continue
        shuffled = episodes_array.copy()
        rng.shuffle(shuffled)
        val_count = int(round(episodes_array.size * float(val_ratio)))
        val_count = min(max(val_count, 1), episodes_array.size - 1)
        val_episodes.update(int(ep) for ep in shuffled[:val_count])

    anchor_episodes = cache.episode[anchors]
    val_mask = np.asarray([int(ep) in val_episodes for ep in anchor_episodes], dtype=bool)
    train = np.sort(anchors[~val_mask])
    val = np.sort(anchors[val_mask])
    return train, val


def compute_normalizer(cache: SequenceCache) -> Normalizer:
    z_mean = cache.z.mean(axis=0)
    z_std = cache.z.std(axis=0)
    action_mean = cache.action.mean(axis=0)
    action_std = cache.action.std(axis=0)
    reward_mean = float(cache.reward.mean())
    reward_std = float(cache.reward.std())
    return Normalizer(
        z_mean=z_mean.astype(float).tolist(),
        z_std=np.maximum(z_std, 1e-6).astype(float).tolist(),
        action_mean=action_mean.astype(float).tolist(),
        action_std=np.maximum(action_std, 1e-6).astype(float).tolist(),
        reward_mean=reward_mean,
        reward_std=max(reward_std, 1e-6),
    )


def cache_summary(cache: SequenceCache, anchors: np.ndarray | None = None) -> dict[str, Any]:
    episodes = np.unique(cache.episode)
    tasks = np.unique(cache.task_id)
    summary = {
        "source_path": cache.source_path,
        "steps": cache.num_steps,
        "z_dim": cache.z_dim,
        "action_dim": cache.action_dim,
        "episodes": int(episodes.size),
        "tasks": int(tasks.size),
        "reward_min": float(np.min(cache.reward)) if cache.reward.size else 0.0,
        "reward_mean": float(np.mean(cache.reward)) if cache.reward.size else 0.0,
        "reward_max": float(np.max(cache.reward)) if cache.reward.size else 0.0,
        "action_mean_abs": float(np.mean(np.abs(cache.action))) if cache.action.size else 0.0,
        "done_count": int(np.sum(cache.done)) if cache.done.size else 0,
        "done_fraction": float(np.mean(cache.done)) if cache.done.size else 0.0,
    }
    if anchors is not None:
        summary["valid_anchors"] = int(len(anchors))
    return summary


def make_smoke_sequence_cache(
    path: str | Path,
    *,
    episodes: int = 8,
    steps: int = 48,
    z_dim: int = 16,
    action_dim: int = 3,
    persistence: float = 0.65,
    action_scale: float = 0.35,
    noise_scale: float = 0.02,
    random_action_fraction: float = 0.65,
) -> Path:
    """
    Create a tiny deterministic cache where future actions causally affect
    future latents. The action includes an exogenous component so action-shuffle
    controls should be measurably worse than normal action conditioning.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12345)
    total = episodes * steps
    z = np.zeros((total, z_dim), dtype=np.float32)
    action = np.zeros((total, action_dim), dtype=np.float32)
    reward = np.zeros(total, dtype=np.float32)
    episode = np.zeros(total, dtype=np.int64)
    step = np.zeros(total, dtype=np.int64)
    task_id = np.zeros(total, dtype=np.int64)
    done = np.zeros(total, dtype=bool)

    w = rng.normal(size=(z_dim, action_dim)).astype(np.float32) / np.sqrt(z_dim)
    idx = 0
    for ep in range(episodes):
        state = rng.normal(size=z_dim).astype(np.float32)
        task = ep % 2
        for t in range(steps):
            z[idx] = state
            policy_action = np.tanh(state @ w + 0.15 * task)
            random_action = np.tanh(rng.normal(size=action_dim).astype(np.float32))
            action[idx] = (
                (1.0 - float(random_action_fraction)) * policy_action
                + float(random_action_fraction) * random_action
            )
            reward[idx] = float(1.0 - np.linalg.norm(action[idx]) / max(action_dim, 1))
            episode[idx] = ep
            step[idx] = t
            task_id[idx] = task
            done[idx] = t == steps - 1
            task_bias = 0.03 if task else -0.03
            state = (
                float(persistence) * state
                + float(noise_scale) * rng.normal(size=z_dim).astype(np.float32)
                + float(action_scale) * (w @ action[idx])
                + task_bias
            )
            idx += 1

    np.savez_compressed(
        path,
        z=z,
        action=action,
        reward=reward,
        episode=episode,
        step=step,
        task_id=task_id,
        done=done,
    )
    return path


if nn is not None:

    def make_mlp_head(
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> nn.Module:
        """Build a configurable output head while preserving linear-head defaults."""

        input_dim = int(input_dim)
        output_dim = int(output_dim)
        hidden_dim = int(hidden_dim or input_dim)
        num_layers = int(num_layers)
        if num_layers <= 1:
            return nn.Linear(input_dim, output_dim)

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()])
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)


    class ActionConditionedLatentDynamics(nn.Module):
        """
        Dreamer-style controllable latent simulator over frozen Kairos/Wan states.

        It learns:

            z[t-C+1:t], a[t-C+1:t], a[t:t+H-1], task -> z[t+1:t+H]

        The frozen Kairos/Sensenova model is not updated. This head is the
        missing controllable simulator needed before imagination RL.
        """

        def __init__(
            self,
            *,
            z_dim: int,
            action_dim: int,
            hidden_dim: int = 256,
            context_len: int = 8,
            prediction_horizon: int = 8,
            num_tasks: int = 1,
            num_layers: int = 2,
            num_heads: int = 4,
            dropout: float = 0.0,
            predict_delta: bool = True,
            head_hidden_dim: int | None = None,
            head_layers: int = 1,
            architecture: str = "pooled",
            residual_mode: str = "none",
        ):
            super().__init__()
            if hidden_dim % num_heads != 0:
                raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
            if architecture not in {"pooled", "action_query"}:
                raise ValueError(f"Unsupported dynamics architecture: {architecture}")
            if residual_mode not in {"none", "action_gated"}:
                raise ValueError(f"Unsupported dynamics residual mode: {residual_mode}")
            self.z_dim = int(z_dim)
            self.action_dim = int(action_dim)
            self.hidden_dim = int(hidden_dim)
            self.context_len = int(context_len)
            self.prediction_horizon = int(prediction_horizon)
            self.num_tasks = max(1, int(num_tasks))
            self.predict_delta = bool(predict_delta)
            self.architecture = architecture
            self.residual_mode = residual_mode
            self.head_hidden_dim = int(head_hidden_dim or hidden_dim)
            self.head_layers = int(head_layers)
            self.z_token = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
            self.action_token = nn.Sequential(nn.Linear(action_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
            self.task_token = nn.Embedding(self.num_tasks, hidden_dim)
            self.rollout_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.query_token = nn.Parameter(torch.zeros(1, self.prediction_horizon, hidden_dim))
            self.token_type = nn.Parameter(torch.empty(6, hidden_dim))
            nn.init.normal_(self.token_type, std=0.02)
            max_tokens = 2 * self.context_len + 2 * self.prediction_horizon + 2
            self.position = nn.Parameter(torch.zeros(1, max_tokens, hidden_dim))
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.post = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
            self.future_heads = nn.ModuleList(
                make_mlp_head(
                    hidden_dim,
                    z_dim,
                    hidden_dim=self.head_hidden_dim,
                    num_layers=self.head_layers,
                    dropout=dropout,
                )
                for _ in range(self.prediction_horizon)
            )
            self.gate_heads = nn.ModuleList(
                make_mlp_head(
                    hidden_dim,
                    z_dim,
                    hidden_dim=self.head_hidden_dim,
                    num_layers=self.head_layers,
                    dropout=dropout,
                )
                for _ in range(self.prediction_horizon)
            )

        def forward(
            self,
            z_context: torch.Tensor,
            action_context: torch.Tensor,
            future_action: torch.Tensor,
            task_id: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            if z_context.dim() != 3:
                raise ValueError(f"z_context must have shape (B,C,Z), got {tuple(z_context.shape)}")
            if action_context.dim() != 3:
                raise ValueError(f"action_context must have shape (B,C,A), got {tuple(action_context.shape)}")
            if future_action.dim() != 3:
                raise ValueError(f"future_action must have shape (B,H,A), got {tuple(future_action.shape)}")
            if z_context.shape[:2] != action_context.shape[:2]:
                raise ValueError("z_context and action_context must have the same batch/context shape")
            if z_context.shape[1] != self.context_len:
                raise ValueError(f"expected context_len={self.context_len}, got {z_context.shape[1]}")
            if future_action.shape[1] != self.prediction_horizon:
                raise ValueError(f"expected prediction_horizon={self.prediction_horizon}, got {future_action.shape[1]}")

            batch = z_context.shape[0]
            if task_id is None:
                task_id = torch.zeros(batch, dtype=torch.long, device=z_context.device)
            task_id = task_id.long().clamp(min=0, max=self.num_tasks - 1)

            z_tokens = self.z_token(z_context) + self.token_type[0].view(1, 1, -1)
            action_tokens = self.action_token(action_context) + self.token_type[1].view(1, 1, -1)
            context_tokens = torch.stack([z_tokens, action_tokens], dim=2).flatten(1, 2)
            future_action_tokens = self.action_token(future_action) + self.token_type[2].view(1, 1, -1)
            task_token = self.task_token(task_id).unsqueeze(1) + self.token_type[3].view(1, 1, -1)
            if self.architecture == "action_query":
                query_tokens = (
                    self.query_token[:, : self.prediction_horizon, :].expand(batch, -1, -1)
                    + self.token_type[5].view(1, 1, -1)
                )
                future_tokens = torch.stack([future_action_tokens, query_tokens], dim=2).flatten(1, 2)
                tokens = torch.cat([task_token, context_tokens, future_tokens], dim=1)
            else:
                rollout_token = self.rollout_token.expand(batch, -1, -1) + self.token_type[4].view(1, 1, -1)
                tokens = torch.cat([task_token, context_tokens, future_action_tokens, rollout_token], dim=1)
            tokens = tokens + self.position[:, : tokens.shape[1], :]
            encoded = self.encoder(tokens)
            if self.architecture == "action_query":
                query_start = 1 + 2 * self.context_len + 1
                query_indices = torch.arange(
                    query_start,
                    query_start + 2 * self.prediction_horizon,
                    2,
                    device=encoded.device,
                )
                h_steps = self.post(encoded.index_select(1, query_indices))
                h_t = h_steps[:, -1, :]
            else:
                h_t = self.post(encoded[:, -1, :])
                h_steps = h_t.unsqueeze(1).expand(-1, self.prediction_horizon, -1)

            residual = torch.stack(
                [head(h_steps[:, index, :]) for index, head in enumerate(self.future_heads)],
                dim=1,
            )
            last_z = z_context[:, -1:, :]
            gates = None
            if self.residual_mode == "action_gated":
                gates = torch.stack(
                    [torch.sigmoid(head(h_steps[:, index, :])) for index, head in enumerate(self.gate_heads)],
                    dim=1,
                )
                predicted = last_z + gates * residual
            elif self.predict_delta:
                predicted = last_z + residual
            else:
                predicted = residual
            return {"h_t": h_t, "predicted_z": predicted, "residual_gate": gates}

    class BehaviorCloningMidtrainingHead(nn.Module):
        """
        Dreamer-style phase-2 head.

        It does not update Kairos/Sensenova. It receives frozen world-model
        states z_t, previous action context, and an optional task id, then
        predicts multi-token future actions and rewards:

            h_t -> a[t:t+L], r[t:t+L]
        """

        def __init__(
            self,
            *,
            z_dim: int,
            action_dim: int,
            hidden_dim: int = 256,
            context_len: int = 8,
            mtp_horizon: int = 8,
            num_tasks: int = 1,
            num_layers: int = 2,
            num_heads: int = 4,
            dropout: float = 0.0,
            agent_token_isolation: bool = True,
            head_hidden_dim: int | None = None,
            head_layers: int = 1,
        ):
            super().__init__()
            if hidden_dim % num_heads != 0:
                raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
            self.z_dim = int(z_dim)
            self.action_dim = int(action_dim)
            self.hidden_dim = int(hidden_dim)
            self.context_len = int(context_len)
            self.mtp_horizon = int(mtp_horizon)
            self.num_tasks = max(1, int(num_tasks))
            self.agent_token_isolation = bool(agent_token_isolation)
            self.head_hidden_dim = int(head_hidden_dim or hidden_dim)
            self.head_layers = int(head_layers)
            self.z_token = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
            self.action_token = nn.Sequential(nn.Linear(action_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
            self.task_token = nn.Embedding(self.num_tasks, hidden_dim)
            self.agent_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.token_type = nn.Parameter(torch.empty(4, hidden_dim))
            nn.init.normal_(self.token_type, std=0.02)
            max_tokens = 2 * self.context_len + 2
            self.position = nn.Parameter(torch.zeros(1, max_tokens, hidden_dim))
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.post = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
            outputs = self.mtp_horizon + 1
            self.action_heads = nn.ModuleList(
                make_mlp_head(
                    hidden_dim,
                    action_dim,
                    hidden_dim=self.head_hidden_dim,
                    num_layers=self.head_layers,
                    dropout=dropout,
                )
                for _ in range(outputs)
            )
            self.reward_heads = nn.ModuleList(
                make_mlp_head(
                    hidden_dim,
                    1,
                    hidden_dim=self.head_hidden_dim,
                    num_layers=self.head_layers,
                    dropout=dropout,
                )
                for _ in range(outputs)
            )
            self.value_head = make_mlp_head(
                hidden_dim,
                1,
                hidden_dim=self.head_hidden_dim,
                num_layers=self.head_layers,
                dropout=dropout,
            )

        def forward(
            self,
            z_context: torch.Tensor,
            prev_action_context: torch.Tensor,
            task_id: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            if z_context.dim() != 3:
                raise ValueError(f"z_context must have shape (B,C,Z), got {tuple(z_context.shape)}")
            if prev_action_context.dim() != 3:
                raise ValueError(
                    f"prev_action_context must have shape (B,C,A), got {tuple(prev_action_context.shape)}"
                )
            if z_context.shape[:2] != prev_action_context.shape[:2]:
                raise ValueError("z_context and prev_action_context must have the same batch/context shape")
            if z_context.shape[1] != self.context_len:
                raise ValueError(f"expected context_len={self.context_len}, got {z_context.shape[1]}")
            batch = z_context.shape[0]
            if task_id is None:
                task_id = torch.zeros(batch, dtype=torch.long, device=z_context.device)
            task_id = task_id.long().clamp(min=0, max=self.num_tasks - 1)

            z_tokens = self.z_token(z_context) + self.token_type[0].view(1, 1, -1)
            action_tokens = self.action_token(prev_action_context) + self.token_type[1].view(1, 1, -1)
            sequence_tokens = torch.stack([z_tokens, action_tokens], dim=2).flatten(1, 2)
            task_token = self.task_token(task_id).unsqueeze(1) + self.token_type[2].view(1, 1, -1)
            agent_token = self.agent_token.expand(batch, -1, -1) + self.token_type[3].view(1, 1, -1)
            tokens = torch.cat([task_token, sequence_tokens, agent_token], dim=1)
            tokens = tokens + self.position[:, : tokens.shape[1], :]
            encoded = self.encoder(tokens, mask=self._agent_isolation_mask(tokens.shape[1], tokens.device))
            h_t = self.post(encoded[:, -1, :])
            action_pred = torch.stack([head(h_t) for head in self.action_heads], dim=1)
            reward_pred = torch.cat([head(h_t) for head in self.reward_heads], dim=1)
            return {
                "h_t": h_t,
                "action_pred": action_pred,
                "reward_pred": reward_pred,
                "value": self.value_head(h_t).squeeze(-1),
            }

        def _agent_isolation_mask(self, seq_len: int, device: torch.device) -> torch.Tensor | None:
            if not self.agent_token_isolation:
                return None
            mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
            if seq_len <= 2:
                return mask
            sequence_rows = torch.arange(1, seq_len - 1, device=device)
            # Context/world tokens cannot read task or agent tokens. The agent token
            # remains free to read task and context tokens for decision heads.
            mask[sequence_rows, 0] = True
            mask[sequence_rows, seq_len - 1] = True
            return mask


    class MidtrainingSequenceDataset(Dataset):
        def __init__(
            self,
            cache: SequenceCache,
            anchors: np.ndarray,
            normalizer: Normalizer,
            *,
            context_len: int,
            mtp_horizon: int,
            control_mode: str = "normal",
            control_seed: int = 0,
        ):
            if control_mode not in CONTROL_MODES:
                raise ValueError(f"Unsupported control_mode={control_mode!r}; expected one of {sorted(CONTROL_MODES)}")
            self.cache = cache
            self.anchors = np.asarray(anchors, dtype=np.int64)
            self.normalizer = normalizer
            self.context_len = int(context_len)
            self.mtp_horizon = int(mtp_horizon)
            self.control_mode = str(control_mode)
            self.z = torch.from_numpy(normalize_np(cache.z, normalizer.z_mean, normalizer.z_std))
            self.action = torch.from_numpy(normalize_np(cache.action, normalizer.action_mean, normalizer.action_std))
            reward = ((cache.reward.astype(np.float32) - normalizer.reward_mean) / normalizer.reward_std).astype(np.float32)
            self.reward = torch.from_numpy(reward)
            self.reward_raw = torch.from_numpy(cache.reward.astype(np.float32))
            self.task_id = torch.from_numpy(cache.task_id.astype(np.int64))
            self._control_anchors = self._make_control_anchors(control_seed)

        def __len__(self) -> int:
            return int(self.anchors.shape[0])

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            anchor = int(self.anchors[index])
            target_anchor = int(self._control_anchors[index]) if self.control_mode == "shuffle_targets" else anchor
            z_anchor = int(self._control_anchors[index]) if self.control_mode == "shuffle_z_context" else anchor
            prev_action_start = anchor - self.context_len
            z_context_start = z_anchor - self.context_len + 1
            target_end = target_anchor + self.mtp_horizon + 1
            z_context = self.z[z_context_start : z_anchor + 1]
            prev_action_context = self.action[prev_action_start:anchor]
            if self.control_mode == "zero_z_context":
                z_context = torch.zeros_like(z_context)
            if self.control_mode == "zero_prev_actions":
                prev_action_context = torch.zeros_like(prev_action_context)
            return {
                "z_context": z_context,
                "prev_action_context": prev_action_context,
                "target_action": self.action[target_anchor:target_end],
                "target_reward": self.reward[target_anchor:target_end],
                "target_reward_raw": self.reward_raw[target_anchor:target_end],
                "task_id": self.task_id[anchor],
            }

        def _make_control_anchors(self, control_seed: int) -> np.ndarray:
            shuffled = self.anchors.copy()
            if shuffled.size <= 1:
                return shuffled
            rng = np.random.default_rng(int(control_seed))
            rng.shuffle(shuffled)
            if np.all(shuffled == self.anchors):
                shuffled = np.roll(shuffled, 1)
            return shuffled


    class LatentDynamicsSequenceDataset(Dataset):
        def __init__(
            self,
            cache: SequenceCache,
            anchors: np.ndarray,
            normalizer: Normalizer,
            *,
            context_len: int,
            prediction_horizon: int,
            future_action_offset: int = 0,
            control_mode: str = "normal",
            control_seed: int = 0,
        ):
            if control_mode not in DYNAMICS_CONTROL_MODES:
                raise ValueError(
                    f"Unsupported control_mode={control_mode!r}; expected one of {sorted(DYNAMICS_CONTROL_MODES)}"
                )
            self.cache = cache
            self.anchors = np.asarray(anchors, dtype=np.int64)
            self.normalizer = normalizer
            self.context_len = int(context_len)
            self.prediction_horizon = int(prediction_horizon)
            self.future_action_offset = int(future_action_offset)
            self.control_mode = str(control_mode)
            self.z = torch.from_numpy(normalize_np(cache.z, normalizer.z_mean, normalizer.z_std))
            self.action = torch.from_numpy(normalize_np(cache.action, normalizer.action_mean, normalizer.action_std))
            self.task_id = torch.from_numpy(cache.task_id.astype(np.int64))
            self._control_anchors = self._make_control_anchors(control_seed)

        def __len__(self) -> int:
            return int(self.anchors.shape[0])

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            anchor = int(self.anchors[index])
            control_anchor = int(self._control_anchors[index])
            target_anchor = control_anchor if self.control_mode == "shuffle_targets" else anchor
            z_anchor = control_anchor if self.control_mode == "shuffle_z_context" else anchor
            future_action_anchor = (
                control_anchor if self.control_mode == "shuffle_future_actions" else anchor + self.future_action_offset
            )

            z_context_start = z_anchor - self.context_len + 1
            action_context_start = anchor - self.context_len + 1
            target_start = target_anchor + 1
            target_end = target_anchor + self.prediction_horizon + 1
            future_action_end = future_action_anchor + self.prediction_horizon

            z_context = self.z[z_context_start : z_anchor + 1]
            action_context = self.action[action_context_start : anchor + 1]
            future_action = self.action[future_action_anchor:future_action_end]
            target_z = self.z[target_start:target_end]

            if self.control_mode == "zero_z_context":
                z_context = torch.zeros_like(z_context)
            if self.control_mode == "zero_action_context":
                action_context = torch.zeros_like(action_context)
            if self.control_mode == "zero_future_actions":
                future_action = torch.zeros_like(future_action)

            return {
                "z_context": z_context,
                "action_context": action_context,
                "future_action": future_action,
                "target_z": target_z,
                "task_id": self.task_id[anchor],
            }

        def _make_control_anchors(self, control_seed: int) -> np.ndarray:
            shuffled = self.anchors.copy()
            if shuffled.size <= 1:
                return shuffled
            rng = np.random.default_rng(int(control_seed))
            rng.shuffle(shuffled)
            if np.all(shuffled == self.anchors):
                shuffled = np.roll(shuffled, 1)
            return shuffled


def normalize_np(array: np.ndarray, mean: list[float] | np.ndarray, std: list[float] | np.ndarray) -> np.ndarray:
    return ((array.astype(np.float32) - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)).astype(
        np.float32
    )


if nn is None:

    class BehaviorCloningMidtrainingHead:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)
            raise RuntimeError("torch is required to instantiate BehaviorCloningMidtrainingHead.")


    class MidtrainingSequenceDataset:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)
            raise RuntimeError("torch is required to instantiate MidtrainingSequenceDataset.")


    class ActionConditionedLatentDynamics:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)
            raise RuntimeError("torch is required to instantiate ActionConditionedLatentDynamics.")


    class LatentDynamicsSequenceDataset:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)
            raise RuntimeError("torch is required to instantiate LatentDynamicsSequenceDataset.")

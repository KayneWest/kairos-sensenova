#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sensenova_drone.action_risk_model import load_action_risk_planner_runner
from sensenova_drone.bc_data import ACTION_VOCAB
from sensenova_drone.bc_infer import load_bc_policy_runner
from sensenova_drone.gym_drone_game import DroneGameConfig, DroneMazeEnv
from sensenova_drone.risk_aware_policy import load_risk_aware_policy_runner
from sensenova_drone.world_model_decision import load_world_model_decision_runner
from scripts.train_gym_drone_game import StateDQN
from scripts.train_gym_drone_game_cnn_dqn import CnnGoalDQN
from scripts.train_gym_drone_game_world_model import ActionConditionedWorldModel
from scripts.train_gym_drone_game_world_model_dqn import WorldModelDQN
from scripts.train_gym_drone_game_world_model_policy import FrozenWorldModelPolicy

try:
    import torch
except ModuleNotFoundError:
    torch = None


DEFAULT_ENABLED_ACTIONS = "hover,yaw_left,yaw_right,forward,strafe_left,strafe_right"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Gym drone-game decision models on matched seeds.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=900000)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--enabled-actions", default=DEFAULT_ENABLED_ACTIONS)
    parser.add_argument("--models", default="random,heuristic,state_dqn,image_bc,world_model_policy")
    parser.add_argument("--state-dqn-checkpoint", default="output/gym_drone_game_dqn_overnight_20260509T032655Z/best.pt")
    parser.add_argument("--image-bc-checkpoint", default="output/bc_policy_gym_drone_game_dqn_teacher_v3_unbalanced/best.pt")
    parser.add_argument("--world-model-policy-checkpoint", default="output/gym_drone_game_world_model_policy_v1/best.pt")
    parser.add_argument("--world-model-dqn-checkpoint", default="output/gym_drone_game_world_model_dqn/best.pt")
    parser.add_argument("--cnn-dqn-checkpoint", default="output/gym_drone_game_cnn_dqn/best.pt")
    parser.add_argument("--risk-visual-policy-checkpoint", default="output/gym_drone_game_risk_policy_v1/best.pt")
    parser.add_argument("--action-risk-checkpoint", default="output/gym_drone_game_action_risk_scorer_v2/best.pt")
    parser.add_argument("--world-model-decision-checkpoint", default="output/gym_drone_game_world_model_decision_heads_v2_weighted/best.pt")
    parser.add_argument("--risk-policy-no-shield", action="store_true")
    parser.add_argument("--risk-shield-collision-threshold", type=float, default=None)
    parser.add_argument("--risk-shield-front-clearance-m", type=float, default=None)
    parser.add_argument("--action-risk-collision-penalty", type=float, default=None)
    parser.add_argument("--action-risk-out-of-bounds-penalty", type=float, default=None)
    parser.add_argument("--action-risk-success-bonus", type=float, default=None)
    parser.add_argument("--action-risk-progress-weight", type=float, default=None)
    parser.add_argument("--action-risk-clearance-weight", type=float, default=None)
    parser.add_argument("--action-risk-hover-penalty", type=float, default=None)
    parser.add_argument("--action-risk-yaw-penalty", type=float, default=None)
    parser.add_argument("--world-model-decision-policy-logprob-weight", type=float, default=None)
    parser.add_argument("--world-model-decision-collision-penalty", type=float, default=None)
    parser.add_argument("--world-model-decision-progress-weight", type=float, default=None)
    parser.add_argument(
        "--world-model-dqn-shield-front-clearance-m",
        type=float,
        default=None,
        help="If set, block forward when current front clearance is below this threshold.",
    )
    parser.add_argument(
        "--cnn-dqn-shield-front-clearance-m",
        type=float,
        default=None,
        help="If set, block forward for cnn_dqn when current front clearance is below this threshold.",
    )
    parser.add_argument("--world-size-m", type=float, default=16.0)
    parser.add_argument("--obstacle-count", type=int, default=14)
    parser.add_argument("--max-episode-steps", type=int, default=80)
    parser.add_argument("--image-width", type=int, default=64)
    parser.add_argument("--image-height", type=int, default=48)
    parser.add_argument("--trace-episodes", type=int, default=4)
    parser.add_argument("--trace-frames", type=int, default=10)
    parser.add_argument("--print-episode-records", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    enabled_actions = parse_enabled_actions(args.enabled_actions)
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    env_cfg = DroneGameConfig(
        world_size_m=args.world_size_m,
        obstacle_count=args.obstacle_count,
        image_width=args.image_width,
        image_height=args.image_height,
        max_episode_steps=args.max_episode_steps,
    )
    seeds = [args.seed + index for index in range(args.episodes)]

    models = build_models(args, model_names, env_cfg, device=device, enabled_actions=enabled_actions)
    results = []
    for model in models:
        result = evaluate_model(
            model,
            env_cfg,
            seeds=seeds,
            enabled_actions=enabled_actions,
            trace_path=out_dir / f"{model.name}_trace.png",
            trace_episodes=args.trace_episodes,
            trace_frames=args.trace_frames,
        )
        results.append(result)
        printable = result if args.print_episode_records else compact_result(result)
        print(json.dumps(printable, indent=2), flush=True)

    summary = {
        "out_dir": str(out_dir.resolve()),
        "episodes": args.episodes,
        "seed_start": args.seed,
        "enabled_actions": [ACTION_VOCAB[index] for index in enabled_actions],
        "env_config": env_cfg.__dict__,
        "results": results,
        "ranking_by_success_then_collision": sorted(
            [
                {
                    "model": result["model"],
                    "success_rate": result["success_rate"],
                    "collision_rate": result["collision_rate"],
                    "mean_return": result["mean_return"],
                }
                for result in results
            ],
            key=lambda item: (-item["success_rate"], item["collision_rate"], -item["mean_return"]),
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out_dir, summary)
    print(json.dumps(summary["ranking_by_success_then_collision"], indent=2))
    return 0


class BenchmarkModel:
    def __init__(self, name: str):
        self.name = name

    def reset(self) -> None:
        return None

    def act(self, env: DroneMazeEnv, obs: dict[str, np.ndarray], info: dict[str, Any]) -> int:
        raise NotImplementedError

    def extra_summary(self) -> dict[str, Any]:
        return {}


class RandomModel(BenchmarkModel):
    def __init__(self, enabled_actions: list[int], seed: int):
        super().__init__("random")
        self.enabled_actions = list(enabled_actions)
        self.rng = random.Random(seed)

    def act(self, env: DroneMazeEnv, obs: dict[str, np.ndarray], info: dict[str, Any]) -> int:
        _ = (env, obs, info)
        return int(self.rng.choice(self.enabled_actions))


class HeuristicModel(BenchmarkModel):
    def __init__(self):
        super().__init__("heuristic")

    def act(self, env: DroneMazeEnv, obs: dict[str, np.ndarray], info: dict[str, Any]) -> int:
        _ = (obs, info)
        return env.expert_action_index()


class StateDQNModel(BenchmarkModel):
    def __init__(self, checkpoint_path: Path, *, device: str, enabled_actions: list[int]):
        super().__init__("state_dqn")
        if torch is None:
            raise RuntimeError("torch is required for state_dqn.")
        payload = torch.load(checkpoint_path, map_location=device)
        self.device = device
        self.enabled_actions = enabled_actions
        state_dim = 12
        self.model = StateDQN(state_dim, len(payload.get("action_vocab") or ACTION_VOCAB)).to(device)
        self.model.load_state_dict(payload["policy_state_dict"])
        self.model.eval()
        self.checkpoint_path = str(checkpoint_path.resolve())

    def act(self, env: DroneMazeEnv, obs: dict[str, np.ndarray], info: dict[str, Any]) -> int:
        _ = (env, info)
        with torch.no_grad():
            state = torch.tensor(obs["state"][None, :], dtype=torch.float32, device=self.device)
            logits = self.model(state)
            return masked_argmax(logits, self.enabled_actions)


class ImageBCModel(BenchmarkModel):
    def __init__(self, checkpoint_path: Path, *, device: str, enabled_actions: list[int]):
        super().__init__("image_bc")
        self.runner = load_bc_policy_runner(checkpoint_path, device=device)
        self.enabled_actions = enabled_actions
        self.checkpoint_path = str(checkpoint_path.resolve())

    def reset(self) -> None:
        self.runner.reset_history()

    def act(self, env: DroneMazeEnv, obs: dict[str, np.ndarray], info: dict[str, Any]) -> int:
        _ = env
        pred = self.runner.predict(Image.fromarray(obs["image"]), goal_features=goal_features_from_info(info))
        return masked_argmax_list(pred.probabilities, self.enabled_actions)


class WorldModelPolicyModel(BenchmarkModel):
    def __init__(self, checkpoint_path: Path, *, device: str, enabled_actions: list[int]):
        super().__init__("world_model_policy")
        if torch is None:
            raise RuntimeError("torch is required for world_model_policy.")
        payload = torch.load(checkpoint_path, map_location=device)
        wm_payload = torch.load(payload["world_model_checkpoint"], map_location=device)
        wm_config = dict(payload["world_model_config"])
        world_model = ActionConditionedWorldModel(
            num_actions=len(ACTION_VOCAB),
            image_width=int(wm_config["image_width"]),
            image_height=int(wm_config["image_height"]),
            latent_dim=int(wm_config["latent_dim"]),
        ).to(device)
        if "world_model_state_dict" in payload:
            world_model.load_state_dict(payload["world_model_state_dict"])
        else:
            world_model.load_state_dict(wm_payload["model_state_dict"])
        world_model.eval()
        self.policy = FrozenWorldModelPolicy(
            world_model,
            latent_dim=int(wm_config["latent_dim"]),
            num_actions=len(ACTION_VOCAB),
        ).to(device)
        self.policy.head.load_state_dict(payload["policy_state_dict"])
        self.policy.eval()
        self.device = device
        self.enabled_actions = enabled_actions
        self.image_width = int(wm_config["image_width"])
        self.image_height = int(wm_config["image_height"])
        self.checkpoint_path = str(checkpoint_path.resolve())

    def act(self, env: DroneMazeEnv, obs: dict[str, np.ndarray], info: dict[str, Any]) -> int:
        _ = env
        image = Image.fromarray(obs["image"]).resize((self.image_width, self.image_height), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        image_tensor = torch.tensor(array[None, ...], dtype=torch.float32, device=self.device)
        goal_tensor = torch.tensor([goal_features_from_info(info)], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.policy(image_tensor, goal_tensor)
            return masked_argmax(logits, self.enabled_actions)


class WorldModelDQNModel(BenchmarkModel):
    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device: str,
        enabled_actions: list[int],
        shield_front_clearance_m: float | None,
    ):
        super().__init__("world_model_dqn")
        if torch is None:
            raise RuntimeError("torch is required for world_model_dqn.")
        payload = torch.load(checkpoint_path, map_location=device)
        wm_payload = torch.load(payload["world_model_checkpoint"], map_location=device)
        wm_config = dict(payload["world_model_config"])
        world_model = ActionConditionedWorldModel(
            num_actions=len(ACTION_VOCAB),
            image_width=int(wm_config["image_width"]),
            image_height=int(wm_config["image_height"]),
            latent_dim=int(wm_config["latent_dim"]),
        ).to(device)
        if "world_model_state_dict" in payload:
            world_model.load_state_dict(payload["world_model_state_dict"])
        else:
            world_model.load_state_dict(wm_payload["model_state_dict"])
        world_model.eval()
        for parameter in world_model.parameters():
            parameter.requires_grad_(False)
        self.world_model = world_model
        self.q_net = WorldModelDQN(
            int(payload.get("input_dim", int(wm_config["latent_dim"]) + 4)),
            len(payload.get("action_vocab") or ACTION_VOCAB),
            hidden_dim=int(payload.get("hidden_dim", 256)),
        ).to(device)
        self.q_net.load_state_dict(payload["q_state_dict"])
        self.q_net.eval()
        self.device = device
        self.enabled_actions = enabled_actions
        self.shield_front_clearance_m = shield_front_clearance_m
        self.shielded_steps = 0
        self.shield_counts: Counter[str] = Counter()
        self.image_width = int(wm_config["image_width"])
        self.image_height = int(wm_config["image_height"])
        self.checkpoint_path = str(checkpoint_path.resolve())
        self.encoder_source = str(payload.get("encoder_source", "pretrained"))

    def reset(self) -> None:
        return None

    def act(self, env: DroneMazeEnv, obs: dict[str, np.ndarray], info: dict[str, Any]) -> int:
        _ = env
        image = Image.fromarray(obs["image"]).resize((self.image_width, self.image_height), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        image_tensor = torch.tensor(array[None, ...], dtype=torch.float32, device=self.device)
        goal_tensor = torch.tensor([goal_features_from_info(info)], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            latent = self.world_model.encode(image_tensor)
            q_values = self.q_net(torch.cat([latent, goal_tensor], dim=1))
            enabled_actions = self._shielded_actions(info)
            return masked_argmax(q_values, enabled_actions)

    def _shielded_actions(self, info: dict[str, Any]) -> list[int]:
        if self.shield_front_clearance_m is None:
            return self.enabled_actions
        forward_index = ACTION_VOCAB.index("forward")
        front = dict(info.get("clearance_m") or {}).get("front_m")
        if front is None or float(front) >= float(self.shield_front_clearance_m):
            return self.enabled_actions
        shielded = [index for index in self.enabled_actions if index != forward_index]
        if not shielded:
            return self.enabled_actions
        self.shielded_steps += 1
        self.shield_counts["front_clearance"] += 1
        return shielded

    def extra_summary(self) -> dict[str, Any]:
        return {
            "shielded_steps": int(self.shielded_steps),
            "shield_counts": dict(self.shield_counts),
            "shield_front_clearance_m": self.shield_front_clearance_m,
            "encoder_source": self.encoder_source,
        }


class CnnDQNModel(BenchmarkModel):
    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device: str,
        enabled_actions: list[int],
        shield_front_clearance_m: float | None,
    ):
        super().__init__("cnn_dqn")
        if torch is None:
            raise RuntimeError("torch is required for cnn_dqn.")
        payload = torch.load(checkpoint_path, map_location=device)
        self.q_net = CnnGoalDQN(
            len(payload.get("action_vocab") or ACTION_VOCAB),
            hidden_dim=int(payload.get("hidden_dim", 256)),
        ).to(device)
        self.q_net.load_state_dict(payload["q_state_dict"])
        self.q_net.eval()
        env_config = dict(payload.get("env_config") or {})
        self.image_width = int(env_config.get("image_width", 64))
        self.image_height = int(env_config.get("image_height", 48))
        self.device = device
        self.enabled_actions = enabled_actions
        self.shield_front_clearance_m = shield_front_clearance_m
        self.shielded_steps = 0
        self.shield_counts: Counter[str] = Counter()
        self.checkpoint_path = str(checkpoint_path.resolve())

    def act(self, env: DroneMazeEnv, obs: dict[str, np.ndarray], info: dict[str, Any]) -> int:
        _ = env
        image = Image.fromarray(obs["image"]).resize((self.image_width, self.image_height), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        image_tensor = torch.tensor(array[None, ...], dtype=torch.float32, device=self.device)
        goal_tensor = torch.tensor([goal_features_from_info(info)], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self.q_net(image_tensor, goal_tensor)
            enabled_actions = self._shielded_actions(info)
            return masked_argmax(q_values, enabled_actions)

    def _shielded_actions(self, info: dict[str, Any]) -> list[int]:
        if self.shield_front_clearance_m is None:
            return self.enabled_actions
        forward_index = ACTION_VOCAB.index("forward")
        front = dict(info.get("clearance_m") or {}).get("front_m")
        if front is None or float(front) >= float(self.shield_front_clearance_m):
            return self.enabled_actions
        shielded = [index for index in self.enabled_actions if index != forward_index]
        if not shielded:
            return self.enabled_actions
        self.shielded_steps += 1
        self.shield_counts["front_clearance"] += 1
        return shielded

    def extra_summary(self) -> dict[str, Any]:
        return {
            "shielded_steps": int(self.shielded_steps),
            "shield_counts": dict(self.shield_counts),
            "shield_front_clearance_m": self.shield_front_clearance_m,
        }



class RiskAwareVisualModel(BenchmarkModel):
    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device: str,
        enabled_actions: list[int],
        shield_enabled: bool,
        shield_collision_threshold: float | None,
        shield_front_clearance_m: float | None,
    ):
        super().__init__("risk_visual_policy")
        self.runner = load_risk_aware_policy_runner(
            checkpoint_path,
            device=device,
            shield_enabled=shield_enabled,
            shield_collision_threshold=shield_collision_threshold,
            shield_front_clearance_m=shield_front_clearance_m,
        )
        self.enabled_actions = enabled_actions
        self.checkpoint_path = str(checkpoint_path.resolve())
        self.shielded_steps = 0
        self.shield_counts: Counter[str] = Counter()
        self.unshielded_action_counts: Counter[str] = Counter()
        self.collision_risks: list[float] = []
        self.predicted_clearances: list[float] = []

    def reset(self) -> None:
        self.runner.reset_history()

    def act(self, env: DroneMazeEnv, obs: dict[str, np.ndarray], info: dict[str, Any]) -> int:
        _ = env
        pred = self.runner.predict(
            Image.fromarray(obs["image"]),
            goal_features=goal_features_from_info(info),
            enabled_actions=self.enabled_actions,
        )
        unshielded = str(pred.metadata.get("unshielded_action"))
        self.unshielded_action_counts[unshielded] += 1
        self.collision_risks.append(float(pred.collision_risk))
        self.predicted_clearances.append(float(pred.front_clearance_m))
        if pred.shield_reason:
            self.shielded_steps += 1
            self.shield_counts[pred.shield_reason] += 1
        return int(pred.action_index)

    def extra_summary(self) -> dict[str, Any]:
        return {
            "shielded_steps": int(self.shielded_steps),
            "shield_counts": dict(self.shield_counts),
            "unshielded_action_counts": dict(self.unshielded_action_counts),
            "mean_predicted_collision_risk": mean(self.collision_risks),
            "mean_predicted_front_clearance_m": mean(self.predicted_clearances),
        }


class ActionRiskPlannerModel(BenchmarkModel):
    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device: str,
        enabled_actions: list[int],
        collision_penalty: float | None,
        out_of_bounds_penalty: float | None,
        success_bonus: float | None,
        progress_weight: float | None,
        clearance_weight: float | None,
        hover_penalty: float | None,
        yaw_penalty: float | None,
    ):
        super().__init__("action_risk_planner")
        self.runner = load_action_risk_planner_runner(
            checkpoint_path,
            device=device,
            collision_penalty=collision_penalty,
            out_of_bounds_penalty=out_of_bounds_penalty,
            success_bonus=success_bonus,
            progress_weight=progress_weight,
            clearance_weight=clearance_weight,
            hover_penalty=hover_penalty,
            yaw_penalty=yaw_penalty,
        )
        self.enabled_actions = enabled_actions
        self.checkpoint_path = str(checkpoint_path.resolve())
        self.top_collision_risks: list[float] = []
        self.top_clearances: list[float] = []
        self.top_utilities: list[float] = []

    def reset(self) -> None:
        self.runner.reset_history()

    def act(self, env: DroneMazeEnv, obs: dict[str, np.ndarray], info: dict[str, Any]) -> int:
        _ = env
        decision = self.runner.predict(
            Image.fromarray(obs["image"]),
            goal_features=goal_features_from_info(info),
            enabled_actions=self.enabled_actions,
        )
        top = decision.candidates[0]
        self.top_collision_risks.append(float(top.collision_risk))
        self.top_clearances.append(float(top.front_clearance_m))
        self.top_utilities.append(float(top.utility))
        return int(decision.action_index)

    def extra_summary(self) -> dict[str, Any]:
        return {
            "mean_predicted_collision_risk": mean(self.top_collision_risks),
            "mean_predicted_front_clearance_m": mean(self.top_clearances),
            "mean_predicted_utility": mean(self.top_utilities),
        }


class WorldModelDecisionModel(BenchmarkModel):
    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device: str,
        enabled_actions: list[int],
        policy_logprob_weight: float | None,
        collision_penalty: float | None,
        progress_weight: float | None,
    ):
        super().__init__("world_model_decision_heads")
        self.runner = load_world_model_decision_runner(
            checkpoint_path,
            device=device,
            policy_logprob_weight=policy_logprob_weight,
            collision_penalty=collision_penalty,
            progress_weight=progress_weight,
        )
        self.enabled_actions = enabled_actions
        self.checkpoint_path = str(checkpoint_path.resolve())
        self.top_collision_risks: list[float] = []
        self.top_clearances: list[float] = []
        self.top_utilities: list[float] = []

    def reset(self) -> None:
        self.runner.reset_history()

    def act(self, env: DroneMazeEnv, obs: dict[str, np.ndarray], info: dict[str, Any]) -> int:
        _ = env
        decision = self.runner.predict(
            Image.fromarray(obs["image"]),
            goal_features=goal_features_from_info(info),
            enabled_actions=self.enabled_actions,
        )
        top = decision.candidates[0]
        self.top_collision_risks.append(float(top.collision_risk))
        self.top_clearances.append(float(top.front_clearance_m))
        self.top_utilities.append(float(top.utility))
        return int(decision.action_index)

    def extra_summary(self) -> dict[str, Any]:
        return {
            "mean_predicted_collision_risk": mean(self.top_collision_risks),
            "mean_predicted_front_clearance_m": mean(self.top_clearances),
            "mean_predicted_utility": mean(self.top_utilities),
        }


def build_models(
    args: argparse.Namespace,
    model_names: list[str],
    env_cfg: DroneGameConfig,
    *,
    device: str,
    enabled_actions: list[int],
) -> list[BenchmarkModel]:
    _ = env_cfg
    models: list[BenchmarkModel] = []
    for name in model_names:
        if name == "random":
            models.append(RandomModel(enabled_actions, seed=args.seed))
        elif name == "heuristic":
            models.append(HeuristicModel())
        elif name == "state_dqn":
            models.append(StateDQNModel(Path(args.state_dqn_checkpoint), device=device, enabled_actions=enabled_actions))
        elif name == "image_bc":
            models.append(ImageBCModel(Path(args.image_bc_checkpoint), device=device, enabled_actions=enabled_actions))
        elif name == "world_model_policy":
            models.append(WorldModelPolicyModel(Path(args.world_model_policy_checkpoint), device=device, enabled_actions=enabled_actions))
        elif name == "world_model_dqn":
            models.append(
                WorldModelDQNModel(
                    Path(args.world_model_dqn_checkpoint),
                    device=device,
                    enabled_actions=enabled_actions,
                    shield_front_clearance_m=args.world_model_dqn_shield_front_clearance_m,
                )
            )
        elif name == "cnn_dqn":
            models.append(
                CnnDQNModel(
                    Path(args.cnn_dqn_checkpoint),
                    device=device,
                    enabled_actions=enabled_actions,
                    shield_front_clearance_m=args.cnn_dqn_shield_front_clearance_m,
                )
            )
        elif name == "risk_visual_policy":
            models.append(
                RiskAwareVisualModel(
                    Path(args.risk_visual_policy_checkpoint),
                    device=device,
                    enabled_actions=enabled_actions,
                    shield_enabled=not args.risk_policy_no_shield,
                    shield_collision_threshold=args.risk_shield_collision_threshold,
                    shield_front_clearance_m=args.risk_shield_front_clearance_m,
                )
            )
        elif name == "action_risk_planner":
            models.append(
                ActionRiskPlannerModel(
                    Path(args.action_risk_checkpoint),
                    device=device,
                    enabled_actions=enabled_actions,
                    collision_penalty=args.action_risk_collision_penalty,
                    out_of_bounds_penalty=args.action_risk_out_of_bounds_penalty,
                    success_bonus=args.action_risk_success_bonus,
                    progress_weight=args.action_risk_progress_weight,
                    clearance_weight=args.action_risk_clearance_weight,
                    hover_penalty=args.action_risk_hover_penalty,
                    yaw_penalty=args.action_risk_yaw_penalty,
                )
            )
        elif name == "world_model_decision_heads":
            models.append(
                WorldModelDecisionModel(
                    Path(args.world_model_decision_checkpoint),
                    device=device,
                    enabled_actions=enabled_actions,
                    policy_logprob_weight=args.world_model_decision_policy_logprob_weight,
                    collision_penalty=args.world_model_decision_collision_penalty,
                    progress_weight=args.world_model_decision_progress_weight,
                )
            )
        else:
            raise ValueError(f"Unsupported model: {name!r}")
    return models


def evaluate_model(
    model: BenchmarkModel,
    env_cfg: DroneGameConfig,
    *,
    seeds: list[int],
    enabled_actions: list[int],
    trace_path: Path,
    trace_episodes: int,
    trace_frames: int,
) -> dict[str, Any]:
    _ = enabled_actions
    returns = []
    lengths = []
    successes = []
    collisions = []
    timeouts = []
    out_of_bounds = []
    min_fronts = []
    action_counts: Counter[str] = Counter()
    trace_rows: list[tuple[str, list[np.ndarray]]] = []
    episodes: list[dict[str, Any]] = []

    for episode_idx, seed in enumerate(seeds):
        env = DroneMazeEnv(env_cfg)
        obs, info = env.reset(seed=seed)
        model.reset()
        done = False
        total_reward = 0.0
        step_idx = 0
        min_front = float("inf")
        episode_actions: Counter[str] = Counter()
        frames: list[np.ndarray] = []
        while not done:
            if episode_idx < trace_episodes and len(frames) < trace_frames:
                frames.append(obs["image"])
            action = model.act(env, obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            step_idx += 1
            action_name = ACTION_VOCAB[action]
            action_counts[action_name] += 1
            episode_actions[action_name] += 1
            front = info.get("clearance_m", {}).get("front_m")
            if front is not None:
                min_front = min(min_front, float(front))
            done = bool(terminated or truncated)
        success = bool(info.get("success", False))
        collision = bool(info.get("collision", False))
        timeout = bool(info.get("truncated", False))
        oob = bool(info.get("out_of_bounds", False))
        returns.append(total_reward)
        lengths.append(step_idx)
        successes.append(1.0 if success else 0.0)
        collisions.append(1.0 if collision else 0.0)
        timeouts.append(1.0 if timeout else 0.0)
        out_of_bounds.append(1.0 if oob else 0.0)
        if min_front != float("inf"):
            min_fronts.append(min_front)
        episodes.append(
            {
                "seed": seed,
                "return": total_reward,
                "length": step_idx,
                "success": success,
                "collision": collision,
                "timeout": timeout,
                "out_of_bounds": oob,
                "action_counts": dict(episode_actions),
            }
        )
        if frames:
            trace_rows.append((f"seed {seed}", frames))

    if trace_rows:
        make_trace_sheet(trace_rows, trace_path)
    episodes_jsonl = trace_path.with_name(f"{model.name}_episodes.jsonl")
    with episodes_jsonl.open("w", encoding="utf-8") as f:
        for record in episodes:
            f.write(json.dumps(record) + "\n")

    result = {
        "model": model.name,
        "num_episodes": len(seeds),
        "success_rate": mean(successes),
        "collision_rate": mean(collisions),
        "timeout_rate": mean(timeouts),
        "out_of_bounds_rate": mean(out_of_bounds),
        "mean_return": mean(returns),
        "median_return": float(np.median(returns)) if returns else 0.0,
        "mean_length": mean(lengths),
        "mean_min_front_clearance_m": mean(min_fronts) if min_fronts else None,
        "action_counts": dict(action_counts),
        "trace_path": str(trace_path.resolve()) if trace_rows else None,
        "episodes_jsonl": str(episodes_jsonl.resolve()),
    }
    result.update(model.extra_summary())
    return result


def resolve_device(requested: str) -> str:
    requested = requested.strip().lower()
    if requested == "auto":
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    return requested


def parse_enabled_actions(raw: str) -> list[int]:
    indices = []
    for name in [item.strip() for item in raw.split(",") if item.strip()]:
        if name not in ACTION_VOCAB:
            raise ValueError(f"Unsupported action: {name!r}")
        indices.append(ACTION_VOCAB.index(name))
    if not indices:
        raise ValueError("No enabled actions.")
    return list(dict.fromkeys(indices))


def masked_argmax(logits, enabled_actions: list[int]) -> int:
    mask = torch.full_like(logits, fill_value=-1e9)
    mask[:, enabled_actions] = logits[:, enabled_actions]
    return int(torch.argmax(mask, dim=-1).item())


def masked_argmax_list(values: list[float], enabled_actions: list[int]) -> int:
    best = enabled_actions[0]
    best_value = float("-inf")
    for index in enabled_actions:
        value = float(values[index])
        if value > best_value:
            best = index
            best_value = value
    return best


def goal_features_from_info(info: dict[str, Any]) -> list[float]:
    forward, right = info.get("goal_body_xy_m") or [0.0, 0.0]
    heading = math.degrees(math.atan2(float(right), max(float(forward), 1e-6)))
    return [
        float(np.clip(float(forward) / 10.0, -2.0, 2.0)),
        float(np.clip(float(right) / 5.0, -2.0, 2.0)),
        0.0,
        float(np.clip(heading / 180.0, -1.0, 1.0)),
    ]


def mean(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "model",
        "num_episodes",
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "out_of_bounds_rate",
        "mean_return",
        "median_return",
        "mean_length",
        "mean_min_front_clearance_m",
        "action_counts",
        "trace_path",
        "episodes_jsonl",
        "shielded_steps",
        "mean_predicted_collision_risk",
        "mean_predicted_front_clearance_m",
        "mean_predicted_utility",
    ]
    return {key: result.get(key) for key in keys}


def make_trace_sheet(rows: list[tuple[str, list[np.ndarray]]], out_path: Path) -> None:
    if not rows:
        return
    frame_h, frame_w = rows[0][1][0].shape[:2]
    label_w = 96
    label_h = 16
    cols = max(len(frames) for _, frames in rows)
    sheet = Image.new("RGB", (label_w + frame_w * cols, label_h + frame_h * len(rows)), color=(28, 28, 28))
    draw = ImageDraw.Draw(sheet)
    for col in range(cols):
        draw.text((label_w + col * frame_w + 4, 2), f"t={col}", fill=(240, 240, 240))
    for row_idx, (label, frames) in enumerate(rows):
        y = label_h + row_idx * frame_h
        draw.text((4, y + 4), label, fill=(240, 240, 240))
        for col, frame in enumerate(frames):
            sheet.paste(Image.fromarray(frame.astype(np.uint8), mode="RGB"), (label_w + col * frame_w, y))
    sheet.save(out_path)


def write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    rows = []
    for result in sorted(summary["results"], key=lambda item: (-item["success_rate"], item["collision_rate"])):
        rows.append(
            "<tr>"
            f"<td>{result['model']}</td>"
            f"<td>{result['success_rate']:.4f}</td>"
            f"<td>{result['collision_rate']:.4f}</td>"
            f"<td>{result['timeout_rate']:.4f}</td>"
            f"<td>{result['mean_return']:.4f}</td>"
            f"<td>{result['mean_length']:.2f}</td>"
            f"<td>{result['mean_min_front_clearance_m'] if result['mean_min_front_clearance_m'] is not None else 'n/a'}</td>"
            f"<td>{result.get('shielded_steps', '')}</td>"
            f"<td>{result.get('mean_predicted_collision_risk', '')}</td>"
            "</tr>"
        )
    traces = []
    for result in summary["results"]:
        trace = result.get("trace_path")
        if trace:
            traces.append(f"<h3>{result['model']}</h3><img src=\"{Path(trace).name}\" />")
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Gym Drone Model Benchmark</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f4f1e8; color: #202020; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid #c9c1ad; padding: 6px 8px; text-align: right; }}
    th {{ background: #292f25; color: white; }}
    td:first-child, th:first-child {{ text-align: left; }}
    img {{ max-width: 100%; border: 1px solid #9f967f; background: white; }}
    code {{ background: #ebe4d4; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Gym Drone Model Benchmark</h1>
  <p>Episodes: <code>{summary['episodes']}</code>, seed start: <code>{summary['seed_start']}</code></p>
  <table>
    <thead><tr><th>Model</th><th>Success</th><th>Collision</th><th>Timeout</th><th>Mean Return</th><th>Mean Length</th><th>Mean Min Front</th><th>Shielded Steps</th><th>Pred Risk</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <h2>Trace Sheets</h2>
  {''.join(traces)}
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

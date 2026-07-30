from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from sensenova_drone.actions import DiscreteDroneAction
from sensenova_drone.bc_data import ACTION_VOCAB

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:
    gym, spaces = None, None


@dataclass(frozen=True)
class TreeObstacle:
    x_m: float
    y_m: float
    radius_m: float


@dataclass
class DroneGameConfig:
    world_size_m: float = 16.0
    image_width: int = 96
    image_height: int = 72
    max_episode_steps: int = 80
    max_depth_m: float = 10.0
    fov_deg: float = 90.0
    forward_step_m: float = 0.45
    strafe_step_m: float = 0.45
    yaw_step_deg: float = 12.0
    drone_radius_m: float = 0.25
    goal_radius_m: float = 0.75
    obstacle_count: int = 14
    front_blocked_threshold_m: float = 2.2
    side_clearance_threshold_m: float = 2.0
    render_depth: bool = False
    reward_progress_scale: float = 2.0
    reward_clearance_scale: float = 0.08
    reward_success: float = 10.0
    reward_collision: float = -8.0
    reward_out_of_bounds: float = -5.0
    reward_step_penalty: float = -0.02
    reward_oscillation_penalty: float = -0.15
    metadata: dict[str, Any] = field(default_factory=dict)


class _DiscreteFallback:
    def __init__(self, n: int):
        self.n = int(n)

    def sample(self) -> int:
        return int(np.random.randint(0, self.n))


class _BoxFallback:
    def __init__(self, low: Any, high: Any, shape: tuple[int, ...], dtype: Any):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype


class _DictFallback(dict):
    pass


class _EnvBase:
    if gym is not None:
        base = gym.Env
    else:
        base = object


class DroneMazeEnv(_EnvBase.base):
    """
    Lightweight first-person drone navigation game.

    The environment is intentionally faster and more controllable than PX4/Gazebo.
    It emits real observations from its own simulator, not generated Kairos frames,
    and uses the same discrete action vocabulary as the SITL drone agent.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(self, config: DroneGameConfig | None = None):
        self.cfg = config or DroneGameConfig()
        self._rng = np.random.default_rng()
        self._step_index = 0
        self._position_xy = np.zeros(2, dtype=np.float32)
        self._yaw_rad = 0.0
        self._goal_xy = np.zeros(2, dtype=np.float32)
        self._obstacles: list[TreeObstacle] = []
        self._last_action: DiscreteDroneAction | None = None
        self._last_distance_to_goal_m = 0.0
        self._last_front_clearance_m = self.cfg.max_depth_m
        self._last_info: dict[str, Any] = {}

        space_module = spaces if spaces is not None else _FallbackSpaces
        self.action_space = space_module.Discrete(len(ACTION_VOCAB))
        self.observation_space = space_module.Dict(
            {
                "image": space_module.Box(
                    low=0,
                    high=255,
                    shape=(self.cfg.image_height, self.cfg.image_width, 3),
                    dtype=np.uint8,
                ),
                "state": space_module.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(12,),
                    dtype=np.float32,
                ),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        options = options or {}
        self._step_index = 0
        self._last_action = None
        self._position_xy = np.array([0.0, 0.0], dtype=np.float32)
        self._yaw_rad = float(self._rng.uniform(-0.10, 0.10))
        self._goal_xy = np.array(
            [
                float(self._rng.uniform(10.0, 13.0)),
                float(self._rng.uniform(-2.5, 2.5)),
            ],
            dtype=np.float32,
        )
        self._obstacles = self._sample_obstacles(options)
        self._last_distance_to_goal_m = self._distance_to_goal()
        self._last_front_clearance_m = self._compute_clearances()["front_m"]
        obs = self._get_observation()
        info = self._build_info(reward_terms={}, terminated=False, truncated=False)
        self._last_info = info
        return obs, info

    def step(
        self,
        action: int | str | DiscreteDroneAction,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        drone_action = self._coerce_action(action)
        previous_distance = self._distance_to_goal()
        previous_front = self._compute_clearances()["front_m"]
        previous_action = self._last_action

        candidate_position = self._position_xy.copy()
        candidate_yaw = float(self._yaw_rad)

        if drone_action == DiscreteDroneAction.YAW_LEFT:
            candidate_yaw -= math.radians(self.cfg.yaw_step_deg)
        elif drone_action == DiscreteDroneAction.YAW_RIGHT:
            candidate_yaw += math.radians(self.cfg.yaw_step_deg)
        elif drone_action == DiscreteDroneAction.FORWARD:
            candidate_position += self._forward_vector() * self.cfg.forward_step_m
        elif drone_action == DiscreteDroneAction.BACKWARD:
            candidate_position -= self._forward_vector() * self.cfg.forward_step_m
        elif drone_action == DiscreteDroneAction.STRAFE_LEFT:
            candidate_position -= self._right_vector() * self.cfg.strafe_step_m
        elif drone_action == DiscreteDroneAction.STRAFE_RIGHT:
            candidate_position += self._right_vector() * self.cfg.strafe_step_m

        self._position_xy = candidate_position.astype(np.float32)
        self._yaw_rad = self._wrap_angle(candidate_yaw)
        self._step_index += 1
        self._last_action = drone_action

        collision = self._collides(self._position_xy)
        out_of_bounds = self._out_of_bounds(self._position_xy)
        distance = self._distance_to_goal()
        success = distance <= self.cfg.goal_radius_m
        truncated = self._step_index >= self.cfg.max_episode_steps and not success and not collision
        terminated = bool(success or collision or out_of_bounds)

        clearances = self._compute_clearances()
        front = clearances["front_m"]
        progress = previous_distance - distance
        front_delta = front - previous_front
        oscillation = self._is_oscillation(previous_action, drone_action)
        reward_terms = {
            "progress": self.cfg.reward_progress_scale * progress,
            "clearance_delta": self.cfg.reward_clearance_scale * front_delta,
            "step_penalty": self.cfg.reward_step_penalty,
            "near_obstacle": -0.35 * max(0.0, 1.0 - front),
            "oscillation": self.cfg.reward_oscillation_penalty if oscillation else 0.0,
            "success": self.cfg.reward_success if success else 0.0,
            "collision": self.cfg.reward_collision if collision else 0.0,
            "out_of_bounds": self.cfg.reward_out_of_bounds if out_of_bounds else 0.0,
        }
        if drone_action in {DiscreteDroneAction.HOVER, DiscreteDroneAction.ASCEND, DiscreteDroneAction.DESCEND}:
            reward_terms["non_navigation_action"] = -0.04
        reward = float(sum(reward_terms.values()))

        self._last_distance_to_goal_m = distance
        self._last_front_clearance_m = front
        obs = self._get_observation()
        info = self._build_info(
            reward_terms=reward_terms,
            terminated=terminated,
            truncated=truncated,
            collision=collision,
            out_of_bounds=out_of_bounds,
            success=success,
            progress_m=progress,
            front_delta_m=front_delta,
            oscillation=oscillation,
        )
        self._last_info = info
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return self._render_rgb()

    def close(self) -> None:
        return None

    @property
    def action_names(self) -> list[str]:
        return list(ACTION_VOCAB)

    def heuristic_action(self) -> DiscreteDroneAction:
        clearances = self._compute_clearances()
        goal_forward, goal_right = self._goal_body_xy()
        heading_error_deg = math.degrees(math.atan2(goal_right, max(goal_forward, 1e-6)))

        if clearances["front_m"] < self.cfg.front_blocked_threshold_m:
            return (
                DiscreteDroneAction.STRAFE_LEFT
                if clearances["left_m"] >= clearances["right_m"]
                else DiscreteDroneAction.STRAFE_RIGHT
            )
        if abs(heading_error_deg) > 18.0:
            return DiscreteDroneAction.YAW_RIGHT if heading_error_deg > 0.0 else DiscreteDroneAction.YAW_LEFT
        return DiscreteDroneAction.FORWARD

    def expert_action_index(self) -> int:
        return ACTION_VOCAB.index(self.heuristic_action().value)

    def snapshot(self) -> dict[str, Any]:
        return {
            "step_index": int(self._step_index),
            "position_xy_m": self._position_xy.astype(float).tolist(),
            "yaw_rad": float(self._yaw_rad),
            "goal_xy_m": self._goal_xy.astype(float).tolist(),
            "obstacles": [asdict(obstacle) for obstacle in self._obstacles],
            "last_action": self._last_action.value if self._last_action is not None else None,
            "last_distance_to_goal_m": float(self._last_distance_to_goal_m),
            "last_front_clearance_m": float(self._last_front_clearance_m),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        self._step_index = int(snapshot["step_index"])
        self._position_xy = np.asarray(snapshot["position_xy_m"], dtype=np.float32)
        self._yaw_rad = float(snapshot["yaw_rad"])
        self._goal_xy = np.asarray(snapshot["goal_xy_m"], dtype=np.float32)
        self._obstacles = [
            TreeObstacle(
                x_m=float(item["x_m"]),
                y_m=float(item["y_m"]),
                radius_m=float(item["radius_m"]),
            )
            for item in snapshot.get("obstacles", [])
        ]
        last_action = snapshot.get("last_action")
        self._last_action = DiscreteDroneAction(last_action) if last_action else None
        self._last_distance_to_goal_m = float(snapshot.get("last_distance_to_goal_m", self._distance_to_goal()))
        self._last_front_clearance_m = float(snapshot.get("last_front_clearance_m", self._compute_clearances()["front_m"]))
        self._last_info = self._build_info(reward_terms={}, terminated=False, truncated=False)

    def branch_step(
        self,
        action: int | str | DiscreteDroneAction,
    ) -> dict[str, Any]:
        snapshot = self.snapshot()
        before_info = self._build_info(reward_terms={}, terminated=False, truncated=False)
        _, reward, terminated, truncated, after_info = self.step(action)
        self.restore(snapshot)
        return {
            "action": self._coerce_action(action).value,
            "action_index": ACTION_VOCAB.index(self._coerce_action(action).value),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "before": before_info,
            "after": after_info,
        }

    def _sample_obstacles(self, options: dict[str, Any]) -> list[TreeObstacle]:
        target_family = str(options.get("target_family") or self._rng.choice(["left", "right"]))
        obstacles: list[TreeObstacle] = []

        if target_family == "left":
            obstacles.extend(
                [
                    TreeObstacle(3.3, 0.0, 0.75),
                    TreeObstacle(4.6, 1.15, 0.65),
                    TreeObstacle(6.1, 0.45, 0.55),
                ]
            )
        else:
            obstacles.extend(
                [
                    TreeObstacle(3.3, 0.0, 0.75),
                    TreeObstacle(4.6, -1.15, 0.65),
                    TreeObstacle(6.1, -0.45, 0.55),
                ]
            )

        for _ in range(max(0, self.cfg.obstacle_count - len(obstacles))):
            obstacles.append(
                TreeObstacle(
                    x_m=float(self._rng.uniform(2.0, self.cfg.world_size_m - 1.0)),
                    y_m=float(self._rng.uniform(-self.cfg.world_size_m * 0.45, self.cfg.world_size_m * 0.45)),
                    radius_m=float(self._rng.uniform(0.35, 0.75)),
                )
            )
        return obstacles

    def _get_observation(self) -> dict[str, np.ndarray]:
        return {
            "image": self._render_rgb(),
            "state": self._state_vector(),
        }

    def _state_vector(self) -> np.ndarray:
        goal_forward, goal_right = self._goal_body_xy()
        clearances = self._compute_clearances()
        distance = self._distance_to_goal()
        heading_error = math.atan2(goal_right, max(goal_forward, 1e-6))
        last_action_index = (
            ACTION_VOCAB.index(self._last_action.value)
            if self._last_action is not None
            else -1
        )
        max_depth = max(self.cfg.max_depth_m, 1e-6)
        return np.array(
            [
                np.clip(distance / self.cfg.world_size_m, 0.0, 2.0),
                np.clip(goal_forward / 10.0, -2.0, 2.0),
                np.clip(goal_right / 5.0, -2.0, 2.0),
                math.sin(heading_error),
                math.cos(heading_error),
                clearances["front_m"] / max_depth,
                clearances["left_m"] / max_depth,
                clearances["right_m"] / max_depth,
                min(clearances.values()) / max_depth,
                last_action_index / max(1, len(ACTION_VOCAB) - 1),
                self._step_index / max(1, self.cfg.max_episode_steps),
                1.0 if clearances["front_m"] < self.cfg.front_blocked_threshold_m else 0.0,
            ],
            dtype=np.float32,
        )

    def _render_rgb(self) -> np.ndarray:
        width = self.cfg.image_width
        height = self.cfg.image_height
        horizon = int(height * 0.48)
        image = Image.new("RGB", (width, height), color=(183, 196, 203))
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, horizon, width, height], fill=(141, 145, 136))
        draw.line([0, horizon, width, horizon], fill=(116, 120, 118), width=1)

        fov_rad = math.radians(self.cfg.fov_deg)
        visible = []
        for obstacle in self._obstacles:
            forward_m, right_m = self._point_body_xy(np.array([obstacle.x_m, obstacle.y_m], dtype=np.float32))
            if forward_m <= 0.1:
                continue
            angle = math.atan2(right_m, forward_m)
            if abs(angle) > fov_rad * 0.6:
                continue
            visible.append((forward_m, right_m, angle, obstacle))
        visible.sort(key=lambda item: item[0], reverse=True)

        for forward_m, _, angle, obstacle in visible:
            center_x = int((0.5 + angle / fov_rad) * width)
            scale = 1.0 / max(forward_m, 0.2)
            trunk_width = max(2, int(width * obstacle.radius_m * scale * 0.85))
            trunk_height = max(4, int(height * min(1.25, 1.7 * scale)))
            bottom = height
            top = max(0, bottom - trunk_height)
            shade = int(np.clip(120 + forward_m * 8, 70, 175))
            color = (shade, int(shade * 0.72), int(shade * 0.42))
            draw.rectangle(
                [center_x - trunk_width, top, center_x + trunk_width, bottom],
                fill=color,
                outline=(69, 53, 39),
            )

        goal_forward, goal_right = self._goal_body_xy()
        if goal_forward > 0.1:
            goal_angle = math.atan2(goal_right, goal_forward)
            if abs(goal_angle) <= fov_rad * 0.5:
                center_x = int((0.5 + goal_angle / fov_rad) * width)
                marker_h = max(6, int(height * min(0.45, 1.1 / max(goal_forward, 0.5))))
                center_y = max(6, horizon - marker_h // 2)
                draw.ellipse(
                    [center_x - 4, center_y - 4, center_x + 4, center_y + 4],
                    fill=(55, 210, 91),
                    outline=(22, 86, 39),
                )

        return np.asarray(image, dtype=np.uint8)

    def _build_info(
        self,
        *,
        reward_terms: dict[str, float],
        terminated: bool,
        truncated: bool,
        collision: bool = False,
        out_of_bounds: bool = False,
        success: bool = False,
        progress_m: float = 0.0,
        front_delta_m: float = 0.0,
        oscillation: bool = False,
    ) -> dict[str, Any]:
        clearances = self._compute_clearances()
        goal_forward, goal_right = self._goal_body_xy()
        return {
            "step_index": self._step_index,
            "position_xy_m": self._position_xy.astype(float).tolist(),
            "yaw_deg": math.degrees(self._yaw_rad),
            "goal_xy_m": self._goal_xy.astype(float).tolist(),
            "goal_body_xy_m": [float(goal_forward), float(goal_right)],
            "distance_to_goal_m": self._distance_to_goal(),
            "clearance_m": clearances,
            "last_action": self._last_action.value if self._last_action is not None else None,
            "reward_terms": reward_terms,
            "progress_m": progress_m,
            "front_delta_m": front_delta_m,
            "collision": collision,
            "out_of_bounds": out_of_bounds,
            "success": success,
            "oscillation": oscillation,
            "terminated": terminated,
            "truncated": truncated,
            "config": asdict(self.cfg),
        }

    def _compute_clearances(self) -> dict[str, float]:
        return {
            "front_m": self._sector_clearance(self._yaw_rad, math.radians(24.0), 11),
            "left_m": self._sector_clearance(self._yaw_rad - math.radians(55.0), math.radians(24.0), 9),
            "right_m": self._sector_clearance(self._yaw_rad + math.radians(55.0), math.radians(24.0), 9),
        }

    def _sector_clearance(self, center_angle: float, width_rad: float, num_rays: int) -> float:
        values = []
        for angle in np.linspace(center_angle - width_rad * 0.5, center_angle + width_rad * 0.5, num_rays):
            values.append(self._raycast(float(angle)))
        return float(np.quantile(values, 0.25))

    def _raycast(self, angle_rad: float) -> float:
        origin = self._position_xy.astype(np.float64)
        direction = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=np.float64)
        best = self.cfg.max_depth_m
        for obstacle in self._obstacles:
            center = np.array([obstacle.x_m, obstacle.y_m], dtype=np.float64)
            oc = origin - center
            b = 2.0 * float(np.dot(direction, oc))
            c = float(np.dot(oc, oc) - (obstacle.radius_m + self.cfg.drone_radius_m) ** 2)
            disc = b * b - 4.0 * c
            if disc < 0.0:
                continue
            root = math.sqrt(disc)
            for t in [(-b - root) * 0.5, (-b + root) * 0.5]:
                if 0.0 < t < best:
                    best = t
        return float(best)

    def _coerce_action(self, action: int | str | DiscreteDroneAction) -> DiscreteDroneAction:
        if isinstance(action, DiscreteDroneAction):
            return action
        if isinstance(action, str):
            return DiscreteDroneAction(action)
        return DiscreteDroneAction(ACTION_VOCAB[int(action)])

    def _forward_vector(self) -> np.ndarray:
        return np.array([math.cos(self._yaw_rad), math.sin(self._yaw_rad)], dtype=np.float32)

    def _right_vector(self) -> np.ndarray:
        return np.array([-math.sin(self._yaw_rad), math.cos(self._yaw_rad)], dtype=np.float32)

    def _goal_body_xy(self) -> tuple[float, float]:
        return self._point_body_xy(self._goal_xy)

    def _point_body_xy(self, point_xy: np.ndarray) -> tuple[float, float]:
        delta = point_xy.astype(np.float32) - self._position_xy
        cos_yaw = math.cos(self._yaw_rad)
        sin_yaw = math.sin(self._yaw_rad)
        forward = float(delta[0] * cos_yaw + delta[1] * sin_yaw)
        right = float(-delta[0] * sin_yaw + delta[1] * cos_yaw)
        return forward, right

    def _distance_to_goal(self) -> float:
        return float(np.linalg.norm(self._goal_xy - self._position_xy))

    def _collides(self, position_xy: np.ndarray) -> bool:
        for obstacle in self._obstacles:
            distance = float(np.linalg.norm(position_xy - np.array([obstacle.x_m, obstacle.y_m], dtype=np.float32)))
            if distance <= obstacle.radius_m + self.cfg.drone_radius_m:
                return True
        return False

    def _out_of_bounds(self, position_xy: np.ndarray) -> bool:
        half_width = self.cfg.world_size_m * 0.55
        return (
            position_xy[0] < -1.0
            or position_xy[0] > self.cfg.world_size_m
            or position_xy[1] < -half_width
            or position_xy[1] > half_width
        )

    def _is_oscillation(
        self,
        previous_action: DiscreteDroneAction | None,
        action: DiscreteDroneAction,
    ) -> bool:
        pairs = {
            (DiscreteDroneAction.STRAFE_LEFT, DiscreteDroneAction.STRAFE_RIGHT),
            (DiscreteDroneAction.STRAFE_RIGHT, DiscreteDroneAction.STRAFE_LEFT),
            (DiscreteDroneAction.YAW_LEFT, DiscreteDroneAction.YAW_RIGHT),
            (DiscreteDroneAction.YAW_RIGHT, DiscreteDroneAction.YAW_LEFT),
        }
        return previous_action is not None and (previous_action, action) in pairs

    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


class _FallbackSpaces:
    Discrete = _DiscreteFallback
    Box = _BoxFallback
    Dict = _DictFallback

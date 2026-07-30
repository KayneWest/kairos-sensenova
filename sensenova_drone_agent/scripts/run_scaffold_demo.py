#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

try:
    import yaml
except ModuleNotFoundError as exc:
    print("pyyaml is required to run this demo.", file=sys.stderr)
    raise SystemExit(1) from exc

from PIL import Image

from sensenova_drone.control_adapter import DroneToKairosControlAdapter
from sensenova_drone.kairos_adapter import SubprocessKairosAdapter
from sensenova_drone.loop import ClosedLoopAgent
from sensenova_drone.memory import RealObservationMemory
from sensenova_drone.observation import Observation
from sensenova_drone.observation_adapter import ObservationAdapter
from sensenova_drone.planner import KairosMPCPlanner
from sensenova_drone.safety import SafetyShield
from sensenova_drone.scoring import GoalSpec, RolloutScorer
from sensenova_drone.state_estimator import StateEstimator
from sensenova_drone.telemetry import TelemetryLogger


class DemoDrone:
    def __init__(self, frame_path: Path):
        self.frame_path = frame_path
        self.sent_commands = []

    async def read_observation(self) -> Observation:
        frame = Image.open(self.frame_path).convert("RGB")
        return Observation(frame_rgb=frame, metadata={"source": "real_gazebo_camera_demo"})

    async def send_command(self, command) -> None:
        self.sent_commands.append(command)


def build_demo_cfg(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    cfg.setdefault("state_estimator", {})
    cfg["state_estimator"]["pose_source"] = "mock"
    cfg["state_estimator"]["mock_pose"] = {
        "position_xyz": [0.0, 0.0, 2.0],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }

    cfg.setdefault("kairos", {})
    cfg["kairos"]["execute_subprocess"] = False

    cfg.setdefault("scoring", {})
    action_bias = cfg["scoring"].setdefault("action_bias", {})
    action_bias["yaw_left"] = 0.05
    return cfg


async def run_demo(config_path: Path, frame_path: Path, out_dir: Path, goal_prompt: str) -> dict:
    cfg = build_demo_cfg(config_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    drone = DemoDrone(frame_path)
    observation_adapter = ObservationAdapter()
    memory = RealObservationMemory()
    world_model = SubprocessKairosAdapter(cfg)
    scorer = RolloutScorer(cfg)
    safety_shield = SafetyShield(cfg)
    control_adapter = DroneToKairosControlAdapter(cfg)
    planner = KairosMPCPlanner(
        world_model=world_model,
        scorer=scorer,
        safety_shield=safety_shield,
        control_adapter=control_adapter,
        cfg=cfg,
    )
    state_estimator = StateEstimator(cfg)
    telemetry_logger = TelemetryLogger(out_dir)

    agent = ClosedLoopAgent(
        drone=drone,
        observation_adapter=observation_adapter,
        state_estimator=state_estimator,
        memory=memory,
        world_model=world_model,
        planner=planner,
        safety_shield=safety_shield,
        telemetry_logger=telemetry_logger,
        cfg=cfg,
    )

    goal = GoalSpec(prompt=goal_prompt)
    plan, executed_command = await agent.step(goal)

    step_dir = out_dir / "step_000001"
    telemetry_path = step_dir / "telemetry.json"

    rollout_requests = {}
    for candidate_dir in sorted(step_dir.glob("candidate_*")):
        request_path = candidate_dir / "kairos_request.json"
        if request_path.exists():
            rollout_requests[candidate_dir.name] = json.loads(request_path.read_text(encoding="utf-8"))

    return {
        "chosen_action": plan.action_sequence.actions[0].value,
        "executed_command": {
            "forward_m_s": executed_command.forward_m_s,
            "right_m_s": executed_command.right_m_s,
            "down_m_s": executed_command.down_m_s,
            "yawspeed_deg_s": executed_command.yawspeed_deg_s,
            "duration_s": executed_command.duration_s,
        },
        "telemetry_path": str(telemetry_path),
        "memory_size": len(memory),
        "candidate_rollout_requests": rollout_requests,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a one-step MPC scaffold demo.")
    parser.add_argument(
        "--config",
        default="config/runtime.example.yaml",
        help="Path to runtime config YAML, relative to the drone agent root.",
    )
    parser.add_argument(
        "--frame",
        default="sim_assets/sample_frames/gazebo_rgb_000001.png",
        help="Path to a real Gazebo RGB frame, relative to the drone agent root.",
    )
    parser.add_argument(
        "--out-dir",
        default="logs/demo_scaffold",
        help="Directory for demo telemetry and rollout requests, relative to the drone agent root.",
    )
    parser.add_argument(
        "--goal",
        default="Find a safe opening while keeping the camera stable enough to inspect the scene.",
        help="Goal prompt passed into the planner.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = (project_root / args.config).resolve()
    frame_path = (project_root / args.frame).resolve()
    out_dir = (project_root / args.out_dir).resolve()

    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1
    if not frame_path.is_file():
        print(f"Frame not found: {frame_path}", file=sys.stderr)
        return 1

    result = asyncio.run(run_demo(config_path, frame_path, out_dir, args.goal))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

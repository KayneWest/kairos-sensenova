#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:
    print("pyyaml is required to run this demo.", file=sys.stderr)
    raise SystemExit(1) from exc

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sensenova_drone.actions import DiscreteDroneAction, DroneCommand, discrete_to_command
from sensenova_drone.bc_infer import load_bc_policy_runner
from sensenova_drone.grounded_world_model import (
    BCGroundedWorldModelAdapter,
    GroundedMovementProposal,
    GroundedWorldModelMovementPlanner,
)
from sensenova_drone.loop import ClosedLoopAgent
from sensenova_drone.memory import RealObservationMemory
from sensenova_drone.observation import Observation
from sensenova_drone.observation_adapter import ObservationAdapter
from sensenova_drone.safety import SafetyShield
from sensenova_drone.scoring import GoalSpec
from sensenova_drone.state_estimator import StateEstimator
from sensenova_drone.telemetry import TelemetryLogger
from sensenova_drone.world_state import ObservationEncoding, WorldState


class DemoDrone:
    def __init__(self, frame_path: Path):
        self.frame_path = frame_path
        self.sent_commands: list[DroneCommand] = []

    async def read_observation(self) -> Observation:
        frame = Image.open(self.frame_path).convert("RGB")
        return Observation(frame_rgb=frame, metadata={"source": "real_gazebo_camera_demo"})

    async def send_command(self, command: DroneCommand) -> None:
        self.sent_commands.append(command)


class StaticGroundedWorldModel:
    def __init__(self, action: DiscreteDroneAction, action_cfg: dict[str, Any]):
        self.action = action
        self.action_cfg = action_cfg

    def encode_observation(self, frame_rgb, frame_path: str | None = None) -> ObservationEncoding:
        _ = frame_rgb
        return ObservationEncoding(
            frame_path=frame_path,
            metadata={
                "backend": "static_grounded_demo",
                "latent_available": False,
                "native_kairos_state": False,
            },
        )

    def propose_movement(
        self,
        world_state: WorldState,
        memory: RealObservationMemory,
        goal: GoalSpec,
        episode_step_dir: str | None = None,
    ) -> GroundedMovementProposal:
        _ = (world_state, memory, goal, episode_step_dir)
        return GroundedMovementProposal(
            action=self.action,
            command=discrete_to_command(self.action, self.action_cfg),
            confidence=1.0,
            metadata={"backend": "static_grounded_demo"},
        )


def build_cfg(config_path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg.setdefault("runtime", {})["mode"] = "grounded_world_model"
    cfg.setdefault("state_estimator", {})
    cfg["state_estimator"]["pose_source"] = "mock"
    cfg["state_estimator"]["mock_pose"] = {
        "position_xyz": [0.0, 0.0, 2.0],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    return cfg


def build_world_model(args: argparse.Namespace, cfg: dict[str, Any]):
    checkpoint = str(args.checkpoint or "").strip()
    if checkpoint:
        runner = load_bc_policy_runner(
            (PROJECT_ROOT / checkpoint).resolve() if not Path(checkpoint).is_absolute() else checkpoint,
            device=args.device,
            action_cfg=cfg.get("actions", {}),
        )
        return BCGroundedWorldModelAdapter(runner)

    return StaticGroundedWorldModel(
        DiscreteDroneAction(args.stub_action),
        cfg.get("actions", {}),
    )


async def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    config_path = (PROJECT_ROOT / args.config).resolve()
    frame_path = (PROJECT_ROOT / args.frame).resolve()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not frame_path.is_file():
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    cfg = build_cfg(config_path)
    world_model = build_world_model(args, cfg)
    safety_shield = SafetyShield(cfg)
    memory = RealObservationMemory()
    drone = DemoDrone(frame_path)
    telemetry_logger = TelemetryLogger(out_dir)
    planner = GroundedWorldModelMovementPlanner(
        world_model=world_model,
        action_cfg=cfg.get("actions", {}),
        cfg=cfg,
    )
    agent = ClosedLoopAgent(
        drone=drone,
        observation_adapter=ObservationAdapter(),
        state_estimator=StateEstimator(cfg),
        memory=memory,
        world_model=world_model,
        planner=planner,
        safety_shield=safety_shield,
        telemetry_logger=telemetry_logger,
        cfg=cfg,
    )

    goal = GoalSpec(
        prompt=args.goal,
        metadata={"goal_features": [float(value) for value in args.goal_features.split(",")]},
    )
    plan, executed_command = await agent.step(goal)
    telemetry_path = out_dir / "step_000001" / "telemetry.json"
    return {
        "mode": "grounded_world_model",
        "chosen_action": plan.action.value,
        "confidence": plan.confidence,
        "executed_command": {
            "forward_m_s": executed_command.forward_m_s,
            "right_m_s": executed_command.right_m_s,
            "down_m_s": executed_command.down_m_s,
            "yawspeed_deg_s": executed_command.yawspeed_deg_s,
            "duration_s": executed_command.duration_s,
        },
        "telemetry_path": str(telemetry_path),
        "memory_size": len(memory),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one grounded world-model control step offline.")
    parser.add_argument("--config", default="config/runtime.example.yaml")
    parser.add_argument("--frame", default="sim_assets/sample_frames/gazebo_rgb_000001.png")
    parser.add_argument("--out-dir", default="logs/demo_grounded_world_model")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stub-action", default="hover", choices=[action.value for action in DiscreteDroneAction])
    parser.add_argument("--goal", default="Avoid trees and keep moving only when clearance is safe.")
    parser.add_argument("--goal-features", default="0,0,0,0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = asyncio.run(run_demo(args))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

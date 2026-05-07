#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import suppress
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_multiscene_waypoint_training_session as multiscene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-set", required=True)
    parser.add_argument("--labels", default="")
    parser.add_argument("--depth-topic", default="/depth_camera")
    parser.add_argument("--bridge-wait-s", type=float, default=4.0)
    parser.add_argument("--takeoff-altitude-m", type=float, default=0.8)
    parser.add_argument("--min-offboard-ready-altitude-m", type=float, default=0.4)
    parser.add_argument("--front-blocked-threshold-m", type=float, default=4.5)
    parser.add_argument("--initial-blocked-check-window-s", type=float, default=2.0)
    parser.add_argument("--initial-blocked-check-max-samples", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--i-understand-this-is-sitl", action="store_true", dest="i_understand_this_is_sitl")
    return parser.parse_args()


def _selected_scenarios(args: argparse.Namespace) -> list[multiscene.Scenario]:
    scenarios = multiscene._scenario_set(args.scenario_set)
    labels = {label.strip() for label in args.labels.split(",") if label.strip()}
    if labels:
        scenarios = [scenario for scenario in scenarios if scenario.label in labels]
    if not scenarios:
        raise SystemExit("No scenarios selected for probing.")
    return scenarios


def main() -> int:
    args = parse_args()
    if not args.i_understand_this_is_sitl:
        raise SystemExit("Refusing to run without --i-understand-this-is-sitl")

    session_id = datetime.now(timezone.utc).strftime("tree_probe_%Y%m%dT%H%M%SZ")
    out_json = (
        Path(args.out_json).expanduser().resolve()
        if args.out_json.strip()
        else PROJECT_ROOT / "logs" / f"{session_id}.json"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)

    scenarios = _selected_scenarios(args)
    tools_container_name = f"{multiscene.TOOLS_SESSION_PREFIX}{session_id}"
    results: list[dict[str, object]] = []

    multiscene._stop_all_sim_containers()
    multiscene._stop_all_tools_containers()
    multiscene._launch_tools_container(container_name=tools_container_name)
    try:
        for scenario in scenarios:
            container_name = f"sensenova_drone_agent_probe_{session_id}_{scenario.label}"
            try:
                multiscene._launch_sim(scenario=scenario, container_name=container_name)
                multiscene._wait_for_connection(tools_container_name=tools_container_name, timeout=args.timeout)
                topic = multiscene.DEFAULT_GAZEBO_TOPIC_TEMPLATE.format(world=scenario.world)
                tool_command = multiscene._tools_exec_command(
                    tools_container_name,
                    [
                        "python3",
                        "scripts/eval_tree_avoidance_policy.py",
                        "--probe-only",
                        "--controller",
                        "reactive_obstacle_teacher",
                        "--gazebo-topic",
                        topic,
                        "--depth-topic",
                        args.depth_topic,
                        "--world-label",
                        scenario.label,
                        "--run-id",
                        f"{session_id}_{scenario.label}",
                        "--bridge-wait-s",
                        str(args.bridge_wait_s),
                        "--takeoff-altitude-m",
                        str(args.takeoff_altitude_m),
                        "--min-offboard-ready-altitude-m",
                        str(args.min_offboard_ready_altitude_m),
                        "--front-blocked-threshold-m",
                        str(args.front_blocked_threshold_m),
                        "--initial-blocked-check-window-s",
                        str(args.initial_blocked_check_window_s),
                        "--initial-blocked-check-max-samples",
                        str(args.initial_blocked_check_max_samples),
                        "--timeout",
                        str(args.timeout),
                        "--i-understand-this-is-sitl",
                    ],
                )
                completed = subprocess.run(
                    tool_command,
                    cwd=PROJECT_ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                )
                payload = json.loads(completed.stdout)
                results.append(
                    {
                        "scenario": scenario.label,
                        "world": scenario.world,
                        "pose": scenario.pose,
                        "target_families": scenario.target_families,
                        "status": "completed",
                        "probe": payload.get("probe", {}),
                    }
                )
            except subprocess.TimeoutExpired as exc:
                stdout_text = (exc.stdout or "").strip()
                parsed_payload = None
                if stdout_text:
                    with suppress(Exception):
                        parsed_payload = json.loads(stdout_text)
                if parsed_payload is not None:
                    results.append(
                        {
                            "scenario": scenario.label,
                            "world": scenario.world,
                            "pose": scenario.pose,
                            "target_families": scenario.target_families,
                            "status": "completed_timeout",
                            "probe": parsed_payload.get("probe", {}),
                        }
                    )
                else:
                    results.append(
                        {
                            "scenario": scenario.label,
                            "world": scenario.world,
                            "pose": scenario.pose,
                            "target_families": scenario.target_families,
                            "status": "timeout",
                            "stderr": (exc.stderr or "")[-4000:],
                            "stdout": stdout_text[-4000:],
                        }
                    )
            except subprocess.CalledProcessError as exc:
                results.append(
                    {
                        "scenario": scenario.label,
                        "world": scenario.world,
                        "pose": scenario.pose,
                        "target_families": scenario.target_families,
                        "status": "failed",
                        "stderr": exc.stderr[-4000:],
                        "stdout": exc.stdout[-4000:],
                    }
                )
            finally:
                multiscene._stop_container(container_name)
    finally:
        multiscene._stop_container(tools_container_name)

    payload = {
        "run_id": session_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_set": args.scenario_set,
        "labels": [scenario.label for scenario in scenarios],
        "results": results,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"out_json": str(out_json), "num_results": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

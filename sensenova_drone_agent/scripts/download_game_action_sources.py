#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


@dataclass(frozen=True)
class Source:
    name: str
    kind: str
    priority: int
    status: str
    local_path: str
    estimated_size: str
    action_signal: str
    notes: str
    url: str


SOURCES: dict[str, Source] = {
    "procgen": Source(
        name="procgen",
        kind="pip_simulator",
        priority=1,
        status="lightweight",
        local_path="docker image: sensenova_drone_agent-game-sources:local",
        estimated_size="<1 GB image layer incremental if procgen image already exists",
        action_signal="strong discrete action -> next-frame/reward",
        notes="Best first source for counterfactual pixel-action data.",
        url="https://github.com/openai/procgen",
    ),
    "vizdoom": Source(
        name="vizdoom",
        kind="pip_simulator",
        priority=2,
        status="lightweight",
        local_path="docker image: sensenova_drone_agent-game-sources:local",
        estimated_size="small Python package plus Doom assets bundled with package",
        action_signal="strong first-person 3D action -> camera/game-state",
        notes="Best GTA-ish/FPS-style lightweight simulator.",
        url="https://github.com/Farama-Foundation/ViZDoom",
    ),
    "minigrid": Source(
        name="minigrid",
        kind="pip_simulator",
        priority=3,
        status="lightweight",
        local_path="docker image: sensenova_drone_agent-game-sources:local",
        estimated_size="small Python package",
        action_signal="very strong discrete action -> symbolic/pixel state",
        notes="Not visually rich, but excellent for action-identifiability smoke tests.",
        url="https://github.com/Farama-Foundation/Minigrid",
    ),
    "ai2thor": Source(
        name="ai2thor",
        kind="pip_simulator_with_runtime_asset",
        priority=4,
        status="medium",
        local_path="sensenova_drone_agent/data/game_action_sources/ai2thor",
        estimated_size="Python package small; Unity runtime/cache can be hundreds of MB",
        action_signal="strong embodied navigation/interactions in indoor scenes",
        notes="Runtime scene launch may need Xvfb or GPU display support.",
        url="https://github.com/allenai/ai2thor",
    ),
    "minerl": Source(
        name="minerl",
        kind="large_dataset_and_simulator",
        priority=5,
        status="large_legacy",
        local_path="sensenova_drone_agent/data/game_action_sources/minerl",
        estimated_size="tens to hundreds of GB depending on tasks",
        action_signal="strong Minecraft video + keyboard/mouse demonstrations",
        notes="Good conceptual match to Dreamer4; Python/Java dependency stack is legacy and data is large.",
        url="https://minerl.readthedocs.io/",
    ),
    "carla": Source(
        name="carla",
        kind="large_simulator",
        priority=6,
        status="large",
        local_path="sensenova_drone_agent/data/game_action_sources/carla",
        estimated_size="many GB to tens of GB depending on release/assets",
        action_signal="strong driving control -> camera/sensor futures",
        notes="Closest GTA-like open simulator, but server download/build is large.",
        url="https://github.com/carla-simulator/carla",
    ),
    "habitat": Source(
        name="habitat",
        kind="simulator_plus_gated_scene_assets",
        priority=7,
        status="gated_assets",
        local_path="sensenova_drone_agent/data/game_action_sources/habitat",
        estimated_size="package small/medium; realistic scenes can be tens to hundreds of GB and may require terms",
        action_signal="strong embodied navigation action -> egocentric frame",
        notes="Useful later; full HM3D/Matterport assets usually need explicit dataset setup.",
        url="https://github.com/facebookresearch/habitat-sim",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap action-identifiable game/simulator sources. Lightweight sources "
            "are installed into a Docker image; large/gated sources are manifested unless "
            "explicitly enabled."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        choices=sorted([*SOURCES.keys(), "all", "lightweight"]),
        help="Source to prepare. Repeatable. Defaults to lightweight.",
    )
    parser.add_argument("--out-dir", default="sensenova_drone_agent/data/game_action_sources")
    parser.add_argument("--build-image", action="store_true", help="Build the game source Docker image.")
    parser.add_argument("--smoke", action="store_true", help="Run quick import/API smoke tests in Docker.")
    parser.add_argument("--allow-large", action="store_true", help="Allow large-source bootstrap hooks where implemented.")
    parser.add_argument("--dry-run", action="store_true", help="Write manifests only; do not build or run smoke tests.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = select_sources(args.source)

    manifest = {
        "phase": "game_action_source_bootstrap",
        "created_unix_s": time.time(),
        "repo_root": str(REPO_ROOT),
        "out_dir": str(out_dir),
        "selected": selected,
        "sources": [asdict(SOURCES[name]) for name in selected],
        "large_sources_enabled": bool(args.allow_large),
        "docker_image": "sensenova_drone_agent-game-sources:local",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(render_readme(manifest), encoding="utf-8")

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    if args.build_image or any(SOURCES[name].status in {"lightweight", "medium"} for name in selected):
        build_image()

    smoke_results: dict[str, Any] = {}
    if args.smoke:
        if "procgen" in selected:
            smoke_results["procgen"] = smoke_procgen()
        if "vizdoom" in selected:
            smoke_results["vizdoom"] = smoke_vizdoom()
        if "minigrid" in selected:
            smoke_results["minigrid"] = smoke_minigrid()
        if "ai2thor" in selected:
            smoke_results["ai2thor_import"] = smoke_ai2thor_import()
            # Full AI2-THOR controller launch is intentionally not automatic:
            # it downloads a Unity runtime and may require Xvfb/GPU display.

    large_notes = {}
    for name in selected:
        source = SOURCES[name]
        source_dir = out_dir / name
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "manifest.json").write_text(json.dumps(asdict(source), indent=2), encoding="utf-8")
        if source.status in {"large", "large_legacy", "gated_assets"} and not args.allow_large:
            large_notes[name] = "not downloaded; rerun with --allow-large after choosing exact release/assets"

    result = {
        **manifest,
        "smoke_results": smoke_results,
        "large_notes": large_notes,
        "completed_unix_s": time.time(),
    }
    (out_dir / "bootstrap_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


def select_sources(values: list[str] | None) -> list[str]:
    values = values or ["lightweight"]
    selected: list[str] = []
    for value in values:
        if value == "all":
            selected.extend(SOURCES)
        elif value == "lightweight":
            selected.extend(name for name, source in SOURCES.items() if source.status in {"lightweight", "medium"})
        else:
            selected.append(value)
    out = []
    seen = set()
    for name in selected:
        if name not in SOURCES:
            raise ValueError(f"unknown source: {name}")
        if name not in seen:
            out.append(name)
            seen.add(name)
    return sorted(out, key=lambda name: SOURCES[name].priority)


def build_image() -> None:
    subprocess_run([str(REPO_ROOT / "sensenova_drone_agent/scripts/build_game_action_sources_image.sh")])


def smoke_procgen() -> dict[str, Any]:
    code = """
import numpy as np
from procgen import ProcgenEnv
env = ProcgenEnv(num_envs=1, env_name='coinrun', num_levels=1, start_level=0, distribution_mode='easy')
obs = env.reset()
action = env.action_space.sample()
obs, rew, done, info = env.step(np.asarray([action], dtype=np.int32))
env.close()
print({'action_n': env.action_space.n, 'reward': float(rew[0]), 'done': bool(done[0])})
"""
    return docker_python(code)


def smoke_vizdoom() -> dict[str, Any]:
    code = """
from vizdoom import DoomGame, scenarios_path
from pathlib import Path
game = DoomGame()
game.load_config(str(Path(scenarios_path) / 'basic.cfg'))
game.set_window_visible(False)
game.init()
state = game.get_state()
reward = game.make_action([1, 0, 0])
shape = tuple(state.screen_buffer.shape) if state is not None else None
game.close()
print({'screen_shape': shape, 'reward': float(reward)})
"""
    return docker_python(code)


def smoke_minigrid() -> dict[str, Any]:
    code = """
import gymnasium as gym
import minigrid  # noqa: F401
env = gym.make('MiniGrid-Empty-5x5-v0', render_mode='rgb_array')
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
frame = env.render()
env.close()
print({'action_n': env.action_space.n, 'frame_shape': tuple(frame.shape), 'reward': float(reward)})
"""
    return docker_python(code)


def smoke_ai2thor_import() -> dict[str, Any]:
    code = """
import ai2thor
print({'ai2thor_version': getattr(ai2thor, '__version__', 'unknown')})
"""
    return docker_python(code)


def docker_python(code: str) -> dict[str, Any]:
    cmd = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/workspace/.docker-home",
        "-v",
        f"{REPO_ROOT}:/workspace",
        "-v",
        "/etc/passwd:/etc/passwd:ro",
        "-v",
        "/etc/group:/etc/group:ro",
        "-w",
        "/workspace",
        "sensenova_drone_agent-game-sources:local",
        "python",
        "-c",
        code,
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr_tail": proc.stderr.strip()[-2000:],
        "success": proc.returncode == 0,
    }


def subprocess_run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def render_readme(manifest: dict[str, Any]) -> str:
    lines = [
        "# Game Action Sources",
        "",
        "This directory records action-identifiable game/simulator sources for world-model dynamics training.",
        "",
        "| Source | Status | Action Signal | Size | Notes |",
        "|---|---|---|---|---|",
    ]
    for source in manifest["sources"]:
        lines.append(
            f"| {source['name']} | {source['status']} | {source['action_signal']} | "
            f"{source['estimated_size']} | {source['notes']} |"
        )
    lines.extend(
        [
            "",
            "Large/gated sources are not downloaded automatically. For action-conditioning experiments, prefer small resettable sources first, then scale to heavier simulators once the collector/training/eval loop is proven.",
            "",
            "Recommended order:",
            "",
            "1. Procgen counterfactual data.",
            "2. ViZDoom first-person counterfactual data.",
            "3. MiniGrid action-identifiability smoke data.",
            "4. AI2-THOR indoor embodied data.",
            "5. MineRL, CARLA, Habitat only after choosing exact release/assets.",
        ]
    )
    return "\n".join(lines) + "\n"


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
